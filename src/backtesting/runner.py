"""
BacktestRunner: walk-forward evaluation engine with temporal bias prevention.

Slides overlapping evaluation windows across the prediction history,
re-runs each resolved prediction using temporally-isolated components
(weight snapshots + ephemeral ChromaDB indexes), computes metrics per
window, and persists results to PostgreSQL with cancellation support.

Design principles:
- Fresh EnsemblePredictor per prediction (mutable _forecast_output state).
- Cache heavy components (RAG, TKG, orchestrator) across predictions
  within a window.
- DB-based cancellation: polls backtest_runs.status between windows.
- No pickling of sessions: creates own DB engine in-process via init_db().
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.backtesting.evaluator import (
    compute_brier_score,
    compute_calibration_bins,
    compute_hit_rate,
    compute_mrr,
)
from src.backtesting.schemas import BacktestRunConfig, BacktestRunResult, WindowResult
from src.backtesting.temporal_index import build_temporal_chromadb_index
from src.backtesting.weight_snapshot import snapshot_calibration_weights
from src.db.models import (
    BacktestResult,
    BacktestRun,
    OutcomeRecord,
    PolymarketComparison,
    Prediction,
)

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Orchestrates walk-forward backtesting with temporal isolation.

    Usage (typically called from a ProcessPoolExecutor worker)::

        config = BacktestRunConfig.from_json(config_json)
        runner = BacktestRunner(config, async_session_factory)
        result = await runner.run()
    """

    def __init__(
        self,
        run_config: BacktestRunConfig,
        async_session_factory: Any,
    ) -> None:
        """
        Args:
            run_config: Frozen run configuration.
            async_session_factory: Callable returning async context-manager
                sessions (async_sessionmaker). Created in-process via
                init_db() when running in ProcessPoolExecutor.
        """
        self._config = run_config
        self._session_factory = async_session_factory
        self._predictor_cache: Any = None

    # ------------------------------------------------------------------
    # Predictor construction (follows auto_forecaster.py pattern)
    # ------------------------------------------------------------------

    def _build_predictor(
        self,
        checkpoint_name: str | None = None,
        checkpoint_path: str | None = None,
    ) -> Any:
        """Build a fresh EnsemblePredictor with optional checkpoint override.

        Caches heavy components (RAG pipeline, TKG predictor, orchestrator)
        across calls. Returns a new EnsemblePredictor each time because
        mutable _forecast_output state prevents reuse.

        Args:
            checkpoint_name: Friendly name for the checkpoint (for logging).
            checkpoint_path: If provided, loads this specific TKG checkpoint
                instead of the default. Enables RE-GCN vs TiRGN comparison.
        """
        if self._predictor_cache is not None:
            llm_orch, tkg_pred, weight_loader = self._predictor_cache
        else:
            llm_orch = None
            tkg_pred = None
            weight_loader = None

            # TKG predictor
            try:
                from src.forecasting.tkg_predictor import TKGPredictor

                tkg_pred = TKGPredictor()
                if not tkg_pred.trained:
                    logger.warning("TKG predictor has no trained model")
                    tkg_pred = None
            except Exception as exc:
                logger.warning("TKG predictor init failed: %s", exc)

            # LLM orchestrator (requires Gemini client)
            try:
                from src.forecasting.gemini_client import get_gemini_client
                from src.forecasting.graph_validator import GraphValidator
                from src.forecasting.rag_pipeline import RAGPipeline
                from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator
                from src.forecasting.scenario_generator import ScenarioGenerator

                gemini = get_gemini_client()
                rag_pipeline = RAGPipeline()
                scenario_gen = ScenarioGenerator(gemini)
                graph_validator = GraphValidator(tkg_predictor=tkg_pred)
                llm_orch = ReasoningOrchestrator(
                    client=gemini,
                    generator=scenario_gen,
                    rag_pipeline=rag_pipeline,
                    graph_validator=graph_validator,
                )
            except Exception as exc:
                logger.warning("LLM orchestrator init failed: %s", exc)

            # Weight loader for per-CAMEO calibration
            try:
                from src.calibration.weight_loader import WeightLoader

                weight_loader = WeightLoader(
                    async_session_factory=self._session_factory,
                )
            except Exception as exc:
                logger.warning("WeightLoader init failed: %s", exc)

            self._predictor_cache = (llm_orch, tkg_pred, weight_loader)

        from src.forecasting.ensemble_predictor import EnsemblePredictor

        return EnsemblePredictor(
            llm_orchestrator=llm_orch,
            tkg_predictor=tkg_pred,
            weight_loader=weight_loader,
        )

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_windows(
        earliest: datetime,
        latest: datetime,
        window_size_days: int,
        slide_step_days: int,
    ) -> list[dict[str, datetime]]:
        """Generate sliding evaluation windows.

        Each window has:
        - window_start / window_end: The "training" context window.
        - prediction_start / prediction_end: The prediction evaluation
          window (immediately after the context window, same duration).

        The slide step moves the window_start forward; windows may overlap.

        Returns:
            List of dicts with keys window_start, window_end,
            prediction_start, prediction_end.
        """
        window_size = timedelta(days=window_size_days)
        slide_step = timedelta(days=slide_step_days)

        windows: list[dict[str, datetime]] = []
        cursor = earliest

        while cursor + 2 * window_size <= latest + timedelta(days=1):
            w_start = cursor
            w_end = cursor + window_size
            p_start = w_end
            p_end = w_end + window_size

            windows.append({
                "window_start": w_start,
                "window_end": w_end,
                "prediction_start": p_start,
                "prediction_end": p_end,
            })
            cursor += slide_step

        return windows

    # ------------------------------------------------------------------
    # Cancellation check
    # ------------------------------------------------------------------

    async def _check_cancelled(self, session: AsyncSession) -> bool:
        """Poll the backtest_runs row for cancellation signal.

        Returns True if the run's status has been set to 'cancelling'
        by the admin endpoint.
        """
        stmt = select(BacktestRun.status).where(
            BacktestRun.id == self._config.run_id
        )
        result = await session.execute(stmt)
        status = result.scalar_one_or_none()
        return status in ("cancelling", "cancelled")

    # ------------------------------------------------------------------
    # Polymarket comparison data
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_polymarket_brier(
        session: AsyncSession,
        prediction_id: str,
    ) -> dict[str, Any] | None:
        """Fetch resolved Polymarket comparison data for a prediction.

        Returns dict with polymarket_brier, geopol_brier if the
        comparison is resolved, None otherwise.
        """
        stmt = (
            select(PolymarketComparison)
            .where(
                and_(
                    PolymarketComparison.geopol_prediction_id == prediction_id,
                    PolymarketComparison.status == "resolved",
                )
            )
        )
        result = await session.execute(stmt)
        comp = result.scalar_one_or_none()
        if comp is None:
            return None

        return {
            "polymarket_brier": comp.polymarket_brier,
            "geopol_brier": comp.geopol_brier,
        }

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> BacktestRunResult:
        """Execute the full walk-forward backtest.

        Orchestration:
        1. Determine date range from predictions table.
        2. Generate sliding windows.
        3. Per window, per checkpoint:
           a. Check cancellation.
           b. Snapshot calibration weights at window_end.
           c. Build ephemeral ChromaDB temporal index.
           d. Query resolved predictions in the prediction window.
           e. Re-predict each via fresh EnsemblePredictor.
           f. Compute metrics (Brier, calibration bins, hit rate).
           g. Persist BacktestResult row.
           h. Update progress.
        4. Compute aggregate metrics.
        5. Update BacktestRun status.

        Returns:
            BacktestRunResult with all window metrics and aggregates.
        """
        run_result = BacktestRunResult(
            run_id=self._config.run_id,
            status="running",
        )

        try:
            async with self._session_factory() as session:
                # Mark run as started.
                await session.execute(
                    update(BacktestRun)
                    .where(BacktestRun.id == self._config.run_id)
                    .values(
                        status="running",
                        started_at=datetime.now(timezone.utc),
                    )
                )
                await session.commit()

            # Determine date range from predictions.
            async with self._session_factory() as session:
                stmt = select(
                    Prediction.created_at,
                ).order_by(Prediction.created_at.asc())
                result = await session.execute(stmt)
                all_dates = [row[0] for row in result.all()]

            if not all_dates:
                run_result.status = "failed"
                run_result.error_message = "No predictions found in database"
                await self._finalize_run(run_result)
                return run_result

            earliest = all_dates[0]
            latest = all_dates[-1]

            # Generate windows.
            windows = self._generate_windows(
                earliest=earliest,
                latest=latest,
                window_size_days=self._config.window_size_days,
                slide_step_days=self._config.slide_step_days,
            )

            if not windows:
                run_result.status = "completed"
                run_result.error_message = (
                    "Insufficient date range for any evaluation window"
                )
                await self._finalize_run(run_result)
                return run_result

            # Total windows = windows * checkpoints.
            total_window_steps = len(windows) * max(1, len(self._config.checkpoints))

            async with self._session_factory() as session:
                await session.execute(
                    update(BacktestRun)
                    .where(BacktestRun.id == self._config.run_id)
                    .values(total_windows=total_window_steps)
                )
                await session.commit()

            completed_count = 0

            for window in windows:
                w_start = window["window_start"]
                w_end = window["window_end"]
                p_start = window["prediction_start"]
                p_end = window["prediction_end"]

                # Iterate over checkpoints (at least one pass even if empty).
                checkpoint_items = (
                    list(self._config.checkpoints.items())
                    if self._config.checkpoints
                    else [("default", None)]
                )

                for cp_name, cp_file in checkpoint_items:
                    # Check cancellation.
                    async with self._session_factory() as session:
                        if await self._check_cancelled(session):
                            logger.info(
                                "Backtest %s cancelled at window %s",
                                self._config.run_id,
                                w_start.isoformat(),
                            )
                            run_result.status = "cancelled"
                            await self._finalize_run(run_result)
                            return run_result

                    # Process this window+checkpoint.
                    window_result = await self._process_window(
                        w_start=w_start,
                        w_end=w_end,
                        p_start=p_start,
                        p_end=p_end,
                        checkpoint_name=cp_name,
                        checkpoint_file=cp_file,
                    )

                    if window_result is not None:
                        run_result.windows.append(window_result)

                    completed_count += 1

                    # Update progress.
                    async with self._session_factory() as session:
                        await session.execute(
                            update(BacktestRun)
                            .where(BacktestRun.id == self._config.run_id)
                            .values(completed_windows=completed_count)
                        )
                        await session.commit()

            # Compute aggregates.
            self._compute_aggregates(run_result)
            run_result.status = "completed"

        except Exception as exc:
            logger.exception("Backtest run %s failed", self._config.run_id)
            run_result.status = "failed"
            run_result.error_message = str(exc)

        await self._finalize_run(run_result)
        return run_result

    # ------------------------------------------------------------------
    # Per-window processing
    # ------------------------------------------------------------------

    async def _process_window(
        self,
        w_start: datetime,
        w_end: datetime,
        p_start: datetime,
        p_end: datetime,
        checkpoint_name: str,
        checkpoint_file: str | None,
    ) -> WindowResult | None:
        """Process a single evaluation window for one checkpoint.

        Returns WindowResult or None if the window has insufficient data.
        """
        async with self._session_factory() as session:
            # Snapshot calibration weights at window end.
            weight_snapshot = await snapshot_calibration_weights(session, w_end)

            # Query resolved predictions in the prediction window.
            stmt = (
                select(Prediction, OutcomeRecord.outcome)
                .join(
                    OutcomeRecord,
                    OutcomeRecord.prediction_id == Prediction.id,
                )
                .where(
                    and_(
                        Prediction.created_at >= p_start,
                        Prediction.created_at < p_end,
                    )
                )
                .order_by(Prediction.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.all()

        if len(rows) < self._config.min_predictions_per_window:
            logger.info(
                "Window %s-%s: only %d resolved predictions (min=%d), skipping",
                w_start.date(),
                w_end.date(),
                len(rows),
                self._config.min_predictions_per_window,
            )
            return None

        # Build temporal ChromaDB index.
        # The RAGPipeline default persist_dir is "./chroma_db".
        try:
            _temporal_col = build_temporal_chromadb_index(
                source_persist_dir="./chroma_db",
                source_collection_name="rss_articles",
                cutoff_date=w_end,
            )
        except Exception as exc:
            logger.warning(
                "Temporal ChromaDB index build failed for window %s: %s",
                w_end.date(),
                exc,
            )
            _temporal_col = None

        # Re-predict each resolved prediction.
        predicted_probs: list[float] = []
        actual_outcomes: list[float] = []
        prediction_details: list[dict[str, Any]] = []
        pm_brier_values: list[float] = []
        geopol_wins = 0
        pm_wins = 0
        total_predictions = 0

        for prediction, outcome in rows:
            try:
                predictor = self._build_predictor(
                    checkpoint_name=checkpoint_name,
                    checkpoint_path=checkpoint_file,
                )

                # Resolve alpha from weight snapshot.
                cameo_code = prediction.cameo_root_code
                alpha = weight_snapshot.get(
                    cameo_code,
                    weight_snapshot.get("global", 0.58),
                ) if cameo_code else weight_snapshot.get("global", 0.58)

                # Run prediction (synchronous call).
                ensemble_pred, _forecast_output = predictor.predict(
                    question=prediction.question,
                    cameo_root_code=cameo_code,
                    alpha_override=alpha,
                )

                re_predicted_prob = ensemble_pred.final_probability
                predicted_probs.append(re_predicted_prob)
                actual_outcomes.append(float(outcome))
                total_predictions += 1

                # Per-prediction detail record.
                detail = {
                    "prediction_id": prediction.id,
                    "question": prediction.question[:200],
                    "original_prob": prediction.probability,
                    "re_predicted_prob": re_predicted_prob,
                    "outcome": float(outcome),
                    "brier": (re_predicted_prob - float(outcome)) ** 2,
                    "cameo_root_code": cameo_code,
                    "country_iso": prediction.country_iso,
                }
                prediction_details.append(detail)

                # Polymarket comparison data.
                async with self._session_factory() as session:
                    pm_data = await self._get_polymarket_brier(
                        session, prediction.id
                    )
                if pm_data is not None:
                    pm_b = pm_data.get("polymarket_brier")
                    gp_b = pm_data.get("geopol_brier")
                    if pm_b is not None:
                        pm_brier_values.append(pm_b)
                    if pm_b is not None and gp_b is not None:
                        if gp_b < pm_b:
                            geopol_wins += 1
                        elif pm_b < gp_b:
                            pm_wins += 1

            except Exception as exc:
                logger.warning(
                    "Re-prediction failed for %s in window %s: %s",
                    prediction.id,
                    w_end.date(),
                    exc,
                )

        if not predicted_probs:
            logger.info(
                "Window %s-%s: no successful re-predictions, skipping",
                w_start.date(),
                w_end.date(),
            )
            return None

        # Compute metrics.
        brier = compute_brier_score(predicted_probs, actual_outcomes)
        cal_bins = compute_calibration_bins(predicted_probs, actual_outcomes)
        hit_rate_data = compute_hit_rate(predicted_probs, actual_outcomes)

        window_result = WindowResult(
            window_start=w_start,
            window_end=w_end,
            prediction_start=p_start,
            prediction_end=p_end,
            checkpoint_name=checkpoint_name,
            num_predictions=total_predictions,
            brier_score=brier,
            mrr=None,  # MRR requires TKG ranking data not available from re-prediction.
            hits_at_1=hit_rate_data.get("hit_rate"),
            hits_at_10=None,
            calibration_bins=cal_bins,
            prediction_details=prediction_details,
            polymarket_brier=float(sum(pm_brier_values) / len(pm_brier_values)) if pm_brier_values else None,
            geopol_vs_pm_wins=geopol_wins if pm_brier_values else None,
            pm_vs_geopol_wins=pm_wins if pm_brier_values else None,
            weight_snapshot=weight_snapshot,
        )

        # Persist to database.
        await self._persist_window_result(window_result)

        # Update total predictions count.
        async with self._session_factory() as session:
            run_row = await session.get(BacktestRun, self._config.run_id)
            if run_row is not None:
                run_row.total_predictions = (
                    (run_row.total_predictions or 0) + total_predictions
                )
                await session.commit()

        logger.info(
            "Window %s-%s [%s]: brier=%.4f, n=%d, hit_rate=%.2f%%",
            w_start.date(),
            w_end.date(),
            checkpoint_name,
            brier,
            total_predictions,
            hit_rate_data.get("hit_rate", 0) * 100,
        )

        return window_result

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _persist_window_result(self, wr: WindowResult) -> None:
        """Write a WindowResult to backtest_results."""
        async with self._session_factory() as session:
            row = BacktestResult(
                run_id=self._config.run_id,
                window_start=wr.window_start,
                window_end=wr.window_end,
                prediction_start=wr.prediction_start,
                prediction_end=wr.prediction_end,
                checkpoint_name=wr.checkpoint_name,
                num_predictions=wr.num_predictions,
                brier_score=wr.brier_score,
                mrr=wr.mrr,
                hits_at_1=wr.hits_at_1,
                hits_at_10=wr.hits_at_10,
                calibration_bins_json=wr.calibration_bins,
                prediction_details_json=wr.prediction_details,
                polymarket_brier=wr.polymarket_brier,
                geopol_vs_pm_wins=wr.geopol_vs_pm_wins,
                pm_vs_geopol_wins=wr.pm_vs_geopol_wins,
                weight_snapshot_json=wr.weight_snapshot,
            )
            session.add(row)
            await session.commit()

    async def _finalize_run(self, result: BacktestRunResult) -> None:
        """Update the BacktestRun row with final status and aggregates."""
        async with self._session_factory() as session:
            values: dict[str, Any] = {
                "status": result.status,
                "completed_at": datetime.now(timezone.utc),
            }
            if result.aggregate_brier is not None:
                values["aggregate_brier"] = result.aggregate_brier
            if result.aggregate_mrr is not None:
                values["aggregate_mrr"] = result.aggregate_mrr
            if result.vs_polymarket_record is not None:
                values["vs_polymarket_record_json"] = result.vs_polymarket_record
            if result.error_message:
                values["error_message"] = result.error_message

            await session.execute(
                update(BacktestRun)
                .where(BacktestRun.id == self._config.run_id)
                .values(**values)
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Aggregate computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_aggregates(result: BacktestRunResult) -> None:
        """Compute aggregate metrics from all completed windows."""
        if not result.windows:
            return

        brier_scores = [
            w.brier_score for w in result.windows if w.brier_score is not None
        ]
        mrr_scores = [
            w.mrr for w in result.windows if w.mrr is not None
        ]

        if brier_scores:
            result.aggregate_brier = sum(brier_scores) / len(brier_scores)
        if mrr_scores:
            result.aggregate_mrr = sum(mrr_scores) / len(mrr_scores)

        # Polymarket head-to-head record.
        total_geopol_wins = sum(
            w.geopol_vs_pm_wins or 0 for w in result.windows
        )
        total_pm_wins = sum(
            w.pm_vs_geopol_wins or 0 for w in result.windows
        )
        if total_geopol_wins + total_pm_wins > 0:
            result.vs_polymarket_record = {
                "geopol_wins": total_geopol_wins,
                "polymarket_wins": total_pm_wins,
            }
