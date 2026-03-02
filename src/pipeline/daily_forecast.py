"""
Daily forecast pipeline orchestrator.

Runs the complete 6-phase daily forecast cycle:
1. Dequeue carryover questions from yesterday's budget exhaustion
2. Generate fresh questions from recent GDELT events (budget permitting)
3. Predict and persist forecasts (EnsemblePredictor -> ForecastService)
4. Resolve expired predictions against GDELT ground truth
5. Run monitoring checks (feed, drift, disk) with alert dispatch
6. Trigger weekly calibration on configured day (Monday by default)

Handles budget exhaustion by queuing overflow questions to PendingQuestion.
Consecutive failure alerting at >= 2 failures sends email via AlertManager.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.services.forecast_service import ForecastService
from src.forecasting.ensemble_predictor import EnsemblePredictor
from src.pipeline.budget_tracker import BudgetTracker
from src.pipeline.outcome_resolver import OutcomeResolver
from src.pipeline.question_generator import GeneratedQuestion, QuestionGenerator

if TYPE_CHECKING:
    from src.calibration.weight_loader import WeightLoader
    from src.calibration.weight_optimizer import CalibrationResult, WeightOptimizer
    from src.monitoring.alert_manager import AlertManager
    from src.monitoring.disk_monitor import DiskMonitor
    from src.monitoring.drift_monitor import DriftMonitor
    from src.monitoring.feed_monitor import FeedMonitor

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a daily pipeline run."""

    questions_generated: int = 0
    forecasts_produced: int = 0
    questions_queued: int = 0
    outcomes_resolved: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = False
    monitor_results: dict[str, Any] = field(default_factory=dict)
    calibration_result: Optional[CalibrationResult] = None


class DailyPipeline:
    """Orchestrate the complete daily forecast cycle.

    The pipeline runs six phases sequentially:
    1. Dequeue pending (carryover) questions from previous runs.
    2. Generate fresh questions from GDELT events via Gemini.
    3. Predict each question via EnsemblePredictor and persist via ForecastService.
    4. Resolve expired predictions against GDELT ground truth.
    5. Run monitoring checks (feed staleness, calibration drift, disk usage).
    6. Trigger weekly calibration on configured day.

    Budget exhaustion during phase 3 queues remaining questions for the next run.

    Attributes:
        question_generator: QuestionGenerator for LLM-based question gen.
        budget_tracker: BudgetTracker for Gemini budget management.
        outcome_resolver: OutcomeResolver for expired prediction resolution.
        ensemble_predictor: EnsemblePredictor for forecast generation.
        _session_factory: SQLAlchemy async session factory.
        alert_manager: Optional AlertManager for email alerts.
        weight_loader: Optional WeightLoader for dynamic alpha resolution.
        weight_optimizer: Optional WeightOptimizer for weekly calibration.
        feed_monitor: Optional FeedMonitor for GDELT staleness detection.
        drift_monitor: Optional DriftMonitor for calibration drift detection.
        disk_monitor: Optional DiskMonitor for disk usage monitoring.
    """

    def __init__(
        self,
        question_generator: QuestionGenerator,
        budget_tracker: BudgetTracker,
        outcome_resolver: OutcomeResolver,
        ensemble_predictor: EnsemblePredictor,
        async_session_factory: async_sessionmaker[AsyncSession],
        alert_manager: Optional[AlertManager] = None,
        weight_loader: Optional[WeightLoader] = None,
        weight_optimizer: Optional[WeightOptimizer] = None,
        feed_monitor: Optional[FeedMonitor] = None,
        drift_monitor: Optional[DriftMonitor] = None,
        disk_monitor: Optional[DiskMonitor] = None,
        calibration_day: int = 0,
    ) -> None:
        self.question_generator = question_generator
        self.budget_tracker = budget_tracker
        self.outcome_resolver = outcome_resolver
        self.ensemble_predictor = ensemble_predictor
        self._session_factory = async_session_factory
        self._consecutive_failures: int = 0

        # Monitoring subsystems (all optional for backward compat)
        self.alert_manager = alert_manager
        self.weight_loader = weight_loader
        self.weight_optimizer = weight_optimizer
        self.feed_monitor = feed_monitor
        self.drift_monitor = drift_monitor
        self.disk_monitor = disk_monitor
        self._calibration_day = calibration_day  # 0=Monday (weekday index)

    async def run_daily(
        self,
        max_questions: int | None = None,
        skip_outcomes: bool = False,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute the complete daily forecast cycle.

        Args:
            max_questions: Override maximum questions to process.
            skip_outcomes: Skip phase 4 (outcome resolution).
            dry_run: Generate questions but don't predict or persist.

        Returns:
            PipelineResult with counts and success status.
        """
        start = time.monotonic()
        result = PipelineResult()
        all_questions: list[tuple[GeneratedQuestion, int | None]] = []
        # tuple: (question, pending_question_id or None for fresh)

        try:
            # Phase 1: Dequeue carryover questions
            logger.info("Phase 1: Dequeuing carryover questions")
            pending = await self.budget_tracker.dequeue_pending(limit=10)
            for p in pending:
                q = GeneratedQuestion(
                    question=p.question,
                    country_iso=p.country_iso or "XX",
                    horizon_days=p.horizon_days,
                    category=p.category,
                )
                all_questions.append((q, p.id))
            logger.info("Dequeued %d carryover questions", len(pending))

            # Phase 2: Generate fresh questions (budget permitting)
            budget_remaining = await self.budget_tracker.check_budget()
            logger.info("Phase 2: Budget remaining = %d", budget_remaining)

            if budget_remaining > len(all_questions):
                fresh_limit = budget_remaining - len(all_questions)
                if max_questions is not None:
                    fresh_limit = min(fresh_limit, max_questions - len(all_questions))
                    fresh_limit = max(0, fresh_limit)

                if fresh_limit > 0:
                    fresh = await self.question_generator.generate_questions(
                        n_questions=fresh_limit
                    )
                    result.questions_generated = len(fresh)
                    for q in fresh:
                        all_questions.append((q, None))
                    logger.info("Generated %d fresh questions", len(fresh))
            else:
                logger.warning("Budget exhausted, skipping fresh question generation")

            if max_questions is not None:
                all_questions = all_questions[:max_questions]

            if dry_run:
                logger.info("DRY RUN: skipping prediction and persistence")
                result.questions_generated = len(all_questions)
                result.success = True
                result.duration_seconds = time.monotonic() - start
                return result

            # Phase 3: Predict and persist
            logger.info("Phase 3: Predicting %d questions", len(all_questions))
            for question, pending_id in all_questions:
                # Check budget before each prediction (Gemini is called by EnsemblePredictor)
                remaining = await self.budget_tracker.check_budget()
                if remaining <= 0:
                    logger.warning(
                        "Budget exhausted mid-pipeline, queuing remaining questions"
                    )
                    # Queue this and all remaining questions
                    idx = all_questions.index((question, pending_id))
                    for q, pid in all_questions[idx:]:
                        if pid is None:  # Only queue fresh questions, carryover stays
                            await self.budget_tracker.queue_question(q)
                            result.questions_queued += 1
                    break

                try:
                    # Resolve dynamic alpha if weight_loader is available
                    alpha_override = await self._resolve_alpha(question.category)

                    # EnsemblePredictor.predict() is synchronous
                    predict_kwargs: dict[str, Any] = {
                        "question": question.question,
                        "category": question.category,
                    }
                    if alpha_override is not None:
                        predict_kwargs["alpha_override"] = alpha_override

                    ensemble_pred, forecast_output = await asyncio.to_thread(
                        self.ensemble_predictor.predict,
                        **predict_kwargs,
                    )

                    # Persist via ForecastService
                    async with self._session_factory() as session:
                        service = ForecastService(session)
                        await service.persist_forecast(
                            forecast_output=forecast_output,
                            ensemble_prediction=ensemble_pred,
                            country_iso=question.country_iso,
                            horizon_days=question.horizon_days,
                        )
                        await session.commit()

                    # Increment budget counter
                    await self.budget_tracker.increment()

                    # Mark carryover question as completed
                    if pending_id is not None:
                        await self.budget_tracker.mark_completed(pending_id)

                    result.forecasts_produced += 1
                    logger.info(
                        "Forecast produced: p=%.3f alpha=%.4f country=%s q='%s'",
                        forecast_output.probability,
                        alpha_override if alpha_override is not None else self.ensemble_predictor.alpha,
                        question.country_iso,
                        question.question[:60],
                    )

                except Exception as exc:
                    error_msg = f"Prediction failed for '{question.question[:50]}': {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            # Phase 4: Resolve outcomes
            if not skip_outcomes:
                logger.info("Phase 4: Resolving expired predictions")
                try:
                    outcomes = await self.outcome_resolver.resolve_expired_predictions()
                    result.outcomes_resolved = len(outcomes)
                    logger.info("Resolved %d expired predictions", len(outcomes))
                except Exception as exc:
                    error_msg = f"Outcome resolution failed: {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            # Phase 5: Run monitors
            logger.info("Phase 5: Running monitoring checks")
            result.monitor_results = await self._run_monitors()

            # Phase 6: Weekly calibration
            logger.info("Phase 6: Checking weekly calibration schedule")
            result.calibration_result = await self._maybe_run_calibration()

            result.success = len(result.errors) == 0
            result.duration_seconds = time.monotonic() - start

            logger.info(
                "Pipeline complete: generated=%d, produced=%d, queued=%d, "
                "resolved=%d, monitors=%d, calibrated=%s, errors=%d, duration=%.1fs",
                result.questions_generated,
                result.forecasts_produced,
                result.questions_queued,
                result.outcomes_resolved,
                len(result.monitor_results),
                "yes" if result.calibration_result is not None else "no",
                len(result.errors),
                result.duration_seconds,
            )
            return result

        except Exception as exc:
            result.errors.append(f"Pipeline failed: {exc}")
            result.duration_seconds = time.monotonic() - start
            logger.error("Pipeline failed with unhandled exception: %s", exc)
            raise

    async def _resolve_alpha(self, category: str | None) -> float | None:
        """Resolve dynamic alpha from WeightLoader if available.

        Args:
            category: Heuristic category (conflict/diplomatic/economic).

        Returns:
            Resolved alpha float, or None if WeightLoader is not configured.
        """
        if self.weight_loader is None:
            return None

        try:
            alpha = await self.weight_loader.resolve_alpha(
                keyword_category=category,
            )
            return alpha
        except Exception as exc:
            logger.warning(
                "WeightLoader.resolve_alpha failed, using default alpha: %s", exc,
            )
            return None

    async def _run_monitors(self) -> dict[str, Any]:
        """Run all configured monitors and dispatch alerts.

        Each monitor runs independently; individual failures are logged
        but do not block other monitors. Results are keyed by monitor name.

        Returns:
            Dict of monitor_name -> status dict.
        """
        results: dict[str, Any] = {}

        if self.alert_manager is None:
            logger.debug("No AlertManager configured, skipping monitors")
            return results

        # Feed staleness
        if self.feed_monitor is not None:
            try:
                results["feed"] = await self.feed_monitor.check_and_alert(
                    self.alert_manager
                )
            except Exception as exc:
                logger.error("FeedMonitor check failed: %s", exc)
                results["feed"] = {"error": str(exc)[:200]}

        # Calibration drift
        if self.drift_monitor is not None:
            try:
                results["drift"] = await self.drift_monitor.check_and_alert(
                    self.alert_manager
                )
            except Exception as exc:
                logger.error("DriftMonitor check failed: %s", exc)
                results["drift"] = {"error": str(exc)[:200]}

        # Disk usage
        if self.disk_monitor is not None:
            try:
                results["disk"] = await self.disk_monitor.check_and_alert(
                    self.alert_manager
                )
            except Exception as exc:
                logger.error("DiskMonitor check failed: %s", exc)
                results["disk"] = {"error": str(exc)[:200]}

        logger.info(
            "Monitors complete: %d checks ran",
            len(results),
        )
        return results

    async def _maybe_run_calibration(self) -> CalibrationResult | None:
        """Trigger weekly calibration if today is the configured day.

        Runs WeightOptimizer.run_weekly_calibration() on the configured
        weekday (default Monday=0). On completion, refreshes the
        WeightLoader cache to pick up new weights immediately.

        Guardrail violations in CalibrationResult.flagged_weights trigger
        an alert via AlertManager if available.

        Returns:
            CalibrationResult if calibration ran, None otherwise.
        """
        if self.weight_optimizer is None:
            logger.debug("No WeightOptimizer configured, skipping calibration")
            return None

        today_weekday = datetime.now(timezone.utc).weekday()
        if today_weekday != self._calibration_day:
            logger.debug(
                "Today is weekday %d, calibration day is %d -- skipping",
                today_weekday,
                self._calibration_day,
            )
            return None

        logger.info("Weekly calibration triggered (weekday=%d)", today_weekday)

        try:
            cal_result = await self.weight_optimizer.run_weekly_calibration()
        except Exception as exc:
            logger.error("Weekly calibration failed: %s", exc)
            if self.alert_manager is not None:
                await self.alert_manager.send_alert(
                    "calibration_error",
                    "Weekly Calibration Failed",
                    f"WeightOptimizer.run_weekly_calibration() raised:\n{exc}",
                )
            return None

        logger.info(
            "Calibration complete: %d applied, %d held",
            len(cal_result.applied_weights),
            len(cal_result.held_weights),
        )

        # Alert on guardrail violations
        if cal_result.flagged_weights and self.alert_manager is not None:
            flagged_summary = "\n".join(
                f"  {code}: {reason}"
                for code, reason in cal_result.flagged_weights.items()
            )
            await self.alert_manager.send_alert(
                "calibration_guardrail",
                "Calibration Guardrail Violations",
                f"The following weights exceeded deviation thresholds:\n\n"
                f"{flagged_summary}\n\n"
                f"Applied: {len(cal_result.applied_weights)}, "
                f"Held: {len(cal_result.held_weights)}",
            )

        # Refresh weight_loader cache to pick up new weights
        if self.weight_loader is not None:
            try:
                await self.weight_loader.load_weights()
                logger.info("WeightLoader cache refreshed after calibration")
            except Exception as exc:
                logger.warning("Failed to refresh WeightLoader cache: %s", exc)

        return cal_result

    async def run_with_retry(
        self,
        max_retries: int = 2,
        retry_delay_seconds: float = 300.0,
        **kwargs,
    ) -> PipelineResult:
        """Run the daily pipeline with retry on failure.

        On success, resets the consecutive failure counter.
        On failure, increments the counter and emits CRITICAL alert at >= 2
        (both as a log and via AlertManager email if configured).
        Retries up to max_retries with retry_delay_seconds between attempts.

        Args:
            max_retries: Maximum retry attempts.
            retry_delay_seconds: Delay between retries in seconds.
            **kwargs: Passed to run_daily().

        Returns:
            PipelineResult from the first successful run, or the last failed run.
        """
        last_result = PipelineResult()

        for attempt in range(max_retries + 1):
            try:
                result = await self.run_daily(**kwargs)
                if result.success:
                    self._consecutive_failures = 0
                    return result
                last_result = result
            except Exception as exc:
                last_result = PipelineResult(
                    errors=[str(exc)],
                    success=False,
                )

            self._consecutive_failures += 1
            logger.error(
                "Pipeline attempt %d/%d failed (consecutive failures: %d)",
                attempt + 1,
                max_retries + 1,
                self._consecutive_failures,
            )

            if self._consecutive_failures >= 2:
                logger.critical(
                    "ALERT: %d consecutive daily pipeline failures. "
                    "Manual investigation required.",
                    self._consecutive_failures,
                )
                # Send email alert for consecutive failures
                if self.alert_manager is not None:
                    errors_summary = "\n".join(
                        f"  - {e}" for e in last_result.errors[:5]
                    )
                    await self.alert_manager.send_alert(
                        "pipeline_consecutive_failure",
                        f"Pipeline: {self._consecutive_failures} Consecutive Failures",
                        f"The daily forecast pipeline has failed "
                        f"{self._consecutive_failures} consecutive times.\n\n"
                        f"Recent errors:\n{errors_summary}\n\n"
                        f"Manual investigation required.",
                    )

            if attempt < max_retries:
                logger.info(
                    "Retrying in %.0f seconds...", retry_delay_seconds
                )
                await asyncio.sleep(retry_delay_seconds)

        return last_result
