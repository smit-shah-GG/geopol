"""
L-BFGS-B alpha weight optimizer and weekly calibration pipeline.

Replaces the fixed alpha=0.6 ensemble blend with per-CAMEO dynamic
weights optimized from accumulated prediction-outcome data.

The optimization objective is Brier score minimization:
    BS(alpha) = (1/N) * sum((alpha * p_llm_i + (1-alpha) * p_tkg_i - o_i)^2)

L-BFGS-B is used because:
- Single bounded variable (alpha in [0, 1]) -- gradient-based is overkill
  but L-BFGS-B is robust, fast, and handles bounds natively.
- Consistent with the existing temperature_scaler.py scipy.optimize pattern.
- Deterministic convergence for this convex problem.

Weekly calibration pipeline (run_weekly_calibration):
1. Query all resolved prediction-outcome pairs from PostgreSQL.
2. Group by CAMEO root code and optimize per-group.
3. Aggregate under-sampled codes to super-categories.
4. Compute global alpha from all resolved pairs.
5. Apply deviation guardrails (20% max change).
6. Persist results to calibration_weights + calibration_weight_history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.calibration.priors import (
    CAMEO_TO_SUPER,
    COLD_START_PRIORS,
    SUPER_CATEGORIES,
)
from src.db.models import (
    CalibrationWeight,
    CalibrationWeightHistory,
    OutcomeRecord,
    Prediction,
)
from src.settings import Settings

logger = logging.getLogger(__name__)


def optimize_alpha_for_category(
    outcomes: list[float],
    tkg_probs: list[float],
    llm_probs: list[float],
) -> tuple[float, float]:
    """Minimize Brier score over alpha for a set of prediction-outcome pairs.

    Finds the alpha that minimizes:
        BS(alpha) = mean((alpha * llm + (1-alpha) * tkg - outcome)^2)

    Args:
        outcomes: Ground-truth outcomes (0.0 or 1.0 each).
        tkg_probs: TKG model probabilities.
        llm_probs: LLM model probabilities.

    Returns:
        Tuple of (optimal_alpha, achieved_brier_score). Alpha is the LLM
        weight in [0.0, 1.0].

    Raises:
        ValueError: If input arrays have mismatched lengths or are empty.
    """
    if not (len(outcomes) == len(tkg_probs) == len(llm_probs)):
        raise ValueError(
            f"Input arrays must have same length: outcomes={len(outcomes)}, "
            f"tkg={len(tkg_probs)}, llm={len(llm_probs)}"
        )
    if len(outcomes) == 0:
        raise ValueError("Cannot optimize with empty data")

    o = np.asarray(outcomes, dtype=np.float64)
    tkg = np.asarray(tkg_probs, dtype=np.float64)
    llm = np.asarray(llm_probs, dtype=np.float64)

    def brier_loss(alpha_arr: np.ndarray) -> float:
        a = alpha_arr[0]
        blended = a * llm + (1.0 - a) * tkg
        return float(np.mean((blended - o) ** 2))

    result = minimize(
        brier_loss,
        x0=[0.6],
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)],
        options={"maxiter": 200, "ftol": 1e-10},
    )

    optimal_alpha = float(result.x[0])
    achieved_brier = float(result.fun)

    logger.debug(
        "L-BFGS-B converged: alpha=%.4f, brier=%.6f, success=%s, nit=%d",
        optimal_alpha,
        achieved_brier,
        result.success,
        result.nit,
    )

    return optimal_alpha, achieved_brier


@dataclass
class CalibrationResult:
    """Output of a weekly calibration run.

    Partitions computed weights into applied (auto-written to
    calibration_weights) and held (flagged for manual review due to
    guardrail violations).
    """

    # cameo_code -> (alpha, brier_score, sample_size)
    new_weights: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    # cameo_code -> flag_reason
    flagged_weights: dict[str, str] = field(default_factory=dict)
    # Subset of new_weights that were auto-applied
    applied_weights: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    # Subset of new_weights that were held back
    held_weights: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WeightOptimizer:
    """Orchestrates weekly calibration: query, optimize, guardrail, persist."""

    def __init__(
        self,
        async_session_factory: Any,
        settings: Settings,
    ) -> None:
        self._session_factory = async_session_factory
        self._min_samples = settings.calibration_min_samples
        self._max_deviation = settings.calibration_max_deviation

    async def run_weekly_calibration(self) -> CalibrationResult:
        """Execute the full calibration pipeline.

        Steps:
            1. Fetch resolved prediction-outcome pairs.
            2. Group by CAMEO root code, optimize per-group.
            3. Aggregate under-sampled codes to super-categories.
            4. Compute global alpha.
            5. Apply guardrails.
            6. Persist results.

        Returns:
            CalibrationResult with all computed, applied, and held weights.
        """
        async with self._session_factory() as session:
            pairs = await self._fetch_resolved_pairs(session)

        if not pairs:
            logger.warning(
                "No resolved prediction-outcome pairs found; "
                "skipping calibration run"
            )
            return CalibrationResult()

        logger.info("Calibration: %d resolved pairs fetched", len(pairs))

        # ---- Step 2: Group by CAMEO root code and optimize ----
        new_weights: dict[str, tuple[float, float, int]] = {}
        under_sampled_by_super: dict[str, list[dict]] = {
            sc: [] for sc in SUPER_CATEGORIES
        }

        cameo_groups: dict[str, list[dict]] = {}
        for p in pairs:
            code = p.get("cameo_root_code")
            if code is None:
                continue
            cameo_groups.setdefault(code, []).append(p)

        for code, group in cameo_groups.items():
            if len(group) >= self._min_samples:
                alpha, brier = self._safe_optimize(group)
                if alpha is not None:
                    new_weights[code] = (alpha, brier, len(group))
            else:
                # Aggregate to super-category for later optimization
                super_cat = CAMEO_TO_SUPER.get(code)
                if super_cat:
                    under_sampled_by_super[super_cat].extend(group)

        # ---- Step 3: Optimize super-category aggregates ----
        for super_cat, group in under_sampled_by_super.items():
            super_key = f"super:{super_cat}"
            # Only optimize if we don't already have enough per-code data
            # and the aggregate meets minimum samples
            if len(group) >= self._min_samples:
                alpha, brier = self._safe_optimize(group)
                if alpha is not None:
                    new_weights[super_key] = (alpha, brier, len(group))

        # ---- Step 4: Compute global alpha from ALL pairs ----
        if len(pairs) >= self._min_samples:
            alpha, brier = self._safe_optimize(pairs)
            if alpha is not None:
                new_weights["global"] = (alpha, brier, len(pairs))

        if not new_weights:
            logger.warning("No weights computed (all optimizations failed or insufficient data)")
            return CalibrationResult()

        # ---- Step 5: Apply guardrails ----
        async with self._session_factory() as session:
            current_weights = await self._load_current_weights(session)
            flagged = self._check_guardrails(
                new_weights, current_weights, self._max_deviation
            )

            # ---- Step 6: Partition and persist ----
            result = CalibrationResult(new_weights=new_weights, flagged_weights=flagged)

            for code, (alpha, brier, n) in new_weights.items():
                is_flagged = code in flagged
                if is_flagged:
                    result.held_weights[code] = (alpha, brier, n)
                else:
                    result.applied_weights[code] = (alpha, brier, n)

            # Auto-apply unflagged weights
            for code, (alpha, brier, n) in result.applied_weights.items():
                await self._upsert_weight(session, code, alpha, n, brier)

            # Persist ALL to history (flagged and unflagged)
            for code, (alpha, brier, n) in new_weights.items():
                is_flagged = code in flagged
                await self._write_history(
                    session,
                    code,
                    alpha,
                    n,
                    brier,
                    auto_applied=not is_flagged,
                    flagged=is_flagged,
                    flag_reason=flagged.get(code),
                )

            await session.commit()

        logger.info(
            "Calibration complete: %d weights computed, %d applied, %d held",
            len(new_weights),
            len(result.applied_weights),
            len(result.held_weights),
        )

        return result

    async def _fetch_resolved_pairs(
        self, session: AsyncSession
    ) -> list[dict[str, Any]]:
        """Query predictions joined with outcome_records.

        Extracts llm_probability and tkg_probability from the
        ensemble_info_json blob, plus the CAMEO root code and category.

        Returns:
            List of dicts with keys: prediction_id, cameo_root_code,
            llm_probability, tkg_probability, outcome, category.
        """
        stmt = (
            select(
                Prediction.id,
                Prediction.cameo_root_code,
                Prediction.category,
                Prediction.ensemble_info_json,
                OutcomeRecord.outcome,
            )
            .join(
                OutcomeRecord,
                OutcomeRecord.prediction_id == Prediction.id,
            )
        )

        result = await session.execute(stmt)
        rows = result.all()

        pairs: list[dict[str, Any]] = []
        for row in rows:
            pred_id, cameo_code, category, ensemble_json, outcome = row

            # Extract component probabilities from ensemble_info_json
            llm_prob, tkg_prob = self._extract_component_probs(ensemble_json)
            if llm_prob is None or tkg_prob is None:
                logger.debug(
                    "Skipping prediction %s: missing component probabilities "
                    "in ensemble_info_json",
                    pred_id,
                )
                continue

            pairs.append(
                {
                    "prediction_id": pred_id,
                    "cameo_root_code": cameo_code,
                    "llm_probability": llm_prob,
                    "tkg_probability": tkg_prob,
                    "outcome": float(outcome),
                    "category": category,
                }
            )

        return pairs

    @staticmethod
    def _extract_component_probs(
        ensemble_json: dict[str, Any] | str | None,
    ) -> tuple[float | None, float | None]:
        """Extract llm_probability and tkg_probability from ensemble_info_json.

        The blob may be a dict or a JSON string depending on the driver.
        Handles both gracefully.

        Returns:
            (llm_probability, tkg_probability) or (None, None) on failure.
        """
        if ensemble_json is None:
            return None, None

        if isinstance(ensemble_json, str):
            try:
                ensemble_json = json.loads(ensemble_json)
            except (json.JSONDecodeError, TypeError):
                return None, None

        if not isinstance(ensemble_json, dict):
            return None, None

        llm = ensemble_json.get("llm_probability")
        tkg = ensemble_json.get("tkg_probability")

        if llm is None or tkg is None:
            return None, None

        try:
            return float(llm), float(tkg)
        except (ValueError, TypeError):
            return None, None

    @staticmethod
    def _safe_optimize(
        pairs: list[dict[str, Any]],
    ) -> tuple[float | None, float]:
        """Run optimize_alpha_for_category with error handling.

        Returns:
            (alpha, brier) on success, (None, 0.0) on failure.
        """
        outcomes = [p["outcome"] for p in pairs]
        tkg_probs = [p["tkg_probability"] for p in pairs]
        llm_probs = [p["llm_probability"] for p in pairs]

        try:
            return optimize_alpha_for_category(outcomes, tkg_probs, llm_probs)
        except Exception:
            logger.warning(
                "L-BFGS-B optimization failed for %d pairs", len(pairs),
                exc_info=True,
            )
            return None, 0.0

    @staticmethod
    def _check_guardrails(
        new_weights: dict[str, tuple[float, float, int]],
        current_weights: dict[str, float],
        max_deviation: float,
    ) -> dict[str, str]:
        """Flag weights that deviate too far from current values.

        Compares absolute deviation of each new alpha against the current
        alpha. If the deviation exceeds max_deviation (fraction of current
        value), the weight is flagged.

        Args:
            new_weights: cameo_code -> (alpha, brier, sample_size).
            current_weights: cameo_code -> current_alpha.
            max_deviation: Maximum allowed relative deviation (e.g. 0.20).

        Returns:
            Dict of cameo_code -> flag_reason for weights that exceed
            the deviation threshold.
        """
        flagged: dict[str, str] = {}

        for code, (new_alpha, _, _) in new_weights.items():
            current_alpha = current_weights.get(code)
            if current_alpha is None:
                # No existing weight -- no guardrail to apply
                continue

            if current_alpha == 0.0:
                # Avoid division by zero; any nonzero new alpha is a change
                if new_alpha > 0.0:
                    flagged[code] = (
                        f"Current alpha=0.0, new alpha={new_alpha:.4f} "
                        f"(infinite relative deviation)"
                    )
                continue

            relative_deviation = abs(new_alpha - current_alpha) / abs(current_alpha)

            if relative_deviation > max_deviation:
                flagged[code] = (
                    f"Relative deviation {relative_deviation:.2%} exceeds "
                    f"threshold {max_deviation:.0%}: "
                    f"current={current_alpha:.4f} -> new={new_alpha:.4f}"
                )
                logger.info(
                    "Flagged weight %s: %s", code, flagged[code],
                )

        return flagged

    @staticmethod
    async def _load_current_weights(
        session: AsyncSession,
    ) -> dict[str, float]:
        """Load current CalibrationWeight rows as code -> alpha dict."""
        stmt = select(CalibrationWeight.cameo_code, CalibrationWeight.alpha)
        result = await session.execute(stmt)
        return {row[0]: row[1] for row in result.all()}

    @staticmethod
    async def _upsert_weight(
        session: AsyncSession,
        cameo_code: str,
        alpha: float,
        sample_size: int,
        brier_score: float,
    ) -> None:
        """Insert or update a CalibrationWeight row."""
        stmt = select(CalibrationWeight).where(
            CalibrationWeight.cameo_code == cameo_code
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            existing.alpha = alpha
            existing.sample_size = sample_size
            existing.brier_score = brier_score
            existing.updated_at = datetime.now(timezone.utc)
        else:
            session.add(
                CalibrationWeight(
                    cameo_code=cameo_code,
                    alpha=alpha,
                    sample_size=sample_size,
                    brier_score=brier_score,
                )
            )

    @staticmethod
    async def _write_history(
        session: AsyncSession,
        cameo_code: str,
        alpha: float,
        sample_size: int,
        brier_score: float,
        *,
        auto_applied: bool,
        flagged: bool,
        flag_reason: str | None,
    ) -> None:
        """Append a row to calibration_weight_history."""
        session.add(
            CalibrationWeightHistory(
                cameo_code=cameo_code,
                alpha=alpha,
                sample_size=sample_size,
                brier_score=brier_score,
                auto_applied=auto_applied,
                flagged=flagged,
                flag_reason=flag_reason,
            )
        )
