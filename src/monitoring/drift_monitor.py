"""
Rolling Brier score computation and calibration drift detection.

Replaces the legacy DriftDetector (JSON file-backed, synchronous) with a
PostgreSQL-backed async implementation.  Queries predictions joined with
outcome_records to compute rolling and baseline Brier scores, then
detects drift when the rolling score degrades beyond a configurable
threshold relative to the all-time baseline.

Minimum sample requirement (default 20) prevents noisy alerts from
small windows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, select
from sqlalchemy import cast as sa_cast
from sqlalchemy import func

from src.db.models import OutcomeRecord, Prediction

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.monitoring.alert_manager import AlertManager
    from src.settings import Settings

logger = logging.getLogger(__name__)

# Minimum resolved predictions required for reliable drift detection.
_MIN_SAMPLES = 20


class DriftMonitor:
    """PostgreSQL-backed calibration drift detector.

    Compares a rolling Brier score (default 30-day window) against the
    all-time baseline.  Drift is declared when:

        rolling > (1 + threshold_pct / 100) * baseline

    Attributes:
        _threshold_pct: Percentage degradation that triggers a drift alert.
    """

    def __init__(
        self,
        async_session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = async_session_factory
        self._threshold_pct: float = settings.drift_threshold_pct

    async def compute_rolling_brier(
        self, window_days: int = 30
    ) -> dict[str, Any]:
        """Compute Brier score over a rolling time window.

        Brier score = mean((predicted_prob - outcome)^2) where outcome
        is 0 or 1.  Joins predictions with outcome_records on
        prediction_id and filters by resolution_date.

        Args:
            window_days: Size of the rolling window in days.

        Returns:
            Dict with brier_score, sample_count, window_days,
            sufficient_data.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        try:
            async with self._session_factory() as session:
                # Brier = mean((p - o)^2)
                stmt = (
                    select(
                        func.avg(
                            func.pow(
                                sa_cast(Prediction.probability, Float)
                                - sa_cast(OutcomeRecord.outcome, Float),
                                2,
                            )
                        ).label("brier"),
                        func.count().label("cnt"),
                    )
                    .join(
                        OutcomeRecord,
                        OutcomeRecord.prediction_id == Prediction.id,
                    )
                    .where(OutcomeRecord.resolution_date >= cutoff)
                )
                result = await session.execute(stmt)
                row = result.one()

                brier = float(row.brier) if row.brier is not None else None
                count = int(row.cnt)
        except Exception as exc:
            logger.error("Failed to compute rolling Brier score: %s", exc)
            return {
                "brier_score": None,
                "sample_count": 0,
                "window_days": window_days,
                "sufficient_data": False,
                "error": str(exc)[:200],
            }

        return {
            "brier_score": round(brier, 6) if brier is not None else None,
            "sample_count": count,
            "window_days": window_days,
            "sufficient_data": count >= _MIN_SAMPLES,
        }

    async def compute_baseline_brier(self) -> float | None:
        """Compute the all-time Brier score from all resolved predictions.

        Returns:
            All-time Brier score, or None if no resolved predictions exist.
        """
        try:
            async with self._session_factory() as session:
                stmt = select(
                    func.avg(
                        func.pow(
                            sa_cast(Prediction.probability, Float)
                            - sa_cast(OutcomeRecord.outcome, Float),
                            2,
                        )
                    )
                ).join(
                    OutcomeRecord,
                    OutcomeRecord.prediction_id == Prediction.id,
                )
                result = await session.execute(stmt)
                val = result.scalar_one_or_none()
                return round(float(val), 6) if val is not None else None
        except Exception as exc:
            logger.error("Failed to compute baseline Brier score: %s", exc)
            return None

    async def check_drift(self) -> dict[str, Any]:
        """Detect calibration drift by comparing rolling vs baseline.

        Drift is declared when:
            rolling_brier > (1 + threshold_pct/100) * baseline_brier

        Returns:
            Dict with drift_detected, rolling_brier, baseline_brier,
            degradation_pct, sufficient_data.
        """
        rolling = await self.compute_rolling_brier()
        baseline = await self.compute_baseline_brier()

        rolling_brier = rolling["brier_score"]
        sufficient = rolling["sufficient_data"]

        # Cannot detect drift without both scores and enough data
        if rolling_brier is None or baseline is None or not sufficient:
            return {
                "drift_detected": False,
                "rolling_brier": rolling_brier,
                "baseline_brier": baseline,
                "degradation_pct": None,
                "sufficient_data": sufficient,
                "sample_count": rolling["sample_count"],
            }

        # Avoid division by zero on a perfect baseline
        if baseline == 0.0:
            degradation_pct = 0.0 if rolling_brier == 0.0 else float("inf")
        else:
            degradation_pct = ((rolling_brier - baseline) / baseline) * 100.0

        drift_detected = rolling_brier > (1 + self._threshold_pct / 100) * baseline

        if drift_detected:
            logger.warning(
                "Calibration drift detected: rolling=%.4f baseline=%.4f (+%.1f%%)",
                rolling_brier,
                baseline,
                degradation_pct,
            )

        return {
            "drift_detected": drift_detected,
            "rolling_brier": rolling_brier,
            "baseline_brier": baseline,
            "degradation_pct": round(degradation_pct, 2),
            "sufficient_data": sufficient,
            "sample_count": rolling["sample_count"],
        }

    async def check_and_alert(self, alert_manager: AlertManager) -> dict[str, Any]:
        """Check for drift and send an alert if detected.

        Args:
            alert_manager: AlertManager instance for email dispatch.

        Returns:
            The drift status dict (same as check_drift).
        """
        status = await self.check_drift()

        if status["drift_detected"]:
            body = (
                f"Calibration drift detected.\n\n"
                f"Rolling Brier (30d): {status['rolling_brier']:.4f}\n"
                f"Baseline Brier:      {status['baseline_brier']:.4f}\n"
                f"Degradation:         {status['degradation_pct']:.1f}%\n"
                f"Threshold:           {self._threshold_pct:.1f}%\n"
                f"Sample count:        {status['sample_count']}\n\n"
                f"Action: Consider triggering calibration weight recompute."
            )
            await alert_manager.send_alert(
                "calibration_drift", "Calibration Drift Detected", body
            )

        return status
