"""
GDELT feed staleness detection.

Queries the most recent successful GDELT ingest run from PostgreSQL and
determines whether the feed is stale based on the configured threshold
(default: 1 hour).  Staleness triggers an email alert via AlertManager.

Designed to be called from a periodic health check loop or the daily
pipeline -- does NOT run a background thread itself.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.db.models import IngestRun

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.monitoring.alert_manager import AlertManager
    from src.settings import Settings

logger = logging.getLogger(__name__)


class FeedMonitor:
    """Detect GDELT feed staleness from the ingest_runs audit trail.

    Attributes:
        _threshold_hours: Maximum acceptable age (hours) of the last
            successful GDELT ingest before declaring staleness.
    """

    def __init__(
        self,
        async_session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self._session_factory = async_session_factory
        self._threshold_hours: float = settings.feed_staleness_hours

    async def check_feed_freshness(self) -> dict[str, Any]:
        """Query the most recent successful GDELT ingest run.

        Returns:
            Dict with keys: stale, hours_since_last, last_run_at,
            threshold_hours.
        """
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(IngestRun)
                    .where(
                        IngestRun.status == "success",
                        IngestRun.daemon_type == "gdelt",
                    )
                    .order_by(IngestRun.completed_at.desc())
                    .limit(1)
                )
                last_run = result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Failed to query ingest_runs: %s", exc)
            return {
                "stale": True,
                "hours_since_last": None,
                "last_run_at": None,
                "threshold_hours": self._threshold_hours,
                "error": str(exc)[:200],
            }

        if last_run is None:
            logger.warning("No successful GDELT ingest runs found -- feed considered stale")
            return {
                "stale": True,
                "hours_since_last": None,
                "last_run_at": None,
                "threshold_hours": self._threshold_hours,
            }

        now = datetime.now(timezone.utc)
        completed = last_run.completed_at
        # Ensure timezone-aware comparison
        if completed is not None and completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        hours_since = (now - completed).total_seconds() / 3600 if completed else None
        stale = hours_since is None or hours_since > self._threshold_hours

        return {
            "stale": stale,
            "hours_since_last": round(hours_since, 2) if hours_since is not None else None,
            "last_run_at": completed.isoformat() if completed else None,
            "threshold_hours": self._threshold_hours,
        }

    async def check_and_alert(self, alert_manager: AlertManager) -> dict[str, Any]:
        """Check feed freshness and send an alert if stale.

        Args:
            alert_manager: AlertManager instance for email dispatch.

        Returns:
            The freshness status dict (same as check_feed_freshness).
        """
        status = await self.check_feed_freshness()

        if status["stale"]:
            hours = status.get("hours_since_last")
            if hours is not None:
                body = (
                    f"GDELT feed has not ingested new data for {hours:.1f} hours.\n"
                    f"Threshold: {self._threshold_hours}h\n"
                    f"Last successful run: {status.get('last_run_at', 'unknown')}"
                )
            else:
                body = (
                    "No successful GDELT ingest runs found in the database.\n"
                    "The feed monitor cannot determine data freshness."
                )

            await alert_manager.send_alert("feed_stale", "GDELT Feed Stale", body)

        return status
