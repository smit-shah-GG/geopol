"""
Daily operational digest email builder.

Assembles a structured plain-text summary of forecasting system operations:
predictions, accuracy, feed health, budget status, Polymarket data, and active
alerts. Delivered via AlertManager with "daily_digest" alert type.

Usage::

    digest = DigestBuilder(alert_manager)
    sent = await digest.send_daily_digest(
        pipeline_result=...,
        feed_status=...,
        drift_status=...,
        budget_status=...,
        polymarket_summary=...,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.monitoring.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class DigestBuilder:
    """Assembles and sends the daily operational digest email.

    Consumes status dicts from various monitoring subsystems and formats
    them into a single structured plain-text email.

    Args:
        alert_manager: AlertManager instance for SMTP delivery.
    """

    def __init__(self, alert_manager: AlertManager) -> None:
        self._alert_manager = alert_manager

    async def send_daily_digest(
        self,
        pipeline_result: dict[str, Any] | None = None,
        feed_status: dict[str, Any] | None = None,
        drift_status: dict[str, Any] | None = None,
        budget_status: dict[str, Any] | None = None,
        polymarket_summary: dict[str, Any] | None = None,
    ) -> bool:
        """Build and send the daily digest email.

        All parameters are optional -- missing sections are omitted with
        a "No data available" placeholder. This ensures the digest is
        always sendable even if some subsystems failed to report.

        Args:
            pipeline_result: Forecast pipeline output (count, errors, timing).
            feed_status: RSS/GDELT feed health from FeedMonitor.
            drift_status: Calibration drift metrics from DriftMonitor.
            budget_status: API budget usage from BudgetMonitor.
            polymarket_summary: Polymarket comparison summary stats.

        Returns:
            True if the email was sent successfully, False otherwise.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

        # Extract headline numbers for subject line
        forecast_count = 0
        brier_score = 0.0

        if pipeline_result:
            forecast_count = pipeline_result.get("predictions_created", 0)

        if drift_status:
            brier_score = drift_status.get("rolling_brier", 0.0) or 0.0

        subject = f"Daily Digest: {date_str} -- {forecast_count} forecasts, Brier {brier_score:.3f}"

        # Build sections
        sections: list[str] = []
        sections.append(f"GEOPOL DAILY DIGEST -- {date_str}")
        sections.append("=" * 60)
        sections.append("")

        sections.append(self._build_forecasts_section(pipeline_result))
        sections.append(self._build_accuracy_section(drift_status))
        sections.append(self._build_operations_section(feed_status, budget_status))
        sections.append(self._build_polymarket_section(polymarket_summary))
        sections.append(self._build_alerts_section(pipeline_result, feed_status))

        sections.append("-" * 60)
        sections.append(f"Generated: {now.isoformat()}")
        sections.append("Geopol Forecasting System v2.0")

        body = "\n".join(sections)

        sent = await self._alert_manager.send_alert(
            alert_type="daily_digest",
            subject=subject,
            body=body,
        )

        if sent:
            logger.info("Daily digest sent for %s", date_str)
        else:
            logger.warning("Daily digest not sent (suppressed or failed)")

        return sent

    @staticmethod
    def _build_forecasts_section(
        pipeline_result: dict[str, Any] | None,
    ) -> str:
        """Format the FORECASTS section."""
        lines = ["FORECASTS", "-" * 40]

        if pipeline_result is None:
            lines.append("  No pipeline data available.")
            lines.append("")
            return "\n".join(lines)

        created = pipeline_result.get("predictions_created", 0)
        resolved = pipeline_result.get("predictions_resolved", 0)
        errors = pipeline_result.get("errors", 0)
        duration_s = pipeline_result.get("duration_seconds", 0)

        lines.append(f"  New predictions:   {created}")
        lines.append(f"  Resolved:          {resolved}")
        lines.append(f"  Errors:            {errors}")
        if duration_s:
            lines.append(f"  Pipeline duration: {duration_s:.1f}s")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_accuracy_section(
        drift_status: dict[str, Any] | None,
    ) -> str:
        """Format the ACCURACY section."""
        lines = ["ACCURACY", "-" * 40]

        if drift_status is None:
            lines.append("  No accuracy data available.")
            lines.append("")
            return "\n".join(lines)

        rolling_brier = drift_status.get("rolling_brier")
        window_size = drift_status.get("window_size", 0)
        drift_detected = drift_status.get("drift_detected", False)
        baseline_brier = drift_status.get("baseline_brier")

        if rolling_brier is not None:
            lines.append(f"  Rolling Brier:     {rolling_brier:.4f}")
        else:
            lines.append("  Rolling Brier:     N/A (insufficient data)")

        if baseline_brier is not None:
            lines.append(f"  Baseline Brier:    {baseline_brier:.4f}")

        lines.append(f"  Window size:       {window_size} predictions")
        lines.append(f"  Drift detected:    {'YES' if drift_detected else 'No'}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_operations_section(
        feed_status: dict[str, Any] | None,
        budget_status: dict[str, Any] | None,
    ) -> str:
        """Format the OPERATIONS section (feeds + budget)."""
        lines = ["OPERATIONS", "-" * 40]

        # Feed health
        if feed_status:
            healthy = feed_status.get("healthy_feeds", 0)
            stale = feed_status.get("stale_feeds", 0)
            total = feed_status.get("total_feeds", 0)
            lines.append(f"  Feed health:       {healthy}/{total} healthy, {stale} stale")
            last_ingest = feed_status.get("last_ingest_at")
            if last_ingest:
                lines.append(f"  Last ingest:       {last_ingest}")
        else:
            lines.append("  Feed health:       No data")

        lines.append("")

        # Budget
        if budget_status:
            used = budget_status.get("daily_used", 0)
            limit = budget_status.get("daily_limit", 0)
            remaining = budget_status.get("daily_remaining", 0)
            pct = (used / limit * 100) if limit > 0 else 0
            lines.append(f"  API budget:        {used}/{limit} used ({pct:.0f}%)")
            lines.append(f"  Remaining:         {remaining}")
        else:
            lines.append("  API budget:        No data")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_polymarket_section(
        polymarket_summary: dict[str, Any] | None,
    ) -> str:
        """Format the POLYMARKET COMPARISON section."""
        lines = ["POLYMARKET COMPARISON", "-" * 40]

        if polymarket_summary is None:
            lines.append("  Polymarket comparison not enabled.")
            lines.append("")
            return "\n".join(lines)

        active = polymarket_summary.get("active_count", 0)
        resolved = polymarket_summary.get("resolved_count", 0)
        geopol_brier = polymarket_summary.get("geopol_avg_brier")
        pm_brier = polymarket_summary.get("polymarket_avg_brier")
        wins = polymarket_summary.get("geopol_wins", 0)

        lines.append(f"  Active matches:    {active}")
        lines.append(f"  Resolved:          {resolved}")

        if geopol_brier is not None and pm_brier is not None:
            leader = "Geopol" if geopol_brier < pm_brier else "Market"
            lines.append(f"  Geopol avg Brier:  {geopol_brier:.4f}")
            lines.append(f"  Market avg Brier:  {pm_brier:.4f}")
            lines.append(f"  Geopol wins:       {wins}")
            lines.append(f"  Leader:            {leader}")
        else:
            lines.append("  Brier comparison:  Insufficient resolved data")

        if active < 5:
            lines.append("  NOTE: Seeking more geopolitical market overlaps")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_alerts_section(
        pipeline_result: dict[str, Any] | None,
        feed_status: dict[str, Any] | None,
    ) -> str:
        """Format the ALERTS TODAY section."""
        lines = ["ALERTS TODAY", "-" * 40]

        alerts: list[str] = []

        if pipeline_result:
            errors = pipeline_result.get("errors", 0)
            if errors > 0:
                alerts.append(f"  [WARN] {errors} pipeline error(s) occurred")

        if feed_status:
            stale = feed_status.get("stale_feeds", 0)
            if stale > 0:
                alerts.append(f"  [WARN] {stale} feed(s) stale")

        if not alerts:
            lines.append("  No alerts.")
        else:
            lines.extend(alerts)

        lines.append("")
        return "\n".join(lines)
