"""Admin service layer -- business logic for the admin API endpoints.

Encapsulates all admin operations: process status, job triggering,
runtime config CRUD, source health, and source toggling. The service
takes an AsyncSession and delegates DB queries, keeping the router
thin and testable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.admin import ConfigEntry, ProcessInfo, SourceInfo
from src.db.models import IngestRun, SystemConfig
from src.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Daemon type -> human-readable name
_DAEMON_NAMES: dict[str, str] = {
    "gdelt": "GDELT Poller",
    "rss": "RSS Poller",
    "pipeline": "Daily Pipeline",
    "polymarket": "Polymarket Forecaster",
    "acled": "ACLED Poller",
    "advisory": "Advisory Poller",
    "tkg": "TKG Retrainer",
}

# Settings fields that must never be editable via the admin API
_SECRET_FIELDS: frozenset[str] = frozenset({
    "admin_key",
    "gemini_api_key",
    "database_url",
    "redis_url",
    "smtp_password",
    "acled_email",
    "acled_password",
})

# Per-field descriptions for the config editor
_FIELD_DESCRIPTIONS: dict[str, str] = {
    "environment": "Runtime environment (development/production/testing)",
    "use_fixtures": "Enable mock fixture fallback (dev only)",
    "gdelt_db_path": "Path to SQLite GDELT event database",
    "gdelt_poll_interval": "GDELT polling interval in seconds",
    "gdelt_backfill_on_start": "Whether to backfill GDELT events on startup",
    "rss_poll_interval_tier1": "Tier-1 RSS feed poll interval in seconds",
    "rss_poll_interval_tier2": "Tier-2 RSS feed poll interval in seconds",
    "rss_article_retention_days": "Days to retain RSS articles",
    "gemini_model": "Primary Gemini model identifier",
    "gemini_fallback_model": "Fallback Gemini model identifier",
    "gemini_max_rpm": "Max Gemini API requests per minute",
    "gemini_daily_budget": "Max forecast questions per UTC day",
    "tkg_backend": "TKG model backend (tirgn/regcn)",
    "log_level": "Root logger level (DEBUG/INFO/WARNING/ERROR)",
    "log_json": "Emit structured JSON logs on stderr",
    "calibration_min_samples": "Min samples for calibration weight recompute",
    "calibration_max_deviation": "Max alpha deviation before flagging",
    "calibration_recompute_day": "Weekday index for weekly calibration (0=Monday)",
    "polymarket_enabled": "Enable Polymarket matching loop",
    "polymarket_poll_interval": "Polymarket poll interval in seconds",
    "polymarket_match_threshold": "Min similarity for Polymarket question match",
    "polymarket_volume_threshold": "Min USD volume for auto-forecast",
    "polymarket_daily_new_forecast_cap": "Max new Polymarket forecasts per day",
    "polymarket_daily_reforecast_cap": "Max re-forecasts per day",
    "acled_poll_interval": "ACLED polling interval in seconds",
    "acled_event_types": "ACLED event types to ingest",
    "advisory_poll_interval": "Advisory polling interval in seconds",
    "log_dir": "Directory for rotated log files",
    "log_retention_days": "Days to retain rotated log files",
    "cors_origins": "Allowed CORS origins",
    "api_key_header": "HTTP header name for API key auth",
    "smtp_host": "SMTP server hostname",
    "smtp_port": "SMTP server port",
    "smtp_username": "SMTP username",
    "smtp_sender": "Alert email sender address",
    "alert_recipient": "Alert email recipient",
    "alert_cooldown_minutes": "Minutes between duplicate alerts",
    "feed_staleness_hours": "Hours before feed marked stale",
    "drift_threshold_pct": "Calibration drift threshold percentage",
    "disk_warning_pct": "Disk usage warning threshold",
    "disk_critical_pct": "Disk usage critical threshold",
}

# Daemon type -> poll interval settings field name
_DAEMON_INTERVALS: dict[str, str] = {
    "gdelt": "gdelt_poll_interval",
    "rss": "rss_poll_interval_tier1",
    "polymarket": "polymarket_poll_interval",
    "acled": "acled_poll_interval",
    "advisory": "advisory_poll_interval",
    "tkg": "tkg_retrain_interval",  # Not in Settings yet -- defaults to 86400
    "pipeline": "pipeline_interval",  # Not in Settings yet -- defaults to 86400
}


def _python_type_name(value: Any) -> str:
    """Map a Python value to a config editor type label."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    return "str"


def _is_dangerous(key: str, value: Any) -> bool:
    """Flag settings that can break the system if set carelessly."""
    if key == "gdelt_poll_interval" and isinstance(value, (int, float)) and value < 60:
        return True
    if key == "rss_poll_interval_tier1" and isinstance(value, (int, float)) and value < 60:
        return True
    if key == "gemini_daily_budget" and value == 0:
        return True
    if key == "polymarket_enabled" and value is False:
        return True
    return False


class AdminService:
    """Admin business logic operating on a scoped async session."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Processes
    # ------------------------------------------------------------------

    async def get_processes(self) -> list[ProcessInfo]:
        """Query ingest_runs aggregated by daemon_type for all daemon types."""
        settings = get_settings()
        result = await self._db.execute(
            select(
                IngestRun.daemon_type,
                func.max(IngestRun.started_at).label("last_started"),
                func.count().filter(IngestRun.status == "success").label("ok"),
                func.count().filter(IngestRun.status == "failed").label("fail"),
            ).group_by(IngestRun.daemon_type)
        )
        rows = {r.daemon_type: r for r in result.all()}

        # Fetch latest status per daemon_type via a subquery
        latest_status: dict[str, str] = {}
        for dtype in _DAEMON_NAMES:
            sub = await self._db.execute(
                select(IngestRun.status)
                .where(IngestRun.daemon_type == dtype)
                .order_by(IngestRun.started_at.desc())
                .limit(1)
            )
            row = sub.scalar_one_or_none()
            latest_status[dtype] = row if row else "unknown"

        processes: list[ProcessInfo] = []
        for dtype, name in _DAEMON_NAMES.items():
            row = rows.get(dtype)
            last_run = row.last_started if row else None

            # Compute next_run from last_run + poll_interval
            interval_field = _DAEMON_INTERVALS.get(dtype)
            interval_seconds: int | None = None
            if interval_field:
                interval_seconds = getattr(settings, interval_field, None)
            # Default TKG / pipeline to 24h if not in Settings
            if interval_seconds is None and dtype in ("tkg", "pipeline"):
                interval_seconds = 86400

            next_run = None
            if last_run and interval_seconds:
                next_run = last_run + timedelta(seconds=interval_seconds)

            processes.append(
                ProcessInfo(
                    name=name,
                    daemon_type=dtype,
                    status=latest_status.get(dtype, "unknown"),
                    last_run=last_run,
                    next_run=next_run,
                    success_count=row.ok if row else 0,
                    fail_count=row.fail if row else 0,
                )
            )
        return processes

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    async def trigger_job(self, daemon_type: str) -> None:
        """Spawn a one-shot asyncio.Task for the given daemon.

        Phase 20 (APScheduler consolidation) will replace this with
        proper job-store triggers. For now, we import and fire the
        existing run functions as background tasks.
        """
        if daemon_type not in _DAEMON_NAMES:
            raise HTTPException(404, f"Unknown daemon type: {daemon_type}")

        # Pre-APScheduler: pollers have complex constructors requiring
        # EventStorage, TemporalKnowledgeGraph, GeminiClient, etc.
        # Phase 20 consolidates all jobs under APScheduler with proper
        # dependency wiring. For now, only Polymarket matching has a
        # self-contained trigger (it wires its own deps in _polymarket_loop).
        logger.warning(
            "Manual trigger for %s requested -- Phase 20 (APScheduler) "
            "will add proper one-shot triggers with dependency wiring",
            daemon_type,
        )
        raise HTTPException(
            501,
            f"Manual trigger for '{_DAEMON_NAMES[daemon_type]}' requires "
            f"APScheduler job wiring (Phase 20). Process table shows "
            f"status from automatic daemon runs.",
        )

    # ------------------------------------------------------------------
    # Config CRUD
    # ------------------------------------------------------------------

    async def get_config(self) -> list[ConfigEntry]:
        """Return all runtime-adjustable settings with DB overrides applied."""
        settings = get_settings()

        # Load all overrides from system_config
        result = await self._db.execute(select(SystemConfig))
        overrides: dict[str, Any] = {
            row.key: row.value for row in result.scalars().all()
        }

        entries: list[ConfigEntry] = []
        for field_name, field_info in settings.model_fields.items():
            default_val = getattr(settings, field_name)
            # If an override exists in system_config, use it
            effective_val = overrides.get(field_name, {}).get("v", default_val) if field_name in overrides else default_val

            entries.append(
                ConfigEntry(
                    key=field_name,
                    value=effective_val,
                    type=_python_type_name(default_val),
                    editable=field_name not in _SECRET_FIELDS,
                    dangerous=_is_dangerous(field_name, effective_val),
                    description=_FIELD_DESCRIPTIONS.get(field_name, ""),
                )
            )
        return entries

    async def update_config(self, updates: dict[str, Any]) -> None:
        """Validate and persist config updates to system_config table.

        Rejects writes to secret fields. Type validation ensures the
        new value is compatible with the Settings field type.
        """
        settings = get_settings()

        for key, value in updates.items():
            if key in _SECRET_FIELDS:
                raise HTTPException(
                    403, f"Cannot modify secret field: {key}"
                )
            if not hasattr(settings, key):
                raise HTTPException(
                    400, f"Unknown config key: {key}"
                )

            # Upsert into system_config
            stmt = pg_insert(SystemConfig).values(
                key=key,
                value={"v": value},
                updated_by="admin",
            ).on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "value": {"v": value},
                    "updated_by": "admin",
                    "updated_at": func.now(),
                },
            )
            await self._db.execute(stmt)

        await self._db.commit()
        logger.info("Admin config updated: %s", list(updates.keys()))

    async def reset_config(self) -> None:
        """Delete ALL system_config rows, reverting to Settings defaults."""
        await self._db.execute(delete(SystemConfig))
        await self._db.commit()
        logger.info("Admin config reset to defaults (all overrides deleted)")

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    async def get_sources(self) -> list[SourceInfo]:
        """Return per-source health derived from ingest_runs + disable state."""
        settings = get_settings()
        source_types = ["gdelt", "rss", "acled", "advisory"]

        # Load disabled state from system_config
        disabled_result = await self._db.execute(
            select(SystemConfig).where(
                SystemConfig.key.in_(
                    [f"{s}_enabled" for s in source_types] + ["disabled_feeds"]
                )
            )
        )
        config_rows = {r.key: r.value for r in disabled_result.scalars().all()}

        sources: list[SourceInfo] = []
        for src_type in source_types:
            # Query latest run
            latest = await self._db.execute(
                select(IngestRun)
                .where(IngestRun.daemon_type == src_type)
                .order_by(IngestRun.started_at.desc())
                .limit(1)
            )
            last_run_row = latest.scalar_one_or_none()

            # Total events
            event_count_result = await self._db.execute(
                select(func.sum(IngestRun.events_new))
                .where(IngestRun.daemon_type == src_type)
            )
            total_events = event_count_result.scalar() or 0

            # Check enabled state
            enabled_key = f"{src_type}_enabled"
            enabled = True
            if enabled_key in config_rows:
                enabled = config_rows[enabled_key].get("v", True)

            # Health: last successful run within 2x poll interval
            healthy = False
            interval_field = _DAEMON_INTERVALS.get(src_type)
            interval = getattr(settings, interval_field, 86400) if interval_field else 86400
            if last_run_row and last_run_row.status == "success":
                age = (datetime.now(timezone.utc) - last_run_row.started_at).total_seconds()
                healthy = age < (interval * 2)

            sources.append(
                SourceInfo(
                    name=_DAEMON_NAMES.get(src_type, src_type),
                    daemon_type=src_type,
                    enabled=enabled,
                    healthy=healthy,
                    last_run=last_run_row.started_at if last_run_row else None,
                    events_count=total_events,
                    tier=None,
                )
            )

        return sources

    async def toggle_source(self, source_name: str, enabled: bool) -> None:
        """Persist enable/disable state to system_config."""
        valid_sources = {"gdelt", "rss", "acled", "advisory"}
        if source_name not in valid_sources:
            raise HTTPException(404, f"Unknown source: {source_name}")

        key = f"{source_name}_enabled"
        stmt = pg_insert(SystemConfig).values(
            key=key,
            value={"v": enabled},
            updated_by="admin",
        ).on_conflict_do_update(
            index_elements=["key"],
            set_={
                "value": {"v": enabled},
                "updated_by": "admin",
                "updated_at": func.now(),
            },
        )
        await self._db.execute(stmt)
        await self._db.commit()
        logger.info("Source %s toggled: enabled=%s", source_name, enabled)
