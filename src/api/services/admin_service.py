"""Admin service layer -- business logic for the admin API endpoints.

Encapsulates all admin operations: process status, job triggering,
pause/resume, runtime config CRUD, source health, and source toggling.
Delegates job control to APScheduler via ``get_scheduler()``, merging
live scheduler state with DB aggregates for the process table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger
from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.admin import (
    AccuracyResponse,
    AccuracySummary,
    AddFeedRequest,
    BacktestResultDTO,
    BacktestRunDTO,
    BacktestRunDetailDTO,
    CheckpointInfo,
    ConfigEntry,
    FeedInfo,
    ProcessInfo,
    ResolvedComparisonDTO,
    SourceInfo,
    StartBacktestRequest,
    UpdateFeedRequest,
)
from src.db.models import BacktestResult, BacktestRun, IngestRun, PolymarketComparison, Prediction, RSSFeed, SystemConfig
from src.scheduler.retry import JobFailureTracker
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

# APScheduler job_id -> daemon_type mapping
_JOB_ID_TO_DAEMON: dict[str, str] = {
    "gdelt_poller": "gdelt",
    "rss_tier1": "rss",
    "rss_tier2": "rss",
    "rss_prune": "rss",
    "acled_poller": "acled",
    "advisory_poller": "advisory",
    "daily_pipeline": "pipeline",
    "polymarket": "polymarket",
    "tkg_retrain": "tkg",
}

# daemon_type -> list of APScheduler job IDs (RSS has 3 sub-jobs)
_DAEMON_TO_JOB_IDS: dict[str, list[str]] = {
    "gdelt": ["gdelt_poller"],
    "rss": ["rss_tier1", "rss_tier2", "rss_prune"],
    "pipeline": ["daily_pipeline"],
    "polymarket": ["polymarket"],
    "acled": ["acled_poller"],
    "advisory": ["advisory_poller"],
    "tkg": ["tkg_retrain"],
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


def _get_scheduler_or_none():
    """Try to get the APScheduler singleton, return None if not initialized."""
    try:
        from src.scheduler.core import get_scheduler
        return get_scheduler()
    except RuntimeError:
        return None


class AdminService:
    """Admin business logic operating on a scoped async session.

    Optionally receives a ``failure_tracker`` for APScheduler job stats.
    When the scheduler is not initialized (e.g. in tests), the service
    falls back to DB-only process information.
    """

    def __init__(
        self,
        db: AsyncSession,
        failure_tracker: JobFailureTracker | None = None,
    ) -> None:
        self._db = db
        self._failure_tracker = failure_tracker

    # ------------------------------------------------------------------
    # Processes
    # ------------------------------------------------------------------

    async def get_processes(self) -> list[ProcessInfo]:
        """Return process status for all daemon types.

        When APScheduler is running, merges live scheduler state (next_run,
        paused status) with DB aggregates (success/fail counts, last_run)
        and failure tracker stats (duration, errors, consecutive failures).

        Falls back to DB-only logic when scheduler is not initialized.
        """
        scheduler = _get_scheduler_or_none()

        # Always query DB aggregates (success/fail counts, last_run)
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

        # Fetch latest status per daemon_type
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

        # Build APScheduler job state map if scheduler is available
        job_state: dict[str, dict] = {}  # job_id -> {next_run, paused}
        if scheduler is not None:
            for job in scheduler.get_jobs():
                job_state[job.id] = {
                    "next_run_time": job.next_run_time,
                    "paused": job.next_run_time is None,
                    "name": job.name,
                }

        processes: list[ProcessInfo] = []
        for dtype, name in _DAEMON_NAMES.items():
            db_row = rows.get(dtype)
            last_run = db_row.last_started if db_row else None
            success_count = db_row.ok if db_row else 0
            fail_count = db_row.fail if db_row else 0

            # Merge APScheduler state for this daemon's jobs
            job_ids = _DAEMON_TO_JOB_IDS.get(dtype, [])
            next_run: datetime | None = None
            is_paused = False
            last_duration: float | None = None
            last_error: str | None = None
            consecutive_failures = 0

            if scheduler is not None and job_ids:
                # Determine next_run as the earliest next_run_time across sub-jobs
                next_runs = []
                all_paused = True
                for jid in job_ids:
                    js = job_state.get(jid)
                    if js is None:
                        continue  # Job not registered (e.g. polymarket disabled)
                    if js["next_run_time"] is not None:
                        next_runs.append(js["next_run_time"])
                        all_paused = False
                    # else: this sub-job is paused

                next_run = min(next_runs) if next_runs else None
                # Daemon is paused only if ALL its sub-jobs are paused
                is_paused = all_paused and any(
                    jid in job_state for jid in job_ids
                )

                # Aggregate failure tracker stats across sub-jobs
                if self._failure_tracker is not None:
                    max_failures = 0
                    for jid in job_ids:
                        stats = self._failure_tracker.get_job_stats(jid)
                        if stats["consecutive_failures"] > max_failures:
                            max_failures = stats["consecutive_failures"]
                            last_error = stats["last_error"]
                        if stats["last_duration"] is not None:
                            if last_duration is None or stats["last_duration"] > last_duration:
                                last_duration = stats["last_duration"]
                    consecutive_failures = max_failures
            else:
                # Fallback: compute next_run from last_run + interval
                interval_field = _DAEMON_INTERVALS.get(dtype)
                interval_seconds: int | None = None
                if interval_field:
                    interval_seconds = getattr(settings, interval_field, None)
                if interval_seconds is None and dtype in ("tkg", "pipeline"):
                    interval_seconds = 86400
                if last_run and interval_seconds:
                    next_run = last_run + timedelta(seconds=interval_seconds)

            # Determine display status
            status = latest_status.get(dtype, "unknown")
            if is_paused:
                status = "paused"
            elif scheduler is not None and next_run is not None:
                status = "scheduled"

            processes.append(
                ProcessInfo(
                    name=name,
                    daemon_type=dtype,
                    status=status,
                    last_run=last_run,
                    next_run=next_run,
                    success_count=success_count,
                    fail_count=fail_count,
                    last_duration=last_duration,
                    last_error=last_error,
                    consecutive_failures=consecutive_failures,
                    paused=is_paused,
                )
            )
        return processes

    # ------------------------------------------------------------------
    # Trigger / Pause / Resume / Reschedule
    # ------------------------------------------------------------------

    async def trigger_job(self, daemon_type: str) -> None:
        """Trigger immediate execution of a daemon's job(s) via APScheduler.

        Sets next_run_time to now, causing APScheduler to fire the job
        on the next scheduler tick (~1 second).

        Raises:
            HTTPException 404: Unknown daemon_type or job not registered.
            HTTPException 503: Scheduler not initialized.
        """
        if daemon_type not in _DAEMON_NAMES:
            raise HTTPException(404, f"Unknown daemon type: {daemon_type}")

        scheduler = _get_scheduler_or_none()
        if scheduler is None:
            raise HTTPException(503, "Scheduler not initialized")

        job_ids = _DAEMON_TO_JOB_IDS.get(daemon_type, [])
        if not job_ids:
            raise HTTPException(404, f"No jobs registered for daemon: {daemon_type}")

        now = datetime.now(timezone.utc)
        triggered = []
        for jid in job_ids:
            job = scheduler.get_job(jid)
            if job is None:
                logger.warning("Job %s not found in scheduler (skipped trigger)", jid)
                continue
            scheduler.modify_job(jid, next_run_time=now)
            triggered.append(jid)

        if not triggered:
            raise HTTPException(404, f"No active jobs found for daemon: {daemon_type}")

        logger.info("Triggered %s job(s): %s", daemon_type, triggered)

    async def pause_job(self, daemon_type: str) -> None:
        """Pause all APScheduler jobs for a daemon type.

        Sets next_run_time to None via scheduler.pause_job(), preventing
        future executions until resumed. Resets failure counters.

        Raises:
            HTTPException 404: Unknown daemon_type.
            HTTPException 503: Scheduler not initialized.
        """
        if daemon_type not in _DAEMON_NAMES:
            raise HTTPException(404, f"Unknown daemon type: {daemon_type}")

        scheduler = _get_scheduler_or_none()
        if scheduler is None:
            raise HTTPException(503, "Scheduler not initialized")

        job_ids = _DAEMON_TO_JOB_IDS.get(daemon_type, [])
        paused = []
        for jid in job_ids:
            job = scheduler.get_job(jid)
            if job is None:
                continue
            scheduler.pause_job(jid)
            if self._failure_tracker is not None:
                self._failure_tracker.reset_failures(jid)
            paused.append(jid)

        logger.info("Paused %s job(s): %s", daemon_type, paused)

    async def resume_job(self, daemon_type: str) -> None:
        """Resume all paused APScheduler jobs for a daemon type.

        Re-enables jobs via scheduler.resume_job() and resets failure
        counters so the auto-pause threshold starts fresh.

        Raises:
            HTTPException 404: Unknown daemon_type.
            HTTPException 503: Scheduler not initialized.
        """
        if daemon_type not in _DAEMON_NAMES:
            raise HTTPException(404, f"Unknown daemon type: {daemon_type}")

        scheduler = _get_scheduler_or_none()
        if scheduler is None:
            raise HTTPException(503, "Scheduler not initialized")

        job_ids = _DAEMON_TO_JOB_IDS.get(daemon_type, [])
        resumed = []
        for jid in job_ids:
            job = scheduler.get_job(jid)
            if job is None:
                continue
            scheduler.resume_job(jid)
            if self._failure_tracker is not None:
                self._failure_tracker.reset_failures(jid)
            resumed.append(jid)

        logger.info("Resumed %s job(s): %s", daemon_type, resumed)

    async def reschedule_job(
        self, daemon_type: str, new_interval_seconds: int,
    ) -> None:
        """Change the interval trigger for a daemon's jobs.

        Persists the new interval to system_config for restart survivability.

        Raises:
            HTTPException 404: Unknown daemon_type.
            HTTPException 503: Scheduler not initialized.
        """
        if daemon_type not in _DAEMON_NAMES:
            raise HTTPException(404, f"Unknown daemon type: {daemon_type}")

        scheduler = _get_scheduler_or_none()
        if scheduler is None:
            raise HTTPException(503, "Scheduler not initialized")

        job_ids = _DAEMON_TO_JOB_IDS.get(daemon_type, [])
        for jid in job_ids:
            job = scheduler.get_job(jid)
            if job is None:
                continue
            scheduler.reschedule_job(
                jid, trigger=IntervalTrigger(seconds=new_interval_seconds),
            )

        # Persist to system_config for restart survivability
        key = f"{daemon_type}_schedule_override"
        stmt = pg_insert(SystemConfig).values(
            key=key,
            value={"v": new_interval_seconds},
            updated_by="admin",
        ).on_conflict_do_update(
            index_elements=["key"],
            set_={
                "value": {"v": new_interval_seconds},
                "updated_by": "admin",
                "updated_at": func.now(),
            },
        )
        await self._db.execute(stmt)
        await self._db.commit()
        logger.info(
            "Rescheduled %s to %ds interval (persisted to system_config)",
            daemon_type, new_interval_seconds,
        )

    async def get_jobs(self) -> list[dict]:
        """Return raw APScheduler job list (complement to get_processes).

        Returns per-job info without daemon-type grouping, useful for
        debugging and the admin /jobs endpoint.
        """
        scheduler = _get_scheduler_or_none()
        if scheduler is None:
            return []

        jobs = []
        for job in scheduler.get_jobs():
            paused = job.next_run_time is None
            trigger_str = str(job.trigger) if job.trigger else "unknown"

            job_info: dict[str, Any] = {
                "job_id": job.id,
                "name": job.name,
                "trigger": trigger_str,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "paused": paused,
                "daemon_type": _JOB_ID_TO_DAEMON.get(job.id, "unknown"),
            }

            # Enrich with failure tracker stats
            if self._failure_tracker is not None:
                stats = self._failure_tracker.get_job_stats(job.id)
                job_info["consecutive_failures"] = stats["consecutive_failures"]
                job_info["last_error"] = stats["last_error"]
                job_info["last_duration"] = stats["last_duration"]

            jobs.append(job_info)

        return jobs

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

    # ------------------------------------------------------------------
    # Accuracy (22-03)
    # ------------------------------------------------------------------

    async def get_accuracy(self) -> AccuracyResponse:
        """Return head-to-head accuracy: summary stats + resolved comparisons.

        Computes summary (wins/losses/draws, cumulative and rolling 30d Brier)
        from live polymarket_comparisons data, plus a list of up to 200
        resolved/voided comparisons with prediction metadata (country, category).
        """
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        # ------ Summary: resolved counts + cumulative Brier ------
        resolved_stmt = select(
            func.count().label("total"),
            func.avg(PolymarketComparison.geopol_brier).label("avg_geopol"),
            func.avg(PolymarketComparison.polymarket_brier).label("avg_pm"),
        ).where(PolymarketComparison.status == "resolved")

        resolved_row = (await self._db.execute(resolved_stmt)).one()
        total_resolved: int = resolved_row.total or 0
        geopol_cumulative = float(resolved_row.avg_geopol) if resolved_row.avg_geopol is not None else None
        pm_cumulative = float(resolved_row.avg_pm) if resolved_row.avg_pm is not None else None

        # Voided count
        voided_count_result = await self._db.execute(
            select(func.count()).where(PolymarketComparison.status == "voided")
        )
        total_voided: int = voided_count_result.scalar() or 0

        # Win/loss/draw counts (only resolved with both Brier scores)
        wins_stmt = select(
            func.count().filter(
                PolymarketComparison.geopol_brier < PolymarketComparison.polymarket_brier
            ).label("geopol_wins"),
            func.count().filter(
                PolymarketComparison.geopol_brier > PolymarketComparison.polymarket_brier
            ).label("pm_wins"),
        ).where(
            PolymarketComparison.status == "resolved",
            PolymarketComparison.geopol_brier.is_not(None),
            PolymarketComparison.polymarket_brier.is_not(None),
        )
        wins_row = (await self._db.execute(wins_stmt)).one()
        geopol_wins: int = wins_row.geopol_wins or 0
        pm_wins: int = wins_row.pm_wins or 0
        draws = total_resolved - geopol_wins - pm_wins

        # ------ Rolling 30d Brier ------
        rolling_stmt = select(
            func.count().label("cnt"),
            func.avg(PolymarketComparison.geopol_brier).label("avg_geopol"),
            func.avg(PolymarketComparison.polymarket_brier).label("avg_pm"),
        ).where(
            PolymarketComparison.status == "resolved",
            PolymarketComparison.resolved_at >= thirty_days_ago,
        )
        rolling_row = (await self._db.execute(rolling_stmt)).one()
        rolling_30d_count: int = rolling_row.cnt or 0
        rolling_30d_geopol = float(rolling_row.avg_geopol) if rolling_row.avg_geopol is not None else None
        rolling_30d_pm = float(rolling_row.avg_pm) if rolling_row.avg_pm is not None else None

        summary = AccuracySummary(
            total_resolved=total_resolved,
            total_voided=total_voided,
            geopol_wins=geopol_wins,
            polymarket_wins=pm_wins,
            draws=draws,
            geopol_cumulative_brier=geopol_cumulative,
            polymarket_cumulative_brier=pm_cumulative,
            rolling_30d_geopol_brier=rolling_30d_geopol,
            rolling_30d_polymarket_brier=rolling_30d_pm,
            rolling_30d_count=rolling_30d_count,
        )

        # ------ Comparisons list ------
        comp_stmt = (
            select(PolymarketComparison, Prediction.country_iso, Prediction.category)
            .outerjoin(Prediction, Prediction.id == PolymarketComparison.geopol_prediction_id)
            .where(PolymarketComparison.status.in_(["resolved", "voided"]))
            .order_by(PolymarketComparison.resolved_at.desc())
            .limit(200)
        )
        rows = (await self._db.execute(comp_stmt)).all()

        comparisons: list[ResolvedComparisonDTO] = []
        for row in rows:
            comp: PolymarketComparison = row[0]
            country_iso: str | None = row[1]
            category: str | None = row[2]

            # Determine winner
            winner: str | None = None
            if comp.status == "voided":
                winner = None
            elif comp.geopol_brier is not None and comp.polymarket_brier is not None:
                if comp.geopol_brier < comp.polymarket_brier:
                    winner = "geopol"
                elif comp.geopol_brier > comp.polymarket_brier:
                    winner = "polymarket"
                else:
                    winner = "draw"

            comparisons.append(
                ResolvedComparisonDTO(
                    id=comp.id,
                    polymarket_title=comp.polymarket_title,
                    polymarket_event_id=comp.polymarket_event_id,
                    geopol_probability=comp.geopol_probability,
                    polymarket_price=comp.polymarket_price,
                    polymarket_outcome=comp.polymarket_outcome,
                    geopol_brier=comp.geopol_brier,
                    polymarket_brier=comp.polymarket_brier,
                    winner=winner,
                    status=comp.status,
                    resolved_at=comp.resolved_at.isoformat() if comp.resolved_at else None,
                    created_at=comp.created_at.isoformat(),
                    country_iso=country_iso,
                    category=category,
                )
            )

        return AccuracyResponse(summary=summary, comparisons=comparisons)

    # ------------------------------------------------------------------
    # Feed CRUD (21-01)
    # ------------------------------------------------------------------

    @staticmethod
    def _feed_to_dto(feed: RSSFeed) -> FeedInfo:
        """Convert an RSSFeed ORM instance to a FeedInfo DTO."""
        return FeedInfo(
            id=feed.id,
            name=feed.name,
            url=feed.url,
            tier=feed.tier,
            category=feed.category,
            lang=feed.lang,
            enabled=feed.enabled,
            last_poll_at=feed.last_poll_at.isoformat() if feed.last_poll_at else None,
            last_error=feed.last_error,
            error_count=feed.error_count,
            articles_24h=feed.articles_24h,
            articles_total=feed.articles_total,
            avg_articles_per_poll=feed.avg_articles_per_poll,
            created_at=feed.created_at.isoformat(),
        )

    async def get_feeds(self) -> list[FeedInfo]:
        """Return all non-deleted feeds ordered by tier then name."""
        result = await self._db.execute(
            select(RSSFeed)
            .where(RSSFeed.deleted_at.is_(None))
            .order_by(RSSFeed.tier, RSSFeed.name)
        )
        feeds = result.scalars().all()
        return [self._feed_to_dto(f) for f in feeds]

    async def add_feed(self, data: AddFeedRequest) -> FeedInfo:
        """Insert a new feed. Raises 409 on duplicate name."""
        # Check for name collision (including soft-deleted)
        existing = await self._db.execute(
            select(RSSFeed).where(RSSFeed.name == data.name)
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(409, f"Feed with name {data.name!r} already exists")

        feed = RSSFeed(
            name=data.name,
            url=data.url,
            tier=data.tier,
            category=data.category,
            lang=data.lang,
        )
        self._db.add(feed)
        await self._db.commit()
        await self._db.refresh(feed)
        logger.info("Feed added: %s (tier %d)", feed.name, feed.tier)
        return self._feed_to_dto(feed)

    async def update_feed(self, feed_id: int, data: UpdateFeedRequest) -> FeedInfo:
        """Update non-None fields on a feed. Raises 404 if not found or deleted."""
        result = await self._db.execute(
            select(RSSFeed).where(
                RSSFeed.id == feed_id,
                RSSFeed.deleted_at.is_(None),
            )
        )
        feed = result.scalar_one_or_none()
        if feed is None:
            raise HTTPException(404, f"Feed {feed_id} not found")

        update_fields = data.model_dump(exclude_none=True)
        if not update_fields:
            raise HTTPException(400, "No fields to update")

        for field_name, value in update_fields.items():
            setattr(feed, field_name, value)

        await self._db.commit()
        await self._db.refresh(feed)
        logger.info("Feed updated: id=%d, fields=%s", feed_id, list(update_fields.keys()))
        return self._feed_to_dto(feed)

    async def delete_feed(self, feed_id: int, purge: bool = False) -> None:
        """Soft-delete (default) or hard-delete a feed. Raises 404 if not found."""
        result = await self._db.execute(
            select(RSSFeed).where(RSSFeed.id == feed_id)
        )
        feed = result.scalar_one_or_none()
        if feed is None:
            raise HTTPException(404, f"Feed {feed_id} not found")

        if purge:
            await self._db.delete(feed)
            logger.info("Feed hard-deleted: id=%d, name=%s", feed_id, feed.name)
        else:
            if feed.deleted_at is not None:
                raise HTTPException(404, f"Feed {feed_id} already deleted")
            feed.deleted_at = datetime.now(timezone.utc)
            logger.info("Feed soft-deleted: id=%d, name=%s", feed_id, feed.name)

        await self._db.commit()

    # ------------------------------------------------------------------
    # Backtesting (23-02)
    # ------------------------------------------------------------------

    @staticmethod
    def _backtest_run_to_dto(run: BacktestRun) -> BacktestRunDTO:
        """Convert a BacktestRun ORM instance to a BacktestRunDTO."""
        return BacktestRunDTO(
            id=run.id,
            label=run.label,
            description=run.description,
            window_size_days=run.window_size_days,
            slide_step_days=run.slide_step_days,
            min_predictions_per_window=run.min_predictions_per_window,
            checkpoints_json=run.checkpoints_json,
            status=run.status,
            started_at=run.started_at.isoformat() if run.started_at else None,
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            total_windows=run.total_windows,
            completed_windows=run.completed_windows,
            total_predictions=run.total_predictions,
            aggregate_brier=run.aggregate_brier,
            aggregate_mrr=run.aggregate_mrr,
            vs_polymarket_record_json=run.vs_polymarket_record_json,
            error_message=run.error_message,
            created_at=run.created_at.isoformat() if run.created_at else None,
        )

    @staticmethod
    def _backtest_result_to_dto(r: BacktestResult) -> BacktestResultDTO:
        """Convert a BacktestResult ORM instance to a BacktestResultDTO."""
        return BacktestResultDTO(
            id=r.id,
            run_id=r.run_id,
            window_start=r.window_start.isoformat() if r.window_start else None,
            window_end=r.window_end.isoformat() if r.window_end else None,
            prediction_start=r.prediction_start.isoformat() if r.prediction_start else None,
            prediction_end=r.prediction_end.isoformat() if r.prediction_end else None,
            checkpoint_name=r.checkpoint_name,
            num_predictions=r.num_predictions,
            brier_score=r.brier_score,
            mrr=r.mrr,
            hits_at_1=r.hits_at_1,
            hits_at_10=r.hits_at_10,
            calibration_bins_json=r.calibration_bins_json,
            prediction_details_json=r.prediction_details_json,
            polymarket_brier=r.polymarket_brier,
            geopol_vs_pm_wins=r.geopol_vs_pm_wins,
            pm_vs_geopol_wins=r.pm_vs_geopol_wins,
            weight_snapshot_json=r.weight_snapshot_json,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )

    async def get_backtest_runs(self) -> list[BacktestRunDTO]:
        """Return all backtest runs ordered by created_at desc."""
        result = await self._db.execute(
            select(BacktestRun).order_by(BacktestRun.created_at.desc())
        )
        runs = result.scalars().all()
        return [self._backtest_run_to_dto(r) for r in runs]

    async def start_backtest_run(
        self, request: StartBacktestRequest,
    ) -> BacktestRunDTO:
        """Create a new BacktestRun and dispatch the heavy backtest job.

        The endpoint returns immediately with a 'pending' run DTO.
        Execution proceeds asynchronously via asyncio.create_task ->
        heavy_backtest -> ProcessPoolExecutor -> BacktestRunner.

        Raises:
            HTTPException 400: Invalid checkpoint configuration.
        """
        import asyncio as _asyncio

        from src.backtesting.schemas import BacktestRunConfig

        # Create DB row first with status='pending'.
        run = BacktestRun(
            label=request.label,
            description=request.description,
            window_size_days=request.window_size_days,
            slide_step_days=request.slide_step_days,
            min_predictions_per_window=request.min_predictions_per_window,
            checkpoints_json=request.checkpoints,
            status="pending",
        )
        self._db.add(run)
        await self._db.commit()
        await self._db.refresh(run)

        # Build BacktestRunConfig with the assigned run_id.
        config = BacktestRunConfig(
            label=request.label,
            checkpoints=request.checkpoints,
            window_size_days=request.window_size_days,
            slide_step_days=request.slide_step_days,
            min_predictions_per_window=request.min_predictions_per_window,
            description=request.description,
            run_id=run.id,
        )
        config_json = config.to_json()

        # Fire-and-forget dispatch to ProcessPoolExecutor.
        from src.scheduler.job_wrappers import heavy_backtest

        _asyncio.create_task(heavy_backtest(config_json))
        logger.info("Backtest run %s dispatched (label=%s)", run.id, run.label)

        return self._backtest_run_to_dto(run)

    async def get_backtest_run_detail(
        self, run_id: str,
    ) -> BacktestRunDetailDTO:
        """Fetch a backtest run with all window-level results.

        Raises:
            HTTPException 404: Run not found.
        """
        run = await self._db.get(BacktestRun, run_id)
        if run is None:
            raise HTTPException(404, f"Backtest run {run_id} not found")

        result = await self._db.execute(
            select(BacktestResult)
            .where(BacktestResult.run_id == run_id)
            .order_by(
                BacktestResult.window_start.asc(),
                BacktestResult.checkpoint_name.asc(),
            )
        )
        results = result.scalars().all()

        return BacktestRunDetailDTO(
            run=self._backtest_run_to_dto(run),
            results=[self._backtest_result_to_dto(r) for r in results],
        )

    async def cancel_backtest_run(self, run_id: str) -> BacktestRunDTO:
        """Set a backtest run's status to 'cancelling'.

        The runner polls status between windows and transitions to 'cancelled'
        upon detecting the 'cancelling' signal.

        Raises:
            HTTPException 404: Run not found.
            HTTPException 409: Run is not in a cancellable state.
        """
        run = await self._db.get(BacktestRun, run_id)
        if run is None:
            raise HTTPException(404, f"Backtest run {run_id} not found")

        if run.status not in ("running", "pending"):
            raise HTTPException(
                409,
                f"Cannot cancel run in status '{run.status}' "
                f"(must be 'running' or 'pending')",
            )

        run.status = "cancelling"
        await self._db.commit()
        await self._db.refresh(run)
        logger.info("Backtest run %s set to cancelling", run_id)
        return self._backtest_run_to_dto(run)

    async def export_backtest_run(
        self, run_id: str, fmt: str,
    ) -> str | dict:
        """Export backtest results as CSV string or JSON dict.

        Raises:
            HTTPException 404: Run not found.
            HTTPException 409: Run not in an exportable state.
            HTTPException 400: Invalid format.
        """
        from src.backtesting.export import export_run_csv, export_run_json

        run = await self._db.get(BacktestRun, run_id)
        if run is None:
            raise HTTPException(404, f"Backtest run {run_id} not found")

        exportable = {"completed", "cancelled"}
        if run.status not in exportable:
            raise HTTPException(
                409,
                f"Cannot export run in status '{run.status}' "
                f"(must be 'completed' or 'cancelled')",
            )

        if fmt == "csv":
            return await export_run_csv(self._db, run_id)
        elif fmt == "json":
            return await export_run_json(self._db, run_id)
        else:
            raise HTTPException(400, f"Invalid format: {fmt} (must be 'csv' or 'json')")

    async def get_checkpoints(self) -> list[CheckpointInfo]:
        """Scan models/tkg/ for available TiRGN and RE-GCN checkpoints.

        Reads JSON metadata files to extract model type, training metrics,
        and creation timestamp. Malformed or unreadable files are skipped.

        Returns:
            Sorted list: by model_type then name.
        """
        import json
        from pathlib import Path

        model_dir = Path("models/tkg")
        if not model_dir.is_dir():
            logger.warning("Checkpoint directory %s not found", model_dir)
            return []

        checkpoints: list[CheckpointInfo] = []

        for json_path in sorted(model_dir.glob("*.json")):
            name = json_path.stem  # e.g. "tirgn_best"

            # Skip non-checkpoint metadata (e.g. last_trained.json)
            is_tirgn = name.startswith("tirgn_")
            is_regcn = name.startswith("regcn_jraph_")
            if not is_tirgn and not is_regcn:
                continue

            # Verify corresponding weight file exists
            if is_tirgn:
                weight_path = json_path.with_suffix(".npz")
            else:
                weight_path = json_path.with_suffix(".npz")
            if not weight_path.exists():
                logger.warning(
                    "Checkpoint %s has metadata but no weight file, skipping",
                    name,
                )
                continue

            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to read checkpoint metadata %s: %s", json_path, exc,
                )
                continue

            model_type = meta.get("model_type", "regcn" if is_regcn else "tirgn")
            metrics_raw = meta.get("metrics")
            metrics: dict[str, float] | None = None
            if isinstance(metrics_raw, dict):
                metrics = {
                    k: float(v)
                    for k, v in metrics_raw.items()
                    if isinstance(v, (int, float))
                }

            # Use file modification time as creation timestamp
            try:
                mtime = json_path.stat().st_mtime
                created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                created_at = None

            checkpoints.append(
                CheckpointInfo(
                    name=name,
                    model_type=model_type,
                    path=str(weight_path),
                    metrics=metrics,
                    created_at=created_at,
                )
            )

        # Sort by model_type then name for stable ordering
        checkpoints.sort(key=lambda c: (c.model_type, c.name))
        return checkpoints
