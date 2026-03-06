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
    AddFeedRequest,
    ConfigEntry,
    FeedInfo,
    ProcessInfo,
    SourceInfo,
    UpdateFeedRequest,
)
from src.db.models import IngestRun, RSSFeed, SystemConfig
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
