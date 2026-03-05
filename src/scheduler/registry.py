"""
Job registration for all 9 background tasks.

Registers each job with the correct trigger type, interval, executor,
misfire grace time, and human-readable name. All jobs inherit scheduler
defaults (coalesce=True, max_instances=1) but set them explicitly for
clarity and auditability.

Schedule overrides from the system_config DB table are applied if available.
"""

from __future__ import annotations

import logging

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.scheduler.job_wrappers import (
    acled_poll_cycle,
    advisory_poll_cycle,
    gdelt_poll_cycle,
    heavy_daily_pipeline,
    heavy_polymarket_cycle,
    heavy_tkg_retrain,
    rss_poll_tier1,
    rss_poll_tier2,
    rss_prune,
)
from src.scheduler.retry import JobFailureTracker
from src.settings import get_settings

logger = logging.getLogger(__name__)


def register_all_jobs(
    scheduler: AsyncIOScheduler,
    failure_tracker: JobFailureTracker,
) -> None:
    """Register all 9 background jobs and the failure tracker listener.

    Light jobs (6) use the default AsyncIOExecutor.
    Heavy jobs (3) use the default executor too -- the async wrappers
    internally acquire the heavy_job_lock and dispatch to ProcessPoolExecutor.

    Args:
        scheduler: The initialized AsyncIOScheduler.
        failure_tracker: The JobFailureTracker for auto-pause.
    """
    settings = get_settings()

    # -------------------------------------------------------------------
    # Light jobs (default executor = AsyncIOExecutor)
    # -------------------------------------------------------------------

    scheduler.add_job(
        gdelt_poll_cycle,
        trigger=IntervalTrigger(seconds=settings.gdelt_poll_interval),
        id="gdelt_poller",
        name="GDELT Poller",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.gdelt_poll_interval,
    )
    logger.info(
        "Registered gdelt_poller (interval=%ds)", settings.gdelt_poll_interval,
    )

    scheduler.add_job(
        rss_poll_tier1,
        trigger=IntervalTrigger(seconds=settings.rss_poll_interval_tier1),
        id="rss_tier1",
        name="RSS Tier-1 Poller",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.rss_poll_interval_tier1,
    )
    logger.info(
        "Registered rss_tier1 (interval=%ds)", settings.rss_poll_interval_tier1,
    )

    scheduler.add_job(
        rss_poll_tier2,
        trigger=IntervalTrigger(seconds=settings.rss_poll_interval_tier2),
        id="rss_tier2",
        name="RSS Tier-2 Poller",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.rss_poll_interval_tier2,
    )
    logger.info(
        "Registered rss_tier2 (interval=%ds)", settings.rss_poll_interval_tier2,
    )

    scheduler.add_job(
        rss_prune,
        trigger=IntervalTrigger(seconds=86400),
        id="rss_prune",
        name="RSS Article Pruner",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=7200,
    )
    logger.info("Registered rss_prune (interval=86400s)")

    scheduler.add_job(
        acled_poll_cycle,
        trigger=IntervalTrigger(seconds=settings.acled_poll_interval),
        id="acled_poller",
        name="ACLED Poller",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    logger.info(
        "Registered acled_poller (interval=%ds)", settings.acled_poll_interval,
    )

    scheduler.add_job(
        advisory_poll_cycle,
        trigger=IntervalTrigger(seconds=settings.advisory_poll_interval),
        id="advisory_poller",
        name="Advisory Poller",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    logger.info(
        "Registered advisory_poller (interval=%ds)",
        settings.advisory_poll_interval,
    )

    # -------------------------------------------------------------------
    # Heavy jobs (async wrappers handle ProcessPoolExecutor internally)
    # -------------------------------------------------------------------

    scheduler.add_job(
        heavy_daily_pipeline,
        trigger=CronTrigger(hour=6, minute=0),
        id="daily_pipeline",
        name="Daily Forecast Pipeline",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    logger.info("Registered daily_pipeline (cron: 06:00 daily)")

    if settings.polymarket_enabled:
        scheduler.add_job(
            heavy_polymarket_cycle,
            trigger=IntervalTrigger(seconds=settings.polymarket_poll_interval),
            id="polymarket",
            name="Polymarket Forecaster",
            coalesce=True,
            max_instances=1,
            misfire_grace_time=1800,
        )
        logger.info(
            "Registered polymarket (interval=%ds)",
            settings.polymarket_poll_interval,
        )
    else:
        logger.info("Polymarket job skipped (polymarket_enabled=False)")

    scheduler.add_job(
        heavy_tkg_retrain,
        trigger=CronTrigger(day_of_week="sun", hour=2),
        id="tkg_retrain",
        name="TKG Weekly Retrain",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=7200,
    )
    logger.info("Registered tkg_retrain (cron: Sun 02:00)")

    # -------------------------------------------------------------------
    # Register failure tracker as event listener
    # -------------------------------------------------------------------

    scheduler.add_listener(
        failure_tracker.on_job_event,
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
    )
    logger.info(
        "Failure tracker registered (auto-pause at %d consecutive failures)",
        failure_tracker._max_failures,
    )

    job_count = len(scheduler.get_jobs())
    logger.info("All jobs registered: %d total", job_count)
