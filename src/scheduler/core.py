"""
AsyncIOScheduler factory with dual executors.

Creates a singleton scheduler with:
  - default: AsyncIOExecutor for lightweight async job wrappers
  - processpool: ProcessPoolExecutor(max_workers=1) for heavy jobs

Coalesce + max_instances=1 prevent overlapping runs. Misfire grace of
15 minutes accommodates transient delays.
"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def create_scheduler() -> AsyncIOScheduler:
    """Create and return the singleton AsyncIOScheduler.

    Configures dual executors:
      - ``default``: AsyncIOExecutor for light async wrappers
      - ``processpool``: ProcessPoolExecutor(max_workers=1) for heavy jobs

    Job defaults: coalesce=True, max_instances=1, misfire_grace_time=900s.

    Raises:
        RuntimeError: If scheduler was already created.
    """
    global _scheduler  # noqa: PLW0603

    if _scheduler is not None:
        raise RuntimeError("Scheduler already created -- call get_scheduler() instead")

    executors = {
        "default": AsyncIOExecutor(),
        "processpool": ProcessPoolExecutor(max_workers=1),
    }

    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 900,
    }

    _scheduler = AsyncIOScheduler(
        executors=executors,
        job_defaults=job_defaults,
    )

    logger.info("Scheduler created (executors: default=AsyncIO, processpool=Process(1))")
    return _scheduler


def get_scheduler() -> AsyncIOScheduler:
    """Return the singleton scheduler.

    Raises:
        RuntimeError: If scheduler has not been created yet.
    """
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized -- call create_scheduler() first")
    return _scheduler


async def shutdown_scheduler(
    scheduler: AsyncIOScheduler,
    timeout: float = 30.0,
) -> None:
    """Gracefully shut down the scheduler.

    1. Pauses the scheduler to prevent new job starts.
    2. Attempts graceful shutdown (wait=True) within timeout.
    3. On timeout, forces immediate shutdown (wait=False).

    Args:
        scheduler: The running scheduler instance.
        timeout: Maximum seconds to wait for running jobs to finish.
    """
    global _scheduler  # noqa: PLW0603

    logger.info("Shutting down scheduler (timeout=%.0fs)...", timeout)

    # Pause to prevent new job starts while existing ones finish
    scheduler.pause()

    try:
        await asyncio.wait_for(
            asyncio.to_thread(scheduler.shutdown, wait=True),
            timeout=timeout,
        )
        logger.info("Scheduler shut down gracefully")
    except asyncio.TimeoutError:
        logger.warning(
            "Scheduler shutdown timed out after %.0fs, forcing immediate shutdown",
            timeout,
        )
        scheduler.shutdown(wait=False)

    _scheduler = None
