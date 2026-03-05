"""
Job failure tracking with auto-pause and dependency cascade.

JobFailureTracker listens to APScheduler events and:
  - Resets consecutive failure count on successful execution
  - Increments failures on error, auto-pauses job at threshold
  - Exposes per-job stats for the admin API dashboard
  - Tracks last duration and last_run_time for monitoring

JOB_DEPENDENCIES defines a simple upstream graph. check_upstream_health()
prevents downstream jobs from running when their upstreams have failures.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    JobEvent,
    JobExecutionEvent,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


class JobFailureTracker:
    """Track per-job consecutive failures and auto-pause at threshold.

    Registers as an APScheduler event listener. On consecutive failures
    reaching ``max_consecutive_failures``, the job is paused via
    ``scheduler.pause_job()``. The admin API can resume it and reset
    the failure counter.

    Args:
        scheduler: The APScheduler instance (for pause_job calls).
        max_consecutive_failures: Threshold before auto-pause (default 5).
    """

    def __init__(
        self,
        scheduler: AsyncIOScheduler,
        max_consecutive_failures: int = 5,
    ) -> None:
        self._scheduler = scheduler
        self._max_failures = max_consecutive_failures

        # Per-job tracking state
        self._consecutive_failures: dict[str, int] = {}
        self._last_error: dict[str, str] = {}
        self._last_duration: dict[str, float] = {}
        self._last_run_time: dict[str, datetime] = {}

    def on_job_event(self, event: JobEvent) -> None:
        """APScheduler event listener callback.

        Handles EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, and EVENT_JOB_MISSED.
        Must be registered via ``scheduler.add_listener()``.
        """
        job_id = event.job_id

        if event.code == EVENT_JOB_EXECUTED:
            self._consecutive_failures[job_id] = 0
            self._last_run_time[job_id] = datetime.now(timezone.utc)

            # Extract duration from scheduled_run_time if available
            if isinstance(event, JobExecutionEvent) and event.scheduled_run_time:
                duration = (
                    datetime.now(timezone.utc) - event.scheduled_run_time.replace(tzinfo=timezone.utc)
                ).total_seconds()
                # Clamp to non-negative (clock skew edge case)
                self._last_duration[job_id] = max(0.0, duration)

            logger.debug("Job %s executed successfully", job_id)

        elif event.code == EVENT_JOB_ERROR:
            count = self._consecutive_failures.get(job_id, 0) + 1
            self._consecutive_failures[job_id] = count
            self._last_run_time[job_id] = datetime.now(timezone.utc)

            # Store error message from the exception
            error_msg = "unknown error"
            if isinstance(event, JobExecutionEvent) and event.exception:
                error_msg = str(event.exception)
            self._last_error[job_id] = error_msg

            logger.error(
                "Job %s failed (consecutive=%d/%d): %s",
                job_id, count, self._max_failures, error_msg,
            )

            if count >= self._max_failures:
                try:
                    self._scheduler.pause_job(job_id)
                    logger.warning(
                        "Job %s AUTO-PAUSED after %d consecutive failures",
                        job_id, count,
                    )
                except Exception:
                    # Job may have been removed between event and pause
                    logger.warning(
                        "Failed to pause job %s (may have been removed)",
                        job_id,
                    )

        elif event.code == EVENT_JOB_MISSED:
            logger.warning("Job %s missed execution", job_id)

    def get_job_stats(self, job_id: str) -> dict:
        """Return stats for a single job.

        Returns:
            Dict with keys: consecutive_failures, last_error,
            last_duration, last_run_time.
        """
        return {
            "consecutive_failures": self._consecutive_failures.get(job_id, 0),
            "last_error": self._last_error.get(job_id),
            "last_duration": self._last_duration.get(job_id),
            "last_run_time": self._last_run_time.get(job_id),
        }

    def get_all_stats(self) -> dict[str, dict]:
        """Return stats for all tracked jobs.

        Returns:
            Dict mapping job_id -> stats dict.
        """
        all_job_ids = set()
        all_job_ids.update(self._consecutive_failures.keys())
        all_job_ids.update(self._last_error.keys())
        all_job_ids.update(self._last_duration.keys())
        all_job_ids.update(self._last_run_time.keys())

        return {job_id: self.get_job_stats(job_id) for job_id in sorted(all_job_ids)}

    def reset_failures(self, job_id: str) -> None:
        """Manually reset the failure counter for a job.

        Called by the admin API when a paused job is resumed.
        """
        self._consecutive_failures[job_id] = 0
        self._last_error.pop(job_id, None)
        logger.info("Failure counter reset for job %s", job_id)


# ---------------------------------------------------------------------------
# Job dependency graph
# ---------------------------------------------------------------------------

# Upstream dependencies: job_id -> list of upstream job_ids that must be healthy.
# daily_pipeline needs fresh GDELT + RSS data before running forecasts.
JOB_DEPENDENCIES: dict[str, list[str]] = {
    "daily_pipeline": ["gdelt_poller", "rss_tier1"],
}


def check_upstream_health(job_id: str, tracker: JobFailureTracker) -> bool:
    """Check if all upstream jobs for a given job are healthy (0 failures).

    Args:
        job_id: The job to check upstream health for.
        tracker: The JobFailureTracker instance.

    Returns:
        True if all upstreams are healthy or if no upstreams are defined.
        False if any upstream has consecutive failures > 0.
    """
    upstreams = JOB_DEPENDENCIES.get(job_id, [])
    if not upstreams:
        return True

    for upstream_id in upstreams:
        failures = tracker._consecutive_failures.get(upstream_id, 0)
        if failures > 0:
            logger.warning(
                "Job %s skipped: upstream %s has %d consecutive failures",
                job_id, upstream_id, failures,
            )
            return False

    return True
