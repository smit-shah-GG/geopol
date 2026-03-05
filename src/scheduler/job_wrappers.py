"""
Async wrappers for all 9 background jobs.

Light jobs (6): run directly in the AsyncIOExecutor event loop.
Heavy jobs (3): acquire a mutual exclusion lock, then dispatch to a
ProcessPoolExecutor(max_workers=1) via loop.run_in_executor().

The heavy job lock prevents concurrent heavy workloads (daily pipeline,
polymarket cycle, TKG retrain) from overwhelming the single-worker
process pool. Jobs queue in FIFO order on the asyncio.Lock.

Every wrapper catches exceptions, logs them with full traceback, then
re-raises so the APScheduler listener (JobFailureTracker) can track
consecutive failures and auto-pause if needed.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

from src.scheduler.dependencies import get_shared_deps

logger = logging.getLogger(__name__)

# Mutual exclusion for heavy jobs -- prevents concurrent subprocess/process work
_heavy_job_lock = asyncio.Lock()

# Lazy singleton process pool for heavy job dispatch
_process_executor: ProcessPoolExecutor | None = None


def _get_process_executor() -> ProcessPoolExecutor:
    """Return the lazy singleton ProcessPoolExecutor(max_workers=1)."""
    global _process_executor  # noqa: PLW0603
    if _process_executor is None:
        _process_executor = ProcessPoolExecutor(max_workers=1)
        logger.info("ProcessPoolExecutor created (max_workers=1)")
    return _process_executor


# ---------------------------------------------------------------------------
# Light job wrappers (AsyncIOExecutor -- default)
# ---------------------------------------------------------------------------


async def gdelt_poll_cycle() -> None:
    """Execute a single GDELT poll cycle using the singleton poller.

    The poller's ``_last_url`` state persists across cycles (singleton in
    SharedDeps), enabling the URL-dedup fast path that skips re-downloading
    unchanged exports.
    """
    try:
        deps = get_shared_deps()
        await deps.gdelt_poller._poll_once()
    except Exception:
        logger.exception("gdelt_poll_cycle failed")
        raise


async def rss_poll_tier1() -> None:
    """Poll tier-1 RSS feeds (high-priority news sources)."""
    try:
        from src.ingest.feed_config import TIER_1_FEEDS
        from src.ingest.rss_daemon import DaemonConfig, RSSDaemon

        deps = get_shared_deps()
        config = DaemonConfig(
            tier1_interval=deps.settings.rss_poll_interval_tier1,
            tier2_interval=deps.settings.rss_poll_interval_tier2,
            retention_days=deps.settings.rss_article_retention_days,
        )
        daemon = RSSDaemon(config=config)
        try:
            metrics = await daemon.poll_feeds(TIER_1_FEEDS)
            logger.info(
                "rss_tier1: feeds=%d, new=%d, dup=%d, chunks=%d, %.1fs",
                metrics.feeds_polled,
                metrics.articles_new,
                metrics.articles_duplicate,
                metrics.chunks_indexed,
                metrics.duration_seconds,
            )
            await daemon._record_ingest_run(metrics, "tier-1")
        finally:
            # Clean up the aiohttp session created by RSSDaemon
            if daemon._session and not daemon._session.closed:
                await daemon._session.close()
    except Exception:
        logger.exception("rss_poll_tier1 failed")
        raise


async def rss_poll_tier2() -> None:
    """Poll tier-2 RSS feeds (lower-priority / regional sources)."""
    try:
        from src.ingest.feed_config import TIER_2_FEEDS
        from src.ingest.rss_daemon import DaemonConfig, RSSDaemon

        deps = get_shared_deps()
        config = DaemonConfig(
            tier1_interval=deps.settings.rss_poll_interval_tier1,
            tier2_interval=deps.settings.rss_poll_interval_tier2,
            retention_days=deps.settings.rss_article_retention_days,
        )
        daemon = RSSDaemon(config=config)
        try:
            metrics = await daemon.poll_feeds(TIER_2_FEEDS)
            logger.info(
                "rss_tier2: feeds=%d, new=%d, dup=%d, chunks=%d, %.1fs",
                metrics.feeds_polled,
                metrics.articles_new,
                metrics.articles_duplicate,
                metrics.chunks_indexed,
                metrics.duration_seconds,
            )
            await daemon._record_ingest_run(metrics, "tier-2")
        finally:
            if daemon._session and not daemon._session.closed:
                await daemon._session.close()
    except Exception:
        logger.exception("rss_poll_tier2 failed")
        raise


async def rss_prune() -> None:
    """Prune old articles from ChromaDB beyond retention window."""
    try:
        from src.ingest.rss_daemon import DaemonConfig, RSSDaemon

        deps = get_shared_deps()
        config = DaemonConfig(
            retention_days=deps.settings.rss_article_retention_days,
        )
        daemon = RSSDaemon(config=config)
        try:
            await daemon._maybe_prune()
        finally:
            if daemon._session and not daemon._session.closed:
                await daemon._session.close()
    except Exception:
        logger.exception("rss_prune failed")
        raise


async def acled_poll_cycle() -> None:
    """Execute a single ACLED conflict event poll cycle."""
    try:
        from src.ingest.acled_poller import ACLEDPoller

        deps = get_shared_deps()
        poller = ACLEDPoller(event_storage=deps.event_storage)
        await poller._poll_once()
    except Exception:
        logger.exception("acled_poll_cycle failed")
        raise


async def advisory_poll_cycle() -> None:
    """Execute a single government travel advisory poll cycle."""
    try:
        from src.ingest.advisory_poller import AdvisoryPoller

        poller = AdvisoryPoller()
        await poller._poll_once()
    except Exception:
        logger.exception("advisory_poll_cycle failed")
        raise


# ---------------------------------------------------------------------------
# Heavy job wrappers (acquire lock, dispatch to ProcessPoolExecutor)
# ---------------------------------------------------------------------------


async def heavy_daily_pipeline() -> None:
    """Run the daily forecast pipeline in a subprocess.

    Acquires the heavy job lock to prevent concurrent heavy workloads,
    then dispatches ``run_daily_pipeline`` to a ProcessPoolExecutor.
    """
    try:
        from src.scheduler.heavy_runner import run_daily_pipeline

        async with _heavy_job_lock:
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(
                _get_process_executor(), run_daily_pipeline
            )
            if returncode != 0:
                raise RuntimeError(
                    f"daily_pipeline subprocess exited with code {returncode}"
                )
    except Exception:
        logger.exception("heavy_daily_pipeline failed")
        raise


async def heavy_polymarket_cycle() -> None:
    """Run the Polymarket matching + auto-forecast cycle in a subprocess.

    Acquires the heavy job lock, then dispatches ``run_polymarket_cycle``
    to a ProcessPoolExecutor. The polymarket cycle creates its own
    asyncio event loop via asyncio.run() inside the subprocess worker.
    """
    try:
        from src.scheduler.heavy_runner import run_polymarket_cycle

        async with _heavy_job_lock:
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(
                _get_process_executor(), run_polymarket_cycle
            )
            if returncode != 0:
                raise RuntimeError(
                    f"polymarket_cycle exited with code {returncode}"
                )
    except Exception:
        logger.exception("heavy_polymarket_cycle failed")
        raise


async def heavy_tkg_retrain() -> None:
    """Run weekly TKG model retraining in a subprocess.

    Acquires the heavy job lock, then dispatches ``run_tkg_retrain``
    to a ProcessPoolExecutor. Uses subprocess.run() internally because
    scripts/retrain_tkg.py relies on argparse and is not importable.
    """
    try:
        from src.scheduler.heavy_runner import run_tkg_retrain

        async with _heavy_job_lock:
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(
                _get_process_executor(), run_tkg_retrain
            )
            if returncode != 0:
                raise RuntimeError(
                    f"tkg_retrain subprocess exited with code {returncode}"
                )
    except Exception:
        logger.exception("heavy_tkg_retrain failed")
        raise
