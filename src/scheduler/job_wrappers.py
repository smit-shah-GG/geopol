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


async def _get_feeds_from_db(tier: int) -> list | None:
    """Query rss_feeds for enabled, non-deleted feeds of the given tier.

    Returns a list of FeedSource objects, or None if the DB is unreachable
    (caller should fall back to feed_config.py constants).
    """
    try:
        from sqlalchemy import select

        from src.db.models import RSSFeed
        from src.db.postgres import async_session_factory, init_db
        from src.ingest.feed_config import FeedSource, FeedTier

        if async_session_factory is None:
            init_db()

        async with async_session_factory() as session:
            result = await session.execute(
                select(RSSFeed).where(
                    RSSFeed.enabled.is_(True),
                    RSSFeed.deleted_at.is_(None),
                    RSSFeed.tier == tier,
                )
            )
            rows = result.scalars().all()

        return [
            FeedSource(
                name=row.name,
                url=row.url,
                tier=FeedTier(row.tier),
                category=row.category,
                lang=row.lang,
            )
            for row in rows
        ]
    except Exception as exc:
        logger.warning(
            "Failed to query rss_feeds for tier %d, falling back to feed_config: %s",
            tier,
            exc,
        )
        return None


async def _update_feed_health(
    per_feed_results: list,
) -> None:
    """Update rss_feeds health metrics after a poll cycle.

    For each feed result:
    - On success: reset error_count, clear last_error, update article counts
    - On error: increment error_count, set last_error
    - Auto-disable at 5 consecutive failures

    Non-fatal: logs and continues on DB errors.
    """
    try:
        from sqlalchemy import select, update

        from src.db.models import RSSFeed
        from src.db.postgres import async_session_factory, init_db

        if async_session_factory is None:
            init_db()

        now = datetime.now(timezone.utc)
        async with async_session_factory() as session:
            for result in per_feed_results:
                feed_name = result.feed_name

                # Look up the feed row
                row = await session.execute(
                    select(RSSFeed).where(RSSFeed.name == feed_name)
                )
                feed = row.scalar_one_or_none()
                if feed is None:
                    continue  # Feed was deleted between query and poll

                feed.last_poll_at = now

                if result.error:
                    # Feed fetch failed
                    feed.error_count += 1
                    feed.last_error = result.error[:2000]  # Truncate long errors

                    if feed.error_count >= 5 and feed.enabled:
                        feed.enabled = False
                        logger.warning(
                            "Feed %s auto-disabled after %d consecutive failures",
                            feed_name,
                            feed.error_count,
                        )
                else:
                    # Feed fetch succeeded -- reset error state
                    feed.error_count = 0
                    feed.last_error = None
                    feed.articles_total += result.articles_new
                    feed.articles_24h = result.articles_found  # Approximation
                    # Running average: weighted towards recent polls
                    total_polls = max(
                        feed.articles_total / max(feed.avg_articles_per_poll, 1.0),
                        1.0,
                    ) if feed.avg_articles_per_poll > 0 else 1.0
                    feed.avg_articles_per_poll = (
                        (feed.avg_articles_per_poll * (total_polls - 1) + result.articles_found)
                        / total_polls
                    )

            await session.commit()
    except Exception as exc:
        logger.warning("Failed to update feed health metrics: %s", exc)


async def _rss_poll_tier(tier: int, tier_label: str) -> None:
    """Shared logic for tier-1 and tier-2 RSS polling.

    1. Query rss_feeds DB for enabled feeds of the given tier (fallback to
       feed_config.py constants if DB is unreachable).
    2. Poll feeds via RSSDaemon.
    3. Update per-feed health metrics in rss_feeds table.
    4. Record IngestRun audit row.
    """
    from src.ingest.rss_daemon import CycleMetrics, DaemonConfig, RSSDaemon

    deps = get_shared_deps()
    config = DaemonConfig(
        tier1_interval=deps.settings.rss_poll_interval_tier1,
        tier2_interval=deps.settings.rss_poll_interval_tier2,
        retention_days=deps.settings.rss_article_retention_days,
    )

    # Step 1: Get feeds from DB (preferred) or fallback to constants
    feeds = await _get_feeds_from_db(tier)
    if feeds is None:
        # DB unavailable -- use hardcoded fallback
        if tier == 1:
            from src.ingest.feed_config import TIER_1_FEEDS
            feeds = TIER_1_FEEDS
        else:
            from src.ingest.feed_config import TIER_2_FEEDS
            feeds = TIER_2_FEEDS
        logger.warning(
            "rss_%s: using feed_config.py fallback (%d feeds)",
            tier_label,
            len(feeds),
        )

    if not feeds:
        logger.info("rss_%s: no enabled feeds for tier %d, skipping", tier_label, tier)
        return

    # Step 2: Poll
    daemon = RSSDaemon(config=config)
    try:
        metrics = await daemon.poll_feeds(feeds)
        logger.info(
            "rss_%s: feeds=%d, new=%d, dup=%d, chunks=%d, %.1fs",
            tier_label,
            metrics.feeds_polled,
            metrics.articles_new,
            metrics.articles_duplicate,
            metrics.chunks_indexed,
            metrics.duration_seconds,
        )

        # Step 3: Update per-feed health metrics
        if metrics.per_feed:
            await _update_feed_health(metrics.per_feed)

        # Step 4: Record IngestRun
        await daemon._record_ingest_run(metrics, tier_label)
    finally:
        if daemon._session and not daemon._session.closed:
            await daemon._session.close()


async def rss_poll_tier1() -> None:
    """Poll tier-1 RSS feeds from the rss_feeds DB table.

    Falls back to feed_config.py TIER_1_FEEDS if the database is
    unreachable. Updates per-feed health metrics after polling.
    """
    try:
        await _rss_poll_tier(tier=1, tier_label="tier-1")
    except Exception:
        logger.exception("rss_poll_tier1 failed")
        raise


async def rss_poll_tier2() -> None:
    """Poll tier-2 RSS feeds from the rss_feeds DB table.

    Falls back to feed_config.py TIER_2_FEEDS if the database is
    unreachable. Updates per-feed health metrics after polling.
    """
    try:
        await _rss_poll_tier(tier=2, tier_label="tier-2")
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


async def heavy_backtest(config_json: str) -> None:
    """Run a backtest in a ProcessPoolExecutor worker.

    Acquires the heavy job lock to prevent concurrent heavy workloads,
    then dispatches ``run_backtest`` with the serialized config. This
    wrapper is NOT registered as an APScheduler interval job -- it's
    called on-demand via asyncio.create_task() from AdminService.

    Args:
        config_json: JSON-serialized BacktestRunConfig string.
    """
    try:
        from src.scheduler.heavy_runner import run_backtest

        async with _heavy_job_lock:
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(
                _get_process_executor(), run_backtest, config_json
            )
            if returncode != 0:
                raise RuntimeError(
                    f"backtest run exited with code {returncode}"
                )
    except Exception:
        logger.exception("heavy_backtest failed")
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
