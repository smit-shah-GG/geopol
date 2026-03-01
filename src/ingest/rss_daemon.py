"""
Async RSS polling daemon with tiered scheduling.

Polls TIER_1 feeds every 15 minutes and TIER_2 feeds every 60 minutes.
Fetches article URLs from RSS entries, extracts text via trafilatura,
chunks on paragraph boundaries, and indexes into ChromaDB.

Concurrency is bounded by a semaphore (default max 10 parallel fetches)
to avoid overwhelming downstream services or the host NIC.

Each poll cycle records an IngestRun row with daemon_type='rss' for
audit and monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import feedparser

from .article_processor import ArticleIndexer, ExtractionResult, extract_article_text
from .feed_config import (
    ALL_FEEDS,
    TIER_1_FEEDS,
    TIER_2_FEEDS,
    FeedSource,
    FeedTier,
)

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Metrics for a single poll cycle."""

    feeds_polled: int = 0
    articles_found: int = 0
    articles_new: int = 0
    articles_duplicate: int = 0
    articles_failed: int = 0
    chunks_indexed: int = 0
    duration_seconds: float = 0.0


@dataclass
class DaemonConfig:
    """Runtime configuration for the RSS daemon."""

    tier1_interval: int = 900  # 15 minutes in seconds
    tier2_interval: int = 3600  # 60 minutes in seconds
    max_concurrent_fetches: int = 10
    prune_interval: int = 86400  # Prune once per day
    retention_days: int = 90
    chroma_persist_dir: str = "./chroma_db"
    request_timeout: int = 30
    max_articles_per_feed: int = 20  # Cap per-feed to prevent runaway


async def _fetch_feed(
    session: aiohttp.ClientSession,
    feed: FeedSource,
    semaphore: asyncio.Semaphore,
    timeout: int = 30,
) -> list[dict[str, str]]:
    """
    Fetch and parse an RSS feed, returning article entries.

    Each entry is a dict with keys: url, title, published.
    Returns empty list on failure (logged, not raised).
    """
    async with semaphore:
        try:
            async with session.get(
                feed.url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": "geopol-rss/2.0 (+https://github.com/smit-shah-GG/geopol)"},
            ) as resp:
                if resp.status != 200:
                    logger.warning("Feed %s returned HTTP %d", feed.name, resp.status)
                    return []
                body = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Failed to fetch feed %s: %s", feed.name, exc)
            return []

    # feedparser is synchronous -- run in executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    parsed = await loop.run_in_executor(None, feedparser.parse, body)

    entries: list[dict[str, str]] = []
    for entry in parsed.entries:
        url = getattr(entry, "link", None)
        if not url:
            continue
        entries.append({
            "url": url,
            "title": getattr(entry, "title", ""),
            "published": getattr(entry, "published", ""),
        })

    return entries


async def _process_entry(
    entry: dict[str, str],
    feed: FeedSource,
    indexer: ArticleIndexer,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Extract and index a single article entry.

    Returns a result dict with status info for metric aggregation.
    """
    url = entry["url"]
    async with semaphore:
        # trafilatura is synchronous -- offload to executor
        loop = asyncio.get_running_loop()
        result: ExtractionResult = await loop.run_in_executor(
            None, extract_article_text, url
        )

    if not result.success or not result.text:
        return {"status": "failed", "url": url, "error": result.error}

    # Index (also sync -- offload)
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(
        None,
        indexer.index_article,
        url,
        feed.name,
        result.text,
        result.title or entry.get("title"),
        result.published_at or entry.get("published"),
    )

    if stats.skipped_duplicate:
        return {"status": "duplicate", "url": url}
    if stats.error:
        return {"status": "failed", "url": url, "error": stats.error}

    return {"status": "new", "url": url, "chunks": stats.chunks_indexed}


class RSSDaemon:
    """
    Async RSS polling daemon with tiered scheduling.

    Lifecycle:
      1. start() -- enters main loop
      2. Polls tier-1 feeds at tier1_interval, tier-2 at tier2_interval
      3. Periodic pruning of articles older than retention_days
      4. stop() or SIGTERM/SIGINT -- graceful shutdown
    """

    def __init__(self, config: Optional[DaemonConfig] = None) -> None:
        self.config = config or DaemonConfig()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._indexer = ArticleIndexer(
            persist_dir=self.config.chroma_persist_dir,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_fetches)
        self._last_tier1_poll: float = 0.0
        self._last_tier2_poll: float = 0.0
        self._last_prune: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("Received %s, initiating graceful shutdown...", sig.name)
        self._shutdown_event.set()

    async def poll_feeds(self, feeds: list[FeedSource]) -> CycleMetrics:
        """
        Poll a set of feeds: fetch RSS, extract articles, index chunks.

        Returns aggregated metrics for the cycle.
        """
        metrics = CycleMetrics()
        start = time.monotonic()
        session = await self._get_session()

        # Phase 1: fetch all feeds concurrently (bounded by semaphore)
        feed_tasks = [
            _fetch_feed(session, feed, self._semaphore, self.config.request_timeout)
            for feed in feeds
        ]
        feed_results = await asyncio.gather(*feed_tasks, return_exceptions=True)

        # Flatten entries with source attribution
        all_entries: list[tuple[dict[str, str], FeedSource]] = []
        for feed, result in zip(feeds, feed_results):
            if isinstance(result, Exception):
                logger.error("Feed %s raised exception: %s", feed.name, result)
                continue
            metrics.feeds_polled += 1
            # Cap entries per feed
            capped = result[: self.config.max_articles_per_feed]
            metrics.articles_found += len(capped)
            for entry in capped:
                all_entries.append((entry, feed))

        # Phase 2: process articles concurrently (bounded by semaphore)
        article_tasks = [
            _process_entry(entry, feed, self._indexer, self._semaphore)
            for entry, feed in all_entries
        ]
        article_results = await asyncio.gather(*article_tasks, return_exceptions=True)

        for result in article_results:
            if isinstance(result, Exception):
                metrics.articles_failed += 1
                continue
            status = result.get("status")
            if status == "new":
                metrics.articles_new += 1
                metrics.chunks_indexed += result.get("chunks", 0)
            elif status == "duplicate":
                metrics.articles_duplicate += 1
            else:
                metrics.articles_failed += 1

        metrics.duration_seconds = time.monotonic() - start
        return metrics

    async def _record_ingest_run(
        self, metrics: CycleMetrics, tier_label: str,
    ) -> None:
        """
        Record an IngestRun row in PostgreSQL.

        Logs and continues on failure -- the daemon must not crash
        because of a transient DB issue.
        """
        try:
            from ..db.postgres import get_async_session
            from ..db.models import IngestRun

            started_at = datetime.now(timezone.utc)
            async with get_async_session() as session:
                run = IngestRun(
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    status="success",
                    daemon_type="rss",
                    events_fetched=metrics.articles_found,
                    events_new=metrics.articles_new,
                    events_duplicate=metrics.articles_duplicate,
                    error_message=None,
                )
                session.add(run)
                await session.commit()

        except Exception as exc:
            # Non-fatal: log and continue. DB may not be available in dev.
            logger.warning(
                "Failed to record IngestRun for %s cycle: %s", tier_label, exc
            )

    async def _maybe_prune(self) -> None:
        """Prune old articles if enough time has passed since last prune."""
        now = time.monotonic()
        if now - self._last_prune < self.config.prune_interval:
            return

        logger.info("Starting periodic article pruning (retention=%d days)", self.config.retention_days)
        loop = asyncio.get_running_loop()
        deleted = await loop.run_in_executor(
            None,
            self._indexer.prune_old_articles,
            self.config.retention_days,
        )
        self._last_prune = now
        logger.info("Pruning complete: %d chunks deleted", deleted)

    async def _tick(self) -> None:
        """Single iteration of the main loop. Polls whichever tier is due."""
        now = time.monotonic()

        if now - self._last_tier1_poll >= self.config.tier1_interval:
            logger.info("Polling %d tier-1 feeds...", len(TIER_1_FEEDS))
            metrics = await self.poll_feeds(TIER_1_FEEDS)
            self._last_tier1_poll = now
            logger.info(
                "Tier-1 cycle: %d feeds, %d found, %d new, %d dup, %d failed, %.1fs",
                metrics.feeds_polled, metrics.articles_found,
                metrics.articles_new, metrics.articles_duplicate,
                metrics.articles_failed, metrics.duration_seconds,
            )
            await self._record_ingest_run(metrics, "tier-1")

        if now - self._last_tier2_poll >= self.config.tier2_interval:
            logger.info("Polling %d tier-2 feeds...", len(TIER_2_FEEDS))
            metrics = await self.poll_feeds(TIER_2_FEEDS)
            self._last_tier2_poll = now
            logger.info(
                "Tier-2 cycle: %d feeds, %d found, %d new, %d dup, %d failed, %.1fs",
                metrics.feeds_polled, metrics.articles_found,
                metrics.articles_new, metrics.articles_duplicate,
                metrics.articles_failed, metrics.duration_seconds,
            )
            await self._record_ingest_run(metrics, "tier-2")

        await self._maybe_prune()

    async def start(self) -> None:
        """
        Enter the main daemon loop.

        Runs until stop() is called or SIGTERM/SIGINT received.
        First iteration fires immediately for both tiers.
        """
        self._running = True
        self._shutdown_event.clear()
        self._install_signal_handlers()

        logger.info(
            "RSS daemon starting: %d tier-1 feeds (every %ds), %d tier-2 feeds (every %ds)",
            len(TIER_1_FEEDS), self.config.tier1_interval,
            len(TIER_2_FEEDS), self.config.tier2_interval,
        )

        # Fire immediately on startup
        self._last_tier1_poll = 0.0
        self._last_tier2_poll = 0.0
        self._last_prune = 0.0

        try:
            while not self._shutdown_event.is_set():
                await self._tick()
                # Sleep in short intervals so shutdown is responsive
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    pass  # Normal: just loop back
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("RSS daemon stopped.")

    def stop(self) -> None:
        """Signal the daemon to stop at the next loop iteration."""
        self._shutdown_event.set()

    @property
    def running(self) -> bool:
        return self._running
