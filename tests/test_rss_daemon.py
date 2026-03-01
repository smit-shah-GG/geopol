"""
Unit tests for the RSS ingestion pipeline.

Tests cover:
  1. Article chunking (paragraph boundaries, sentence fallback)
  2. Article text extraction (success + failure paths)
  3. URL deduplication in ArticleIndexer
  4. ChromaDB indexing flow
  5. Feed config validation (no duplicates, valid URLs)
  6. Daemon lifecycle (start/stop)
  7. Concurrency limits (semaphore bounds)
  8. Feed fetching (HTTP success + failure)
  9. Article pruning
  10. Cycle metrics aggregation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingest.article_processor import (
    ArticleIndexer,
    ExtractionResult,
    chunk_article,
    extract_article_text,
)
from src.ingest.feed_config import (
    ALL_FEEDS,
    TIER_1_FEEDS,
    TIER_2_FEEDS,
    FeedSource,
    FeedTier,
    get_feeds_by_category,
    get_feeds_by_tier,
    get_propaganda_risk,
    validate_feeds,
)
from src.ingest.rss_daemon import (
    CycleMetrics,
    DaemonConfig,
    RSSDaemon,
    _fetch_feed,
)


# ---------------------------------------------------------------------------
# 1. Chunking: paragraph boundaries
# ---------------------------------------------------------------------------


class TestChunkArticle:
    def test_single_short_paragraph(self) -> None:
        text = "This is a short paragraph about sanctions on Russia."
        # Below min_length threshold (80 chars), should be dropped
        result = chunk_article(text, min_length=80)
        assert result == []

    def test_single_paragraph_above_min(self) -> None:
        text = (
            "The United Nations Security Council met today to discuss "
            "the ongoing humanitarian crisis in the Horn of Africa. "
            "Representatives from 15 member states participated."
        )
        result = chunk_article(text, min_length=50)
        assert len(result) == 1
        assert "United Nations" in result[0]

    def test_multiple_paragraphs_merged(self) -> None:
        """Short paragraphs should merge up to target chunk size."""
        text = "Short paragraph one.\n\nShort paragraph two.\n\nShort paragraph three."
        result = chunk_article(text, min_length=10)
        # All three are short, should merge into one chunk
        assert len(result) == 1
        assert "one" in result[0] and "three" in result[0]

    def test_oversized_paragraph_splits_on_sentences(self) -> None:
        """Paragraphs exceeding max_length split on sentence boundaries."""
        sentences = [f"Sentence number {i} is here." for i in range(50)]
        text = " ".join(sentences)
        result = chunk_article(text, max_length=200, min_length=10)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 200 + 50  # allow some slack for sentence boundary

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_article("") == []
        assert chunk_article("   \n\n  ") == []


# ---------------------------------------------------------------------------
# 2. Article extraction (mocked trafilatura)
# ---------------------------------------------------------------------------


class TestExtractArticleText:
    @patch("trafilatura.settings.use_config")
    @patch("trafilatura.extract")
    @patch("trafilatura.fetch_url")
    def test_successful_extraction(
        self,
        mock_fetch: MagicMock,
        mock_extract: MagicMock,
        mock_use_config: MagicMock,
    ) -> None:
        mock_use_config.return_value = MagicMock()
        mock_fetch.return_value = "<html>content</html>"
        long_text = "Extracted article text that is long enough to pass the minimum length check. " * 3
        mock_extract.side_effect = [
            long_text,
            '{"title": "Test Title", "date": "2026-03-01"}',
        ]

        result = extract_article_text("https://example.com/article")

        assert result.success is True
        assert result.url == "https://example.com/article"
        assert result.title == "Test Title"

    @patch("trafilatura.settings.use_config")
    @patch("trafilatura.fetch_url")
    def test_fetch_returns_none(
        self,
        mock_fetch: MagicMock,
        mock_use_config: MagicMock,
    ) -> None:
        mock_use_config.return_value = MagicMock()
        mock_fetch.return_value = None

        result = extract_article_text("https://example.com/404")

        assert result.success is False
        assert "None" in (result.error or "")


# ---------------------------------------------------------------------------
# 3. URL deduplication
# ---------------------------------------------------------------------------


class TestArticleIndexerDedup:
    def test_is_url_indexed_returns_false_for_new_url(self) -> None:
        indexer = ArticleIndexer.__new__(ArticleIndexer)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        indexer._collection = mock_collection
        indexer._client = MagicMock()

        assert indexer.is_url_indexed("https://example.com/new") is False

    def test_is_url_indexed_returns_true_for_existing(self) -> None:
        indexer = ArticleIndexer.__new__(ArticleIndexer)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["abc123"]}
        indexer._collection = mock_collection
        indexer._client = MagicMock()

        assert indexer.is_url_indexed("https://example.com/existing") is True


# ---------------------------------------------------------------------------
# 4. ChromaDB indexing flow
# ---------------------------------------------------------------------------


class TestArticleIndexerIndex:
    def test_index_article_skips_duplicate(self) -> None:
        indexer = ArticleIndexer.__new__(ArticleIndexer)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["existing"]}
        indexer._collection = mock_collection
        indexer._client = MagicMock()

        stats = indexer.index_article(
            url="https://example.com/dup",
            source_name="Test",
            text="Some long enough article text. " * 10,
        )
        assert stats.skipped_duplicate is True
        assert stats.chunks_indexed == 0

    def test_index_article_adds_chunks(self) -> None:
        indexer = ArticleIndexer.__new__(ArticleIndexer)
        mock_collection = MagicMock()
        # First call: dedup check returns empty
        # collection.get for dedup, then collection.add for indexing
        mock_collection.get.return_value = {"ids": []}
        indexer._collection = mock_collection
        indexer._client = MagicMock()

        text = (
            "A long article about geopolitical events that will produce chunks. " * 20
            + "\n\n"
            + "Another paragraph with significant content about international relations. " * 20
        )
        stats = indexer.index_article(
            url="https://example.com/new-article",
            source_name="BBC World",
            text=text,
        )
        assert stats.skipped_duplicate is False
        assert stats.chunks_indexed > 0
        mock_collection.add.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Feed config validation
# ---------------------------------------------------------------------------


class TestFeedConfig:
    def test_no_duplicate_names_or_urls(self) -> None:
        errors = validate_feeds()
        assert errors == [], f"Feed validation errors: {errors}"

    def test_tier_counts(self) -> None:
        assert len(TIER_1_FEEDS) >= 20, "Expected at least 20 tier-1 feeds"
        assert len(TIER_2_FEEDS) >= 40, "Expected at least 40 tier-2 feeds"
        assert len(ALL_FEEDS) == len(TIER_1_FEEDS) + len(TIER_2_FEEDS)

    def test_all_urls_are_http(self) -> None:
        for feed in ALL_FEEDS:
            assert feed.url.startswith(("http://", "https://")), (
                f"Feed {feed.name} has invalid URL: {feed.url}"
            )

    def test_get_feeds_by_tier(self) -> None:
        t1 = get_feeds_by_tier(FeedTier.TIER_1)
        t2 = get_feeds_by_tier(FeedTier.TIER_2)
        assert len(t1) == len(TIER_1_FEEDS)
        assert len(t2) == len(TIER_2_FEEDS)

    def test_propaganda_risk(self) -> None:
        assert get_propaganda_risk("Xinhua") == "high"
        assert get_propaganda_risk("TASS") == "high"
        assert get_propaganda_risk("Reuters World") == "low"


# ---------------------------------------------------------------------------
# 6. Daemon lifecycle (start/stop)
# ---------------------------------------------------------------------------


class TestRSSDaemonLifecycle:
    @pytest.mark.asyncio
    async def test_stop_terminates_loop(self) -> None:
        config = DaemonConfig(tier1_interval=1, tier2_interval=1)
        daemon = RSSDaemon(config=config)

        # Patch poll_feeds to avoid real network calls
        daemon.poll_feeds = AsyncMock(return_value=CycleMetrics())
        daemon._record_ingest_run = AsyncMock()

        # Start in background, stop after brief delay
        async def stop_after_delay() -> None:
            await asyncio.sleep(0.2)
            daemon.stop()

        stop_task = asyncio.create_task(stop_after_delay())

        # Override signal handler installation (not available in test context)
        daemon._install_signal_handlers = lambda: None

        await daemon.start()
        await stop_task

        assert daemon.running is False


# ---------------------------------------------------------------------------
# 7. Concurrency limits
# ---------------------------------------------------------------------------


class TestConcurrencyLimits:
    @pytest.mark.asyncio
    async def test_semaphore_bounds_concurrent_fetches(self) -> None:
        """Verify semaphore prevents more than max_concurrent simultaneous fetches."""
        config = DaemonConfig(max_concurrent_fetches=2)
        daemon = RSSDaemon(config=config)

        active_count = 0
        max_active = 0

        original_fetch = _fetch_feed

        async def counting_fetch(session, feed, semaphore, timeout=30):
            nonlocal active_count, max_active
            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1
            return []

        feeds = [
            FeedSource("Test1", "https://example.com/1", FeedTier.TIER_1, "wire"),
            FeedSource("Test2", "https://example.com/2", FeedTier.TIER_1, "wire"),
            FeedSource("Test3", "https://example.com/3", FeedTier.TIER_1, "wire"),
            FeedSource("Test4", "https://example.com/4", FeedTier.TIER_1, "wire"),
        ]

        with patch("src.ingest.rss_daemon._fetch_feed", counting_fetch):
            with patch("src.ingest.rss_daemon._process_entry", AsyncMock(return_value={"status": "new", "chunks": 1})):
                daemon._session = MagicMock()
                daemon._session.closed = False
                metrics = await daemon.poll_feeds(feeds)

        # Semaphore should have bounded concurrency
        assert max_active <= config.max_concurrent_fetches


# ---------------------------------------------------------------------------
# 8. Feed fetching
# ---------------------------------------------------------------------------


class TestFeedFetching:
    @pytest.mark.asyncio
    async def test_fetch_feed_success(self) -> None:
        rss_body = """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Test Article</title>
              <link>https://example.com/article1</link>
            </item>
          </channel>
        </rss>"""

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value=rss_body)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        feed = FeedSource("Test", "https://example.com/rss", FeedTier.TIER_1, "wire")
        sem = asyncio.Semaphore(10)

        entries = await _fetch_feed(mock_session, feed, sem)
        assert len(entries) == 1
        assert entries[0]["url"] == "https://example.com/article1"
        assert entries[0]["title"] == "Test Article"

    @pytest.mark.asyncio
    async def test_fetch_feed_http_error(self) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        feed = FeedSource("Test", "https://example.com/rss", FeedTier.TIER_1, "wire")
        sem = asyncio.Semaphore(10)

        entries = await _fetch_feed(mock_session, feed, sem)
        assert entries == []


# ---------------------------------------------------------------------------
# 9. Article pruning
# ---------------------------------------------------------------------------


class TestArticlePruning:
    def test_prune_removes_old_chunks(self) -> None:
        indexer = ArticleIndexer.__new__(ArticleIndexer)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["old1", "old2", "old3"]}
        indexer._collection = mock_collection
        indexer._client = MagicMock()

        deleted = indexer.prune_old_articles(retention_days=90)
        assert deleted == 3
        mock_collection.delete.assert_called_once_with(ids=["old1", "old2", "old3"])

    def test_prune_no_old_chunks(self) -> None:
        indexer = ArticleIndexer.__new__(ArticleIndexer)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        indexer._collection = mock_collection
        indexer._client = MagicMock()

        deleted = indexer.prune_old_articles(retention_days=90)
        assert deleted == 0


# ---------------------------------------------------------------------------
# 10. Cycle metrics aggregation
# ---------------------------------------------------------------------------


class TestCycleMetrics:
    def test_default_metrics(self) -> None:
        m = CycleMetrics()
        assert m.feeds_polled == 0
        assert m.articles_found == 0
        assert m.articles_new == 0
        assert m.articles_duplicate == 0
        assert m.articles_failed == 0
        assert m.chunks_indexed == 0
        assert m.duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_poll_feeds_aggregates_metrics(self) -> None:
        config = DaemonConfig(max_concurrent_fetches=5)
        daemon = RSSDaemon(config=config)

        test_feeds = [
            FeedSource("Test1", "https://example.com/1", FeedTier.TIER_1, "wire"),
        ]

        async def mock_fetch(session, feed, sem, timeout=30):
            return [{"url": "https://example.com/a1", "title": "Article", "published": ""}]

        async def mock_process(entry, feed, indexer, sem):
            return {"status": "new", "url": entry["url"], "chunks": 3}

        with patch("src.ingest.rss_daemon._fetch_feed", mock_fetch):
            with patch("src.ingest.rss_daemon._process_entry", mock_process):
                daemon._session = MagicMock()
                daemon._session.closed = False
                metrics = await daemon.poll_feeds(test_feeds)

        assert metrics.feeds_polled == 1
        assert metrics.articles_found == 1
        assert metrics.articles_new == 1
        assert metrics.chunks_indexed == 3
