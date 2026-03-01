"""
Unit tests for the GDELT micro-batch poller.

Tests cover:
    1. parse_lastupdate_txt -- parsing the GDELT feed format
    2. URL-dedup fast path -- skipping when URL unchanged
    3. BackoffStrategy -- exponential delay calculation
    4. IngestRun recording -- persisting metrics to PostgreSQL
    5. Shutdown signal handling -- clean exit on SIGTERM
    6. Poll failure -> backoff trigger
    7. Incremental graph update -- add_event_from_db_row via to_thread

All external I/O is mocked (aiohttp, PostgreSQL, SQLite, graph builder).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.ingest.gdelt_poller import (
    BackoffStrategy,
    GDELTPoller,
    GDELTUpdate,
    _gdelt_row_to_event,
    parse_lastupdate_txt,
)


# ---------------------------------------------------------------------------
# 1. parse_lastupdate_txt
# ---------------------------------------------------------------------------

class TestParseLastupdateTxt:
    """Verify parsing of the three-line GDELT lastupdate.txt format."""

    def test_valid_content(self) -> None:
        """Standard three-line format should extract the export CSV entry."""
        content = (
            "46566 05e1a247a6f0b62b1463e6f10bb7f465 "
            "http://data.gdeltproject.org/gdeltv2/20260301111500.export.CSV.zip\n"
            "78114 363c6917b7fad577f185e387d946592d "
            "http://data.gdeltproject.org/gdeltv2/20260301111500.mentions.CSV.zip\n"
            "2233462 c32977d11768f35949cd06d460bd5ab7 "
            "http://data.gdeltproject.org/gdeltv2/20260301111500.gkg.csv.zip\n"
        )
        result = parse_lastupdate_txt(content)
        assert result is not None
        assert result.filesize == 46566
        assert result.md5_hash == "05e1a247a6f0b62b1463e6f10bb7f465"
        assert result.url.endswith(".export.CSV.zip")

    def test_empty_content(self) -> None:
        """Empty string returns None."""
        assert parse_lastupdate_txt("") is None

    def test_no_export_line(self) -> None:
        """Content without .export.CSV.zip returns None."""
        content = (
            "78114 abc123 http://example.com/file.mentions.CSV.zip\n"
            "2233462 def456 http://example.com/file.gkg.csv.zip\n"
        )
        assert parse_lastupdate_txt(content) is None

    def test_malformed_filesize(self) -> None:
        """Non-integer filesize returns None."""
        content = "notanumber abc123 http://example.com/20260301.export.CSV.zip\n"
        assert parse_lastupdate_txt(content) is None


# ---------------------------------------------------------------------------
# 2. URL-dedup fast path (skip duplicate URL)
# ---------------------------------------------------------------------------

class TestUrlDedupFastPath:
    """When lastupdate.txt returns the same URL as the previous poll,
    the poller should skip download and record events_fetched=0."""

    @pytest.mark.asyncio
    async def test_skip_duplicate_url(self) -> None:
        """Second poll with same URL should not download the CSV."""
        storage = MagicMock()
        graph = MagicMock()

        poller = GDELTPoller(
            event_storage=storage,
            graph=graph,
            poll_interval=10,
        )

        # Simulate that last poll already processed this URL
        poller._last_url = "http://data.gdeltproject.org/gdeltv2/20260301111500.export.CSV.zip"

        lastupdate_text = (
            "46566 abc123 "
            "http://data.gdeltproject.org/gdeltv2/20260301111500.export.CSV.zip\n"
            "78114 def456 http://example.com/mentions.CSV.zip\n"
            "2233462 ghi789 http://example.com/gkg.csv.zip\n"
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value=lastupdate_text)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("src.ingest.gdelt_poller.aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch.object(poller, "_record_run", new_callable=AsyncMock) as mock_record:
                await poller._poll_once()

                # Should have recorded a run with 0 events fetched
                mock_record.assert_called_once()
                call_kwargs = mock_record.call_args
                assert call_kwargs.kwargs.get("events_fetched", call_kwargs[1].get("events_fetched")) == 0

        # insert_events should NOT have been called (no download)
        storage.insert_events.assert_not_called()


# ---------------------------------------------------------------------------
# 3. BackoffStrategy
# ---------------------------------------------------------------------------

class TestBackoffStrategy:
    """Verify exponential delay calculation with cap."""

    def test_initial_delay(self) -> None:
        """First failure should produce base delay."""
        bs = BackoffStrategy(base=60.0, max_delay=1800.0, jitter_fraction=0.0)
        delay = bs.next_delay()
        assert delay == 60.0

    def test_exponential_growth(self) -> None:
        """Delays should double with each failure (no jitter)."""
        bs = BackoffStrategy(base=60.0, max_delay=1800.0, jitter_fraction=0.0)
        delays = [bs.next_delay() for _ in range(5)]
        assert delays == [60.0, 120.0, 240.0, 480.0, 960.0]

    def test_max_cap(self) -> None:
        """Delay should not exceed max_delay."""
        bs = BackoffStrategy(base=60.0, max_delay=1800.0, jitter_fraction=0.0)
        for _ in range(20):
            delay = bs.next_delay()
        assert delay == 1800.0

    def test_reset_clears_failures(self) -> None:
        """Reset should bring delay back to base."""
        bs = BackoffStrategy(base=60.0, max_delay=1800.0, jitter_fraction=0.0)
        bs.next_delay()
        bs.next_delay()
        assert bs.failures == 2
        bs.reset()
        assert bs.failures == 0
        assert bs.next_delay() == 60.0

    def test_jitter_within_bounds(self) -> None:
        """With jitter, delay should be in [base, base * (1 + jitter_fraction)]."""
        bs = BackoffStrategy(base=60.0, max_delay=1800.0, jitter_fraction=0.1)
        for _ in range(100):
            bs._failures = 0  # Reset for consistent base
            delay = bs.next_delay()
            assert 60.0 <= delay <= 66.0  # 60 + 10% jitter


# ---------------------------------------------------------------------------
# 4. IngestRun recording
# ---------------------------------------------------------------------------

class TestRecordIngestRun:
    """Verify that _record_run persists an IngestRun via async session."""

    @pytest.mark.asyncio
    async def test_record_run_creates_ingest_run(self) -> None:
        """A successful poll should persist an IngestRun row."""
        storage = MagicMock()
        graph = MagicMock()

        poller = GDELTPoller(event_storage=storage, graph=graph)

        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        async def _fake_get_session():
            yield mock_session

        with patch(
            "src.ingest.gdelt_poller.get_async_session",
            return_value=_fake_get_session(),
        ):
            await poller._record_run(
                started_at=datetime.now(timezone.utc),
                status="success",
                events_fetched=100,
                events_new=42,
                events_duplicate=58,
            )

        # session.add should have been called with an IngestRun
        mock_session.add.assert_called_once()
        run = mock_session.add.call_args[0][0]
        assert run.status == "success"
        assert run.daemon_type == "gdelt"
        assert run.events_fetched == 100
        assert run.events_new == 42
        assert run.events_duplicate == 58


# ---------------------------------------------------------------------------
# 5. Shutdown signal handling
# ---------------------------------------------------------------------------

class TestShutdownSignal:
    """Verify that setting the shutdown event terminates the poll loop."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_loop(self) -> None:
        """Calling _handle_shutdown should cause run() to exit."""
        storage = MagicMock()
        graph = MagicMock()
        poller = GDELTPoller(event_storage=storage, graph=graph, poll_interval=1)

        # Pre-set shutdown so run() exits immediately
        poller._shutdown.set()

        with patch("src.ingest.gdelt_poller.init_db"):
            with patch("src.ingest.gdelt_poller.get_settings") as mock_settings:
                mock_settings.return_value.gdelt_backfill_on_start = False
                mock_settings.return_value.gdelt_poll_interval = 1
                mock_settings.return_value.log_level = "WARNING"

                # run() should return without hanging
                await asyncio.wait_for(poller.run(), timeout=5.0)

    def test_handle_shutdown_sets_event(self) -> None:
        """_handle_shutdown should set the asyncio.Event."""
        storage = MagicMock()
        graph = MagicMock()
        poller = GDELTPoller(event_storage=storage, graph=graph)

        assert not poller._shutdown.is_set()
        poller._handle_shutdown()
        assert poller._shutdown.is_set()


# ---------------------------------------------------------------------------
# 6. Poll failure triggers backoff
# ---------------------------------------------------------------------------

class TestPollFailureTriggersBackoff:
    """When _poll_once raises, the main loop should increment backoff."""

    @pytest.mark.asyncio
    async def test_failure_increments_backoff(self) -> None:
        """A failed poll should cause backoff counter to increase."""
        bs = BackoffStrategy(base=0.01, max_delay=0.1, jitter_fraction=0.0)
        storage = MagicMock()
        graph = MagicMock()
        poller = GDELTPoller(
            event_storage=storage,
            graph=graph,
            poll_interval=1,
            backoff=bs,
        )

        call_count = 0

        async def _failing_poll() -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("GDELT feed unreachable")
            # Third call succeeds -- then we trigger shutdown
            poller._handle_shutdown()

        with patch("src.ingest.gdelt_poller.init_db"):
            with patch("src.ingest.gdelt_poller.get_settings") as mock_settings:
                mock_settings.return_value.gdelt_backfill_on_start = False
                mock_settings.return_value.gdelt_poll_interval = 1
                mock_settings.return_value.log_level = "WARNING"

                with patch.object(poller, "_poll_once", side_effect=_failing_poll):
                    with patch.object(poller, "_record_run", new_callable=AsyncMock):
                        await asyncio.wait_for(poller.run(), timeout=5.0)

        # Two failures should have happened before the success + shutdown
        assert call_count == 3


# ---------------------------------------------------------------------------
# 7. Incremental graph update
# ---------------------------------------------------------------------------

class TestIncrementalGraphUpdate:
    """Verify that new events are added to the graph via add_event_from_db_row."""

    @pytest.mark.asyncio
    async def test_incremental_update_calls_graph(self) -> None:
        """Events with quad_class 1 or 4 should be passed to the graph."""
        from src.database.models import Event

        storage = MagicMock()
        graph = MagicMock()
        graph.add_event_from_db_row = MagicMock(return_value=("USA", "RUS"))

        poller = GDELTPoller(event_storage=storage, graph=graph)

        events = [
            Event(
                gdelt_id="1001",
                content_hash="abc",
                time_window="2026-03-01",
                event_date="2026-03-01",
                actor1_code="USA",
                actor2_code="RUS",
                event_code="042",
                quad_class=4,  # Material conflict
            ),
            Event(
                gdelt_id="1002",
                content_hash="def",
                time_window="2026-03-01",
                event_date="2026-03-01",
                actor1_code="CHN",
                actor2_code="JPN",
                event_code="010",
                quad_class=1,  # Verbal cooperation
            ),
            Event(
                gdelt_id="1003",
                content_hash="ghi",
                time_window="2026-03-01",
                event_date="2026-03-01",
                actor1_code="FRA",
                actor2_code="DEU",
                event_code="020",
                quad_class=2,  # Material cooperation -- should be skipped
            ),
        ]

        added = await poller._update_graph_incremental(events)

        # Only quad_class 1 and 4 should be processed
        assert graph.add_event_from_db_row.call_count == 2
        assert added == 2


# ---------------------------------------------------------------------------
# 8. GDELT row to Event conversion
# ---------------------------------------------------------------------------

class TestGdeltRowToEvent:
    """Verify _gdelt_row_to_event converts DataFrame rows correctly."""

    def test_standard_row(self) -> None:
        """Standard GDELT CSV row should produce a valid Event."""
        row = pd.Series({
            "GLOBALEVENTID": "123456789",
            "SQLDATE": "20260301",
            "Actor1Code": "USA",
            "Actor2Code": "RUS",
            "EventCode": "042",
            "QuadClass": 4,
            "GoldsteinScale": -5.0,
            "NumMentions": 150,
            "NumSources": 25,
            "AvgTone": -3.5,
            "ActionGeo_FullName": "Moscow, Russia",
            "SOURCEURL": "http://example.com/article",
        })

        event = _gdelt_row_to_event(row)

        assert event.gdelt_id == "123456789"
        assert event.event_date == "2026-03-01"
        assert event.actor1_code == "USA"
        assert event.actor2_code == "RUS"
        assert event.quad_class == 4
        assert event.goldstein_scale == -5.0
        assert event.num_mentions == 150
        assert event.tone == -3.5
        assert event.content_hash is not None
        assert len(event.content_hash) == 32

    def test_missing_optional_fields(self) -> None:
        """Row with NaN optional fields should produce None values."""
        row = pd.Series({
            "GLOBALEVENTID": "999",
            "SQLDATE": "20260301",
            "Actor1Code": float("nan"),
            "Actor2Code": float("nan"),
            "EventCode": float("nan"),
            "QuadClass": float("nan"),
            "GoldsteinScale": float("nan"),
            "NumMentions": float("nan"),
            "NumSources": float("nan"),
            "AvgTone": float("nan"),
            "ActionGeo_FullName": "",
            "SOURCEURL": float("nan"),
        })

        event = _gdelt_row_to_event(row)

        assert event.gdelt_id == "999"
        assert event.actor1_code is None
        assert event.actor2_code is None
        assert event.quad_class is None
        assert event.goldstein_scale is None
