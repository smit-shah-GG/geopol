"""
Async GDELT micro-batch polling daemon.

Fetches lastupdate.txt every poll_interval seconds, downloads new
export CSVs when the URL changes, deduplicates events via INSERT OR
IGNORE on gdelt_id, inserts into SQLite, and incrementally updates
the temporal knowledge graph.  Per-run metrics are persisted to the
PostgreSQL IngestRun table.

Designed for systemd: graceful SIGTERM shutdown, exponential backoff
on feed failures, and gap recovery on startup (backfill module).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import random
import signal
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import pandas as pd

from src.database.models import Event
from src.database.storage import EventStorage
from src.db.models import IngestRun
from src.db.postgres import get_async_session, init_db
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from src.settings import get_settings

logger = logging.getLogger(__name__)

LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GDELT 2.0 Event Database CSV columns (61 columns, tab-separated, no header)
GDELT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]


@dataclass
class GDELTUpdate:
    """Parsed entry from lastupdate.txt -- the export CSV line."""

    filesize: int
    md5_hash: str
    url: str


class BackoffStrategy:
    """Exponential backoff: 1min -> 2min -> 4min -> ... -> max 30min.

    Includes 10% jitter to avoid thundering-herd synchronisation.
    """

    def __init__(
        self,
        base: float = 60.0,
        max_delay: float = 1800.0,
        jitter_fraction: float = 0.1,
    ) -> None:
        self.base = base
        self.max_delay = max_delay
        self.jitter_fraction = jitter_fraction
        self._failures = 0

    @property
    def failures(self) -> int:
        return self._failures

    def next_delay(self) -> float:
        """Compute the next delay without sleeping (useful for testing)."""
        delay = min(self.base * (2 ** self._failures), self.max_delay)
        jitter = random.uniform(0, delay * self.jitter_fraction)
        self._failures += 1
        return delay + jitter

    async def wait(self) -> float:
        """Sleep for the computed backoff duration.  Returns actual seconds waited."""
        actual = self.next_delay()
        await asyncio.sleep(actual)
        return actual

    def reset(self) -> None:
        self._failures = 0


def parse_lastupdate_txt(text: str) -> Optional[GDELTUpdate]:
    """Parse the GDELT lastupdate.txt content, return the export CSV entry.

    Format: three lines, space-separated ``{filesize} {md5} {url}``.
    We only care about the ``.export.CSV.zip`` line.
    """
    for line in text.strip().split("\n"):
        parts = line.strip().split(" ")
        if len(parts) == 3 and parts[2].endswith(".export.CSV.zip"):
            try:
                return GDELTUpdate(
                    filesize=int(parts[0]),
                    md5_hash=parts[1],
                    url=parts[2],
                )
            except (ValueError, IndexError):
                logger.warning("Malformed lastupdate.txt line: %s", line)
                return None
    return None


def _gdelt_row_to_event(row: pd.Series) -> Event:
    """Convert a single GDELT CSV DataFrame row to an Event dataclass."""
    gdelt_id = str(row.get("GLOBALEVENTID", ""))
    actor1 = str(row.get("Actor1Code", "")) if pd.notna(row.get("Actor1Code")) else None
    actor2 = str(row.get("Actor2Code", "")) if pd.notna(row.get("Actor2Code")) else None
    event_code = str(row.get("EventCode", "")) if pd.notna(row.get("EventCode")) else None

    # Build content hash for composite dedup
    hash_input = f"{actor1}|{actor2}|{event_code}|{row.get('ActionGeo_FullName', '')}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    # Parse SQLDATE (YYYYMMDD) -> YYYY-MM-DD
    raw_date = str(row.get("SQLDATE", ""))
    if len(raw_date) == 8:
        event_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        event_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Time window: hour-resolution bucket for dedup index
    time_window = event_date  # day-level granularity for 15-min updates

    quad_class: Optional[int] = None
    if pd.notna(row.get("QuadClass")):
        try:
            quad_class = int(row["QuadClass"])
        except (ValueError, TypeError):
            pass

    goldstein: Optional[float] = None
    if pd.notna(row.get("GoldsteinScale")):
        try:
            goldstein = float(row["GoldsteinScale"])
        except (ValueError, TypeError):
            pass

    mentions: Optional[int] = None
    if pd.notna(row.get("NumMentions")):
        try:
            mentions = int(row["NumMentions"])
        except (ValueError, TypeError):
            pass

    sources: Optional[int] = None
    if pd.notna(row.get("NumSources")):
        try:
            sources = int(row["NumSources"])
        except (ValueError, TypeError):
            pass

    tone: Optional[float] = None
    if pd.notna(row.get("AvgTone")):
        try:
            tone = float(row["AvgTone"])
        except (ValueError, TypeError):
            pass

    return Event(
        gdelt_id=gdelt_id,
        content_hash=content_hash,
        time_window=time_window,
        event_date=event_date,
        actor1_code=actor1,
        actor2_code=actor2,
        event_code=event_code,
        quad_class=quad_class,
        goldstein_scale=goldstein,
        num_mentions=mentions,
        num_sources=sources,
        tone=tone,
        url=str(row.get("SOURCEURL", "")) if pd.notna(row.get("SOURCEURL")) else None,
    )


class GDELTPoller:
    """Micro-batch GDELT event ingestion daemon.

    Lifecycle:
        1. (optional) Backfill gap from last successful IngestRun.
        2. Poll lastupdate.txt every ``poll_interval`` seconds.
        3. URL-dedup fast path: skip if same URL as last poll.
        4. Download + extract export CSV, convert rows to Event objects.
        5. Insert via EventStorage.insert_events() (INSERT OR IGNORE on gdelt_id).
        6. Incrementally add new events to TemporalKnowledgeGraph.
        7. Record IngestRun metrics to PostgreSQL.
        8. On SIGTERM, mark current run as interrupted and exit cleanly.
    """

    def __init__(
        self,
        event_storage: EventStorage,
        graph: TemporalKnowledgeGraph,
        poll_interval: Optional[int] = None,
        backoff: Optional[BackoffStrategy] = None,
    ) -> None:
        settings = get_settings()
        self.event_storage = event_storage
        self.graph = graph
        self.poll_interval = poll_interval or settings.gdelt_poll_interval
        self.backoff = backoff or BackoffStrategy()

        self._shutdown = asyncio.Event()
        self._last_url: Optional[str] = None
        self._current_run_id: Optional[int] = None

    async def run(self) -> None:
        """Main event loop with graceful shutdown support."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Initialise PostgreSQL engine (needed for IngestRun persistence)
        init_db()

        logger.info(
            "GDELT poller starting (interval=%ds, backfill=%s)",
            self.poll_interval,
            get_settings().gdelt_backfill_on_start,
        )

        if get_settings().gdelt_backfill_on_start:
            from src.ingest.backfill import backfill_from_last_run

            await backfill_from_last_run(self.event_storage, self.graph)

        while not self._shutdown.is_set():
            run_start = datetime.now(timezone.utc)
            try:
                await self._poll_once()
                self.backoff.reset()
            except Exception as exc:
                logger.error("Poll cycle failed: %s", exc, exc_info=True)
                await self._record_run(
                    started_at=run_start,
                    status="failed",
                    error_message=str(exc),
                )
                delay = self.backoff.next_delay()
                logger.info("Backing off for %.0fs (failure #%d)", delay, self.backoff.failures)
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                continue

            # Wait for next poll interval (or shutdown signal)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                pass  # Normal timeout -- poll again

        # Graceful cleanup
        if self._current_run_id is not None:
            await self._mark_interrupted(self._current_run_id)
        logger.info("GDELT poller shut down cleanly")

    def _handle_shutdown(self) -> None:
        """Signal handler for SIGTERM/SIGINT."""
        logger.info("Received shutdown signal, finishing current cycle...")
        self._shutdown.set()

    async def _poll_once(self) -> None:
        """Execute a single poll cycle: fetch -> dedup -> insert -> graph."""
        run_start = datetime.now(timezone.utc)

        async with aiohttp.ClientSession() as session:
            # 1. Fetch lastupdate.txt
            update = await self._fetch_lastupdate(session)
            if update is None:
                raise RuntimeError("Failed to fetch or parse lastupdate.txt")

            # 2. URL-dedup fast path
            if update.url == self._last_url:
                logger.debug("Same URL as last poll, skipping: %s", update.url)
                await self._record_run(
                    started_at=run_start,
                    status="success",
                    events_fetched=0,
                    events_new=0,
                    events_duplicate=0,
                )
                return

            # 3. Download and parse CSV
            logger.info("New GDELT export: %s (%d bytes)", update.url, update.filesize)
            df = await self._download_and_parse(session, update.url)
            events = [_gdelt_row_to_event(row) for _, row in df.iterrows()]
            events_fetched = len(events)

            # 4. Insert into SQLite (dedup via INSERT OR IGNORE on gdelt_id)
            events_new = await asyncio.to_thread(
                self.event_storage.insert_events, events
            )
            events_duplicate = events_fetched - events_new

            # 5. Incrementally update knowledge graph (O(N_new) only)
            graph_added = 0
            if events_new > 0:
                graph_added = await self._update_graph_incremental(events)

            logger.info(
                "Poll complete: fetched=%d, new=%d, dup=%d, graph_added=%d",
                events_fetched,
                events_new,
                events_duplicate,
                graph_added,
            )

            # 6. Record metrics
            await self._record_run(
                started_at=run_start,
                status="success",
                events_fetched=events_fetched,
                events_new=events_new,
                events_duplicate=events_duplicate,
            )

            self._last_url = update.url

    async def _fetch_lastupdate(
        self, session: aiohttp.ClientSession
    ) -> Optional[GDELTUpdate]:
        """Fetch and parse lastupdate.txt."""
        try:
            async with session.get(LASTUPDATE_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.warning("lastupdate.txt returned HTTP %d", resp.status)
                    return None
                text = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Failed to fetch lastupdate.txt: %s", exc)
            return None

        return parse_lastupdate_txt(text)

    async def _download_and_parse(
        self, session: aiohttp.ClientSession, url: str
    ) -> pd.DataFrame:
        """Download GDELT export CSV.zip and parse to DataFrame."""
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"GDELT CSV download failed: HTTP {resp.status}")
            data = await resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".CSV")]
            if not csv_names:
                raise RuntimeError(f"No CSV found in {url}")
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=GDELT_COLUMNS,
                    dtype={"GLOBALEVENTID": str, "EventCode": str},
                    low_memory=False,
                )
        return df

    async def _update_graph_incremental(self, events: list[Event]) -> int:
        """Add newly inserted events to the knowledge graph.

        Only processes events that have quad_class 1 or 4 (diplomatic /
        material conflict) to match the graph builder's default filter.
        Wraps the synchronous graph builder in asyncio.to_thread().
        """
        relevant = [
            e for e in events
            if e.quad_class in (1, 4) and e.actor1_code and e.actor2_code
        ]

        if not relevant:
            return 0

        def _add_to_graph() -> int:
            added = 0
            for event in relevant:
                row_dict = event.to_dict()
                result = self.graph.add_event_from_db_row(row_dict)
                if result is not None:
                    added += 1
            return added

        return await asyncio.to_thread(_add_to_graph)

    async def _record_run(
        self,
        started_at: datetime,
        status: str,
        events_fetched: int = 0,
        events_new: int = 0,
        events_duplicate: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Persist an IngestRun row to PostgreSQL."""
        try:
            async for session in get_async_session():
                run = IngestRun(
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    status=status,
                    daemon_type="gdelt",
                    events_fetched=events_fetched,
                    events_new=events_new,
                    events_duplicate=events_duplicate,
                    error_message=error_message,
                )
                session.add(run)
                # session commits on exit via get_async_session context manager
        except Exception as exc:
            # PostgreSQL may be down -- log but do not crash the daemon
            logger.warning("Failed to record IngestRun: %s", exc)

    async def _mark_interrupted(self, run_id: int) -> None:
        """Mark a running IngestRun as interrupted on shutdown."""
        try:
            async for session in get_async_session():
                from sqlalchemy import update

                stmt = (
                    update(IngestRun)
                    .where(IngestRun.id == run_id)
                    .values(
                        status="interrupted",
                        completed_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
        except Exception as exc:
            logger.warning("Failed to mark run %d as interrupted: %s", run_id, exc)
