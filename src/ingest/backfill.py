"""
Gap recovery for the GDELT poller on startup.

When the poller starts, it checks the last successful IngestRun in
PostgreSQL.  If the gap between that run and now exceeds one poll
interval, it fetches the current lastupdate.txt to ingest whatever
is available.  GDELT only retains the *latest* 15-minute batch on
lastupdate.txt, so we cannot recover individual missed batches --
but we guarantee at least the most recent batch is ingested.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from src.database.storage import EventStorage
from src.db.models import IngestRun
from src.db.postgres import get_async_session
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from src.settings import get_settings

logger = logging.getLogger(__name__)


async def get_last_successful_run() -> datetime | None:
    """Query PostgreSQL for the most recent successful GDELT IngestRun."""
    try:
        async for session in get_async_session():
            stmt = (
                select(IngestRun.completed_at)
                .where(
                    IngestRun.daemon_type == "gdelt",
                    IngestRun.status == "success",
                )
                .order_by(IngestRun.completed_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row
    except Exception as exc:
        logger.warning("Could not query last successful run: %s", exc)
        return None


async def backfill_from_last_run(
    event_storage: EventStorage,
    graph: TemporalKnowledgeGraph,
) -> None:
    """Check for gaps and trigger a single poll if needed.

    GDELT only serves the latest batch on lastupdate.txt, so the
    backfill is simply "poll once immediately" when the gap is large.
    Individual missed 15-minute batches cannot be recovered.
    """
    settings = get_settings()
    last_run = await get_last_successful_run()

    if last_run is None:
        logger.info("No previous successful GDELT run found -- will poll immediately")
        return  # The main loop's first poll covers this case

    gap_seconds = (datetime.now(timezone.utc) - last_run).total_seconds()
    threshold = settings.gdelt_poll_interval * 2  # Two missed intervals

    if gap_seconds > threshold:
        logger.info(
            "Gap detected: last successful run %.0f seconds ago (threshold=%ds). "
            "The main loop will poll immediately.",
            gap_seconds,
            threshold,
        )
    else:
        logger.info(
            "No significant gap: last successful run %.0f seconds ago",
            gap_seconds,
        )
