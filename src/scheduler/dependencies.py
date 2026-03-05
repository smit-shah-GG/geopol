"""
Shared dependency container for scheduler job wrappers.

Holds singleton instances of heavy-to-construct objects (EventStorage,
TemporalKnowledgeGraph, GDELTPoller) that are reused across poll cycles.

The GDELT poller MUST be a singleton so ``_last_url`` persists between
poll cycles (avoids re-downloading the same GDELT export every interval).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.database.storage import EventStorage
from src.ingest.gdelt_poller import GDELTPoller
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from src.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_shared_deps: SharedDeps | None = None


@dataclass
class SharedDeps:
    """Container for shared, long-lived dependencies injected into job wrappers.

    Attributes:
        settings: Application configuration singleton.
        event_storage: SQLite event storage (thread-safe via INSERT OR IGNORE).
        graph: Temporal knowledge graph (in-memory NetworkX MultiDiGraph).
        gdelt_poller: Singleton GDELT poller -- ``_last_url`` state persists
            between cycles to enable URL-dedup fast path.
    """

    settings: Settings
    event_storage: EventStorage
    graph: TemporalKnowledgeGraph
    gdelt_poller: GDELTPoller


def init_shared_deps() -> SharedDeps:
    """Initialize and return the shared dependency container.

    Creates singleton instances of EventStorage, TemporalKnowledgeGraph,
    and GDELTPoller. The GDELT poller is constructed once so that
    ``_last_url`` state persists across poll cycles.

    Raises:
        RuntimeError: If deps were already initialized.

    Returns:
        The initialized SharedDeps instance.
    """
    global _shared_deps  # noqa: PLW0603

    if _shared_deps is not None:
        raise RuntimeError("SharedDeps already initialized -- call get_shared_deps()")

    settings = get_settings()
    event_storage = EventStorage(db_path=settings.gdelt_db_path)
    graph = TemporalKnowledgeGraph()
    gdelt_poller = GDELTPoller(
        event_storage=event_storage,
        graph=graph,
        poll_interval=settings.gdelt_poll_interval,
    )

    _shared_deps = SharedDeps(
        settings=settings,
        event_storage=event_storage,
        graph=graph,
        gdelt_poller=gdelt_poller,
    )

    logger.info(
        "SharedDeps initialized (gdelt_db=%s, graph_nodes=%d)",
        settings.gdelt_db_path,
        graph.graph.number_of_nodes(),
    )
    return _shared_deps


def get_shared_deps() -> SharedDeps:
    """Return the shared dependency container.

    Raises:
        RuntimeError: If not yet initialized via ``init_shared_deps()``.
    """
    if _shared_deps is None:
        raise RuntimeError("SharedDeps not initialized -- call init_shared_deps() first")
    return _shared_deps
