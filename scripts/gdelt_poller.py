#!/usr/bin/env python
"""
Entry point for the GDELT micro-batch polling daemon.

Usage:
    python scripts/gdelt_poller.py [--interval SECONDS] [--no-backfill]
    python -m scripts.gdelt_poller

Designed to run under systemd (geopol-gdelt-poller.service) with
SIGTERM for graceful shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.database.storage import EventStorage
from src.ingest.gdelt_poller import GDELTPoller
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph
from src.settings import get_settings


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GDELT micro-batch polling daemon"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Poll interval in seconds (default: from settings, 900)",
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Skip gap recovery on startup",
    )
    args = parser.parse_args()

    _configure_logging()
    settings = get_settings()

    if args.no_backfill:
        # Temporarily override the setting
        settings.gdelt_backfill_on_start = False

    event_storage = EventStorage(db_path=settings.gdelt_db_path)
    graph = TemporalKnowledgeGraph()

    poller = GDELTPoller(
        event_storage=event_storage,
        graph=graph,
        poll_interval=args.interval,
    )

    asyncio.run(poller.run())


if __name__ == "__main__":
    main()
