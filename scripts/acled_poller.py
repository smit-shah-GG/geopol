#!/usr/bin/env python
"""
Entry point for the ACLED armed conflict event polling daemon.

Usage:
    uv run python scripts/acled_poller.py
    uv run python scripts/acled_poller.py --interval 86400

Requires ACLED_EMAIL and ACLED_PASSWORD environment variables (or in .env).
Designed to run under systemd (geopol-acled-poller.service) with
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
from src.ingest.acled_poller import ACLEDPoller
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
        description="ACLED armed conflict event polling daemon"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Poll interval in seconds (default: from settings)",
    )
    args = parser.parse_args()

    _configure_logging()
    settings = get_settings()

    event_storage = EventStorage(db_path=settings.gdelt_db_path)

    poller = ACLEDPoller(
        event_storage=event_storage,
        poll_interval=args.interval,
    )

    asyncio.run(poller.run())


if __name__ == "__main__":
    main()
