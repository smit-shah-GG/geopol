#!/usr/bin/env python
"""
Entry point for the geopol RSS ingestion daemon.

Usage:
    uv run python scripts/rss_daemon.py
    uv run python scripts/rss_daemon.py --tier1-interval 300 --tier2-interval 1800
    uv run python scripts/rss_daemon.py --chroma-dir /data/chroma_db
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geopol RSS feed ingestion daemon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier1-interval",
        type=int,
        default=900,
        help="Tier-1 poll interval in seconds (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--tier2-interval",
        type=int,
        default=3600,
        help="Tier-2 poll interval in seconds (default: 3600 = 60 min)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent feed/article fetches",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="./chroma_db",
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=90,
        help="Article retention period in days",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    from src.ingest.rss_daemon import DaemonConfig, RSSDaemon

    config = DaemonConfig(
        tier1_interval=args.tier1_interval,
        tier2_interval=args.tier2_interval,
        max_concurrent_fetches=args.max_concurrent,
        chroma_persist_dir=args.chroma_dir,
        retention_days=args.retention_days,
    )

    daemon = RSSDaemon(config=config)

    logging.getLogger(__name__).info(
        "Starting RSS daemon (tier1=%ds, tier2=%ds, concurrency=%d)",
        config.tier1_interval,
        config.tier2_interval,
        config.max_concurrent_fetches,
    )

    asyncio.run(daemon.start())


if __name__ == "__main__":
    main()
