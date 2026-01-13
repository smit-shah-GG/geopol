#!/usr/bin/env python
"""
Collect 30 days of historical GDELT event data for TKG training.

Usage:
    uv run python scripts/collect_training_data.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_collector import GDELTHistoricalCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """Collect 30 days of GDELT events."""
    logger.info("=" * 60)
    logger.info("GDELT Historical Data Collection")
    logger.info("=" * 60)

    # Initialize collector
    collector = GDELTHistoricalCollector(
        base_delay=2.0,  # 2 seconds between requests
        max_retries=3,
    )

    # Collect all QuadClasses (1-4) as per CONTEXT.md
    # All event types matter for geopolitics
    quad_classes = [1, 2, 3, 4]

    logger.info(f"Collecting last 30 days of GDELT events")
    logger.info(f"QuadClasses: {quad_classes} (all types)")
    logger.info(f"Output: {collector.output_dir}")

    start_time = datetime.now()

    try:
        df = collector.collect_last_n_days(
            n_days=30,
            quad_classes=quad_classes,
        )

        elapsed = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("Collection Complete")
        logger.info("=" * 60)
        logger.info(f"Total events: {len(df):,}")
        logger.info(f"Time elapsed: {elapsed}")

        # Show file stats
        raw_dir = collector.output_dir
        csv_files = list(raw_dir.glob("*.csv"))
        total_size = sum(f.stat().st_size for f in csv_files)
        logger.info(f"Files created: {len(csv_files)}")
        logger.info(f"Total size: {total_size / 1024 / 1024:.1f} MB")

        # Show QuadClass distribution
        if not df.empty and "QuadClass" in df.columns:
            logger.info("\nQuadClass distribution:")
            for qc, count in df["QuadClass"].value_counts().sort_index().items():
                logger.info(f"  QuadClass {qc}: {count:,} events")

    except KeyboardInterrupt:
        logger.warning("\nCollection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
