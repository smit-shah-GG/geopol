#!/usr/bin/env python
"""
Test script for GDELT event fetching with filtering.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.fetch_events import fetch_conflict_diplomatic_events, get_event_statistics
from src.gdelt_client import GDELTClient
from src.config import DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test event fetching functionality."""

    # Use timespan for more reliable recent data fetching
    timespan = "24h"

    logger.info("=" * 60)
    logger.info(f"GDELT Event Fetching Test - Last {timespan}")
    logger.info("=" * 60)

    # Initialize client
    logger.info("\n1. Initializing GDELT client...")
    client = GDELTClient()

    # Test connection
    logger.info("\n2. Testing connection to GDELT API...")
    if not client.test_connection():
        logger.error("Failed to connect to GDELT API")
        sys.exit(1)
    logger.info("✓ Connection successful")

    # Fetch events for the timespan
    logger.info(f"\n3. Fetching conflict and diplomatic events for last {timespan}...")
    try:
        events_df = fetch_conflict_diplomatic_events(timespan=timespan, client=client)

        if events_df.empty:
            logger.warning(f"No events found for {timespan}")
            # Try a longer timespan if no events
            longer_timespan = "7d"
            logger.info(f"\n4. Trying longer timespan ({longer_timespan})...")
            events_df = fetch_conflict_diplomatic_events(timespan=longer_timespan, client=client)

    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        sys.exit(1)

    if events_df.empty:
        logger.warning("No events found")
        sys.exit(0)

    # Display statistics
    logger.info("\n4. Event Statistics:")
    logger.info("-" * 40)

    stats = get_event_statistics(events_df)

    logger.info(f"Total events retrieved: {stats['total_events']}")
    logger.info(f"Unique URLs: {stats.get('unique_urls', 'N/A')}")
    logger.info(f"Unique domains: {stats.get('unique_domains', 'N/A')}")

    if 'event_types' in stats:
        logger.info("\nEvent type distribution:")
        for event_type, count in stats['event_types'].items():
            percentage = (count / stats['total_events']) * 100
            logger.info(f"  - {event_type}: {count} ({percentage:.1f}%)")

    if 'tone_stats' in stats:
        tone = stats['tone_stats']
        logger.info("\nTone statistics:")
        logger.info(f"  - Mean: {tone['mean']:.2f}")
        logger.info(f"  - Std Dev: {tone['std']:.2f}")
        logger.info(f"  - Range: [{tone['min']:.2f}, {tone['max']:.2f}]")

    if 'quad_classes' in stats:
        logger.info("\nQuadClass distribution:")
        quad_names = {
            1: "Verbal Cooperation",
            2: "Material Cooperation",
            3: "Verbal Conflict",
            4: "Material Conflict"
        }
        for quad_class, count in stats['quad_classes'].items():
            name = quad_names.get(quad_class, f"Class {quad_class}")
            percentage = (count / stats['total_events']) * 100
            logger.info(f"  - {name}: {count} ({percentage:.1f}%)")

    # Display sample events
    logger.info("\n5. Sample Events (first 5):")
    logger.info("-" * 40)

    sample_cols = ['title', 'domain', 'tone', 'event_type'] if 'event_type' in events_df.columns else ['title', 'domain', 'tone']

    for col in sample_cols:
        if col not in events_df.columns:
            sample_cols.remove(col)

    if sample_cols:
        for idx, row in events_df.head(5).iterrows():
            logger.info(f"\nEvent {idx + 1}:")
            for col in sample_cols:
                if col == 'title':
                    # Truncate long titles
                    title = str(row[col])[:100]
                    if len(str(row[col])) > 100:
                        title += "..."
                    logger.info(f"  Title: {title}")
                elif col == 'tone':
                    logger.info(f"  Tone: {row[col]:.2f}")
                else:
                    logger.info(f"  {col.capitalize()}: {row[col]}")

    # Save sample to CSV
    output_path = DATA_DIR / "sample_events.csv"
    logger.info(f"\n6. Saving sample events to {output_path}...")

    try:
        events_df.to_csv(output_path, index=False)
        logger.info(f"✓ Sample events saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Test completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()