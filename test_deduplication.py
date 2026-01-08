#!/usr/bin/env python
"""
Test script for deduplication system.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.deduplication import (
    deduplicate_events,
    process_events_with_deduplication,
    calculate_duplicate_statistics
)
from src.database import EventStorage
from src.fetch_events import fetch_conflict_diplomatic_events
from src.gdelt_client import GDELTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_deduplication():
    """Test the deduplication system with real GDELT data."""

    logger.info("=" * 60)
    logger.info("GDELT Deduplication System Test")
    logger.info("=" * 60)

    # Initialize components
    logger.info("\n1. Initializing components...")
    client = GDELTClient()
    storage = EventStorage()

    # Get initial database statistics
    initial_count = storage.get_event_count()
    logger.info(f"Initial database event count: {initial_count}")

    # Fetch recent events (last 24 hours)
    logger.info("\n2. Fetching events from GDELT (last 24 hours)...")
    try:
        events_df = fetch_conflict_diplomatic_events(timespan="24h", client=client)
        logger.info(f"Fetched {len(events_df)} events from GDELT")
    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        sys.exit(1)

    if events_df.empty:
        logger.warning("No events fetched, trying longer timespan (7 days)...")
        events_df = fetch_conflict_diplomatic_events(timespan="7d", client=client)

    if events_df.empty:
        logger.error("No events available for testing")
        sys.exit(1)

    # Apply deduplication
    logger.info("\n3. Applying deduplication...")
    start_time = time.time()

    deduplicated_df, batch_duplicates, db_duplicates = process_events_with_deduplication(
        events_df, storage
    )

    processing_time = time.time() - start_time

    logger.info(f"Deduplication completed in {processing_time:.2f} seconds")
    logger.info(f"  - Original events: {len(events_df)}")
    logger.info(f"  - Batch duplicates removed: {batch_duplicates}")
    logger.info(f"  - Database duplicates found: {db_duplicates}")
    logger.info(f"  - New unique events: {len(deduplicated_df)}")

    # Insert new events into database
    if not deduplicated_df.empty:
        logger.info("\n4. Inserting new events into database...")

        # Convert DataFrame to Event objects
        from src.database.models import Event
        events_to_insert = []
        for _, row in deduplicated_df.iterrows():
            event = Event.from_gdelt_row(row.to_dict())
            event.content_hash = row.get('content_hash', '')
            event.time_window = row.get('time_window', '')
            events_to_insert.append(event)

        inserted_count = storage.insert_events(events_to_insert)
        logger.info(f"Inserted {inserted_count} new events into database")

        # Record ingestion statistics
        storage.record_ingestion_stats(
            events_fetched=len(events_df),
            events_deduplicated=batch_duplicates + db_duplicates,
            events_inserted=inserted_count,
            processing_time=processing_time
        )
    else:
        logger.info("\n4. No new events to insert (all were duplicates)")

    # Get updated statistics
    logger.info("\n5. Database statistics after insertion:")
    final_count = storage.get_event_count()
    logger.info(f"  - Total events: {final_count}")
    logger.info(f"  - Events added this run: {final_count - initial_count}")

    # Calculate duplicate statistics
    logger.info("\n6. Duplicate statistics:")
    dup_stats = calculate_duplicate_statistics(storage)
    logger.info(f"  - Total events: {dup_stats['total_events']}")
    logger.info(f"  - Duplicate groups: {dup_stats['duplicate_groups']}")
    logger.info(f"  - Total duplicates: {dup_stats['total_duplicates']}")
    logger.info(f"  - Duplicate rate: {dup_stats['duplicate_rate']:.1f}%")

    if dup_stats['most_duplicated']:
        logger.info("\n  Most duplicated events:")
        for event in dup_stats['most_duplicated']:
            logger.info(
                f"    - {event['actors']} ({event['event']}): "
                f"{event['count']} occurrences"
            )

    # Test second run to verify deduplication
    logger.info("\n7. Testing second run (should find all duplicates)...")
    logger.info("Fetching same timespan again...")

    events_df2 = fetch_conflict_diplomatic_events(timespan="24h", client=client)
    logger.info(f"Fetched {len(events_df2)} events again")

    deduplicated_df2, batch_dup2, db_dup2 = process_events_with_deduplication(
        events_df2, storage
    )

    logger.info(f"Second run results:")
    logger.info(f"  - Batch duplicates: {batch_dup2}")
    logger.info(f"  - Database duplicates: {db_dup2}")
    logger.info(f"  - New events found: {len(deduplicated_df2)}")

    if len(deduplicated_df2) == 0:
        logger.info("✓ SUCCESS: All events identified as duplicates on second run")
    else:
        logger.warning(f"⚠ Found {len(deduplicated_df2)} new events on second run")
        logger.info("This is expected if new events were published between runs")

    # Get overall database statistics
    logger.info("\n8. Overall database statistics:")
    db_stats = storage.get_statistics()
    logger.info(f"  - Total events: {db_stats['total_events']}")
    logger.info(f"  - High confidence (100+ mentions): {db_stats['high_confidence_events']}")

    if db_stats['quad_class_distribution']:
        logger.info("  - QuadClass distribution:")
        quad_names = {
            1: "Verbal Cooperation",
            2: "Material Cooperation",
            3: "Verbal Conflict",
            4: "Material Conflict"
        }
        for quad_class, count in db_stats['quad_class_distribution'].items():
            name = quad_names.get(quad_class, f"Class {quad_class}")
            percentage = (count / db_stats['total_events']) * 100 if db_stats['total_events'] > 0 else 0
            logger.info(f"      {name}: {count} ({percentage:.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("Deduplication test completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_deduplication()