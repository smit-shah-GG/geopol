"""
Deduplication system using content hashing for GDELT events.
"""

import hashlib
import logging
from typing import List, Optional, Set, Tuple
from datetime import datetime
import pandas as pd

from .database.models import Event
from .database.storage import EventStorage

logger = logging.getLogger(__name__)


def generate_content_hash(
    actor1_code: Optional[str],
    actor2_code: Optional[str],
    event_code: Optional[str],
    location: Optional[str] = None,
) -> str:
    """
    Generate content hash from event key fields.

    Args:
        actor1_code: First actor code
        actor2_code: Second actor code
        event_code: Event type code
        location: Optional location information

    Returns:
        MD5 hash string for deduplication
    """
    # Normalize None values to empty strings
    actor1 = (actor1_code or '').strip().upper()
    actor2 = (actor2_code or '').strip().upper()
    event = (event_code or '').strip().upper()
    loc = (location or '').strip().upper()

    # Create canonical representation (order matters for consistency)
    content = f"{actor1}|{actor2}|{event}|{loc}"

    # Generate MD5 hash
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return hash_obj.hexdigest()


def generate_time_window(timestamp: str) -> str:
    """
    Generate time window identifier by flooring to nearest hour.

    Args:
        timestamp: Timestamp string (ISO format, GDELT format, or YYYY-MM-DD)

    Returns:
        Hour-based time window string (YYYY-MM-DD-HH)
    """
    try:
        # Handle different timestamp formats
        if 'T' in timestamp:
            # Check for GDELT format (YYYYMMDDTHHMMSSZ or truncated versions)
            if 'Z' in timestamp:
                # Full GDELT format: 20260108T034500Z
                clean_ts = timestamp.replace('Z', '')
                if len(clean_ts) >= 11:  # At least YYYYMMDDTHH
                    year = clean_ts[:4]
                    month = clean_ts[4:6]
                    day = clean_ts[6:8]
                    hour = clean_ts[9:11] if len(clean_ts) > 10 else '00'
                    # Pad hour if single digit
                    if len(hour) == 1:
                        hour = '0' + hour
                    return f"{year}-{month}-{day}-{hour}"
            elif timestamp.count('T') == 1 and len(timestamp.split('T')[0]) == 8:
                # GDELT format without Z: 20260108T0 or 20260108T03
                date_part = timestamp.split('T')[0]
                time_part = timestamp.split('T')[1]
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                # Handle single digit hour
                hour = time_part[:2] if len(time_part) >= 2 else time_part.zfill(2)
                return f"{year}-{month}-{day}-{hour}"
            else:
                # Standard ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d-%H')
        elif len(timestamp) == 10:
            # Date only (YYYY-MM-DD)
            dt = datetime.strptime(timestamp, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d-00')
        else:
            # Try parsing as datetime
            dt = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d-%H')

    except Exception as e:
        logger.debug(f"Failed to parse timestamp '{timestamp}': {e}")
        # Fallback to date only
        if len(timestamp) >= 10:
            return timestamp[:10] + "-00"
        return timestamp


def deduplicate_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate events based on content hash and time window.

    Args:
        events_df: DataFrame containing GDELT events

    Returns:
        DataFrame with duplicates removed
    """
    if events_df.empty:
        return events_df

    original_count = len(events_df)

    # Generate content hashes for each event
    events_df['content_hash'] = events_df.apply(
        lambda row: generate_content_hash(
            row.get('Actor1Code') or row.get('actor1_code'),
            row.get('Actor2Code') or row.get('actor2_code'),
            row.get('EventCode') or row.get('event_code'),
            row.get('ActionGeo') or row.get('location')
        ),
        axis=1
    )

    # Generate time windows
    date_column = 'seendate' if 'seendate' in events_df.columns else 'event_date'
    if date_column in events_df.columns:
        events_df['time_window'] = events_df[date_column].apply(generate_time_window)
    else:
        # Use current date as fallback
        events_df['time_window'] = generate_time_window(datetime.now().isoformat())

    # Remove duplicates based on content_hash + time_window
    # Keep first occurrence (typically the earliest report)
    deduplicated_df = events_df.drop_duplicates(
        subset=['content_hash', 'time_window'],
        keep='first'
    ).copy()

    duplicate_count = original_count - len(deduplicated_df)
    duplicate_percentage = (duplicate_count / original_count) * 100 if original_count > 0 else 0

    logger.info(
        f"Deduplication: {original_count} → {len(deduplicated_df)} events "
        f"({duplicate_count} duplicates removed, {duplicate_percentage:.1f}%)"
    )

    # Log duplicate statistics by time window
    if duplicate_count > 0:
        duplicates_by_window = (
            events_df.groupby('time_window')
            .size()
            .subtract(deduplicated_df.groupby('time_window').size(), fill_value=0)
            .sort_values(ascending=False)
        )

        top_windows = duplicates_by_window.head(5)
        if not top_windows.empty:
            logger.debug("Top time windows with duplicates:")
            for window, count in top_windows.items():
                if count > 0:
                    logger.debug(f"  {window}: {int(count)} duplicates")

    return deduplicated_df


def is_duplicate(event: Event, storage: EventStorage) -> bool:
    """
    Check if an event already exists in the database.

    Args:
        event: Event to check
        storage: EventStorage instance for database access

    Returns:
        True if event is a duplicate, False otherwise
    """
    # Ensure event has content hash and time window
    if not event.content_hash:
        event.content_hash = generate_content_hash(
            event.actor1_code,
            event.actor2_code,
            event.event_code,
            None  # Location not available in Event model
        )

    if not event.time_window:
        event.time_window = generate_time_window(event.event_date)

    # Query database for existing event with same hash and time window
    with storage.db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) as count
            FROM events
            WHERE content_hash = ? AND time_window = ?
            """,
            (event.content_hash, event.time_window)
        )
        result = cursor.fetchone()
        return result['count'] > 0 if result else False


def process_events_with_deduplication(
    events_df: pd.DataFrame,
    storage: Optional[EventStorage] = None,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Process events with deduplication against database.

    Args:
        events_df: DataFrame with events to process
        storage: Optional EventStorage instance (creates new if not provided)

    Returns:
        Tuple of (deduplicated_df, duplicates_removed, database_duplicates)
    """
    if storage is None:
        storage = EventStorage()

    # First pass: Remove duplicates within the batch
    deduplicated_df = deduplicate_events(events_df)
    batch_duplicates = len(events_df) - len(deduplicated_df)

    # Second pass: Check against database
    database_duplicates = 0
    events_to_insert = []

    for _, row in deduplicated_df.iterrows():
        # Convert row to Event
        event = Event.from_gdelt_row(row.to_dict())

        # Add content hash and time window
        event.content_hash = row.get('content_hash', generate_content_hash(
            event.actor1_code,
            event.actor2_code,
            event.event_code,
            None
        ))
        event.time_window = row.get('time_window', generate_time_window(event.event_date))

        # Check if duplicate in database
        if not is_duplicate(event, storage):
            events_to_insert.append(event)
        else:
            database_duplicates += 1

    logger.info(
        f"Deduplication complete: "
        f"Batch duplicates: {batch_duplicates}, "
        f"Database duplicates: {database_duplicates}, "
        f"New events: {len(events_to_insert)}"
    )

    # Convert events back to DataFrame for return
    if events_to_insert:
        new_events_df = pd.DataFrame([e.to_dict() for e in events_to_insert])
    else:
        new_events_df = pd.DataFrame()

    return new_events_df, batch_duplicates, database_duplicates


def calculate_duplicate_statistics(storage: EventStorage) -> dict:
    """
    Calculate comprehensive duplicate statistics from database.

    Args:
        storage: EventStorage instance

    Returns:
        Dictionary with duplicate statistics
    """
    with storage.db.get_connection() as conn:
        cursor = conn.cursor()

        stats = {}

        # Total events
        cursor.execute("SELECT COUNT(*) as count FROM events")
        stats['total_events'] = cursor.fetchone()['count']

        # Events with duplicate content hashes
        cursor.execute("""
            SELECT COUNT(*) as duplicate_groups, SUM(count - 1) as total_duplicates
            FROM (
                SELECT content_hash, COUNT(*) as count
                FROM events
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            )
        """)
        result = cursor.fetchone()
        stats['duplicate_groups'] = result['duplicate_groups'] or 0
        stats['total_duplicates'] = result['total_duplicates'] or 0

        # Duplicate rate
        if stats['total_events'] > 0:
            stats['duplicate_rate'] = (stats['total_duplicates'] / stats['total_events']) * 100
        else:
            stats['duplicate_rate'] = 0.0

        # Most duplicated events
        cursor.execute("""
            SELECT content_hash, COUNT(*) as count,
                   MIN(actor1_code) as actor1, MIN(actor2_code) as actor2,
                   MIN(event_code) as event_code
            FROM events
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            ORDER BY count DESC
            LIMIT 5
        """)

        stats['most_duplicated'] = [
            {
                'hash': row['content_hash'][:8],  # Shortened hash for display
                'count': row['count'],
                'actors': f"{row['actor1'] or 'N/A'} - {row['actor2'] or 'N/A'}",
                'event': row['event_code'] or 'N/A'
            }
            for row in cursor.fetchall()
        ]

        return stats