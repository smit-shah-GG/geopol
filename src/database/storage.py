"""
Event storage layer with batch operations and efficient querying.
"""

import sqlite3
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .connection import DatabaseConnection
from .models import Event

logger = logging.getLogger(__name__)


class EventStorage:
    """Handles storage and retrieval of GDELT events with batch operations."""

    def __init__(self, db_path: str = "data/events.db"):
        """
        Initialize event storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db = DatabaseConnection(db_path)
        self.db_path = Path(db_path)

        # Initialize database if needed
        self.init_database()

    def init_database(self):
        """Create database tables if they don't exist."""
        if not self.db.table_exists('events'):
            schema_path = Path(__file__).parent / 'schema.sql'
            if schema_path.exists():
                self.db.execute_script(str(schema_path))
                logger.info("Database initialized with schema")
            else:
                logger.error(f"Schema file not found: {schema_path}")
                raise FileNotFoundError(f"Database schema not found at {schema_path}")

    def insert_events(self, events: List[Event], batch_size: int = 1000) -> int:
        """
        Insert events in batches with proper error handling.

        Args:
            events: List of Event objects to insert
            batch_size: Number of events to insert per batch

        Returns:
            Number of events successfully inserted
        """
        if not events:
            return 0

        inserted_count = 0
        failed_count = 0

        # Process events in batches
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]

            with self.db.transaction() as conn:
                cursor = conn.cursor()

                for event in batch:
                    try:
                        # Prepare event data
                        data = event.to_dict()

                        # Remove id if it's None (let database auto-increment)
                        if 'id' in data and data['id'] is None:
                            del data['id']

                        # Build INSERT statement with placeholders
                        columns = list(data.keys())
                        placeholders = ['?' for _ in columns]

                        sql = f"""
                            INSERT OR IGNORE INTO events ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                        """

                        cursor.execute(sql, list(data.values()))

                        if cursor.rowcount > 0:
                            inserted_count += 1
                        else:
                            logger.debug(f"Event already exists (duplicate): {event.gdelt_id}")

                    except sqlite3.IntegrityError as e:
                        logger.debug(f"Duplicate event skipped: {e}")
                        failed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to insert event: {e}")
                        failed_count += 1

                logger.info(
                    f"Batch {i // batch_size + 1}: Inserted {inserted_count} events "
                    f"({failed_count} duplicates/failures)"
                )

        return inserted_count

    def get_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        quad_classes: Optional[List[int]] = None,
        min_mentions: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Retrieve events with filtering options.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            quad_classes: List of QuadClass values to filter (1-4)
            min_mentions: Minimum number of mentions
            limit: Maximum number of events to return

        Returns:
            List of Event objects
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Build query with filters
            sql = "SELECT * FROM events WHERE 1=1"
            params = []

            if start_date:
                sql += " AND event_date >= ?"
                params.append(start_date)

            if end_date:
                sql += " AND event_date <= ?"
                params.append(end_date)

            if quad_classes:
                placeholders = ','.join(['?' for _ in quad_classes])
                sql += f" AND quad_class IN ({placeholders})"
                params.extend(quad_classes)

            if min_mentions:
                sql += " AND num_mentions >= ?"
                params.append(min_mentions)

            sql += " ORDER BY event_date DESC, num_mentions DESC"

            if limit:
                sql += " LIMIT ?"
                params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            # Convert rows to Event objects
            events = []
            for row in rows:
                # Convert Row to dict
                row_dict = dict(row)
                events.append(Event(**row_dict))

            logger.info(f"Retrieved {len(events)} events from database")
            return events

    def get_event_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Get count of events in database with optional filters.

        Args:
            filters: Optional dictionary of filters

        Returns:
            Number of events matching filters
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            sql = "SELECT COUNT(*) as count FROM events WHERE 1=1"
            params = []

            if filters:
                if 'start_date' in filters:
                    sql += " AND event_date >= ?"
                    params.append(filters['start_date'])

                if 'end_date' in filters:
                    sql += " AND event_date <= ?"
                    params.append(filters['end_date'])

                if 'quad_class' in filters:
                    sql += " AND quad_class = ?"
                    params.append(filters['quad_class'])

                if 'min_mentions' in filters:
                    sql += " AND num_mentions >= ?"
                    params.append(filters['min_mentions'])

            cursor.execute(sql, params)
            result = cursor.fetchone()
            return result['count'] if result else 0

    def get_duplicate_count(self, time_window: str) -> int:
        """
        Get count of duplicate events in a time window.

        Args:
            time_window: Time window identifier

        Returns:
            Number of duplicates found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            sql = """
                SELECT content_hash, COUNT(*) as count
                FROM events
                WHERE time_window = ?
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            """

            cursor.execute(sql, (time_window,))
            duplicates = cursor.fetchall()

            total_duplicates = sum(row['count'] - 1 for row in duplicates)
            return total_duplicates

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with various statistics
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total events
            cursor.execute("SELECT COUNT(*) as count FROM events")
            stats['total_events'] = cursor.fetchone()['count']

            # Events by QuadClass
            cursor.execute("""
                SELECT quad_class, COUNT(*) as count
                FROM events
                WHERE quad_class IS NOT NULL
                GROUP BY quad_class
            """)
            stats['quad_class_distribution'] = {
                row['quad_class']: row['count']
                for row in cursor.fetchall()
            }

            # High confidence events (GDELT100)
            cursor.execute("SELECT COUNT(*) as count FROM events WHERE num_mentions >= 100")
            stats['high_confidence_events'] = cursor.fetchone()['count']

            # Date range
            cursor.execute("SELECT MIN(event_date) as min_date, MAX(event_date) as max_date FROM events")
            result = cursor.fetchone()
            stats['date_range'] = {
                'earliest': result['min_date'],
                'latest': result['max_date']
            }

            # Average tone by QuadClass
            cursor.execute("""
                SELECT quad_class, AVG(tone) as avg_tone
                FROM events
                WHERE quad_class IS NOT NULL AND tone IS NOT NULL
                GROUP BY quad_class
            """)
            stats['avg_tone_by_class'] = {
                row['quad_class']: round(row['avg_tone'], 2)
                for row in cursor.fetchall()
            }

            return stats

    def clear_old_events(self, days_to_keep: int = 90) -> int:
        """
        Remove events older than specified number of days.

        Args:
            days_to_keep: Number of days of events to keep

        Returns:
            Number of events deleted
        """
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE event_date < ?", (cutoff_date,))
            deleted_count = cursor.rowcount

            logger.info(f"Deleted {deleted_count} events older than {cutoff_date}")
            return deleted_count

    def record_ingestion_stats(
        self,
        events_fetched: int,
        events_deduplicated: int,
        events_inserted: int,
        processing_time: float
    ):
        """
        Record statistics about an ingestion run.

        Args:
            events_fetched: Number of events fetched from API
            events_deduplicated: Number of duplicate events removed
            events_inserted: Number of new events inserted
            processing_time: Time taken in seconds
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ingestion_stats
                (events_fetched, events_deduplicated, events_inserted, processing_time_seconds)
                VALUES (?, ?, ?, ?)
            """, (events_fetched, events_deduplicated, events_inserted, processing_time))

            logger.info(
                f"Recorded ingestion stats: fetched={events_fetched}, "
                f"deduped={events_deduplicated}, inserted={events_inserted}, "
                f"time={processing_time:.1f}s"
            )