"""
SQLite-backed partition index for entity-to-partition mapping.

This module provides O(1) entity-to-partition lookups persisted in SQLite,
enabling efficient routing of queries to relevant partitions.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class EntityPartitionIndex:
    """
    SQLite-backed index mapping entities to their containing partitions.

    Provides:
    - Entity-to-partition lookup for query routing
    - Partition metadata storage (time windows, file paths, stats)
    - Home partition tracking for boundary entity resolution
    - Persistence across process restarts

    Schema:
    - entity_partitions: (entity_id, partition_id, is_home, edge_count)
    - partition_meta: (partition_id, time_window_start, time_window_end, ...)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entity_partitions (
        entity_id TEXT NOT NULL,
        partition_id TEXT NOT NULL,
        is_home INTEGER DEFAULT 1,
        edge_count INTEGER DEFAULT 0,
        PRIMARY KEY (entity_id, partition_id)
    );

    CREATE INDEX IF NOT EXISTS idx_entity_id
        ON entity_partitions(entity_id);

    CREATE INDEX IF NOT EXISTS idx_partition_id
        ON entity_partitions(partition_id);

    CREATE TABLE IF NOT EXISTS partition_meta (
        partition_id TEXT PRIMARY KEY,
        time_window_start TEXT,
        time_window_end TEXT,
        node_count INTEGER,
        edge_count INTEGER,
        file_path TEXT,
        created_at TEXT
    );
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize partition index with SQLite database.

        Args:
            db_path: Path to SQLite database file. Created if not exists.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def register_partition(
        self,
        partition_id: str,
        entities: Set[str],
        time_window: Tuple[str, str],
        stats: Dict[str, int],
        file_path: str
    ) -> None:
        """
        Register a partition and its entities in the index.

        Args:
            partition_id: Unique identifier for the partition
            entities: Set of entity IDs contained in the partition
            time_window: Tuple of (start, end) ISO timestamps
            stats: Dict with 'nodes' and 'edges' counts
            file_path: Path to the partition GraphML file
        """
        cursor = self.conn.cursor()

        # Insert partition metadata (idempotent)
        cursor.execute("""
            INSERT OR REPLACE INTO partition_meta
            (partition_id, time_window_start, time_window_end,
             node_count, edge_count, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            partition_id,
            time_window[0],
            time_window[1],
            stats.get('nodes', 0),
            stats.get('edges', 0),
            file_path,
            datetime.now(timezone.utc).isoformat()
        ))

        # Bulk insert entity mappings (idempotent via INSERT OR IGNORE)
        entity_rows = [
            (entity_id, partition_id, 1, 0)
            for entity_id in entities
        ]
        cursor.executemany("""
            INSERT OR IGNORE INTO entity_partitions
            (entity_id, partition_id, is_home, edge_count)
            VALUES (?, ?, ?, ?)
        """, entity_rows)

        self.conn.commit()

    def get_entity_partitions(self, entity_id: str) -> List[str]:
        """
        Get all partitions containing an entity.

        Returns partitions ordered by:
        1. is_home DESC (home partition first)
        2. edge_count DESC (partitions with more edges for this entity ranked higher)

        Args:
            entity_id: Entity to look up

        Returns:
            List of partition IDs containing the entity
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT partition_id FROM entity_partitions
            WHERE entity_id = ?
            ORDER BY is_home DESC, edge_count DESC
        """, (entity_id,))
        return [row[0] for row in cursor.fetchall()]

    def get_partitions_in_time_range(
        self,
        time_start: str,
        time_end: str
    ) -> List[str]:
        """
        Get partitions whose time window overlaps the given range.

        A partition overlaps if:
        - partition.start <= query.end AND partition.end >= query.start

        Args:
            time_start: ISO timestamp for range start
            time_end: ISO timestamp for range end

        Returns:
            List of partition IDs overlapping the time range
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT partition_id FROM partition_meta
            WHERE time_window_start <= ? AND time_window_end >= ?
            ORDER BY time_window_start
        """, (time_end, time_start))
        return [row[0] for row in cursor.fetchall()]

    def get_partition_meta(self, partition_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific partition.

        Args:
            partition_id: Partition to look up

        Returns:
            Dict with partition metadata, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT partition_id, time_window_start, time_window_end,
                   node_count, edge_count, file_path, created_at
            FROM partition_meta
            WHERE partition_id = ?
        """, (partition_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            'partition_id': row['partition_id'],
            'time_window_start': row['time_window_start'],
            'time_window_end': row['time_window_end'],
            'node_count': row['node_count'],
            'edge_count': row['edge_count'],
            'file_path': row['file_path'],
            'created_at': row['created_at']
        }

    def update_entity_edge_count(
        self,
        entity_id: str,
        partition_id: str,
        edge_count: int
    ) -> None:
        """
        Update the edge count for an entity in a specific partition.

        Used to track entity density per partition for query routing optimization.

        Args:
            entity_id: Entity to update
            partition_id: Partition containing the entity
            edge_count: New edge count value
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE entity_partitions
            SET edge_count = ?
            WHERE entity_id = ? AND partition_id = ?
        """, (edge_count, entity_id, partition_id))
        self.conn.commit()

    def mark_home_partition(
        self,
        entity_id: str,
        partition_id: str
    ) -> None:
        """
        Mark a partition as the home partition for an entity.

        Sets is_home=1 for the specified partition and is_home=0 for all other
        partitions containing this entity. The home partition is where the
        entity's canonical data resides.

        Args:
            entity_id: Entity to update
            partition_id: Partition to mark as home
        """
        cursor = self.conn.cursor()

        # Clear all home flags for this entity
        cursor.execute("""
            UPDATE entity_partitions
            SET is_home = 0
            WHERE entity_id = ?
        """, (entity_id,))

        # Set home flag for specified partition
        cursor.execute("""
            UPDATE entity_partitions
            SET is_home = 1
            WHERE entity_id = ? AND partition_id = ?
        """, (entity_id, partition_id))

        self.conn.commit()

    def list_all_partitions(self) -> List[str]:
        """
        List all registered partition IDs.

        Returns:
            List of all partition IDs in the index
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT partition_id FROM partition_meta ORDER BY partition_id")
        return [row[0] for row in cursor.fetchall()]

    def get_entity_count(self, partition_id: str) -> int:
        """
        Get the number of unique entities in a partition.

        Args:
            partition_id: Partition to count entities for

        Returns:
            Number of entities in the partition
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT entity_id) FROM entity_partitions
            WHERE partition_id = ?
        """, (partition_id,))
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> 'EntityPartitionIndex':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
