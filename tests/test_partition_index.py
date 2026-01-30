"""
Tests for EntityPartitionIndex SQLite persistence.

Verifies:
- Schema creation on fresh database
- Partition registration with entities
- Entity-to-partition lookup ordering
- Time range filtering
- Multi-partition entity handling
- Persistence across process restart (close/reopen)
"""

import tempfile
from pathlib import Path

import pytest

from src.knowledge_graph.partition_index import EntityPartitionIndex


class TestEntityPartitionIndex:
    """Test suite for EntityPartitionIndex."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_index.db"

    def test_schema_creation(self, temp_db):
        """Fresh database has correct tables and indexes."""
        index = EntityPartitionIndex(temp_db)

        # Check tables exist
        cursor = index.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        assert "entity_partitions" in tables
        assert "partition_meta" in tables

        # Check indexes exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index'
            ORDER BY name
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_entity_id" in indexes
        assert "idx_partition_id" in indexes

        index.close()

    def test_register_partition(self, temp_db):
        """Partition metadata and entities stored correctly."""
        index = EntityPartitionIndex(temp_db)

        entities = {"entity_a", "entity_b", "entity_c"}
        time_window = ("2024-01-01T00:00:00", "2024-01-31T23:59:59")
        stats = {"nodes": 3, "edges": 5}
        file_path = "/data/partitions/2024-01/graph.graphml"

        index.register_partition(
            partition_id="2024-01",
            entities=entities,
            time_window=time_window,
            stats=stats,
            file_path=file_path
        )

        # Verify partition metadata
        meta = index.get_partition_meta("2024-01")
        assert meta is not None
        assert meta["partition_id"] == "2024-01"
        assert meta["time_window_start"] == "2024-01-01T00:00:00"
        assert meta["time_window_end"] == "2024-01-31T23:59:59"
        assert meta["node_count"] == 3
        assert meta["edge_count"] == 5
        assert meta["file_path"] == file_path

        # Verify entities registered
        for entity_id in entities:
            partitions = index.get_entity_partitions(entity_id)
            assert "2024-01" in partitions

        index.close()

    def test_get_entity_partitions_ordering(self, temp_db):
        """Returns partitions in correct order: is_home DESC, edge_count DESC."""
        index = EntityPartitionIndex(temp_db)

        # Register entity in two partitions
        index.register_partition(
            partition_id="part_a",
            entities={"entity_x"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 10},
            file_path="/a.graphml"
        )
        index.register_partition(
            partition_id="part_b",
            entities={"entity_x"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 20},
            file_path="/b.graphml"
        )

        # Update edge counts
        index.update_entity_edge_count("entity_x", "part_a", 5)
        index.update_entity_edge_count("entity_x", "part_b", 15)

        # Both are home (is_home=1), so order by edge_count DESC
        partitions = index.get_entity_partitions("entity_x")
        assert partitions == ["part_b", "part_a"]

        # Mark part_a as home, part_b as not home
        index.mark_home_partition("entity_x", "part_a")

        # Now part_a comes first (is_home=1) despite lower edge_count
        partitions = index.get_entity_partitions("entity_x")
        assert partitions == ["part_a", "part_b"]

        index.close()

    def test_get_partitions_in_time_range(self, temp_db):
        """Time range filtering returns overlapping partitions."""
        index = EntityPartitionIndex(temp_db)

        # Create partitions for Jan, Feb, Mar
        index.register_partition(
            partition_id="2024-01",
            entities={"e1"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 1},
            file_path="/jan.graphml"
        )
        index.register_partition(
            partition_id="2024-02",
            entities={"e2"},
            time_window=("2024-02-01", "2024-02-29"),
            stats={"nodes": 1, "edges": 1},
            file_path="/feb.graphml"
        )
        index.register_partition(
            partition_id="2024-03",
            entities={"e3"},
            time_window=("2024-03-01", "2024-03-31"),
            stats={"nodes": 1, "edges": 1},
            file_path="/mar.graphml"
        )

        # Query for Feb 15 - Mar 15 (should get Feb and Mar)
        partitions = index.get_partitions_in_time_range(
            time_start="2024-02-15",
            time_end="2024-03-15"
        )
        assert "2024-01" not in partitions
        assert "2024-02" in partitions
        assert "2024-03" in partitions

        # Query for Jan 15 - Jan 20 (should get only Jan)
        partitions = index.get_partitions_in_time_range(
            time_start="2024-01-15",
            time_end="2024-01-20"
        )
        assert partitions == ["2024-01"]

        # Query for entire period (should get all three)
        partitions = index.get_partitions_in_time_range(
            time_start="2024-01-01",
            time_end="2024-03-31"
        )
        assert len(partitions) == 3

        index.close()

    def test_entity_appears_in_multiple_partitions(self, temp_db):
        """Entity correctly mapped to multiple partitions."""
        index = EntityPartitionIndex(temp_db)

        # Same entity in three different partitions
        index.register_partition(
            partition_id="p1",
            entities={"shared_entity", "unique_1"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 2, "edges": 3},
            file_path="/p1.graphml"
        )
        index.register_partition(
            partition_id="p2",
            entities={"shared_entity", "unique_2"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 2, "edges": 5},
            file_path="/p2.graphml"
        )
        index.register_partition(
            partition_id="p3",
            entities={"shared_entity", "unique_3"},
            time_window=("2024-03-01", "2024-03-31"),
            stats={"nodes": 2, "edges": 2},
            file_path="/p3.graphml"
        )

        # Shared entity appears in all three partitions
        partitions = index.get_entity_partitions("shared_entity")
        assert len(partitions) == 3
        assert set(partitions) == {"p1", "p2", "p3"}

        # Unique entities appear in only one partition
        assert index.get_entity_partitions("unique_1") == ["p1"]
        assert index.get_entity_partitions("unique_2") == ["p2"]
        assert index.get_entity_partitions("unique_3") == ["p3"]

        # Non-existent entity returns empty list
        assert index.get_entity_partitions("nonexistent") == []

        index.close()

    def test_persistence_across_restart(self, temp_db):
        """Data survives close and reopen of database."""
        # First session: create and populate
        index1 = EntityPartitionIndex(temp_db)
        index1.register_partition(
            partition_id="persistent_partition",
            entities={"persistent_entity"},
            time_window=("2024-01-01", "2024-12-31"),
            stats={"nodes": 100, "edges": 500},
            file_path="/persistent.graphml"
        )
        index1.update_entity_edge_count("persistent_entity", "persistent_partition", 42)
        index1.close()

        # Second session: verify data persists
        index2 = EntityPartitionIndex(temp_db)

        # Partition metadata persists
        meta = index2.get_partition_meta("persistent_partition")
        assert meta is not None
        assert meta["node_count"] == 100
        assert meta["edge_count"] == 500
        assert meta["file_path"] == "/persistent.graphml"

        # Entity mapping persists
        partitions = index2.get_entity_partitions("persistent_entity")
        assert "persistent_partition" in partitions

        # Edge count update persists
        cursor = index2.conn.cursor()
        cursor.execute("""
            SELECT edge_count FROM entity_partitions
            WHERE entity_id = ? AND partition_id = ?
        """, ("persistent_entity", "persistent_partition"))
        row = cursor.fetchone()
        assert row[0] == 42

        index2.close()

    def test_idempotent_registration(self, temp_db):
        """Re-registering partition doesn't create duplicates."""
        index = EntityPartitionIndex(temp_db)

        # Register same partition twice
        for _ in range(2):
            index.register_partition(
                partition_id="idempotent_test",
                entities={"e1", "e2"},
                time_window=("2024-01-01", "2024-01-31"),
                stats={"nodes": 2, "edges": 3},
                file_path="/idem.graphml"
            )

        # Should only have one partition entry
        all_partitions = index.list_all_partitions()
        assert all_partitions.count("idempotent_test") == 1

        # Entity should only appear once per partition
        cursor = index.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM entity_partitions
            WHERE entity_id = 'e1' AND partition_id = 'idempotent_test'
        """)
        assert cursor.fetchone()[0] == 1

        index.close()

    def test_get_partition_meta_not_found(self, temp_db):
        """Non-existent partition returns None."""
        index = EntityPartitionIndex(temp_db)
        assert index.get_partition_meta("nonexistent") is None
        index.close()

    def test_get_entity_count(self, temp_db):
        """Entity count per partition is accurate."""
        index = EntityPartitionIndex(temp_db)

        index.register_partition(
            partition_id="counting_test",
            entities={"e1", "e2", "e3", "e4", "e5"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 5, "edges": 10},
            file_path="/count.graphml"
        )

        assert index.get_entity_count("counting_test") == 5
        assert index.get_entity_count("nonexistent") == 0

        index.close()

    def test_context_manager(self, temp_db):
        """Context manager properly closes connection."""
        with EntityPartitionIndex(temp_db) as index:
            index.register_partition(
                partition_id="ctx_test",
                entities={"e1"},
                time_window=("2024-01-01", "2024-01-31"),
                stats={"nodes": 1, "edges": 1},
                file_path="/ctx.graphml"
            )

        # After context exit, connection should be closed
        # Verify by opening new connection and reading data
        with EntityPartitionIndex(temp_db) as index2:
            meta = index2.get_partition_meta("ctx_test")
            assert meta is not None
