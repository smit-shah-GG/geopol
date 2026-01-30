"""
Tests for PartitionManager temporal partitioning and LRU caching.

Verifies:
- Temporal partitioning by time windows
- GraphML persistence
- Entity indexing during partition
- LRU cache behavior
- Partition roundtrip integrity
- Edge cases (empty graph, no-timestamp edges)
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

from src.knowledge_graph.partition_index import EntityPartitionIndex
from src.knowledge_graph.partition_manager import PartitionManager


class TestPartitionManager:
    """Test suite for PartitionManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def index(self, temp_dir):
        """Create a fresh EntityPartitionIndex."""
        db_path = temp_dir / "index.db"
        idx = EntityPartitionIndex(db_path)
        yield idx
        idx.close()

    @pytest.fixture
    def manager(self, temp_dir, index):
        """Create a PartitionManager with test settings."""
        partition_dir = temp_dir / "partitions"
        return PartitionManager(
            partition_dir=partition_dir,
            index=index,
            max_cached_partitions=4,
            max_memory_mb=1024
        )

    def _create_test_graph(self, months: int = 3) -> nx.MultiDiGraph:
        """
        Create a test graph with edges spanning multiple months.

        Args:
            months: Number of months to span.

        Returns:
            MultiDiGraph with timestamped edges.
        """
        graph = nx.MultiDiGraph()
        base_date = datetime(2024, 1, 1)

        for month in range(months):
            # Create edges in each month
            for day in [5, 15, 25]:
                ts = base_date + timedelta(days=month * 30 + day)
                u = f"entity_{month}_a"
                v = f"entity_{month}_b"

                # Add nodes with attributes
                if u not in graph:
                    graph.add_node(u, type="actor", label=u)
                if v not in graph:
                    graph.add_node(v, type="actor", label=v)

                graph.add_edge(
                    u, v,
                    timestamp=ts.isoformat(),
                    event_type="interaction",
                    weight=1.0
                )

        return graph

    def test_partition_by_time_windows(self, manager):
        """Graph with edges across 3 months creates 3+ partitions."""
        graph = self._create_test_graph(months=3)

        partition_ids = manager.partition_graph(graph, window_days=30)

        # Should have multiple partitions (at least 3 for 3 months)
        assert len(partition_ids) >= 3

        # Partition IDs should be date strings
        for pid in partition_ids:
            if pid != "no-timestamp":
                # Should parse as date
                datetime.fromisoformat(pid)

    def test_partition_saves_graphml(self, manager):
        """Each partition directory contains graph.graphml file."""
        graph = self._create_test_graph(months=2)

        partition_ids = manager.partition_graph(graph, window_days=30)

        for pid in partition_ids:
            file_path = manager.get_partition_path(pid)
            assert file_path.exists(), f"GraphML not found for partition {pid}"
            assert file_path.suffix == ".graphml"

            # File should have content
            assert file_path.stat().st_size > 0

    def test_partition_entities_indexed(self, manager, index):
        """All entities appear in partition index after partitioning."""
        graph = self._create_test_graph(months=2)

        partition_ids = manager.partition_graph(graph, window_days=30)

        # All entities from original graph should be findable
        for node in graph.nodes():
            partitions = index.get_entity_partitions(node)
            assert len(partitions) > 0, f"Entity {node} not indexed"
            # Entity should be in one of the created partitions
            assert any(p in partition_ids for p in partitions)

    def test_load_partition_caching(self, manager):
        """Second load returns cached copy without file I/O."""
        graph = self._create_test_graph(months=1)
        partition_ids = manager.partition_graph(graph, window_days=30)

        pid = partition_ids[0]

        # First load - from disk
        graph1 = manager.load_partition(pid)
        assert manager.is_cached(pid)

        # Second load - from cache (same object)
        graph2 = manager.load_partition(pid)
        assert graph1 is graph2  # Same object reference

    def test_lru_eviction(self, temp_dir):
        """Loading 5 partitions with max_cached=4 evicts oldest."""
        # Create manager with max_cached=4
        db_path = temp_dir / "lru_index.db"
        partition_dir = temp_dir / "lru_partitions"

        index = EntityPartitionIndex(db_path)
        manager = PartitionManager(
            partition_dir=partition_dir,
            index=index,
            max_cached_partitions=4
        )

        # Create graph spanning 5 months to produce 5 partitions
        graph = nx.MultiDiGraph()
        base = datetime(2024, 1, 1)
        for month in range(5):
            ts = base + timedelta(days=month * 30)
            graph.add_edge(
                f"A{month}", f"B{month}",
                timestamp=ts.isoformat()
            )

        partition_ids = manager.partition_graph(graph, window_days=30)
        assert len(partition_ids) >= 5, f"Expected 5+ partitions, got {len(partition_ids)}"

        # Load all 5 partitions
        for pid in partition_ids[:5]:
            manager.load_partition(pid)

        # Cache should respect max_cached_partitions
        cache_size = len(manager._cache)
        assert cache_size == 4, f"Cache has {cache_size} partitions, expected 4"

        # Oldest partition (first loaded) should be evicted
        first_partition = partition_ids[0]
        assert first_partition not in manager._cache, \
            f"Oldest partition {first_partition} should be evicted"

        # Most recent 4 should still be cached
        for pid in partition_ids[1:5]:
            assert pid in manager._cache, f"Partition {pid} should be in cache"

        index.close()

    def test_partition_roundtrip(self, manager):
        """Loaded partition has same nodes/edges as original subgraph."""
        graph = self._create_test_graph(months=1)

        # Add some node attributes for verification
        for node in graph.nodes():
            graph.nodes[node]['verified'] = True

        partition_ids = manager.partition_graph(graph, window_days=30)

        # Load and verify each partition
        total_edges = 0
        total_nodes = set()

        for pid in partition_ids:
            loaded = manager.load_partition(pid)

            # Verify nodes have attributes
            for node in loaded.nodes():
                total_nodes.add(node)
                # GraphML may convert booleans to strings
                assert 'verified' in loaded.nodes[node] or 'type' in loaded.nodes[node]

            # Verify edges have attributes
            for u, v, data in loaded.edges(data=True):
                assert 'timestamp' in data or 'event_type' in data
                total_edges += 1

        # Total should match original
        assert total_nodes == set(graph.nodes())
        assert total_edges == graph.number_of_edges()

    def test_empty_graph(self, manager):
        """Empty graph produces no partitions."""
        graph = nx.MultiDiGraph()

        partition_ids = manager.partition_graph(graph, window_days=30)

        assert partition_ids == []

    def test_no_timestamp_edges(self, manager):
        """Edges without timestamps go to fallback partition."""
        graph = nx.MultiDiGraph()

        # Add edges without timestamp attribute
        graph.add_edge("A", "B", event_type="unknown")
        graph.add_edge("B", "C", event_type="unknown")

        partition_ids = manager.partition_graph(graph, window_days=30)

        # Should have exactly one partition for no-timestamp edges
        assert len(partition_ids) == 1
        assert partition_ids[0] == "no-timestamp"

        # Load and verify
        loaded = manager.load_partition("no-timestamp")
        assert loaded.number_of_edges() == 2

    def test_mixed_timestamp_edges(self, manager):
        """Graph with mixed timestamped and non-timestamped edges."""
        graph = nx.MultiDiGraph()

        # Timestamped edges
        graph.add_edge("A", "B", timestamp="2024-01-15T00:00:00")
        graph.add_edge("B", "C", timestamp="2024-02-15T00:00:00")

        # Non-timestamped edge
        graph.add_edge("C", "D", event_type="unknown")

        partition_ids = manager.partition_graph(graph, window_days=30)

        # Should have 3 partitions: Jan, Feb, no-timestamp
        assert len(partition_ids) == 3
        assert "no-timestamp" in partition_ids

    def test_partition_metadata_in_index(self, manager, index):
        """Partition metadata is correctly stored in index."""
        graph = self._create_test_graph(months=2)
        partition_ids = manager.partition_graph(graph, window_days=30)

        for pid in partition_ids:
            if pid == "no-timestamp":
                continue

            meta = index.get_partition_meta(pid)
            assert meta is not None
            assert meta['partition_id'] == pid
            assert meta['node_count'] > 0
            assert meta['edge_count'] > 0
            assert meta['file_path'] is not None

            # Time window should be set
            assert meta['time_window_start'] != ""
            assert meta['time_window_end'] != ""

    def test_list_partitions(self, manager, index):
        """list_partitions returns all registered partition IDs."""
        graph = self._create_test_graph(months=3)
        partition_ids = manager.partition_graph(graph, window_days=30)

        listed = manager.list_partitions()

        assert set(listed) == set(partition_ids)

    def test_clear_cache(self, manager):
        """clear_cache empties the cache."""
        graph = self._create_test_graph(months=2)
        partition_ids = manager.partition_graph(graph, window_days=30)

        # Load partitions
        for pid in partition_ids:
            manager.load_partition(pid)

        assert len(manager._cache) > 0

        # Clear cache
        manager.clear_cache()

        assert len(manager._cache) == 0

    def test_partition_by_time_windows_alias(self, manager):
        """partition_by_time_windows is alias for partition_graph."""
        graph = self._create_test_graph(months=2)

        # Use the alias method (matches RESEARCH.md pattern name)
        partition_ids = manager.partition_by_time_windows(graph, window_days=30)

        assert len(partition_ids) >= 2

    def test_load_nonexistent_partition(self, manager):
        """Loading non-existent partition raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            manager.load_partition("nonexistent-partition")

    def test_lru_access_order_update(self, temp_dir):
        """Accessing cached partition moves it to end of LRU order."""
        db_path = temp_dir / "access_index.db"
        partition_dir = temp_dir / "access_partitions"

        index = EntityPartitionIndex(db_path)
        manager = PartitionManager(
            partition_dir=partition_dir,
            index=index,
            max_cached_partitions=3
        )

        # Create 3 partitions
        graph = nx.MultiDiGraph()
        for i in range(3):
            ts = datetime(2024, i + 1, 15).isoformat()
            graph.add_edge(f"A{i}", f"B{i}", timestamp=ts)

        partition_ids = manager.partition_graph(graph, window_days=30)
        assert len(partition_ids) == 3

        # Load all three: order is [0, 1, 2]
        for pid in partition_ids:
            manager.load_partition(pid)

        # Access first partition again - moves to end
        manager.load_partition(partition_ids[0])

        # Order should now be [1, 2, 0]
        cache_keys = list(manager._cache.keys())
        assert cache_keys[-1] == partition_ids[0], "First partition should be at end after re-access"

        # Add a 4th partition to trigger eviction
        graph.add_edge("A3", "B3", timestamp=datetime(2024, 4, 15).isoformat())
        partition_ids = manager.partition_graph(graph, window_days=30)

        # Load the new partition
        new_partition = [p for p in partition_ids if p not in cache_keys][0]
        manager.load_partition(new_partition)

        # partition_ids[1] (second in original order) should be evicted
        # because it's now the oldest after we re-accessed partition_ids[0]
        assert partition_ids[1] not in manager._cache

        index.close()
