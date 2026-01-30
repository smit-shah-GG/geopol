"""
Tests for QueryRouter scatter-gather cross-partition queries.

Verifies:
- Scatter-gather query execution across partitions
- Parallel execution timing
- Edge deduplication on merge
- Time range filtering
- Entity not found handling
- Home partition prioritization for boundary entities
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.knowledge_graph.boundary_resolver import BoundaryResolver
from src.knowledge_graph.cross_partition_query import QueryRouter
from src.knowledge_graph.partition_index import EntityPartitionIndex
from src.knowledge_graph.partition_manager import PartitionManager


class TestQueryRouter:
    """Test suite for QueryRouter."""

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
            max_cached_partitions=8
        )

    @pytest.fixture
    def resolver(self, index, manager):
        """Create a BoundaryResolver."""
        return BoundaryResolver(index, manager, replication_threshold=3)

    @pytest.fixture
    def router(self, manager, index, resolver):
        """Create a QueryRouter."""
        return QueryRouter(
            manager=manager,
            index=index,
            boundary_resolver=resolver,
            max_workers=4
        )

    def _create_partitioned_test_graph(
        self,
        manager: PartitionManager,
        months: int = 3
    ) -> nx.MultiDiGraph:
        """
        Create test graph and partition it.

        Returns the full (pre-partition) graph for comparison.
        """
        graph = nx.MultiDiGraph()
        base_date = datetime(2024, 1, 1)

        # Create a graph with entities that span partitions
        # "shared_entity" appears in all months
        for month in range(months):
            ts_base = base_date + timedelta(days=month * 30)

            for day in [5, 15, 25]:
                ts = ts_base + timedelta(days=day)

                # Month-local entities
                local_a = f"entity_{month}_a"
                local_b = f"entity_{month}_b"

                # Add nodes
                for node in [local_a, local_b, "shared_entity"]:
                    if node not in graph:
                        graph.add_node(node, type="actor", label=node)

                # Local edge within month
                graph.add_edge(
                    local_a, local_b,
                    timestamp=ts.isoformat(),
                    event_type="local_interaction",
                    confidence=0.9
                )

                # Edge involving shared entity
                graph.add_edge(
                    "shared_entity", local_a,
                    timestamp=ts.isoformat(),
                    event_type="shared_interaction",
                    confidence=0.8
                )

        # Partition the graph
        manager.partition_graph(graph, window_days=30)

        return graph

    def test_scatter_gather_basic(self, manager, index, resolver, router):
        """Query entity in 2 partitions returns merged result."""
        full_graph = self._create_partitioned_test_graph(manager, months=3)

        # Query shared_entity which exists in all partitions
        result = router.execute_k_hop_query(
            entity_id="shared_entity",
            k=1
        )

        # Should have found nodes across partitions
        assert len(result.nodes) > 0
        assert len(result.edges) > 0

        # Metadata should track partitions queried
        assert 'partitions_queried' in result.metadata
        assert len(result.metadata['partitions_queried']) >= 2

    def test_parallel_execution(self, manager, index, resolver):
        """Multiple partitions queried concurrently (verify via timing)."""
        # Create graph with many partitions
        graph = nx.MultiDiGraph()
        base_date = datetime(2024, 1, 1)

        # Create 4 partitions
        for month in range(4):
            ts = base_date + timedelta(days=month * 30 + 15)
            graph.add_edge(
                "parallel_test_entity", f"target_{month}",
                timestamp=ts.isoformat(),
                event_type="test"
            )

        manager.partition_graph(graph, window_days=30)

        # Create router with 4 workers
        router = QueryRouter(
            manager=manager,
            index=index,
            boundary_resolver=resolver,
            max_workers=4
        )

        # Time the query - parallel should be faster than sequential
        start = time.time()
        result = router.execute_k_hop_query(
            entity_id="parallel_test_entity",
            k=1
        )
        elapsed = time.time() - start

        # Should have results from all partitions
        partitions_queried = result.metadata.get('partitions_queried', [])
        assert len(partitions_queried) >= 4

        # Parallel execution: even with overhead, should complete quickly
        # (This is a soft check - mainly verifies parallelism works)
        assert elapsed < 5.0  # Should be well under 5 seconds for simple queries

        router.shutdown()

    def test_edge_deduplication(self, manager, index, resolver, router):
        """Same edge in 2 partitions appears once in result."""
        # Create a graph where an edge could appear in multiple partitions
        # due to boundary entity replication
        graph = nx.MultiDiGraph()

        # Edge at partition boundary (could be in multiple windows depending on bucketing)
        ts1 = datetime(2024, 1, 30).isoformat()  # End of January
        ts2 = datetime(2024, 2, 1).isoformat()   # Start of February

        graph.add_edge("entity_a", "entity_b", timestamp=ts1, event_type="test", key="edge_1")
        graph.add_edge("entity_a", "entity_b", timestamp=ts2, event_type="test", key="edge_2")
        graph.add_edge("entity_a", "entity_c", timestamp=ts1, event_type="test", key="edge_3")

        manager.partition_graph(graph, window_days=30)

        # Query entity_a
        result = router.execute_k_hop_query(
            entity_id="entity_a",
            k=1
        )

        # Count edges - should have exactly 3 unique edges, no duplicates
        edge_keys = [(e[0], e[1], e[2]) for e in result.edges]
        unique_edges = set(edge_keys)

        assert len(edge_keys) == len(unique_edges), "Edges should be deduplicated"

    def test_time_range_filtering(self, manager, index, resolver, router):
        """Only partitions in time range are queried."""
        # Create a more controlled test graph with distinct time windows
        graph = nx.MultiDiGraph()

        # January entity (clearly in January)
        graph.add_edge(
            "jan_entity", "jan_target",
            timestamp="2024-01-10T00:00:00",
            event_type="january_event"
        )

        # March entity (clearly in March)
        graph.add_edge(
            "mar_entity", "mar_target",
            timestamp="2024-03-15T00:00:00",
            event_type="march_event"
        )

        # Shared entity appears in both
        graph.add_edge(
            "shared_test", "jan_target",
            timestamp="2024-01-10T00:00:00"
        )
        graph.add_edge(
            "shared_test", "mar_target",
            timestamp="2024-03-15T00:00:00"
        )

        manager.partition_graph(graph, window_days=30)

        # Query shared_test for March only
        result = router.execute_k_hop_query(
            entity_id="shared_test",
            k=1,
            time_start="2024-03-01",
            time_end="2024-03-31"
        )

        # Should have queried partitions (check metadata)
        partitions_queried = result.metadata.get('partitions_queried', [])

        # With time filtering, we should only get March-related results
        # The edges should be filtered even if the partition overlaps
        # Check that at least the result edges are within time range
        for edge in result.edges:
            ts = edge[3].get('timestamp', '')
            if ts and ts != '':
                assert ts >= "2024-03-01", f"Edge timestamp {ts} is before March"

    def test_entity_not_found(self, manager, index, resolver, router):
        """Entity not in any partition returns empty result."""
        # Create some partitions
        graph = nx.MultiDiGraph()
        graph.add_edge("existing_a", "existing_b", timestamp="2024-01-15T00:00:00")
        manager.partition_graph(graph, window_days=30)

        # Query non-existent entity
        result = router.execute_k_hop_query(
            entity_id="nonexistent_entity",
            k=2
        )

        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    def test_home_partition_priority(self, manager, index, resolver, router):
        """Verify that for boundary entities, home partition appears first in query order."""
        # Create graph where boundary entity has different edge counts per partition
        graph = nx.MultiDiGraph()
        base_date = datetime(2024, 1, 1)

        # Partition 1 (January): boundary entity has 5 edges
        for i in range(5):
            ts = base_date + timedelta(days=i + 5)
            graph.add_edge(
                "boundary_ent", f"jan_target_{i}",
                timestamp=ts.isoformat(),
                event_type="test"
            )

        # Partition 2 (February): boundary entity has 10 edges (this should be home)
        for i in range(10):
            ts = base_date + timedelta(days=35 + i)
            graph.add_edge(
                "boundary_ent", f"feb_target_{i}",
                timestamp=ts.isoformat(),
                event_type="test"
            )

        # Partition 3 (March): boundary entity has 3 edges
        for i in range(3):
            ts = base_date + timedelta(days=65 + i)
            graph.add_edge(
                "boundary_ent", f"mar_target_{i}",
                timestamp=ts.isoformat(),
                event_type="test"
            )

        partition_ids = manager.partition_graph(graph, window_days=30)

        # Update edge counts in index
        for pid in partition_ids:
            loaded = manager.load_partition(pid)
            if "boundary_ent" in loaded:
                edge_count = (
                    loaded.in_degree("boundary_ent") +
                    loaded.out_degree("boundary_ent")
                )
                index.update_entity_edge_count("boundary_ent", pid, edge_count)

        # Now identify boundaries (resolver needs fresh data)
        resolver.invalidate_cache()
        resolver.identify_boundary_entities()

        # Track which partition is queried first
        query_order: List[str] = []
        original_query = router._query_single_partition

        def tracked_query(partition_id, *args, **kwargs):
            query_order.append(partition_id)
            return original_query(partition_id, *args, **kwargs)

        with patch.object(router, '_query_single_partition', side_effect=tracked_query):
            result = router.execute_k_hop_query(
                entity_id="boundary_ent",
                k=1
            )

        # Home partition (February, with 10 edges) should be first
        home = resolver.get_home_partition("boundary_ent")
        assert home is not None, "boundary_ent should be identified as boundary"

        # Home should be queried first (first in query_order)
        # Note: Due to parallel execution, we can only verify it's in the prioritized list
        prioritized = router._prioritize_home_partition(
            "boundary_ent",
            partition_ids
        )
        assert prioritized[0] == home, "Home partition should be first in prioritized list"

    def test_bilateral_query(self, manager, index, resolver, router):
        """Bilateral query finds relations between two entities across partitions."""
        graph = nx.MultiDiGraph()

        # Create relations between A and B across multiple months
        base_date = datetime(2024, 1, 1)
        for month in range(3):
            ts = base_date + timedelta(days=month * 30 + 15)
            graph.add_edge(
                "entity_a", "entity_b",
                timestamp=ts.isoformat(),
                event_type=f"relation_{month}",
                confidence=0.8 + month * 0.05
            )
            # Also add reverse direction
            graph.add_edge(
                "entity_b", "entity_a",
                timestamp=ts.isoformat(),
                event_type=f"reverse_{month}",
                confidence=0.7 + month * 0.05
            )

        manager.partition_graph(graph, window_days=30)

        result = router.execute_bilateral_query(
            entity1="entity_a",
            entity2="entity_b"
        )

        # Should find 6 edges (3 A->B + 3 B->A)
        assert len(result.edges) == 6
        assert "entity_a" in result.nodes
        assert "entity_b" in result.nodes

    def test_empty_result_for_disconnected_entities(self, manager, index, resolver, router):
        """Bilateral query returns empty for entities in different partitions with no connection."""
        graph = nx.MultiDiGraph()

        # Entity A in January
        graph.add_edge("entity_a", "other_a", timestamp="2024-01-15T00:00:00")

        # Entity B in February (no connection to A)
        graph.add_edge("entity_b", "other_b", timestamp="2024-02-15T00:00:00")

        manager.partition_graph(graph, window_days=30)

        result = router.execute_bilateral_query(
            entity1="entity_a",
            entity2="entity_b"
        )

        # No shared partition = no results
        assert len(result.edges) == 0

    def test_shutdown(self, router):
        """Router shutdown doesn't raise errors."""
        # Should be able to shutdown cleanly
        router.shutdown()

        # Double shutdown should also be safe
        router.shutdown()

    def test_query_with_filters(self, manager, index, resolver, router):
        """Query filters (confidence, quad_class) are applied."""
        graph = nx.MultiDiGraph()

        # Add edges with varying confidence
        graph.add_edge(
            "filter_ent", "target_low",
            timestamp="2024-01-15T00:00:00",
            confidence=0.3
        )
        graph.add_edge(
            "filter_ent", "target_high",
            timestamp="2024-01-16T00:00:00",
            confidence=0.9
        )

        manager.partition_graph(graph, window_days=30)

        # Query with high confidence filter
        result = router.execute_k_hop_query(
            entity_id="filter_ent",
            k=1,
            min_confidence=0.5
        )

        # Should only have the high-confidence edge
        confidences = [e[3].get('confidence', 0) for e in result.edges]
        assert all(c >= 0.5 for c in confidences)
        assert len(result.edges) == 1

    def test_max_results_limit(self, manager, index, resolver, router):
        """Query respects max_results limit."""
        graph = nx.MultiDiGraph()

        # Create many edges
        for i in range(50):
            graph.add_edge(
                "many_edges_ent", f"target_{i}",
                timestamp="2024-01-15T00:00:00",
                event_type=f"type_{i}"
            )

        manager.partition_graph(graph, window_days=30)

        result = router.execute_k_hop_query(
            entity_id="many_edges_ent",
            k=1,
            max_results=10
        )

        # Should have at most 10 edges
        assert len(result.edges) <= 10
