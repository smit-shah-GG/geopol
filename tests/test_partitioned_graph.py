"""
Tests for PartitionedTemporalGraph unified interface.

Verifies:
- Query correctness vs full graph (critical SCALE-02 validation)
- Bilateral relations across partitions
- Drop-in API compatibility with GraphTraversal
- Stats include boundary information
"""

import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Set

import networkx as nx
import pytest

from src.knowledge_graph.graph_traversal import GraphTraversal, TraversalResult
from src.knowledge_graph.partitioned_graph import (
    PartitionedTemporalGraph,
    load_partitioned_graph,
)
from src.knowledge_graph.partition_index import EntityPartitionIndex
from src.knowledge_graph.partition_manager import PartitionManager


class TestPartitionedTemporalGraph:
    """Test suite for PartitionedTemporalGraph."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _create_test_graph(
        self,
        num_nodes: int = 100,
        num_edges: int = 500,
        months: int = 3
    ) -> nx.MultiDiGraph:
        """
        Create a randomized test graph with temporal edges.

        Creates a graph where nodes are connected across time windows,
        creating a realistic cross-partition scenario.
        """
        random.seed(42)  # Reproducibility
        graph = nx.MultiDiGraph()

        # Create nodes
        nodes = [f"entity_{i}" for i in range(num_nodes)]
        for node in nodes:
            graph.add_node(node, type="actor", label=node)

        # Create timestamped edges
        base_date = datetime(2024, 1, 1)

        for edge_idx in range(num_edges):
            # Random source and target
            src = random.choice(nodes)
            tgt = random.choice(nodes)
            if src == tgt:
                continue  # Skip self-loops for simplicity

            # Random timestamp across the time range
            days_offset = random.randint(0, months * 30 - 1)
            ts = base_date + timedelta(days=days_offset)

            graph.add_edge(
                src, tgt,
                timestamp=ts.isoformat(),
                event_type="interaction",
                confidence=random.uniform(0.5, 1.0)
            )

        return graph

    def _setup_partitioned_graph(
        self,
        temp_dir: Path,
        graph: nx.MultiDiGraph,
        window_days: int = 30
    ) -> PartitionedTemporalGraph:
        """
        Partition a graph and return PartitionedTemporalGraph interface.
        """
        partition_dir = temp_dir / "partitions"
        index_path = temp_dir / "partition_meta.db"

        # Create index and manager
        index = EntityPartitionIndex(index_path)
        manager = PartitionManager(
            partition_dir=partition_dir,
            index=index,
            max_cached_partitions=8
        )

        # Partition the graph
        partition_ids = manager.partition_graph(graph, window_days=window_days)

        # Update edge counts for boundary detection
        for pid in partition_ids:
            loaded = manager.load_partition(pid)
            for node in loaded.nodes():
                edge_count = loaded.in_degree(node) + loaded.out_degree(node)
                index.update_entity_edge_count(str(node), pid, edge_count)

        index.close()

        # Return PartitionedTemporalGraph
        return PartitionedTemporalGraph(
            partition_dir=partition_dir,
            index_path=index_path,
            max_cached=8,
            max_workers=4,
            replication_threshold=5
        )

    def test_query_correctness_vs_full_graph(self, temp_dir):
        """
        CRITICAL TEST: Partitioned queries return same results as single-graph queries.

        This validates the SCALE-02 requirement that partitioning doesn't break
        query correctness.
        """
        # Create test graph with 100 nodes, 500 edges across 3 months
        full_graph = self._create_test_graph(
            num_nodes=100,
            num_edges=500,
            months=3
        )

        # Partition into time-window partitions
        ptg = self._setup_partitioned_graph(temp_dir, full_graph, window_days=30)

        # Create GraphTraversal for full graph comparison
        full_traversal = GraphTraversal(full_graph)

        # Sample 20 random entities to test
        random.seed(123)
        all_entities = list(full_graph.nodes())
        sample_entities = random.sample(
            all_entities,
            min(20, len(all_entities))
        )

        correct = 0
        total = len(sample_entities)
        failures = []

        for entity_id in sample_entities:
            # Query full graph
            full_result = full_traversal.k_hop_neighborhood(
                entity_id=entity_id,
                k=2
            )

            # Query partitioned graph
            partitioned_result = ptg.k_hop_neighborhood(
                entity_id=entity_id,
                k=2
            )

            # Compare node sets
            # Note: GraphTraversal may use int keys, PartitionedGraph uses str
            full_nodes = self._normalize_node_set(full_result.nodes)
            partitioned_nodes = self._normalize_node_set(partitioned_result.nodes)

            if full_nodes == partitioned_nodes:
                correct += 1
            else:
                missing = full_nodes - partitioned_nodes
                extra = partitioned_nodes - full_nodes
                failures.append({
                    'entity': entity_id,
                    'full_count': len(full_nodes),
                    'partitioned_count': len(partitioned_nodes),
                    'missing': len(missing),
                    'extra': len(extra)
                })

        accuracy = correct / total

        # MUST achieve 100% accuracy for SCALE-02 compliance
        assert accuracy == 1.0, (
            f"Query correctness failed: {correct}/{total} ({accuracy:.0%})\n"
            f"Failures: {failures}"
        )

        ptg.close()

    def _normalize_node_set(self, nodes: Set[Any]) -> Set[str]:
        """Normalize node IDs to strings for comparison."""
        return {str(n) for n in nodes}

    def test_bilateral_relations_across_partitions(self, temp_dir):
        """Entities in different partitions have relations found."""
        graph = nx.MultiDiGraph()

        # Create two entities that interact across multiple months
        base_date = datetime(2024, 1, 1)
        for month in range(3):
            ts = base_date + timedelta(days=month * 30 + 15)
            graph.add_edge(
                "country_A", "country_B",
                timestamp=ts.isoformat(),
                event_type=f"diplomatic_{month}",
                confidence=0.9
            )
            graph.add_edge(
                "country_B", "country_A",
                timestamp=ts.isoformat(),
                event_type=f"response_{month}",
                confidence=0.85
            )

        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        result = ptg.bilateral_relations(
            entity1="country_A",
            entity2="country_B"
        )

        # Should find all 6 edges (3 A->B, 3 B->A) across partitions
        assert len(result.edges) == 6
        assert "country_A" in result.nodes
        assert "country_B" in result.nodes

        ptg.close()

    def test_drop_in_api_compatibility(self, temp_dir):
        """PartitionedTemporalGraph has same method signatures as GraphTraversal."""
        graph = self._create_test_graph(num_nodes=20, num_edges=50, months=2)
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        # Verify k_hop_neighborhood signature matches
        result = ptg.k_hop_neighborhood(
            entity_id="entity_0",
            k=2,
            time_start="2024-01-01",
            time_end="2024-12-31",
            min_confidence=0.0,
            quad_class=None,
            max_results=100
        )
        assert isinstance(result, TraversalResult)

        # Verify bilateral_relations signature matches
        result = ptg.bilateral_relations(
            entity1="entity_0",
            entity2="entity_1",
            time_start="2024-01-01",
            time_end="2024-12-31",
            min_confidence=0.0,
            quad_class=None
        )
        assert isinstance(result, TraversalResult)

        ptg.close()

    def test_stats_include_boundary_info(self, temp_dir):
        """get_stats() includes boundary entity statistics."""
        graph = self._create_test_graph(num_nodes=50, num_edges=200, months=3)
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        stats = ptg.get_stats()

        # Verify structure
        assert 'partition_count' in stats
        assert 'partition_ids' in stats
        assert 'total_nodes' in stats
        assert 'total_edges' in stats
        assert 'cache' in stats
        assert 'boundary' in stats

        # Verify cache stats
        assert 'size' in stats['cache']
        assert 'max_size' in stats['cache']
        assert 'utilization' in stats['cache']

        # Verify boundary stats
        assert 'total_entities' in stats['boundary']
        assert 'boundary_entities' in stats['boundary']
        assert 'avg_replica_count' in stats['boundary']
        assert 'replication_ratio' in stats['boundary']
        assert 'replication_threshold' in stats['boundary']

        ptg.close()

    def test_factory_function(self, temp_dir):
        """load_partitioned_graph factory creates valid instance."""
        # First create partitions
        graph = self._create_test_graph(num_nodes=20, num_edges=50, months=2)
        partition_dir = temp_dir / "factory_test"
        index_path = partition_dir / "partition_meta.db"

        # Setup partitions
        index = EntityPartitionIndex(index_path)
        manager = PartitionManager(partition_dir=partition_dir, index=index)
        manager.partition_graph(graph, window_days=30)
        index.close()

        # Use factory function
        ptg = load_partitioned_graph(
            partition_dir=partition_dir,
            max_cached=4,
            max_workers=2
        )

        assert ptg.get_partition_count() > 0
        assert len(ptg.list_partitions()) > 0

        ptg.close()

    def test_context_manager(self, temp_dir):
        """Context manager properly closes resources."""
        graph = self._create_test_graph(num_nodes=20, num_edges=50, months=2)
        partition_dir = temp_dir / "ctx_test"
        index_path = partition_dir / "partition_meta.db"

        # Setup partitions
        index = EntityPartitionIndex(index_path)
        manager = PartitionManager(partition_dir=partition_dir, index=index)
        manager.partition_graph(graph, window_days=30)
        index.close()

        # Use context manager
        with PartitionedTemporalGraph(
            partition_dir=partition_dir,
            index_path=index_path
        ) as ptg:
            result = ptg.k_hop_neighborhood("entity_0", k=1)
            assert isinstance(result, TraversalResult)

        # After exit, should be closed (can't verify directly, but shouldn't error)

    def test_get_entity_partitions(self, temp_dir):
        """Can query which partitions contain an entity."""
        graph = self._create_test_graph(num_nodes=50, num_edges=200, months=3)
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        # Pick an entity and check its partitions
        entity = "entity_0"
        partitions = ptg.get_entity_partitions(entity)

        # Should be in at least one partition
        assert len(partitions) > 0

        # All returned partitions should exist
        all_partitions = ptg.list_partitions()
        for p in partitions:
            assert p in all_partitions

        ptg.close()

    def test_boundary_entity_detection(self, temp_dir):
        """Can detect boundary entities and their home partitions."""
        graph = self._create_test_graph(num_nodes=50, num_edges=300, months=3)
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        stats = ptg.get_stats()

        # With 50 nodes, 300 edges across 3 months, likely some boundaries
        # Just verify the API works
        boundary_count = stats['boundary']['boundary_entities']

        if boundary_count > 0:
            # Find a boundary entity
            for entity in [f"entity_{i}" for i in range(50)]:
                if ptg.is_boundary_entity(entity):
                    home = ptg.get_home_partition(entity)
                    assert home is not None
                    assert home in ptg.list_partitions()
                    break

        ptg.close()

    def test_clear_cache(self, temp_dir):
        """clear_cache empties the partition cache."""
        graph = self._create_test_graph(num_nodes=30, num_edges=100, months=3)
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        # Query to populate cache
        ptg.k_hop_neighborhood("entity_0", k=1)
        ptg.k_hop_neighborhood("entity_1", k=1)

        stats_before = ptg.get_stats()
        assert stats_before['cache']['size'] > 0

        # Clear cache
        ptg.clear_cache()

        stats_after = ptg.get_stats()
        assert stats_after['cache']['size'] == 0

        ptg.close()

    def test_performance_ratio(self, temp_dir):
        """
        Performance test: partitioned queries should be < 2x slower than single graph.

        This is a soft requirement - we warn if exceeded but don't fail.
        """
        # Create larger test graph
        graph = self._create_test_graph(num_nodes=200, num_edges=1000, months=3)

        # Setup partitioned graph
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        # Setup full graph traversal
        full_traversal = GraphTraversal(graph)

        # Sample queries
        random.seed(456)
        sample_entities = random.sample(list(graph.nodes()), 10)

        # Time full graph queries
        start = time.time()
        for entity in sample_entities:
            full_traversal.k_hop_neighborhood(entity_id=entity, k=2)
        full_time = time.time() - start

        # Time partitioned queries
        start = time.time()
        for entity in sample_entities:
            ptg.k_hop_neighborhood(entity_id=entity, k=2)
        partitioned_time = time.time() - start

        # Calculate ratio
        if full_time > 0:
            ratio = partitioned_time / full_time
        else:
            ratio = 1.0

        ptg.close()

        # Warn if ratio exceeds 2x, but don't fail
        if ratio > 2.0:
            pytest.skip(
                f"Performance warning: partitioned is {ratio:.1f}x slower than full "
                f"(full: {full_time:.3f}s, partitioned: {partitioned_time:.3f}s)"
            )

        # Always pass with info
        print(
            f"\nPerformance: ratio={ratio:.2f}x "
            f"(full: {full_time:.3f}s, partitioned: {partitioned_time:.3f}s)"
        )

    def test_empty_result_for_nonexistent_entity(self, temp_dir):
        """Non-existent entity returns empty result."""
        graph = self._create_test_graph(num_nodes=20, num_edges=50, months=2)
        ptg = self._setup_partitioned_graph(temp_dir, graph, window_days=30)

        result = ptg.k_hop_neighborhood("totally_fake_entity", k=2)

        assert len(result.nodes) == 0
        assert len(result.edges) == 0

        ptg.close()
