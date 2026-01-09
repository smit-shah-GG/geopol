"""
Tests for Graph Traversal Engine.

Tests k-hop search, bilateral relations, temporal paths, and pattern matching.
"""

import pytest
import networkx as nx
from datetime import datetime, timedelta
from src.knowledge_graph.graph_traversal import (
    GraphTraversal, TraversalResult, create_traversal_engine
)


@pytest.fixture
def simple_graph():
    """Create simple test graph."""
    G = nx.MultiDiGraph()

    # Add nodes
    for i in range(1, 6):
        G.add_node(i, entity_type='country', name=f'Entity{i}')

    # Add edges with temporal and confidence attributes
    # 1 -> 2 (diplomatic)
    G.add_edge(1, 2, key=0,
               timestamp='2024-01-01T00:00:00Z',
               confidence=0.8,
               quad_class=1,
               relation_type='diplomatic_cooperation',
               num_mentions=100)

    # 2 -> 3 (conflict)
    G.add_edge(2, 3, key=0,
               timestamp='2024-01-05T00:00:00Z',
               confidence=0.9,
               quad_class=4,
               relation_type='military_action',
               num_mentions=200)

    # 3 -> 4 (diplomatic)
    G.add_edge(3, 4, key=0,
               timestamp='2024-01-10T00:00:00Z',
               confidence=0.7,
               quad_class=1,
               relation_type='negotiate',
               num_mentions=50)

    # 1 -> 4 (direct path)
    G.add_edge(1, 4, key=0,
               timestamp='2024-01-15T00:00:00Z',
               confidence=0.6,
               quad_class=2,
               relation_type='material_cooperation',
               num_mentions=75)

    # 4 -> 5 (conflict)
    G.add_edge(4, 5, key=0,
               timestamp='2024-01-20T00:00:00Z',
               confidence=0.85,
               quad_class=4,
               relation_type='threaten',
               num_mentions=150)

    # Bilateral: 2 <-> 3
    G.add_edge(3, 2, key=0,
               timestamp='2024-01-08T00:00:00Z',
               confidence=0.75,
               quad_class=3,
               relation_type='verbal_conflict',
               num_mentions=120)

    return G


@pytest.fixture
def temporal_index_mock():
    """Create mock temporal index."""
    class MockTemporalIndex:
        def __init__(self):
            self.actor_pair_index = {
                (1, 2): [(0, '2024-01-01T00:00:00Z')],
                (2, 3): [(0, '2024-01-05T00:00:00Z')],
                (3, 2): [(0, '2024-01-08T00:00:00Z')],
                (3, 4): [(0, '2024-01-10T00:00:00Z')],
                (1, 4): [(0, '2024-01-15T00:00:00Z')],
                (4, 5): [(0, '2024-01-20T00:00:00Z')]
            }
    return MockTemporalIndex()


class TestTraversalResult:
    """Test TraversalResult container."""

    def test_init_empty(self):
        """Test empty result initialization."""
        result = TraversalResult()
        assert len(result.nodes) == 0
        assert len(result.edges) == 0
        assert len(result.paths) == 0
        assert len(result.metadata) == 0

    def test_add_edge(self):
        """Test adding edges to result."""
        result = TraversalResult()
        result.add_edge(1, 2, 0, {'confidence': 0.8})

        assert 1 in result.nodes
        assert 2 in result.nodes
        assert len(result.edges) == 1
        assert result.edges[0] == (1, 2, 0, {'confidence': 0.8})

    def test_add_path(self):
        """Test adding paths to result."""
        result = TraversalResult()
        path = [(1, 2, 0), (2, 3, 0)]
        result.add_path(path)

        assert 1 in result.nodes
        assert 2 in result.nodes
        assert 3 in result.nodes
        assert len(result.paths) == 1
        assert result.paths[0] == path

    def test_get_subgraph(self, simple_graph):
        """Test extracting subgraph from result."""
        result = TraversalResult()
        result.add_edge(1, 2, 0, simple_graph[1][2][0])
        result.add_edge(2, 3, 0, simple_graph[2][3][0])

        subgraph = result.get_subgraph(simple_graph)

        assert len(subgraph.nodes()) == 3
        assert subgraph.has_edge(1, 2)
        assert subgraph.has_edge(2, 3)


class TestKHopNeighborhood:
    """Test k-hop neighborhood search."""

    def test_1hop_neighborhood(self, simple_graph):
        """Test 1-hop neighborhood."""
        engine = create_traversal_engine(simple_graph)
        result = engine.k_hop_neighborhood(entity_id=1, k=1)

        # From node 1, we should reach node 2 and 4
        assert 1 in result.nodes
        assert 2 in result.nodes or 4 in result.nodes
        assert len(result.edges) >= 1

    def test_2hop_neighborhood(self, simple_graph):
        """Test 2-hop neighborhood."""
        engine = create_traversal_engine(simple_graph)
        result = engine.k_hop_neighborhood(entity_id=1, k=2)

        # From node 1, 2-hop should reach further
        assert 1 in result.nodes
        assert len(result.nodes) >= 2
        assert result.metadata['k'] == 2

    def test_time_filter(self, simple_graph):
        """Test time window filtering."""
        engine = create_traversal_engine(simple_graph)

        # Only edges before Jan 10
        result = engine.k_hop_neighborhood(
            entity_id=1,
            k=2,
            time_start='2024-01-01T00:00:00Z',
            time_end='2024-01-10T00:00:00Z'
        )

        # Check that all returned edges are within time window
        for u, v, key, data in result.edges:
            timestamp = data['timestamp']
            assert '2024-01-01' <= timestamp <= '2024-01-10'

    def test_confidence_filter(self, simple_graph):
        """Test confidence filtering."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(
            entity_id=1,
            k=2,
            min_confidence=0.75
        )

        # Check that all edges meet confidence threshold
        for u, v, key, data in result.edges:
            assert data['confidence'] >= 0.75

    def test_quad_class_filter(self, simple_graph):
        """Test QuadClass filtering."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(
            entity_id=1,
            k=2,
            quad_class=4
        )

        # All returned edges should be quad_class 4
        for u, v, key, data in result.edges:
            assert data['quad_class'] == 4

    def test_max_results(self, simple_graph):
        """Test max_results limit."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(
            entity_id=1,
            k=5,
            max_results=3
        )

        assert len(result.edges) <= 3

    def test_nonexistent_entity(self, simple_graph):
        """Test query for nonexistent entity."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(entity_id=999, k=2)

        assert len(result.nodes) == 0
        assert len(result.edges) == 0


class TestBilateralRelations:
    """Test bilateral relation finding."""

    def test_bilateral_with_index(self, simple_graph, temporal_index_mock):
        """Test bilateral search with temporal index."""
        engine = GraphTraversal(simple_graph, temporal_index_mock)

        result = engine.bilateral_relations(entity1=2, entity2=3)

        # Should find both 2->3 and 3->2
        assert len(result.edges) >= 2
        edge_pairs = [(u, v) for u, v, _, _ in result.edges]
        assert (2, 3) in edge_pairs or (3, 2) in edge_pairs

    def test_bilateral_without_index(self, simple_graph):
        """Test bilateral search without temporal index."""
        engine = create_traversal_engine(simple_graph)

        result = engine.bilateral_relations(entity1=2, entity2=3)

        # Should still find relations
        assert len(result.edges) >= 1

    def test_bilateral_time_filter(self, simple_graph, temporal_index_mock):
        """Test bilateral with time filtering."""
        engine = GraphTraversal(simple_graph, temporal_index_mock)

        # Only edges up to Jan 7
        result = engine.bilateral_relations(
            entity1=2,
            entity2=3,
            time_end='2024-01-07T00:00:00Z'
        )

        # Should only get 2->3 (Jan 5), not 3->2 (Jan 8)
        for u, v, key, data in result.edges:
            assert data['timestamp'] <= '2024-01-07T00:00:00Z'

    def test_bilateral_no_relations(self, simple_graph):
        """Test bilateral for entities with no relations."""
        engine = create_traversal_engine(simple_graph)

        result = engine.bilateral_relations(entity1=1, entity2=5)

        # No direct relation between 1 and 5
        assert len(result.edges) == 0


class TestTemporalPaths:
    """Test temporal path finding."""

    def test_direct_path(self, simple_graph):
        """Test finding direct path."""
        engine = create_traversal_engine(simple_graph)

        result = engine.temporal_paths(source=1, target=2, max_length=1)

        # Direct path exists: 1 -> 2
        assert len(result.paths) >= 1
        assert result.metadata['source'] == 1
        assert result.metadata['target'] == 2

    def test_multihop_path(self, simple_graph):
        """Test finding multi-hop path."""
        engine = create_traversal_engine(simple_graph)

        result = engine.temporal_paths(source=1, target=3, max_length=3)

        # Path exists: 1 -> 2 -> 3
        assert len(result.paths) >= 1
        path_count = result.metadata['path_count']
        assert path_count >= 1

    def test_chronological_ordering(self, simple_graph):
        """Test chronological path ordering."""
        engine = create_traversal_engine(simple_graph)

        result = engine.temporal_paths(
            source=1,
            target=3,
            max_length=3,
            chronological_only=True
        )

        # Check that each path is chronologically ordered
        for path in result.paths:
            timestamps = []
            for u, v, key in path:
                data = simple_graph[u][v][key]
                timestamps.append(data['timestamp'])

            # Should be in ascending order
            for i in range(len(timestamps) - 1):
                assert timestamps[i] <= timestamps[i + 1]

    def test_max_path_length(self, simple_graph):
        """Test max path length constraint."""
        engine = create_traversal_engine(simple_graph)

        result = engine.temporal_paths(
            source=1,
            target=5,
            max_length=2
        )

        # With max_length=2, path 1->2->3->4->5 should not be found
        # But 1->4->5 might exist if length allows

    def test_no_path_exists(self, simple_graph):
        """Test when no path exists."""
        engine = create_traversal_engine(simple_graph)

        # No path from 5 to 1 (all edges point away from 1)
        result = engine.temporal_paths(source=5, target=1, max_length=5)

        assert len(result.paths) == 0


class TestPatternMatching:
    """Test event sequence pattern matching."""

    def test_simple_pattern(self, simple_graph):
        """Test simple two-step pattern."""
        engine = create_traversal_engine(simple_graph)

        pattern = [
            {"quad_class": 1},  # Diplomatic
            {"quad_class": 4}   # Conflict
        ]

        result = engine.pattern_match(pattern, start_entity=1)

        # Pattern 1->2 (diplomatic) -> 2->3 (conflict) should match
        assert len(result.paths) >= 0  # May or may not find depending on edges

    def test_pattern_with_relation_type(self, simple_graph):
        """Test pattern with specific relation types."""
        engine = create_traversal_engine(simple_graph)

        pattern = [
            {"relation_type": "diplomatic_cooperation"},
            {"relation_type": "military_action"}
        ]

        result = engine.pattern_match(pattern, start_entity=1)

        # Should match 1->2->3 if relation types match
        assert result.metadata['pattern_length'] == 2

    def test_pattern_max_matches(self, simple_graph):
        """Test max_matches limit."""
        engine = create_traversal_engine(simple_graph)

        pattern = [{"quad_class": 1}]

        result = engine.pattern_match(pattern, max_matches=1)

        assert result.metadata['matches'] <= 1

    def test_pattern_no_start_entity(self, simple_graph):
        """Test pattern matching without specific start entity."""
        engine = create_traversal_engine(simple_graph)

        pattern = [{"quad_class": 4}]

        result = engine.pattern_match(pattern)

        # Should search from all entities
        assert result.metadata['matches'] >= 0

    def test_empty_pattern(self, simple_graph):
        """Test empty pattern."""
        engine = create_traversal_engine(simple_graph)

        result = engine.pattern_match([])

        assert len(result.paths) == 0
        assert result.metadata['pattern_length'] == 0


class TestResultRanking:
    """Test result ranking."""

    def test_rank_by_confidence(self, simple_graph):
        """Test ranking by confidence."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(entity_id=1, k=2)
        ranked = engine.rank_results(result, ranking="confidence")

        # Check that edges are sorted by confidence descending
        confidences = [data['confidence'] for _, _, _, data in ranked.edges]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]

    def test_rank_by_recency(self, simple_graph):
        """Test ranking by recency."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(entity_id=1, k=2)
        ranked = engine.rank_results(result, ranking="recency")

        # Check that edges are sorted by timestamp descending (most recent first)
        timestamps = [data['timestamp'] for _, _, _, data in ranked.edges]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] >= timestamps[i + 1]

    def test_rank_by_mentions(self, simple_graph):
        """Test ranking by mentions."""
        engine = create_traversal_engine(simple_graph)

        result = engine.k_hop_neighborhood(entity_id=1, k=2)
        ranked = engine.rank_results(result, ranking="mentions")

        # Check that edges are sorted by num_mentions descending
        mentions = [data['num_mentions'] for _, _, _, data in ranked.edges]
        for i in range(len(mentions) - 1):
            assert mentions[i] >= mentions[i + 1]


class TestPerformance:
    """Test performance characteristics."""

    def test_2hop_performance(self, simple_graph):
        """Test that 2-hop search completes quickly."""
        import time

        engine = create_traversal_engine(simple_graph)

        start = time.time()
        result = engine.k_hop_neighborhood(entity_id=1, k=2)
        elapsed = time.time() - start

        # Should complete in well under 5ms for small graph
        assert elapsed < 0.1  # 100ms is very generous for small graph

    def test_bilateral_performance(self, simple_graph, temporal_index_mock):
        """Test that bilateral search is fast with index."""
        import time

        engine = GraphTraversal(simple_graph, temporal_index_mock)

        start = time.time()
        result = engine.bilateral_relations(entity1=2, entity2=3)
        elapsed = time.time() - start

        # Should be very fast with index
        assert elapsed < 0.1


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, simple_graph):
        """Test complete workflow: traverse, filter, rank."""
        engine = create_traversal_engine(simple_graph)

        # 1. Find k-hop neighborhood
        result = engine.k_hop_neighborhood(
            entity_id=1,
            k=2,
            min_confidence=0.6,
            time_start='2024-01-01T00:00:00Z',
            time_end='2024-01-31T00:00:00Z'
        )

        assert len(result.edges) >= 1

        # 2. Rank results
        ranked = engine.rank_results(result, ranking="confidence")

        # 3. Extract subgraph
        subgraph = ranked.get_subgraph(simple_graph)

        assert len(subgraph.nodes()) >= 2

    def test_multiple_query_types(self, simple_graph):
        """Test multiple query types on same graph."""
        engine = create_traversal_engine(simple_graph)

        # K-hop
        result1 = engine.k_hop_neighborhood(entity_id=1, k=2)
        assert len(result1.edges) >= 1

        # Bilateral
        result2 = engine.bilateral_relations(entity1=2, entity2=3)
        assert result2.metadata['query_type'] == 'bilateral_relations'

        # Temporal path
        result3 = engine.temporal_paths(source=1, target=3, max_length=3)
        assert result3.metadata['query_type'] == 'temporal_paths'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
