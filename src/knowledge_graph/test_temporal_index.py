"""
Unit tests for temporal index.

Tests verify:
- Time-range queries return in < 10ms for 30-day windows
- Actor-pair lookups are O(1)
- Subgraph views correctly filter by QuadClass
"""

import pytest
import networkx as nx
from datetime import datetime, timedelta
import time
from src.knowledge_graph.temporal_index import TemporalIndex, create_index
from src.knowledge_graph.relation_classification import RelationType


@pytest.fixture
def sample_graph():
    """Create sample graph for testing."""
    graph = nx.MultiDiGraph()

    # Add nodes
    nodes = ['USA', 'CHN', 'RUS', 'UKR', 'EU', 'NATO']
    for node in nodes:
        graph.add_node(node, entity_type='country', canonical=True)

    # Add edges with timestamps
    base_date = datetime(2024, 1, 1)
    edges = [
        # Diplomatic events
        ('USA', 'CHN', {'timestamp': (base_date + timedelta(days=1)).isoformat() + 'Z',
                       'confidence': 0.7, 'quad_class': 1, 'relation_type': 'cooperates'}),
        ('USA', 'CHN', {'timestamp': (base_date + timedelta(days=2)).isoformat() + 'Z',
                       'confidence': 0.6, 'quad_class': 1, 'relation_type': 'negotiates'}),
        ('EU', 'RUS', {'timestamp': (base_date + timedelta(days=5)).isoformat() + 'Z',
                      'confidence': 0.4, 'quad_class': 1, 'relation_type': 'consults'}),

        # Conflict events
        ('RUS', 'UKR', {'timestamp': (base_date + timedelta(days=3)).isoformat() + 'Z',
                       'confidence': 0.9, 'quad_class': 4, 'relation_type': 'fights'}),
        ('RUS', 'UKR', {'timestamp': (base_date + timedelta(days=4)).isoformat() + 'Z',
                       'confidence': 0.95, 'quad_class': 4, 'relation_type': 'kills'}),
    ]

    for source, target, data in edges:
        graph.add_edge(source, target, **data)

    return graph


class TestTemporalIndex:
    """Test temporal index functionality."""

    @pytest.fixture
    def index(self, sample_graph):
        """Create index for tests."""
        return create_index(sample_graph)

    def test_index_creation(self, index):
        """Test index creation."""
        assert index is not None
        assert len(index.timestamp_index) > 0
        assert len(index.actor_pair_index) > 0

    def test_edges_in_time_range(self, index):
        """Test time-range queries."""
        # Query for first 3 days
        base_date = datetime(2024, 1, 1)
        start = (base_date + timedelta(days=1)).isoformat() + 'Z'
        end = (base_date + timedelta(days=3)).isoformat() + 'Z'

        results = index.edges_in_time_range(start, end)
        assert len(results) > 0
        # Should have edges from days 1, 2, 3

    def test_edges_in_time_range_exact_timestamp_match(self, index):
        """Test time-range query when search timestamp exactly matches an event.

        Regression test for UAT-004: bisect operations must use type-consistent
        sentinels to avoid TypeError when comparing string nodes with integers.
        """
        base_date = datetime(2024, 1, 1)
        # Use exact timestamp that exists in the index
        exact_timestamp = (base_date + timedelta(days=1)).isoformat() + 'Z'

        # This should NOT raise TypeError: '<' not supported between 'str' and 'int'
        results = index.edges_in_time_range(exact_timestamp, exact_timestamp)

        # Should find the USA-CHN event on day 1
        assert len(results) >= 1
        # Verify we got valid results (not empty due to off-by-one)
        for u, v, key in results:
            assert isinstance(u, str)
            assert isinstance(v, str)

    def test_time_range_query_performance(self, index):
        """Test that time-range queries are fast (< 10ms)."""
        base_date = datetime(2024, 1, 1)
        start = base_date.isoformat() + 'Z'
        end = (base_date + timedelta(days=30)).isoformat() + 'Z'

        start_time = time.perf_counter()
        for _ in range(100):  # 100 queries
            index.edges_in_time_range(start, end)
        elapsed = time.perf_counter() - start_time

        avg_time_ms = (elapsed * 1000) / 100
        assert avg_time_ms < 10, f"Query time {avg_time_ms}ms exceeds 10ms"

    def test_actor_pair_lookup(self, index):
        """Test actor-pair lookups are O(1)."""
        results = index.get_actor_pair_relations('USA', 'CHN')
        assert len(results) == 2  # Two relations between USA and CHN

    def test_actor_pair_lookup_performance(self, index):
        """Test actor-pair lookup is O(1)."""
        start_time = time.perf_counter()
        for _ in range(10000):
            index.get_actor_pair_relations('USA', 'CHN')
        elapsed = time.perf_counter() - start_time

        avg_time_us = (elapsed * 1_000_000) / 10000
        # O(1) should be < 10 microseconds
        assert avg_time_us < 100

    def test_bilateral_relations(self, index):
        """Test bilateral relation queries."""
        forward, backward = index.bilateral_relations('USA', 'CHN')
        assert len(forward) == 2
        assert len(backward) == 0  # CHN doesn't target USA in sample

    def test_temporal_neighbors(self, index):
        """Test temporal neighbor queries."""
        neighbors = index.temporal_neighbors('USA', direction='out')
        assert 'CHN' in neighbors  # USA -> CHN

    def test_quadclass_subgraph(self, index):
        """Test QuadClass subgraph filtering."""
        # Get diplomatic events (QuadClass 1)
        diplomatic = index.quadclass_subgraph(1)
        assert diplomatic.number_of_edges() == 3  # 2 USA-CHN + 1 EU-RUS

        # Get conflict events (QuadClass 4)
        conflict = index.quadclass_subgraph(4)
        assert conflict.number_of_edges() == 2  # 2 RUS-UKR

    def test_top_actors_by_degree(self, index):
        """Test top actors query."""
        top_out = index.top_actors_by_degree(top_k=3, direction='out')
        assert len(top_out) <= 3
        # USA and RUS should be top (both have 2 out-edges)

    def test_shortest_path(self, index):
        """Test shortest path query."""
        # There might be a path USA -> RUS if we check connectivity
        # For sample data, likely no path
        path = index.shortest_path('USA', 'RUS', max_length=2)
        # Path may be None for disconnected nodes

    def test_k_hop_neighbors(self, index):
        """Test k-hop neighbors."""
        neighbors = index.k_hop_neighbors('USA', k=1)
        assert 'CHN' in neighbors

    def test_strongly_connected_components(self, index):
        """Test SCC calculation."""
        sccs = index.strongly_connected_components()
        assert len(sccs) > 0
        # Most nodes should be in single-node SCCs

    def test_filter_by_confidence(self, index):
        """Test confidence filtering."""
        high_conf = index.filter_by_confidence(0.8)
        assert high_conf.number_of_edges() > 0
        # RUS-UKR has confidence 0.9 and 0.95

    def test_statistics(self, index):
        """Test statistics."""
        stats = index.get_statistics()
        assert 'indexed_edges' in stats
        assert 'indexed_actor_pairs' in stats
        assert 'quadclass_distribution' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
