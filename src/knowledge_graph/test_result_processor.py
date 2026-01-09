"""
Tests for Result Processor (Tasks 4 & 5 combined).

Tests temporal filtering/aggregation and result formatting/explanation.
"""

import pytest
import networkx as nx
from datetime import datetime, timedelta
from unittest.mock import Mock
from src.knowledge_graph.result_processor import (
    TemporalFilterAggregator, ResultFormatter,
    FormattedResult, ExplanationPath,
    create_temporal_aggregator, create_result_formatter
)


@pytest.fixture
def sample_edges():
    """Create sample edges with temporal data."""
    edges = [
        (1, 2, 0, {
            'timestamp': '2024-01-01T00:00:00Z',
            'confidence': 0.8,
            'num_mentions': 100,
            'quad_class': 1,
            'relation_type': 'diplomatic_cooperation'
        }),
        (2, 3, 0, {
            'timestamp': '2024-01-05T00:00:00Z',
            'confidence': 0.9,
            'num_mentions': 200,
            'quad_class': 4,
            'relation_type': 'military_action'
        }),
        (3, 4, 0, {
            'timestamp': '2024-01-10T00:00:00Z',
            'confidence': 0.7,
            'num_mentions': 50,
            'quad_class': 1,
            'relation_type': 'negotiate'
        }),
        (1, 4, 0, {
            'timestamp': '2024-01-15T00:00:00Z',
            'confidence': 0.6,
            'num_mentions': 75,
            'quad_class': 2,
            'relation_type': 'material_cooperation'
        }),
    ]
    return edges


@pytest.fixture
def sample_graph():
    """Create sample graph for formatting tests."""
    G = nx.MultiDiGraph()

    for i in range(1, 5):
        G.add_node(i, entity_type='country', name=f'Entity{i}')

    G.add_edge(1, 2, key=0, timestamp='2024-01-01T00:00:00Z',
               confidence=0.8, relation_type='diplomatic_cooperation')
    G.add_edge(2, 3, key=0, timestamp='2024-01-05T00:00:00Z',
               confidence=0.9, relation_type='military_action')

    return G


@pytest.fixture
def mock_traversal_result():
    """Create mock traversal result."""
    result = Mock()
    result.nodes = {1, 2, 3}
    result.edges = [
        (1, 2, 0, {'confidence': 0.8, 'relation_type': 'diplomatic', 'timestamp': '2024-01-01T00:00:00Z'}),
        (2, 3, 0, {'confidence': 0.9, 'relation_type': 'conflict', 'timestamp': '2024-01-05T00:00:00Z'})
    ]
    result.paths = [[(1, 2, 0), (2, 3, 0)]]
    result.metadata = {'query_type': 'test'}
    return result


class TestTemporalFilterAggregator:
    """Test temporal filtering and aggregation."""

    def test_filter_by_time_window(self, sample_edges):
        """Test time window filtering."""
        aggregator = create_temporal_aggregator()

        filtered = aggregator.filter_by_time_window(
            edges=sample_edges,
            start_time='2024-01-01T00:00:00Z',
            end_time='2024-01-10T00:00:00Z'
        )

        # Should include first 3 edges, exclude last
        assert len(filtered) == 3
        assert all(e[3]['timestamp'] <= '2024-01-10T00:00:00Z' for e in filtered)

    def test_aggregate_by_daily(self, sample_edges):
        """Test daily aggregation."""
        aggregator = create_temporal_aggregator()

        aggregations = aggregator.aggregate_by_time_period(
            edges=sample_edges,
            period="daily"
        )

        # Should have 4 daily buckets
        assert len(aggregations) == 4

        # Check first aggregation
        first_agg = aggregations[0]
        assert first_agg.event_count == 1
        assert first_agg.total_mentions == 100

    def test_aggregate_by_weekly(self, sample_edges):
        """Test weekly aggregation."""
        aggregator = create_temporal_aggregator()

        aggregations = aggregator.aggregate_by_time_period(
            edges=sample_edges,
            period="weekly"
        )

        # All events within same week (Jan 1-15)
        assert len(aggregations) >= 1

    def test_sliding_window_trends(self, sample_edges):
        """Test sliding window trend analysis."""
        aggregator = create_temporal_aggregator()

        trends = aggregator.compute_sliding_window_trends(
            edges=sample_edges,
            window_days=7,
            slide_days=3
        )

        # Should have multiple windows
        assert len(trends) >= 1

        # Check trend structure
        assert 'window_start' in trends[0]
        assert 'window_end' in trends[0]
        assert 'event_count' in trends[0]
        assert 'avg_confidence' in trends[0]

    def test_temporal_decay(self, sample_edges):
        """Test exponential temporal decay."""
        aggregator = create_temporal_aggregator()

        weighted = aggregator.apply_temporal_decay(
            edges=sample_edges,
            reference_time='2024-01-15T00:00:00Z',
            half_life_days=7.0
        )

        assert len(weighted) == len(sample_edges)

        # Recent events should have higher weight
        # Last event (Jan 15) should have weight ~1.0
        last_edge = [w for w in weighted if w[3]['timestamp'] == '2024-01-15T00:00:00Z'][0]
        assert last_edge[4] > 0.9  # Weight close to 1.0

        # Older events should have lower weight
        first_edge = [w for w in weighted if w[3]['timestamp'] == '2024-01-01T00:00:00Z'][0]
        assert first_edge[4] < last_edge[4]

    def test_detect_cooccurrence(self, sample_edges):
        """Test temporal co-occurrence detection."""
        aggregator = create_temporal_aggregator()

        # Add two events close in time
        close_edges = sample_edges[:2].copy()

        cooccurrences = aggregator.detect_temporal_cooccurrence(
            edges=close_edges,
            time_threshold_hours=120  # 5 days
        )

        # Events on Jan 1 and Jan 5 (4 days apart) should co-occur
        assert len(cooccurrences) >= 1
        assert cooccurrences[0][2] <= 120  # Time diff within threshold

    def test_empty_edges(self):
        """Test handling of empty edge list."""
        aggregator = create_temporal_aggregator()

        filtered = aggregator.filter_by_time_window([], '2024-01-01T00:00:00Z', '2024-01-31T00:00:00Z')
        assert len(filtered) == 0

        aggregations = aggregator.aggregate_by_time_period([], period="daily")
        assert len(aggregations) == 0

        trends = aggregator.compute_sliding_window_trends([])
        assert len(trends) == 0


class TestResultFormatter:
    """Test result formatting and explanation generation."""

    def test_format_query_result(self, sample_graph, mock_traversal_result):
        """Test complete query result formatting."""
        id_to_entity = {1: 'USA', 2: 'CHN', 3: 'RUS'}

        formatter = create_result_formatter(
            graph=sample_graph,
            id_to_entity=id_to_entity
        )

        result = formatter.format_query_result(
            query_id='test-001',
            query_type='entity_pair',
            query_params={'entity1': 'USA', 'entity2': 'CHN'},
            traversal_result=mock_traversal_result,
            execution_time_ms=15.5
        )

        assert isinstance(result, FormattedResult)
        assert result.query_id == 'test-001'
        assert result.query_type == 'entity_pair'
        assert result.execution_time_ms == 15.5
        assert len(result.entities) == 3
        assert len(result.edges) == 2

    def test_generate_explanation(self, sample_graph):
        """Test explanation generation for path."""
        id_to_entity = {1: 'USA', 2: 'CHN', 3: 'RUS'}

        formatter = create_result_formatter(
            graph=sample_graph,
            id_to_entity=id_to_entity
        )

        path = [(1, 2, 0), (2, 3, 0)]
        explanation = formatter.generate_explanation(path, sample_graph)

        assert isinstance(explanation, ExplanationPath)
        assert explanation.source_entity == 1
        assert explanation.target_entity == 3
        assert explanation.path_length == 2
        assert explanation.total_confidence > 0
        assert 'USA' in explanation.explanation_text
        assert 'RUS' in explanation.explanation_text

    def test_calculate_confidence_harmonic(self, sample_edges):
        """Test harmonic mean confidence calculation."""
        formatter = create_result_formatter()

        confidence = formatter.calculate_confidence_score(
            edges=sample_edges,
            method="harmonic_mean"
        )

        assert 0.0 <= confidence <= 1.0
        # Harmonic mean should be dominated by lower values
        assert confidence < 0.8  # Lower than max confidence

    def test_calculate_confidence_geometric(self, sample_edges):
        """Test geometric mean confidence calculation."""
        formatter = create_result_formatter()

        confidence = formatter.calculate_confidence_score(
            edges=sample_edges,
            method="geometric_mean"
        )

        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_min(self, sample_edges):
        """Test minimum confidence calculation."""
        formatter = create_result_formatter()

        confidence = formatter.calculate_confidence_score(
            edges=sample_edges,
            method="min"
        )

        # Should be minimum confidence (0.6)
        assert confidence == 0.6

    def test_format_entity(self, sample_graph):
        """Test entity formatting."""
        id_to_entity = {1: 'USA'}
        formatter = create_result_formatter(graph=sample_graph, id_to_entity=id_to_entity)

        entity_data = formatter._format_entity(1)

        assert entity_data['id'] == 1
        # Graph node has 'name' = 'Entity1', id_to_entity would be 'USA'
        # But graph metadata takes precedence
        assert entity_data['name'] == 'Entity1'  # From graph node
        assert 'entity_type' in entity_data

    def test_format_edge(self):
        """Test edge formatting."""
        id_to_entity = {1: 'USA', 2: 'CHN'}
        formatter = create_result_formatter(id_to_entity=id_to_entity)

        edge_data = formatter._format_edge(
            u=1, v=2, key=0,
            data={
                'relation_type': 'diplomatic',
                'confidence': 0.85,
                'timestamp': '2024-01-01T00:00:00Z'
            }
        )

        assert edge_data['source'] == 1
        assert edge_data['target'] == 2
        assert edge_data['source_name'] == 'USA'
        assert edge_data['target_name'] == 'CHN'
        assert edge_data['confidence'] == 0.85

    def test_to_json(self, sample_graph, mock_traversal_result):
        """Test JSON serialization."""
        formatter = create_result_formatter(graph=sample_graph)

        result = formatter.format_query_result(
            query_id='test-002',
            query_type='test',
            query_params={},
            traversal_result=mock_traversal_result
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert 'test-002' in json_str
        assert 'entities' in json_str

    def test_confidence_components(self, mock_traversal_result):
        """Test confidence component calculation."""
        formatter = create_result_formatter()

        components = formatter._calculate_confidence_components(mock_traversal_result)

        assert 'edge_confidence' in components
        assert 'mention_strength' in components
        assert 0.0 <= components['edge_confidence'] <= 1.0


class TestIntegration:
    """Integration tests combining temporal and formatting."""

    def test_full_pipeline(self, sample_edges, sample_graph):
        """Test full result processing pipeline."""
        # 1. Temporal aggregation
        aggregator = create_temporal_aggregator()

        filtered = aggregator.filter_by_time_window(
            edges=sample_edges,
            start_time='2024-01-01T00:00:00Z',
            end_time='2024-01-31T00:00:00Z'
        )

        aggregations = aggregator.aggregate_by_time_period(
            edges=filtered,
            period="daily"
        )

        assert len(aggregations) >= 1

        # 2. Result formatting
        id_to_entity = {1: 'USA', 2: 'CHN', 3: 'RUS', 4: 'EU'}
        formatter = create_result_formatter(graph=sample_graph, id_to_entity=id_to_entity)

        # Mock traversal result
        traversal_result = Mock()
        traversal_result.nodes = {1, 2, 3}
        traversal_result.edges = filtered
        traversal_result.paths = [[(1, 2, 0), (2, 3, 0)]]
        traversal_result.metadata = {}

        result = formatter.format_query_result(
            query_id='integration-001',
            query_type='temporal_analysis',
            query_params={'start': '2024-01-01', 'end': '2024-01-31'},
            traversal_result=traversal_result,
            execution_time_ms=25.0
        )

        assert result.result_count >= 1
        assert len(result.entities) >= 1
        assert result.overall_confidence > 0

    def test_temporal_decay_with_formatting(self, sample_edges):
        """Test combining temporal decay with result formatting."""
        aggregator = create_temporal_aggregator()

        weighted_edges = aggregator.apply_temporal_decay(
            edges=sample_edges,
            reference_time='2024-01-15T00:00:00Z',
            half_life_days=7.0
        )

        # Create weighted edge list
        weighted_edge_list = [(u, v, k, d) for u, v, k, d, w in weighted_edges]

        # Format results
        formatter = create_result_formatter()
        confidence = formatter.calculate_confidence_score(weighted_edge_list)

        assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
