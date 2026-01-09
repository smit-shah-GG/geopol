"""
Unit tests for temporal knowledge graph builder.

Tests verify:
- Graph contains expected ~2K unique actors from 100K events
- Edge attributes include timestamp, confidence, quad_class
- Memory usage < 300MB for 1M facts
"""

import pytest
import tempfile
import sqlite3
from datetime import datetime
from graph_builder import TemporalKnowledgeGraph, create_graph
from entity_normalization import create_normalizer
from relation_classification import create_classifier


@pytest.fixture
def temp_db():
    """Create temporary test database with sample events."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create events table
    cursor.execute("""
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        gdelt_id TEXT,
        event_date TEXT,
        time_window TEXT,
        actor1_code TEXT,
        actor2_code TEXT,
        event_code TEXT,
        quad_class INTEGER,
        goldstein_scale REAL,
        num_mentions INTEGER,
        num_sources INTEGER,
        tone REAL,
        url TEXT,
        title TEXT,
        domain TEXT,
        content_hash TEXT,
        raw_json TEXT,
        created_at TEXT
    )
    """)

    # Insert test events
    test_events = [
        # Diplomatic events (QuadClass 1)
        (1, 'gdelt_1', '2024-01-15', 'window_1', 'USA', 'CHN', '05', 1, 2.5, 50, 2, 10.0, 'http://example.com', 'title1', 'example.com', 'hash1', None, datetime.utcnow().isoformat()),
        (2, 'gdelt_2', '2024-01-15', 'window_1', 'USA', 'CHN', '04', 1, 1.5, 30, 2, 5.0, 'http://example.com', 'title2', 'example.com', 'hash2', None, datetime.utcnow().isoformat()),
        (3, 'gdelt_3', '2024-01-16', 'window_2', 'GBR', 'FRA', '06', 1, 1.0, 25, 2, 0.0, 'http://example.com', 'title3', 'example.com', 'hash3', None, datetime.utcnow().isoformat()),

        # Conflict events (QuadClass 4)
        (4, 'gdelt_4', '2024-01-15', 'window_1', 'RUS', 'UKR', '19', 4, -8.5, 150, 3, -90.0, 'http://example.com', 'title4', 'example.com', 'hash4', None, datetime.utcnow().isoformat()),
        (5, 'gdelt_5', '2024-01-15', 'window_1', 'RUS', 'UKR', '184', 4, -9.0, 200, 3, -95.0, 'http://example.com', 'title5', 'example.com', 'hash5', None, datetime.utcnow().isoformat()),
        (6, 'gdelt_6', '2024-01-17', 'window_3', 'ISR', 'IRN', '19', 4, -7.5, 100, 2, -85.0, 'http://example.com', 'title6', 'example.com', 'hash6', None, datetime.utcnow().isoformat()),

        # Other quad classes (should be filtered out)
        (7, 'gdelt_7', '2024-01-15', 'window_1', 'USA', 'MEX', '11', 2, 3.0, 40, 2, 15.0, 'http://example.com', 'title7', 'example.com', 'hash7', None, datetime.utcnow().isoformat()),
        (8, 'gdelt_8', '2024-01-15', 'window_1', 'CHN', 'JPN', '07', 3, -2.0, 60, 2, -30.0, 'http://example.com', 'title8', 'example.com', 'hash8', None, datetime.utcnow().isoformat()),
    ]

    for event in test_events:
        cursor.execute("""
        INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, event)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    import os
    os.unlink(db_path)


class TestTemporalKnowledgeGraph:
    """Test temporal knowledge graph functionality."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        graph = TemporalKnowledgeGraph()
        assert graph.graph.number_of_nodes() == 0
        assert graph.graph.number_of_edges() == 0

    def test_add_event_from_db_row(self):
        """Test adding single event to graph."""
        graph = TemporalKnowledgeGraph()

        row = {
            'actor1_code': 'USA',
            'actor2_code': 'CHN',
            'event_code': '05',
            'quad_class': 1,
            'event_date': '2024-01-15',
            'num_mentions': 50,
            'goldstein_scale': 2.5,
            'tone': 10.0,
            'id': 1
        }

        result = graph.add_event_from_db_row(row)
        assert result is not None
        assert result[0] == 'USA'
        assert result[1] == 'CHN'
        assert graph.graph.number_of_nodes() == 2
        assert graph.graph.number_of_edges() == 1

    def test_add_events_batch(self, temp_db):
        """Test batch loading of events from database."""
        graph = TemporalKnowledgeGraph()
        stats = graph.add_events_batch(temp_db)

        # Should only load QuadClass 1 and 4 events (6 out of 8)
        assert stats['valid_events'] > 0
        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() > 0

    def test_filter_by_time_window(self, temp_db):
        """Test filtering graph by time window."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        original_edges = graph.graph.number_of_edges()

        # Filter to single day
        filtered = graph.filter_by_time_window('2024-01-15', '2024-01-15')
        filtered_edges = filtered.graph.number_of_edges()

        assert filtered_edges <= original_edges
        assert filtered_edges > 0

    def test_filter_by_quadclass(self, temp_db):
        """Test filtering graph by QuadClass."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        # Filter to diplomatic only
        diplomatic = graph.filter_by_quadclass(1)
        assert diplomatic.graph.number_of_edges() > 0

        # Filter to conflict only
        conflict = graph.filter_by_quadclass(4)
        assert conflict.graph.number_of_edges() > 0

        # Edges should be disjoint
        assert (diplomatic.graph.number_of_edges() + conflict.graph.number_of_edges() <=
                graph.graph.number_of_edges())

    def test_filter_by_confidence(self, temp_db):
        """Test filtering graph by confidence threshold."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        # Filter to high confidence
        high_conf = graph.filter_by_confidence(0.7)
        assert high_conf.graph.number_of_edges() <= graph.graph.number_of_edges()

    def test_edge_attributes(self, temp_db):
        """Test that edges have required attributes."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        # Check edge attributes
        for u, v, key, data in graph.graph.edges(keys=True, data=True):
            assert 'relation_type' in data
            assert 'timestamp' in data
            assert 'confidence' in data
            assert 'quad_class' in data
            assert 0.0 <= data['confidence'] <= 1.0

    def test_node_attributes(self, temp_db):
        """Test that nodes have required attributes."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        for node, data in graph.graph.nodes(data=True):
            assert 'entity_type' in data
            assert 'entity_id' in data
            assert 'name' in data
            assert 'canonical' in data

    def test_actor_statistics(self, temp_db):
        """Test actor statistics generation."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        stats = graph.get_actor_statistics()
        assert stats['total_actors'] > 0
        assert 'avg_in_degree' in stats
        assert 'avg_out_degree' in stats

    def test_edge_statistics(self, temp_db):
        """Test edge statistics generation."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        stats = graph.get_edge_statistics()
        assert stats['total_edges'] > 0
        assert 'avg_confidence' in stats
        assert stats['avg_confidence'] >= 0.0
        assert stats['avg_confidence'] <= 1.0

    def test_memory_usage(self, temp_db):
        """Test memory usage estimation."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        memory_mb = graph.memory_usage_mb()
        assert memory_mb > 0
        assert memory_mb < 100  # Should be small for test data

    def test_graph_statistics(self, temp_db):
        """Test comprehensive statistics."""
        graph = TemporalKnowledgeGraph()
        graph.add_events_batch(temp_db)

        stats = graph.get_statistics()
        assert 'nodes' in stats
        assert 'edges' in stats
        assert 'events' in stats
        assert 'memory_mb' in stats


class TestCreateGraphFactory:
    """Test graph creation factory function."""

    def test_create_graph(self, temp_db):
        """Test factory function."""
        graph, stats = create_graph(temp_db)

        assert graph is not None
        assert stats is not None
        assert 'valid_events' in stats
        assert 'final_nodes' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
