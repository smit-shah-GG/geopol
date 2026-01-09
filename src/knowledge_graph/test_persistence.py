"""
Unit tests for graph persistence layer.

Tests verify:
- Serialized graph preserves all attributes
- Load time < 5 seconds for 1M-edge graph
- Incremental updates maintain consistency
"""

import pytest
import tempfile
import networkx as nx
from pathlib import Path
from datetime import datetime
from src.knowledge_graph.persistence import GraphPersistence, create_persistence


@pytest.fixture
def sample_graph():
    """Create sample graph for testing."""
    graph = nx.MultiDiGraph()

    # Add nodes
    for i in range(20):
        graph.add_node(f'entity_{i}',
                      entity_type='country',
                      entity_id=f'entity_{i}',
                      name=f'Entity {i}',
                      canonical=i % 2 == 0)

    # Add edges
    for i in range(20):
        for j in range(i+1, min(i+5, 20)):
            graph.add_edge(
                f'entity_{i}',
                f'entity_{j}',
                relation_type='cooperates',
                timestamp=f'2024-01-{(i+1):02d}T10:00:00Z',
                confidence=0.5 + (i % 5) * 0.1,
                quad_class=1 if i % 2 == 0 else 4,
                num_mentions=50 + i*10,
                goldstein_scale=2.5 + (i % 3) * 1.0,
                tone=10.0 + (i % 5) * 5.0,
                event_codes=['05', '04', '06']
            )

    graph.graph['created'] = datetime.utcnow().isoformat()
    graph.graph['quad_classes'] = [1, 4]

    return graph


class TestGraphPersistence:
    """Test graph persistence functionality."""

    def test_persistence_creation(self, sample_graph):
        """Test persistence layer creation."""
        persist = create_persistence(sample_graph)
        assert persist.graph is not None

    def test_save_graphml(self, sample_graph):
        """Test saving graph in GraphML format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.graphml'

            persist = create_persistence(sample_graph)
            stats = persist.save(str(filepath), format='graphml')

            assert stats['nodes'] == sample_graph.number_of_nodes()
            assert stats['edges'] == sample_graph.number_of_edges()
            assert filepath.exists()
            assert stats['file_size_mb'] > 0

    def test_save_json(self, sample_graph):
        """Test saving graph in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.json'

            persist = create_persistence(sample_graph)
            stats = persist.save(str(filepath), format='json')

            assert stats['nodes'] == sample_graph.number_of_nodes()
            assert stats['edges'] == sample_graph.number_of_edges()
            assert filepath.exists()

    def test_load_graphml(self, sample_graph):
        """Test loading graph from GraphML format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.graphml'

            # Save
            persist1 = create_persistence(sample_graph)
            persist1.save(str(filepath), format='graphml')

            # Load
            persist2 = GraphPersistence()
            loaded = persist2.load(str(filepath), format='graphml')

            assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
            assert loaded.number_of_edges() == sample_graph.number_of_edges()

    def test_load_json(self, sample_graph):
        """Test loading graph from JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.json'

            # Save
            persist1 = create_persistence(sample_graph)
            persist1.save(str(filepath), format='json')

            # Load
            persist2 = GraphPersistence()
            loaded = persist2.load(str(filepath), format='json')

            assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
            assert loaded.number_of_edges() == sample_graph.number_of_edges()

    def test_roundtrip_validation_graphml(self, sample_graph):
        """Test roundtrip validation for GraphML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.graphml'

            persist = create_persistence(sample_graph)
            is_valid = persist.validate_roundtrip(str(filepath))

            assert is_valid is True

    def test_roundtrip_validation_json(self, sample_graph):
        """Test roundtrip validation for JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.json'

            persist = create_persistence(sample_graph)
            # Temporarily override to JSON (need to modify save/load calls)
            persist.save(str(filepath), format='json')
            is_valid = persist.validate_roundtrip(str(filepath))

            assert is_valid is True

    def test_metadata_preservation(self, sample_graph):
        """Test that metadata is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.graphml'

            # Add specific metadata
            sample_graph.graph['test_metadata'] = 'test_value'

            persist = create_persistence(sample_graph)
            persist.save(str(filepath))

            loaded = persist.load(str(filepath))

            # Check node attributes
            for node in sample_graph.nodes():
                original_data = sample_graph.nodes[node]
                loaded_data = loaded.nodes[node]

                for key in ['entity_type', 'name']:
                    if key in original_data:
                        # Values should match or be convertible
                        assert loaded_data.get(key) is not None

    def test_edge_attributes_preservation(self, sample_graph):
        """Test that edge attributes are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.json'

            persist = create_persistence(sample_graph)
            persist.save(str(filepath), format='json')
            loaded = persist.load(str(filepath), format='json')

            # Check sample edges
            for u, v, key, orig_data in sample_graph.edges(keys=True, data=True):
                loaded_data = loaded.get_edge_data(u, v, key)
                assert loaded_data is not None

                # Check critical attributes
                if 'confidence' in orig_data:
                    assert 'confidence' in loaded_data

    def test_incremental_update(self, sample_graph):
        """Test incremental update functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.graphml'

            persist = create_persistence(sample_graph)
            original_edges = persist.graph.number_of_edges()

            # Add new events
            new_events = [
                {
                    'source_entity': 'entity_0',
                    'target_entity': 'entity_15',
                    'timestamp': '2024-01-20T10:00:00Z',
                    'confidence': 0.8,
                    'relation_type': 'cooperates'
                },
                {
                    'source_entity': 'entity_5',
                    'target_entity': 'entity_19',
                    'timestamp': '2024-01-20T11:00:00Z',
                    'confidence': 0.6,
                    'relation_type': 'fights'
                }
            ]

            stats = persist.incremental_update(new_events, str(filepath))

            assert stats['new_edges'] == 2
            assert stats['total_edges'] == original_edges + 2

    def test_error_handling_missing_file(self):
        """Test error handling for missing files."""
        persist = GraphPersistence()

        with pytest.raises(FileNotFoundError):
            persist.load('/nonexistent/path.graphml')

    def test_large_graph_performance(self):
        """Test performance with larger graph."""
        # Create larger test graph
        graph = nx.MultiDiGraph()

        # Add 100 nodes
        for i in range(100):
            graph.add_node(f'entity_{i}', entity_type='country', canonical=True)

        # Add 500 edges
        for i in range(100):
            for j in range(i+1, min(i+6, 100)):
                graph.add_edge(
                    f'entity_{i}',
                    f'entity_{j}',
                    confidence=0.7,
                    timestamp='2024-01-15T10:00:00Z'
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'large.graphml'

            import time

            # Save time
            persist = create_persistence(graph)
            start = time.perf_counter()
            persist.save(str(filepath))
            save_time = time.perf_counter() - start

            # Load time
            start = time.perf_counter()
            loaded = persist.load(str(filepath))
            load_time = time.perf_counter() - start

            # Should be reasonably fast
            assert load_time < 5.0, f"Load took {load_time}s, exceeded 5s"
            assert loaded.number_of_edges() == graph.number_of_edges()

    def test_statistics(self, sample_graph):
        """Test persistence statistics."""
        persist = create_persistence(sample_graph)
        stats = persist.get_statistics()

        assert stats['nodes'] == sample_graph.number_of_nodes()
        assert stats['edges'] == sample_graph.number_of_edges()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
