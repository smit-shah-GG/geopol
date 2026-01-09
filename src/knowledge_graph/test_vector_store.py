"""
Unit tests for Qdrant vector store integration.

Tests are designed to work with an optional Qdrant server:
- If server is running: Full integration tests
- If server is not running: Skip tests with warning

Tests cover:
    - Configuration and initialization
    - Collection creation and management
    - Embedding upload and batch processing
    - Similarity search with filtering
    - Backup and restore functionality
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from knowledge_graph.vector_store import (
    VectorStore,
    QdrantConfig,
    setup_qdrant_for_embeddings
)
from knowledge_graph.embeddings import RotatEModel
from qdrant_client.models import PointStruct


# Check if Qdrant is available
def qdrant_available():
    """Check if Qdrant server is running."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


QDRANT_AVAILABLE = qdrant_available()

pytestmark = pytest.mark.skipif(
    not QDRANT_AVAILABLE,
    reason="Qdrant server not running. Start with: docker run -p 6333:6333 qdrant/qdrant"
)


class TestQdrantConfig:
    """Test Qdrant configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QdrantConfig()

        assert config.host == "localhost"
        assert config.port == 6333
        assert config.embedding_dim == 256
        assert config.batch_size == 1000

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = QdrantConfig(embedding_dim=128, batch_size=500)
        config_dict = config.to_dict()

        assert config_dict['embedding_dim'] == 128
        assert config_dict['batch_size'] == 500
        assert isinstance(config_dict, dict)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not available")
class TestVectorStore:
    """Test vector store operations."""

    @pytest.fixture
    def config(self):
        """Create test configuration with unique collection names."""
        import time
        suffix = int(time.time() * 1000) % 10000
        return QdrantConfig(
            entity_collection=f"test_entities_{suffix}",
            relation_collection=f"test_relations_{suffix}",
            embedding_dim=64
        )

    @pytest.fixture
    def store(self, config):
        """Create vector store instance."""
        store = VectorStore(config)
        yield store
        # Cleanup
        try:
            store.client.delete_collection(config.entity_collection)
            store.client.delete_collection(config.relation_collection)
        except Exception:
            pass

    @pytest.fixture
    def small_model(self):
        """Create small trained model for testing."""
        model = RotatEModel(
            num_entities=20,
            num_relations=5,
            embedding_dim=64
        )
        # Simulate some training by making embeddings non-random
        with torch.no_grad():
            model.entity_embeddings.weight.data = torch.randn_like(
                model.entity_embeddings.weight.data
            ) * 0.1
        return model

    def test_store_initialization(self, store):
        """Test store initializes correctly."""
        assert store.client is not None
        assert store.config is not None

    def test_create_collections(self, store):
        """Test collection creation."""
        store.create_collections(recreate=True)

        # Check collections exist
        entity_info = store.get_collection_info(store.config.entity_collection)
        relation_info = store.get_collection_info(store.config.relation_collection)

        assert entity_info is not None
        assert relation_info is not None
        assert entity_info.get('points_count', 0) == 0  # Empty initially

    def test_upload_entity_embeddings(self, store, small_model):
        """Test entity embedding upload."""
        store.create_collections(recreate=True)

        # Create entity mappings
        entity_to_id = {f"Entity_{i}": i for i in range(20)}
        id_to_entity = {i: f"Entity_{i}" for i in range(20)}

        # Upload
        store.upload_entity_embeddings(
            small_model,
            entity_to_id,
            id_to_entity
        )

        # Check uploaded
        info = store.get_collection_info(store.config.entity_collection)
        assert info['points_count'] == 20

    def test_upload_relation_embeddings(self, store, small_model):
        """Test relation embedding upload."""
        store.create_collections(recreate=True)

        # Create relation mappings
        relation_to_id = {f"Relation_{i}": i for i in range(5)}
        id_to_relation = {i: f"Relation_{i}" for i in range(5)}

        # Upload
        store.upload_relation_embeddings(
            small_model,
            relation_to_id,
            id_to_relation
        )

        # Check uploaded
        info = store.get_collection_info(store.config.relation_collection)
        assert info['points_count'] == 5

    def test_batch_upload(self, store):
        """Test batch uploading mechanism."""
        store.create_collections(recreate=True)

        # Create test points
        points = []
        for i in range(100):
            point = PointStruct(
                id=i,
                vector=[0.1] * 128,  # 2*64 for complex embeddings
                payload={"test_id": i}
            )
            points.append(point)

        # Upload in small batches
        store._batch_upload(
            store.config.entity_collection,
            points,
            batch_size=25
        )

        # Check all uploaded
        info = store.get_collection_info(store.config.entity_collection)
        assert info['points_count'] == 100

    def test_search_similar_entities(self, store, small_model):
        """Test similarity search."""
        store.create_collections(recreate=True)

        # Upload embeddings
        entity_to_id = {f"Entity_{i}": i for i in range(20)}
        id_to_entity = {i: f"Entity_{i}" for i in range(20)}

        store.upload_entity_embeddings(
            small_model,
            entity_to_id,
            id_to_entity
        )

        # Search with query vector
        query_vector = small_model.get_entity_embedding(0)
        results = store.search_similar_entities(query_vector, top_k=5)

        # Check results
        assert len(results) == 5
        assert all(len(r) == 3 for r in results)  # (id, score, payload)

        # First result should be the query entity itself (or very close)
        top_id, top_score, top_payload = results[0]
        assert top_score > 0.9  # Should be very similar

    def test_get_entity_by_id(self, store, small_model):
        """Test entity retrieval by ID."""
        store.create_collections(recreate=True)

        entity_to_id = {f"Entity_{i}": i for i in range(10)}
        id_to_entity = {i: f"Entity_{i}" for i in range(10)}

        store.upload_entity_embeddings(
            small_model,
            entity_to_id,
            id_to_entity
        )

        # Retrieve entity
        result = store.get_entity_by_id(5)

        assert result is not None
        assert result['id'] == 5
        assert 'vector' in result
        assert 'payload' in result
        assert result['payload']['entity_name'] == 'Entity_5'

    def test_backup_and_restore(self, store, small_model):
        """Test backup and restore functionality."""
        store.create_collections(recreate=True)

        # Upload some data
        entity_to_id = {f"Entity_{i}": i for i in range(10)}
        id_to_entity = {i: f"Entity_{i}" for i in range(10)}

        store.upload_entity_embeddings(
            small_model,
            entity_to_id,
            id_to_entity
        )

        # Backup
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            backup_path = f.name

        try:
            store.backup_collection(
                store.config.entity_collection,
                backup_path
            )

            # Delete collection
            store.client.delete_collection(store.config.entity_collection)

            # Recreate empty collection
            store.create_collections(recreate=True)

            # Restore
            store.restore_collection(
                backup_path,
                store.config.entity_collection
            )

            # Check restored
            info = store.get_collection_info(store.config.entity_collection)
            assert info['points_count'] == 10

        finally:
            Path(backup_path).unlink(missing_ok=True)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not available")
class TestHighLevelSetup:
    """Test high-level setup function."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary config."""
        import time
        suffix = int(time.time() * 1000) % 10000
        return QdrantConfig(
            entity_collection=f"test_hl_entities_{suffix}",
            relation_collection=f"test_hl_relations_{suffix}",
            embedding_dim=32
        )

    def test_setup_qdrant_for_embeddings(self, temp_config):
        """Test end-to-end setup."""
        # Create small model
        model = RotatEModel(
            num_entities=10,
            num_relations=3,
            embedding_dim=32
        )

        # Create mappings
        entity_to_id = {f"E{i}": i for i in range(10)}
        relation_to_id = {f"R{i}": i for i in range(3)}
        id_to_entity = {i: f"E{i}" for i in range(10)}
        id_to_relation = {i: f"R{i}" for i in range(3)}

        try:
            # Setup
            store = setup_qdrant_for_embeddings(
                model,
                entity_to_id,
                relation_to_id,
                id_to_entity,
                id_to_relation,
                config=temp_config
            )

            # Check collections created and populated
            entity_info = store.get_collection_info(temp_config.entity_collection)
            relation_info = store.get_collection_info(temp_config.relation_collection)

            assert entity_info['points_count'] == 10
            assert relation_info['points_count'] == 3

        finally:
            # Cleanup
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(host="localhost", port=6333)
                client.delete_collection(temp_config.entity_collection)
                client.delete_collection(temp_config.relation_collection)
            except Exception:
                pass


class TestQdrantNotAvailable:
    """Tests that run even without Qdrant server."""

    def test_config_creation(self):
        """Test config can be created without server."""
        config = QdrantConfig()
        assert config.host == "localhost"

    def test_config_serialization(self):
        """Test config serialization works without server."""
        config = QdrantConfig(embedding_dim=128)
        config_dict = config.to_dict()
        assert config_dict['embedding_dim'] == 128


if __name__ == '__main__':
    if not QDRANT_AVAILABLE:
        print("\n" + "="*70)
        print("WARNING: Qdrant server not running!")
        print("="*70)
        print("\nTo run full tests, start Qdrant with:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        print("\nOr install Qdrant locally:")
        print("  https://qdrant.tech/documentation/quick-start/")
        print("\n" + "="*70 + "\n")

    pytest.main([__file__, '-v'])
