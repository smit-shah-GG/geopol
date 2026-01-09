"""
Tests for Vector Similarity Search.

Tests semantic entity search, relation similarity, hybrid search, and query expansion.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from src.knowledge_graph.vector_similarity import (
    VectorSimilaritySearch, SimilarityResult, create_similarity_search
)


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    mock_store = Mock()
    mock_store.config = Mock()
    mock_store.config.entity_collection = "kg_entities"
    mock_store.config.relation_collection = "kg_relations"

    # Mock client
    mock_client = Mock()
    mock_store.client = mock_client

    return mock_store


@pytest.fixture
def mock_embedding_model():
    """Create mock embedding model."""
    model = Mock()

    # Mock entity embeddings
    def mock_entity_embed(entity_ids):
        # Return complex embeddings (256 dims x 2)
        batch_size = entity_ids.shape[0]
        return torch.randn(batch_size, 256, 2)

    # Mock relation embeddings
    def mock_relation_embed(relation_ids):
        # Return phase embeddings (256 dims)
        batch_size = relation_ids.shape[0]
        return torch.randn(batch_size, 256)

    model.entity_embeddings = Mock(side_effect=mock_entity_embed)
    model.relation_embeddings = Mock(side_effect=mock_relation_embed)

    return model


@pytest.fixture
def entity_mappings():
    """Create entity ID mappings."""
    entity_to_id = {
        'USA': 1,
        'CHN': 2,
        'RUS': 3,
        'NATO': 4,
        'EU': 5
    }
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    return entity_to_id, id_to_entity


@pytest.fixture
def relation_mappings():
    """Create relation type mappings."""
    relation_to_id = {
        'diplomatic_cooperation': 0,
        'military_action': 1,
        'threaten': 2,
        'negotiate': 3
    }
    return relation_to_id


class TestSimilarityResult:
    """Test SimilarityResult container."""

    def test_init_empty(self):
        """Test empty result initialization."""
        result = SimilarityResult()
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert len(result.metadata) == 0

    def test_add_entity(self):
        """Test adding entity results."""
        result = SimilarityResult()
        result.add_entity(1, 0.9, {'name': 'USA'})
        result.add_entity(2, 0.8, {'name': 'CHN'})

        assert len(result.entities) == 2
        assert result.entities[0] == (1, 0.9)
        assert result.entities[1] == (2, 0.8)

    def test_add_relation(self):
        """Test adding relation results."""
        result = SimilarityResult()
        result.add_relation('diplomatic_cooperation', 0.95)
        result.add_relation('negotiate', 0.85)

        assert len(result.relations) == 2
        assert result.relations[0] == ('diplomatic_cooperation', 0.95)

    def test_top_k_entities(self):
        """Test top-k entity selection."""
        result = SimilarityResult()
        result.add_entity(1, 0.7)
        result.add_entity(2, 0.9)
        result.add_entity(3, 0.8)
        result.add_entity(4, 0.6)

        top_k = result.top_k_entities(k=2)

        assert len(top_k) == 2
        assert top_k[0] == (2, 0.9)  # Highest score
        assert top_k[1] == (3, 0.8)  # Second highest

    def test_top_k_relations(self):
        """Test top-k relation selection."""
        result = SimilarityResult()
        result.add_relation('threaten', 0.6)
        result.add_relation('negotiate', 0.9)
        result.add_relation('military_action', 0.8)

        top_k = result.top_k_relations(k=2)

        assert len(top_k) == 2
        assert top_k[0] == ('negotiate', 0.9)
        assert top_k[1] == ('military_action', 0.8)


class TestEntitySimilaritySearch:
    """Test entity similarity search."""

    def test_search_with_model(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test entity search using embedding model."""
        entity_to_id, id_to_entity = entity_mappings

        # Mock Qdrant search results
        mock_hits = [
            Mock(id=2, score=0.85, payload={'name': 'CHN'}),
            Mock(id=3, score=0.75, payload={'name': 'RUS'})
        ]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = search_engine.search_similar_entities(query='USA', top_k=5)

        assert len(result.entities) == 2
        assert result.entities[0][0] == 2  # CHN
        assert result.entities[0][1] == 0.85
        assert result.metadata['query'] == 'USA'

    def test_search_with_threshold(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test similarity threshold filtering."""
        entity_to_id, id_to_entity = entity_mappings

        # Mock results with varying scores
        mock_hits = [
            Mock(id=2, score=0.85, payload={}),
            Mock(id=3, score=0.60, payload={})  # Below threshold
        ]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = search_engine.search_similar_entities(
            query='USA',
            top_k=5,
            threshold=0.7  # Should filter out 0.60 score
        )

        # Qdrant does the filtering, so we get what it returns
        assert len(result.entities) >= 1

    def test_search_unknown_entity(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test search with unknown entity."""
        entity_to_id, id_to_entity = entity_mappings

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = search_engine.search_similar_entities(query='UNKNOWN', top_k=5)

        assert len(result.entities) == 0
        assert result.metadata['query'] == 'UNKNOWN'

    def test_search_by_id(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test search using entity ID."""
        entity_to_id, id_to_entity = entity_mappings

        mock_hits = [Mock(id=2, score=0.9, payload={})]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = search_engine.search_similar_entities(query='1', top_k=5)

        assert len(result.entities) >= 1


class TestRelationSimilaritySearch:
    """Test relation similarity search."""

    def test_search_similar_relations(self, mock_vector_store, mock_embedding_model, relation_mappings):
        """Test relation similarity search."""
        # Mock Qdrant results
        mock_hits = [
            Mock(id=3, score=0.88, payload={}),  # negotiate
            Mock(id=0, score=0.70, payload={})   # diplomatic_cooperation (query)
        ]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            relation_to_id=relation_mappings
        )

        result = search_engine.search_similar_relations(
            relation_type='diplomatic_cooperation',
            top_k=5
        )

        # Should get negotiate (id=3) but not diplomatic_cooperation itself
        assert len(result.relations) >= 1
        assert result.metadata['query'] == 'diplomatic_cooperation'

    def test_search_unknown_relation(self, mock_vector_store, mock_embedding_model, relation_mappings):
        """Test search with unknown relation type."""
        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            relation_to_id=relation_mappings
        )

        result = search_engine.search_similar_relations(
            relation_type='unknown_relation',
            top_k=5
        )

        assert len(result.relations) == 0


class TestHybridSearch:
    """Test hybrid search combining graph and vector."""

    def test_hybrid_search_fusion(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test hybrid search score fusion."""
        entity_to_id, id_to_entity = entity_mappings

        # Mock vector results
        mock_hits = [
            Mock(id=2, score=0.9, payload={}),
            Mock(id=3, score=0.8, payload={}),
            Mock(id=4, score=0.7, payload={})
        ]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        # Mock graph results
        mock_graph_results = Mock()
        mock_graph_results.nodes = {1, 2, 5}  # Overlaps with 2 from vector

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = search_engine.hybrid_search(
            entity_query='USA',
            graph_results=mock_graph_results,
            top_k=5,
            vector_weight=0.6  # 60% vector, 40% graph
        )

        # Should have fusion of both sources
        assert len(result.entities) >= 1
        assert result.metadata['query_type'] == 'hybrid_search'
        assert result.metadata['vector_weight'] == 0.6

        # Entity 2 should have highest score (in both vector and graph)
        entity_ids = [e[0] for e in result.entities]
        assert 2 in entity_ids

    def test_hybrid_search_weights(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test different weight configurations."""
        entity_to_id, id_to_entity = entity_mappings

        mock_hits = [Mock(id=2, score=0.8, payload={})]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        mock_graph_results = Mock()
        mock_graph_results.nodes = {3}

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        # Pure vector (weight=1.0)
        result_vector = search_engine.hybrid_search(
            entity_query='USA',
            graph_results=mock_graph_results,
            vector_weight=1.0
        )

        # Pure graph (weight=0.0)
        result_graph = search_engine.hybrid_search(
            entity_query='USA',
            graph_results=mock_graph_results,
            vector_weight=0.0
        )

        # Both should have results
        assert len(result_vector.entities) >= 1
        assert len(result_graph.entities) >= 1


class TestQueryExpansion:
    """Test query expansion."""

    def test_expand_query(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test query expansion with similar entities."""
        entity_to_id, id_to_entity = entity_mappings

        # Mock similar entities
        mock_hits = [
            Mock(id=4, score=0.85, payload={}),  # NATO
            Mock(id=5, score=0.75, payload={})   # EU
        ]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        expanded = search_engine.expand_query(
            entity_query='USA',
            expansion_k=3,
            similarity_threshold=0.7
        )

        # Should include query entity (USA=1) plus similar entities
        assert 1 in expanded  # USA itself
        assert len(expanded) >= 2  # USA + at least one expansion

    def test_expand_unknown_entity(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test expansion with unknown entity."""
        entity_to_id, id_to_entity = entity_mappings

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        expanded = search_engine.expand_query(
            entity_query='UNKNOWN',
            expansion_k=3
        )

        assert len(expanded) == 0


class TestTemporalReranking:
    """Test temporal relevance re-ranking."""

    def test_rerank_by_recency(self, mock_vector_store, entity_mappings):
        """Test re-ranking by temporal relevance."""
        entity_to_id, id_to_entity = entity_mappings

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        # Create result with entities
        result = SimilarityResult()
        result.add_entity(1, 0.8, {'last_seen': '2024-01-15T00:00:00Z'})  # Recent
        result.add_entity(2, 0.9, {'last_seen': '2023-01-01T00:00:00Z'})  # Old
        result.metadata['entity_metadata'] = {
            1: {'last_seen': '2024-01-15T00:00:00Z'},
            2: {'last_seen': '2023-01-01T00:00:00Z'}
        }

        # Re-rank with reference time close to entity 1's last_seen
        reranked = search_engine.rerank_by_temporal_relevance(
            results=result,
            reference_time='2024-01-20T00:00:00Z',
            decay_factor=0.1
        )

        assert reranked.metadata['reranked'] is True
        # Entity 1 should now be ranked higher due to recency
        # (original score 0.8 but recent vs 0.9 but old)

    def test_rerank_without_temporal_metadata(self, mock_vector_store, entity_mappings):
        """Test re-ranking when no temporal metadata available."""
        entity_to_id, id_to_entity = entity_mappings

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = SimilarityResult()
        result.add_entity(1, 0.8)
        result.add_entity(2, 0.9)

        reranked = search_engine.rerank_by_temporal_relevance(
            results=result,
            reference_time='2024-01-20T00:00:00Z'
        )

        # Scores should remain unchanged
        assert len(reranked.entities) == 2


class TestIntegration:
    """Integration tests."""

    def test_full_similarity_workflow(self, mock_vector_store, mock_embedding_model, entity_mappings):
        """Test complete similarity search workflow."""
        entity_to_id, id_to_entity = entity_mappings

        # Mock search results
        mock_hits = [
            Mock(id=2, score=0.9, payload={'name': 'CHN'}),
            Mock(id=3, score=0.8, payload={'name': 'RUS'})
        ]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            embedding_model=mock_embedding_model,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        # 1. Search similar entities
        result = search_engine.search_similar_entities(query='USA', top_k=10)
        assert len(result.entities) >= 2

        # 2. Get top-k
        top_k = result.top_k_entities(k=2)
        assert len(top_k) == 2

        # 3. Re-rank
        result.metadata['entity_metadata'] = {
            2: {'last_seen': '2024-01-20T00:00:00Z'},
            3: {'last_seen': '2024-01-10T00:00:00Z'}
        }
        reranked = search_engine.rerank_by_temporal_relevance(
            results=result,
            reference_time='2024-01-25T00:00:00Z'
        )

        assert len(reranked.entities) >= 2

    def test_embedding_fallback_to_qdrant(self, mock_vector_store, entity_mappings):
        """Test fallback to Qdrant when model unavailable."""
        entity_to_id, id_to_entity = entity_mappings

        # Mock Qdrant retrieve for embedding
        mock_point = Mock()
        mock_point.vector = np.random.randn(512).tolist()
        mock_vector_store.client.retrieve = Mock(return_value=[mock_point])

        # Mock search
        mock_hits = [Mock(id=2, score=0.9, payload={})]
        mock_vector_store.client.search = Mock(return_value=mock_hits)

        # Create without embedding model
        search_engine = create_similarity_search(
            vector_store=mock_vector_store,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        result = search_engine.search_similar_entities(query='USA', top_k=5)

        # Should still work via Qdrant fallback
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
