"""
Vector Similarity Search for Temporal Knowledge Graphs.

Implements semantic search using Qdrant embeddings:
- Entity search by semantic similarity
- Relation type similarity search
- Hybrid search combining graph and vectors
- Query expansion using similar entities
- Re-ranking based on temporal relevance
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SimilarityResult:
    """Container for similarity search results."""

    def __init__(self):
        """Initialize empty result."""
        self.entities: List[Tuple[int, float]] = []  # (entity_id, similarity_score)
        self.relations: List[Tuple[str, float]] = []  # (relation_type, similarity_score)
        self.metadata: Dict[str, Any] = {}

    def add_entity(self, entity_id: int, score: float, metadata: Optional[Dict] = None):
        """Add entity result."""
        self.entities.append((entity_id, score))
        if metadata:
            if 'entity_metadata' not in self.metadata:
                self.metadata['entity_metadata'] = {}
            self.metadata['entity_metadata'][entity_id] = metadata

    def add_relation(self, relation_type: str, score: float):
        """Add relation result."""
        self.relations.append((relation_type, score))

    def top_k_entities(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k most similar entities."""
        sorted_entities = sorted(self.entities, key=lambda x: x[1], reverse=True)
        return sorted_entities[:k]

    def top_k_relations(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k most similar relations."""
        sorted_relations = sorted(self.relations, key=lambda x: x[1], reverse=True)
        return sorted_relations[:k]


class VectorSimilaritySearch:
    """Vector similarity search engine using Qdrant embeddings."""

    def __init__(
        self,
        vector_store,
        embedding_model=None,
        temporal_model=None,
        entity_to_id: Optional[Dict[str, int]] = None,
        id_to_entity: Optional[Dict[int, str]] = None,
        relation_to_id: Optional[Dict[str, int]] = None
    ):
        """Initialize vector similarity search.

        Args:
            vector_store: VectorStore instance for Qdrant access
            embedding_model: Base embedding model (RotatEModel)
            temporal_model: Temporal embedding model (TemporalRotatEModel)
            entity_to_id: Mapping from entity names to IDs
            id_to_entity: Mapping from entity IDs to names
            relation_to_id: Mapping from relation types to IDs
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.temporal_model = temporal_model
        self.entity_to_id = entity_to_id or {}
        self.id_to_entity = id_to_entity or {}
        self.relation_to_id = relation_to_id or {}

        # Reverse mapping for relations
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

    def search_similar_entities(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
        time_filter: Optional[Tuple[str, str]] = None,
        entity_type_filter: Optional[str] = None
    ) -> SimilarityResult:
        """Search for entities similar to query.

        Args:
            query: Entity name or ID to search for
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            time_filter: Optional (start, end) time window
            entity_type_filter: Optional entity type filter

        Returns:
            SimilarityResult with similar entities
        """
        result = SimilarityResult()
        result.metadata['query'] = query
        result.metadata['query_type'] = 'entity_similarity'

        # Get query entity ID
        if query in self.entity_to_id:
            query_id = self.entity_to_id[query]
        elif query.isdigit():
            query_id = int(query)
        else:
            logger.warning(f"Query entity '{query}' not found")
            return result

        # Get query embedding
        query_embedding = self._get_entity_embedding(query_id)
        if query_embedding is None:
            logger.warning(f"No embedding found for entity {query_id}")
            return result

        # Search in Qdrant
        try:
            # Build filter
            qdrant_filter = self._build_entity_filter(time_filter, entity_type_filter)

            # Search
            search_results = self.vector_store.client.search(
                collection_name=self.vector_store.config.entity_collection,
                query_vector=query_embedding.tolist(),
                limit=top_k + 1,  # +1 to exclude query itself
                score_threshold=threshold,
                query_filter=qdrant_filter if qdrant_filter else None
            )

            # Process results
            for hit in search_results:
                entity_id = hit.id
                similarity = hit.score

                # Skip query entity itself
                if entity_id == query_id:
                    continue

                result.add_entity(entity_id, similarity, hit.payload)

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")

        return result

    def search_similar_relations(
        self,
        relation_type: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> SimilarityResult:
        """Search for relations similar to given relation type.

        Args:
            relation_type: Relation type to search for
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            SimilarityResult with similar relations
        """
        result = SimilarityResult()
        result.metadata['query'] = relation_type
        result.metadata['query_type'] = 'relation_similarity'

        # Get relation ID
        if relation_type not in self.relation_to_id:
            logger.warning(f"Relation type '{relation_type}' not found")
            return result

        relation_id = self.relation_to_id[relation_type]

        # Get relation embedding
        relation_embedding = self._get_relation_embedding(relation_id)
        if relation_embedding is None:
            logger.warning(f"No embedding found for relation {relation_id}")
            return result

        # Search in Qdrant
        try:
            search_results = self.vector_store.client.search(
                collection_name=self.vector_store.config.relation_collection,
                query_vector=relation_embedding.tolist(),
                limit=top_k + 1,  # +1 to exclude query itself
                score_threshold=threshold
            )

            # Process results
            for hit in search_results:
                rel_id = hit.id
                similarity = hit.score

                # Skip query relation itself
                if rel_id == relation_id:
                    continue

                # Map back to relation type
                if rel_id in self.id_to_relation:
                    result.add_relation(self.id_to_relation[rel_id], similarity)

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")

        return result

    def hybrid_search(
        self,
        entity_query: str,
        graph_results,
        top_k: int = 20,
        vector_weight: float = 0.5,
        time_filter: Optional[Tuple[str, str]] = None
    ) -> SimilarityResult:
        """Hybrid search combining graph traversal and vector similarity.

        Args:
            entity_query: Entity to search around
            graph_results: Results from graph traversal (TraversalResult)
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            time_filter: Optional time window

        Returns:
            SimilarityResult with fused results
        """
        result = SimilarityResult()
        result.metadata['query'] = entity_query
        result.metadata['query_type'] = 'hybrid_search'
        result.metadata['vector_weight'] = vector_weight

        # Get vector similarity results
        vector_results = self.search_similar_entities(
            query=entity_query,
            top_k=top_k * 2,  # Get more for fusion
            time_filter=time_filter
        )

        # Get entities from graph results
        graph_entities = graph_results.nodes if hasattr(graph_results, 'nodes') else set()

        # Fusion: combine scores
        entity_scores = {}

        # Add vector results with weight
        for entity_id, sim_score in vector_results.entities:
            entity_scores[entity_id] = vector_weight * sim_score

        # Add graph results with weight
        graph_weight = 1.0 - vector_weight
        for entity_id in graph_entities:
            if entity_id in entity_scores:
                entity_scores[entity_id] += graph_weight  # Presence bonus
            else:
                entity_scores[entity_id] = graph_weight

        # Sort and take top-k
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        for entity_id, score in sorted_entities[:top_k]:
            result.add_entity(entity_id, score)

        result.metadata['fusion_count'] = len(entity_scores)

        return result

    def expand_query(
        self,
        entity_query: str,
        expansion_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> Set[int]:
        """Expand query with similar entities.

        Args:
            entity_query: Entity to expand
            expansion_k: Number of similar entities to add
            similarity_threshold: Minimum similarity for expansion

        Returns:
            Set of entity IDs including query and expansions
        """
        expanded = set()

        # Add query entity
        if entity_query in self.entity_to_id:
            query_id = self.entity_to_id[entity_query]
            expanded.add(query_id)

            # Find similar entities
            similar = self.search_similar_entities(
                query=entity_query,
                top_k=expansion_k,
                threshold=similarity_threshold
            )

            # Add similar entities
            for entity_id, score in similar.entities:
                expanded.add(entity_id)

        return expanded

    def rerank_by_temporal_relevance(
        self,
        results: SimilarityResult,
        reference_time: str,
        decay_factor: float = 0.1
    ) -> SimilarityResult:
        """Re-rank results based on temporal relevance.

        Args:
            results: Similarity results to re-rank
            reference_time: Reference timestamp (ISO format)
            decay_factor: Exponential decay factor for time distance

        Returns:
            Re-ranked SimilarityResult
        """
        reranked = SimilarityResult()
        reranked.metadata = results.metadata.copy()
        reranked.metadata['reranked'] = True
        reranked.metadata['reference_time'] = reference_time

        try:
            ref_dt = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"Invalid reference time: {e}")
            return results

        # Re-rank entities
        for entity_id, score in results.entities:
            # Get temporal metadata if available
            temporal_score = score  # Base similarity score

            if 'entity_metadata' in results.metadata:
                entity_meta = results.metadata['entity_metadata'].get(entity_id, {})
                if 'last_seen' in entity_meta:
                    try:
                        last_seen = datetime.fromisoformat(entity_meta['last_seen'].replace('Z', '+00:00'))
                        time_diff_days = abs((ref_dt - last_seen).days)

                        # Exponential decay
                        recency_factor = np.exp(-decay_factor * time_diff_days / 30.0)  # 30-day scale
                        temporal_score = score * (0.7 + 0.3 * recency_factor)
                    except Exception:
                        pass

            reranked.add_entity(entity_id, temporal_score)

        # Sort by new scores
        reranked.entities.sort(key=lambda x: x[1], reverse=True)

        return reranked

    # Helper methods

    def _get_entity_embedding(self, entity_id: int) -> Optional[np.ndarray]:
        """Get entity embedding from model or Qdrant.

        Args:
            entity_id: Entity ID

        Returns:
            Embedding vector or None
        """
        # Try to get from embedding model first
        if self.embedding_model is not None:
            try:
                entity_tensor = torch.tensor([entity_id], dtype=torch.long)
                with torch.no_grad():
                    embedding = self.embedding_model.entity_embeddings(entity_tensor)
                    # Complex embedding: flatten real and imaginary parts
                    embedding = embedding.squeeze(0).cpu().numpy()
                    # Concatenate real and imaginary
                    return embedding.flatten()
            except Exception as e:
                logger.debug(f"Could not get embedding from model: {e}")

        # Fallback: retrieve from Qdrant
        try:
            points = self.vector_store.client.retrieve(
                collection_name=self.vector_store.config.entity_collection,
                ids=[entity_id]
            )
            if points and len(points) > 0:
                return np.array(points[0].vector)
        except Exception as e:
            logger.debug(f"Could not retrieve from Qdrant: {e}")

        return None

    def _get_relation_embedding(self, relation_id: int) -> Optional[np.ndarray]:
        """Get relation embedding from model or Qdrant.

        Args:
            relation_id: Relation ID

        Returns:
            Embedding vector or None
        """
        # Try to get from embedding model first
        if self.embedding_model is not None:
            try:
                relation_tensor = torch.tensor([relation_id], dtype=torch.long)
                with torch.no_grad():
                    embedding = self.embedding_model.relation_embeddings(relation_tensor)
                    return embedding.squeeze(0).cpu().numpy()
            except Exception as e:
                logger.debug(f"Could not get relation embedding from model: {e}")

        # Fallback: retrieve from Qdrant
        try:
            points = self.vector_store.client.retrieve(
                collection_name=self.vector_store.config.relation_collection,
                ids=[relation_id]
            )
            if points and len(points) > 0:
                return np.array(points[0].vector)
        except Exception as e:
            logger.debug(f"Could not retrieve relation from Qdrant: {e}")

        return None

    def _build_entity_filter(
        self,
        time_filter: Optional[Tuple[str, str]],
        entity_type_filter: Optional[str]
    ):
        """Build Qdrant filter for entity search.

        Args:
            time_filter: Optional (start, end) time window
            entity_type_filter: Optional entity type

        Returns:
            Qdrant Filter object or None
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

        conditions = []

        # Entity type filter
        if entity_type_filter:
            conditions.append(
                FieldCondition(
                    key="entity_type",
                    match=MatchValue(value=entity_type_filter)
                )
            )

        # Time filter (if temporal metadata exists)
        if time_filter:
            start_time, end_time = time_filter
            # Note: This assumes entities have temporal metadata
            # Implementation depends on metadata schema

        if not conditions:
            return None

        return Filter(must=conditions)


def create_similarity_search(
    vector_store,
    embedding_model=None,
    temporal_model=None,
    entity_to_id: Optional[Dict[str, int]] = None,
    id_to_entity: Optional[Dict[int, str]] = None,
    relation_to_id: Optional[Dict[str, int]] = None
) -> VectorSimilaritySearch:
    """Create vector similarity search instance.

    Args:
        vector_store: VectorStore instance
        embedding_model: Optional embedding model
        temporal_model: Optional temporal model
        entity_to_id: Entity name to ID mapping
        id_to_entity: Entity ID to name mapping
        relation_to_id: Relation type to ID mapping

    Returns:
        Initialized VectorSimilaritySearch
    """
    return VectorSimilaritySearch(
        vector_store=vector_store,
        embedding_model=embedding_model,
        temporal_model=temporal_model,
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        relation_to_id=relation_to_id
    )
