"""
Qdrant Vector Database Integration for Knowledge Graph Embeddings

Manages storage and retrieval of entity and relation embeddings in Qdrant
vector database with temporal metadata support.

Features:
    - Collection creation and configuration with HNSW indexing
    - Batch upload of embeddings with metadata
    - Temporal filtering for time-aware queries
    - Similarity search with metadata constraints
    - Backup and restore functionality
    - Health monitoring and connection pooling

Architecture:
    - Separate collections for entities and relations
    - HNSW index for fast approximate nearest neighbor search
    - Payload schema includes: entity_type, temporal_bounds, confidence, etc.
    - CPU-optimized configuration (no GPU required)
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationSearchParams
)

from knowledge_graph.embeddings import RotatEModel
from knowledge_graph.temporal_embeddings import TemporalRotatEModel, HyTETemporalExtension


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector store."""

    # Connection
    host: str = "localhost"
    port: int = 6333
    timeout: int = 60  # seconds

    # Collection configuration
    entity_collection: str = "kg_entities"
    relation_collection: str = "kg_relations"
    embedding_dim: int = 256

    # HNSW index parameters (CPU-optimized)
    hnsw_m: int = 16  # Number of edges per node
    hnsw_ef_construct: int = 100  # Size of candidate list during construction
    hnsw_ef_search: int = 64  # Size of candidate list during search

    # Indexing parameters
    batch_size: int = 1000  # Vectors per batch upload
    parallel: int = 4  # Parallel indexing threads

    # Quantization (optional, for memory efficiency)
    use_quantization: bool = False
    quantization_type: str = "scalar"  # scalar or product

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class VectorStore:
    """
    Qdrant-based vector store for knowledge graph embeddings.

    Manages entity and relation embeddings with temporal metadata,
    similarity search, and batch operations.
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize vector store.

        Args:
            config: Qdrant configuration (uses defaults if None)
        """
        self.config = config or QdrantConfig()

        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout
        )

        # Check connection
        self._check_health()

    def _check_health(self):
        """Check Qdrant server health."""
        try:
            # Simple health check
            collections = self.client.get_collections()
            print(f"Connected to Qdrant at {self.config.host}:{self.config.port}")
            print(f"Existing collections: {len(collections.collections)}")
        except Exception as e:
            print(f"Warning: Qdrant connection check failed: {e}")
            print("Qdrant server may not be running. Operations will fail.")

    def create_collections(self, recreate: bool = False):
        """
        Create entity and relation collections with appropriate configuration.

        Args:
            recreate: If True, delete existing collections and recreate
        """
        collections_to_create = [
            (self.config.entity_collection, f"Entity embeddings ({self.config.embedding_dim}D)"),
            (self.config.relation_collection, f"Relation embeddings ({self.config.embedding_dim//2}D for RotatE)")
        ]

        for collection_name, description in collections_to_create:
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(collection_name)
                if recreate:
                    print(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    print(f"Collection {collection_name} already exists")
                    continue
            except Exception:
                # Collection doesn't exist, will create
                pass

            # Determine vector size
            if "entity" in collection_name:
                vector_size = 2 * self.config.embedding_dim  # Complex embeddings
            else:
                vector_size = self.config.embedding_dim  # Phase embeddings

            # Create collection with HNSW configuration
            print(f"Creating collection: {collection_name} (dim={vector_size})")

            # HNSW configuration for fast similarity search
            hnsw_config = HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
                full_scan_threshold=10000  # Use HNSW above this many vectors
            )

            # Optimizer configuration
            optimizer_config = OptimizersConfigDiff(
                indexing_threshold=10000,  # Start indexing after this many vectors
                memmap_threshold=20000  # Use memory mapping above this
            )

            # Quantization configuration (optional)
            quantization_config = None
            if self.config.use_quantization:
                quantization_config = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # Cosine similarity for normalized vectors
                ),
                hnsw_config=hnsw_config,
                optimizers_config=optimizer_config,
                quantization_config=quantization_config
            )

            print(f"Created collection: {collection_name}")

            # Create payload indices for filtering
            self._create_payload_indices(collection_name)

    def _create_payload_indices(self, collection_name: str):
        """
        Create payload field indices for fast filtering.

        Args:
            collection_name: Name of collection
        """
        # Common indices for both collections
        indices = [
            "entity_id",
            "entity_type",
            "temporal_bucket_start",
            "temporal_bucket_end"
        ]

        for field_name in indices:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword" if "entity" in field_name or "type" in field_name else "integer"
                )
                print(f"  Created index on {field_name}")
            except Exception as e:
                # Index might already exist
                pass

    def upload_entity_embeddings(
        self,
        model: Union[RotatEModel, TemporalRotatEModel],
        entity_to_id: Dict[str, int],
        id_to_entity: Dict[int, str],
        entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Upload entity embeddings to Qdrant.

        Args:
            model: Trained embedding model
            entity_to_id: Mapping from entity names to IDs
            id_to_entity: Mapping from IDs to entity names
            entity_metadata: Optional metadata for each entity
        """
        print(f"Uploading {len(entity_to_id)} entity embeddings...")

        # Extract embeddings
        points = []
        for entity_name, entity_id in entity_to_id.items():
            # Get embedding
            if isinstance(model, TemporalRotatEModel):
                embedding = model.base_model.get_entity_embedding(entity_id)
            else:
                embedding = model.get_entity_embedding(entity_id)

            # Prepare payload
            payload = {
                "entity_id": str(entity_id),
                "entity_name": entity_name,
                "entity_type": "entity"  # Can be enriched with actual types
            }

            # Add metadata if available
            if entity_metadata and entity_name in entity_metadata:
                payload.update(entity_metadata[entity_name])

            # Create point
            point = PointStruct(
                id=entity_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

        # Batch upload
        self._batch_upload(
            self.config.entity_collection,
            points,
            batch_size=self.config.batch_size
        )

        print(f"Uploaded {len(points)} entity embeddings")

    def upload_relation_embeddings(
        self,
        model: Union[RotatEModel, TemporalRotatEModel],
        relation_to_id: Dict[str, int],
        id_to_relation: Dict[int, str]
    ):
        """
        Upload relation embeddings to Qdrant.

        Args:
            model: Trained embedding model
            relation_to_id: Mapping from relation types to IDs
            id_to_relation: Mapping from IDs to relation types
        """
        print(f"Uploading {len(relation_to_id)} relation embeddings...")

        points = []
        for relation_name, relation_id in relation_to_id.items():
            # Get embedding
            if isinstance(model, TemporalRotatEModel):
                embedding = model.base_model.get_relation_embedding(relation_id)
            else:
                embedding = model.get_relation_embedding(relation_id)

            # Prepare payload
            payload = {
                "relation_id": str(relation_id),
                "relation_name": relation_name,
                "entity_type": "relation"
            }

            # Create point
            point = PointStruct(
                id=relation_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

        # Batch upload
        self._batch_upload(
            self.config.relation_collection,
            points,
            batch_size=self.config.batch_size
        )

        print(f"Uploaded {len(points)} relation embeddings")

    def _batch_upload(
        self,
        collection_name: str,
        points: List[PointStruct],
        batch_size: int
    ):
        """
        Upload points in batches.

        Args:
            collection_name: Target collection
            points: List of points to upload
            batch_size: Number of points per batch
        """
        total_batches = (len(points) + batch_size - 1) // batch_size

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                print(f"  Uploaded batch {batch_num}/{total_batches} ({len(batch)} points)")
            except Exception as e:
                print(f"  Error uploading batch {batch_num}: {e}")
                raise

    def search_similar_entities(
        self,
        query_vector: Union[torch.Tensor, List[float]],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar entities.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional filter conditions

        Returns:
            List of (entity_id, score, payload) tuples
        """
        # Convert tensor to list if needed
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.tolist()

        # Build filter
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)

        # Search
        results = self.client.search(
            collection_name=self.config.entity_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        # Format results
        return [
            (result.id, result.score, result.payload)
            for result in results
        ]

    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from conditions.

        Args:
            conditions: Dictionary of field->value conditions

        Returns:
            Qdrant Filter object
        """
        must_conditions = []

        for field, value in conditions.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                # Range condition
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        range=Range(gte=value[0], lte=value[1])
                    )
                )
            else:
                # Exact match
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )

        return Filter(must=must_conditions)

    def get_entity_by_id(self, entity_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Dictionary with vector and payload, or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.config.entity_collection,
                ids=[entity_id],
                with_vectors=True,
                with_payload=True
            )

            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
        except Exception as e:
            print(f"Error retrieving entity {entity_id}: {e}")

        return None

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}

    def backup_collection(
        self,
        collection_name: str,
        backup_path: str
    ):
        """
        Backup collection to file.

        Args:
            collection_name: Name of collection to backup
            backup_path: Path to save backup
        """
        print(f"Backing up collection {collection_name}...")

        # Get all points
        points = []
        offset = None
        batch_size = 1000

        while True:
            result = self.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True
            )

            batch_points, next_offset = result

            if not batch_points:
                break

            # Convert to serializable format
            for point in batch_points:
                points.append({
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                })

            offset = next_offset
            if offset is None:
                break

        # Save to file
        backup_data = {
            "collection_name": collection_name,
            "points_count": len(points),
            "points": points
        }

        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f)

        print(f"Backed up {len(points)} points to {backup_path}")

    def restore_collection(
        self,
        backup_path: str,
        collection_name: Optional[str] = None
    ):
        """
        Restore collection from backup file.

        Args:
            backup_path: Path to backup file
            collection_name: Target collection name (uses backup name if None)
        """
        print(f"Restoring collection from {backup_path}...")

        # Load backup
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        target_collection = collection_name or backup_data["collection_name"]
        points_data = backup_data["points"]

        # Convert to PointStruct
        points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"]
            )
            for p in points_data
        ]

        # Upload in batches
        self._batch_upload(target_collection, points, self.config.batch_size)

        print(f"Restored {len(points)} points to {target_collection}")


def setup_qdrant_for_embeddings(
    model: Union[RotatEModel, TemporalRotatEModel],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    id_to_entity: Dict[int, str],
    id_to_relation: Dict[int, str],
    config: Optional[QdrantConfig] = None,
    entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> VectorStore:
    """
    High-level function to set up Qdrant and upload embeddings.

    Args:
        model: Trained embedding model
        entity_to_id: Entity name to ID mapping
        relation_to_id: Relation name to ID mapping
        id_to_entity: ID to entity name mapping
        id_to_relation: ID to relation name mapping
        config: Qdrant configuration
        entity_metadata: Optional metadata for entities

    Returns:
        Configured VectorStore instance
    """
    if config is None:
        config = QdrantConfig()

    # Initialize vector store
    store = VectorStore(config)

    # Create collections
    store.create_collections(recreate=True)

    # Upload embeddings
    store.upload_entity_embeddings(
        model, entity_to_id, id_to_entity, entity_metadata
    )
    store.upload_relation_embeddings(
        model, relation_to_id, id_to_relation
    )

    # Print statistics
    entity_info = store.get_collection_info(config.entity_collection)
    relation_info = store.get_collection_info(config.relation_collection)

    print(f"\nQdrant Setup Complete:")
    print(f"  Entities: {entity_info.get('points_count', 0)} points")
    print(f"  Relations: {relation_info.get('points_count', 0)} points")

    return store
