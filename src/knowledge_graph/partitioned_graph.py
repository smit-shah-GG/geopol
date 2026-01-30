"""
PartitionedTemporalGraph: Unified interface over partitioned temporal knowledge graphs.

Provides a drop-in replacement for GraphTraversal API that transparently handles
cross-partition queries through the QueryRouter scatter-gather mechanism.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .boundary_resolver import BoundaryResolver
from .cross_partition_query import QueryRouter
from .graph_traversal import TraversalResult
from .partition_index import EntityPartitionIndex
from .partition_manager import PartitionManager

logger = logging.getLogger(__name__)


class PartitionedTemporalGraph:
    """
    Unified interface over partitioned temporal knowledge graphs.

    Provides the same API as GraphTraversal but transparently handles:
    - Entity-to-partition routing via EntityPartitionIndex
    - Parallel scatter-gather queries via QueryRouter
    - Home partition prioritization for boundary entities
    - Result merging with edge deduplication

    This enables drop-in replacement of single-graph queries with
    partitioned queries, preserving query correctness.

    Usage:
        # Load from existing partition directory
        ptg = PartitionedTemporalGraph(
            partition_dir=Path("data/partitions"),
            index_path=Path("data/partitions/partition_meta.db")
        )

        # Query just like GraphTraversal
        result = ptg.k_hop_neighborhood("entity_id", k=2)

        # Or use factory function
        ptg = load_partitioned_graph(Path("data/partitions"))
    """

    def __init__(
        self,
        partition_dir: Path,
        index_path: Path,
        max_cached: int = 4,
        max_workers: int = 4,
        replication_threshold: int = 10
    ) -> None:
        """
        Initialize PartitionedTemporalGraph.

        Args:
            partition_dir: Directory containing partition GraphML files.
            index_path: Path to SQLite partition index database.
            max_cached: Maximum partitions to cache in memory.
            max_workers: Maximum parallel workers for scatter-gather.
            replication_threshold: Minimum edge count for boundary replica tracking.
        """
        self.partition_dir = Path(partition_dir)
        self.index_path = Path(index_path)

        # Initialize components
        self._index = EntityPartitionIndex(index_path)
        self._manager = PartitionManager(
            partition_dir=partition_dir,
            index=self._index,
            max_cached_partitions=max_cached
        )
        self._boundary_resolver = BoundaryResolver(
            index=self._index,
            manager=self._manager,
            replication_threshold=replication_threshold
        )
        self._router = QueryRouter(
            manager=self._manager,
            index=self._index,
            boundary_resolver=self._boundary_resolver,
            max_workers=max_workers
        )

        logger.info(
            f"PartitionedTemporalGraph initialized with "
            f"{len(self._manager.list_partitions())} partitions"
        )

    def k_hop_neighborhood(
        self,
        entity_id: Any,
        k: int = 2,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        quad_class: Optional[int] = None,
        max_results: int = 1000
    ) -> TraversalResult:
        """
        Find k-hop neighborhood of entity with temporal constraints.

        Matches GraphTraversal.k_hop_neighborhood() API for drop-in compatibility.

        Args:
            entity_id: Starting entity.
            k: Number of hops (1-5 recommended).
            time_start: Optional start time filter (ISO format).
            time_end: Optional end time filter (ISO format).
            min_confidence: Minimum edge confidence.
            quad_class: Optional QuadClass filter.
            max_results: Maximum edges to return.

        Returns:
            TraversalResult with k-hop subgraph from all relevant partitions.
        """
        return self._router.execute_k_hop_query(
            entity_id=str(entity_id),  # Ensure string for index lookup
            k=k,
            time_start=time_start,
            time_end=time_end,
            min_confidence=min_confidence,
            quad_class=quad_class,
            max_results=max_results
        )

    def bilateral_relations(
        self,
        entity1: Any,
        entity2: Any,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        quad_class: Optional[int] = None
    ) -> TraversalResult:
        """
        Find all relations between two entities across partitions.

        Matches GraphTraversal.bilateral_relations() API for drop-in compatibility.

        Args:
            entity1: First entity.
            entity2: Second entity.
            time_start: Optional start time filter.
            time_end: Optional end time filter.
            min_confidence: Minimum edge confidence.
            quad_class: Optional QuadClass filter.

        Returns:
            TraversalResult with bilateral edges from all relevant partitions.
        """
        return self._router.execute_bilateral_query(
            entity1=str(entity1),
            entity2=str(entity2),
            time_start=time_start,
            time_end=time_end,
            min_confidence=min_confidence,
            quad_class=quad_class
        )

    def get_stats(self) -> Dict:
        """
        Get statistics about the partitioned graph.

        Returns:
            Dict with partition count, total nodes, total edges,
            cache stats, and boundary entity statistics.
        """
        partitions = self._manager.list_partitions()

        total_nodes = 0
        total_edges = 0

        for pid in partitions:
            meta = self._index.get_partition_meta(pid)
            if meta:
                total_nodes += meta.get('node_count', 0)
                total_edges += meta.get('edge_count', 0)

        # Get cache stats
        cache_size = len(self._manager._cache)
        max_cache = self._manager.max_cached

        # Get boundary stats
        boundary_stats = self._boundary_resolver.get_boundary_stats()

        return {
            'partition_count': len(partitions),
            'partition_ids': partitions,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'cache': {
                'size': cache_size,
                'max_size': max_cache,
                'utilization': cache_size / max_cache if max_cache > 0 else 0.0
            },
            'boundary': boundary_stats
        }

    def get_partition_count(self) -> int:
        """Get number of partitions."""
        return len(self._manager.list_partitions())

    def list_partitions(self) -> List[str]:
        """List all partition IDs."""
        return self._manager.list_partitions()

    def get_partition_meta(self, partition_id: str) -> Optional[Dict]:
        """Get metadata for a specific partition."""
        return self._index.get_partition_meta(partition_id)

    def get_entity_partitions(self, entity_id: str) -> List[str]:
        """Get all partitions containing an entity."""
        return self._index.get_entity_partitions(entity_id)

    def is_boundary_entity(self, entity_id: str) -> bool:
        """Check if an entity is a boundary entity."""
        return self._boundary_resolver.is_boundary_entity(entity_id)

    def get_home_partition(self, entity_id: str) -> Optional[str]:
        """Get home partition for a boundary entity."""
        return self._boundary_resolver.get_home_partition(entity_id)

    def clear_cache(self) -> None:
        """Clear the partition cache."""
        self._manager.clear_cache()

    def close(self) -> None:
        """Close resources (index connection, thread pool)."""
        self._router.shutdown()
        self._index.close()

    def __enter__(self) -> 'PartitionedTemporalGraph':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def load_partitioned_graph(
    partition_dir: Path,
    max_cached: int = 4,
    max_workers: int = 4
) -> PartitionedTemporalGraph:
    """
    Factory function to load a partitioned graph.

    Assumes index database is at {partition_dir}/partition_meta.db.

    Args:
        partition_dir: Directory containing partitions.
        max_cached: Maximum partitions to cache.
        max_workers: Maximum parallel workers.

    Returns:
        PartitionedTemporalGraph instance.
    """
    partition_dir = Path(partition_dir)
    index_path = partition_dir / "partition_meta.db"

    return PartitionedTemporalGraph(
        partition_dir=partition_dir,
        index_path=index_path,
        max_cached=max_cached,
        max_workers=max_workers
    )
