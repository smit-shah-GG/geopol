"""
Boundary entity resolver for cross-partition entity identification.

This module identifies entities that appear across multiple partitions and
tracks their home/replica assignments. Used by QueryRouter to prioritize
home partitions and avoid duplicate work on replicas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .partition_index import EntityPartitionIndex
from .partition_manager import PartitionManager

logger = logging.getLogger(__name__)


@dataclass
class BoundaryEntity:
    """
    Entity that appears in multiple partitions.

    Attributes:
        entity_id: The unique identifier for this entity.
        home_partition: Partition with highest edge count for this entity.
        replica_partitions: Other partitions where entity appears with
            edge_count >= replication_threshold.
        edge_count_per_partition: Map of partition_id to edge count.
    """
    entity_id: str
    home_partition: str
    replica_partitions: Set[str] = field(default_factory=set)
    edge_count_per_partition: Dict[str, int] = field(default_factory=dict)


class BoundaryResolver:
    """
    Resolves cross-partition boundary entities.

    Provides:
    - Identification of entities appearing in multiple partitions
    - Home partition assignment (partition with most edges)
    - Replica tracking for entities above replication threshold
    - Boundary statistics for monitoring

    The resolver does NOT copy entity data between partitions - it only
    tracks which entities are boundaries and their home/replica status.

    Usage:
        resolver = BoundaryResolver(index, manager, replication_threshold=10)
        boundaries = resolver.identify_boundary_entities()
        home = resolver.get_home_partition("some_entity")
    """

    def __init__(
        self,
        index: EntityPartitionIndex,
        manager: PartitionManager,
        replication_threshold: int = 10
    ) -> None:
        """
        Initialize boundary resolver.

        Args:
            index: EntityPartitionIndex for entity-to-partition lookups.
            manager: PartitionManager for partition metadata access.
            replication_threshold: Minimum edge count in a secondary partition
                for it to be considered a replica. Partitions below this
                threshold are not tracked as replicas.
        """
        self.index = index
        self.manager = manager
        self.replication_threshold = replication_threshold

        # Cache of identified boundary entities
        self._boundary_entities: Dict[str, BoundaryEntity] = {}
        self._is_identified = False

    def identify_boundary_entities(self) -> Dict[str, BoundaryEntity]:
        """
        Identify all boundary entities across partitions.

        A boundary entity is one that appears in multiple partitions.
        For each boundary entity, determine:
        - Home partition: partition with highest edge count
        - Replica partitions: other partitions with edge_count >= threshold

        This is a read-only analysis - no data is copied between partitions.

        Returns:
            Dict mapping entity_id to BoundaryEntity for all boundaries.
        """
        # Get all partitions
        partition_ids = self.manager.list_partitions()

        if not partition_ids:
            logger.info("No partitions registered - no boundary entities")
            self._is_identified = True
            return {}

        # Collect entity -> partition -> edge_count mapping
        entity_partitions: Dict[str, Dict[str, int]] = {}

        # Query the index for all entities and their edge counts
        cursor = self.index.conn.cursor()
        cursor.execute("""
            SELECT entity_id, partition_id, edge_count
            FROM entity_partitions
            ORDER BY entity_id
        """)

        for row in cursor.fetchall():
            entity_id = row[0]
            partition_id = row[1]
            edge_count = row[2] or 0

            if entity_id not in entity_partitions:
                entity_partitions[entity_id] = {}
            entity_partitions[entity_id][partition_id] = edge_count

        # Identify boundary entities
        boundary_entities: Dict[str, BoundaryEntity] = {}

        for entity_id, partition_counts in entity_partitions.items():
            if len(partition_counts) <= 1:
                # Entity only in one partition - not a boundary
                continue

            # Sort partitions by edge count descending
            sorted_partitions = sorted(
                partition_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Home = highest edge count
            home_partition = sorted_partitions[0][0]

            # Replicas = other partitions with edge_count >= threshold
            replica_partitions = {
                p for p, count in sorted_partitions[1:]
                if count >= self.replication_threshold
            }

            boundary_entities[entity_id] = BoundaryEntity(
                entity_id=entity_id,
                home_partition=home_partition,
                replica_partitions=replica_partitions,
                edge_count_per_partition=partition_counts
            )

        # Check for boundary explosion (Pitfall 5)
        total_entities = len(entity_partitions)
        boundary_count = len(boundary_entities)
        if total_entities > 0:
            replication_ratio = boundary_count / total_entities
            if replication_ratio > 0.30:
                logger.warning(
                    f"Boundary entity explosion detected: {boundary_count}/{total_entities} "
                    f"({replication_ratio:.1%}) entities are boundaries. "
                    "Consider adjusting partitioning strategy or replication threshold."
                )

        self._boundary_entities = boundary_entities
        self._is_identified = True

        logger.info(
            f"Identified {boundary_count} boundary entities across "
            f"{len(partition_ids)} partitions"
        )

        return boundary_entities

    def get_home_partition(self, entity_id: str) -> Optional[str]:
        """
        Get the home partition for an entity.

        For boundary entities, returns the partition with the highest edge
        count for this entity. For non-boundary entities, returns None.

        This method is used by QueryRouter to prioritize home partition
        in scatter-gather queries.

        Args:
            entity_id: Entity to look up.

        Returns:
            Home partition ID if entity is a boundary, None otherwise.
        """
        if not self._is_identified:
            self.identify_boundary_entities()

        boundary = self._boundary_entities.get(entity_id)
        if boundary is None:
            return None

        return boundary.home_partition

    def is_boundary_entity(self, entity_id: str) -> bool:
        """
        Check if an entity is a boundary entity.

        Args:
            entity_id: Entity to check.

        Returns:
            True if entity appears in multiple partitions.
        """
        if not self._is_identified:
            self.identify_boundary_entities()

        return entity_id in self._boundary_entities

    def get_boundary_entity(self, entity_id: str) -> Optional[BoundaryEntity]:
        """
        Get full BoundaryEntity info for an entity.

        Args:
            entity_id: Entity to look up.

        Returns:
            BoundaryEntity if entity is a boundary, None otherwise.
        """
        if not self._is_identified:
            self.identify_boundary_entities()

        return self._boundary_entities.get(entity_id)

    def get_boundary_stats(self) -> Dict:
        """
        Get statistics about boundary entities.

        Returns:
            Dict with:
            - total_entities: Total entities across all partitions
            - boundary_entities: Number of boundary entities
            - avg_replica_count: Average number of replica partitions per boundary
            - replication_ratio: Fraction of entities that are boundaries
            - replication_threshold: Current threshold setting
        """
        if not self._is_identified:
            self.identify_boundary_entities()

        # Count total unique entities
        cursor = self.index.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT entity_id) FROM entity_partitions")
        total_entities = cursor.fetchone()[0] or 0

        boundary_count = len(self._boundary_entities)

        # Calculate average replica count
        if boundary_count > 0:
            total_replicas = sum(
                len(b.replica_partitions)
                for b in self._boundary_entities.values()
            )
            avg_replica_count = total_replicas / boundary_count
        else:
            avg_replica_count = 0.0

        # Calculate replication ratio
        replication_ratio = (
            boundary_count / total_entities if total_entities > 0 else 0.0
        )

        return {
            'total_entities': total_entities,
            'boundary_entities': boundary_count,
            'avg_replica_count': avg_replica_count,
            'replication_ratio': replication_ratio,
            'replication_threshold': self.replication_threshold
        }

    def invalidate_cache(self) -> None:
        """
        Invalidate the cached boundary identification.

        Call this after partitions are modified (added/removed/re-partitioned)
        to force re-identification on next access.
        """
        self._boundary_entities.clear()
        self._is_identified = False
