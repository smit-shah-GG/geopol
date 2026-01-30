"""
Cross-partition query router with scatter-gather execution.

This module provides QueryRouter for executing queries across multiple
partitions in parallel, with home partition prioritization for boundary
entities.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple, Any

from .boundary_resolver import BoundaryResolver
from .graph_traversal import GraphTraversal, TraversalResult
from .partition_index import EntityPartitionIndex
from .partition_manager import PartitionManager

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Routes queries across partitions using scatter-gather pattern.

    Provides:
    - Parallel query execution across relevant partitions
    - Home partition prioritization for boundary entities
    - Result merging with edge deduplication
    - Time range filtering for partition selection

    Correctness guarantee: Returns the same result as querying a single
    unified graph by querying all partitions containing the entity and
    merging the results.

    Usage:
        router = QueryRouter(manager, index, boundary_resolver)
        result = router.execute_k_hop_query("entity_id", k=2)
    """

    def __init__(
        self,
        manager: PartitionManager,
        index: EntityPartitionIndex,
        boundary_resolver: BoundaryResolver,
        max_workers: int = 4
    ) -> None:
        """
        Initialize query router.

        Args:
            manager: PartitionManager for loading partitions.
            index: EntityPartitionIndex for entity-to-partition lookups.
            boundary_resolver: BoundaryResolver for home partition prioritization.
            max_workers: Maximum parallel workers for scatter queries.
        """
        self.manager = manager
        self.index = index
        self.boundary_resolver = boundary_resolver
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_k_hop_query(
        self,
        entity_id: str,
        k: int,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        quad_class: Optional[int] = None,
        max_results: int = 1000
    ) -> TraversalResult:
        """
        Execute k-hop neighborhood query across partitions.

        Correctness guarantee: Returns same result as single-graph query
        by building a merged view of all relevant partitions and executing
        k-hop traversal on the merged graph.

        For boundary entities, the home partition is queried first to
        reduce duplicate work on replicas.

        Args:
            entity_id: Starting entity for traversal.
            k: Number of hops (1-5 recommended).
            time_start: Optional start time filter (ISO format).
            time_end: Optional end time filter (ISO format).
            min_confidence: Minimum edge confidence.
            quad_class: Optional QuadClass filter.
            max_results: Maximum edges to return.

        Returns:
            TraversalResult with merged k-hop subgraph from all partitions.
        """
        # Step 1: For k-hop queries, we must consider ALL partitions since
        # edges are partitioned by time, not entity. A k-hop neighborhood
        # can span any number of time windows.
        all_partitions = self.manager.list_partitions()

        if not all_partitions:
            logger.debug("No partitions registered")
            return TraversalResult()

        # Step 2: Filter by time window if specified
        if time_start or time_end:
            all_partitions = self._filter_partitions_by_time(
                all_partitions, time_start, time_end
            )
            if not all_partitions:
                logger.debug(f"No partitions in time range for entity {entity_id}")
                return TraversalResult()

        # Step 3: Prioritize home partition for boundary entities
        partitions = self._prioritize_home_partition(entity_id, all_partitions)

        logger.debug(
            f"Building merged view from {len(partitions)} partitions for entity {entity_id}, k={k}"
        )

        # Step 4: Build merged graph from all partitions
        # This ensures k-hop traversal sees ALL edges, regardless of which
        # time partition they're in.
        merged_graph = self._build_merged_graph(partitions)

        # Check if entity exists in merged graph
        if entity_id not in merged_graph and str(entity_id) not in merged_graph:
            logger.debug(f"Entity {entity_id} not in merged graph")
            return TraversalResult()

        # Use string entity_id if needed
        query_entity = entity_id if entity_id in merged_graph else str(entity_id)

        # Step 5: Execute k-hop on merged graph
        from .graph_traversal import GraphTraversal
        traversal = GraphTraversal(merged_graph)
        result = traversal.k_hop_neighborhood(
            entity_id=query_entity,
            k=k,
            time_start=time_start,
            time_end=time_end,
            min_confidence=min_confidence,
            quad_class=quad_class,
            max_results=max_results
        )

        result.metadata['partitions_queried'] = partitions

        return result

    def _build_merged_graph(
        self,
        partition_ids: List[str]
    ) -> 'nx.MultiDiGraph':
        """
        Build a merged graph view from multiple partitions.

        Loads all specified partitions and combines them into a single
        graph for k-hop traversal. This is necessary because k-hop
        traversal needs to see ALL edges, regardless of which time
        partition they belong to.

        Args:
            partition_ids: Partitions to merge.

        Returns:
            NetworkX MultiDiGraph containing all nodes and edges.
        """
        import networkx as nx

        merged = nx.MultiDiGraph()

        for partition_id in partition_ids:
            try:
                partition = self.manager.load_partition(partition_id)

                # Add all nodes with their attributes
                for node, data in partition.nodes(data=True):
                    if node not in merged:
                        merged.add_node(node, **data)

                # Add all edges
                for u, v, key, data in partition.edges(keys=True, data=True):
                    # Check for duplicate edge
                    if not merged.has_edge(u, v, key=key):
                        merged.add_edge(u, v, key=key, **data)

            except FileNotFoundError:
                logger.warning(f"Partition {partition_id} not found, skipping")
            except Exception as e:
                logger.error(f"Error loading partition {partition_id}: {e}")

        logger.debug(
            f"Merged {len(partition_ids)} partitions: "
            f"{merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges"
        )

        return merged

    def _filter_partitions_by_time(
        self,
        partition_ids: List[str],
        time_start: Optional[str],
        time_end: Optional[str]
    ) -> List[str]:
        """
        Filter partition list to those overlapping the time range.

        Args:
            partition_ids: Candidate partitions.
            time_start: Range start (inclusive).
            time_end: Range end (inclusive).

        Returns:
            Filtered list of partition IDs.
        """
        if not time_start and not time_end:
            return partition_ids

        # Get partitions in time range from index
        time_range_partitions = set(
            self.index.get_partitions_in_time_range(
                time_start or "",
                time_end or ""
            )
        )

        # Include "no-timestamp" partition as it could have relevant edges
        if "no-timestamp" in partition_ids:
            time_range_partitions.add("no-timestamp")

        return [p for p in partition_ids if p in time_range_partitions]

    def _prioritize_home_partition(
        self,
        entity_id: str,
        partitions: List[str]
    ) -> List[str]:
        """
        Reorder partitions to put home partition first for boundary entities.

        This optimization reduces unnecessary duplicate work by querying the
        authoritative partition first.

        Args:
            entity_id: Entity being queried.
            partitions: List of partition IDs to reorder.

        Returns:
            Reordered partition list with home first (if applicable).
        """
        home_partition = self.boundary_resolver.get_home_partition(entity_id)

        if home_partition is None or home_partition not in partitions:
            # Not a boundary entity or home not in list - no change
            return partitions

        # Move home partition to front
        reordered = [home_partition]
        reordered.extend(p for p in partitions if p != home_partition)

        logger.debug(f"Prioritized home partition {home_partition} for {entity_id}")

        return reordered

    def _query_single_partition(
        self,
        partition_id: str,
        entity_id: str,
        k: int,
        time_start: Optional[str],
        time_end: Optional[str],
        min_confidence: float,
        quad_class: Optional[int],
        max_results: int
    ) -> TraversalResult:
        """
        Query a single partition for k-hop neighborhood.

        Args:
            partition_id: Partition to query.
            entity_id: Starting entity.
            k: Number of hops.
            time_start: Time range start.
            time_end: Time range end.
            min_confidence: Minimum confidence.
            quad_class: QuadClass filter.
            max_results: Max edges.

        Returns:
            TraversalResult from this partition.
        """
        graph = self.manager.load_partition(partition_id)

        # Handle entity_id type mismatch (GraphML may use str keys)
        # NetworkX may have loaded entity as string
        if entity_id not in graph:
            # Try string representation
            str_entity = str(entity_id)
            if str_entity not in graph:
                # Entity not in this partition (may have been removed)
                return TraversalResult()
            entity_id = str_entity

        traversal = GraphTraversal(graph)
        result = traversal.k_hop_neighborhood(
            entity_id=entity_id,
            k=k,
            time_start=time_start,
            time_end=time_end,
            min_confidence=min_confidence,
            quad_class=quad_class,
            max_results=max_results
        )

        result.metadata['source_partition'] = partition_id
        return result

    def _merge_result(
        self,
        target: TraversalResult,
        source: TraversalResult,
        seen_edges: Set[Tuple[Any, Any, Any]]
    ) -> None:
        """
        Merge source result into target, deduplicating edges.

        Args:
            target: Target result to merge into.
            source: Source result to merge from.
            seen_edges: Set of (u, v, key) tuples already seen.
        """
        # Merge nodes (set union)
        target.nodes.update(source.nodes)

        # Merge edges with deduplication
        for edge in source.edges:
            u, v, key, data = edge
            edge_key = (u, v, key)

            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                target.edges.append(edge)

        # Merge paths
        target.paths.extend(source.paths)

    def execute_bilateral_query(
        self,
        entity1: str,
        entity2: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        quad_class: Optional[int] = None
    ) -> TraversalResult:
        """
        Execute bilateral relations query across partitions.

        Finds all relations between two entities across all partitions.

        Args:
            entity1: First entity.
            entity2: Second entity.
            time_start: Optional start time filter.
            time_end: Optional end time filter.
            min_confidence: Minimum edge confidence.
            quad_class: Optional QuadClass filter.

        Returns:
            TraversalResult with all bilateral edges.
        """
        # Get partitions containing either entity
        partitions_1 = set(self.index.get_entity_partitions(entity1))
        partitions_2 = set(self.index.get_entity_partitions(entity2))

        # Only query partitions containing BOTH entities
        common_partitions = list(partitions_1 & partitions_2)

        if not common_partitions:
            return TraversalResult()

        # Filter by time range if specified
        if time_start or time_end:
            common_partitions = self._filter_partitions_by_time(
                common_partitions, time_start, time_end
            )

        # Scatter-gather
        futures = {}
        for partition_id in common_partitions:
            future = self._executor.submit(
                self._query_bilateral_single_partition,
                partition_id,
                entity1,
                entity2,
                time_start,
                time_end,
                min_confidence,
                quad_class
            )
            futures[future] = partition_id

        merged_result = TraversalResult()
        merged_result.metadata['query_type'] = 'bilateral_relations'
        merged_result.metadata['entity1'] = entity1
        merged_result.metadata['entity2'] = entity2
        merged_result.metadata['partitions_queried'] = common_partitions

        seen_edges: Set[Tuple[Any, Any, Any]] = set()

        for future in as_completed(futures):
            partition_id = futures[future]
            try:
                partial_result = future.result()
                self._merge_result(merged_result, partial_result, seen_edges)
            except Exception as e:
                logger.error(f"Error querying partition {partition_id}: {e}")

        return merged_result

    def _query_bilateral_single_partition(
        self,
        partition_id: str,
        entity1: str,
        entity2: str,
        time_start: Optional[str],
        time_end: Optional[str],
        min_confidence: float,
        quad_class: Optional[int]
    ) -> TraversalResult:
        """
        Query a single partition for bilateral relations.

        Args:
            partition_id: Partition to query.
            entity1: First entity.
            entity2: Second entity.
            time_start: Time range start.
            time_end: Time range end.
            min_confidence: Minimum confidence.
            quad_class: QuadClass filter.

        Returns:
            TraversalResult from this partition.
        """
        graph = self.manager.load_partition(partition_id)

        # Handle entity_id type
        e1 = entity1 if entity1 in graph else str(entity1)
        e2 = entity2 if entity2 in graph else str(entity2)

        if e1 not in graph or e2 not in graph:
            return TraversalResult()

        traversal = GraphTraversal(graph)
        result = traversal.bilateral_relations(
            entity1=e1,
            entity2=e2,
            time_start=time_start,
            time_end=time_end,
            min_confidence=min_confidence,
            quad_class=quad_class
        )

        result.metadata['source_partition'] = partition_id
        return result

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
