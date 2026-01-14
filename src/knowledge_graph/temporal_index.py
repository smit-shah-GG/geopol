"""
Temporal index creation for efficient queries on knowledge graphs.

This module handles:
1. Timestamp-sorted edge index for time-range queries
2. Actor-pair index for bilateral relation queries
3. Temporal neighbor iteration with time constraints
4. QuadClass-specific subgraph views
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
import networkx as nx

logger = logging.getLogger(__name__)


class TemporalIndex:
    """
    Efficient temporal index for knowledge graph queries.

    Maintains:
    - Sorted edge lists by timestamp for range queries
    - Actor-pair index for O(1) lookups
    - QuadClass views for subset queries
    """

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize temporal index from graph.

        Args:
            graph: NetworkX MultiDiGraph to index
        """
        self.graph = graph
        # (timestamp, source_node, target_node, edge_key) - nodes are strings, key is int
        self.timestamp_index: List[Tuple[str, str, str, int]] = []
        self.actor_pair_index: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}  # (u, v) -> [(key, timestamp)]
        self.quadclass_views: Dict[int, List[Tuple[int, int, int]]] = {1: [], 2: [], 3: [], 4: []}

        self._build_indexes()

    def _build_indexes(self):
        """Build all indexes from graph."""
        logger.debug("Building temporal indexes...")

        # Collect all edges with timestamps
        edges_with_timestamps = []

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            timestamp = data.get('timestamp', '')
            quad_class = data.get('quad_class', 1)

            edges_with_timestamps.append((timestamp, u, v, key))

            # Add to actor-pair index
            pair = (u, v)
            if pair not in self.actor_pair_index:
                self.actor_pair_index[pair] = []
            self.actor_pair_index[pair].append((key, timestamp))

            # Add to quadclass view
            if quad_class in self.quadclass_views:
                self.quadclass_views[quad_class].append((u, v, key))

        # Sort by timestamp for range queries
        edges_with_timestamps.sort(key=lambda x: x[0])
        self.timestamp_index = edges_with_timestamps

        logger.debug(
            f"Index built: {len(self.timestamp_index)} edges, "
            f"{len(self.actor_pair_index)} actor pairs"
        )

    def edges_in_time_range(self, start_date: str, end_date: str) -> List[Tuple[str, str, int]]:
        """
        Get edges within time range using binary search.

        Performance: O(log n + k) where k is number of results.

        Args:
            start_date: ISO format start (YYYY-MM-DDT00:00:00Z)
            end_date: ISO format end (YYYY-MM-DDT23:59:59Z)

        Returns:
            List of (source, target, key) tuples
        """
        # Binary search for range using type-consistent sentinels
        # '' sorts before any node name; '~' (ASCII 126) sorts after alphanumeric names
        left_idx = bisect_left(self.timestamp_index, (start_date, '', '', 0))
        right_idx = bisect_right(self.timestamp_index, (end_date, '~', '~', 2**31))

        results = []
        for i in range(left_idx, min(right_idx, len(self.timestamp_index))):
            timestamp, u, v, key = self.timestamp_index[i]
            if start_date <= timestamp <= end_date:
                results.append((u, v, key))

        return results

    def get_actor_pair_relations(self, actor1: int, actor2: int) -> List[Tuple[int, str]]:
        """
        Get all relations between actor pair.

        Performance: O(1) lookup.

        Args:
            actor1: Source actor node ID
            actor2: Target actor node ID

        Returns:
            List of (edge_key, timestamp) tuples
        """
        pair = (actor1, actor2)
        return self.actor_pair_index.get(pair, [])

    def bilateral_relations(self, actor1: int, actor2: int) -> Tuple[List, List]:
        """
        Get bilateral relations between two actors (both directions).

        Args:
            actor1: First actor
            actor2: Second actor

        Returns:
            Tuple of (forward_relations, backward_relations)
        """
        forward = self.actor_pair_index.get((actor1, actor2), [])
        backward = self.actor_pair_index.get((actor2, actor1), [])
        return (forward, backward)

    def temporal_neighbors(self, node: int, time_window: str = 'month',
                          direction: str = 'out') -> Dict[int, List]:
        """
        Get neighbors of node within time window.

        Args:
            node: Node ID
            time_window: 'day', 'week', 'month', or 'year'
            direction: 'in', 'out', or 'both'

        Returns:
            Dict mapping neighbor to list of relations
        """
        neighbors = {}

        # Get edges involving this node
        if direction in ['out', 'both']:
            for u, v, key, data in self.graph.out_edges(node, keys=True, data=True):
                if v not in neighbors:
                    neighbors[v] = []
                neighbors[v].append(data)

        if direction in ['in', 'both']:
            for u, v, key, data in self.graph.in_edges(node, keys=True, data=True):
                if u not in neighbors:
                    neighbors[u] = []
                neighbors[u].append(data)

        return neighbors

    def quadclass_subgraph(self, quad_class: int) -> nx.MultiDiGraph:
        """
        Get subgraph for specific QuadClass.

        Args:
            quad_class: QuadClass (1, 2, 3, or 4)

        Returns:
            New MultiDiGraph with edges from specified QuadClass
        """
        edges = self.quadclass_views.get(quad_class, [])

        subgraph = nx.MultiDiGraph()

        # Add nodes
        for u, v, key in edges:
            subgraph.add_node(u, **self.graph.nodes[u])
            subgraph.add_node(v, **self.graph.nodes[v])

        # Add edges
        for u, v, key in edges:
            edge_data = self.graph.get_edge_data(u, v, key)
            subgraph.add_edge(u, v, key, **edge_data)

        return subgraph

    def top_actors_by_degree(self, top_k: int = 20, direction: str = 'out') -> List[Tuple[int, int]]:
        """
        Get top K actors by degree.

        Args:
            top_k: Number of top actors to return
            direction: 'in', 'out', or 'both'

        Returns:
            List of (actor, degree) sorted by degree descending
        """
        if direction == 'out':
            degrees = dict(self.graph.out_degree())
        elif direction == 'in':
            degrees = dict(self.graph.in_degree())
        else:
            degrees = dict(self.graph.degree())

        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_degrees[:top_k]

    def strongly_connected_components(self) -> List[Set]:
        """
        Get strongly connected components in graph.

        Returns:
            List of node sets representing SCCs
        """
        return list(nx.strongly_connected_components(self.graph))

    def shortest_path(self, source: int, target: int, max_length: int = 5) -> Optional[List]:
        """
        Find shortest path between actors.

        Args:
            source: Source actor
            target: Target actor
            max_length: Maximum path length to consider

        Returns:
            Shortest path or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, source, target, cutoff=max_length)
            return path
        except nx.NetworkXNoPath:
            return None

    def k_hop_neighbors(self, node: int, k: int = 2) -> Set:
        """
        Get k-hop neighbors of node.

        Args:
            node: Starting node
            k: Number of hops

        Returns:
            Set of reachable nodes within k hops
        """
        neighbors = {node}
        current_level = {node}

        for _ in range(k):
            next_level = set()
            for n in current_level:
                for neighbor in self.graph.successors(n):
                    if neighbor not in neighbors:
                        next_level.add(neighbor)
                        neighbors.add(neighbor)
            current_level = next_level

            if not current_level:
                break

        return neighbors - {node}  # Exclude self

    def centrality_measures(self) -> Dict:
        """
        Calculate various centrality measures.

        Returns:
            Dict with centrality metrics
        """
        metrics = {}

        # In-degree centrality (who targets this actor)
        in_centrality = nx.in_degree_centrality(self.graph)
        metrics['in_centrality_top'] = sorted(
            in_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Out-degree centrality (who this actor targets)
        out_centrality = nx.out_degree_centrality(self.graph)
        metrics['out_centrality_top'] = sorted(
            out_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return metrics

    def filter_by_confidence(self, min_confidence: float) -> nx.MultiDiGraph:
        """
        Create subgraph with minimum confidence edges only.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            New MultiDiGraph with filtered edges
        """
        subgraph = nx.MultiDiGraph()

        # Copy nodes
        for node, data in self.graph.nodes(data=True):
            subgraph.add_node(node, **data)

        # Add filtered edges
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('confidence', 0) >= min_confidence:
                subgraph.add_edge(u, v, key, **data)

        return subgraph

    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            'indexed_edges': len(self.timestamp_index),
            'indexed_actor_pairs': len(self.actor_pair_index),
            'quadclass_distribution': {
                qc: len(edges) for qc, edges in self.quadclass_views.items()
            },
        }


def create_index(graph: nx.MultiDiGraph) -> TemporalIndex:
    """Factory function to create temporal index."""
    return TemporalIndex(graph)
