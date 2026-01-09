"""
Graph Traversal Engine for Temporal Knowledge Graphs.

Implements efficient subgraph extraction with temporal constraints:
- k-hop neighborhood search
- Bilateral relation finding
- Temporal path discovery
- Event sequence pattern matching
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime
from collections import deque, defaultdict
import networkx as nx

from .query_parser import ParsedQuery, QueryType

logger = logging.getLogger(__name__)


class TraversalResult:
    """Container for graph traversal results."""

    def __init__(self):
        """Initialize empty result."""
        self.nodes: Set[int] = set()
        self.edges: List[Tuple[int, int, int, Dict]] = []  # (u, v, key, data)
        self.paths: List[List[Tuple[int, int, int]]] = []  # List of paths (edge sequences)
        self.metadata: Dict[str, Any] = {}

    def add_edge(self, u: int, v: int, key: int, data: Dict):
        """Add edge to result."""
        self.edges.append((u, v, key, data))
        self.nodes.add(u)
        self.nodes.add(v)

    def add_path(self, path: List[Tuple[int, int, int]]):
        """Add path to result."""
        self.paths.append(path)
        for u, v, key in path:
            self.nodes.add(u)
            self.nodes.add(v)

    def get_subgraph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Extract subgraph containing result nodes and edges.

        Args:
            graph: Original graph

        Returns:
            Subgraph with result nodes and edges
        """
        subgraph = nx.MultiDiGraph()

        # Add nodes with attributes
        for node in self.nodes:
            if node in graph:
                subgraph.add_node(node, **graph.nodes[node])

        # Add edges with attributes
        for u, v, key, data in self.edges:
            subgraph.add_edge(u, v, key=key, **data)

        return subgraph


class GraphTraversal:
    """Graph traversal engine for temporal knowledge graphs."""

    def __init__(self, graph: nx.MultiDiGraph, temporal_index=None):
        """Initialize graph traversal engine.

        Args:
            graph: NetworkX MultiDiGraph to traverse
            temporal_index: Optional TemporalIndex for efficient queries
        """
        self.graph = graph
        self.temporal_index = temporal_index

    def k_hop_neighborhood(
        self,
        entity_id: int,
        k: int = 2,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        quad_class: Optional[int] = None,
        max_results: int = 1000
    ) -> TraversalResult:
        """Find k-hop neighborhood of entity with temporal constraints.

        Performance: O(n*k) for discovery, < 5ms for dense nodes (k=2)

        Args:
            entity_id: Starting entity
            k: Number of hops (1-5)
            time_start: Optional start time filter (ISO format)
            time_end: Optional end time filter (ISO format)
            min_confidence: Minimum edge confidence
            quad_class: Optional QuadClass filter
            max_results: Maximum edges to return

        Returns:
            TraversalResult with k-hop subgraph
        """
        result = TraversalResult()
        result.metadata['query_type'] = 'k_hop_neighborhood'
        result.metadata['k'] = k
        result.metadata['root_entity'] = entity_id

        if entity_id not in self.graph:
            logger.warning(f"Entity {entity_id} not in graph")
            return result

        # BFS traversal
        visited = {entity_id}
        queue = deque([(entity_id, 0)])  # (node, depth)

        while queue and len(result.edges) < max_results:
            node, depth = queue.popleft()

            if depth >= k:
                continue

            # Outgoing edges
            for neighbor in self.graph.successors(node):
                if len(result.edges) >= max_results:
                    break
                # Get all edges between node and neighbor
                for key, data in self.graph[node][neighbor].items():
                    if len(result.edges) >= max_results:
                        break
                    # Apply filters
                    if not self._edge_matches_filters(
                        data, time_start, time_end, min_confidence, quad_class
                    ):
                        continue

                    result.add_edge(node, neighbor, key, data)

                    if neighbor not in visited and depth + 1 < k:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

            # Incoming edges
            if len(result.edges) < max_results:
                for predecessor in self.graph.predecessors(node):
                    if len(result.edges) >= max_results:
                        break
                    for key, data in self.graph[predecessor][node].items():
                        if len(result.edges) >= max_results:
                            break
                        if not self._edge_matches_filters(
                            data, time_start, time_end, min_confidence, quad_class
                        ):
                            continue

                        result.add_edge(predecessor, node, key, data)

                        if predecessor not in visited and depth + 1 < k:
                            visited.add(predecessor)
                            queue.append((predecessor, depth + 1))

        return result

    def bilateral_relations(
        self,
        entity1: int,
        entity2: int,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        quad_class: Optional[int] = None
    ) -> TraversalResult:
        """Find all relations between two entities.

        Performance: O(1) with actor-pair index

        Args:
            entity1: First entity
            entity2: Second entity
            time_start: Optional start time filter
            time_end: Optional end time filter
            min_confidence: Minimum edge confidence
            quad_class: Optional QuadClass filter

        Returns:
            TraversalResult with bilateral edges
        """
        result = TraversalResult()
        result.metadata['query_type'] = 'bilateral_relations'
        result.metadata['entity1'] = entity1
        result.metadata['entity2'] = entity2

        # Direct actor-pair index lookup if available
        if self.temporal_index and hasattr(self.temporal_index, 'actor_pair_index'):
            pairs = [
                (entity1, entity2),  # e1 -> e2
                (entity2, entity1)   # e2 -> e1
            ]
            for u, v in pairs:
                pair = (u, v)
                if pair in self.temporal_index.actor_pair_index:
                    for key, timestamp in self.temporal_index.actor_pair_index[pair]:
                        data = self.graph[u][v][key]
                        if self._edge_matches_filters(
                            data, time_start, time_end, min_confidence, quad_class
                        ):
                            result.add_edge(u, v, key, data)
        else:
            # Fallback: direct graph lookup
            if self.graph.has_edge(entity1, entity2):
                for key, data in self.graph[entity1][entity2].items():
                    if self._edge_matches_filters(
                        data, time_start, time_end, min_confidence, quad_class
                    ):
                        result.add_edge(entity1, entity2, key, data)

            if self.graph.has_edge(entity2, entity1):
                for key, data in self.graph[entity2][entity1].items():
                    if self._edge_matches_filters(
                        data, time_start, time_end, min_confidence, quad_class
                    ):
                        result.add_edge(entity2, entity1, key, data)

        return result

    def temporal_paths(
        self,
        source: int,
        target: int,
        max_length: int = 3,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        min_confidence: float = 0.0,
        chronological_only: bool = True
    ) -> TraversalResult:
        """Find temporal paths from source to target.

        Paths respect chronological ordering if chronological_only=True.

        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length (hops)
            time_start: Optional start time filter
            time_end: Optional end time filter
            min_confidence: Minimum edge confidence
            chronological_only: Require edges to be chronologically ordered

        Returns:
            TraversalResult with paths
        """
        result = TraversalResult()
        result.metadata['query_type'] = 'temporal_paths'
        result.metadata['source'] = source
        result.metadata['target'] = target
        result.metadata['max_length'] = max_length

        if source not in self.graph or target not in self.graph:
            return result

        # DFS to find paths
        paths = []
        self._dfs_temporal_paths(
            current=source,
            target=target,
            current_path=[],
            visited=set(),
            paths=paths,
            max_length=max_length,
            time_start=time_start,
            time_end=time_end,
            min_confidence=min_confidence,
            chronological_only=chronological_only,
            last_timestamp=time_start
        )

        # Add paths to result
        for path in paths:
            result.add_path(path)
            for u, v, key in path:
                data = self.graph[u][v][key]
                result.add_edge(u, v, key, data)

        result.metadata['path_count'] = len(paths)

        return result

    def pattern_match(
        self,
        pattern: List[Dict[str, Any]],
        start_entity: Optional[int] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        max_matches: int = 100
    ) -> TraversalResult:
        """Find event sequences matching pattern.

        Pattern format: List of edge specifications
        [
            {"relation_type": "threaten", "quad_class": 3},
            {"relation_type": "military_action", "quad_class": 4}
        ]

        Args:
            pattern: List of edge specifications to match
            start_entity: Optional starting entity
            time_start: Optional start time filter
            time_end: Optional end time filter
            max_matches: Maximum matches to return

        Returns:
            TraversalResult with matching sequences
        """
        result = TraversalResult()
        result.metadata['query_type'] = 'pattern_match'
        result.metadata['pattern_length'] = len(pattern)
        result.metadata['matches'] = 0

        if not pattern:
            return result

        # Determine starting entities
        start_entities = [start_entity] if start_entity else list(self.graph.nodes())

        matches = []
        for entity in start_entities:
            if len(matches) >= max_matches:
                break

            # Try to match pattern starting from this entity
            self._match_pattern_from_entity(
                entity=entity,
                pattern=pattern,
                current_path=[],
                matches=matches,
                time_start=time_start,
                time_end=time_end,
                max_matches=max_matches
            )

        # Add matches to result
        for match in matches:
            result.add_path(match)
            for u, v, key in match:
                data = self.graph[u][v][key]
                result.add_edge(u, v, key, data)

        result.metadata['matches'] = len(matches)

        return result

    def rank_results(
        self,
        result: TraversalResult,
        ranking: str = "confidence"
    ) -> TraversalResult:
        """Rank traversal results by specified metric.

        Args:
            result: Traversal result to rank
            ranking: Ranking method ("confidence", "recency", "mentions")

        Returns:
            Ranked result
        """
        if ranking == "confidence":
            result.edges.sort(key=lambda x: x[3].get('confidence', 0.0), reverse=True)
        elif ranking == "recency":
            result.edges.sort(key=lambda x: x[3].get('timestamp', ''), reverse=True)
        elif ranking == "mentions":
            result.edges.sort(key=lambda x: x[3].get('num_mentions', 0), reverse=True)

        return result

    # Helper methods

    def _edge_matches_filters(
        self,
        edge_data: Dict,
        time_start: Optional[str],
        time_end: Optional[str],
        min_confidence: float,
        quad_class: Optional[int]
    ) -> bool:
        """Check if edge matches filters."""
        # Time filter
        if time_start or time_end:
            timestamp = edge_data.get('timestamp', '')
            if time_start and timestamp < time_start:
                return False
            if time_end and timestamp > time_end:
                return False

        # Confidence filter
        if edge_data.get('confidence', 0.0) < min_confidence:
            return False

        # QuadClass filter
        if quad_class is not None and edge_data.get('quad_class') != quad_class:
            return False

        return True

    def _dfs_temporal_paths(
        self,
        current: int,
        target: int,
        current_path: List[Tuple[int, int, int]],
        visited: Set[int],
        paths: List[List[Tuple[int, int, int]]],
        max_length: int,
        time_start: Optional[str],
        time_end: Optional[str],
        min_confidence: float,
        chronological_only: bool,
        last_timestamp: Optional[str]
    ):
        """DFS helper for temporal path finding."""
        if current == target and current_path:
            paths.append(current_path[:])
            return

        if len(current_path) >= max_length:
            return

        visited.add(current)

        # Explore neighbors
        for neighbor in self.graph.successors(current):
            for key, data in self.graph[current][neighbor].items():
                # Check filters
                if not self._edge_matches_filters(
                    data, time_start, time_end, min_confidence, None
                ):
                    continue

                # Check chronological ordering
                timestamp = data.get('timestamp', '')
                if chronological_only and last_timestamp:
                    if timestamp < last_timestamp:
                        continue

                # Avoid cycles
                if neighbor in visited and neighbor != target:
                    continue

                # Add edge to path
                current_path.append((current, neighbor, key))

                # Recurse
                self._dfs_temporal_paths(
                    current=neighbor,
                    target=target,
                    current_path=current_path,
                    visited=visited.copy() if neighbor != target else visited,
                    paths=paths,
                    max_length=max_length,
                    time_start=time_start,
                    time_end=time_end,
                    min_confidence=min_confidence,
                    chronological_only=chronological_only,
                    last_timestamp=timestamp if chronological_only else last_timestamp
                )

                # Backtrack
                current_path.pop()

        visited.discard(current)

    def _match_pattern_from_entity(
        self,
        entity: int,
        pattern: List[Dict[str, Any]],
        current_path: List[Tuple[int, int, int]],
        matches: List[List[Tuple[int, int, int]]],
        time_start: Optional[str],
        time_end: Optional[str],
        max_matches: int
    ):
        """Recursively match pattern from entity."""
        if len(current_path) == len(pattern):
            # Complete match
            matches.append(current_path[:])
            return

        if len(matches) >= max_matches:
            return

        pattern_idx = len(current_path)
        pattern_spec = pattern[pattern_idx]

        # Try all outgoing edges
        for neighbor in self.graph.successors(entity):
            for key, data in self.graph[entity][neighbor].items():
                # Check if edge matches pattern specification
                if not self._edge_matches_pattern_spec(
                    data, pattern_spec, time_start, time_end
                ):
                    continue

                # Add to path and recurse
                current_path.append((entity, neighbor, key))

                self._match_pattern_from_entity(
                    entity=neighbor,
                    pattern=pattern,
                    current_path=current_path,
                    matches=matches,
                    time_start=time_start,
                    time_end=time_end,
                    max_matches=max_matches
                )

                current_path.pop()

    def _edge_matches_pattern_spec(
        self,
        edge_data: Dict,
        spec: Dict[str, Any],
        time_start: Optional[str],
        time_end: Optional[str]
    ) -> bool:
        """Check if edge matches pattern specification."""
        # Relation type
        if 'relation_type' in spec:
            if edge_data.get('relation_type') != spec['relation_type']:
                return False

        # QuadClass
        if 'quad_class' in spec:
            if edge_data.get('quad_class') != spec['quad_class']:
                return False

        # Confidence
        if 'min_confidence' in spec:
            if edge_data.get('confidence', 0.0) < spec['min_confidence']:
                return False

        # Time window
        if time_start or time_end:
            timestamp = edge_data.get('timestamp', '')
            if time_start and timestamp < time_start:
                return False
            if time_end and timestamp > time_end:
                return False

        return True


def create_traversal_engine(
    graph: nx.MultiDiGraph,
    temporal_index=None
) -> GraphTraversal:
    """Create graph traversal engine instance.

    Args:
        graph: NetworkX MultiDiGraph to traverse
        temporal_index: Optional TemporalIndex for optimization

    Returns:
        Initialized GraphTraversal engine
    """
    return GraphTraversal(graph, temporal_index)
