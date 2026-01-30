"""
Partition manager for temporal graph partitioning with LRU caching.

This module handles:
- Creating partitions from a graph using temporal windowing
- Saving partitions as GraphML files
- Loading partitions on-demand with LRU eviction
- Integration with EntityPartitionIndex for entity routing
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .partition_index import EntityPartitionIndex


class PartitionManager:
    """
    Manages graph partition lifecycle: creation, storage, and LRU-cached loading.

    Provides:
    - Temporal partitioning by time windows
    - GraphML persistence per partition
    - LRU cache for loaded partitions (prevents memory exhaustion)
    - Integration with EntityPartitionIndex for routing

    The partition strategy is temporal-first: edges are bucketed by timestamp
    into time windows, and each window becomes a separate partition.
    """

    def __init__(
        self,
        partition_dir: Path,
        index: EntityPartitionIndex,
        max_cached_partitions: int = 4,
        max_memory_mb: int = 8192
    ) -> None:
        """
        Initialize partition manager.

        Args:
            partition_dir: Directory for partition storage. Created if not exists.
            index: EntityPartitionIndex for entity-to-partition mapping.
            max_cached_partitions: Maximum partitions to hold in LRU cache.
            max_memory_mb: Memory limit in MB (for future memory-aware eviction).
        """
        self.partition_dir = Path(partition_dir)
        self.partition_dir.mkdir(parents=True, exist_ok=True)

        self.index = index
        self.max_cached = max_cached_partitions
        self.max_memory_mb = max_memory_mb

        # LRU cache: OrderedDict maintains insertion/access order
        # move_to_end() on access, popitem(last=False) for eviction
        self._cache: OrderedDict[str, nx.MultiDiGraph] = OrderedDict()

    def partition_graph(
        self,
        graph: nx.MultiDiGraph,
        window_days: int = 30
    ) -> List[str]:
        """
        Partition graph by time windows and persist to disk.

        Each partition:
        - Contains edges within a time window
        - Is saved as GraphML in {partition_dir}/{partition_id}/graph.graphml
        - Is registered in EntityPartitionIndex

        Args:
            graph: Full temporal knowledge graph to partition.
            window_days: Days per time window partition.

        Returns:
            List of partition IDs created.
        """
        # Group edges by time window
        edge_windows = self._bucket_edges_by_time(graph, window_days)

        if not edge_windows:
            return []

        partition_ids = []

        for window_id, edges in edge_windows.items():
            # Create partition subgraph
            partition = nx.MultiDiGraph()
            partition.graph['time_window'] = window_id
            partition.graph['edge_count'] = len(edges)

            for u, v, key, data in edges:
                # Copy node attributes if node exists in source graph
                if u not in partition:
                    node_attrs = dict(graph.nodes[u]) if u in graph.nodes else {}
                    partition.add_node(u, **node_attrs)
                if v not in partition:
                    node_attrs = dict(graph.nodes[v]) if v in graph.nodes else {}
                    partition.add_node(v, **node_attrs)
                partition.add_edge(u, v, key=key, **data)

            # Save partition to disk
            partition_path = self._save_partition(window_id, partition)

            # Calculate time window bounds
            time_window = self._calculate_window_bounds(window_id, window_days)

            # Register in index
            entities = set(partition.nodes())
            stats = {
                'nodes': partition.number_of_nodes(),
                'edges': partition.number_of_edges()
            }
            self.index.register_partition(
                partition_id=window_id,
                entities=entities,
                time_window=time_window,
                stats=stats,
                file_path=str(partition_path)
            )

            partition_ids.append(window_id)

        return sorted(partition_ids)

    def _bucket_edges_by_time(
        self,
        graph: nx.MultiDiGraph,
        window_days: int
    ) -> Dict[str, List[Tuple]]:
        """
        Group edges by time window based on timestamp attribute.

        Args:
            graph: Source graph with timestamped edges.
            window_days: Days per window.

        Returns:
            Dict mapping window_id to list of (u, v, key, data) tuples.
        """
        edge_windows: Dict[str, List[Tuple]] = {}
        no_timestamp_key = "no-timestamp"

        for u, v, key, data in graph.edges(keys=True, data=True):
            timestamp = data.get('timestamp')

            if timestamp:
                try:
                    # Parse timestamp - handle various ISO formats
                    ts_str = str(timestamp)
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1] + '+00:00'

                    dt = datetime.fromisoformat(ts_str)

                    # Calculate window start: align to window_days boundary
                    # Using day-of-year modulo for consistent bucketing
                    year_start = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
                    days_since_year_start = (dt - year_start).days
                    window_offset = (days_since_year_start // window_days) * window_days
                    window_start = year_start + timedelta(days=window_offset)
                    window_id = window_start.strftime('%Y-%m-%d')

                except (ValueError, TypeError):
                    # Invalid timestamp format - assign to no-timestamp partition
                    window_id = no_timestamp_key
            else:
                # No timestamp - assign to fallback partition
                window_id = no_timestamp_key

            if window_id not in edge_windows:
                edge_windows[window_id] = []
            edge_windows[window_id].append((u, v, key, data))

        return edge_windows

    def _calculate_window_bounds(
        self,
        window_id: str,
        window_days: int
    ) -> Tuple[str, str]:
        """
        Calculate time window start and end from window_id.

        Args:
            window_id: Window identifier (YYYY-MM-DD format or 'no-timestamp').
            window_days: Days per window.

        Returns:
            Tuple of (start, end) ISO timestamps.
        """
        if window_id == "no-timestamp":
            # Special partition for edges without timestamps
            return ("", "")

        try:
            window_start = datetime.fromisoformat(window_id)
            window_end = window_start + timedelta(days=window_days) - timedelta(seconds=1)
            return (
                window_start.isoformat(),
                window_end.isoformat()
            )
        except ValueError:
            return ("", "")

    def _save_partition(
        self,
        partition_id: str,
        graph: nx.MultiDiGraph
    ) -> Path:
        """
        Save partition graph to GraphML file.

        Args:
            partition_id: Unique partition identifier.
            graph: Partition graph to save.

        Returns:
            Path to saved GraphML file.
        """
        partition_path = self.partition_dir / partition_id
        partition_path.mkdir(parents=True, exist_ok=True)

        file_path = partition_path / "graph.graphml"
        nx.write_graphml(graph, str(file_path))

        return file_path

    def load_partition(self, partition_id: str) -> nx.MultiDiGraph:
        """
        Load partition with LRU caching.

        On cache hit: move to end of access order, return cached graph.
        On cache miss: evict oldest if at capacity, load from disk, cache.

        Args:
            partition_id: Partition to load.

        Returns:
            Loaded partition graph.

        Raises:
            FileNotFoundError: If partition GraphML file doesn't exist.
        """
        # Cache hit: move to end (most recently used)
        if partition_id in self._cache:
            self._cache.move_to_end(partition_id)
            return self._cache[partition_id]

        # Cache miss: evict if at capacity
        while len(self._cache) >= self.max_cached:
            self._evict_oldest()

        # Load from disk
        file_path = self.partition_dir / partition_id / "graph.graphml"
        if not file_path.exists():
            raise FileNotFoundError(f"Partition file not found: {file_path}")

        graph = nx.read_graphml(str(file_path), force_multigraph=True)

        # Add to cache
        self._cache[partition_id] = graph

        return graph

    def _evict_oldest(self) -> None:
        """
        Evict least recently used partition from cache.

        Calls gc.collect() after eviction to encourage memory release
        (Python memory fragmentation pitfall mitigation).
        """
        if not self._cache:
            return

        # popitem(last=False) removes oldest (first inserted/accessed)
        evicted_id, _ = self._cache.popitem(last=False)
        gc.collect()  # Encourage memory release

    def get_partition_path(self, partition_id: str) -> Path:
        """
        Get path to partition's GraphML file.

        Args:
            partition_id: Partition to get path for.

        Returns:
            Path to GraphML file (may not exist if partition not created).
        """
        return self.partition_dir / partition_id / "graph.graphml"

    def list_partitions(self) -> List[str]:
        """
        List all partition IDs from the index.

        Returns:
            List of all registered partition IDs.
        """
        return self.index.list_all_partitions()

    def is_cached(self, partition_id: str) -> bool:
        """
        Check if partition is currently in cache.

        Args:
            partition_id: Partition to check.

        Returns:
            True if partition is cached, False otherwise.
        """
        return partition_id in self._cache

    def clear_cache(self) -> None:
        """Clear all cached partitions and run garbage collection."""
        self._cache.clear()
        gc.collect()

    def partition_by_time_windows(
        self,
        graph: nx.MultiDiGraph,
        window_days: int = 30
    ) -> List[str]:
        """
        Alias for partition_graph() for API compatibility with RESEARCH.md pattern.

        Args:
            graph: Full temporal knowledge graph to partition.
            window_days: Days per time window partition.

        Returns:
            List of partition IDs created.
        """
        return self.partition_graph(graph, window_days)
