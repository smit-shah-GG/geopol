# Phase 8: Graph Partitioning - Research

**Researched:** 2026-01-30
**Domain:** Large-scale graph partitioning, temporal knowledge graphs, cross-partition query routing
**Confidence:** MEDIUM

## Summary

The core challenge is scaling a NetworkX-based temporal knowledge graph beyond 1M events while preserving query correctness across partition boundaries. Research reveals that NetworkX consumes approximately 100 bytes per node and 100+ bytes per edge in-memory, meaning a 1M-event graph with typical entity fan-out would require 2-4GB RAM minimum. A standard research workstation (32-64GB RAM) can theoretically hold this, but memory pressure from Python object overhead, edge attributes, and concurrent operations makes this fragile.

The recommended approach is a **hybrid temporal-entity partitioning strategy** using:
1. **Primary partitioning by time window** (temporal sharding) - events within the same time window stay together
2. **Secondary partitioning by entity locality** using METIS for entity-dense subgraphs
3. **Boundary entity replication** for entities that appear across partitions (cross-partition bridges)
4. **SQLite-backed partition index** for partition routing and entity resolution lookup

This preserves query correctness by: (a) routing queries to relevant time-window partitions, (b) using replicated boundary entities to answer cross-partition traversals without cross-shard joins, and (c) maintaining a global entity resolution index that maps entity IDs to their home partition(s).

**Primary recommendation:** Implement temporal partitioning with METIS-based entity locality optimization, SQLite partition index for routing, and boundary entity replication for cross-partition query correctness. Target partition size: 100K-200K events per partition for memory efficiency.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `networkx` | >=3.0 | Individual partition graph storage | Already in project, well-tested |
| `pymetis` | 2025.2.2 | Graph partitioning algorithm | METIS is the gold-standard multilevel partitioner |
| `sqlite3` | builtin | Partition index and routing metadata | Already used for events, zero dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `metis` (ctypes) | latest | Alternative to pymetis if compilation fails | NetworkX-native API, pure Python |
| `networkx-metis` | 1.0 | Direct NetworkX integration | If simpler API preferred over pymetis |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Multiple NetworkX partitions | igraph | igraph uses 32 bytes/edge vs 100+ for NetworkX; but requires API rewrite |
| Multiple NetworkX partitions | graph-tool | Fastest option (C++), but harder compilation, Linux-only |
| METIS partitioning | KaHIP | Higher quality for social networks, but more complex |
| SQLite index | Redis | Faster lookups, but adds infrastructure dependency |
| Partition files (GraphML) | SQLite graph tables | Better for truly massive graphs, but significant refactor |

**Installation:**
```bash
pip install pymetis  # Preferred, includes METIS 5.2.1
# OR
pip install metis networkx  # Pure Python alternative via ctypes
```

**Note on pymetis:** Requires C++ compiler. If installation fails on target hardware, fall back to `metis` package (pure Python/ctypes).

## Architecture Patterns

### Recommended Project Structure
```
src/
  knowledge_graph/
    partitioned_graph.py      # PartitionedTemporalGraph main class
    partition_manager.py      # Partition creation, loading, routing
    partition_index.py        # SQLite-backed entity->partition index
    boundary_resolver.py      # Cross-partition entity resolution
    cross_partition_query.py  # Scatter-gather query orchestration

data/
  partitions/
    partition_meta.db         # SQLite: partition index, entity routing
    partition_0/
      graph.graphml           # NetworkX partition
      entities.json           # Boundary entity metadata
    partition_1/
      ...
```

### Pattern 1: Temporal-First Partitioning
**What:** Partition events by time window first, then optimize within windows
**When to use:** Temporal knowledge graphs where queries typically have time constraints
**Why:** Time-window queries only need to load relevant partitions; reduces cross-partition edges

```python
# Source: Research synthesis from temporal graph partitioning literature
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import networkx as nx

def partition_by_time_windows(
    graph: nx.MultiDiGraph,
    window_days: int = 30,
    target_partition_size: int = 150_000
) -> Dict[str, nx.MultiDiGraph]:
    """
    Partition graph by time windows.

    Args:
        graph: Full temporal knowledge graph
        window_days: Days per time partition
        target_partition_size: Target events per partition

    Returns:
        Dict mapping partition_id to subgraph
    """
    partitions = {}

    # Group edges by time window
    edge_windows: Dict[str, List] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        timestamp = data.get('timestamp', '')
        if timestamp:
            # Parse timestamp and bucket by window
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            window_start = dt - timedelta(days=dt.day % window_days)
            window_id = window_start.strftime('%Y-%m-%d')

            if window_id not in edge_windows:
                edge_windows[window_id] = []
            edge_windows[window_id].append((u, v, key, data))

    # Create partition subgraphs
    for window_id, edges in edge_windows.items():
        partition = nx.MultiDiGraph()
        partition.graph['time_window'] = window_id
        partition.graph['edge_count'] = len(edges)

        for u, v, key, data in edges:
            # Copy node attributes
            if u not in partition:
                partition.add_node(u, **graph.nodes[u])
            if v not in partition:
                partition.add_node(v, **graph.nodes[v])
            partition.add_edge(u, v, key=key, **data)

        partitions[window_id] = partition

    return partitions
```

### Pattern 2: METIS Entity Locality Optimization
**What:** Use METIS to minimize edge cuts within time partitions
**When to use:** After temporal partitioning, when partitions are still too large
**Why:** METIS produces balanced partitions with minimal cross-partition edges

```python
# Source: pymetis documentation + NetworkX integration patterns
import pymetis
import numpy as np
from typing import List, Tuple

def optimize_partition_locality(
    graph: nx.MultiDiGraph,
    num_subpartitions: int = 4
) -> Tuple[int, List[int]]:
    """
    Use METIS to partition graph minimizing edge cuts.

    Args:
        graph: NetworkX graph to partition
        num_subpartitions: Number of partitions to create

    Returns:
        Tuple of (edge_cuts, membership_list)
    """
    # Convert to undirected for METIS (required)
    undirected = graph.to_undirected()

    # Build node index mapping
    nodes = list(undirected.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build adjacency list for METIS
    adjacency = []
    for node in nodes:
        neighbors = [node_to_idx[n] for n in undirected.neighbors(node)]
        adjacency.append(np.array(neighbors, dtype=np.int32))

    # Run METIS partitioning
    n_cuts, membership = pymetis.part_graph(
        num_subpartitions,
        adjacency=adjacency
    )

    return n_cuts, membership
```

### Pattern 3: Boundary Entity Replication
**What:** Replicate entities that appear in multiple partitions to avoid cross-partition lookups
**When to use:** For entities with high cross-partition edge counts
**Why:** Enables local query resolution without distributed joins

```python
# Source: Research synthesis from federated graph literature
from dataclasses import dataclass
from typing import Set, Dict

@dataclass
class BoundaryEntity:
    """Entity that appears in multiple partitions."""
    entity_id: str
    home_partition: str  # Primary partition
    replica_partitions: Set[str]  # Where replicated
    edge_count_per_partition: Dict[str, int]

def identify_boundary_entities(
    partitions: Dict[str, nx.MultiDiGraph],
    replication_threshold: int = 10
) -> Dict[str, BoundaryEntity]:
    """
    Identify entities that should be replicated across partitions.

    Args:
        partitions: Dict of partition_id -> graph
        replication_threshold: Min edges in secondary partition to replicate

    Returns:
        Dict of entity_id -> BoundaryEntity
    """
    # Count entity appearances per partition
    entity_partitions: Dict[str, Dict[str, int]] = {}  # entity -> {partition: edge_count}

    for partition_id, graph in partitions.items():
        for node in graph.nodes():
            if node not in entity_partitions:
                entity_partitions[node] = {}

            # Count edges in this partition
            edge_count = graph.in_degree(node) + graph.out_degree(node)
            entity_partitions[node][partition_id] = edge_count

    # Identify boundary entities
    boundary_entities = {}
    for entity_id, partition_counts in entity_partitions.items():
        if len(partition_counts) > 1:
            # Entity appears in multiple partitions
            sorted_partitions = sorted(
                partition_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            home_partition = sorted_partitions[0][0]
            replica_partitions = {
                p for p, count in sorted_partitions[1:]
                if count >= replication_threshold
            }

            if replica_partitions:
                boundary_entities[entity_id] = BoundaryEntity(
                    entity_id=entity_id,
                    home_partition=home_partition,
                    replica_partitions=replica_partitions,
                    edge_count_per_partition=partition_counts
                )

    return boundary_entities
```

### Pattern 4: Scatter-Gather Query Router
**What:** Route queries to relevant partitions, aggregate results
**When to use:** For all cross-partition queries
**Why:** Ensures query correctness equivalent to single-graph query

```python
# Source: Research synthesis from distributed query processing
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class QueryRouter:
    """Routes queries to partitions and aggregates results."""

    def __init__(
        self,
        partition_manager: 'PartitionManager',
        entity_index: 'EntityPartitionIndex',
        max_workers: int = 4
    ):
        self.partition_manager = partition_manager
        self.entity_index = entity_index
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_k_hop_query(
        self,
        entity_id: str,
        k: int,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None
    ) -> 'TraversalResult':
        """
        Execute k-hop neighborhood query across partitions.

        Correctness guarantee: Returns same result as single-graph query
        by querying all partitions containing the entity and merging.
        """
        # Step 1: Identify relevant partitions
        partitions = self.entity_index.get_entity_partitions(entity_id)

        # Step 2: Filter by time window if specified
        if time_start or time_end:
            partitions = self.partition_manager.filter_by_time(
                partitions, time_start, time_end
            )

        # Step 3: Scatter - query each partition in parallel
        futures = []
        for partition_id in partitions:
            future = self.executor.submit(
                self._query_partition,
                partition_id,
                entity_id,
                k,
                time_start,
                time_end
            )
            futures.append(future)

        # Step 4: Gather - merge results
        merged_result = TraversalResult()
        for future in as_completed(futures):
            partial_result = future.result()
            merged_result.merge(partial_result)

        return merged_result

    def _query_partition(
        self,
        partition_id: str,
        entity_id: str,
        k: int,
        time_start: Optional[str],
        time_end: Optional[str]
    ) -> 'TraversalResult':
        """Query single partition."""
        graph = self.partition_manager.load_partition(partition_id)
        traversal = GraphTraversal(graph)
        return traversal.k_hop_neighborhood(
            entity_id=entity_id,
            k=k,
            time_start=time_start,
            time_end=time_end
        )
```

### Anti-Patterns to Avoid
- **Loading all partitions into memory:** Defeats the purpose; load on-demand only
- **Cross-partition joins without boundary entities:** O(n*m) complexity explosion
- **Ignoring temporal locality in partitioning:** Leads to excessive cross-partition queries
- **Single global entity index without caching:** Index lookup becomes bottleneck
- **Synchronous cross-partition queries:** Use parallel scatter-gather pattern

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph partitioning algorithm | Custom balanced partition | pymetis/METIS | NP-hard problem, METIS is state-of-art multilevel solver |
| Entity-to-partition index | In-memory dict | SQLite table | Persistence, crash recovery, SQL query flexibility |
| Cross-partition merge logic | Manual result concatenation | Set union with dedup | Edge cases in duplicate detection |
| Partition file format | Custom binary | GraphML/JSON | Human-readable, debuggable, NetworkX native |
| Memory estimation | Guesswork | `sys.getsizeof` + sampling | Accurate capacity planning |

**Key insight:** The hard problem is partitioning algorithm design (NP-hard). METIS has 30+ years of research. The implementation challenge is the routing/aggregation layer, which is straightforward but must be correct.

## Common Pitfalls

### Pitfall 1: Memory Fragmentation from Partition Loading/Unloading
**What goes wrong:** Repeatedly loading/unloading partitions fragments Python heap, causing OOM even with sufficient total RAM
**Why it happens:** Python's memory allocator doesn't return memory to OS efficiently
**How to avoid:** Use partition LRU cache with fixed size; consider subprocess isolation for partition loading
**Warning signs:** Memory usage grows over time despite stable partition access patterns

### Pitfall 2: Cross-Partition Query Returns Incomplete Results
**What goes wrong:** Entity exists in partition A and B; query only checks A; misses edges in B
**Why it happens:** Entity-to-partition index is stale or incomplete
**How to avoid:** Build complete entity index at partition time; validate index covers all entities
**Warning signs:** Query results differ when running on full vs partitioned graph

### Pitfall 3: Temporal Partition Boundaries Split Related Events
**What goes wrong:** Event sequence spanning midnight gets split across partitions
**Why it happens:** Naive time-bucket partitioning without overlap
**How to avoid:** Add small time overlap at partition boundaries (e.g., 1 hour); deduplicate in query
**Warning signs:** Path queries fail to find paths that exist in full graph

### Pitfall 4: METIS Partitioning on Directed Graph
**What goes wrong:** METIS requires undirected graph; naive conversion doubles edges
**Why it happens:** METIS API limitation not documented clearly
**How to avoid:** Convert to undirected for partitioning only; keep original directed graph for storage
**Warning signs:** "Invalid graph" errors from METIS; 2x expected partition sizes

### Pitfall 5: Boundary Entity Explosion
**What goes wrong:** 50%+ of entities are "boundary" and get replicated everywhere
**Why it happens:** Partitioning strategy doesn't respect entity locality
**How to avoid:** Tune replication threshold; use METIS edge-cut optimization; consider re-partitioning
**Warning signs:** Replicated entity storage exceeds original graph size

### Pitfall 6: Query Performance Regression on Small Graphs
**What goes wrong:** 100K event graph queries slower with partitioning than without
**Why it happens:** Partition overhead (index lookup, file I/O) exceeds single-graph query time
**How to avoid:** Only partition when graph exceeds threshold (e.g., 500K events); bypass partitioning for small graphs
**Warning signs:** p95 latency increases after partitioning deployment

## Code Examples

### Creating Partition Index in SQLite
```python
# Source: Standard SQLite pattern for entity indexing
import sqlite3
from pathlib import Path
from typing import List, Set

class EntityPartitionIndex:
    """SQLite-backed index mapping entities to their partitions."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entity_partitions (
        entity_id TEXT NOT NULL,
        partition_id TEXT NOT NULL,
        is_home INTEGER DEFAULT 1,
        edge_count INTEGER DEFAULT 0,
        PRIMARY KEY (entity_id, partition_id)
    );

    CREATE INDEX IF NOT EXISTS idx_entity
        ON entity_partitions(entity_id);

    CREATE INDEX IF NOT EXISTS idx_partition
        ON entity_partitions(partition_id);

    CREATE TABLE IF NOT EXISTS partition_meta (
        partition_id TEXT PRIMARY KEY,
        time_window_start TEXT,
        time_window_end TEXT,
        node_count INTEGER,
        edge_count INTEGER,
        file_path TEXT,
        created_at TEXT
    );
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def register_partition(
        self,
        partition_id: str,
        entities: Set[str],
        time_window: tuple,
        stats: dict,
        file_path: str
    ):
        """Register partition and its entities in index."""
        cursor = self.conn.cursor()

        # Insert partition metadata
        cursor.execute("""
            INSERT OR REPLACE INTO partition_meta
            (partition_id, time_window_start, time_window_end,
             node_count, edge_count, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            partition_id,
            time_window[0],
            time_window[1],
            stats['nodes'],
            stats['edges'],
            file_path
        ))

        # Insert entity mappings
        for entity_id in entities:
            cursor.execute("""
                INSERT OR IGNORE INTO entity_partitions
                (entity_id, partition_id, is_home)
                VALUES (?, ?, 1)
            """, (entity_id, partition_id))

        self.conn.commit()

    def get_entity_partitions(self, entity_id: str) -> List[str]:
        """Get all partitions containing an entity."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT partition_id FROM entity_partitions
            WHERE entity_id = ?
            ORDER BY is_home DESC, edge_count DESC
        """, (entity_id,))
        return [row[0] for row in cursor.fetchall()]

    def get_partitions_in_time_range(
        self,
        time_start: str,
        time_end: str
    ) -> List[str]:
        """Get partitions overlapping time range."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT partition_id FROM partition_meta
            WHERE time_window_start <= ? AND time_window_end >= ?
            ORDER BY time_window_start
        """, (time_end, time_start))
        return [row[0] for row in cursor.fetchall()]
```

### Memory-Aware Partition Loading
```python
# Source: Best practice for memory-constrained environments
import gc
import sys
from functools import lru_cache
from pathlib import Path
import networkx as nx

class PartitionManager:
    """Manages loading/unloading of graph partitions."""

    def __init__(
        self,
        partition_dir: Path,
        max_cached_partitions: int = 4,
        max_memory_mb: int = 8192  # 8GB default
    ):
        self.partition_dir = partition_dir
        self.max_cached = max_cached_partitions
        self.max_memory_mb = max_memory_mb
        self._cache = {}
        self._access_order = []

    def load_partition(self, partition_id: str) -> nx.MultiDiGraph:
        """Load partition with LRU eviction."""
        if partition_id in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(partition_id)
            self._access_order.append(partition_id)
            return self._cache[partition_id]

        # Evict if necessary
        while len(self._cache) >= self.max_cached:
            self._evict_oldest()

        # Check memory before loading
        self._check_memory_pressure()

        # Load from disk
        file_path = self.partition_dir / partition_id / "graph.graphml"
        graph = nx.read_graphml(file_path, force_multigraph=True)

        self._cache[partition_id] = graph
        self._access_order.append(partition_id)

        return graph

    def _evict_oldest(self):
        """Evict least recently used partition."""
        if not self._access_order:
            return

        oldest = self._access_order.pop(0)
        if oldest in self._cache:
            del self._cache[oldest]
            gc.collect()  # Encourage memory release

    def _check_memory_pressure(self):
        """Check if memory usage is too high."""
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        if memory_mb > self.max_memory_mb * 0.9:
            # Evict half of cache
            evict_count = len(self._cache) // 2
            for _ in range(evict_count):
                self._evict_oldest()
            gc.collect()

    def estimate_partition_memory(self, partition_id: str) -> float:
        """Estimate memory needed for partition in MB."""
        file_path = self.partition_dir / partition_id / "graph.graphml"
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # GraphML is verbose; actual memory ~2-3x smaller typically
        # But NetworkX overhead adds ~2x
        # Net estimate: roughly equal to file size
        return file_size_mb
```

### Verifying Query Correctness
```python
# Source: Testing pattern for partition correctness
def verify_partition_query_correctness(
    full_graph: nx.MultiDiGraph,
    partitioned_graph: 'PartitionedTemporalGraph',
    sample_queries: int = 100
) -> dict:
    """
    Verify partitioned queries return same results as full graph.

    This is the critical validation for SCALE-02 requirement.
    """
    import random

    results = {
        'total_queries': sample_queries,
        'correct': 0,
        'incorrect': 0,
        'failures': []
    }

    # Sample entities
    entities = list(full_graph.nodes())
    sample_entities = random.sample(
        entities,
        min(sample_queries, len(entities))
    )

    for entity in sample_entities:
        # Query full graph
        full_traversal = GraphTraversal(full_graph)
        full_result = full_traversal.k_hop_neighborhood(
            entity_id=entity,
            k=2
        )

        # Query partitioned graph
        partitioned_result = partitioned_graph.k_hop_neighborhood(
            entity_id=entity,
            k=2
        )

        # Compare results
        full_nodes = full_result.nodes
        partitioned_nodes = partitioned_result.nodes

        if full_nodes == partitioned_nodes:
            results['correct'] += 1
        else:
            results['incorrect'] += 1
            results['failures'].append({
                'entity': entity,
                'full_nodes': len(full_nodes),
                'partitioned_nodes': len(partitioned_nodes),
                'missing': full_nodes - partitioned_nodes,
                'extra': partitioned_nodes - full_nodes
            })

    results['accuracy'] = results['correct'] / results['total_queries']
    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single in-memory NetworkX graph | Partitioned with disk-backed index | N/A (new for this project) | Enables 1M+ event scale |
| Random partitioning | METIS multilevel k-way | 2000s (METIS matured) | 10x fewer edge cuts |
| Synchronous cross-partition queries | Parallel scatter-gather | 2020s (common pattern) | Sub-linear latency scaling |
| Full entity replication | Selective boundary replication | Current best practice | 5-10x storage efficiency |

**Deprecated/outdated:**
- **Spectral partitioning** for large graphs: O(n^3) doesn't scale; METIS is O(n log n)
- **Single-partition GraphML files**: GraphML is slow for large graphs; consider Parquet for 10M+ events
- **In-memory entity index**: Doesn't survive restarts; SQLite is standard

## Open Questions

1. **Optimal partition size for target hardware**
   - What we know: NetworkX uses ~100 bytes/node + ~100 bytes/edge minimum
   - What's unclear: Exact memory overhead with edge attributes (timestamp, confidence, etc.)
   - Recommendation: Benchmark with real data; start with 100K events/partition; adjust based on profiling

2. **Time window granularity**
   - What we know: 30-day windows are common for geopolitical analysis
   - What's unclear: Whether weekly vs monthly windows affect query patterns significantly
   - Recommendation: Make configurable; default to 30 days; expose as partition parameter

3. **Handling partition splits during growth**
   - What we know: METIS can re-partition; but invalidates entity index
   - What's unclear: Online re-partitioning strategy vs batch rebuild
   - Recommendation: Start with offline rebuild; defer online partitioning to post-v1.0

4. **Alternative to NetworkX for partitions**
   - What we know: igraph uses 3x less memory; graph-tool is 10x faster
   - What's unclear: Migration cost from current NetworkX-based implementation
   - Recommendation: Profile NetworkX first; migrate individual partitions if memory-bound

## Sources

### Primary (HIGH confidence)
- [pymetis PyPI](https://pypi.org/project/pymetis/) - Version 2025.2.2, API documentation
- [NetworkX-METIS documentation](https://networkx-metis.readthedocs.io/en/latest/reference/generated/nxmetis.partition.html) - Function signatures
- [metis readthedocs](https://metis.readthedocs.io/) - NetworkX integration examples
- Codebase analysis: `src/knowledge_graph/graph_builder.py`, `persistence.py`, `graph_traversal.py`, `query_engine.py`

### Secondary (MEDIUM confidence)
- [NetworkX discuss: Large graph suitability](https://groups.google.com/g/networkx-discuss/c/dmfkwgY2llQ) - Memory characteristics
- [Memgraph NetworkX challenges](https://memgraph.com/blog/data-persistency-large-scale-data-analytics-and-visualizations-biggest-networkx-challenges) - Memory limitations
- [graph-tool performance](https://graph-tool.skewed.de/performance.html) - Alternative benchmarks
- [DDIA Ch6 Partitioning notes](https://notes.shichao.io/dda/ch6/) - Scatter-gather pattern
- [CUTTANA VLDB paper](https://www.vldb.org/pvldb/vol18/p14-hajidehi.pdf) - Streaming graph partitioner research

### Tertiary (LOW confidence)
- [simple-graph-sqlite GitHub](https://github.com/dpapathanasiou/simple-graph) - SQLite graph storage pattern
- WebSearch results on research workstation RAM specs (32-64GB typical)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - pymetis API verified via official docs; NetworkX memory estimates from community sources
- Architecture: MEDIUM - Patterns synthesized from multiple sources; no single authoritative reference for temporal graph partitioning
- Pitfalls: MEDIUM - Based on general distributed systems knowledge; not all validated against temporal KG use case

**Research date:** 2026-01-30
**Valid until:** 60 days (stable problem domain; library APIs unlikely to change)

## Memory Budget Analysis

For planning purposes, assuming a "standard research workstation" with 32-64GB RAM:

| Graph Size | Nodes (est) | Edges | NetworkX Memory | Recommended Partitions |
|------------|-------------|-------|-----------------|------------------------|
| 100K events | 10K | 100K | ~300 MB | 1 (no partitioning) |
| 500K events | 50K | 500K | ~1.5 GB | 1-4 |
| 1M events | 100K | 1M | ~3 GB | 4-8 |
| 5M events | 500K | 5M | ~15 GB | 25-50 |
| 10M events | 1M | 10M | ~30 GB | 50-100 |

**Assumptions:**
- 1:10 node:edge ratio (typical for event graphs)
- ~100 bytes per node, ~200 bytes per edge (with temporal attributes)
- 50% overhead for Python object allocation
- Target: each partition fits in 200-500 MB for safe concurrent loading
