"""
Unified Query Engine for Temporal Knowledge Graphs (Tasks 6 + 7).

Combines all query components with performance optimization and API interface:
- Query result caching with TTL
- Batch query processing
- Query execution profiling
- Concurrent query handling
- Production-ready API interface
"""

import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import Lock
import asyncio

from .query_parser import QueryParser, ParsedQuery, QueryOptimizer
from .graph_traversal import GraphTraversal
from .vector_similarity import VectorSimilaritySearch
from .result_processor import TemporalFilterAggregator, ResultFormatter, FormattedResult

logger = logging.getLogger(__name__)


class QueryCache:
    """LRU cache with TTL for query results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cached entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached result if valid.

        Args:
            key: Cache key

        Returns:
            Cached result or None if miss/expired
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            result, timestamp = self.cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            return result

    def put(self, key: str, value: Any):
        """Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)

            self.cache[key] = (value, time.time())

            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0

            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'max_size': self.max_size
            }


class QueryProfiler:
    """Profile query execution time."""

    def __init__(self):
        """Initialize profiler."""
        self.query_times: List[float] = []
        self.component_times: Dict[str, List[float]] = {}

    def record_query_time(self, duration_ms: float):
        """Record total query time."""
        self.query_times.append(duration_ms)

    def record_component_time(self, component: str, duration_ms: float):
        """Record component execution time."""
        if component not in self.component_times:
            self.component_times[component] = []
        self.component_times[component].append(duration_ms)

    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        import numpy as np

        stats = {}

        if self.query_times:
            stats['total_queries'] = len(self.query_times)
            stats['avg_time_ms'] = np.mean(self.query_times)
            stats['p50_time_ms'] = np.percentile(self.query_times, 50)
            stats['p95_time_ms'] = np.percentile(self.query_times, 95)
            stats['p99_time_ms'] = np.percentile(self.query_times, 99)

        stats['components'] = {}
        for component, times in self.component_times.items():
            if times:
                stats['components'][component] = {
                    'count': len(times),
                    'avg_ms': np.mean(times),
                    'p95_ms': np.percentile(times, 95)
                }

        return stats


class QueryEngine:
    """Unified query engine with optimization and caching."""

    def __init__(
        self,
        graph,
        temporal_index=None,
        vector_store=None,
        embedding_model=None,
        entity_to_id: Optional[Dict[str, int]] = None,
        id_to_entity: Optional[Dict[int, str]] = None,
        relation_to_id: Optional[Dict[str, int]] = None,
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        """Initialize query engine.

        Args:
            graph: NetworkX MultiDiGraph
            temporal_index: TemporalIndex for optimization
            vector_store: VectorStore for similarity search
            embedding_model: Embedding model
            entity_to_id: Entity name to ID mapping
            id_to_entity: Entity ID to name mapping
            relation_to_id: Relation type to ID mapping
            enable_cache: Enable result caching
            cache_ttl: Cache TTL in seconds
        """
        # Core components
        self.graph = graph
        self.temporal_index = temporal_index

        # Initialize modules
        self.parser = QueryParser()
        self.optimizer = QueryOptimizer()
        self.traversal = GraphTraversal(graph, temporal_index)
        self.aggregator = TemporalFilterAggregator()
        self.formatter = ResultFormatter(graph, id_to_entity)

        # Optional vector search
        self.vector_search = None
        if vector_store:
            self.vector_search = VectorSimilaritySearch(
                vector_store=vector_store,
                embedding_model=embedding_model,
                entity_to_id=entity_to_id,
                id_to_entity=id_to_entity,
                relation_to_id=relation_to_id
            )

        # Performance optimization
        self.cache = QueryCache(ttl_seconds=cache_ttl) if enable_cache else None
        self.profiler = QueryProfiler()

        # Entity mappings
        self.entity_to_id = entity_to_id or {}
        self.id_to_entity = id_to_entity or {}

    def execute_query(
        self,
        query: Any,
        use_cache: bool = True,
        profile: bool = True
    ) -> FormattedResult:
        """Execute query end-to-end.

        Args:
            query: Query dict or natural language string
            use_cache: Whether to use cache
            profile: Whether to profile execution

        Returns:
            FormattedResult with complete results
        """
        start_time = time.time()

        # Parse query
        if isinstance(query, str):
            parsed_query = self.parser.parse_natural_language(query)
        elif isinstance(query, dict):
            parsed_query = self.parser.parse_dict(query)
        else:
            parsed_query = query

        # Validate
        self.parser.validate(parsed_query)

        # Check cache
        if use_cache and self.cache:
            cache_key = self._get_cache_key(parsed_query)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for query")
                return cached_result

        # Optimize query plan
        component_start = time.time()
        query_plan = self.optimizer.optimize(parsed_query)
        if profile:
            self.profiler.record_component_time(
                'optimization',
                (time.time() - component_start) * 1000
            )

        # Execute based on query type
        component_start = time.time()
        traversal_result = self._execute_graph_query(parsed_query)
        if profile:
            self.profiler.record_component_time(
                'graph_traversal',
                (time.time() - component_start) * 1000
            )

        # Optional vector search
        similarity_result = None
        if self.vector_search and parsed_query.search_mode in ['semantic', 'hybrid']:
            component_start = time.time()
            similarity_result = self._execute_vector_query(parsed_query)
            if profile:
                self.profiler.record_component_time(
                    'vector_search',
                    (time.time() - component_start) * 1000
                )

        # Format results
        component_start = time.time()
        execution_time_ms = (time.time() - start_time) * 1000

        result = self.formatter.format_query_result(
            query_id=self._generate_query_id(),
            query_type=parsed_query.query_type.value,
            query_params=self._query_to_dict(parsed_query),
            traversal_result=traversal_result,
            similarity_result=similarity_result,
            execution_time_ms=execution_time_ms,
            include_explanations=parsed_query.include_explanations
        )

        if profile:
            self.profiler.record_component_time(
                'formatting',
                (time.time() - component_start) * 1000
            )
            self.profiler.record_query_time(execution_time_ms)

        # Cache result
        if use_cache and self.cache:
            self.cache.put(cache_key, result)

        return result

    async def execute_batch_queries(
        self,
        queries: List[Any],
        use_cache: bool = True
    ) -> List[FormattedResult]:
        """Execute multiple queries concurrently.

        Args:
            queries: List of queries
            use_cache: Whether to use cache

        Returns:
            List of FormattedResult objects
        """
        # Execute queries concurrently using asyncio
        tasks = [
            asyncio.to_thread(self.execute_query, q, use_cache=use_cache)
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, FormattedResult)]

        return valid_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dict with cache and profiler stats
        """
        stats = {
            'profiler': self.profiler.get_stats()
        }

        if self.cache:
            stats['cache'] = self.cache.get_stats()

        return stats

    def clear_cache(self):
        """Clear query cache."""
        if self.cache:
            self.cache.clear()

    # Private methods

    def _execute_graph_query(self, query: ParsedQuery):
        """Execute graph traversal based on query type."""
        from .query_parser import QueryType

        if query.query_type == QueryType.ENTITY_PAIR:
            entity1_id = self._resolve_entity(query.entity1)
            entity2_id = self._resolve_entity(query.entity2)

            return self.traversal.bilateral_relations(
                entity1=entity1_id,
                entity2=entity2_id,
                time_start=query.time_window.start.isoformat() if query.time_window and query.time_window.start else None,
                time_end=query.time_window.end.isoformat() if query.time_window and query.time_window.end else None,
                min_confidence=query.constraints.min_confidence,
                quad_class=query.constraints.quad_class
            )

        elif query.query_type == QueryType.ENTITY_RELATIONS:
            entity_id = self._resolve_entity(query.entity1)

            return self.traversal.k_hop_neighborhood(
                entity_id=entity_id,
                k=query.path_length,
                time_start=query.time_window.start.isoformat() if query.time_window and query.time_window.start else None,
                time_end=query.time_window.end.isoformat() if query.time_window and query.time_window.end else None,
                min_confidence=query.constraints.min_confidence,
                quad_class=query.constraints.quad_class,
                max_results=query.max_results
            )

        elif query.query_type == QueryType.TEMPORAL_PATH:
            source_id = self._resolve_entity(query.entity1)
            target_id = self._resolve_entity(query.entity2)

            return self.traversal.temporal_paths(
                source=source_id,
                target=target_id,
                max_length=query.path_length,
                time_start=query.time_window.start.isoformat() if query.time_window and query.time_window.start else None,
                time_end=query.time_window.end.isoformat() if query.time_window and query.time_window.end else None,
                min_confidence=query.constraints.min_confidence
            )

        elif query.query_type == QueryType.PATTERN_MATCH:
            return self.traversal.pattern_match(
                pattern=query.pattern_sequence,
                time_start=query.time_window.start.isoformat() if query.time_window and query.time_window.start else None,
                time_end=query.time_window.end.isoformat() if query.time_window and query.time_window.end else None,
                max_matches=query.max_results
            )

        else:
            # Default: empty result
            from .graph_traversal import TraversalResult
            return TraversalResult()

    def _execute_vector_query(self, query: ParsedQuery):
        """Execute vector similarity search."""
        if not self.vector_search:
            return None

        if query.entity_similarity_query:
            return self.vector_search.search_similar_entities(
                query=query.entity_similarity_query,
                top_k=query.max_results,
                threshold=query.similarity_threshold
            )

        return None

    def _resolve_entity(self, entity_name: str):
        """Resolve entity name to ID or return entity name if using string IDs."""
        # If we have entity mappings, use them
        if self.entity_to_id and entity_name in self.entity_to_id:
            return self.entity_to_id[entity_name]
        # If entity_to_id is None or empty but entity exists in graph, return the name itself
        elif (not self.entity_to_id or entity_name not in self.entity_to_id) and self.graph and entity_name in self.graph.nodes():
            return entity_name
        # Handle numeric string IDs
        elif entity_name.isdigit():
            return int(entity_name)
        else:
            raise ValueError(f"Unknown entity: {entity_name}")

    def _get_cache_key(self, query: ParsedQuery) -> str:
        """Generate cache key for query."""
        query_dict = self._query_to_dict(query)
        query_json = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_json.encode()).hexdigest()

    def _query_to_dict(self, query: ParsedQuery) -> Dict[str, Any]:
        """Convert query to dictionary."""
        query_dict = {
            'query_type': query.query_type.value,
            'entity1': query.entity1,
            'entity2': query.entity2,
            'search_mode': query.search_mode.value,
            'min_confidence': query.constraints.min_confidence,
            'quad_class': query.constraints.quad_class
        }

        if query.time_window:
            query_dict['time_window'] = {
                'start': query.time_window.start.isoformat() if query.time_window.start else None,
                'end': query.time_window.end.isoformat() if query.time_window.end else None
            }

        return query_dict

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]


def create_query_engine(
    graph,
    temporal_index=None,
    vector_store=None,
    embedding_model=None,
    entity_to_id: Optional[Dict[str, int]] = None,
    id_to_entity: Optional[Dict[int, str]] = None,
    relation_to_id: Optional[Dict[str, int]] = None,
    enable_cache: bool = True
) -> QueryEngine:
    """Create query engine instance.

    Args:
        graph: NetworkX MultiDiGraph
        temporal_index: Optional TemporalIndex
        vector_store: Optional VectorStore
        embedding_model: Optional embedding model
        entity_to_id: Entity mappings
        id_to_entity: Reverse entity mappings
        relation_to_id: Relation mappings
        enable_cache: Enable caching

    Returns:
        Initialized QueryEngine
    """
    return QueryEngine(
        graph=graph,
        temporal_index=temporal_index,
        vector_store=vector_store,
        embedding_model=embedding_model,
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        relation_to_id=relation_to_id,
        enable_cache=enable_cache
    )
