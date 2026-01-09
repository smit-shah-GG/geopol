"""
Tests for Unified Query Engine (Tasks 6 + 7).

Tests caching, profiling, batch execution, and end-to-end query flow.
"""

import pytest
import networkx as nx
import asyncio
from unittest.mock import Mock
from src.knowledge_graph.query_engine import (
    QueryEngine, QueryCache, QueryProfiler,
    create_query_engine
)


@pytest.fixture
def simple_graph():
    """Create simple test graph."""
    G = nx.MultiDiGraph()

    for i in range(1, 5):
        G.add_node(i, entity_type='country', name=f'Entity{i}')

    G.add_edge(1, 2, key=0, timestamp='2024-01-01T00:00:00Z',
               confidence=0.8, quad_class=1, relation_type='diplomatic',
               num_mentions=100)
    G.add_edge(2, 3, key=0, timestamp='2024-01-05T00:00:00Z',
               confidence=0.9, quad_class=4, relation_type='conflict',
               num_mentions=200)

    return G


class TestQueryCache:
    """Test query caching."""

    def test_cache_get_put(self):
        """Test basic cache operations."""
        cache = QueryCache(max_size=10, ttl_seconds=60)

        # Miss
        assert cache.get('key1') is None

        # Put and hit
        cache.put('key1', {'result': 'data'})
        assert cache.get('key1') == {'result': 'data'}

    def test_cache_ttl(self):
        """Test TTL expiration."""
        import time
        cache = QueryCache(max_size=10, ttl_seconds=1)

        cache.put('key1', 'value')
        assert cache.get('key1') == 'value'

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get('key1') is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        cache = QueryCache(max_size=2)

        cache.put('key1', 'val1')
        cache.put('key2', 'val2')
        cache.put('key3', 'val3')  # Should evict key1

        assert cache.get('key1') is None  # Evicted
        assert cache.get('key2') == 'val2'
        assert cache.get('key3') == 'val3'

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = QueryCache()

        cache.put('key1', 'val1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss

        stats = cache.get_stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['size'] == 1


class TestQueryProfiler:
    """Test query profiling."""

    def test_record_query_time(self):
        """Test recording query times."""
        profiler = QueryProfiler()

        profiler.record_query_time(10.5)
        profiler.record_query_time(15.2)
        profiler.record_query_time(8.7)

        stats = profiler.get_stats()

        assert stats['total_queries'] == 3
        assert 'avg_time_ms' in stats
        assert 'p95_time_ms' in stats

    def test_record_component_time(self):
        """Test recording component times."""
        profiler = QueryProfiler()

        profiler.record_component_time('parsing', 1.0)
        profiler.record_component_time('parsing', 1.5)
        profiler.record_component_time('execution', 10.0)

        stats = profiler.get_stats()

        assert 'components' in stats
        assert 'parsing' in stats['components']
        assert stats['components']['parsing']['count'] == 2


class TestQueryEngine:
    """Test unified query engine."""

    def test_execute_entity_pair_query(self, simple_graph):
        """Test executing entity pair query."""
        entity_to_id = {'Entity1': 1, 'Entity2': 2}
        id_to_entity = {1: 'Entity1', 2: 'Entity2'}

        engine = create_query_engine(
            graph=simple_graph,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity,
            enable_cache=False
        )

        query = {
            'actor1': 'Entity1',
            'actor2': 'Entity2'
        }

        result = engine.execute_query(query, use_cache=False)

        assert result is not None
        assert result.query_type == 'entity_pair'
        assert result.execution_time_ms >= 0

    def test_execute_natural_language_query(self, simple_graph):
        """Test natural language query execution."""
        # Natural language parser uppercases entity names
        entity_to_id = {'ENTITY1': 1, 'ENTITY2': 2}
        id_to_entity = {1: 'ENTITY1', 2: 'ENTITY2'}

        engine = create_query_engine(
            graph=simple_graph,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity,
            enable_cache=False
        )

        result = engine.execute_query(
            "relations with Entity1",
            use_cache=False
        )

        assert result is not None
        assert result.query_type == 'entity_relations'

    def test_query_caching(self, simple_graph):
        """Test query result caching."""
        entity_to_id = {'Entity1': 1, 'Entity2': 2}
        id_to_entity = {1: 'Entity1', 2: 'Entity2'}

        engine = create_query_engine(
            graph=simple_graph,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity,
            enable_cache=True
        )

        query = {'actor1': 'Entity1', 'actor2': 'Entity2'}

        # First execution
        result1 = engine.execute_query(query, use_cache=True)

        # Second execution (should hit cache)
        result2 = engine.execute_query(query, use_cache=True)

        # Check cache stats
        stats = engine.get_performance_stats()
        assert stats['cache']['hits'] >= 1

    def test_profiling(self, simple_graph):
        """Test query profiling."""
        entity_to_id = {'Entity1': 1}
        id_to_entity = {1: 'Entity1'}

        engine = create_query_engine(
            graph=simple_graph,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        query = {'entity1': 'Entity1'}

        engine.execute_query(query, profile=True)

        stats = engine.get_performance_stats()

        assert 'profiler' in stats
        assert stats['profiler']['total_queries'] >= 1

    def test_batch_queries(self, simple_graph):
        """Test batch query execution."""
        entity_to_id = {'Entity1': 1, 'Entity2': 2}
        id_to_entity = {1: 'Entity1', 2: 'Entity2'}

        engine = create_query_engine(
            graph=simple_graph,
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity
        )

        queries = [
            {'entity1': 'Entity1'},
            {'actor1': 'Entity1', 'actor2': 'Entity2'}
        ]

        # Run async in sync context
        results = asyncio.run(engine.execute_batch_queries(queries))

        assert len(results) >= 1
        assert all(hasattr(r, 'query_id') for r in results)

    def test_clear_cache(self, simple_graph):
        """Test cache clearing."""
        engine = create_query_engine(
            graph=simple_graph,
            entity_to_id={'Entity1': 1},
            id_to_entity={1: 'Entity1'},
            enable_cache=True
        )

        query = {'entity1': 'Entity1'}
        engine.execute_query(query)

        # Clear cache
        engine.clear_cache()

        stats = engine.get_performance_stats()
        assert stats['cache']['size'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
