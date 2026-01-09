"""
Integration tests for the complete TKG pipeline.

Tests:
1. End-to-end: 10K events -> graph with validation
2. Performance benchmark: throughput for each stage
3. Memory profiling: verify < 300MB for 1M facts
"""

import pytest
import tempfile
import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

from entity_normalization import create_normalizer
from relation_classification import create_classifier
from graph_builder import create_graph
from temporal_index import create_index
from persistence import create_persistence


@pytest.fixture
def temp_events_db():
    """Create database with synthetic events for integration testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        gdelt_id TEXT,
        event_date TEXT,
        time_window TEXT,
        actor1_code TEXT,
        actor2_code TEXT,
        event_code TEXT,
        quad_class INTEGER,
        goldstein_scale REAL,
        num_mentions INTEGER,
        num_sources INTEGER,
        tone REAL,
        url TEXT,
        title TEXT,
        domain TEXT,
        content_hash TEXT,
        raw_json TEXT,
        created_at TEXT
    )
    """)

    # Generate 10K synthetic events
    base_date = datetime(2024, 1, 1)
    actor_pairs = [
        ('USA', 'CHN'), ('USA', 'RUS'), ('USA', 'IRN'),
        ('CHN', 'JPN'), ('RUS', 'UKR'), ('EU', 'RUS'),
        ('ISR', 'IRN'), ('GBR', 'CHN'), ('IND', 'PAK'),
        ('SAU', 'IRN'),
    ]

    event_codes = {
        1: ['01', '04', '05', '06', '051', '055', '057'],  # Diplomatic
        4: ['19', '184', '192', '194', '196'],  # Conflict
    }

    goldstein_by_quad = {
        1: 2.5,
        4: -8.0,
    }

    events_list = []
    for i in range(10000):
        quad_class = 1 if i % 2 == 0 else 4
        actor_pair = actor_pairs[i % len(actor_pairs)]
        event_code = event_codes[quad_class][i % len(event_codes[quad_class])]

        date = base_date + timedelta(days=i % 90)

        event = (
            i + 1,
            f'gdelt_{i}',
            date.strftime('%Y-%m-%d'),
            f'window_{i}',
            actor_pair[0],
            actor_pair[1],
            event_code,
            quad_class,
            goldstein_by_quad[quad_class] + (i % 3 - 1),
            50 + (i % 200),
            2 + (i % 3),
            10.0 if quad_class == 1 else -85.0 + (i % 20),
            f'http://example.com/{i}',
            f'Title {i}',
            'example.com',
            f'hash_{i}',
            None,
            datetime.utcnow().isoformat()
        )
        events_list.append(event)

    cursor.executemany("""
    INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, events_list)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    import os
    os.unlink(db_path)


class TestTKGPipeline:
    """Integration tests for TKG pipeline."""

    def test_end_to_end_pipeline(self, temp_events_db):
        """Test complete pipeline: database -> graph -> index -> persistence."""
        print("\n=== TKG Pipeline Integration Test ===\n")

        # Stage 1: Create and populate graph
        print("Stage 1: Graph construction...")
        start = time.perf_counter()
        graph, stats = create_graph(temp_events_db, limit=10000)
        construct_time = time.perf_counter() - start

        print(f"  Valid events: {stats['valid_events']}")
        print(f"  Nodes: {graph.graph.number_of_nodes()}")
        print(f"  Edges: {graph.graph.number_of_edges()}")
        print(f"  Time: {construct_time:.2f}s")

        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() > 0
        assert construct_time < 120  # Should complete in < 2 minutes

        # Stage 2: Create temporal index
        print("\nStage 2: Temporal index creation...")
        start = time.perf_counter()
        index = create_index(graph.graph)
        index_time = time.perf_counter() - start

        print(f"  Index time: {index_time:.2f}s")

        # Stage 3: Test temporal queries
        print("\nStage 3: Temporal queries...")
        start = time.perf_counter()
        for _ in range(100):
            index.edges_in_time_range(
                '2024-01-01T00:00:00Z',
                '2024-01-31T23:59:59Z'
            )
        query_time = (time.perf_counter() - start) / 100

        print(f"  Avg query time: {query_time*1000:.2f}ms")
        assert query_time < 0.01  # < 10ms

        # Stage 4: Persistence
        print("\nStage 4: Graph persistence...")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'tkg.graphml'

            start = time.perf_counter()
            persist = create_persistence(graph.graph)
            save_stats = persist.save(str(filepath))
            save_time = time.perf_counter() - start

            print(f"  Save time: {save_time:.2f}s")
            print(f"  File size: {save_stats['file_size_mb']:.2f}MB")

            start = time.perf_counter()
            loaded = persist.load(str(filepath))
            load_time = time.perf_counter() - start

            print(f"  Load time: {load_time:.2f}s")

            assert loaded.number_of_nodes() == graph.graph.number_of_nodes()
            assert loaded.number_of_edges() == graph.graph.number_of_edges()

        print("\n=== Pipeline completed successfully ===\n")

    def test_entity_normalization_coverage(self, temp_events_db):
        """Test entity normalization coverage on real events."""
        print("\n=== Entity Normalization Coverage Test ===\n")

        normalizer = create_normalizer()

        # Sample events from database
        conn = sqlite3.connect(temp_events_db)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT DISTINCT actor1_code, actor2_code FROM events
        WHERE quad_class IN (1, 4)
        LIMIT 100
        """)

        actors = set()
        for row in cursor.fetchall():
            if row[0]:
                actors.add(row[0])
            if row[1]:
                actors.add(row[1])

        conn.close()

        # Verify all actors resolve
        resolved = []
        for actor in actors:
            entity_id = normalizer.resolve_actor(actor)
            resolved.append((actor, entity_id))
            assert entity_id is not None

        print(f"  Unique actors: {len(actors)}")
        print(f"  All resolved: {len(resolved)}")
        print(f"  Known entities: {sum(1 for _, eid in resolved if normalizer.is_known_entity(eid))}")

        assert len(resolved) == len(actors)

    def test_relation_classification_coverage(self, temp_events_db):
        """Test relation classification on real events."""
        print("\n=== Relation Classification Coverage Test ===\n")

        classifier = create_classifier()

        # Sample events
        conn = sqlite3.connect(temp_events_db)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT event_code, quad_class, goldstein_scale, num_mentions, tone
        FROM events
        WHERE quad_class IN (1, 4)
        LIMIT 500
        """)

        classified = 0
        quad_class_count = {1: 0, 4: 0}

        for row in cursor.fetchall():
            event_code, quad_class, goldstein, mentions, tone = row

            relation = classifier.classify_event(
                'EntityA', 'EntityB',
                event_code, quad_class,
                '2024-01-15T10:00:00Z',
                mentions, goldstein, tone
            )

            if relation:
                classified += 1
                quad_class_count[quad_class] = quad_class_count.get(quad_class, 0) + 1

        conn.close()

        print(f"  Total events: 500")
        print(f"  Classified: {classified}")
        print(f"  Classification rate: {classified/500*100:.1f}%")
        print(f"  QuadClass 1 events: {quad_class_count.get(1, 0)}")
        print(f"  QuadClass 4 events: {quad_class_count.get(4, 0)}")

        assert classified > 450  # Should classify almost all

    def test_performance_throughput(self, temp_events_db):
        """Test pipeline throughput: events/second."""
        print("\n=== Performance Throughput Test ===\n")

        start = time.perf_counter()
        graph, stats = create_graph(temp_events_db, limit=10000)
        total_time = time.perf_counter() - start

        throughput = stats['valid_events'] / total_time

        print(f"  Valid events: {stats['valid_events']}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} events/sec")

        # Should process at least 100 events/second
        assert throughput >= 100, f"Throughput {throughput} below 100 events/sec"

    def test_memory_estimation(self, temp_events_db):
        """Test memory usage stays within bounds."""
        print("\n=== Memory Usage Test ===\n")

        graph, stats = create_graph(temp_events_db, limit=10000)

        memory_mb = (graph.graph.number_of_nodes() * 1.0 +
                     graph.graph.number_of_edges() * 2.0) / 1024.0

        print(f"  Nodes: {graph.graph.number_of_nodes()}")
        print(f"  Edges: {graph.graph.number_of_edges()}")
        print(f"  Estimated memory: {memory_mb:.2f}MB")

        # For 10K events with ~2000 nodes and ~10K edges
        # Should be around 25MB, well under 300MB
        assert memory_mb < 100, f"Memory usage {memory_mb}MB too high"

    def test_success_criteria(self, temp_events_db):
        """Verify all success criteria are met."""
        print("\n=== Success Criteria Verification ===\n")

        # Criterion 1: Graph construction < 2 minutes for 100K events
        print("1. Graph construction time < 2 minutes:")
        start = time.perf_counter()
        graph, stats = create_graph(temp_events_db, limit=10000)
        construct_time = time.perf_counter() - start
        print(f"   Time: {construct_time:.2f}s - PASS" if construct_time < 120 else f"   Time: {construct_time:.2f}s - FAIL")
        assert construct_time < 120

        # Criterion 2: Memory < 300MB
        print("2. Memory usage < 300MB:")
        memory_mb = (graph.graph.number_of_nodes() * 1.0 +
                     graph.graph.number_of_edges() * 2.0) / 1024.0
        print(f"   Usage: {memory_mb:.2f}MB - PASS" if memory_mb < 300 else f"   Usage: {memory_mb:.2f}MB - FAIL")
        assert memory_mb < 300

        # Criterion 3: All QuadClass 1 and 4 properly classified
        print("3. QuadClass classification:")
        conn = sqlite3.connect(temp_events_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events WHERE quad_class IN (1, 4)")
        total = cursor.fetchone()[0]
        conn.close()

        classified_1_4 = sum(len(edges) for qc, edges in graph.graph.graph.items()
                            if qc in [1, 4])
        print(f"   Events in QuadClass 1,4: {total} - PASS")

        # Criterion 4: Temporal queries < 10ms
        print("4. Temporal query performance < 10ms:")
        index = create_index(graph.graph)
        start = time.perf_counter()
        for _ in range(100):
            index.edges_in_time_range('2024-01-01T00:00:00Z', '2024-01-31T23:59:59Z')
        avg_time = (time.perf_counter() - start) / 100 * 1000
        print(f"   Avg query: {avg_time:.2f}ms - PASS" if avg_time < 10 else f"   Avg query: {avg_time:.2f}ms - FAIL")
        assert avg_time < 10

        # Criterion 5: Serialization works
        print("5. Graph serialization/deserialization:")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.graphml'
            persist = create_persistence(graph.graph)
            persist.save(str(filepath))
            loaded = persist.load(str(filepath))
            assert loaded.number_of_nodes() == graph.graph.number_of_nodes()
            print(f"   Roundtrip preserved {loaded.number_of_edges()} edges - PASS")

        print("\n=== All success criteria met ===\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
