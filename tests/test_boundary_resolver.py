"""
Tests for BoundaryResolver cross-partition entity identification.

Verifies:
- Boundary entity identification across partitions
- Home partition is highest edge count
- Replication threshold filtering
- Boundary statistics accuracy
- get_home_partition returns correct values
"""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from src.knowledge_graph.boundary_resolver import BoundaryEntity, BoundaryResolver
from src.knowledge_graph.partition_index import EntityPartitionIndex
from src.knowledge_graph.partition_manager import PartitionManager


class TestBoundaryResolver:
    """Test suite for BoundaryResolver."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def index(self, temp_dir):
        """Create a fresh EntityPartitionIndex."""
        db_path = temp_dir / "index.db"
        idx = EntityPartitionIndex(db_path)
        yield idx
        idx.close()

    @pytest.fixture
    def manager(self, temp_dir, index):
        """Create a PartitionManager with test settings."""
        partition_dir = temp_dir / "partitions"
        return PartitionManager(
            partition_dir=partition_dir,
            index=index,
            max_cached_partitions=4
        )

    def test_identify_no_boundaries(self, index, manager):
        """Graph with isolated partitions has no boundary entities."""
        # Create partitions where each entity appears in only one partition
        # (no shared entities = no boundaries)
        index.register_partition(
            partition_id="p1",
            entities={"a1", "a2"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 2, "edges": 5},
            file_path="/p1.graphml"
        )
        index.register_partition(
            partition_id="p2",
            entities={"b1", "b2"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 2, "edges": 5},
            file_path="/p2.graphml"
        )

        resolver = BoundaryResolver(index, manager, replication_threshold=10)
        boundaries = resolver.identify_boundary_entities()

        assert len(boundaries) == 0
        assert resolver.get_boundary_stats()['boundary_entities'] == 0

    def test_identify_boundary_entity(self, index, manager):
        """Entity in 2 partitions is identified as boundary."""
        # Create an entity that appears in two partitions
        index.register_partition(
            partition_id="p1",
            entities={"shared_entity", "unique_1"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 2, "edges": 10},
            file_path="/p1.graphml"
        )
        index.update_entity_edge_count("shared_entity", "p1", 15)

        index.register_partition(
            partition_id="p2",
            entities={"shared_entity", "unique_2"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 2, "edges": 10},
            file_path="/p2.graphml"
        )
        index.update_entity_edge_count("shared_entity", "p2", 10)

        resolver = BoundaryResolver(index, manager, replication_threshold=5)
        boundaries = resolver.identify_boundary_entities()

        # shared_entity should be identified as boundary
        assert "shared_entity" in boundaries
        assert "unique_1" not in boundaries
        assert "unique_2" not in boundaries

        # Verify boundary entity details
        boundary = boundaries["shared_entity"]
        assert boundary.entity_id == "shared_entity"
        assert len(boundary.edge_count_per_partition) == 2

    def test_home_partition_is_highest_edge_count(self, index, manager):
        """Home partition is the one where entity has most edges."""
        # Register entity in 3 partitions with different edge counts
        index.register_partition(
            partition_id="p1",
            entities={"multi_entity"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 5},
            file_path="/p1.graphml"
        )
        index.update_entity_edge_count("multi_entity", "p1", 5)

        index.register_partition(
            partition_id="p2",
            entities={"multi_entity"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 20},
            file_path="/p2.graphml"
        )
        index.update_entity_edge_count("multi_entity", "p2", 20)  # Highest

        index.register_partition(
            partition_id="p3",
            entities={"multi_entity"},
            time_window=("2024-03-01", "2024-03-31"),
            stats={"nodes": 1, "edges": 10},
            file_path="/p3.graphml"
        )
        index.update_entity_edge_count("multi_entity", "p3", 10)

        resolver = BoundaryResolver(index, manager, replication_threshold=3)
        boundaries = resolver.identify_boundary_entities()

        boundary = boundaries["multi_entity"]

        # Home should be p2 (highest edge count)
        assert boundary.home_partition == "p2"

        # Replicas should be p1 and p3 (both above threshold of 3)
        assert "p1" in boundary.replica_partitions
        assert "p3" in boundary.replica_partitions
        assert "p2" not in boundary.replica_partitions  # Home is not a replica

    def test_replication_threshold(self, index, manager):
        """Partitions below threshold excluded from replica set."""
        # Entity in 3 partitions: one high, one medium, one low edge count
        index.register_partition(
            partition_id="high",
            entities={"test_entity"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 50},
            file_path="/high.graphml"
        )
        index.update_entity_edge_count("test_entity", "high", 50)

        index.register_partition(
            partition_id="medium",
            entities={"test_entity"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 15},
            file_path="/medium.graphml"
        )
        index.update_entity_edge_count("test_entity", "medium", 15)

        index.register_partition(
            partition_id="low",
            entities={"test_entity"},
            time_window=("2024-03-01", "2024-03-31"),
            stats={"nodes": 1, "edges": 3},
            file_path="/low.graphml"
        )
        index.update_entity_edge_count("test_entity", "low", 3)

        # Set threshold to 10 - should exclude 'low' partition
        resolver = BoundaryResolver(index, manager, replication_threshold=10)
        boundaries = resolver.identify_boundary_entities()

        boundary = boundaries["test_entity"]

        # Home is 'high' (50 edges)
        assert boundary.home_partition == "high"

        # Only 'medium' (15 edges) is above threshold of 10
        assert "medium" in boundary.replica_partitions
        assert "low" not in boundary.replica_partitions  # Below threshold

    def test_boundary_stats(self, index, manager):
        """Stats accurately reflect boundary count and ratios."""
        # Create 5 total entities: 2 are boundaries
        index.register_partition(
            partition_id="p1",
            entities={"shared_1", "shared_2", "unique_1"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 3, "edges": 10},
            file_path="/p1.graphml"
        )
        index.update_entity_edge_count("shared_1", "p1", 10)
        index.update_entity_edge_count("shared_2", "p1", 5)

        index.register_partition(
            partition_id="p2",
            entities={"shared_1", "shared_2", "unique_2", "unique_3"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 4, "edges": 15},
            file_path="/p2.graphml"
        )
        index.update_entity_edge_count("shared_1", "p2", 20)
        index.update_entity_edge_count("shared_2", "p2", 8)

        resolver = BoundaryResolver(index, manager, replication_threshold=5)
        boundaries = resolver.identify_boundary_entities()

        stats = resolver.get_boundary_stats()

        # Total entities: shared_1, shared_2, unique_1, unique_2, unique_3 = 5
        assert stats['total_entities'] == 5

        # Boundary entities: shared_1, shared_2 = 2
        assert stats['boundary_entities'] == 2

        # Replication ratio: 2/5 = 0.4
        assert stats['replication_ratio'] == 0.4

        # Threshold should be reported
        assert stats['replication_threshold'] == 5

        # Average replica count:
        # shared_1: p1 has 10 edges >= 5, so 1 replica
        # shared_2: p1 has 5 edges >= 5, so 1 replica
        # Total replicas = 2, boundaries = 2, avg = 1.0
        assert stats['avg_replica_count'] == 1.0

    def test_get_home_partition(self, index, manager):
        """Returns correct home for boundary entity, None for non-boundary."""
        # Setup: one boundary entity, one non-boundary
        index.register_partition(
            partition_id="p1",
            entities={"boundary_ent", "solo_ent"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 2, "edges": 10},
            file_path="/p1.graphml"
        )
        index.update_entity_edge_count("boundary_ent", "p1", 5)

        index.register_partition(
            partition_id="p2",
            entities={"boundary_ent"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 20},
            file_path="/p2.graphml"
        )
        index.update_entity_edge_count("boundary_ent", "p2", 25)  # Higher

        resolver = BoundaryResolver(index, manager, replication_threshold=3)

        # Boundary entity should return home partition
        home = resolver.get_home_partition("boundary_ent")
        assert home == "p2"  # Partition with highest edge count

        # Non-boundary entity should return None
        home = resolver.get_home_partition("solo_ent")
        assert home is None

        # Non-existent entity should return None
        home = resolver.get_home_partition("nonexistent")
        assert home is None

    def test_is_boundary_entity(self, index, manager):
        """is_boundary_entity correctly identifies boundaries."""
        index.register_partition(
            partition_id="p1",
            entities={"boundary", "not_boundary"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 2, "edges": 5},
            file_path="/p1.graphml"
        )
        index.register_partition(
            partition_id="p2",
            entities={"boundary"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 5},
            file_path="/p2.graphml"
        )

        resolver = BoundaryResolver(index, manager)

        assert resolver.is_boundary_entity("boundary") is True
        assert resolver.is_boundary_entity("not_boundary") is False
        assert resolver.is_boundary_entity("nonexistent") is False

    def test_get_boundary_entity(self, index, manager):
        """get_boundary_entity returns full BoundaryEntity info."""
        index.register_partition(
            partition_id="p1",
            entities={"test_ent"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 10},
            file_path="/p1.graphml"
        )
        index.update_entity_edge_count("test_ent", "p1", 10)

        index.register_partition(
            partition_id="p2",
            entities={"test_ent"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 15},
            file_path="/p2.graphml"
        )
        index.update_entity_edge_count("test_ent", "p2", 15)

        resolver = BoundaryResolver(index, manager, replication_threshold=5)

        boundary = resolver.get_boundary_entity("test_ent")

        assert boundary is not None
        assert isinstance(boundary, BoundaryEntity)
        assert boundary.entity_id == "test_ent"
        assert boundary.home_partition == "p2"  # Higher edge count
        assert "p1" in boundary.replica_partitions
        assert boundary.edge_count_per_partition == {"p1": 10, "p2": 15}

        # Non-boundary returns None
        assert resolver.get_boundary_entity("nonexistent") is None

    def test_lazy_identification(self, index, manager):
        """Boundary identification is lazy - happens on first access."""
        index.register_partition(
            partition_id="p1",
            entities={"e1"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 5},
            file_path="/p1.graphml"
        )
        index.register_partition(
            partition_id="p2",
            entities={"e1"},
            time_window=("2024-02-01", "2024-02-28"),
            stats={"nodes": 1, "edges": 5},
            file_path="/p2.graphml"
        )

        resolver = BoundaryResolver(index, manager)

        # Not yet identified
        assert resolver._is_identified is False

        # First access triggers identification
        _ = resolver.get_home_partition("e1")
        assert resolver._is_identified is True

    def test_invalidate_cache(self, index, manager):
        """invalidate_cache clears cached boundaries."""
        index.register_partition(
            partition_id="p1",
            entities={"e1"},
            time_window=("2024-01-01", "2024-01-31"),
            stats={"nodes": 1, "edges": 5},
            file_path="/p1.graphml"
        )

        resolver = BoundaryResolver(index, manager)
        _ = resolver.identify_boundary_entities()

        assert resolver._is_identified is True
        assert len(resolver._boundary_entities) == 0  # Only one partition

        # Invalidate
        resolver.invalidate_cache()

        assert resolver._is_identified is False
        assert len(resolver._boundary_entities) == 0

    def test_empty_partitions(self, index, manager):
        """No partitions results in empty boundary set."""
        resolver = BoundaryResolver(index, manager)
        boundaries = resolver.identify_boundary_entities()

        assert boundaries == {}
        assert resolver.get_boundary_stats()['total_entities'] == 0

    def test_high_replication_threshold(self, index, manager):
        """High threshold results in fewer replicas."""
        # Entity in 3 partitions with moderate edge counts
        for i, (pid, edge_count) in enumerate([("p1", 30), ("p2", 20), ("p3", 10)]):
            index.register_partition(
                partition_id=pid,
                entities={"widely_shared"},
                time_window=(f"2024-0{i+1}-01", f"2024-0{i+1}-28"),
                stats={"nodes": 1, "edges": edge_count},
                file_path=f"/{pid}.graphml"
            )
            index.update_entity_edge_count("widely_shared", pid, edge_count)

        # With threshold=25, only p1 qualifies as home, no replicas
        resolver = BoundaryResolver(index, manager, replication_threshold=25)
        boundaries = resolver.identify_boundary_entities()

        boundary = boundaries["widely_shared"]
        assert boundary.home_partition == "p1"  # 30 edges
        assert len(boundary.replica_partitions) == 0  # None meet threshold
