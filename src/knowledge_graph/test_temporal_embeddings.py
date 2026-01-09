"""
Unit tests for HyTE temporal extensions.

Tests cover:
    - Hyperplane initialization and constraints
    - Timestamp to bucket conversion
    - Orthogonal projection correctness
    - Temporal-aware scoring
    - Integration with RotatE base model
    - Projection quality analysis
"""

import pytest
import torch
import numpy as np
from datetime import datetime

from knowledge_graph.temporal_embeddings import (
    HyTETemporalExtension,
    TemporalRotatEModel,
    analyze_temporal_projection_quality
)
from knowledge_graph.embeddings import RotatEModel


class TestHyTETemporalExtension:
    """Test suite for HyTE temporal extension."""

    @pytest.fixture
    def temporal_extension(self):
        """Create temporal extension for testing."""
        return HyTETemporalExtension(
            embedding_dim=64,
            num_time_buckets=52,
            use_complex=True
        )

    def test_initialization(self, temporal_extension):
        """Test temporal extension initializes correctly."""
        # Check hyperplane normals shape
        assert temporal_extension.time_normals.shape == (52, 128)  # 2*64 for complex

        # Check normals are unit vectors
        norms = torch.norm(temporal_extension.time_normals, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_timestamp_to_bucket(self, temporal_extension):
        """Test timestamp to bucket conversion."""
        # Test specific dates
        # January 1, 2024 should be week 0
        jan1_timestamp = datetime(2024, 1, 1).timestamp()
        bucket = temporal_extension.timestamp_to_bucket(jan1_timestamp)
        assert 0 <= bucket < 52

        # December 31, 2024 should be last week
        dec31_timestamp = datetime(2024, 12, 31).timestamp()
        bucket = temporal_extension.timestamp_to_bucket(dec31_timestamp)
        assert 0 <= bucket < 52

    def test_timestamps_to_buckets_batch(self, temporal_extension):
        """Test batch timestamp conversion."""
        # Create batch of timestamps
        timestamps = torch.tensor([
            datetime(2024, 1, 15).timestamp(),
            datetime(2024, 6, 15).timestamp(),
            datetime(2024, 12, 15).timestamp()
        ])

        buckets = temporal_extension.timestamps_to_buckets(timestamps)

        # Check shape and range
        assert buckets.shape == (3,)
        assert torch.all(buckets >= 0)
        assert torch.all(buckets < 52)

        # Different times should map to different buckets
        assert len(set(buckets.tolist())) > 1

    def test_orthogonal_projection(self, temporal_extension):
        """Test orthogonal projection correctness."""
        # Create test embedding
        batch_size = 8
        embeddings = torch.randn(batch_size, 128)  # Complex embeddings
        time_buckets = torch.randint(0, 52, (batch_size,))

        # Project
        projected = temporal_extension.project_onto_hyperplane(
            embeddings, time_buckets
        )

        # Check shape preserved
        assert projected.shape == embeddings.shape

        # Check projection is orthogonal to normal
        normals = temporal_extension.time_normals[time_buckets]
        dot_products = torch.sum(projected * normals, dim=-1)

        # Dot product should be close to zero (projected vector is orthogonal)
        assert torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-4)

    def test_projection_preserves_dimensionality(self, temporal_extension):
        """Test projection doesn't collapse embeddings."""
        embeddings = torch.randn(16, 128)
        time_buckets = torch.randint(0, 52, (16,))

        projected = temporal_extension.project_onto_hyperplane(
            embeddings, time_buckets
        )

        # Projected embeddings should have reasonable magnitude
        original_norms = torch.norm(embeddings, p=2, dim=-1)
        projected_norms = torch.norm(projected, p=2, dim=-1)

        # Projection should reduce magnitude but not by too much
        # Typically should retain > 70% of magnitude
        ratio = projected_norms / original_norms
        assert torch.all(ratio > 0.5)  # Not collapsed
        assert torch.all(ratio <= 1.0)  # Can't increase magnitude

    def test_forward_pass(self, temporal_extension):
        """Test forward pass with temporal projections."""
        batch_size = 16

        # Create test embeddings
        head_emb = torch.randn(batch_size, 128)
        relation_emb = torch.randn(batch_size, 64)  # Phases for RotatE
        tail_emb = torch.randn(batch_size, 128)
        timestamps = torch.rand(batch_size) * 1e9 + 1e9  # Random timestamps

        # Apply temporal projections
        head_proj, relation_proj, tail_proj = temporal_extension(
            head_emb, relation_emb, tail_emb, timestamps
        )

        # Check shapes
        assert head_proj.shape == head_emb.shape
        assert relation_proj.shape == relation_emb.shape  # Relations not projected
        assert tail_proj.shape == tail_emb.shape

        # Check relations are unchanged (phases aren't projected)
        assert torch.allclose(relation_proj, relation_emb)

    def test_enforce_constraints(self, temporal_extension):
        """Test constraint enforcement."""
        # Corrupt normals
        with torch.no_grad():
            temporal_extension.time_normals.data *= 2.0

        # Enforce constraints
        temporal_extension.enforce_constraints()

        # Check normals are unit vectors again
        norms = torch.norm(temporal_extension.time_normals, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_temporal_differentiation(self, temporal_extension):
        """Test that different times produce different projections."""
        # Same embedding, different times
        embedding = torch.randn(1, 128)

        # Project to different time buckets
        bucket1 = torch.tensor([0])
        bucket2 = torch.tensor([25])

        proj1 = temporal_extension.project_onto_hyperplane(embedding, bucket1)
        proj2 = temporal_extension.project_onto_hyperplane(embedding, bucket2)

        # Projections should be different
        diff = torch.norm(proj1 - proj2, p=2)
        assert diff > 0.1  # Significant difference


class TestTemporalRotatEModel:
    """Test temporal RotatE model."""

    @pytest.fixture
    def temporal_model(self):
        """Create temporal RotatE model for testing."""
        return TemporalRotatEModel(
            num_entities=50,
            num_relations=5,
            embedding_dim=64,
            num_time_buckets=52,
            margin=9.0,
            negative_samples=4
        )

    def test_model_initialization(self, temporal_model):
        """Test model initializes correctly."""
        assert temporal_model.base_model is not None
        assert temporal_model.temporal_extension is not None
        assert temporal_model.num_time_buckets == 52

    def test_forward_with_timestamps(self, temporal_model):
        """Test forward pass with temporal awareness."""
        batch_size = 32

        head_idx = torch.randint(0, 50, (batch_size,))
        relation_idx = torch.randint(0, 5, (batch_size,))
        tail_idx = torch.randint(0, 50, (batch_size,))
        timestamps = torch.rand(batch_size) * 1e9 + 1e9

        scores = temporal_model.forward(
            head_idx, relation_idx, tail_idx, timestamps
        )

        # Check output shape
        assert scores.shape == (batch_size,)
        assert not torch.any(torch.isnan(scores))

    def test_temporal_margin_ranking_loss(self, temporal_model):
        """Test temporal-aware loss computation."""
        batch_size = 16

        head_idx = torch.randint(0, 50, (batch_size,))
        relation_idx = torch.randint(0, 5, (batch_size,))
        tail_idx = torch.randint(0, 50, (batch_size,))
        timestamps = torch.rand(batch_size) * 1e9 + 1e9

        loss, stats = temporal_model.margin_ranking_loss(
            head_idx, relation_idx, tail_idx, timestamps
        )

        # Check loss is valid
        assert loss >= 0
        assert not torch.isnan(loss)

        # Check statistics
        assert 'pos_score' in stats
        assert 'neg_score' in stats
        assert 'violation_rate' in stats

    def test_temporal_loss_gradient_flow(self, temporal_model):
        """Test gradients flow through temporal components."""
        batch_size = 16

        head_idx = torch.randint(0, 50, (batch_size,))
        relation_idx = torch.randint(0, 5, (batch_size,))
        tail_idx = torch.randint(0, 50, (batch_size,))
        timestamps = torch.rand(batch_size) * 1e9 + 1e9

        # Zero gradients
        temporal_model.zero_grad()

        # Compute loss
        loss, _ = temporal_model.margin_ranking_loss(
            head_idx, relation_idx, tail_idx, timestamps
        )

        # Backward
        loss.backward()

        # Check gradients exist for temporal parameters
        assert temporal_model.temporal_extension.time_normals.grad is not None
        temporal_grad_norm = torch.norm(
            temporal_model.temporal_extension.time_normals.grad
        )
        assert temporal_grad_norm > 0

    def test_training_step_with_temporal(self, temporal_model):
        """Test training step with temporal model."""
        batch_size = 16

        head_idx = torch.randint(0, 50, (batch_size,))
        relation_idx = torch.randint(0, 5, (batch_size,))
        tail_idx = torch.randint(0, 50, (batch_size,))
        timestamps = torch.rand(batch_size) * 1e9 + 1e9

        # Initial loss
        loss_initial, _ = temporal_model.margin_ranking_loss(
            head_idx, relation_idx, tail_idx, timestamps
        )

        # Training step
        optimizer = torch.optim.Adam(temporal_model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss_initial.backward()
        optimizer.step()
        temporal_model.enforce_constraints()

        # Loss after step
        loss_after, _ = temporal_model.margin_ranking_loss(
            head_idx, relation_idx, tail_idx, timestamps
        )

        # Check loss is still valid
        assert loss_after >= 0
        assert not torch.isnan(loss_after)

    def test_predict_tail_with_time(self, temporal_model):
        """Test tail prediction with temporal awareness."""
        head_idx = 10
        relation_idx = 2
        timestamp = datetime(2024, 6, 15).timestamp()

        predictions = temporal_model.predict_tail(
            head_idx, relation_idx, timestamp, top_k=5
        )

        # Check predictions
        assert len(predictions) == 5
        assert all(isinstance(idx, int) and isinstance(score, float)
                   for idx, score in predictions)

        # Check sorted
        scores = [score for _, score in predictions]
        assert scores == sorted(scores, reverse=True)

    def test_temporal_sensitivity(self, temporal_model):
        """Test model is sensitive to temporal changes."""
        head_idx = torch.tensor([0])
        relation_idx = torch.tensor([0])
        tail_idx = torch.tensor([1])

        # Same triple at different times
        timestamp1 = torch.tensor([datetime(2024, 1, 1).timestamp()])
        timestamp2 = torch.tensor([datetime(2024, 7, 1).timestamp()])

        score1 = temporal_model.forward(
            head_idx, relation_idx, tail_idx, timestamp1
        )
        score2 = temporal_model.forward(
            head_idx, relation_idx, tail_idx, timestamp2
        )

        # Scores should be different (model is time-aware)
        # Note: might be same by chance, but unlikely
        assert not torch.allclose(score1, score2, atol=1e-3)


class TestProjectionQuality:
    """Test projection quality analysis."""

    def test_analyze_projection_quality(self):
        """Test projection quality analysis function."""
        # Create small model
        base_model = RotatEModel(
            num_entities=20,
            num_relations=3,
            embedding_dim=32
        )

        temporal_extension = HyTETemporalExtension(
            embedding_dim=32,
            num_time_buckets=52,
            use_complex=True
        )

        # Analyze quality for subset of entities
        entity_indices = [0, 1, 2, 3, 4]

        metrics = analyze_temporal_projection_quality(
            base_model,
            temporal_extension,
            entity_indices,
            num_samples=20
        )

        # Check metrics exist
        assert 'avg_similarity_preserved' in metrics
        assert 'avg_projection_magnitude' in metrics
        assert 'avg_cross_time_difference' in metrics

        # Check reasonable values
        # Similarity should be high (projections preserve semantics)
        assert 0.5 <= metrics['avg_similarity_preserved'] <= 1.0

        # Magnitude should be substantial (not collapsing)
        assert 0.5 <= metrics['avg_projection_magnitude'] <= 1.0

        # Cross-time differences should be non-zero (temporal variation)
        assert metrics['avg_cross_time_difference'] > 0

    def test_projection_quality_degrades_minimally(self):
        """Test that projection doesn't degrade quality too much."""
        # Target: < 5% quality degradation as per plan
        base_model = RotatEModel(
            num_entities=50,
            num_relations=5,
            embedding_dim=64
        )

        temporal_extension = HyTETemporalExtension(
            embedding_dim=64,
            num_time_buckets=52,
            use_complex=True
        )

        entity_indices = list(range(10))

        metrics = analyze_temporal_projection_quality(
            base_model,
            temporal_extension,
            entity_indices,
            num_samples=50
        )

        # Similarity should be > 0.95 (< 5% degradation)
        assert metrics['avg_similarity_preserved'] > 0.95

        # Magnitude ratio should be > 0.95
        assert metrics['avg_projection_magnitude'] > 0.95


class TestMemoryOverhead:
    """Test memory overhead of temporal extension."""

    def test_memory_overhead(self):
        """Test temporal parameters use < 10MB as per plan."""
        # Create temporal extension with realistic parameters
        temporal_extension = HyTETemporalExtension(
            embedding_dim=256,  # Production setting
            num_time_buckets=52,
            use_complex=True
        )

        # Calculate memory usage
        # time_normals: (52, 512) float32 = 52 * 512 * 4 bytes
        param_size = temporal_extension.time_normals.numel() * 4  # 4 bytes per float32
        memory_mb = param_size / (1024 * 1024)

        print(f"\nTemporal parameters memory: {memory_mb:.2f} MB")

        # Should be well under 10MB
        assert memory_mb < 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
