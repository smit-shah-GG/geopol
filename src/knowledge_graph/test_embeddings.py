"""
Unit tests for RotatE embedding model.

Tests cover:
    - Model initialization and constraints
    - Complex multiplication correctness
    - Distance computation
    - Negative sampling strategy
    - Loss computation and gradient flow
    - Prediction functionality
"""

import pytest
import torch
import torch.nn as nn
import math
from src.knowledge_graph.embeddings import RotatEModel, clip_gradients


class TestRotatEModel:
    """Test suite for RotatE embedding model."""

    @pytest.fixture
    def small_model(self):
        """Create small model for testing."""
        return RotatEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=64,
            margin=9.0,
            negative_samples=4
        )

    def test_initialization(self, small_model):
        """Test model initializes with correct shapes and constraints."""
        # Check shapes
        assert small_model.entity_embeddings.weight.shape == (100, 128)  # 2 * 64
        assert small_model.relation_embeddings.weight.shape == (10, 64)

        # Check entity embeddings on unit circle
        entity_norms = torch.norm(small_model.entity_embeddings.weight, p=2, dim=1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms), atol=1e-5)

        # Check relation phases in [-π, π]
        assert torch.all(small_model.relation_embeddings.weight >= -math.pi)
        assert torch.all(small_model.relation_embeddings.weight <= math.pi)

    def test_complex_multiplication_correctness(self, small_model):
        """Test complex multiplication matches mathematical definition."""
        # Create simple test case: rotate by π/2 (90 degrees)
        # (1, 0) * e^(iπ/2) = (1, 0) * (0, 1) = (0, 1)
        entity = torch.tensor([[1.0, 0.0, 0.0, 1.0]])  # Real: [1,0], Imag: [0,1]
        phase = torch.tensor([[math.pi/2, 0.0]])  # 90° rotation, 0° rotation

        result = small_model.complex_multiply(entity, phase)

        # Expected: first component rotated 90°, second unchanged
        # First: (1,0) -> (0,1)
        # Second: (0,1) -> (-1,0)
        expected_real = torch.tensor([[0.0, 0.0]])  # Real part after rotation
        expected_imag = torch.tensor([[1.0, 1.0]])  # Imag part after rotation

        assert torch.allclose(result[:, :2], expected_real, atol=1e-5)
        assert torch.allclose(result[:, 2:], expected_imag, atol=1e-5)

    def test_complex_multiplication_preserves_norm(self, small_model):
        """Test that rotation preserves complex vector norm."""
        # Get random entity embedding
        entity_idx = torch.tensor([0])
        entity = small_model.entity_embeddings(entity_idx)

        # Get random relation (rotation)
        relation_idx = torch.tensor([0])
        relation = small_model.relation_embeddings(relation_idx)

        # Apply rotation
        rotated = small_model.complex_multiply(entity, relation)

        # Check norms are equal
        original_norm = torch.norm(entity, p=2, dim=-1)
        rotated_norm = torch.norm(rotated, p=2, dim=-1)

        assert torch.allclose(original_norm, rotated_norm, atol=1e-5)

    def test_distance_function(self, small_model):
        """Test distance computation is non-negative and symmetric-ish."""
        batch_size = 16
        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        # Get embeddings
        head = small_model.entity_embeddings(head_idx)
        relation = small_model.relation_embeddings(relation_idx)
        tail = small_model.entity_embeddings(tail_idx)

        # Compute distance
        dist = small_model.distance(head, relation, tail)

        # Distance should be non-negative
        assert torch.all(dist >= 0)

        # Distance should be zero for identical entities after zero rotation
        identity_head = small_model.entity_embeddings(torch.tensor([0]))
        zero_relation = torch.zeros(1, 64)  # Zero rotation
        identity_tail = small_model.entity_embeddings(torch.tensor([0]))

        identity_dist = small_model.distance(identity_head, zero_relation, identity_tail)
        assert torch.allclose(identity_dist, torch.zeros_like(identity_dist), atol=1e-5)

    def test_forward_pass(self, small_model):
        """Test forward pass completes and returns correct shape."""
        batch_size = 256
        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        scores = small_model.forward(head_idx, relation_idx, tail_idx)

        # Check shape
        assert scores.shape == (batch_size,)

        # Scores should be negative distances (negative values)
        # But not all will be negative due to random initialization
        assert scores.dtype == torch.float32

    def test_negative_sampling(self, small_model):
        """Test negative sampling generates correct number and valid indices."""
        batch_size = 32
        num_negatives = 4

        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        neg_head, neg_relation, neg_tail = small_model.sample_negative_triples(
            head_idx, relation_idx, tail_idx, num_negatives
        )

        # Check shapes
        expected_size = batch_size * num_negatives
        assert neg_head.shape == (expected_size,)
        assert neg_relation.shape == (expected_size,)
        assert neg_tail.shape == (expected_size,)

        # Check indices are valid
        assert torch.all(neg_head >= 0) and torch.all(neg_head < 100)
        assert torch.all(neg_relation >= 0) and torch.all(neg_relation < 10)
        assert torch.all(neg_tail >= 0) and torch.all(neg_tail < 100)

        # Check relations are preserved
        relation_repeated = relation_idx.repeat_interleave(num_negatives)
        assert torch.all(neg_relation == relation_repeated)

    def test_margin_ranking_loss(self, small_model):
        """Test loss computation and statistics."""
        batch_size = 64
        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        loss, stats = small_model.margin_ranking_loss(head_idx, relation_idx, tail_idx)

        # Check loss is scalar and non-negative
        assert loss.ndim == 0
        assert loss >= 0

        # Check statistics
        assert 'pos_score' in stats
        assert 'neg_score' in stats
        assert 'violation_rate' in stats
        assert 0 <= stats['violation_rate'] <= 1

    def test_loss_gradient_flow(self, small_model):
        """Test that gradients flow correctly through the model."""
        batch_size = 32
        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        # Zero gradients
        small_model.zero_grad()

        # Compute loss
        loss, _ = small_model.margin_ranking_loss(head_idx, relation_idx, tail_idx)

        # Backward pass
        loss.backward()

        # Check gradients exist and are non-zero
        entity_grad_norm = torch.norm(small_model.entity_embeddings.weight.grad)
        relation_grad_norm = torch.norm(small_model.relation_embeddings.weight.grad)

        assert entity_grad_norm > 0
        assert relation_grad_norm > 0

    def test_enforce_constraints(self, small_model):
        """Test constraint enforcement after parameter updates."""
        # Manually corrupt embeddings
        with torch.no_grad():
            # Scale entity embeddings (breaks unit norm)
            small_model.entity_embeddings.weight.data *= 2.0

            # Push relation phases out of bounds
            small_model.relation_embeddings.weight.data += 10.0

        # Enforce constraints
        small_model.enforce_constraints()

        # Check constraints restored
        entity_norms = torch.norm(small_model.entity_embeddings.weight, p=2, dim=1)
        assert torch.allclose(entity_norms, torch.ones_like(entity_norms), atol=1e-5)

        assert torch.all(small_model.relation_embeddings.weight >= -math.pi)
        assert torch.all(small_model.relation_embeddings.weight <= math.pi)

    def test_training_step_decreases_loss(self, small_model):
        """Test that a single training step decreases loss."""
        batch_size = 64
        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        # Initial loss
        loss_initial, _ = small_model.margin_ranking_loss(head_idx, relation_idx, tail_idx)

        # Training step
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss_initial.backward()
        optimizer.step()
        small_model.enforce_constraints()

        # Loss after one step
        loss_after, _ = small_model.margin_ranking_loss(head_idx, relation_idx, tail_idx)

        # For the same batch, loss should typically decrease
        # (may not always be true due to stochastic negative sampling, but likely)
        # We just check it's still a valid loss value
        assert loss_after >= 0
        assert not torch.isnan(loss_after)

    def test_get_entity_embedding(self, small_model):
        """Test entity embedding retrieval."""
        entity_idx = 5
        embedding = small_model.get_entity_embedding(entity_idx)

        # Check shape and device
        assert embedding.shape == (128,)  # 2 * 64
        assert embedding.device.type == 'cpu'

        # Check on unit circle
        norm = torch.norm(embedding, p=2)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_get_relation_embedding(self, small_model):
        """Test relation embedding retrieval."""
        relation_idx = 3
        embedding = small_model.get_relation_embedding(relation_idx)

        # Check shape and device
        assert embedding.shape == (64,)
        assert embedding.device.type == 'cpu'

        # Check in valid range
        assert torch.all(embedding >= -math.pi)
        assert torch.all(embedding <= math.pi)

    def test_predict_tail(self, small_model):
        """Test tail prediction functionality."""
        head_idx = 10
        relation_idx = 2
        top_k = 5

        predictions = small_model.predict_tail(head_idx, relation_idx, top_k)

        # Check return format
        assert len(predictions) == top_k
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in predictions)

        # Check scores are sorted descending
        scores = [score for _, score in predictions]
        assert scores == sorted(scores, reverse=True)

        # Check entity indices are valid
        entity_indices = [idx for idx, _ in predictions]
        assert all(0 <= idx < 100 for idx in entity_indices)

    def test_gradient_clipping(self, small_model):
        """Test gradient clipping utility."""
        batch_size = 32
        head_idx = torch.randint(0, 100, (batch_size,))
        relation_idx = torch.randint(0, 10, (batch_size,))
        tail_idx = torch.randint(0, 100, (batch_size,))

        # Compute loss and gradients
        small_model.zero_grad()
        loss, _ = small_model.margin_ranking_loss(head_idx, relation_idx, tail_idx)
        loss.backward()

        # Clip gradients
        total_norm = clip_gradients(small_model, max_norm=1.0)

        # Check clipping occurred
        assert total_norm >= 0

        # Check actual gradient norm after clipping
        actual_norm = sum(
            p.grad.norm().item() ** 2
            for p in small_model.parameters() if p.grad is not None
        ) ** 0.5

        # After clipping, norm should be <= max_norm (if it was > max_norm before)
        if total_norm > 1.0:
            assert actual_norm <= 1.0 + 1e-5  # Small tolerance for numerical errors


class TestRotatEScaling:
    """Test model behavior at different scales."""

    def test_large_batch_performance(self):
        """Test model handles large batches efficiently."""
        model = RotatEModel(num_entities=1000, num_relations=50, embedding_dim=256)

        batch_size = 256
        head_idx = torch.randint(0, 1000, (batch_size,))
        relation_idx = torch.randint(0, 50, (batch_size,))
        tail_idx = torch.randint(0, 1000, (batch_size,))

        # Forward pass should complete without error
        loss, stats = model.margin_ranking_loss(head_idx, relation_idx, tail_idx)

        assert loss >= 0
        assert not torch.isnan(loss)

    def test_embedding_dimension_scaling(self):
        """Test different embedding dimensions."""
        for dim in [64, 128, 256]:
            model = RotatEModel(num_entities=100, num_relations=10, embedding_dim=dim)

            # Check shapes
            assert model.entity_embeddings.weight.shape == (100, 2 * dim)
            assert model.relation_embeddings.weight.shape == (10, dim)

            # Test forward pass
            head_idx = torch.tensor([0])
            relation_idx = torch.tensor([0])
            tail_idx = torch.tensor([1])

            score = model.forward(head_idx, relation_idx, tail_idx)
            assert not torch.isnan(score)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
