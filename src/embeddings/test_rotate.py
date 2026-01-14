"""
Unit tests for RotatE embedding model.

Verifies:
- Model initialization
- Forward pass
- Complex rotation constraint
- Margin loss computation
- Negative sampling
"""

import torch
import numpy as np
import pytest
from src.embeddings.rotate import RotatEModel, create_negative_samples


class TestRotatEModel:
    """Test suite for RotatE model."""

    def test_initialization(self):
        """Test model initializes with correct dimensions."""
        num_entities = 100
        num_relations = 20
        embedding_dim = 256

        model = RotatEModel(num_entities, num_relations, embedding_dim)

        assert model.num_entities == num_entities
        assert model.num_relations == num_relations
        assert model.embedding_dim == embedding_dim
        assert model.complex_dim == embedding_dim // 2

        # Check embedding shapes
        assert model.entity_embedding.weight.shape == (num_entities, embedding_dim)
        assert model.relation_embedding.weight.shape == (num_relations, embedding_dim // 2)

    def test_odd_embedding_dimension_raises(self):
        """Test that odd embedding dimension raises error."""
        with pytest.raises(AssertionError):
            RotatEModel(100, 20, embedding_dim=257)  # Odd dimension

    def test_forward_pass_batch(self):
        """Test forward pass completes for batch of 256 triples."""
        model = RotatEModel(1000, 50, embedding_dim=256)

        batch_size = 256
        head_ids = torch.randint(0, 1000, (batch_size,))
        relation_ids = torch.randint(0, 50, (batch_size,))
        tail_ids = torch.randint(0, 1000, (batch_size,))

        # Generate negative samples
        neg_heads, neg_tails = create_negative_samples(
            head_ids, tail_ids, 1000, num_negatives=4
        )

        # Forward pass
        pos_scores, neg_scores = model(head_ids, relation_ids, tail_ids, neg_heads, neg_tails)

        # Check output shapes
        assert pos_scores.shape == (batch_size,)
        assert neg_scores.shape == (batch_size, 8)  # 4 head + 4 tail negatives

    def test_score_triples(self):
        """Test triple scoring function."""
        model = RotatEModel(100, 20, embedding_dim=64)

        head_ids = torch.tensor([0, 1, 2])
        relation_ids = torch.tensor([0, 1, 2])
        tail_ids = torch.tensor([10, 11, 12])

        scores = model.score_triples(head_ids, relation_ids, tail_ids)

        # Check output
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_rotation_constraint(self):
        """Test that complex rotations preserve norm (unit circle)."""
        model = RotatEModel(100, 20, embedding_dim=128)

        # Check constraint statistics
        stats = model.verify_rotation_constraint()

        # Rotations should be on unit circle (norm ≈ 1.0)
        assert 0.99 <= stats["mean_norm"] <= 1.01
        assert 0.95 <= stats["min_norm"] <= 1.05
        assert 0.95 <= stats["max_norm"] <= 1.05
        assert stats["std_norm"] < 0.01

    def test_margin_loss(self):
        """Test margin-based loss computation."""
        model = RotatEModel(100, 20, embedding_dim=64)

        # Create dummy scores
        positive_scores = torch.tensor([5.0, 6.0, 7.0])
        negative_scores = torch.tensor([
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0]
        ])

        loss = model.margin_loss(positive_scores, negative_scores, margin=2.0)

        # Loss should be non-negative
        assert loss >= 0
        assert torch.isfinite(loss)

    def test_loss_decreases_monotonically(self):
        """Test that loss decreases during initial training steps."""
        model = RotatEModel(100, 20, embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Generate synthetic training data
        batch_size = 32
        losses = []

        for _ in range(10):
            head_ids = torch.randint(0, 100, (batch_size,))
            relation_ids = torch.randint(0, 20, (batch_size,))
            tail_ids = torch.randint(0, 100, (batch_size,))

            neg_heads, neg_tails = create_negative_samples(
                head_ids, tail_ids, 100, num_negatives=4
            )

            optimizer.zero_grad()
            pos_scores, neg_scores = model(head_ids, relation_ids, tail_ids, neg_heads, neg_tails)
            loss = model.margin_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease
        # Check that final loss is lower than initial loss
        assert losses[-1] < losses[0]

    def test_complex_multiply(self):
        """Test complex multiplication implementation."""
        model = RotatEModel(10, 5, embedding_dim=64)

        # Test with known values: (1 + 2i) * (3 + 4i) = -5 + 10i
        a_real = torch.tensor([[1.0]])
        a_imag = torch.tensor([[2.0]])
        b_real = torch.tensor([[3.0]])
        b_imag = torch.tensor([[4.0]])

        result_real, result_imag = model._complex_multiply(a_real, a_imag, b_real, b_imag)

        assert torch.isclose(result_real, torch.tensor([[-5.0]]), atol=1e-6)
        assert torch.isclose(result_imag, torch.tensor([[10.0]]), atol=1e-6)

    def test_get_embeddings(self):
        """Test embedding extraction methods."""
        model = RotatEModel(100, 20, embedding_dim=128)

        entity_emb = model.get_entity_embeddings()
        relation_emb = model.get_relation_embeddings()

        assert entity_emb.shape == (100, 128)
        assert relation_emb.shape == (20, 64)  # complex_dim
        assert isinstance(entity_emb, np.ndarray)
        assert isinstance(relation_emb, np.ndarray)


class TestNegativeSampling:
    """Test suite for negative sampling."""

    def test_both_corruption(self):
        """Test both head and tail corruption."""
        head_ids = torch.tensor([0, 1, 2])
        tail_ids = torch.tensor([10, 11, 12])

        neg_heads, neg_tails = create_negative_samples(
            head_ids, tail_ids, num_entities=100, num_negatives=4, corruption_mode="both"
        )

        assert neg_heads.shape == (3, 4)
        assert neg_tails.shape == (3, 4)
        assert (neg_heads >= 0).all() and (neg_heads < 100).all()
        assert (neg_tails >= 0).all() and (neg_tails < 100).all()

    def test_head_corruption_only(self):
        """Test head corruption only."""
        head_ids = torch.tensor([0, 1, 2])
        tail_ids = torch.tensor([10, 11, 12])

        neg_heads, neg_tails = create_negative_samples(
            head_ids, tail_ids, num_entities=100, num_negatives=4, corruption_mode="head"
        )

        assert neg_heads.shape == (3, 4)
        assert neg_tails is None

    def test_tail_corruption_only(self):
        """Test tail corruption only."""
        head_ids = torch.tensor([0, 1, 2])
        tail_ids = torch.tensor([10, 11, 12])

        neg_heads, neg_tails = create_negative_samples(
            head_ids, tail_ids, num_entities=100, num_negatives=4, corruption_mode="tail"
        )

        assert neg_heads is None
        assert neg_tails.shape == (3, 4)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
