"""
RotatE Embedding Model for Temporal Knowledge Graphs

Implements the RotatE model (Sun et al., 2019) for knowledge graph embeddings
using complex-valued rotations. This is a CPU-optimized implementation targeting
256-dimensional embeddings for entities and relations.

Reference:
    Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019).
    RotatE: Knowledge graph embedding by relational rotation in complex space.
    ICLR 2019.

Key Features:
    - Complex-valued entity embeddings in polar form
    - Relation embeddings as rotations (angles only)
    - Margin-based ranking loss with negative sampling
    - Unit circle constraint preservation
    - Gradient clipping for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class RotatEModel(nn.Module):
    """
    RotatE: Knowledge graph embedding by relational rotation in complex space.

    The model represents entities as complex vectors and relations as rotations
    in the complex plane. For a triple (h, r, t), the scoring function is:

        d(h, r, t) = ||h ∘ r - t||

    where ∘ denotes element-wise complex multiplication (rotation).

    Architecture:
        - Entity embeddings: Complex vectors (2 × embedding_dim real values)
        - Relation embeddings: Phase angles (embedding_dim real values in [0, 2π])
        - Negative sampling: 4 negatives per positive triple
        - Loss: Margin-based ranking loss

    Args:
        num_entities: Total number of unique entities
        num_relations: Total number of unique relation types
        embedding_dim: Dimension of embeddings (256 for CPU optimization)
        margin: Margin value for ranking loss (default: 9.0)
        negative_samples: Number of negative samples per positive (default: 4)
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        negative_samples: int = 4
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.negative_samples = negative_samples

        # Entity embeddings: Complex vectors represented as (real, imaginary) pairs
        # Shape: (num_entities, 2 * embedding_dim)
        # First half = real components, second half = imaginary components
        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)

        # Relation embeddings: Phase angles in [0, 2π]
        # Shape: (num_relations, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """
        Initialize embeddings using Xavier uniform initialization.

        Entity embeddings are initialized to lie near the unit circle.
        Relation embeddings (phases) are uniformly distributed in [-π, π].
        """
        # Initialize entity embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)

        # Normalize entities to unit circle (complex norm = 1)
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data,
                p=2,
                dim=-1
            )

        # Initialize relation phases uniformly in [-π, π]
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            -math.pi,
            math.pi
        )

    def complex_multiply(
        self,
        entity: torch.Tensor,
        phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform complex multiplication: entity ∘ rotation.

        Given entity as (real, imag) and phase θ, computes:
            (real, imag) ∘ (cos(θ), sin(θ))
          = (real*cos(θ) - imag*sin(θ), real*sin(θ) + imag*cos(θ))

        Args:
            entity: Complex entity embedding, shape (*, 2*dim) where
                    [:, :dim] = real, [:, dim:] = imaginary
            phase: Rotation angles, shape (*, dim)

        Returns:
            Rotated complex vector, shape (*, 2*dim)
        """
        dim = phase.size(-1)

        # Split entity into real and imaginary parts
        real = entity[..., :dim]  # Shape: (*, dim)
        imag = entity[..., dim:]  # Shape: (*, dim)

        # Compute rotation components
        cos_phase = torch.cos(phase)  # Shape: (*, dim)
        sin_phase = torch.sin(phase)  # Shape: (*, dim)

        # Complex multiplication: (a + bi) * (cos + i*sin)
        # = (a*cos - b*sin) + i*(a*sin + b*cos)
        real_rotated = real * cos_phase - imag * sin_phase
        imag_rotated = real * sin_phase + imag * cos_phase

        # Concatenate back to complex vector
        return torch.cat([real_rotated, imag_rotated], dim=-1)

    def distance(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RotatE scoring function: ||h ∘ r - t||_L1.

        Lower distance indicates higher plausibility.

        Args:
            head: Head entity embeddings (batch_size, 2*dim)
            relation: Relation phase embeddings (batch_size, dim)
            tail: Tail entity embeddings (batch_size, 2*dim)

        Returns:
            L1 distances (batch_size,)
        """
        # Rotate head by relation
        head_rotated = self.complex_multiply(head, relation)

        # Compute L1 distance: ||h ∘ r - t||_1
        distance = torch.sum(torch.abs(head_rotated - tail), dim=-1)

        return distance

    def forward(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute scores for given triples.

        Args:
            head_idx: Head entity indices (batch_size,)
            relation_idx: Relation indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)

        Returns:
            Negative distances (higher = more plausible)
        """
        # Lookup embeddings
        head_emb = self.entity_embeddings(head_idx)
        relation_emb = self.relation_embeddings(relation_idx)
        tail_emb = self.entity_embeddings(tail_idx)

        # Compute distance (lower = better)
        dist = self.distance(head_emb, relation_emb, tail_emb)

        # Return negative distance (higher = better for consistency with loss)
        return -dist

    def sample_negative_triples(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        num_negatives: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate negative triples by corrupting head or tail entities.

        Strategy: Randomly replace either head or tail with random entity.
        This ensures negative samples are valid entity-relation-entity triples
        but likely do not appear in the graph.

        Args:
            head_idx: Head entity indices (batch_size,)
            relation_idx: Relation indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            num_negatives: Number of negative samples per positive

        Returns:
            Tuple of (neg_heads, neg_relations, neg_tails) each of shape
            (batch_size * num_negatives,)
        """
        batch_size = head_idx.size(0)
        device = head_idx.device

        # Repeat positive triples num_negatives times
        head_repeated = head_idx.repeat_interleave(num_negatives)
        relation_repeated = relation_idx.repeat_interleave(num_negatives)
        tail_repeated = tail_idx.repeat_interleave(num_negatives)

        # Randomly decide whether to corrupt head or tail (50/50 split)
        corrupt_head_mask = torch.rand(batch_size * num_negatives, device=device) < 0.5

        # Generate random entity replacements
        random_entities = torch.randint(
            0,
            self.num_entities,
            (batch_size * num_negatives,),
            device=device
        )

        # Apply corruption
        neg_head = torch.where(corrupt_head_mask, random_entities, head_repeated)
        neg_tail = torch.where(corrupt_head_mask, tail_repeated, random_entities)
        neg_relation = relation_repeated

        return neg_head, neg_relation, neg_tail

    def margin_ranking_loss(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute margin-based ranking loss with negative sampling.

        Loss: L = max(0, γ - score_pos + score_neg)

        where γ is the margin, score_pos is the score of positive triples,
        and score_neg is the score of negative triples.

        Args:
            head_idx: Head entity indices (batch_size,)
            relation_idx: Relation indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)

        Returns:
            loss: Scalar loss value
            stats: Dictionary with statistics (pos_score, neg_score, violations)
        """
        batch_size = head_idx.size(0)

        # Compute positive scores (negative distances)
        pos_scores = self.forward(head_idx, relation_idx, tail_idx)

        # Generate and score negative triples
        neg_head, neg_relation, neg_tail = self.sample_negative_triples(
            head_idx, relation_idx, tail_idx, self.negative_samples
        )
        neg_scores = self.forward(neg_head, neg_relation, neg_tail)

        # Reshape negative scores: (batch_size, num_negatives)
        neg_scores = neg_scores.view(batch_size, self.negative_samples)

        # Compute margin loss for each negative
        # Loss = max(0, margin - pos_score + neg_score)
        pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch_size, 1)
        margin_violations = self.margin - pos_scores_expanded + neg_scores
        loss = torch.clamp(margin_violations, min=0.0).mean()

        # Compute statistics
        with torch.no_grad():
            violation_rate = (margin_violations > 0).float().mean().item()
            pos_score_mean = pos_scores.mean().item()
            neg_score_mean = neg_scores.mean().item()

        stats = {
            'pos_score': pos_score_mean,
            'neg_score': neg_score_mean,
            'violation_rate': violation_rate
        }

        return loss, stats

    def enforce_constraints(self):
        """
        Enforce model constraints after each optimization step.

        Constraints:
            1. Entity embeddings must lie on unit circle (complex norm = 1)
            2. Relation phases must be in [-π, π]
        """
        with torch.no_grad():
            # Normalize entity embeddings to unit circle
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data,
                p=2,
                dim=-1
            )

            # Clip relation phases to [-π, π]
            self.relation_embeddings.weight.data = torch.clamp(
                self.relation_embeddings.weight.data,
                -math.pi,
                math.pi
            )

    def get_entity_embedding(self, entity_idx: int) -> torch.Tensor:
        """
        Retrieve embedding for a single entity.

        Args:
            entity_idx: Entity index

        Returns:
            Complex embedding (2*dim,) on CPU
        """
        return self.entity_embeddings.weight[entity_idx].detach().cpu()

    def get_relation_embedding(self, relation_idx: int) -> torch.Tensor:
        """
        Retrieve embedding for a single relation.

        Args:
            relation_idx: Relation index

        Returns:
            Phase embedding (dim,) on CPU
        """
        return self.relation_embeddings.weight[relation_idx].detach().cpu()

    def predict_tail(
        self,
        head_idx: int,
        relation_idx: int,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Predict most likely tail entities for (head, relation, ?).

        Args:
            head_idx: Head entity index
            relation_idx: Relation index
            top_k: Number of top predictions to return

        Returns:
            List of (entity_idx, score) tuples sorted by score descending
        """
        self.eval()
        with torch.no_grad():
            # Get head and relation embeddings
            head_emb = self.entity_embeddings.weight[head_idx].unsqueeze(0)  # (1, 2*dim)
            rel_emb = self.relation_embeddings.weight[relation_idx].unsqueeze(0)  # (1, dim)

            # Get all tail embeddings
            all_tail_emb = self.entity_embeddings.weight  # (num_entities, 2*dim)

            # Rotate head by relation
            head_rotated = self.complex_multiply(head_emb, rel_emb)  # (1, 2*dim)

            # Compute distances to all tails
            distances = torch.sum(
                torch.abs(head_rotated - all_tail_emb),
                dim=-1
            )  # (num_entities,)

            # Convert to scores (negative distance)
            scores = -distances

            # Get top-k
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

            return [
                (idx.item(), score.item())
                for idx, score in zip(top_indices, top_scores)
            ]


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """
    Clip gradients by global norm for training stability.

    This prevents exploding gradients during complex-valued optimization.

    Args:
        model: The model whose gradients to clip
        max_norm: Maximum gradient norm (default: 1.0)

    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
