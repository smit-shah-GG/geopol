"""
RotatE embedding model for knowledge graph representation.

Implements complex-space rotations for relation modeling:
    h ◦ r ≈ t  where ◦ is Hadamard product in complex space

Reference: Sun et al. (2019) "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class RotatEModel(nn.Module):
    """
    RotatE embedding model with complex-space rotations.

    Key features:
    - Entities: Complex vectors (real and imaginary components)
    - Relations: Phase rotations in complex plane
    - Scoring: Distance after rotation h ◦ r ≈ t
    - Constraint: Relations constrained to unit circle |r| = 1

    Args:
        num_entities: Number of unique entities
        num_relations: Number of unique relation types
        embedding_dim: Dimension of embeddings (must be even for complex)
        margin: Margin for margin-based loss (default: 9.0 per RotatE paper)
        entity_init_scale: Initialization scale for entity embeddings
        relation_init_scale: Initialization scale for relation phases
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        entity_init_scale: float = 1.0,
        relation_init_scale: float = 1.0,
    ):
        super().__init__()

        assert embedding_dim % 2 == 0, "Embedding dimension must be even for complex numbers"

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.complex_dim = embedding_dim // 2  # Real and imaginary parts

        # Entity embeddings: Complex vectors (real and imaginary)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings: Phase angles (constrained to [0, 2π])
        # Stored as angles, will convert to complex rotation
        self.relation_embedding = nn.Embedding(num_relations, self.complex_dim)

        # Initialize embeddings
        self._initialize_embeddings(entity_init_scale, relation_init_scale)

    def _initialize_embeddings(self, entity_scale: float, relation_scale: float):
        """
        Initialize embeddings with appropriate scales.

        Entity embeddings: Uniform distribution scaled by entity_scale
        Relation embeddings: Uniform [0, 2π] for phase angles
        """
        # Entity embeddings: uniform in [-scale, scale]
        nn.init.uniform_(
            self.entity_embedding.weight,
            -entity_scale,
            entity_scale
        )

        # Relation embeddings: phases in [0, 2π]
        nn.init.uniform_(
            self.relation_embedding.weight,
            0.0,
            2 * np.pi * relation_scale
        )

    def _get_complex_embeddings(self, entity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get entity embeddings as complex numbers (real and imaginary parts).

        Args:
            entity_ids: Tensor of entity indices [batch_size]

        Returns:
            Tuple of (real, imaginary) tensors [batch_size, complex_dim]
        """
        embeddings = self.entity_embedding(entity_ids)  # [batch_size, embedding_dim]

        # Split into real and imaginary parts
        real = embeddings[:, :self.complex_dim]
        imag = embeddings[:, self.complex_dim:]

        return real, imag

    def _get_rotation_embeddings(self, relation_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get relation rotation as complex numbers from phase angles.

        Relations are constrained to unit circle: r = exp(iθ) = cos(θ) + i*sin(θ)

        Args:
            relation_ids: Tensor of relation indices [batch_size]

        Returns:
            Tuple of (cos(θ), sin(θ)) tensors [batch_size, complex_dim]
        """
        phases = self.relation_embedding(relation_ids)  # [batch_size, complex_dim]

        # Convert to unit circle: r = exp(iθ)
        real = torch.cos(phases)
        imag = torch.sin(phases)

        return real, imag

    def _complex_multiply(
        self,
        a_real: torch.Tensor,
        a_imag: torch.Tensor,
        b_real: torch.Tensor,
        b_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complex multiplication: (a_real + i*a_imag) * (b_real + i*b_imag)

        Result: (a_real*b_real - a_imag*b_imag) + i*(a_real*b_imag + a_imag*b_real)

        Args:
            a_real, a_imag: Real and imaginary parts of first complex number
            b_real, b_imag: Real and imaginary parts of second complex number

        Returns:
            Tuple of (result_real, result_imag)
        """
        result_real = a_real * b_real - a_imag * b_imag
        result_imag = a_real * b_imag + a_imag * b_real

        return result_real, result_imag

    def score_triples(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Score triples using RotatE distance function.

        RotatE score: -||h ◦ r - t||_L2
        Lower distance = higher score (more likely true triple)

        Args:
            head_ids: Head entity indices [batch_size]
            relation_ids: Relation indices [batch_size]
            tail_ids: Tail entity indices [batch_size]

        Returns:
            Scores [batch_size] (negative distances)
        """
        # Get complex embeddings
        h_real, h_imag = self._get_complex_embeddings(head_ids)
        t_real, t_imag = self._get_complex_embeddings(tail_ids)
        r_real, r_imag = self._get_rotation_embeddings(relation_ids)

        # Apply rotation: h ◦ r
        hr_real, hr_imag = self._complex_multiply(h_real, h_imag, r_real, r_imag)

        # Compute distance: ||h ◦ r - t||
        diff_real = hr_real - t_real
        diff_imag = hr_imag - t_imag

        # L2 distance in complex space
        distance = torch.sqrt(diff_real ** 2 + diff_imag ** 2).sum(dim=1)

        # Return negative distance as score (higher = better)
        return -distance

    def forward(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
        negative_head_ids: Optional[torch.Tensor] = None,
        negative_tail_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with positive and negative samples.

        Args:
            head_ids: Positive head entities [batch_size]
            relation_ids: Relations [batch_size]
            tail_ids: Positive tail entities [batch_size]
            negative_head_ids: Negative head samples [batch_size, num_neg] or None
            negative_tail_ids: Negative tail samples [batch_size, num_neg] or None

        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        # Score positive triples
        positive_scores = self.score_triples(head_ids, relation_ids, tail_ids)

        # Score negative triples
        negative_scores = []

        if negative_head_ids is not None:
            # Corrupt heads: (h', r, t)
            batch_size, num_neg = negative_head_ids.shape

            # Expand relation and tail for all negatives
            rel_expanded = relation_ids.unsqueeze(1).expand(batch_size, num_neg).contiguous().view(-1)
            tail_expanded = tail_ids.unsqueeze(1).expand(batch_size, num_neg).contiguous().view(-1)
            neg_heads = negative_head_ids.view(-1)

            neg_scores = self.score_triples(neg_heads, rel_expanded, tail_expanded)
            negative_scores.append(neg_scores.view(batch_size, num_neg))

        if negative_tail_ids is not None:
            # Corrupt tails: (h, r, t')
            batch_size, num_neg = negative_tail_ids.shape

            # Expand head and relation for all negatives
            head_expanded = head_ids.unsqueeze(1).expand(batch_size, num_neg).contiguous().view(-1)
            rel_expanded = relation_ids.unsqueeze(1).expand(batch_size, num_neg).contiguous().view(-1)
            neg_tails = negative_tail_ids.view(-1)

            neg_scores = self.score_triples(head_expanded, rel_expanded, neg_tails)
            negative_scores.append(neg_scores.view(batch_size, num_neg))

        # Concatenate all negative scores
        if negative_scores:
            negative_scores = torch.cat(negative_scores, dim=1)
        else:
            negative_scores = None

        return positive_scores, negative_scores

    def margin_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        margin: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss.

        Loss = max(0, margin + negative_score - positive_score)

        Encourages positive triples to score higher than negatives by margin.

        Args:
            positive_scores: Scores for positive triples [batch_size]
            negative_scores: Scores for negative triples [batch_size, num_neg]
            margin: Override margin (uses self.margin if None)

        Returns:
            Scalar loss value
        """
        if margin is None:
            margin = self.margin

        # Expand positive scores to match negative scores shape
        batch_size, num_neg = negative_scores.shape
        pos_scores_expanded = positive_scores.unsqueeze(1).expand(batch_size, num_neg)

        # Compute pairwise margin loss
        loss = F.relu(margin + negative_scores - pos_scores_expanded)

        # Average over all samples
        return loss.mean()

    def get_entity_embeddings(self) -> np.ndarray:
        """
        Get all entity embeddings as numpy array.

        Returns:
            Array of shape [num_entities, embedding_dim]
        """
        return self.entity_embedding.weight.detach().cpu().numpy()

    def get_relation_embeddings(self) -> np.ndarray:
        """
        Get all relation embeddings as numpy array (phase angles).

        Returns:
            Array of shape [num_relations, complex_dim]
        """
        return self.relation_embedding.weight.detach().cpu().numpy()

    def verify_rotation_constraint(self) -> Dict[str, float]:
        """
        Verify that relation embeddings satisfy unit circle constraint.

        Returns:
            Dict with constraint violation statistics
        """
        phases = self.relation_embedding.weight.detach()

        # Convert to complex rotations
        real = torch.cos(phases)
        imag = torch.sin(phases)

        # Compute norms (should all be 1.0)
        norms = torch.sqrt(real ** 2 + imag ** 2)

        return {
            "mean_norm": norms.mean().item(),
            "min_norm": norms.min().item(),
            "max_norm": norms.max().item(),
            "std_norm": norms.std().item()
        }


def create_negative_samples(
    head_ids: torch.Tensor,
    tail_ids: torch.Tensor,
    num_entities: int,
    num_negatives: int = 4,
    corruption_mode: str = "both"
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Generate negative samples by corrupting heads or tails.

    Args:
        head_ids: Positive head entities [batch_size]
        tail_ids: Positive tail entities [batch_size]
        num_entities: Total number of entities for sampling
        num_negatives: Number of negative samples per positive
        corruption_mode: "head", "tail", or "both"

    Returns:
        Tuple of (negative_heads, negative_tails)
        Either can be None based on corruption_mode
    """
    batch_size = head_ids.shape[0]

    negative_heads = None
    negative_tails = None

    if corruption_mode in ["head", "both"]:
        # Sample random entities for head corruption
        negative_heads = torch.randint(
            0, num_entities,
            (batch_size, num_negatives),
            device=head_ids.device
        )

    if corruption_mode in ["tail", "both"]:
        # Sample random entities for tail corruption
        negative_tails = torch.randint(
            0, num_entities,
            (batch_size, num_negatives),
            device=tail_ids.device
        )

    return negative_heads, negative_tails
