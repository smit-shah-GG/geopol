"""
HyTE Temporal Extension for Knowledge Graph Embeddings

Implements HyTE (Hyperplane-based Temporal Embeddings) as proposed in:
    Dasgupta, S. S., Ray, S. N., & Talukdar, P. (2018).
    HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding.
    EMNLP 2018.

HyTE extends static embeddings (like RotatE) by projecting entity embeddings
onto time-specific hyperplanes. Each time period has an associated hyperplane
defined by a normal vector, and entities are projected orthogonally onto it.

Key Features:
    - Weekly temporal buckets (52 hyperplanes for annual cycle)
    - Orthogonal projection: e_τ = e - (w·e)w where w is time-normal
    - Compatible with any base embedding model (RotatE, TransE, etc.)
    - Memory efficient: only K hyperplane vectors for K time buckets
    - Maintains embedding quality with minimal degradation

Architecture:
    - Base embeddings: From RotatE or other models
    - Temporal hyperplanes: One per time bucket (52 for weekly)
    - Projection function: Projects embeddings onto time-specific hyperplane
    - Temporal-aware scoring: Uses projected embeddings for scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import numpy as np

from knowledge_graph.embeddings import RotatEModel


class HyTETemporalExtension(nn.Module):
    """
    HyTE temporal extension for knowledge graph embeddings.

    Projects entity and relation embeddings onto time-specific hyperplanes
    to model temporal dynamics in knowledge graphs.

    Args:
        embedding_dim: Dimension of base embeddings
        num_time_buckets: Number of temporal buckets (default 52 for weekly)
        use_complex: Whether embeddings are complex-valued (for RotatE)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_time_buckets: int = 52,
        use_complex: bool = True
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_time_buckets = num_time_buckets
        self.use_complex = use_complex

        # Temporal hyperplane normal vectors
        # For complex embeddings, we need 2*embedding_dim
        projection_dim = 2 * embedding_dim if use_complex else embedding_dim

        self.time_normals = nn.Parameter(
            torch.randn(num_time_buckets, projection_dim)
        )

        # Initialize and normalize
        self._initialize_hyperplanes()

    def _initialize_hyperplanes(self):
        """
        Initialize temporal hyperplane normals.

        Normals are initialized randomly and then normalized to unit length.
        """
        with torch.no_grad():
            # Normalize to unit vectors
            self.time_normals.data = F.normalize(
                self.time_normals.data,
                p=2,
                dim=-1
            )

    def timestamp_to_bucket(
        self,
        timestamp: float,
        reference_date: Optional[datetime] = None
    ) -> int:
        """
        Convert Unix timestamp to temporal bucket index.

        Uses weekly buckets by default (52 buckets for full year cycle).

        Args:
            timestamp: Unix timestamp (seconds since epoch)
            reference_date: Reference date for bucket calculation
                           (defaults to Unix epoch)

        Returns:
            Bucket index in [0, num_time_buckets-1]
        """
        if reference_date is None:
            reference_date = datetime(1970, 1, 1)

        # Convert timestamp to datetime (handle numpy types)
        dt = datetime.fromtimestamp(float(timestamp))

        # Calculate week of year (0-51)
        week_of_year = dt.isocalendar()[1] - 1  # ISO week, 0-indexed

        # Map to bucket (handle edge cases where week might be 52 or 53)
        bucket = min(week_of_year, self.num_time_buckets - 1)

        return bucket

    def timestamps_to_buckets(
        self,
        timestamps: torch.Tensor,
        reference_date: Optional[datetime] = None
    ) -> torch.Tensor:
        """
        Batch convert timestamps to bucket indices.

        Args:
            timestamps: Tensor of Unix timestamps (batch_size,)
            reference_date: Reference date for bucket calculation

        Returns:
            Bucket indices (batch_size,) in [0, num_time_buckets-1]
        """
        # Convert to numpy for datetime operations
        timestamps_np = timestamps.cpu().numpy()

        buckets = np.array([
            self.timestamp_to_bucket(ts, reference_date)
            for ts in timestamps_np
        ])

        return torch.tensor(buckets, dtype=torch.long, device=timestamps.device)

    def project_onto_hyperplane(
        self,
        embeddings: torch.Tensor,
        time_bucket: torch.Tensor
    ) -> torch.Tensor:
        """
        Project embeddings onto time-specific hyperplane.

        Implements orthogonal projection: e_τ = e - (w·e)w
        where w is the unit normal vector of the hyperplane.

        Args:
            embeddings: Entity/relation embeddings (batch_size, embedding_dim)
            time_bucket: Time bucket indices (batch_size,)

        Returns:
            Projected embeddings (batch_size, embedding_dim)
        """
        # Get time-specific normals
        # Shape: (batch_size, embedding_dim)
        normals = self.time_normals[time_bucket]

        # Compute dot product: (w·e)
        # Shape: (batch_size,)
        dot_products = torch.sum(embeddings * normals, dim=-1, keepdim=True)

        # Compute projection: e - (w·e)w
        # This projects e onto the hyperplane orthogonal to w
        projected = embeddings - dot_products * normals

        return projected

    def forward(
        self,
        head_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        tail_embeddings: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply temporal projections to embeddings.

        Args:
            head_embeddings: Head entity embeddings (batch_size, emb_dim)
            relation_embeddings: Relation embeddings (batch_size, emb_dim or emb_dim/2)
            tail_embeddings: Tail entity embeddings (batch_size, emb_dim)
            timestamps: Unix timestamps (batch_size,)

        Returns:
            Tuple of (projected_heads, projected_relations, projected_tails)
        """
        # Convert timestamps to buckets
        time_buckets = self.timestamps_to_buckets(timestamps)

        # Project entities onto time-specific hyperplanes
        projected_heads = self.project_onto_hyperplane(head_embeddings, time_buckets)
        projected_tails = self.project_onto_hyperplane(tail_embeddings, time_buckets)

        # For complex embeddings (RotatE), relations are phases (half dimension)
        # We don't project phases, only complex entity embeddings
        if self.use_complex:
            # Relation embeddings are phases, not complex vectors
            # No projection needed
            projected_relations = relation_embeddings
        else:
            # For real-valued embeddings, project relations too
            # Expand relation_embeddings to match dimension if needed
            if relation_embeddings.size(-1) != head_embeddings.size(-1):
                # This is a phase embedding, don't project
                projected_relations = relation_embeddings
            else:
                projected_relations = self.project_onto_hyperplane(
                    relation_embeddings, time_buckets
                )

        return projected_heads, projected_relations, projected_tails

    def enforce_constraints(self):
        """
        Enforce unit norm constraint on hyperplane normals.

        Should be called after each parameter update.
        """
        with torch.no_grad():
            self.time_normals.data = F.normalize(
                self.time_normals.data,
                p=2,
                dim=-1
            )


class TemporalRotatEModel(nn.Module):
    """
    RotatE model with HyTE temporal extensions.

    Combines base RotatE embeddings with temporal hyperplane projections
    for time-aware knowledge graph modeling.

    Args:
        num_entities: Number of unique entities
        num_relations: Number of unique relation types
        embedding_dim: Embedding dimension
        num_time_buckets: Number of temporal buckets (default 52)
        margin: Margin for ranking loss
        negative_samples: Number of negative samples per positive
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        num_time_buckets: int = 52,
        margin: float = 9.0,
        negative_samples: int = 4
    ):
        super().__init__()

        # Base RotatE model
        self.base_model = RotatEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            margin=margin,
            negative_samples=negative_samples
        )

        # Temporal extension
        self.temporal_extension = HyTETemporalExtension(
            embedding_dim=embedding_dim,
            num_time_buckets=num_time_buckets,
            use_complex=True  # RotatE uses complex embeddings
        )

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_time_buckets = num_time_buckets
        self.margin = margin
        self.negative_samples = negative_samples

    def forward(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with temporal awareness.

        Args:
            head_idx: Head entity indices (batch_size,)
            relation_idx: Relation indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            timestamps: Unix timestamps (batch_size,)

        Returns:
            Negative distances (higher = more plausible)
        """
        # Get base embeddings
        head_emb = self.base_model.entity_embeddings(head_idx)
        relation_emb = self.base_model.relation_embeddings(relation_idx)
        tail_emb = self.base_model.entity_embeddings(tail_idx)

        # Apply temporal projections
        head_proj, relation_proj, tail_proj = self.temporal_extension(
            head_emb, relation_emb, tail_emb, timestamps
        )

        # Compute distance using projected embeddings
        dist = self.base_model.distance(head_proj, relation_proj, tail_proj)

        return -dist

    def margin_ranking_loss(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute temporal-aware margin ranking loss.

        Args:
            head_idx: Head entity indices (batch_size,)
            relation_idx: Relation indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            timestamps: Unix timestamps (batch_size,)

        Returns:
            loss: Scalar loss value
            stats: Dictionary with statistics
        """
        batch_size = head_idx.size(0)

        # Compute positive scores
        pos_scores = self.forward(head_idx, relation_idx, tail_idx, timestamps)

        # Generate negative triples (same timestamps)
        neg_head, neg_relation, neg_tail = self.base_model.sample_negative_triples(
            head_idx, relation_idx, tail_idx, self.negative_samples
        )

        # Repeat timestamps for negatives
        timestamps_repeated = timestamps.repeat_interleave(self.negative_samples)

        # Compute negative scores
        neg_scores = self.forward(neg_head, neg_relation, neg_tail, timestamps_repeated)
        neg_scores = neg_scores.view(batch_size, self.negative_samples)

        # Compute margin loss
        pos_scores_expanded = pos_scores.unsqueeze(1)
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
        """Enforce all model constraints."""
        self.base_model.enforce_constraints()
        self.temporal_extension.enforce_constraints()

    def predict_tail(
        self,
        head_idx: int,
        relation_idx: int,
        timestamp: float,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Predict most likely tail entities at given time.

        Args:
            head_idx: Head entity index
            relation_idx: Relation index
            timestamp: Unix timestamp
            top_k: Number of top predictions

        Returns:
            List of (entity_idx, score) tuples
        """
        self.eval()
        with torch.no_grad():
            # Prepare inputs
            head_idx_t = torch.tensor([head_idx])
            relation_idx_t = torch.tensor([relation_idx])
            timestamp_t = torch.tensor([timestamp], dtype=torch.float32)

            # Get base embeddings
            head_emb = self.base_model.entity_embeddings(head_idx_t)
            relation_emb = self.base_model.relation_embeddings(relation_idx_t)

            # Project head
            time_bucket = self.temporal_extension.timestamps_to_buckets(timestamp_t)
            head_proj = self.temporal_extension.project_onto_hyperplane(
                head_emb, time_bucket
            )

            # Project all tails
            all_tail_emb = self.base_model.entity_embeddings.weight
            tail_buckets = time_bucket.expand(all_tail_emb.size(0))
            all_tail_proj = self.temporal_extension.project_onto_hyperplane(
                all_tail_emb, tail_buckets
            )

            # Rotate head
            head_rotated = self.base_model.complex_multiply(head_proj, relation_emb)

            # Compute distances
            distances = torch.sum(
                torch.abs(head_rotated - all_tail_proj),
                dim=-1
            )

            # Convert to scores
            scores = -distances

            # Get top-k
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

            return [
                (idx.item(), score.item())
                for idx, score in zip(top_indices, top_scores)
            ]


def analyze_temporal_projection_quality(
    base_model: RotatEModel,
    temporal_extension: HyTETemporalExtension,
    entity_indices: List[int],
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Analyze quality of temporal projections.

    Measures:
        - Semantic similarity preservation (cosine similarity before/after)
        - Projection magnitude (how much is projected away)
        - Cross-time similarity (how different are projections across time)

    Args:
        base_model: Base RotatE model
        temporal_extension: HyTE temporal extension
        entity_indices: List of entity indices to analyze
        num_samples: Number of time samples to test

    Returns:
        Dictionary with quality metrics
    """
    base_model.eval()
    temporal_extension.eval()

    with torch.no_grad():
        # Get base embeddings for selected entities
        entity_idx_t = torch.tensor(entity_indices)
        base_embeddings = base_model.entity_embeddings(entity_idx_t)

        # Sample different time buckets
        time_buckets = torch.randint(
            0, temporal_extension.num_time_buckets,
            (num_samples,)
        )

        similarities_preserved = []
        projection_magnitudes = []
        cross_time_diffs = []

        for i in range(len(entity_indices)):
            emb = base_embeddings[i:i+1]

            # Project to different times
            projections = []
            for bucket in time_buckets:
                proj = temporal_extension.project_onto_hyperplane(
                    emb, bucket.unsqueeze(0)
                )
                projections.append(proj)

            projections = torch.cat(projections, dim=0)

            # Measure similarity preservation
            base_norm = F.normalize(emb, p=2, dim=-1)
            proj_norms = F.normalize(projections, p=2, dim=-1)
            sims = torch.sum(base_norm * proj_norms, dim=-1)
            similarities_preserved.extend(sims.tolist())

            # Measure projection magnitude (how much is removed)
            base_mag = torch.norm(emb, p=2, dim=-1)
            proj_mags = torch.norm(projections, p=2, dim=-1)
            mag_ratios = proj_mags / base_mag
            projection_magnitudes.extend(mag_ratios.tolist())

            # Measure cross-time differences
            for j in range(len(projections) - 1):
                diff = torch.norm(projections[j] - projections[j+1], p=2)
                cross_time_diffs.append(diff.item())

    metrics = {
        'avg_similarity_preserved': np.mean(similarities_preserved),
        'avg_projection_magnitude': np.mean(projection_magnitudes),
        'avg_cross_time_difference': np.mean(cross_time_diffs),
        'similarity_std': np.std(similarities_preserved),
        'magnitude_std': np.std(projection_magnitudes)
    }

    return metrics
