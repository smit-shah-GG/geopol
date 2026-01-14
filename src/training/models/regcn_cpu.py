"""
CPU-optimized RE-GCN implementation without DGL dependency.

Implements the core RE-GCN architecture using pure PyTorch:
- Recurrent evolution of entity embeddings through graph snapshots
- R-GCN style aggregation using sparse matrix operations
- ConvTransE decoder for link prediction scoring

Reference:
    Li et al. (2021). Temporal Knowledge Graph Reasoning Based on
    Evolutional Representation Learning. SIGIR 2021.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class RelationalGraphConv(nn.Module):
    """
    Relational Graph Convolution layer for heterogeneous graphs.

    Aggregates neighbor information across multiple relation types
    using sparse matrix operations. Each relation type has its own
    weight matrix for message transformation.

    Uses basis decomposition when num_relations is large to reduce
    parameter count: W_r = sum_b(a_rb * V_b)

    Attributes:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_relations: Number of relation types (including inverse)
        num_bases: Number of basis matrices for decomposition
        weight: Basis weight matrices (num_bases, in_features, out_features)
        coeff: Basis coefficients per relation (num_relations, num_bases)
        bias: Optional bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        num_bases: int = 30,
        bias: bool = True,
        self_loop: bool = True,
    ):
        """
        Initialize relational graph convolution.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_relations: Number of relation types
            num_bases: Number of basis matrices for weight decomposition
            bias: Whether to include bias term
            self_loop: Whether to include self-loop connections
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)
        self.self_loop = self_loop

        # Basis decomposition for relation weights
        self.weight = nn.Parameter(
            torch.empty(self.num_bases, in_features, out_features)
        )
        self.coeff = nn.Parameter(torch.empty(num_relations, self.num_bases))

        if self_loop:
            self.self_weight = nn.Parameter(
                torch.empty(in_features, out_features)
            )
        else:
            self.register_parameter("self_weight", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.coeff)
        if self.self_loop:
            nn.init.xavier_uniform_(self.self_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Forward pass through relational graph convolution.

        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge indices (2, num_edges) - [source, target] pairs
            edge_type: Relation type for each edge (num_edges,)
            num_nodes: Total number of nodes in graph

        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Compute relation weights from basis decomposition
        # (num_relations, in_features, out_features)
        rel_weight = torch.einsum("rb,bio->rio", self.coeff, self.weight)

        # Initialize output
        out = torch.zeros(num_nodes, self.out_features, device=x.device)

        # Aggregate messages per relation type
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            # Get edges for this relation
            src = source_idx[mask]
            tgt = target_idx[mask]

            # Transform source features
            src_feats = x[src]  # (num_edges_r, in_features)
            msg = src_feats @ rel_weight[r]  # (num_edges_r, out_features)

            # Aggregate to targets (mean aggregation)
            out.index_add_(0, tgt, msg)

        # Normalize by in-degree
        in_degree = torch.zeros(num_nodes, device=x.device)
        in_degree.index_add_(0, target_idx, torch.ones_like(target_idx, dtype=torch.float))
        in_degree = in_degree.clamp(min=1)
        out = out / in_degree.unsqueeze(1)

        # Self-loop contribution
        if self.self_loop:
            out = out + x @ self.self_weight

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out


class ConvTransEDecoder(nn.Module):
    """
    ConvTransE decoder for scoring entity-relation-entity triples.

    Uses 2D convolution over concatenated entity and relation embeddings
    followed by fully-connected layers for scoring.

    Architecture:
        [subject_emb | relation_emb] -> Conv2d -> FC -> score

    Reference:
        Shang et al. (2019). End-to-end Structure-Aware Convolutional Network
        for Knowledge Base Completion. AAAI 2019.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_entities: int,
        num_filters: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize ConvTransE decoder.

        Args:
            embedding_dim: Entity/relation embedding dimension
            num_entities: Total number of entities
            num_filters: Number of convolution filters
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities

        # Reshape embeddings to 2D for convolution
        # Input: (batch, 1, 2, embedding_dim) - [subject; relation]
        self.conv = nn.Conv2d(1, num_filters, (1, kernel_size), padding=(0, kernel_size // 2))
        self.bn = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(dropout)

        # FC layer after convolution
        self.fc = nn.Linear(num_filters * 2 * embedding_dim, embedding_dim)

    def forward(
        self,
        subject_emb: Tensor,
        relation_emb: Tensor,
        all_entity_emb: Tensor,
    ) -> Tensor:
        """
        Compute scores for all possible objects given subject and relation.

        Args:
            subject_emb: Subject embeddings (batch_size, embedding_dim)
            relation_emb: Relation embeddings (batch_size, embedding_dim)
            all_entity_emb: All entity embeddings (num_entities, embedding_dim)

        Returns:
            Scores for all entities (batch_size, num_entities)
        """
        batch_size = subject_emb.size(0)

        # Stack subject and relation: (batch, 2, embedding_dim)
        stacked = torch.stack([subject_emb, relation_emb], dim=1)
        # Add channel dim: (batch, 1, 2, embedding_dim)
        stacked = stacked.unsqueeze(1)

        # Convolution
        x = self.conv(stacked)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Flatten: (batch, num_filters * 2 * embedding_dim)
        x = x.view(batch_size, -1)

        # Project to embedding space
        x = self.fc(x)  # (batch, embedding_dim)

        # Score against all entities via dot product
        scores = x @ all_entity_emb.t()  # (batch, num_entities)

        return scores

    def score_triple(
        self,
        subject_emb: Tensor,
        relation_emb: Tensor,
        object_emb: Tensor,
    ) -> Tensor:
        """
        Score specific subject-relation-object triples.

        Args:
            subject_emb: Subject embeddings (batch_size, embedding_dim)
            relation_emb: Relation embeddings (batch_size, embedding_dim)
            object_emb: Object embeddings (batch_size, embedding_dim)

        Returns:
            Triple scores (batch_size,)
        """
        batch_size = subject_emb.size(0)

        # Stack subject and relation
        stacked = torch.stack([subject_emb, relation_emb], dim=1)
        stacked = stacked.unsqueeze(1)

        # Convolution path
        x = self.conv(stacked)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        # Dot product with object embeddings
        scores = (x * object_emb).sum(dim=1)

        return scores


class REGCN(nn.Module):
    """
    RE-GCN: Recurrent Evolution on Graph Convolutional Network.

    Temporal knowledge graph model that:
    1. Maintains entity embeddings that evolve over time
    2. Uses R-GCN layers to aggregate structural information per snapshot
    3. Feeds aggregated embeddings through GRU for temporal evolution
    4. Uses ConvTransE decoder for link prediction

    Key insight: Entity representations should evolve based on both
    structural context (graph neighbors) and temporal dynamics (history).

    Attributes:
        num_entities: Total number of entities
        num_relations: Number of relation types (forward only, inverse added)
        embedding_dim: Entity embedding dimension
        num_layers: Number of R-GCN layers per snapshot
        entity_embeddings: Initial entity embeddings
        relation_embeddings: Relation embeddings (2x for inverse)
        rgcn_layers: List of relational graph convolution layers
        gru: GRU for temporal evolution
        decoder: ConvTransE decoder for scoring
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2,
    ):
        """
        Initialize RE-GCN model.

        Args:
            num_entities: Total number of entities
            num_relations: Number of forward relation types (inverse added automatically)
            embedding_dim: Entity/relation embedding dimension
            num_layers: Number of R-GCN layers per snapshot
            num_bases: Number of basis matrices for R-GCN weight decomposition
            dropout: Dropout rate
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Entity embeddings (learnable initial state)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # Relation embeddings: num_relations * 2 for forward + inverse
        self.relation_embeddings = nn.Embedding(num_relations * 2, embedding_dim)

        # R-GCN layers for each timestep
        # Each layer aggregates neighbor info per relation type
        self.rgcn_layers = nn.ModuleList([
            RelationalGraphConv(
                embedding_dim,
                embedding_dim,
                num_relations * 2,  # Include inverse relations
                num_bases=num_bases,
                bias=True,
                self_loop=True,
            )
            for _ in range(num_layers)
        ])

        # GRU for temporal evolution
        # Takes aggregated embeddings, outputs evolved embeddings
        # Not using batch_first for easier entity processing
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=False,  # Input: (seq_len, batch, input_size)
        )

        # ConvTransE decoder
        self.decoder = ConvTransEDecoder(
            embedding_dim=embedding_dim,
            num_entities=num_entities,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

        logger.info(
            f"Initialized RE-GCN: {num_entities} entities, "
            f"{num_relations} relations (x2 with inverse), "
            f"dim={embedding_dim}, layers={num_layers}"
        )

    def _init_parameters(self) -> None:
        """Initialize embedding parameters."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def _add_inverse_edges(
        self,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Add inverse edges for bidirectional message passing.

        For edge (u, r, v), adds inverse edge (v, r+num_relations, u).

        Args:
            edge_index: Original edges (2, num_edges)
            edge_type: Original relation types (num_edges,)

        Returns:
            Extended edge_index and edge_type with inverse edges
        """
        # Reverse edge direction
        inv_edge_index = edge_index.flip(0)
        # Inverse relation type = original + num_relations
        inv_edge_type = edge_type + self.num_relations

        # Concatenate original and inverse
        edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
        edge_type = torch.cat([edge_type, inv_edge_type], dim=0)

        return edge_index, edge_type

    def evolve_embeddings(
        self,
        snapshots: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """
        Evolve entity embeddings through temporal graph snapshots.

        For each snapshot:
        1. Apply R-GCN layers to aggregate structural info
        2. Pass aggregated embeddings through GRU
        3. Output represents entity state after seeing this snapshot

        Args:
            snapshots: List of (edge_index, edge_type) tuples per timestep
                      edge_index: (2, num_edges) tensor
                      edge_type: (num_edges,) tensor

        Returns:
            Evolved entity embeddings (num_entities, embedding_dim)
        """
        # Start with initial embeddings
        h = self.entity_embeddings.weight  # (num_entities, embedding_dim)

        # GRU hidden state: (num_layers=1, num_entities_as_batch, hidden_size)
        # Each entity is treated as a separate sequence element in the batch
        gru_hidden = h.unsqueeze(0)  # (1, num_entities, embedding_dim)

        for edge_index, edge_type in snapshots:
            # Add inverse edges for bidirectional aggregation
            edge_index, edge_type = self._add_inverse_edges(edge_index, edge_type)

            # Apply R-GCN layers
            x = h
            for rgcn_layer in self.rgcn_layers:
                x = rgcn_layer(x, edge_index, edge_type, self.num_entities)
                x = F.relu(x)
                x = self.dropout(x)

            # GRU step: update hidden state based on aggregated info
            # With batch_first=False:
            #   Input shape: (seq_len=1, batch=num_entities, input_size=embedding_dim)
            #   Hidden shape: (num_layers=1, batch=num_entities, hidden_size=embedding_dim)
            gru_input = x.unsqueeze(0)  # (1, num_entities, embedding_dim)
            _, gru_hidden = self.gru(gru_input, gru_hidden)

            # Evolved embeddings for next timestep
            h = gru_hidden.squeeze(0)  # (num_entities, embedding_dim)

        return h

    def forward(
        self,
        snapshots: List[Tuple[Tensor, Tensor]],
        query_subjects: Tensor,
        query_relations: Tensor,
    ) -> Tensor:
        """
        Forward pass: evolve embeddings and score all objects.

        Args:
            snapshots: List of (edge_index, edge_type) per timestep
            query_subjects: Subject entity IDs (batch_size,)
            query_relations: Relation type IDs (batch_size,)

        Returns:
            Scores for all entities as objects (batch_size, num_entities)
        """
        # Evolve entity embeddings through history
        entity_emb = self.evolve_embeddings(snapshots)

        # Get query embeddings
        subject_emb = entity_emb[query_subjects]
        relation_emb = self.relation_embeddings(query_relations)

        # Score all entities as potential objects
        scores = self.decoder(subject_emb, relation_emb, entity_emb)

        return scores

    def compute_loss(
        self,
        snapshots: List[Tuple[Tensor, Tensor]],
        positive_triples: Tensor,
        negative_triples: Tensor,
        margin: float = 1.0,
    ) -> Tensor:
        """
        Compute margin-based ranking loss.

        Uses pairwise margin loss: max(0, margin - pos_score + neg_score)

        Args:
            snapshots: List of (edge_index, edge_type) per timestep
            positive_triples: True triples (batch, 3) - [subject, relation, object]
            negative_triples: Corrupted triples (batch, num_neg, 3)
            margin: Margin for ranking loss

        Returns:
            Scalar loss tensor
        """
        # Evolve embeddings
        entity_emb = self.evolve_embeddings(snapshots)

        batch_size = positive_triples.size(0)
        num_neg = negative_triples.size(1)

        # Positive scores
        pos_subjects = positive_triples[:, 0]
        pos_relations = positive_triples[:, 1]
        pos_objects = positive_triples[:, 2]

        pos_subject_emb = entity_emb[pos_subjects]
        pos_relation_emb = self.relation_embeddings(pos_relations)
        pos_object_emb = entity_emb[pos_objects]

        pos_scores = self.decoder.score_triple(
            pos_subject_emb, pos_relation_emb, pos_object_emb
        )  # (batch,)

        # Negative scores
        neg_triples_flat = negative_triples.view(-1, 3)
        neg_subjects = neg_triples_flat[:, 0]
        neg_relations = neg_triples_flat[:, 1]
        neg_objects = neg_triples_flat[:, 2]

        neg_subject_emb = entity_emb[neg_subjects]
        neg_relation_emb = self.relation_embeddings(neg_relations)
        neg_object_emb = entity_emb[neg_objects]

        neg_scores = self.decoder.score_triple(
            neg_subject_emb, neg_relation_emb, neg_object_emb
        )  # (batch * num_neg,)
        neg_scores = neg_scores.view(batch_size, num_neg)

        # Margin ranking loss
        # For each positive, compare against all its negatives
        pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_neg)
        losses = F.relu(margin - pos_scores_expanded + neg_scores)
        loss = losses.mean()

        return loss

    def predict(
        self,
        snapshots: List[Tuple[Tensor, Tensor]],
        subject: int,
        relation: int,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Predict top-k objects for (subject, relation, ?).

        Args:
            snapshots: Historical graph snapshots
            subject: Subject entity ID
            relation: Relation type ID
            k: Number of top predictions

        Returns:
            List of (entity_id, score) tuples
        """
        self.eval()
        with torch.no_grad():
            # Evolve embeddings
            entity_emb = self.evolve_embeddings(snapshots)

            # Query embeddings
            subject_emb = entity_emb[subject].unsqueeze(0)
            relation_emb = self.relation_embeddings(
                torch.tensor([relation], device=entity_emb.device)
            )

            # Score all entities
            scores = self.decoder(subject_emb, relation_emb, entity_emb)
            scores = scores.squeeze(0)

            # Top-k predictions
            top_scores, top_indices = torch.topk(scores, k)
            predictions = [
                (idx.item(), score.item())
                for idx, score in zip(top_indices, top_scores)
            ]

        return predictions


def create_snapshot_from_triples(
    triples: Tensor,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Convert triples array to snapshot format.

    Args:
        triples: (num_triples, 3) array with [subject, relation, object]
        device: Target device

    Returns:
        (edge_index, edge_type) tuple for graph operations
    """
    triples = torch.as_tensor(triples, dtype=torch.long, device=device)
    edge_index = triples[:, [0, 2]].t().contiguous()  # (2, num_edges)
    edge_type = triples[:, 1]  # (num_edges,)
    return edge_index, edge_type
