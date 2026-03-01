"""TiRGN implementation in JAX/Flax NNX.

TiRGN (Time-Guided Recurrent Graph Network) extends RE-GCN with:
1. A global history encoder that captures repeated facts across prior timestamps
   via a sparse binary vocabulary matrix.
2. A Time-ConvTransE decoder that integrates learned periodic and non-periodic
   time embeddings into the scoring function.
3. A copy-generation fusion mechanism that interpolates between raw (open
   vocabulary) and history-constrained distributions via a scalar history_rate.

This is a clean Flax NNX implementation, NOT a wrapper around REGCN. It reuses
the structural components (RelationalGraphConv, GRUCell, GraphSnapshot) from
regcn_jax.py but composes them differently.

Reference:
    Li et al. (2022). TiRGN: Time-Guided Recurrent Graph Network with
    Local-Global Historical Patterns for Temporal Knowledge Graph Reasoning.
    IJCAI 2022.
"""

from __future__ import annotations

import logging
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from src.training.models.components.global_history import (
    GlobalHistoryEncoder,
    HistoryVocab,
    get_history_mask,
)
from src.training.models.components.time_conv_transe import TimeConvTransEDecoder
from src.training.models.regcn_jax import (
    GRUCell,
    GraphSnapshot,
    RelationalGraphConv,
)

logger = logging.getLogger(__name__)


class TiRGN(nnx.Module):
    """TiRGN: Time-Guided Recurrent Graph Network.

    Processes temporal knowledge graph snapshots through:
    1. R-GCN layers for per-snapshot graph structure encoding
    2. Entity GRU for temporal evolution of entity embeddings
    3. Relation GRU for temporal evolution of relation embeddings (TiRGN-specific)
    4. Time-ConvTransE raw decoder for open-vocabulary scoring
    5. Global history encoder for history-constrained scoring
    6. Copy-generation fusion: linear interpolation of raw and history distributions

    Uses jax.checkpoint on the temporal loop for memory efficiency.

    Satisfies TKGModelProtocol (evolve_embeddings, compute_scores, compute_loss).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout_rate: float = 0.2,
        history_rate: float = 0.3,
        history_window: int = 50,
        *,
        rngs: nnx.Rngs,
    ):
        # Protocol-required attributes
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Config
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.history_rate: float = history_rate
        self.history_window: int = history_window

        # Initial entity embeddings -- xavier uniform initialization
        limit = (6.0 / (num_entities + embedding_dim)) ** 0.5
        self.entity_emb = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (num_entities, embedding_dim),
                minval=-limit,
                maxval=limit,
            )
        )

        # Initial relation embeddings (including inverse relations)
        num_rels_with_inv = num_relations * 2
        rel_limit = (6.0 / (num_rels_with_inv + embedding_dim)) ** 0.5
        self.relation_emb = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (num_rels_with_inv, embedding_dim),
                minval=-rel_limit,
                maxval=rel_limit,
            )
        )

        # R-GCN layers (shared structural component with RE-GCN)
        self.rgcn_layers = nnx.List([
            RelationalGraphConv(
                embedding_dim,
                embedding_dim,
                num_rels_with_inv,
                num_bases,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        # Entity GRU: evolves entity embeddings over time
        self.entity_gru = GRUCell(embedding_dim, rngs=rngs)

        # Relation GRU: evolves relation embeddings over time (TiRGN-specific)
        # The reference uses nn.GRUCell(h_dim*2, h_dim) because it concatenates
        # relation embeddings with aggregated context. We use a projection layer
        # + existing GRUCell to avoid modifying the shared GRUCell class.
        self.rel_input_proj = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            rngs=rngs,
        )
        self.relation_gru = GRUCell(embedding_dim, rngs=rngs)

        # Raw decoder: scores using evolved entity embeddings (open vocabulary)
        self.raw_decoder = TimeConvTransEDecoder(
            embedding_dim=embedding_dim,
            num_relations=num_rels_with_inv,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Global history encoder: scores using history-constrained distribution
        self.global_history = GlobalHistoryEncoder(
            num_entities=num_entities,
            num_relations=num_rels_with_inv,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Dropout RNGs for R-GCN layers
        self._dropout_rngs = rngs

    def _encode_snapshot(
        self,
        x: Array,
        snapshot: GraphSnapshot,
        training: bool = True,
    ) -> Array:
        """Encode a single graph snapshot through R-GCN layers.

        Args:
            x: Input entity embeddings (num_entities, embedding_dim).
            snapshot: Graph snapshot with edges.
            training: Whether to apply dropout.

        Returns:
            Updated entity embeddings (num_entities, embedding_dim).
        """
        for layer in self.rgcn_layers:
            x = layer(x, snapshot.edge_index, snapshot.edge_type, self.num_entities)
            x = jax.nn.relu(x)
            if training and self.dropout_rate > 0:
                keep_rate = 1.0 - self.dropout_rate
                mask = jax.random.bernoulli(
                    jax.random.PRNGKey(hash(id(layer)) % 2**31),
                    keep_rate,
                    x.shape,
                )
                x = x * mask / keep_rate
        return x

    def _aggregate_relation_context(
        self,
        snapshot: GraphSnapshot,
        entity_emb: Array,
        num_rels: int,
    ) -> Array:
        """Aggregate per-relation edge features from a snapshot.

        For each relation type, computes the mean of source entity embeddings
        for all edges of that type. This gives a (num_rels, embedding_dim)
        context tensor that feeds into the relation GRU.

        Args:
            snapshot: Graph snapshot.
            entity_emb: Current entity embeddings (num_entities, embedding_dim).
            num_rels: Number of relation types (including inverse).

        Returns:
            Aggregated relation context (num_rels, embedding_dim).
        """
        source_idx = snapshot.edge_index[0]
        edge_type = snapshot.edge_type

        # Gather source entity embeddings for all edges
        src_feats = entity_emb[source_idx]  # (num_edges, embedding_dim)

        # Aggregate by relation type using segment_sum
        rel_sum = jnp.zeros((num_rels, self.embedding_dim))
        rel_sum = rel_sum.at[edge_type].add(src_feats)

        # Count edges per relation for mean computation
        rel_count = jnp.zeros(num_rels)
        rel_count = rel_count.at[edge_type].add(1.0)
        rel_count = jnp.maximum(rel_count, 1.0)

        return rel_sum / rel_count[:, None]

    def evolve_embeddings(
        self,
        snapshots: list,
        training: bool = False,
        **kwargs: object,
    ) -> Array:
        """Evolve entity embeddings through temporal graph snapshots.

        Processes snapshots sequentially through R-GCN encoding + entity GRU +
        relation GRU. Uses jax.checkpoint on each snapshot for memory efficiency.

        Also evolves relation embeddings through a separate relation GRU
        (TiRGN-specific). The relation context is aggregated per-relation
        from edge source features, projected, then fed through the GRU.

        Args:
            snapshots: List of GraphSnapshot instances.
            training: Whether in training mode (dropout active).
            **kwargs: Ignored (protocol compatibility).

        Returns:
            Final entity embeddings (num_entities, embedding_dim).
        """
        h = self.entity_emb[...]
        r = self.relation_emb[...]
        num_rels = self.num_relations * 2

        for snapshot in snapshots:
            # Evolve relation embeddings via relation GRU
            rel_context = self._aggregate_relation_context(snapshot, h, num_rels)
            rel_projected = self.rel_input_proj(rel_context)
            r = self.relation_gru(r, rel_projected)

            # R-GCN message passing on current snapshot
            @jax.checkpoint
            def process_snapshot(h_in, snap):
                x = self._encode_snapshot(h_in, snap, training)
                h_out = self.entity_gru(h_in, x)
                return h_out

            h = process_snapshot(h, snapshot)

        return h

    def compute_scores(
        self,
        entity_emb: Array,
        triples: Array,
    ) -> Array:
        """Score individual triples using the raw decoder only.

        This method satisfies TKGModelProtocol.compute_scores. It does NOT
        include history fusion -- that happens in compute_loss and predict.
        The protocol requires (batch,) output shape per triple, which we
        achieve by gathering the diagonal from the all-entity scores.

        Args:
            entity_emb: Entity embeddings (num_entities, embedding_dim).
            triples: (batch, 3) [subject, relation, object] indices.

        Returns:
            Scores of shape (batch,).
        """
        # Use a zero time index as fallback when called without temporal context
        time_indices = jnp.zeros(triples.shape[0], dtype=jnp.int32)

        # Get all-entity scores from raw decoder
        all_scores = self.raw_decoder(
            entity_emb, triples, time_indices, training=False
        )

        # Extract the score for the actual object in each triple
        objects = triples[:, 2]
        scores = all_scores[jnp.arange(triples.shape[0]), objects]

        return scores

    def _compute_fused_distribution(
        self,
        entity_emb: Array,
        triples: Array,
        time_indices: Array,
        history_mask: Array | None,
        training: bool = True,
    ) -> Array:
        """Compute copy-generation fused probability distribution.

        Args:
            entity_emb: (num_entities, embedding_dim) evolved embeddings.
            triples: (batch, 3) [subject, relation, object].
            time_indices: (batch,) integer time step indices.
            history_mask: (batch, num_entities) boolean mask or None.
            training: Training mode flag.

        Returns:
            Fused probability distribution (batch, num_entities).
        """
        # Raw scores + softmax (open vocabulary / generation mode)
        raw_scores = self.raw_decoder(
            entity_emb, triples, time_indices, training=training
        )
        raw_probs = jax.nn.softmax(raw_scores, axis=-1)

        if history_mask is not None:
            # History scores + masked softmax (copy mode)
            hist_probs = self.global_history(
                entity_emb, triples, time_indices, history_mask, training=training
            )

            # Copy-generation fusion: linear interpolation
            fused_probs = (
                self.history_rate * hist_probs
                + (1.0 - self.history_rate) * raw_probs
            )
        else:
            fused_probs = raw_probs

        return fused_probs

    def compute_loss(
        self,
        snapshots: list,
        pos_triples: Array,
        neg_triples: Array,
        margin: float = 1.0,
        **kwargs: object,
    ) -> Array:
        """Compute NLL loss over the fused copy-generation distribution.

        TiRGN uses NLL loss, NOT margin ranking loss. The fused distribution
        assigns probability to every entity; the loss is -log(P[target]).
        The neg_triples and margin parameters are accepted for protocol
        compatibility with RE-GCN but are NOT used.

        Args:
            snapshots: List of GraphSnapshot instances.
            pos_triples: (batch, 3) positive triples.
            neg_triples: (batch * num_neg, 3) IGNORED -- protocol compatibility.
            margin: IGNORED -- protocol compatibility.
            **kwargs: Must include 'history_vocab' (HistoryVocab) for history
                fusion. If absent, raw-only scoring is used. May also include
                'time_indices' (Array) for temporal encoding.

        Returns:
            Scalar NLL loss.
        """
        if neg_triples is not None and neg_triples.shape[0] > 0:
            logger.debug(
                "TiRGN.compute_loss: neg_triples (%d rows) and margin (%.1f) "
                "are accepted for protocol compatibility but not used. "
                "TiRGN uses NLL loss over the full entity distribution.",
                neg_triples.shape[0],
                margin,
            )

        # Evolve embeddings through temporal snapshots
        entity_emb = self.evolve_embeddings(snapshots, training=True)

        # Extract time indices (default to zeros if not provided)
        time_indices = kwargs.get("time_indices", None)
        if time_indices is None:
            time_indices = jnp.zeros(pos_triples.shape[0], dtype=jnp.int32)

        # Build history mask if vocabulary is available
        history_mask = None
        history_vocab: HistoryVocab | None = kwargs.get("history_vocab", None)
        if history_vocab is not None:
            subjects_np = np.asarray(pos_triples[:, 0])
            relations_np = np.asarray(pos_triples[:, 1])
            history_mask = get_history_mask(
                history_vocab, subjects_np, relations_np, self.num_entities
            )

        # Compute fused distribution
        fused_probs = self._compute_fused_distribution(
            entity_emb, pos_triples, time_indices, history_mask, training=True
        )

        # NLL loss: -log(P[target_entity]) for each triple
        target_entities = pos_triples[:, 2]
        target_probs = fused_probs[
            jnp.arange(pos_triples.shape[0]), target_entities
        ]

        # Clamp for numerical stability in log
        target_probs = jnp.maximum(target_probs, 1e-10)
        nll_loss = -jnp.mean(jnp.log(target_probs))

        return nll_loss

    def predict(
        self,
        snapshots: list,
        query_triples: Array,
        time_indices: Array | None = None,
        history_vocab: HistoryVocab | None = None,
    ) -> Array:
        """Predict entity scores for query triples with copy-generation fusion.

        Convenience method (not part of TKGModelProtocol). Evolves embeddings,
        computes fused scores, and returns the full distribution.

        Args:
            snapshots: List of GraphSnapshot instances.
            query_triples: (batch, 3) query triples.
            time_indices: (batch,) time step indices. Defaults to zeros.
            history_vocab: Optional history vocabulary for copy-generation.

        Returns:
            Fused probability distribution (batch, num_entities).
        """
        entity_emb = self.evolve_embeddings(snapshots, training=False)

        if time_indices is None:
            time_indices = jnp.zeros(query_triples.shape[0], dtype=jnp.int32)

        # Build history mask
        history_mask = None
        if history_vocab is not None:
            subjects_np = np.asarray(query_triples[:, 0])
            relations_np = np.asarray(query_triples[:, 1])
            history_mask = get_history_mask(
                history_vocab, subjects_np, relations_np, self.num_entities
            )

        return self._compute_fused_distribution(
            entity_emb, query_triples, time_indices, history_mask, training=False
        )


def create_tirgn_model(
    num_entities: int,
    num_relations: int,
    embedding_dim: int = 200,
    num_layers: int = 2,
    num_bases: int = 30,
    dropout_rate: float = 0.2,
    history_rate: float = 0.3,
    history_window: int = 50,
    seed: int = 0,
) -> TiRGN:
    """Factory function to create a TiRGN model.

    Args:
        num_entities: Number of unique entities.
        num_relations: Number of unique relations (without inverse).
        embedding_dim: Embedding dimension.
        num_layers: Number of R-GCN layers.
        num_bases: Number of bases for R-GCN basis decomposition.
        dropout_rate: Dropout rate for R-GCN layers.
        history_rate: Alpha for copy-generation fusion (0.0 = raw only, 1.0 = history only).
        history_window: Number of past timestamps for history vocabulary.
        seed: Random seed.

    Returns:
        Initialized TiRGN model.
    """
    rngs = nnx.Rngs(params=seed, dropout=seed + 1)
    return TiRGN(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_bases=num_bases,
        dropout_rate=dropout_rate,
        history_rate=history_rate,
        history_window=history_window,
        rngs=rngs,
    )
