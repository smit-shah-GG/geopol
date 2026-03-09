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
    PaddedSnapshots,
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
        label_smoothing: float = 0.1,
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
        self.label_smoothing: float = label_smoothing

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

    def _encode_snapshot_masked(
        self,
        x: Array,
        edge_index: Array,
        edge_type: Array,
        edge_mask: Array,
        training: bool = True,
        dropout_keys: Array | None = None,
    ) -> Array:
        """Encode a single snapshot through R-GCN layers with edge masking.

        Args:
            x: Input entity embeddings (num_entities, embedding_dim).
            edge_index: (2, max_edges) padded edge indices.
            edge_type: (max_edges,) padded edge types.
            edge_mask: (max_edges,) float32, 1.0=real 0.0=pad.
            training: Whether to apply dropout.
            dropout_keys: Per-layer PRNG keys for stochastic dropout.
                If None during training, falls back to a fixed key (no
                stochastic regularization — only for backward compat).

        Returns:
            Updated entity embeddings (num_entities, embedding_dim).
        """
        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, edge_index, edge_type, self.num_entities, edge_mask=edge_mask)
            x = jax.nn.relu(x)
            if training and self.dropout_rate > 0:
                key = dropout_keys[i] if dropout_keys is not None else jax.random.PRNGKey(0)
                keep_rate = 1.0 - self.dropout_rate
                mask = jax.random.bernoulli(key, keep_rate, x.shape)
                x = x * mask / keep_rate
        return x

    def _aggregate_relation_context_masked(
        self,
        edge_index: Array,
        edge_type: Array,
        edge_mask: Array,
        entity_emb: Array,
        num_rels: int,
    ) -> Array:
        """Aggregate per-relation edge features with masking for padded edges.

        Args:
            edge_index: (2, max_edges) padded edge indices.
            edge_type: (max_edges,) padded edge types.
            edge_mask: (max_edges,) float32, 1.0=real 0.0=pad.
            entity_emb: Current entity embeddings (num_entities, embedding_dim).
            num_rels: Number of relation types (including inverse).

        Returns:
            Aggregated relation context (num_rels, embedding_dim).
        """
        source_idx = edge_index[0]

        # Gather source features, mask out padded edges
        src_feats = entity_emb[source_idx]  # (max_edges, embedding_dim)
        src_feats = src_feats * edge_mask[:, None]

        # Aggregate by relation type
        rel_sum = jnp.zeros((num_rels, self.embedding_dim))
        rel_sum = rel_sum.at[edge_type].add(src_feats)

        # Count only real edges per relation
        rel_count = jnp.zeros(num_rels)
        rel_count = rel_count.at[edge_type].add(edge_mask)
        rel_count = jnp.maximum(rel_count, 1.0)

        return rel_sum / rel_count[:, None]

    def evolve_embeddings(
        self,
        snapshots: PaddedSnapshots,
        training: bool = False,
        rng_key: Array | None = None,
        **kwargs: object,
    ) -> Array:
        """Evolve entity embeddings through temporal graph snapshots via scan.

        Uses jax.lax.scan to compile ONE R-GCN+GRU iteration template and
        reuse it across all snapshots. Requires uniform-shape PaddedSnapshots.

        Args:
            snapshots: PaddedSnapshots with uniform edge count.
            training: Whether in training mode (dropout active).
            rng_key: PRNG key for stochastic dropout. Split per snapshot
                inside scan so each step gets a unique dropout mask. If None,
                defaults to PRNGKey(0) (deterministic — no regularisation).
            **kwargs: Ignored (protocol compatibility).

        Returns:
            Final entity embeddings (num_entities, embedding_dim).
        """
        h = self.entity_emb[...]
        r = self.relation_emb[...]
        num_rels = self.num_relations * 2

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Guard: empty snapshots (inference without temporal context)
        if snapshots.edge_index.shape[0] == 0:
            return h

        def scan_body(carry, snap_slice):
            h, r, rng_key = carry
            edge_index = snap_slice[0]   # (2, max_edges)
            edge_type = snap_slice[1]    # (max_edges,)
            edge_mask = snap_slice[2]    # (max_edges,)

            # Checkpoint the entire snapshot step: recompute intermediates
            # during backward instead of storing them across all 31 snapshots.
            # Without this, scan stores ~400 MB of basis_msgs per R-GCN layer
            # per snapshot → 2 layers × 31 snapshots × 400 MB ≈ 25 GB OOM.
            @jax.checkpoint
            def _step(h, r, edge_index, edge_type, edge_mask, rng_key):
                # Split key: one sub-key per R-GCN layer + one for next step
                keys = jax.random.split(rng_key, len(self.rgcn_layers) + 1)
                dropout_keys = keys[1:]  # one per layer

                # Relation GRU: aggregate per-relation context, project, evolve
                rel_context = self._aggregate_relation_context_masked(
                    edge_index, edge_type, edge_mask, h, num_rels
                )
                rel_projected = self.rel_input_proj(rel_context)
                r_new = self.relation_gru(r, rel_projected)

                # R-GCN + Entity GRU (with per-layer dropout keys)
                x = self._encode_snapshot_masked(
                    h, edge_index, edge_type, edge_mask, training,
                    dropout_keys=dropout_keys if training else None,
                )
                h_new = self.entity_gru(h, x)
                return h_new, r_new

            h_new, r_new = _step(h, r, edge_index, edge_type, edge_mask, rng_key)
            _, next_key = jax.random.split(rng_key)

            return (h_new, r_new, next_key), None

        xs = (snapshots.edge_index, snapshots.edge_type, snapshots.edge_mask)
        (h_final, _r_final, _), _ = jax.lax.scan(scan_body, (h, r, rng_key), xs)
        return h_final

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
        snapshots: PaddedSnapshots | list,
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
            snapshots: PaddedSnapshots for scan-based evolution.
            pos_triples: (batch, 3) positive triples.
            neg_triples: (batch * num_neg, 3) IGNORED -- protocol compatibility.
            margin: IGNORED -- protocol compatibility.
            **kwargs: Must include 'history_vocab' (HistoryVocab) for history
                fusion. If absent, raw-only scoring is used. May also include
                'time_indices' (Array) for temporal encoding.

        Returns:
            Scalar NLL loss.
        """
        # Evolve embeddings through temporal snapshots
        rng_key = kwargs.get("rng_key", None)
        entity_emb = self.evolve_embeddings(snapshots, training=True, rng_key=rng_key)

        history_mask = kwargs.get("history_mask", None)
        if history_mask is None:
            history_vocab: HistoryVocab | None = kwargs.get("history_vocab", None)
            if history_vocab is not None:
                subjects_np = np.asarray(pos_triples[:, 0])
                relations_np = np.asarray(pos_triples[:, 1])
                history_mask = get_history_mask(
                    history_vocab, subjects_np, relations_np, self.num_entities
                )

        return self.compute_loss_from_embeddings(
            entity_emb, pos_triples, history_mask=history_mask,
        )

    def compute_loss_from_embeddings(
        self,
        entity_emb: Array,
        pos_triples: Array,
        history_mask: Array | None = None,
        time_indices: Array | None = None,
    ) -> Array:
        """Compute NLL loss from pre-evolved entity embeddings.

        Separates the expensive evolve_embeddings scan from the cheap
        decoder scoring so callers can evolve once per epoch and batch
        over this method. Gradients flow through the decoder, history
        encoder, and entity_emb but NOT through the R-GCN/GRU scan
        (entity_emb is treated as a detached input).

        Args:
            entity_emb: (num_entities, embedding_dim) evolved embeddings.
            pos_triples: (batch, 3) positive triples.
            history_mask: (batch, num_entities) boolean mask or None.
            time_indices: (batch,) integer time step indices. Defaults to zeros.

        Returns:
            Scalar NLL loss.
        """
        if time_indices is None:
            time_indices = jnp.zeros(pos_triples.shape[0], dtype=jnp.int32)

        # Compute fused distribution
        fused_probs = self._compute_fused_distribution(
            entity_emb, pos_triples, time_indices, history_mask, training=True
        )

        # Label-smoothed NLL: (1-ε)*NLL_hard + ε*NLL_uniform
        target_entities = pos_triples[:, 2]
        log_probs = jnp.log(jnp.maximum(fused_probs, 1e-10))

        nll_hard = -log_probs[jnp.arange(pos_triples.shape[0]), target_entities]
        nll_uniform = -jnp.mean(log_probs, axis=-1)

        eps = self.label_smoothing
        loss = jnp.mean((1.0 - eps) * nll_hard + eps * nll_uniform)

        return loss

    def predict(
        self,
        snapshots: PaddedSnapshots | list,
        query_triples: Array,
        time_indices: Array | None = None,
        history_vocab: HistoryVocab | None = None,
    ) -> Array:
        """Predict entity scores for query triples with copy-generation fusion.

        Convenience method (not part of TKGModelProtocol). Evolves embeddings,
        computes fused scores, and returns the full distribution.

        Args:
            snapshots: PaddedSnapshots for scan-based evolution.
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
    label_smoothing: float = 0.1,
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
        label_smoothing: Epsilon for label smoothing (0.0 = hard targets).
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
        label_smoothing=label_smoothing,
        rngs=rngs,
    )
