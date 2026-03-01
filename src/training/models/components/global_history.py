"""Global history encoder for TiRGN.

The global history encoder is the key innovation of TiRGN over RE-GCN. It
builds a sparse binary vocabulary matrix tracking which (subject, relation)
pairs have historically produced which objects across all prior timestamps,
then uses this information to constrain entity predictions.

The vocabulary is stored as a dictionary-of-keys mapping
(subject, relation) -> set[object], NOT as a dense matrix. For GDELT-scale
data (~500K entities, ~300 relations), the dense matrix would be ~28GB.
The per-batch mask construction is O(batch_size * avg_vocab_size), which is
tractable.

Reference:
    Li et al. (2022). TiRGN: Time-Guided Recurrent Graph Network with
    Local-Global Historical Patterns for Temporal Knowledge Graph Reasoning.
    IJCAI 2022.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from .time_conv_transe import TimeConvTransEDecoder

# Type alias for the sparse history vocabulary.
# Maps (subject_id, relation_id) -> set of object_ids that appeared with that pair.
HistoryVocab = dict[tuple[int, int], set[int]]


def build_history_vocabulary(
    snapshots: list[Union[np.ndarray, Array]],
    num_entities: int,
    num_relations: int,
    window_size: int = 50,
) -> HistoryVocab:
    """Build a sparse history vocabulary from snapshot triple arrays.

    Scans the last ``window_size`` snapshots and records, for every
    (subject, relation) pair observed, the set of objects that appeared
    with that pair. This is a preprocessing step -- called once before
    training or per-timestamp during incremental updates, NOT during the
    forward pass.

    Args:
        snapshots: List of triple arrays, each (num_triples, 3) with
            columns [subject, relation, object]. Can be numpy or JAX arrays.
        num_entities: Total entity count (for validation only).
        num_relations: Total relation count including inverse relations.
        window_size: Number of most recent snapshots to consider. Default 50
            per CONTEXT.md.

    Returns:
        Dictionary mapping (subject, relation) -> set of object entity ids.
        This is memory-efficient: only non-empty entries are stored.
    """
    vocab: HistoryVocab = defaultdict(set)

    # Only consider the last window_size snapshots
    recent = snapshots[-window_size:] if len(snapshots) > window_size else snapshots

    for snap in recent:
        # Convert to numpy for efficient Python-level iteration
        if hasattr(snap, "numpy"):
            arr = np.asarray(snap)
        elif isinstance(snap, np.ndarray):
            arr = snap
        else:
            arr = np.asarray(snap)

        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                f"Expected snapshot shape (N, 3), got {arr.shape}"
            )

        for row_idx in range(arr.shape[0]):
            s = int(arr[row_idx, 0])
            r = int(arr[row_idx, 1])
            o = int(arr[row_idx, 2])
            vocab[(s, r)].add(o)

    # Convert defaultdict to regular dict to prevent silent insertions
    return dict(vocab)


def get_history_mask(
    vocab: HistoryVocab,
    subjects: np.ndarray,
    relations: np.ndarray,
    num_entities: int,
) -> Array:
    """Construct a per-batch boolean mask from the sparse history vocabulary.

    For each (subject, relation) pair in the batch, looks up the set of
    historical objects and sets those positions to True in the output mask.
    This avoids materializing the full (E*R, E) dense matrix.

    Args:
        vocab: Sparse history vocabulary from ``build_history_vocabulary``.
        subjects: (batch_size,) subject entity indices.
        relations: (batch_size,) relation indices.
        num_entities: Total entity count.

    Returns:
        Boolean mask of shape (batch_size, num_entities). True where the
        entity appeared as an object for the given (subject, relation) in
        history.
    """
    batch_size = len(subjects)
    mask = np.zeros((batch_size, num_entities), dtype=np.bool_)

    for i in range(batch_size):
        s = int(subjects[i])
        r = int(relations[i])
        objects = vocab.get((s, r))
        if objects is not None:
            obj_indices = np.array(list(objects), dtype=np.int32)
            mask[i, obj_indices] = True

    return jnp.array(mask)


class GlobalHistoryEncoder(nnx.Module):
    """History-constrained entity scorer using a dedicated Time-ConvTransE.

    Owns a separate ``TimeConvTransEDecoder`` instance (the "history decoder")
    whose weights are independent from the raw decoder. Scores are computed
    for all entities, then non-historical entities are masked to -inf BEFORE
    softmax, ensuring probability mass concentrates on historically observed
    objects.

    This module does NOT own the history vocabulary. The vocabulary dict is
    passed in from the training loop or data pipeline.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        num_filters: int = 32,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # History decoder -- separate weights from the raw decoder
        self.hist_decoder = TimeConvTransEDecoder(
            embedding_dim=embedding_dim,
            num_relations=num_relations,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(
        self,
        entity_emb: Array,
        triples: Array,
        time_indices: Array,
        history_mask: Array,
        training: bool = True,
    ) -> Array:
        """Score entities with history constraint.

        Args:
            entity_emb: (num_entities, embedding_dim) entity embeddings.
            triples: (batch, 3) [subject, relation, object] indices.
            time_indices: (batch,) integer time step indices.
            history_mask: (batch, num_entities) boolean mask. True for entities
                that appeared in history for the corresponding (s, r) pair.
            training: Training mode flag for dropout and batch norm.

        Returns:
            Softmax probabilities of shape (batch, num_entities), with
            probability mass concentrated on historically observed entities.
        """
        # Score all entities via history decoder
        hist_scores = self.hist_decoder(
            entity_emb, triples, time_indices, training=training
        )

        # Mask non-historical entities BEFORE softmax (Pitfall 2 from research)
        # Setting to -inf ensures zero probability after softmax
        hist_scores = jnp.where(history_mask, hist_scores, -1e9)

        # Softmax over entity dimension
        hist_probs = jax.nn.softmax(hist_scores, axis=-1)

        return hist_probs
