"""Temporal Knowledge Graph model protocol.

Defines TKGModelProtocol -- the contract that any TKG model backend must
satisfy. RE-GCN (regcn_jraph.py) is the current implementation; TiRGN
(Phase 11) will be the second.

The protocol uses @runtime_checkable so downstream code can validate
implementations via isinstance() without importing concrete classes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class TKGModelProtocol(Protocol):
    """Contract for temporal knowledge graph model backends.

    Any class satisfying this protocol can be used as a drop-in TKG model
    for the forecasting pipeline. The three methods cover the full
    predict/train lifecycle:

    - evolve_embeddings: temporal snapshot processing -> entity vectors
    - compute_scores: entity vectors + triples -> scalar scores
    - compute_loss: end-to-end training loss from snapshots + triples
    """

    num_entities: int
    num_relations: int
    embedding_dim: int

    def evolve_embeddings(
        self, snapshots: list, training: bool = False, **kwargs: object
    ) -> Array:
        """Evolve entity embeddings through temporal graph snapshots.

        Args:
            snapshots: Sequence of temporal graph snapshots (format is
                implementation-defined: TemporalGraph, GraphSnapshot, etc.)
            training: Whether dropout / stochastic layers are active.
            **kwargs: Implementation-specific options (e.g. rng_key).

        Returns:
            Entity embedding matrix of shape (num_entities, embedding_dim).
        """
        ...

    def compute_scores(self, entity_emb: Array, triples: Array) -> Array:
        """Score a batch of (subject, relation, object) triples.

        Args:
            entity_emb: Entity embeddings (num_entities, embedding_dim).
            triples: Integer triples (batch, 3) -- [s, r, o].

        Returns:
            Scalar scores (batch,).
        """
        ...

    def compute_loss(
        self,
        snapshots: list,
        pos_triples: Array,
        neg_triples: Array,
        margin: float = 1.0,
        **kwargs: object,
    ) -> Array:
        """Compute training loss from temporal snapshots and triples.

        Args:
            snapshots: Temporal graph snapshots (same type as evolve_embeddings).
            pos_triples: Positive (true) triples (batch, 3).
            neg_triples: Negative (corrupted) triples (batch * num_neg, 3).
            margin: Margin for ranking loss.
            **kwargs: Implementation-specific options (e.g. rng_key).

        Returns:
            Scalar loss value.
        """
        ...


class StubTiRGN:
    """Minimal stub satisfying TKGModelProtocol for contract verification.

    Not a real implementation -- Phase 11 will provide the full TiRGN port.
    This exists solely to prove the protocol is satisfiable by a second,
    independent class.
    """

    def __init__(
        self,
        num_entities: int = 100,
        num_relations: int = 10,
        embedding_dim: int = 64,
    ) -> None:
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

    def evolve_embeddings(
        self, snapshots: list, training: bool = False, **kwargs: object
    ) -> Array:
        import jax.numpy as jnp

        return jnp.zeros((self.num_entities, self.embedding_dim))

    def compute_scores(self, entity_emb: Array, triples: Array) -> Array:
        import jax.numpy as jnp

        return jnp.zeros((triples.shape[0],))

    def compute_loss(
        self,
        snapshots: list,
        pos_triples: Array,
        neg_triples: Array,
        margin: float = 1.0,
        **kwargs: object,
    ) -> Array:
        import jax.numpy as jnp

        return jnp.array(0.0)
