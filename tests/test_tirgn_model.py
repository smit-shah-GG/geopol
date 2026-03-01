"""Unit tests for TiRGN model architecture.

Covers forward pass shapes, loss computation, TKGModelProtocol compliance,
component correctness, copy-generation fusion, and mixed precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from src.training.models.components.global_history import (
    GlobalHistoryEncoder,
    build_history_vocabulary,
    get_history_mask,
)
from src.training.models.components.time_conv_transe import TimeConvTransEDecoder
from src.training.models.regcn_jax import GraphSnapshot
from src.training.models.tirgn_jax import TiRGN, create_tirgn_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_ENTITIES = 10
NUM_RELATIONS = 5
EMBEDDING_DIM = 32
NUM_SNAPSHOTS = 3
EDGES_PER_SNAPSHOT = 8


@pytest.fixture()
def rngs() -> nnx.Rngs:
    return nnx.Rngs(params=42, dropout=43)


@pytest.fixture()
def model() -> TiRGN:
    """Small TiRGN model for testing."""
    return create_tirgn_model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        num_layers=1,
        history_rate=0.3,
        seed=0,
    )


@pytest.fixture()
def snapshots() -> list[GraphSnapshot]:
    """Three random GraphSnapshots."""
    key = jax.random.PRNGKey(42)
    snaps = []
    num_rels_with_inv = NUM_RELATIONS * 2
    for _ in range(NUM_SNAPSHOTS):
        key, sk = jax.random.split(key)
        edge_index = jax.random.randint(sk, (2, EDGES_PER_SNAPSHOT), 0, NUM_ENTITIES)
        key, sk = jax.random.split(key)
        edge_type = jax.random.randint(sk, (EDGES_PER_SNAPSHOT,), 0, num_rels_with_inv)
        snaps.append(
            GraphSnapshot(
                edge_index=edge_index,
                edge_type=edge_type,
                num_edges=EDGES_PER_SNAPSHOT,
            )
        )
    return snaps


@pytest.fixture()
def pos_triples() -> jnp.ndarray:
    return jnp.array([[0, 1, 2], [3, 4, 5], [1, 0, 9], [7, 3, 2], [5, 2, 8]])


@pytest.fixture()
def neg_triples() -> jnp.ndarray:
    return jnp.array([[0, 1, 3], [3, 4, 7], [1, 0, 4], [7, 3, 6], [5, 2, 1]])


@pytest.fixture()
def history_vocab(snapshots: list[GraphSnapshot]) -> dict[tuple[int, int], set[int]]:
    """Build a history vocabulary from the test snapshots."""
    snap_triples = []
    for snap in snapshots:
        src = np.asarray(snap.edge_index[0])
        etype = np.asarray(snap.edge_type)
        dst = np.asarray(snap.edge_index[1])
        snap_triples.append(np.stack([src, etype, dst], axis=1))
    return build_history_vocabulary(
        snap_triples, NUM_ENTITIES, NUM_RELATIONS * 2, window_size=50
    )


# ---------------------------------------------------------------------------
# Test 1: Forward pass shapes
# ---------------------------------------------------------------------------


def test_tirgn_forward_shapes(
    model: TiRGN, snapshots: list[GraphSnapshot]
) -> None:
    """evolve_embeddings returns (num_entities, embedding_dim)."""
    emb = model.evolve_embeddings(snapshots, training=False)
    assert emb.shape == (NUM_ENTITIES, EMBEDDING_DIM), (
        f"Expected ({NUM_ENTITIES}, {EMBEDDING_DIM}), got {emb.shape}"
    )
    assert emb.dtype == jnp.float32, f"Expected float32, got {emb.dtype}"


# ---------------------------------------------------------------------------
# Test 2: compute_scores shape
# ---------------------------------------------------------------------------


def test_tirgn_compute_scores_shape(
    model: TiRGN,
    snapshots: list[GraphSnapshot],
    pos_triples: jnp.ndarray,
) -> None:
    """compute_scores returns (batch,) for given triples."""
    emb = model.evolve_embeddings(snapshots, training=False)
    scores = model.compute_scores(emb, pos_triples)
    assert scores.shape == (pos_triples.shape[0],), (
        f"Expected ({pos_triples.shape[0]},), got {scores.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: compute_loss is scalar, finite, non-negative
# ---------------------------------------------------------------------------


def test_tirgn_compute_loss_scalar(
    model: TiRGN,
    snapshots: list[GraphSnapshot],
    pos_triples: jnp.ndarray,
    neg_triples: jnp.ndarray,
) -> None:
    """compute_loss returns a scalar NLL loss that is finite and non-negative."""
    loss = model.compute_loss(snapshots, pos_triples, neg_triples)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    # NLL loss is -log(prob) where prob in (0, 1], so loss >= 0
    assert float(loss) >= 0.0, f"NLL loss should be non-negative, got {float(loss)}"


# ---------------------------------------------------------------------------
# Test 4: Protocol compliance
# ---------------------------------------------------------------------------


def test_tirgn_protocol_compliance(model: TiRGN) -> None:
    """TiRGN satisfies TKGModelProtocol (isinstance check)."""
    from src.protocols.tkg import TKGModelProtocol

    assert isinstance(model, TKGModelProtocol), (
        "TiRGN does not satisfy TKGModelProtocol -- "
        "missing methods or attributes"
    )

    # Verify method signatures explicitly
    import inspect

    evolve_sig = inspect.signature(model.evolve_embeddings)
    assert "snapshots" in evolve_sig.parameters
    assert "training" in evolve_sig.parameters

    scores_sig = inspect.signature(model.compute_scores)
    assert "entity_emb" in scores_sig.parameters
    assert "triples" in scores_sig.parameters

    loss_sig = inspect.signature(model.compute_loss)
    assert "snapshots" in loss_sig.parameters
    assert "pos_triples" in loss_sig.parameters
    assert "neg_triples" in loss_sig.parameters
    assert "margin" in loss_sig.parameters


# ---------------------------------------------------------------------------
# Test 5: Protocol-required attributes
# ---------------------------------------------------------------------------


def test_tirgn_attributes(model: TiRGN) -> None:
    """Model exposes num_entities, num_relations, embedding_dim."""
    assert model.num_entities == NUM_ENTITIES
    assert model.num_relations == NUM_RELATIONS
    assert model.embedding_dim == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Test 6: TimeConvTransEDecoder standalone shape test
# ---------------------------------------------------------------------------


def test_time_conv_transe_shape(rngs: nnx.Rngs) -> None:
    """TimeConvTransEDecoder produces (batch, num_entities) logits."""
    decoder = TimeConvTransEDecoder(
        embedding_dim=EMBEDDING_DIM,
        num_relations=NUM_RELATIONS * 2,
        num_filters=16,
        kernel_size=3,
        rngs=rngs,
    )

    entity_emb = jax.random.normal(
        jax.random.PRNGKey(0), (NUM_ENTITIES, EMBEDDING_DIM)
    )
    triples = jnp.array([[0, 1, 2], [3, 4, 5]])
    time_indices = jnp.array([0, 10])

    scores = decoder(entity_emb, triples, time_indices, training=False)
    assert scores.shape == (2, NUM_ENTITIES), (
        f"Expected (2, {NUM_ENTITIES}), got {scores.shape}"
    )
    assert scores.dtype == jnp.float32, (
        f"Expected float32 output, got {scores.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 7: Global history mask correctness
# ---------------------------------------------------------------------------


def test_global_history_mask(
    model: TiRGN,
    snapshots: list[GraphSnapshot],
    history_vocab: dict[tuple[int, int], set[int]],
    rngs: nnx.Rngs,
) -> None:
    """History mask correctly zeroes out non-history entities in scores."""
    entity_emb = model.evolve_embeddings(snapshots, training=False)
    triples = jnp.array([[0, 1, 2]])
    time_indices = jnp.array([0])

    # Build mask for this query
    mask = get_history_mask(
        history_vocab,
        np.array([0]),
        np.array([1]),
        NUM_ENTITIES,
    )

    # Get history-constrained probabilities
    hist_probs = model.global_history(
        entity_emb, triples, time_indices, mask, training=False
    )

    assert hist_probs.shape == (1, NUM_ENTITIES)

    # If (0, 1) is in vocab, non-historical entities should have ~0 probability
    if (0, 1) in history_vocab:
        historical_objects = history_vocab[(0, 1)]
        for e in range(NUM_ENTITIES):
            if e not in historical_objects:
                assert float(hist_probs[0, e]) < 1e-6, (
                    f"Entity {e} not in history but has prob {float(hist_probs[0, e])}"
                )

    # All probabilities should be non-negative
    assert jnp.all(hist_probs >= 0.0), "Negative probabilities found"


# ---------------------------------------------------------------------------
# Test 8: Copy-generation fusion produces valid distribution
# ---------------------------------------------------------------------------


def test_copy_generation_fusion(
    model: TiRGN,
    snapshots: list[GraphSnapshot],
    pos_triples: jnp.ndarray,
    history_vocab: dict[tuple[int, int], set[int]],
) -> None:
    """Fused distribution is a valid probability distribution."""
    probs = model.predict(snapshots, pos_triples, history_vocab=history_vocab)

    assert probs.shape == (pos_triples.shape[0], NUM_ENTITIES)

    # Check all values are non-negative
    assert jnp.all(probs >= 0.0), "Negative probabilities in fused distribution"

    # Check rows sum to ~1.0
    row_sums = jnp.sum(probs, axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-3), (
        f"Row sums deviate from 1.0: {row_sums}"
    )


# ---------------------------------------------------------------------------
# Test 9: Mixed precision -- bfloat16 in intermediate computations
# ---------------------------------------------------------------------------


def test_mixed_precision(rngs: nnx.Rngs) -> None:
    """Verify that intermediate computations use bfloat16."""
    decoder = TimeConvTransEDecoder(
        embedding_dim=EMBEDDING_DIM,
        num_relations=NUM_RELATIONS * 2,
        num_filters=16,
        kernel_size=3,
        rngs=rngs,
    )

    # Conv layer should be configured for bfloat16 compute
    assert decoder.conv.dtype == jnp.bfloat16, (
        f"Conv dtype should be bfloat16, got {decoder.conv.dtype}"
    )
    # But parameters should be float32
    assert decoder.conv.kernel[...].dtype == jnp.float32, (
        f"Conv kernel should be float32, got {decoder.conv.kernel[...].dtype}"
    )

    # FC layer should be configured for bfloat16 compute
    assert decoder.fc.dtype == jnp.bfloat16, (
        f"FC dtype should be bfloat16, got {decoder.fc.dtype}"
    )
    assert decoder.fc.kernel[...].dtype == jnp.float32, (
        f"FC kernel should be float32, got {decoder.fc.kernel[...].dtype}"
    )

    # Verify the output is float32 (cast back for numerical stability)
    entity_emb = jax.random.normal(
        jax.random.PRNGKey(0), (NUM_ENTITIES, EMBEDDING_DIM)
    )
    triples = jnp.array([[0, 1, 2]])
    time_indices = jnp.array([0])

    scores = decoder(entity_emb, triples, time_indices, training=False)
    assert scores.dtype == jnp.float32, (
        f"Output should be float32 for stability, got {scores.dtype}"
    )
