"""TiRGN training loop with early stopping, VRAM monitoring, and observability.

Reuses data loading (``load_gdelt_data``, ``create_graph_snapshots``),
negative sampling, and MRR evaluation from ``src.training.train_jax``.
Does NOT duplicate any existing infrastructure.

Key differences from RE-GCN training (``train_regcn``):
1. Uses NLL loss over fused copy-generation distribution, not margin ranking.
2. Builds a global history vocabulary ONCE before training begins.
3. Logs VRAM usage per epoch for GPU envelope validation (TKG-04).
4. Writes all metrics to TensorBoard + optional W&B via ``TrainingLogger``.
5. Supports early stopping on validation MRR.
6. Checkpoints include ``model_type: "tirgn"`` in JSON metadata.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from src.training.models.components.global_history import (
    HistoryVocab,
    build_history_vocabulary,
)
from src.training.models.tirgn_jax import TiRGN, create_tirgn_model
from src.training.train_jax import (
    create_graph_snapshots,
    load_gdelt_data,
    negative_sampling,
)
from src.training.training_logger import TrainingLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TiRGNTrainingConfig:
    """Training hyperparameters for TiRGN.

    Extends the same fields as ``TrainingConfig`` with TiRGN-specific
    history parameters and early stopping patience.
    """

    # Shared with RE-GCN
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 1024
    num_negatives: int = 10
    grad_clip: float = 1.0
    checkpoint_interval: int = 10
    eval_interval: int = 5

    # TiRGN-specific
    history_rate: float = 0.3
    history_window: int = 50
    patience: int = 15
    logdir: str = "runs/tirgn"


# ---------------------------------------------------------------------------
# History vocabulary construction
# ---------------------------------------------------------------------------


def build_history_vocabulary_from_snapshots(
    snapshots_np: list[np.ndarray],
    num_entities: int,
    num_relations: int,
    window_size: int,
) -> HistoryVocab:
    """Build the global history vocabulary from raw numpy snapshots.

    Thin wrapper around ``build_history_vocabulary()`` that logs
    vocabulary statistics.  Called ONCE before training begins.

    Args:
        snapshots_np: List of (N, 3) numpy arrays from ``load_gdelt_data``.
        num_entities: Total entity count.
        num_relations: Total relation count (including inverse relations).
        window_size: Number of most recent snapshots to consider.

    Returns:
        Sparse history vocabulary dict.
    """
    vocab = build_history_vocabulary(
        snapshots_np, num_entities, num_relations, window_size
    )

    # Statistics
    num_pairs = len(vocab)
    if num_pairs > 0:
        avg_objects = sum(len(v) for v in vocab.values()) / num_pairs
    else:
        avg_objects = 0.0

    logger.info(
        "History vocabulary: %d (s,r) pairs, avg %.1f objects/pair",
        num_pairs,
        avg_objects,
    )
    return vocab


# ---------------------------------------------------------------------------
# Checkpoint I/O (TiRGN-specific metadata)
# ---------------------------------------------------------------------------


def save_tirgn_checkpoint(
    model: TiRGN,
    path: Path,
    epoch: int,
    metrics: dict[str, Any],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Save TiRGN model checkpoint with model_type discriminator.

    Follows the same .npz + .json format as RE-GCN ``save_checkpoint``
    but adds ``model_type: "tirgn"`` and TiRGN-specific config fields.

    Args:
        model: Trained TiRGN model.
        path: Destination .npz path.
        epoch: Current epoch number.
        metrics: Evaluation metrics dict.
        entity_to_id: Entity string-to-id mapping.
        relation_to_id: Relation string-to-id mapping.
        extra_config: Additional config fields to include.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    state, _ = nnx.split(model)
    state_dict = jax.tree.map(lambda x: np.array(x), state)

    config_section: dict[str, Any] = {
        "num_entities": model.num_entities,
        "num_relations": model.num_relations,
        "embedding_dim": model.embedding_dim,
        "num_layers": model.num_layers,
        "history_rate": model.history_rate,
        "history_window": model.history_window,
    }
    if extra_config:
        config_section.update(extra_config)

    checkpoint = {
        "model_type": "tirgn",
        "epoch": epoch,
        "metrics": metrics,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "config": config_section,
    }

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    np.savez(
        path,
        **{str(k): v for k, v in jax.tree_util.tree_leaves_with_path(state_dict)},
    )
    logger.info("Saved TiRGN checkpoint to %s", path)


def load_tirgn_checkpoint(
    path: Path,
) -> dict[str, Any]:
    """Load TiRGN checkpoint metadata (JSON sidecar).

    Args:
        path: Path to the .npz checkpoint (the .json sidecar is derived).

    Returns:
        Metadata dict from the JSON sidecar.

    Raises:
        FileNotFoundError: If the JSON sidecar does not exist.
    """
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------


def _get_vram_used_mb() -> float | None:
    """Return peak VRAM usage in MB, or None if unavailable."""
    try:
        devices = jax.devices()
        if devices:
            stats = devices[0].memory_stats()
            if stats and "peak_bytes_in_use" in stats:
                return stats["peak_bytes_in_use"] / (1024 * 1024)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_tirgn(
    data_path: Path,
    config: TiRGNTrainingConfig,
    model_dir: Path,
    max_events: int = 0,
    num_days: int = 30,
    embedding_dim: int = 200,
    num_layers: int = 2,
) -> dict[str, Any]:
    """Train TiRGN model on GDELT data with full observability.

    Args:
        data_path: Path to processed events parquet.
        config: Training hyperparameters.
        model_dir: Directory for model checkpoints.
        max_events: Maximum events to load (0 = unlimited).
        num_days: Number of recent days to include.
        embedding_dim: Entity/relation embedding dimension.
        num_layers: Number of R-GCN layers.

    Returns:
        Result dict with status, epochs_trained, best_mrr, total_time,
        early_stopped, model_type.
    """
    logger.info("=" * 70)
    logger.info("TiRGN Training Pipeline")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Data loading (reused from train_jax)
    # ------------------------------------------------------------------
    snapshots_np, entity_to_id, relation_to_id, train_triples, val_triples = (
        load_gdelt_data(data_path, max_events, num_days)
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    snapshots = create_graph_snapshots(snapshots_np, num_relations)

    logger.info("Graph Statistics:")
    logger.info("  Entities:    %d", num_entities)
    logger.info("  Relations:   %d", num_relations)
    logger.info("  Snapshots:   %d", len(snapshots))
    logger.info("  Train:       %d", len(train_triples))
    logger.info("  Val:         %d", len(val_triples))

    # ------------------------------------------------------------------
    # 2. Model creation
    # ------------------------------------------------------------------
    logger.info("Initializing TiRGN model...")
    model = create_tirgn_model(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        history_rate=config.history_rate,
        history_window=config.history_window,
        seed=0,
    )

    state, _ = nnx.split(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    logger.info("Total parameters: %d", total_params)

    # ------------------------------------------------------------------
    # 3. History vocabulary (built ONCE)
    # ------------------------------------------------------------------
    history_vocab = build_history_vocabulary_from_snapshots(
        snapshots_np, num_entities, num_relations * 2, config.history_window
    )

    # ------------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------------
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # ------------------------------------------------------------------
    # 5. Logger
    # ------------------------------------------------------------------
    run_name = f"tirgn_{datetime.now():%Y%m%d_%H%M}"
    training_logger = TrainingLogger(
        config.logdir, run_name=run_name, config=asdict(config)
    )

    # Log total params once
    training_logger.log_metrics({"system/total_params": float(total_params)}, step=0)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TRAINING STARTED")
    logger.info("=" * 70)

    model_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle training data
        perm = np.random.permutation(len(train_triples))
        train_shuffled = train_triples[perm]

        for batch_start in range(0, len(train_shuffled), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(train_shuffled))
            pos_batch = train_shuffled[batch_start:batch_end]

            # Generate negatives (protocol compat -- TiRGN ignores them but
            # we pass them through for interface consistency)
            neg_batch = negative_sampling(
                pos_batch, num_entities, config.num_negatives
            )

            pos_jax = jnp.array(pos_batch, dtype=jnp.int32)
            neg_jax = jnp.array(neg_batch, dtype=jnp.int32)

            def loss_fn(model: TiRGN) -> jax.Array:
                return model.compute_loss(
                    snapshots, pos_jax, neg_jax, history_vocab=history_vocab
                )

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

            epoch_loss += float(loss)
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_duration = time.time() - epoch_start

        # VRAM monitoring
        vram_mb = _get_vram_used_mb()

        # Log per-epoch metrics
        epoch_metrics: dict[str, float] = {
            "train/loss": avg_loss,
            "train/lr": config.learning_rate,
            "system/epoch_duration_s": epoch_duration,
        }
        if vram_mb is not None:
            epoch_metrics["system/vram_used_mb"] = vram_mb

        # Evaluate periodically
        if epoch % config.eval_interval == 0 or epoch == 1:
            val_sample = val_triples[: min(500, len(val_triples))]
            eval_metrics = _evaluate_tirgn(
                model, snapshots, val_sample, num_entities, history_vocab
            )
            mrr = eval_metrics["mrr"]

            epoch_metrics["eval/mrr"] = mrr
            epoch_metrics["eval/hits_at_1"] = eval_metrics["hits_at_1"]
            epoch_metrics["eval/hits_at_3"] = eval_metrics["hits_at_3"]
            epoch_metrics["eval/hits_at_10"] = eval_metrics["hits_at_10"]

            logger.info(
                "Epoch %3d/%d | Loss: %.4f | MRR: %.4f | H@10: %.4f | Time: %.1fs",
                epoch,
                config.epochs,
                avg_loss,
                mrr,
                eval_metrics["hits_at_10"],
                epoch_duration,
            )

            # Save best model
            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
                epochs_without_improvement = 0
                save_tirgn_checkpoint(
                    model,
                    model_dir / "tirgn_best.npz",
                    epoch,
                    eval_metrics,
                    entity_to_id,
                    relation_to_id,
                )
            else:
                epochs_without_improvement += config.eval_interval

            # Early stopping check
            if epochs_without_improvement >= config.patience:
                logger.info(
                    "Early stopping at epoch %d, best MRR %.4f at epoch %d",
                    epoch,
                    best_mrr,
                    best_epoch,
                )
                early_stopped = True
                training_logger.log_metrics(epoch_metrics, step=epoch)
                break
        else:
            logger.info(
                "Epoch %3d/%d | Loss: %.4f | Time: %.1fs",
                epoch,
                config.epochs,
                avg_loss,
                epoch_duration,
            )

        training_logger.log_metrics(epoch_metrics, step=epoch)

        # Periodic checkpoint
        if epoch % config.checkpoint_interval == 0:
            save_tirgn_checkpoint(
                model,
                model_dir / f"tirgn_epoch_{epoch}.npz",
                epoch,
                {"loss": avg_loss},
                entity_to_id,
                relation_to_id,
            )

    total_time = time.time() - start_time

    training_logger.close()

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time: %.1f minutes", total_time / 60)
    logger.info("Best MRR: %.4f at epoch %d", best_mrr, best_epoch)
    if early_stopped:
        logger.info("Training was halted by early stopping")

    # Save final checkpoint
    save_tirgn_checkpoint(
        model,
        model_dir / "tirgn_trained.npz",
        epoch,
        {"mrr": best_mrr},
        entity_to_id,
        relation_to_id,
    )

    return {
        "status": "complete",
        "epochs_trained": epoch,
        "best_mrr": float(best_mrr),
        "total_time": total_time,
        "early_stopped": early_stopped,
        "model_type": "tirgn",
    }


# ---------------------------------------------------------------------------
# Evaluation helper (wraps TiRGN's predict path into MRR computation)
# ---------------------------------------------------------------------------


def _evaluate_tirgn(
    model: TiRGN,
    snapshots: list,
    triples: np.ndarray,
    num_entities: int,
    history_vocab: HistoryVocab | None = None,
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute MRR/Hits@K for TiRGN using fused copy-generation scores.

    Unlike RE-GCN's ``compute_mrr`` which calls ``model.predict`` per-triple,
    TiRGN uses ``evolve_embeddings`` once then scores via the fused
    distribution.  This avoids re-evolving embeddings per triple.

    Args:
        model: Trained TiRGN model.
        snapshots: JAX graph snapshots.
        triples: (N, 3) evaluation triples.
        num_entities: Total entity count.
        history_vocab: Optional history vocabulary for copy-generation.
        batch_size: Evaluation batch size.

    Returns:
        Dict with mrr, hits_at_1, hits_at_3, hits_at_10.
    """
    # Evolve embeddings once for all evaluation triples
    entity_emb = model.evolve_embeddings(snapshots, training=False)

    ranks: list[int] = []

    for i in range(0, len(triples), batch_size):
        batch = triples[i : i + batch_size]
        batch_jax = jnp.array(batch, dtype=jnp.int32)

        # Time indices default to zeros (consistent with training)
        time_indices = jnp.zeros(batch_jax.shape[0], dtype=jnp.int32)

        # Build history mask if available
        history_mask = None
        if history_vocab is not None:
            from src.training.models.components.global_history import get_history_mask

            subjects_np = np.asarray(batch_jax[:, 0])
            relations_np = np.asarray(batch_jax[:, 1])
            history_mask = get_history_mask(
                history_vocab, subjects_np, relations_np, num_entities
            )

        # Get fused probability distribution (batch, num_entities)
        fused_probs = model._compute_fused_distribution(
            entity_emb, batch_jax, time_indices, history_mask, training=False
        )

        probs_np = np.asarray(fused_probs)

        for j, triple in enumerate(batch):
            _, _, o = triple
            true_score = probs_np[j, o]
            rank = int(np.sum(probs_np[j] >= true_score))
            ranks.append(rank)

    ranks_arr = np.array(ranks)

    return {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "hits_at_1": float(np.mean(ranks_arr <= 1)),
        "hits_at_3": float(np.mean(ranks_arr <= 3)),
        "hits_at_10": float(np.mean(ranks_arr <= 10)),
    }
