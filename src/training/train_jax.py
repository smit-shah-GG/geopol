"""
JAX training utilities for RE-GCN.

Provides data loading, negative sampling, evaluation metrics,
and the training loop using Optax optimizers.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from src.training.models.regcn_jax import REGCN, GraphSnapshot, create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 1024
    num_negatives: int = 10
    margin: float = 1.0
    grad_clip: float = 1.0
    checkpoint_interval: int = 10
    eval_interval: int = 5


def load_gdelt_data(
    data_path: Path,
    max_events: int = 0,
    num_days: int = 30,
) -> Tuple[List[np.ndarray], Dict[str, int], Dict[str, int], np.ndarray, np.ndarray]:
    """
    Load GDELT events and prepare for TKG training.

    Args:
        data_path: Path to processed events parquet
        max_events: Maximum events (0 = unlimited)
        num_days: Number of recent days to include

    Returns:
        (snapshots, entity_to_id, relation_to_id, train_triples, val_triples)
    """
    import pandas as pd

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} events")

    # Filter to recent data
    max_date = df["timestamp"].max()
    cutoff_date = max_date - pd.Timedelta(days=num_days)
    df = df[df["timestamp"] >= cutoff_date].copy()
    logger.info(f"After filtering to last {num_days} days: {len(df):,} events")

    # Sample if needed
    if max_events > 0 and len(df) > max_events:
        logger.info(f"Sampling {max_events:,} events")
        df = df.sample(n=max_events, random_state=42).sort_values("timestamp")

    # Build entity mapping
    entities = set(df["entity1"].unique()) | set(df["entity2"].unique())
    entity_to_id = {e: i for i, e in enumerate(sorted(entities))}

    # Build relation mapping
    relations = df["relation"].unique()
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}

    logger.info(f"Entities: {len(entity_to_id)}, Relations: {len(relation_to_id)}")

    # Convert to integer day indices
    df["day"] = (df["timestamp"] - df["timestamp"].min()).dt.days

    # Create snapshots (one per day)
    snapshots = []
    for day in sorted(df["day"].unique()):
        day_df = df[df["day"] == day]
        triples = np.array([
            [entity_to_id[e1], relation_to_id[r], entity_to_id[e2]]
            for e1, r, e2 in zip(day_df["entity1"], day_df["relation"], day_df["entity2"])
        ])
        snapshots.append(triples)

    logger.info(f"Created {len(snapshots)} temporal snapshots")

    # Combine all triples for train/val split
    all_triples = np.vstack(snapshots)

    # Time-based split: 80% train, 20% val (last 20% of events)
    split_idx = int(len(all_triples) * 0.8)
    train_triples = all_triples[:split_idx]
    val_triples = all_triples[split_idx:]

    logger.info(f"Train: {len(train_triples):,}, Val: {len(val_triples):,}")

    return snapshots, entity_to_id, relation_to_id, train_triples, val_triples


def create_graph_snapshots(
    snapshots_np: List[np.ndarray],
    num_relations: int,
) -> List[GraphSnapshot]:
    """
    Convert numpy snapshots to JAX GraphSnapshot format.

    Adds inverse relations (r + num_relations for inverse).

    Args:
        snapshots_np: List of (N, 3) numpy arrays
        num_relations: Number of original relations

    Returns:
        List of GraphSnapshot objects
    """
    jax_snapshots = []

    for triples in snapshots_np:
        if len(triples) == 0:
            # Empty snapshot
            jax_snapshots.append(GraphSnapshot(
                edge_index=jnp.zeros((2, 0), dtype=jnp.int32),
                edge_type=jnp.zeros((0,), dtype=jnp.int32),
                num_edges=0,
            ))
            continue

        subjects = triples[:, 0]
        relations = triples[:, 1]
        objects = triples[:, 2]

        # Forward edges
        fwd_src = subjects
        fwd_dst = objects
        fwd_rel = relations

        # Inverse edges (object -> subject with inverse relation)
        inv_src = objects
        inv_dst = subjects
        inv_rel = relations + num_relations

        # Combine
        all_src = np.concatenate([fwd_src, inv_src])
        all_dst = np.concatenate([fwd_dst, inv_dst])
        all_rel = np.concatenate([fwd_rel, inv_rel])

        edge_index = jnp.array(np.stack([all_src, all_dst]), dtype=jnp.int32)
        edge_type = jnp.array(all_rel, dtype=jnp.int32)

        jax_snapshots.append(GraphSnapshot(
            edge_index=edge_index,
            edge_type=edge_type,
            num_edges=len(all_src),
        ))

    return jax_snapshots


def negative_sampling(
    pos_triples: np.ndarray,
    num_entities: int,
    num_negatives: int = 10,
    rng_key: Optional[jax.Array] = None,
) -> np.ndarray:
    """
    Generate negative samples by corrupting subject or object.

    Args:
        pos_triples: Positive triples (batch, 3)
        num_entities: Total number of entities
        num_negatives: Negatives per positive
        rng_key: JAX random key (uses numpy if None)

    Returns:
        Negative triples (batch * num_negatives, 3)
    """
    batch_size = len(pos_triples)
    negatives = []

    for i in range(batch_size):
        s, r, o = pos_triples[i]
        for _ in range(num_negatives):
            # Corrupt subject or object with 50% probability
            if np.random.random() < 0.5:
                # Corrupt subject
                new_s = np.random.randint(0, num_entities)
                negatives.append([new_s, r, o])
            else:
                # Corrupt object
                new_o = np.random.randint(0, num_entities)
                negatives.append([s, r, new_o])

    return np.array(negatives)


def compute_mrr(
    model: REGCN,
    snapshots: List[GraphSnapshot],
    triples: np.ndarray,
    num_entities: int,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Compute Mean Reciprocal Rank and Hits@K.

    Args:
        model: Trained model
        snapshots: Graph snapshots
        triples: Evaluation triples
        num_entities: Number of entities
        batch_size: Evaluation batch size

    Returns:
        Dict with mrr, hits_at_1, hits_at_3, hits_at_10
    """
    ranks = []

    for i in range(0, len(triples), batch_size):
        batch = triples[i:i+batch_size]

        for triple in batch:
            s, r, o = triple

            # Score all possible objects
            candidates = np.array([[s, r, e] for e in range(num_entities)])
            candidates_jax = jnp.array(candidates, dtype=jnp.int32)

            scores = model.predict(snapshots, candidates_jax)
            scores = np.array(scores)

            # Rank of true object
            true_score = scores[o]
            rank = np.sum(scores >= true_score)
            ranks.append(rank)

    ranks = np.array(ranks)

    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits_at_1": float(np.mean(ranks <= 1)),
        "hits_at_3": float(np.mean(ranks <= 3)),
        "hits_at_10": float(np.mean(ranks <= 10)),
    }


def save_checkpoint(
    model: REGCN,
    path: Path,
    optimizer_state: optax.OptState,
    epoch: int,
    metrics: Dict,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract state from model
    state, graphdef = nnx.split(model)

    # Convert to serializable format
    state_dict = jax.tree.map(lambda x: np.array(x), state)

    checkpoint = {
        "epoch": epoch,
        "metrics": metrics,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "config": {
            "num_entities": model.num_entities,
            "num_relations": model.num_relations,
            "embedding_dim": model.embedding_dim,
            "num_layers": model.num_layers,
        },
    }

    # Save metadata as JSON
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    # Save state using numpy
    np.savez(path, **{str(k): v for k, v in jax.tree_util.tree_leaves_with_path(state_dict)})

    logger.info(f"Saved checkpoint to {path}")


def train_regcn(
    data_path: Path,
    config: TrainingConfig,
    model_dir: Path,
    max_events: int = 0,
    num_days: int = 30,
    embedding_dim: int = 200,
    num_layers: int = 2,
) -> Dict:
    """
    Train RE-GCN model.

    Args:
        data_path: Path to GDELT parquet data
        config: Training configuration
        model_dir: Directory to save checkpoints
        max_events: Maximum events (0 = unlimited)
        num_days: Number of days to include
        embedding_dim: Embedding dimension
        num_layers: Number of R-GCN layers

    Returns:
        Training results dict
    """
    logger.info("=" * 70)
    logger.info("RE-GCN JAX Training Pipeline")
    logger.info("=" * 70)

    # Load data
    snapshots_np, entity_to_id, relation_to_id, train_triples, val_triples = load_gdelt_data(
        data_path, max_events, num_days
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    # Convert to JAX format
    snapshots = create_graph_snapshots(snapshots_np, num_relations)

    logger.info(f"\nGraph Statistics:")
    logger.info(f"  Entities:    {num_entities:,}")
    logger.info(f"  Relations:   {num_relations}")
    logger.info(f"  Snapshots:   {len(snapshots)}")
    logger.info(f"  Train:       {len(train_triples):,}")
    logger.info(f"  Val:         {len(val_triples):,}")

    # Create model
    logger.info("\nInitializing model...")
    model = create_model(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )

    # Count parameters
    state, _ = nnx.split(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    logger.info(f"Total parameters: {total_params:,}")

    # Setup optimizer using nnx.Optimizer with wrt (Flax 0.11.0+ API)
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING STARTED")
    logger.info("=" * 70)

    model_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = 0.0
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

            # Generate negatives
            neg_batch = negative_sampling(
                pos_batch, num_entities, config.num_negatives
            )

            # Convert to JAX arrays
            pos_jax = jnp.array(pos_batch, dtype=jnp.int32)
            neg_jax = jnp.array(neg_batch, dtype=jnp.int32)

            # Compute loss and gradients
            def loss_fn(model):
                return model.compute_loss(snapshots, pos_jax, neg_jax, config.margin)

            loss, grads = nnx.value_and_grad(loss_fn)(model)

            # Update parameters using nnx.Optimizer (Flax 0.11.0+ API)
            optimizer.update(model, grads)

            epoch_loss += float(loss)
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # Evaluate periodically
        if epoch % config.eval_interval == 0 or epoch == 1:
            # Sample validation for speed
            val_sample = val_triples[:min(500, len(val_triples))]
            metrics = compute_mrr(model, snapshots, val_sample, num_entities)
            mrr = metrics["mrr"]

            logger.info(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"MRR: {mrr:.4f} | "
                f"H@10: {metrics['hits_at_10']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if mrr > best_mrr:
                best_mrr = mrr
                save_checkpoint(
                    model,
                    model_dir / "regcn_best.npz",
                    optimizer.opt_state,
                    epoch,
                    metrics,
                    entity_to_id,
                    relation_to_id,
                )
        else:
            logger.info(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Periodic checkpoint
        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(
                model,
                model_dir / f"regcn_epoch_{epoch}.npz",
                optimizer.opt_state,
                epoch,
                {"loss": avg_loss},
                entity_to_id,
                relation_to_id,
            )

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best MRR: {best_mrr:.4f}")

    # Save final model
    save_checkpoint(
        model,
        model_dir / "regcn_trained.npz",
        optimizer.opt_state,
        config.epochs,
        {"mrr": best_mrr},
        entity_to_id,
        relation_to_id,
    )

    return {
        "status": "complete",
        "epochs": config.epochs,
        "best_mrr": best_mrr,
        "total_time": total_time,
    }
