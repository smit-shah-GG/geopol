"""
JAX training utilities for RE-GCN using jraph.

Key improvements over manual implementation:
- jraph's segment operations are XLA-optimized
- GraphsTuple handles batching efficiently
- Vectorized message passing (no Python loops over relations)
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

from src.training.models.regcn_jraph import (
    REGCNJraph,
    TemporalGraph,
    create_graph,
    create_model,
)

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


def create_temporal_graphs(
    snapshots_np: List[np.ndarray],
    num_entities: int,
    num_relations: int,
) -> List[TemporalGraph]:
    """
    Convert numpy snapshots to jraph TemporalGraph format.

    Adds inverse relations (r + num_relations for inverse).

    Args:
        snapshots_np: List of (N, 3) numpy arrays [subject, relation, object]
        num_entities: Total number of entities
        num_relations: Number of original relations

    Returns:
        List of TemporalGraph objects
    """
    jraph_graphs = []

    for triples in snapshots_np:
        if len(triples) == 0:
            # Empty snapshot - create minimal graph
            graph = create_graph(
                senders=jnp.zeros((0,), dtype=jnp.int32),
                receivers=jnp.zeros((0,), dtype=jnp.int32),
                relation_types=jnp.zeros((0,), dtype=jnp.int32),
                num_nodes=num_entities,
            )
            jraph_graphs.append(graph)
            continue

        subjects = triples[:, 0]
        relations = triples[:, 1]
        objects = triples[:, 2]

        # Forward edges: subject -> object with relation r
        fwd_src = subjects
        fwd_dst = objects
        fwd_rel = relations

        # Inverse edges: object -> subject with relation (r + num_relations)
        inv_src = objects
        inv_dst = subjects
        inv_rel = relations + num_relations

        # Combine forward and inverse
        all_src = np.concatenate([fwd_src, inv_src])
        all_dst = np.concatenate([fwd_dst, inv_dst])
        all_rel = np.concatenate([fwd_rel, inv_rel])

        graph = create_graph(
            senders=jnp.array(all_src, dtype=jnp.int32),
            receivers=jnp.array(all_dst, dtype=jnp.int32),
            relation_types=jnp.array(all_rel, dtype=jnp.int32),
            num_nodes=num_entities,
        )
        jraph_graphs.append(graph)

    return jraph_graphs


def negative_sampling(
    pos_triples: np.ndarray,
    num_entities: int,
    num_negatives: int = 10,
) -> np.ndarray:
    """
    Generate negative samples by corrupting subject or object.

    Args:
        pos_triples: Positive triples (batch, 3)
        num_entities: Total number of entities
        num_negatives: Negatives per positive

    Returns:
        Negative triples (batch * num_negatives, 3)
    """
    batch_size = len(pos_triples)
    total_negs = batch_size * num_negatives

    # Vectorized negative sampling
    # Randomly decide whether to corrupt subject (0) or object (1)
    corrupt_obj = np.random.randint(0, 2, size=total_negs)

    # Generate random entities
    random_entities = np.random.randint(0, num_entities, size=total_negs)

    # Repeat positive triples
    repeated = np.repeat(pos_triples, num_negatives, axis=0)

    # Apply corruption
    negatives = repeated.copy()
    negatives[corrupt_obj == 0, 0] = random_entities[corrupt_obj == 0]  # Corrupt subject
    negatives[corrupt_obj == 1, 2] = random_entities[corrupt_obj == 1]  # Corrupt object

    return negatives


def compute_mrr_batched(
    model: REGCNJraph,
    graphs: List[TemporalGraph],
    triples: np.ndarray,
    num_entities: int,
    eval_batch_size: int = 64,
) -> Dict[str, float]:
    """
    Compute Mean Reciprocal Rank and Hits@K efficiently.

    Uses batched evaluation to avoid Python loops.

    Args:
        model: Trained model
        graphs: Temporal graph snapshots
        triples: Evaluation triples (num_eval, 3)
        num_entities: Number of entities
        eval_batch_size: Batch size for evaluation

    Returns:
        Dict with mrr, hits_at_1, hits_at_3, hits_at_10
    """
    ranks = []

    # Get entity embeddings once (expensive operation)
    entity_emb = model.evolve_embeddings(graphs, training=False)

    for i in range(0, len(triples), eval_batch_size):
        batch = triples[i:i+eval_batch_size]
        batch_ranks = []

        for triple in batch:
            s, r, o = triple

            # Create candidate triples for all possible objects
            candidates = np.zeros((num_entities, 3), dtype=np.int32)
            candidates[:, 0] = s
            candidates[:, 1] = r
            candidates[:, 2] = np.arange(num_entities)

            candidates_jax = jnp.array(candidates, dtype=jnp.int32)

            # Score all candidates
            scores = model.compute_scores(entity_emb, candidates_jax)
            scores = np.array(scores)

            # Compute rank (1-indexed)
            true_score = scores[o]
            rank = int(np.sum(scores > true_score)) + 1
            batch_ranks.append(rank)

        ranks.extend(batch_ranks)

    ranks = np.array(ranks, dtype=np.float32)

    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits_at_1": float(np.mean(ranks <= 1)),
        "hits_at_3": float(np.mean(ranks <= 3)),
        "hits_at_10": float(np.mean(ranks <= 10)),
    }


def save_checkpoint(
    model: REGCNJraph,
    path: Path,
    epoch: int,
    metrics: Dict,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract state from model
    graphdef, state = nnx.split(model)

    # Convert state to flat dict for serialization
    flat_state = {}

    def flatten_state(prefix: str, obj):
        if hasattr(obj, 'value'):
            flat_state[prefix] = np.array(obj.value)
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                flatten_state(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                flatten_state(f"{prefix}[{i}]", item)

    flatten_state("", state)

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
    np.savez(path, **flat_state)

    logger.info(f"Saved checkpoint to {path}")


def train_regcn_jraph(
    data_path: Path,
    config: TrainingConfig,
    model_dir: Path,
    max_events: int = 0,
    num_days: int = 30,
    embedding_dim: int = 200,
    num_layers: int = 2,
) -> Dict:
    """
    Train RE-GCN model using jraph.

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
    logger.info("RE-GCN Jraph Training Pipeline")
    logger.info("=" * 70)

    # Load data
    snapshots_np, entity_to_id, relation_to_id, train_triples, val_triples = load_gdelt_data(
        data_path, max_events, num_days
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    # Convert to jraph format
    graphs = create_temporal_graphs(snapshots_np, num_entities, num_relations)

    logger.info(f"\nGraph Statistics:")
    logger.info(f"  Entities:    {num_entities:,}")
    logger.info(f"  Relations:   {num_relations} (x2 with inverses = {num_relations * 2})")
    logger.info(f"  Snapshots:   {len(graphs)}")
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
    _, state = nnx.split(model)
    leaves = jax.tree_util.tree_leaves(state)
    total_params = sum(
        np.prod(x.shape) for x in leaves
        if hasattr(x, 'shape') and len(x.shape) > 0
    )
    logger.info(f"Total parameters: {total_params:,}")

    # Setup optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # JIT compile the loss function
    @jax.jit
    def loss_fn(model_state, graphs, pos_jax, neg_jax, margin, rng_key):
        # Reconstruct model from state for JIT
        return model.compute_loss(graphs, pos_jax, neg_jax, margin, rng_key)

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING STARTED")
    logger.info("=" * 70)

    model_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = 0.0
    start_time = time.time()
    rng_key = jax.random.PRNGKey(42)

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

            # Split RNG key
            rng_key, subkey = jax.random.split(rng_key)

            # Compute loss and gradients
            def train_loss(model):
                return model.compute_loss(graphs, pos_jax, neg_jax, config.margin, subkey)

            loss, grads = nnx.value_and_grad(train_loss)(model)

            # Update parameters
            optimizer.update(model, grads)

            epoch_loss += float(loss)
            num_batches += 1

            # Progress indicator every 10 batches
            if num_batches % 10 == 0:
                logger.info(f"  Batch {num_batches}: loss = {float(loss):.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        # Evaluate periodically
        if epoch % config.eval_interval == 0 or epoch == 1:
            # Sample validation for speed
            val_sample = val_triples[:min(200, len(val_triples))]
            metrics = compute_mrr_batched(model, graphs, val_sample, num_entities)
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
                    model_dir / "regcn_jraph_best.npz",
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
                model_dir / f"regcn_jraph_epoch_{epoch}.npz",
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
        model_dir / "regcn_jraph_final.npz",
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
