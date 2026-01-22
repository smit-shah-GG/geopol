#!/usr/bin/env python
"""
Train RE-GCN temporal knowledge graph model using JAX + jraph.

Uses jraph for efficient graph neural network operations:
- segment_sum for XLA-optimized message aggregation
- GraphsTuple for efficient batching
- Fully vectorized (no Python loops over relations)

Usage:
    uv run python scripts/train_tkg_jraph.py
    uv run python scripts/train_tkg_jraph.py --max-events 200000 --epochs 50
    uv run python scripts/train_tkg_jraph.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Default paths
DATA_PATH = Path("data/gdelt/processed/events.parquet")
MODEL_DIR = Path("models/tkg")
LOG_DIR = Path("logs/training")


def main():
    parser = argparse.ArgumentParser(
        description="Train RE-GCN temporal knowledge graph model using JAX + jraph"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Training batch size (default: 1024)"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=200,
        help="Embedding dimension (default: 200)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2,
        help="Number of R-GCN layers (default: 2)"
    )
    parser.add_argument(
        "--num-negatives", type=int, default=10,
        help="Negative samples per positive (default: 10)"
    )
    parser.add_argument(
        "--max-events", type=int, default=0,
        help="Maximum events (0 = unlimited)"
    )
    parser.add_argument(
        "--num-days", type=int, default=30,
        help="Number of recent days to include (default: 30)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Initialize model without training"
    )

    args = parser.parse_args()

    # Print JAX device info
    logger.info("=" * 70)
    logger.info("RE-GCN Jraph Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Default backend: {jax.default_backend()}")

    # Check for GPU
    try:
        gpu_devices = jax.devices("gpu")
        logger.info(f"GPU devices: {gpu_devices}")
    except RuntimeError:
        logger.warning("No GPU detected - training will use CPU")

    logger.info("\nConfiguration:")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Batch Size:    {args.batch_size}")
    logger.info(f"  Embedding Dim: {args.embedding_dim}")
    logger.info(f"  Num Layers:    {args.num_layers}")
    logger.info(f"  Max Events:    {args.max_events if args.max_events > 0 else 'unlimited'}")
    logger.info(f"  Num Days:      {args.num_days}")
    logger.info(f"  Dry Run:       {args.dry_run}")

    # Check data exists
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.error("Run scripts/collect_training_data.py first")
        return 1

    if args.dry_run:
        logger.info("\nDry run: testing imports and model creation...")

        from src.training.models.regcn_jraph import create_model
        from src.training.train_jraph import load_gdelt_data, create_temporal_graphs

        # Load a small sample
        snapshots_np, entity_to_id, relation_to_id, _, _ = load_gdelt_data(
            DATA_PATH, max_events=1000, num_days=7
        )

        num_entities = len(entity_to_id)
        num_relations = len(relation_to_id)

        # Create model
        model = create_model(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
        )

        logger.info(f"\nModel created successfully!")
        logger.info(f"  Entities:   {num_entities}")
        logger.info(f"  Relations:  {num_relations}")
        logger.info(f"  Snapshots:  {len(snapshots_np)}")

        # Convert to jraph format
        graphs = create_temporal_graphs(snapshots_np, num_entities, num_relations)
        logger.info(f"  Jraph graphs: {len(graphs)}")

        # Test forward pass
        import jax.numpy as jnp
        test_triple = jnp.array([[0, 0, 1]], dtype=jnp.int32)
        scores = model.predict(graphs, test_triple)
        logger.info(f"  Test score: {float(scores[0]):.4f}")

        # Test loss computation
        pos_triples = jnp.array([[0, 0, 1], [1, 0, 2]], dtype=jnp.int32)
        neg_triples = jnp.array([[0, 0, 2], [1, 0, 0]], dtype=jnp.int32)
        loss = model.compute_loss(graphs, pos_triples, neg_triples, margin=1.0)
        logger.info(f"  Test loss: {float(loss):.4f}")

        logger.info("\nDry run complete - all systems operational")
        return 0

    # Full training
    from src.training.train_jraph import train_regcn_jraph, TrainingConfig

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        checkpoint_interval=args.checkpoint_interval,
    )

    try:
        result = train_regcn_jraph(
            data_path=DATA_PATH,
            config=config,
            model_dir=MODEL_DIR,
            max_events=args.max_events,
            num_days=args.num_days,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
        )

        logger.info("\nTraining completed successfully!")
        logger.info(f"  Best MRR: {result['best_mrr']:.4f}")
        logger.info(f"  Total time: {result['total_time']/60:.1f} minutes")
        logger.info(f"  Model saved to: {MODEL_DIR}/regcn_jraph_final.npz")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
