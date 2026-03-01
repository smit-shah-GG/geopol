#!/usr/bin/env python
"""Train TiRGN temporal knowledge graph model using JAX.

TiRGN extends RE-GCN with:
- Global history encoder (copy-generation mechanism)
- Time-ConvTransE decoder with learned temporal embeddings
- Relation GRU for relation embedding evolution

Training observability:
- TensorBoard logging (always)
- Weights & Biases logging (when WANDB_API_KEY is set)
- VRAM monitoring per epoch
- Early stopping on validation MRR

Usage:
    uv run python scripts/train_tirgn.py
    uv run python scripts/train_tirgn.py --epochs 50 --patience 10
    uv run python scripts/train_tirgn.py --history-rate 0.3 --history-window 50
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train TiRGN temporal knowledge graph model using JAX"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to processed GDELT events parquet (default: data/gdelt/processed/events.parquet)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory for model checkpoints (default: models/tkg)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size (default: 1024)",
    )
    parser.add_argument(
        "--history-rate",
        type=float,
        default=0.3,
        help="Copy-generation fusion rate: 0.0 = raw only, 1.0 = history only (default: 0.3)",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=50,
        help="Number of past snapshots for history vocabulary (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs/tirgn",
        help="TensorBoard / W&B log directory (default: runs/tirgn)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Maximum events to load, 0 = unlimited (default: 0)",
    )
    parser.add_argument(
        "--num-days",
        type=int,
        default=30,
        help="Number of recent days to include (default: 30)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=200,
        help="Entity/relation embedding dimension (default: 200)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of R-GCN layers (default: 2)",
    )

    args = parser.parse_args()

    # Print JAX device info
    logger.info("=" * 70)
    logger.info("TiRGN Training Pipeline")
    logger.info("=" * 70)
    logger.info("JAX version: %s", jax.__version__)
    logger.info("JAX devices: %s", jax.devices())
    logger.info("Default backend: %s", jax.default_backend())

    logger.info("\nConfiguration:")
    logger.info("  Epochs:          %d", args.epochs)
    logger.info("  Learning Rate:   %f", args.lr)
    logger.info("  Batch Size:      %d", args.batch_size)
    logger.info("  Embedding Dim:   %d", args.embedding_dim)
    logger.info("  Num Layers:      %d", args.num_layers)
    logger.info("  History Rate:    %.2f", args.history_rate)
    logger.info("  History Window:  %d", args.history_window)
    logger.info("  Patience:        %d", args.patience)
    logger.info("  Log Dir:         %s", args.logdir)
    logger.info(
        "  Max Events:      %s",
        str(args.max_events) if args.max_events > 0 else "unlimited",
    )
    logger.info("  Num Days:        %d", args.num_days)

    # Check data exists
    if not args.data_path.exists():
        logger.error("Data file not found: %s", args.data_path)
        logger.error("Run scripts/collect_training_data.py first")
        return 1

    from src.training.train_tirgn import TiRGNTrainingConfig, train_tirgn

    config = TiRGNTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        history_rate=args.history_rate,
        history_window=args.history_window,
        patience=args.patience,
        logdir=args.logdir,
    )

    try:
        result = train_tirgn(
            data_path=args.data_path,
            config=config,
            model_dir=args.model_dir,
            max_events=args.max_events,
            num_days=args.num_days,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
        )

        logger.info("\nTraining completed successfully!")
        logger.info("  Model type:      %s", result["model_type"])
        logger.info("  Epochs trained:  %d", result["epochs_trained"])
        logger.info("  Best MRR:        %.4f", result["best_mrr"])
        logger.info("  Total time:      %.1f minutes", result["total_time"] / 60)
        logger.info("  Early stopped:   %s", result["early_stopped"])
        logger.info("  Model saved to:  %s/tirgn_trained.npz", args.model_dir)

        return 0

    except Exception as e:
        logger.error("Training failed: %s", e)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
