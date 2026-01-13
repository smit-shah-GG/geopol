#!/usr/bin/env python
"""
Train RE-GCN temporal knowledge graph model on GDELT data.

This script:
1. Loads processed GDELT events from parquet
2. Creates temporal graph snapshots
3. Trains RE-GCN with progress monitoring
4. Saves checkpoints and final model

Usage:
    uv run python scripts/train_tkg.py             # Full training
    uv run python scripts/train_tkg.py --dry-run   # Initialize only
    uv run python scripts/train_tkg.py --epochs 50 # Reduced epochs
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.training.models.regcn_cpu import REGCN
from src.training.progress_monitor import ProgressMonitor
from src.training.train_utils import (
    compute_hits_at_k,
    compute_mrr,
    create_graph_snapshots,
    negative_sampling,
    save_checkpoint,
)

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


def load_training_data(data_path: Path) -> tuple:
    """
    Load GDELT events and prepare for TKG training.

    Args:
        data_path: Path to processed events parquet file

    Returns:
        Tuple of (snapshots, entity_to_id, relation_to_id, train_triples, val_triples)
    """
    import pandas as pd

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    logger.info(f"Loaded {len(df):,} events")
    logger.info(f"Raw date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Filter to last 30 days of data only (data collection target range)
    # The parquet may contain artifacts from earlier dates
    max_date = df["timestamp"].max()
    cutoff_date = max_date - pd.Timedelta(days=30)
    df = df[df["timestamp"] >= cutoff_date].copy()

    logger.info(f"After filtering to last 30 days: {len(df):,} events")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Convert timestamp to integer days for snapshotting
    # The create_graph_snapshots function expects integer timestamps
    min_date = df["timestamp"].min()
    df["timestamp"] = (df["timestamp"] - min_date).dt.days

    # Create graph snapshots with entity/relation mappings
    num_snapshots = 30
    snapshots, entity_to_id, relation_to_id = create_graph_snapshots(
        df,
        num_snapshots=num_snapshots,
    )

    logger.info(f"Created {len(snapshots)} temporal snapshots")
    logger.info(f"Unique entities: {len(entity_to_id):,}")
    logger.info(f"Unique relations: {len(relation_to_id)}")

    # Split by time: 80% train, 20% validation (last snapshots)
    split_idx = int(len(snapshots) * 0.8)
    train_snapshots = snapshots[:split_idx]
    val_snapshots = snapshots[split_idx:]

    # Aggregate triples for train/val
    train_triples = np.concatenate([s for s in train_snapshots if len(s) > 0])
    val_triples = np.concatenate([s for s in val_snapshots if len(s) > 0])

    logger.info(f"Training triples: {len(train_triples):,}")
    logger.info(f"Validation triples: {len(val_triples):,}")

    return snapshots, entity_to_id, relation_to_id, train_triples, val_triples


def snapshots_to_tensors(
    snapshots: list,
    device: torch.device,
) -> list:
    """
    Convert numpy snapshot arrays to PyTorch tensors.

    Args:
        snapshots: List of (num_triples, 3) numpy arrays
        device: Target torch device

    Returns:
        List of (edge_index, edge_type) tensor tuples
    """
    tensor_snapshots = []
    for snapshot in snapshots:
        if len(snapshot) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty(0, dtype=torch.long, device=device)
        else:
            snapshot_t = torch.tensor(snapshot, dtype=torch.long, device=device)
            edge_index = snapshot_t[:, [0, 2]].t().contiguous()
            edge_type = snapshot_t[:, 1]
        tensor_snapshots.append((edge_index, edge_type))
    return tensor_snapshots


def evaluate_model(
    model: REGCN,
    snapshots: list,
    val_triples: np.ndarray,
    num_entities: int,
    device: torch.device,
    num_samples: int = 500,
) -> dict:
    """
    Evaluate model on validation set.

    Args:
        model: Trained RE-GCN model
        snapshots: Tensor snapshots for embedding evolution
        val_triples: Validation triples (N, 3)
        num_entities: Total number of entities
        device: Torch device
        num_samples: Number of validation samples to evaluate

    Returns:
        Dictionary with MRR and Hits@K metrics
    """
    model.eval()

    # Sample validation triples for efficiency
    if len(val_triples) > num_samples:
        indices = np.random.choice(len(val_triples), num_samples, replace=False)
        sampled_triples = val_triples[indices]
    else:
        sampled_triples = val_triples

    with torch.no_grad():
        # Evolve embeddings
        entity_emb = model.evolve_embeddings(snapshots)

        # Compute scores for all validation triples
        predictions = []
        ground_truth = []

        for s, r, o in sampled_triples:
            # Get subject and relation embeddings
            subject_emb = entity_emb[s].unsqueeze(0)
            relation_emb = model.relation_embeddings(
                torch.tensor([r], device=device)
            )

            # Score all entities as potential objects
            scores = model.decoder(subject_emb, relation_emb, entity_emb)
            scores = scores.squeeze(0).cpu().numpy()

            predictions.append(scores)
            ground_truth.append(o)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Compute metrics
    mrr = compute_mrr(predictions, ground_truth)
    hits_at_1 = compute_hits_at_k(predictions, ground_truth, k=1)
    hits_at_3 = compute_hits_at_k(predictions, ground_truth, k=3)
    hits_at_10 = compute_hits_at_k(predictions, ground_truth, k=10)

    return {
        "mrr": mrr,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "hits_at_10": hits_at_10,
    }


def train_regcn(
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 1024,
    embedding_dim: int = 200,
    num_layers: int = 2,
    dropout: float = 0.2,
    num_negatives: int = 10,
    margin: float = 1.0,
    checkpoint_interval: int = 10,
    dry_run: bool = False,
) -> dict:
    """
    Train RE-GCN model on GDELT data.

    Args:
        epochs: Number of training epochs
        learning_rate: Adam optimizer learning rate
        batch_size: Training batch size
        embedding_dim: Entity/relation embedding dimension
        num_layers: Number of R-GCN layers
        dropout: Dropout rate
        num_negatives: Negative samples per positive
        margin: Margin for ranking loss
        checkpoint_interval: Save checkpoint every N epochs
        dry_run: If True, initialize model but skip training

    Returns:
        Dictionary with training summary
    """
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    snapshots, entity_to_id, relation_to_id, train_triples, val_triples = load_training_data(
        DATA_PATH
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    num_triples = len(train_triples)

    # Convert snapshots to tensors
    tensor_snapshots = snapshots_to_tensors(snapshots, device)

    # Initialize model
    logger.info("\nInitializing RE-GCN model...")
    model = REGCN(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if dry_run:
        logger.info("\nDry run complete - model initialized successfully")
        return {
            "status": "dry_run",
            "num_entities": num_entities,
            "num_relations": num_relations,
            "total_params": total_params,
        }

    # Initialize progress monitor
    monitor = ProgressMonitor(
        total_epochs=epochs,
        log_interval=1,
        save_interval=checkpoint_interval,
        output_dir=LOG_DIR,
    )
    monitor.set_graph_stats(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        num_snapshots=len(tensor_snapshots),
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    monitor.start_training()
    model.train()

    best_mrr = 0.0
    final_loss = 0.0
    final_mrr = 0.0

    for epoch in range(1, epochs + 1):
        monitor.start_epoch(epoch)

        # Shuffle training data
        perm = np.random.permutation(num_triples)
        triples_shuffled = train_triples[perm]

        epoch_loss = 0.0
        num_batches = 0

        for batch_start in range(0, num_triples, batch_size):
            batch_end = min(batch_start + batch_size, num_triples)
            batch_triples = triples_shuffled[batch_start:batch_end]

            # Generate negative samples
            negatives = negative_sampling(
                batch_triples,
                num_entities=num_entities,
                num_negatives=num_negatives,
                strategy="uniform",
            )

            # Convert to tensors
            pos_tensor = torch.tensor(batch_triples, dtype=torch.long, device=device)
            neg_tensor = torch.tensor(negatives, dtype=torch.long, device=device)

            # Forward pass and loss
            optimizer.zero_grad()
            loss = model.compute_loss(
                tensor_snapshots,
                pos_tensor,
                neg_tensor,
                margin=margin,
            )
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        final_loss = avg_loss

        # Evaluate on validation set every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_model(
                model, tensor_snapshots, val_triples, num_entities, device
            )
            mrr = metrics["mrr"]
            final_mrr = mrr

            # Save best model
            if mrr > best_mrr:
                best_mrr = mrr
                save_checkpoint(
                    model,
                    MODEL_DIR / "regcn_best.pt",
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    entity_to_id=entity_to_id,
                    relation_to_id=relation_to_id,
                )
        else:
            mrr = final_mrr  # Use last computed MRR for logging
            metrics = {}

        monitor.end_epoch(
            epoch=epoch,
            loss=avg_loss,
            mrr=mrr,
            hits_at_1=metrics.get("hits_at_1", 0),
            hits_at_3=metrics.get("hits_at_3", 0),
            hits_at_10=metrics.get("hits_at_10", 0),
        )

        # Save checkpoint at intervals
        if epoch % checkpoint_interval == 0:
            save_checkpoint(
                model,
                MODEL_DIR / f"regcn_epoch_{epoch}.pt",
                optimizer=optimizer,
                epoch=epoch,
                metrics={"mrr": mrr, "loss": avg_loss},
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )

    # Final evaluation
    logger.info("\nRunning final evaluation...")
    final_metrics = evaluate_model(
        model, tensor_snapshots, val_triples, num_entities, device, num_samples=1000
    )
    final_mrr = final_metrics["mrr"]

    # Save final trained model
    save_checkpoint(
        model,
        MODEL_DIR / "regcn_trained.pt",
        optimizer=optimizer,
        epoch=epochs,
        metrics=final_metrics,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    # Generate summary
    summary = monitor.end_training(final_mrr=final_mrr, final_loss=final_loss)

    logger.info(f"\nFinal Metrics:")
    logger.info(f"  MRR:       {final_metrics['mrr']:.4f}")
    logger.info(f"  Hits@1:    {final_metrics['hits_at_1']:.4f}")
    logger.info(f"  Hits@3:    {final_metrics['hits_at_3']:.4f}")
    logger.info(f"  Hits@10:   {final_metrics['hits_at_10']:.4f}")
    logger.info(f"\nModel saved to: {MODEL_DIR / 'regcn_trained.pt'}")

    return {
        "status": "complete",
        "epochs": epochs,
        "final_loss": final_loss,
        "final_mrr": final_mrr,
        "best_mrr": best_mrr,
        "metrics": final_metrics,
        "model_path": str(MODEL_DIR / "regcn_trained.pt"),
        **summary,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RE-GCN temporal knowledge graph model on GDELT data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=200,
        help="Entity/relation embedding dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of R-GCN layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=10,
        help="Negative samples per positive",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize model without training (for verification)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("RE-GCN TKG Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"\nConfiguration:")
    logger.info(f"  Epochs:          {args.epochs}")
    logger.info(f"  Learning Rate:   {args.lr}")
    logger.info(f"  Batch Size:      {args.batch_size}")
    logger.info(f"  Embedding Dim:   {args.embedding_dim}")
    logger.info(f"  Num Layers:      {args.num_layers}")
    logger.info(f"  Dropout:         {args.dropout}")
    logger.info(f"  Dry Run:         {args.dry_run}")

    try:
        result = train_regcn(
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_negatives=args.num_negatives,
            checkpoint_interval=args.checkpoint_interval,
            dry_run=args.dry_run,
        )

        logger.info("\n" + "=" * 70)
        logger.info("Training Pipeline Complete")
        logger.info("=" * 70)

        if result["status"] == "dry_run":
            logger.info("Dry run successful - model initialization verified")
        else:
            logger.info(f"Model saved to: {result.get('model_path')}")
            logger.info(f"Final MRR: {result.get('final_mrr', 0):.4f}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
