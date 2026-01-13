"""
Progress monitoring utilities for RE-GCN training.

Provides real-time visibility into training progress:
- Epoch progress with loss/metric curves
- Graph statistics (entities, relations, density)
- Pattern learning indicators (top predicted relations)
- ETA estimation based on moving averages
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics collected for a single training epoch."""

    epoch: int
    loss: float
    mrr: float = 0.0
    hits_at_1: float = 0.0
    hits_at_3: float = 0.0
    hits_at_10: float = 0.0
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GraphStats:
    """Statistics about the training graph."""

    num_entities: int
    num_relations: int
    num_triples: int
    num_snapshots: int
    density: float  # edges / (entities^2 * relations)
    avg_degree: float  # average node degree


class ProgressMonitor:
    """
    Real-time training progress monitor with logging and metrics export.

    Tracks:
    - Training loss curve with smoothed moving average
    - Validation metrics (MRR, Hits@K)
    - Graph statistics per snapshot
    - Top predicted relations (learning indicators)
    - ETA estimation using exponential moving average

    Attributes:
        metrics_history: List of EpochMetrics for all completed epochs
        graph_stats: GraphStats for current training graph
        smoothing_window: Window size for loss smoothing
        log_interval: Epochs between console logs
        save_interval: Epochs between metrics file saves
    """

    def __init__(
        self,
        total_epochs: int,
        log_interval: int = 1,
        save_interval: int = 10,
        smoothing_window: int = 5,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize progress monitor.

        Args:
            total_epochs: Total number of training epochs
            log_interval: Log progress every N epochs
            save_interval: Save metrics to file every N epochs
            smoothing_window: Window for moving average smoothing
            output_dir: Directory for saving metrics (default: logs/training/)
        """
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.smoothing_window = smoothing_window
        self.output_dir = output_dir or Path("logs/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[EpochMetrics] = []
        self.graph_stats: Optional[GraphStats] = None
        self.loss_window: deque = deque(maxlen=smoothing_window)
        self.duration_window: deque = deque(maxlen=smoothing_window)

        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.best_mrr: float = 0.0
        self.best_epoch: int = 0

        # Top relation predictions for pattern learning indicator
        self.top_relations: List[Tuple[int, float]] = []

    def start_training(self) -> None:
        """Mark training start."""
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("RE-GCN TRAINING STARTED")
        logger.info("=" * 70)

    def set_graph_stats(
        self,
        num_entities: int,
        num_relations: int,
        num_triples: int,
        num_snapshots: int,
    ) -> None:
        """
        Set graph statistics for progress display.

        Args:
            num_entities: Total unique entities
            num_relations: Total unique relation types
            num_triples: Total training triples
            num_snapshots: Number of temporal snapshots
        """
        # Calculate density: ratio of actual edges to possible edges
        max_possible = num_entities * num_entities * num_relations
        density = num_triples / max_possible if max_possible > 0 else 0.0

        # Average degree: average edges per node
        avg_degree = (2 * num_triples) / num_entities if num_entities > 0 else 0.0

        self.graph_stats = GraphStats(
            num_entities=num_entities,
            num_relations=num_relations,
            num_triples=num_triples,
            num_snapshots=num_snapshots,
            density=density,
            avg_degree=avg_degree,
        )

        logger.info("\nGraph Statistics:")
        logger.info(f"  Entities:    {num_entities:,}")
        logger.info(f"  Relations:   {num_relations}")
        logger.info(f"  Triples:     {num_triples:,}")
        logger.info(f"  Snapshots:   {num_snapshots}")
        logger.info(f"  Density:     {density:.6f}")
        logger.info(f"  Avg Degree:  {avg_degree:.1f}")
        logger.info("")

    def start_epoch(self, epoch: int) -> None:
        """Mark epoch start for timing."""
        self.epoch_start_time = time.time()

    def end_epoch(
        self,
        epoch: int,
        loss: float,
        mrr: float = 0.0,
        hits_at_1: float = 0.0,
        hits_at_3: float = 0.0,
        hits_at_10: float = 0.0,
    ) -> None:
        """
        Record epoch completion and log progress.

        Args:
            epoch: Current epoch number (1-indexed)
            loss: Training loss for this epoch
            mrr: Validation MRR (Mean Reciprocal Rank)
            hits_at_1: Validation Hits@1
            hits_at_3: Validation Hits@3
            hits_at_10: Validation Hits@10
        """
        duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0

        # Update moving average windows
        self.loss_window.append(loss)
        self.duration_window.append(duration)

        # Record metrics
        metrics = EpochMetrics(
            epoch=epoch,
            loss=loss,
            mrr=mrr,
            hits_at_1=hits_at_1,
            hits_at_3=hits_at_3,
            hits_at_10=hits_at_10,
            duration_seconds=duration,
        )
        self.metrics_history.append(metrics)

        # Track best MRR
        if mrr > self.best_mrr:
            self.best_mrr = mrr
            self.best_epoch = epoch

        # Log progress
        if epoch % self.log_interval == 0 or epoch == 1:
            self._log_progress(epoch, loss, mrr, duration)

        # Save metrics periodically
        if epoch % self.save_interval == 0:
            self._save_metrics()

    def _log_progress(
        self,
        epoch: int,
        loss: float,
        mrr: float,
        duration: float,
    ) -> None:
        """Log formatted progress to console."""
        # Progress bar
        progress = epoch / self.total_epochs
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)

        # Smoothed loss
        smoothed_loss = np.mean(self.loss_window) if self.loss_window else loss

        # ETA calculation
        avg_duration = np.mean(self.duration_window) if self.duration_window else duration
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = remaining_epochs * avg_duration
        eta = timedelta(seconds=int(eta_seconds))

        # Format output
        logger.info(
            f"Epoch {epoch:3d}/{self.total_epochs} [{bar}] "
            f"Loss: {loss:.4f} (avg: {smoothed_loss:.4f}) | "
            f"MRR: {mrr:.4f} | ETA: {eta}"
        )

    def update_top_relations(
        self,
        relation_scores: List[Tuple[int, float]],
        relation_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Update top predicted relations for pattern learning indicator.

        Args:
            relation_scores: List of (relation_id, score) tuples
            relation_names: Optional mapping of relation IDs to names
        """
        self.top_relations = sorted(relation_scores, key=lambda x: x[1], reverse=True)[:10]

        if relation_names:
            logger.info("\nTop Predicted Relations:")
            for rel_id, score in self.top_relations[:5]:
                name = relation_names.get(rel_id, f"rel_{rel_id}")
                logger.info(f"  {name}: {score:.4f}")

    def end_training(self, final_mrr: float, final_loss: float) -> Dict:
        """
        Mark training completion and return summary.

        Args:
            final_mrr: Final validation MRR
            final_loss: Final training loss

        Returns:
            Dictionary with training summary
        """
        total_time = time.time() - self.start_time if self.start_time else 0.0

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total Time:    {timedelta(seconds=int(total_time))}")
        logger.info(f"Final Loss:    {final_loss:.4f}")
        logger.info(f"Final MRR:     {final_mrr:.4f}")
        logger.info(f"Best MRR:      {self.best_mrr:.4f} (epoch {self.best_epoch})")

        if self.graph_stats:
            logger.info(f"Entities:      {self.graph_stats.num_entities:,}")
            logger.info(f"Relations:     {self.graph_stats.num_relations}")

        # Save final metrics
        self._save_metrics()

        summary = {
            "total_epochs": len(self.metrics_history),
            "total_time_seconds": total_time,
            "final_loss": final_loss,
            "final_mrr": final_mrr,
            "best_mrr": self.best_mrr,
            "best_epoch": self.best_epoch,
            "graph_stats": {
                "num_entities": self.graph_stats.num_entities,
                "num_relations": self.graph_stats.num_relations,
                "num_triples": self.graph_stats.num_triples,
            }
            if self.graph_stats
            else None,
        }

        return summary

    def _save_metrics(self) -> None:
        """Save metrics history to JSON file."""
        metrics_file = self.output_dir / "tkg_metrics.json"

        data = {
            "training_started": self.metrics_history[0].timestamp
            if self.metrics_history
            else None,
            "total_epochs": self.total_epochs,
            "completed_epochs": len(self.metrics_history),
            "best_mrr": self.best_mrr,
            "best_epoch": self.best_epoch,
            "graph_stats": {
                "num_entities": self.graph_stats.num_entities,
                "num_relations": self.graph_stats.num_relations,
                "num_triples": self.graph_stats.num_triples,
                "num_snapshots": self.graph_stats.num_snapshots,
                "density": self.graph_stats.density,
                "avg_degree": self.graph_stats.avg_degree,
            }
            if self.graph_stats
            else None,
            "epoch_metrics": [
                {
                    "epoch": m.epoch,
                    "loss": m.loss,
                    "mrr": m.mrr,
                    "hits_at_1": m.hits_at_1,
                    "hits_at_3": m.hits_at_3,
                    "hits_at_10": m.hits_at_10,
                    "duration_seconds": m.duration_seconds,
                    "timestamp": m.timestamp,
                }
                for m in self.metrics_history
            ],
        }

        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Metrics saved to {metrics_file}")

    def get_loss_curve(self) -> Tuple[List[int], List[float]]:
        """
        Get loss curve data for plotting.

        Returns:
            Tuple of (epochs, losses) lists
        """
        epochs = [m.epoch for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        return epochs, losses

    def get_mrr_curve(self) -> Tuple[List[int], List[float]]:
        """
        Get MRR curve data for plotting.

        Returns:
            Tuple of (epochs, mrr_values) lists
        """
        epochs = [m.epoch for m in self.metrics_history]
        mrr_values = [m.mrr for m in self.metrics_history]
        return epochs, mrr_values
