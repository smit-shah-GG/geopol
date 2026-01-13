"""
Training utilities for RE-GCN temporal knowledge graph model.

Provides:
- Graph snapshot creation from event DataFrames
- Sparse adjacency matrix construction
- Negative sampling for contrastive training
- Evaluation metrics (MRR, Hits@K)
- Model checkpoint save/load
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def create_graph_snapshots(
    events_df,
    num_snapshots: int = 30,
    entity_to_id: Optional[Dict[str, int]] = None,
    relation_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[List[np.ndarray], Dict[str, int], Dict[str, int]]:
    """
    Split temporal events into graph snapshots for RE-GCN training.

    Divides events by timestamp into num_snapshots buckets. Each snapshot
    contains triples (subject_id, relation_id, object_id) for that time period.

    Args:
        events_df: DataFrame with columns [entity1, relation, entity2, timestamp]
                  where timestamp is integer representing days since start
        num_snapshots: Number of temporal buckets to create
        entity_to_id: Existing entity mapping (if None, builds from data)
        relation_to_id: Existing relation mapping (if None, builds from data)

    Returns:
        Tuple of:
        - List of snapshot arrays, each (num_triples, 3)
        - entity_to_id mapping dict
        - relation_to_id mapping dict

    Raises:
        ValueError: If required columns missing from DataFrame
    """
    import pandas as pd

    required_cols = {"entity1", "relation", "entity2", "timestamp"}
    if not required_cols.issubset(events_df.columns):
        missing = required_cols - set(events_df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Build entity mapping if not provided
    if entity_to_id is None:
        entities = set(events_df["entity1"].unique()) | set(events_df["entity2"].unique())
        entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        logger.info(f"Built entity mapping with {len(entity_to_id)} entities")

    # Build relation mapping if not provided
    if relation_to_id is None:
        relations = sorted(events_df["relation"].unique())
        relation_to_id = {r: i for i, r in enumerate(relations)}
        logger.info(f"Built relation mapping with {len(relation_to_id)} relations")

    # Convert to IDs, filtering unmapped entities
    df = events_df.copy()
    df["subject_id"] = df["entity1"].map(entity_to_id)
    df["relation_id"] = df["relation"].map(relation_to_id)
    df["object_id"] = df["entity2"].map(entity_to_id)

    # Drop rows with unmapped values
    valid_mask = df[["subject_id", "relation_id", "object_id"]].notna().all(axis=1)
    df = df[valid_mask]

    if len(df) == 0:
        raise ValueError("No valid triples after mapping")

    # Divide into temporal snapshots
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    time_range = max_ts - min_ts + 1

    if time_range < num_snapshots:
        # Fewer unique timestamps than requested snapshots
        num_snapshots = int(time_range)
        logger.warning(f"Reduced num_snapshots to {num_snapshots} (limited time range)")

    bucket_size = time_range / num_snapshots
    df["bucket"] = ((df["timestamp"] - min_ts) / bucket_size).astype(int)
    df["bucket"] = df["bucket"].clip(upper=num_snapshots - 1)

    # Build snapshots
    snapshots = []
    for bucket_idx in range(num_snapshots):
        bucket_df = df[df["bucket"] == bucket_idx]
        if len(bucket_df) == 0:
            # Empty snapshot - create placeholder with no edges
            triples = np.empty((0, 3), dtype=np.int64)
        else:
            triples = bucket_df[["subject_id", "relation_id", "object_id"]].values.astype(np.int64)
        snapshots.append(triples)

    # Log statistics
    total_triples = sum(len(s) for s in snapshots)
    non_empty = sum(1 for s in snapshots if len(s) > 0)
    logger.info(f"Created {num_snapshots} snapshots: {total_triples} total triples, {non_empty} non-empty")

    return snapshots, entity_to_id, relation_to_id


def build_adjacency_matrix(
    snapshot: np.ndarray,
    num_entities: int,
    num_relations: int,
) -> torch.sparse.Tensor:
    """
    Convert snapshot triples to sparse adjacency tensor.

    Creates a sparse tensor of shape (num_relations, num_entities, num_entities)
    where adjacency[r, i, j] = 1 if edge (i, r, j) exists.

    Args:
        snapshot: (num_triples, 3) array with [subject, relation, object]
        num_entities: Total number of entities
        num_relations: Total number of relation types

    Returns:
        Sparse COO tensor (num_relations, num_entities, num_entities)
    """
    if len(snapshot) == 0:
        # Return empty sparse tensor
        indices = torch.empty((3, 0), dtype=torch.long)
        values = torch.empty(0)
        size = (num_relations, num_entities, num_entities)
        return torch.sparse_coo_tensor(indices, values, size)

    # Build indices: (relation, subject, object)
    subjects = snapshot[:, 0]
    relations = snapshot[:, 1]
    objects = snapshot[:, 2]

    indices = torch.tensor(
        np.stack([relations, subjects, objects], axis=0),
        dtype=torch.long,
    )
    values = torch.ones(len(snapshot))

    size = (num_relations, num_entities, num_entities)
    adj = torch.sparse_coo_tensor(indices, values, size).coalesce()

    return adj


def negative_sampling(
    positive_triples: np.ndarray,
    num_entities: int,
    num_negatives: int = 10,
    strategy: str = "corrupt_tail",
) -> np.ndarray:
    """
    Generate negative samples by corrupting positive triples.

    For each positive triple (s, r, o), generates negatives by replacing
    either the subject or object with random entities.

    Args:
        positive_triples: (batch, 3) array of positive triples [s, r, o]
        num_entities: Total number of entities for sampling
        num_negatives: Number of negatives per positive
        strategy: "corrupt_tail" (replace o), "corrupt_head" (replace s),
                 or "uniform" (50/50 mix)

    Returns:
        (batch, num_negatives, 3) array of negative triples
    """
    batch_size = len(positive_triples)
    negatives = np.zeros((batch_size, num_negatives, 3), dtype=np.int64)

    for i, (s, r, o) in enumerate(positive_triples):
        for j in range(num_negatives):
            if strategy == "corrupt_tail":
                # Replace object with random entity
                new_o = np.random.randint(num_entities)
                negatives[i, j] = [s, r, new_o]
            elif strategy == "corrupt_head":
                # Replace subject with random entity
                new_s = np.random.randint(num_entities)
                negatives[i, j] = [new_s, r, o]
            else:
                # Uniform: 50% corrupt head, 50% corrupt tail
                if np.random.random() < 0.5:
                    new_o = np.random.randint(num_entities)
                    negatives[i, j] = [s, r, new_o]
                else:
                    new_s = np.random.randint(num_entities)
                    negatives[i, j] = [new_s, r, o]

    return negatives


def compute_mrr(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    filtered: bool = True,
    known_triples: Optional[set] = None,
) -> float:
    """
    Compute Mean Reciprocal Rank for link prediction evaluation.

    MRR = (1/n) * sum(1/rank_i) where rank_i is the rank of the true entity
    among all candidates for query i.

    Args:
        predictions: (num_queries, num_entities) array of scores
        ground_truth: (num_queries,) array of true entity indices
        filtered: If True, filter out other known correct answers (requires known_triples)
        known_triples: Set of (s, r, o) tuples for filtering

    Returns:
        MRR score in [0, 1]
    """
    num_queries = len(ground_truth)
    if num_queries == 0:
        return 0.0

    reciprocal_ranks = []

    for i, (scores, true_idx) in enumerate(zip(predictions, ground_truth)):
        true_score = scores[true_idx]
        # Count how many entities score higher (rank = num_higher + 1)
        rank = (scores > true_score).sum() + 1
        reciprocal_ranks.append(1.0 / rank)

    mrr = np.mean(reciprocal_ranks)
    return float(mrr)


def compute_hits_at_k(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Hits@K metric: fraction of queries where true answer is in top-K.

    Args:
        predictions: (num_queries, num_entities) array of scores
        ground_truth: (num_queries,) array of true entity indices
        k: Number of top candidates to consider

    Returns:
        Hits@K score in [0, 1]
    """
    num_queries = len(ground_truth)
    if num_queries == 0:
        return 0.0

    hits = 0
    for scores, true_idx in zip(predictions, ground_truth):
        top_k_indices = np.argsort(scores)[-k:]
        if true_idx in top_k_indices:
            hits += 1

    return hits / num_queries


def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    entity_to_id: Optional[Dict[str, int]] = None,
    relation_to_id: Optional[Dict[str, int]] = None,
) -> None:
    """
    Save model checkpoint with training state.

    Args:
        model: PyTorch model to save
        path: Output path for checkpoint
        optimizer: Optional optimizer state
        epoch: Current training epoch
        metrics: Optional validation metrics dict
        entity_to_id: Entity ID mapping for inference
        relation_to_id: Relation ID mapping for inference
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_entities": model.num_entities,
            "num_relations": model.num_relations,
            "embedding_dim": model.embedding_dim,
            "num_layers": model.num_layers,
        },
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if entity_to_id is not None:
        checkpoint["entity_to_id"] = entity_to_id

    if relation_to_id is not None:
        checkpoint["relation_to_id"] = relation_to_id

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        Dict containing:
        - model_state_dict: Model weights
        - model_config: Model configuration
        - epoch: Training epoch
        - optimizer_state_dict: Optional optimizer state
        - metrics: Optional validation metrics
        - entity_to_id: Optional entity mapping
        - relation_to_id: Optional relation mapping

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    logger.info(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


def prepare_batch(
    snapshots: List[np.ndarray],
    positive_triples: np.ndarray,
    num_negatives: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor, Tensor]:
    """
    Prepare training batch with snapshots and negative samples.

    Converts numpy arrays to PyTorch tensors and generates negatives.

    Args:
        snapshots: List of snapshot arrays (num_triples, 3)
        positive_triples: (batch_size, 3) positive samples
        num_negatives: Negatives per positive
        device: Target device

    Returns:
        Tuple of:
        - List of (edge_index, edge_type) snapshot tuples
        - Positive triples tensor (batch, 3)
        - Negative triples tensor (batch, num_neg, 3)
    """
    # Convert snapshots to edge format
    snapshot_tensors = []
    for snapshot in snapshots:
        if len(snapshot) == 0:
            # Empty snapshot
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty(0, dtype=torch.long, device=device)
        else:
            snapshot_t = torch.tensor(snapshot, dtype=torch.long, device=device)
            edge_index = snapshot_t[:, [0, 2]].t().contiguous()
            edge_type = snapshot_t[:, 1]
        snapshot_tensors.append((edge_index, edge_type))

    # Get num_entities for negative sampling
    max_entity = 0
    for snapshot in snapshots:
        if len(snapshot) > 0:
            max_entity = max(max_entity, snapshot[:, 0].max(), snapshot[:, 2].max())
    num_entities = int(max_entity) + 1

    # Generate negative samples
    negatives = negative_sampling(
        positive_triples,
        num_entities=num_entities,
        num_negatives=num_negatives,
        strategy="uniform",
    )

    positive_tensor = torch.tensor(positive_triples, dtype=torch.long, device=device)
    negative_tensor = torch.tensor(negatives, dtype=torch.long, device=device)

    return snapshot_tensors, positive_tensor, negative_tensor


class EarlyStopping:
    """
    Early stopping handler for training.

    Stops training if validation metric doesn't improve for patience epochs.
    Optionally saves best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "max" (higher is better) or "min" (lower is better)
            checkpoint_path: Optional path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path

        self.best_value: Optional[float] = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(
        self,
        value: float,
        epoch: int,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **save_kwargs,
    ) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            epoch: Current epoch
            model: Model to checkpoint if improved
            optimizer: Optimizer state to checkpoint
            **save_kwargs: Additional kwargs for save_checkpoint

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            self._save_if_improved(model, optimizer, epoch, **save_kwargs)
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            self._save_if_improved(model, optimizer, epoch, **save_kwargs)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best: {self.best_value:.4f} at epoch {self.best_epoch}"
            )
            return True

        return False

    def _save_if_improved(
        self,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        **save_kwargs,
    ) -> None:
        """Save checkpoint if path configured and model provided."""
        if self.checkpoint_path is not None and model is not None:
            save_checkpoint(
                model,
                self.checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"best_metric": self.best_value},
                **save_kwargs,
            )
