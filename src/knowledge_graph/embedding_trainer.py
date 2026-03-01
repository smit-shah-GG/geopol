"""
Training Pipeline for RotatE Embeddings

Implements efficient training loop with:
- Batched data loading from NetworkX temporal graphs
- Adam optimizer with exponential learning rate decay
- Early stopping based on validation loss
- Checkpointing every N epochs
- Comprehensive training statistics
- Auto-detection of CUDA for GPU acceleration

Target Performance:
    - 100K facts train in < 10 minutes for 500 epochs
    - Throughput > 10K triples/second on 8-core CPU (faster on GPU)
"""

import logging
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import json
import pickle
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

from src.knowledge_graph.embeddings import RotatEModel, clip_gradients


@dataclass
class TrainingConfig:
    """Configuration for embedding training."""

    # Model parameters
    embedding_dim: int = 256
    margin: float = 9.0
    negative_samples: int = 4

    # Training parameters
    batch_size: int = 256
    num_epochs: int = 500
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Learning rate decay
    lr_decay_rate: float = 0.95  # Multiply LR by this every decay_step epochs
    lr_decay_step: int = 50  # Decay every N epochs

    # Early stopping
    early_stopping_patience: int = 50  # Stop if no improvement for N epochs
    early_stopping_delta: float = 1e-4  # Minimum change to qualify as improvement

    # Checkpointing
    checkpoint_every: int = 100  # Save checkpoint every N epochs
    checkpoint_dir: str = 'checkpoints'

    # Validation
    validation_split: float = 0.1  # Fraction of data for validation

    # Device
    device: str = 'auto'  # 'auto', 'cuda', or 'cpu' - auto detects CUDA

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class TemporalGraphDataset(Dataset):
    """
    PyTorch dataset for temporal knowledge graph triples.

    Converts NetworkX MultiDiGraph edges into (head, relation, tail) triples
    with entity and relation ID mappings.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        entity_to_id: Dict[str, int],
        relation_to_id: Dict[str, int]
    ):
        """
        Initialize dataset from temporal knowledge graph.

        Args:
            graph: NetworkX MultiDiGraph with temporal edges
            entity_to_id: Mapping from entity strings to integer IDs
            relation_to_id: Mapping from relation types to integer IDs
        """
        self.graph = graph
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

        # Extract all triples from graph
        self.triples = self._extract_triples()

    def _extract_triples(self) -> List[Tuple[int, int, int]]:
        """
        Extract (head, relation, tail) triples from graph.

        Returns:
            List of (head_id, relation_id, tail_id) tuples
        """
        triples = []

        for head, tail, key, data in self.graph.edges(keys=True, data=True):
            # Get entity IDs
            head_id = self.entity_to_id.get(head)
            tail_id = self.entity_to_id.get(tail)

            # Get relation ID from edge data
            relation_type = data.get('relation_type', 'UNKNOWN')
            relation_id = self.relation_to_id.get(relation_type)

            # Skip if any ID is missing
            if head_id is None or tail_id is None or relation_id is None:
                continue

            triples.append((head_id, relation_id, tail_id))

        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        return self.triples[idx]


def create_entity_relation_mappings(
    graph: nx.MultiDiGraph
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """
    Create bidirectional mappings between entities/relations and integer IDs.

    Args:
        graph: NetworkX MultiDiGraph

    Returns:
        Tuple of (entity_to_id, relation_to_id, id_to_entity, id_to_relation)
    """
    # Extract unique entities and relations
    entities = set(graph.nodes())
    relations = set()

    for _, _, _, data in graph.edges(keys=True, data=True):
        relation_type = data.get('relation_type', 'UNKNOWN')
        relations.add(relation_type)

    # Create mappings
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    relation_to_id = {relation: idx for idx, relation in enumerate(sorted(relations))}

    # Create reverse mappings
    id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}
    id_to_relation = {idx: relation for relation, idx in relation_to_id.items()}

    return entity_to_id, relation_to_id, id_to_entity, id_to_relation


def collate_triples(batch: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Converts list of triples to batched tensors.

    Args:
        batch: List of (head, relation, tail) tuples

    Returns:
        Tuple of (head_batch, relation_batch, tail_batch) tensors
    """
    heads, relations, tails = zip(*batch)
    return (
        torch.tensor(heads, dtype=torch.long),
        torch.tensor(relations, dtype=torch.long),
        torch.tensor(tails, dtype=torch.long)
    )


class EmbeddingTrainer:
    """
    Trainer for RotatE embeddings with full training loop.

    Handles:
        - Data loading and batching
        - Model training with Adam optimizer
        - Learning rate scheduling
        - Early stopping
        - Checkpointing
        - Training statistics tracking
    """

    def __init__(
        self,
        model: RotatEModel,
        config: TrainingConfig
    ):
        """
        Initialize trainer.

        Args:
            model: RotatE model to train
            config: Training configuration
        """
        self.model = model
        self.config = config

        # Resolve device
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay_rate
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data

        Returns:
            Dictionary with training history (losses, scores, etc.)
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        logger.info(f"Training batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")

        training_start = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_stats = self._train_epoch(train_loader)

            # Validation phase
            if val_loader:
                val_loss, val_stats = self._validate_epoch(val_loader)
            else:
                val_loss, val_stats = None, {}

            # Update learning rate
            self.scheduler.step()

            # Record epoch metrics
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()}
            }
            self.training_history.append(epoch_metrics)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss_str} | "
                            f"LR: {current_lr:.6f} | "
                            f"Time: {epoch_time:.2f}s")

            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1)

            # Early stopping
            if val_loader and self._should_early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch + 1} "
                            f"(no improvement for {self.config.early_stopping_patience} epochs)")
                break

            self.current_epoch = epoch + 1

        training_time = time.time() - training_start
        logger.info(f"Training completed in {training_time:.2f}s "
                    f"({training_time/60:.2f} minutes)")

        return self._format_history()

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Tuple of (average_loss, statistics_dict)
        """
        self.model.train()
        total_loss = 0.0
        total_stats = {'pos_score': 0.0, 'neg_score': 0.0, 'violation_rate': 0.0}
        num_batches = 0

        for head_batch, relation_batch, tail_batch in train_loader:
            # Move to device
            head_batch = head_batch.to(self.device)
            relation_batch = relation_batch.to(self.device)
            tail_batch = tail_batch.to(self.device)

            # Forward pass
            loss, stats = self.model.margin_ranking_loss(
                head_batch, relation_batch, tail_batch
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_gradients(self.model, self.config.gradient_clip_norm)

            # Update parameters
            self.optimizer.step()

            # Enforce model constraints
            self.model.enforce_constraints()

            # Accumulate statistics
            total_loss += loss.item()
            for key, value in stats.items():
                total_stats[key] += value
            num_batches += 1

        # Average statistics
        avg_loss = total_loss / num_batches
        avg_stats = {k: v / num_batches for k, v in total_stats.items()}

        return avg_loss, avg_stats

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (average_loss, statistics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        total_stats = {'pos_score': 0.0, 'neg_score': 0.0, 'violation_rate': 0.0}
        num_batches = 0

        with torch.no_grad():
            for head_batch, relation_batch, tail_batch in val_loader:
                # Move to device
                head_batch = head_batch.to(self.device)
                relation_batch = relation_batch.to(self.device)
                tail_batch = tail_batch.to(self.device)

                # Forward pass only
                loss, stats = self.model.margin_ranking_loss(
                    head_batch, relation_batch, tail_batch
                )

                # Accumulate statistics
                total_loss += loss.item()
                for key, value in stats.items():
                    total_stats[key] += value
                num_batches += 1

        # Average statistics
        avg_loss = total_loss / num_batches
        avg_stats = {k: v / num_batches for k, v in total_stats.items()}

        return avg_loss, avg_stats

    def _should_early_stop(self, val_loss: float) -> bool:
        """
        Check if training should stop early.

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_val_loss - self.config.early_stopping_delta:
            # Improvement
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            # No improvement
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.config.early_stopping_patience

    def _save_checkpoint(self, epoch: int):
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        self.current_epoch = checkpoint['epoch']

        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {self.current_epoch})")

    def _format_history(self) -> Dict[str, List[float]]:
        """Format training history into columnar dictionary."""
        if not self.training_history:
            return {}

        history = {}
        keys = self.training_history[0].keys()

        for key in keys:
            history[key] = [epoch_data[key] for epoch_data in self.training_history]

        return history

    def save_model(self, save_path: str):
        """
        Save trained model to disk.

        Args:
            save_path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }, save_path)
        logger.info(f"Model saved to {save_path}")


def train_embeddings_from_graph(
    graph: nx.MultiDiGraph,
    config: Optional[TrainingConfig] = None,
    save_path: Optional[str] = None
) -> Tuple[RotatEModel, Dict[str, int], Dict[str, int], Dict[str, List[float]]]:
    """
    High-level function to train embeddings from temporal knowledge graph.

    Args:
        graph: NetworkX MultiDiGraph with temporal edges
        config: Training configuration (uses defaults if None)
        save_path: Optional path to save trained model

    Returns:
        Tuple of (trained_model, entity_to_id, relation_to_id, training_history)
    """
    if config is None:
        config = TrainingConfig()

    logger.info("Creating entity and relation mappings...")
    entity_to_id, relation_to_id, id_to_entity, id_to_relation = \
        create_entity_relation_mappings(graph)

    logger.info(f"Entities: {len(entity_to_id)}, Relations: {len(relation_to_id)}")

    # Create dataset
    logger.info("Creating dataset...")
    dataset = TemporalGraphDataset(graph, entity_to_id, relation_to_id)
    logger.info(f"Total triples: {len(dataset)}")

    # Split into train/val
    val_size = int(config.validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_triples,
        num_workers=0  # CPU only
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_triples,
        num_workers=0
    )

    # Initialize model
    logger.info("Initializing RotatE model...")
    model = RotatEModel(
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        embedding_dim=config.embedding_dim,
        margin=config.margin,
        negative_samples=config.negative_samples
    )

    # Initialize trainer
    trainer = EmbeddingTrainer(model, config)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Save if requested
    if save_path:
        trainer.save_model(save_path)

        # Also save mappings
        mappings_path = save_path.replace('.pt', '_mappings.pkl')
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'entity_to_id': entity_to_id,
                'relation_to_id': relation_to_id,
                'id_to_entity': id_to_entity,
                'id_to_relation': id_to_relation
            }, f)
        logger.info(f"Mappings saved to {mappings_path}")

    return model, entity_to_id, relation_to_id, history
