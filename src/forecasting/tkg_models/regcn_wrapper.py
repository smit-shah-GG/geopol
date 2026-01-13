"""
Wrapper for RE-GCN temporal knowledge graph prediction model.

RE-GCN (Recurrent Evolution on Graph Convolutional Network) performs
link prediction on temporal knowledge graphs using:
1. Recurrent graph neural networks to capture temporal dynamics
2. ConvTransE decoder for link prediction
3. Multi-step extrapolation for future events

This wrapper provides:
- Simplified interface to RE-GCN model
- Integration with our NetworkX graph format via DataAdapter
- CPU-compatible inference using pure PyTorch implementation
- Fallback to simple baseline when model unavailable

Reference:
    Li et al. (2021). Temporal Knowledge Graph Reasoning Based on
    Evolutional Representation Learning. SIGIR 2021.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)


class REGCNWrapper:
    """
    Wrapper for RE-GCN temporal knowledge graph model.

    Provides simplified interface for:
    - Model initialization (with or without pre-trained weights)
    - Link prediction: (subject, ?, object) -> relation candidates
    - Entity prediction: (subject, relation, ?) -> object candidates
    - Scoring quadruples for validation

    Uses CPU-optimized RE-GCN implementation from src.training.models.regcn_cpu.
    Falls back to frequency-based baseline if import fails.

    Attributes:
        model: RE-GCN model instance (or None if using baseline)
        data_adapter: DataAdapter instance for format conversion
        num_entities: Total number of entities
        num_relations: Total number of relation types
        use_baseline: Whether using baseline instead of RE-GCN
        device: torch device (CPU)
        snapshots: Cached graph snapshots for prediction
    """

    def __init__(
        self,
        data_adapter: Optional[DataAdapter] = None,
        embedding_dim: int = 200,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_path: Optional[Path] = None
    ):
        """
        Initialize RE-GCN wrapper.

        Args:
            data_adapter: DataAdapter with fitted entity/relation mappings.
                        If None, must call fit() before use.
            embedding_dim: Dimension for entity embeddings (default: 200)
            num_layers: Number of GCN layers (default: 2)
            dropout: Dropout rate (default: 0.2)
            model_path: Path to pre-trained model checkpoint (optional)
        """
        self.data_adapter = data_adapter
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_path = model_path

        self.model: Optional[nn.Module] = None
        self.device = torch.device('cpu')
        self.use_baseline = False

        # Cached snapshots for model inference
        self.snapshots: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Statistics for baseline model
        self.relation_frequency: Dict[int, int] = {}
        self.entity_frequency: Dict[int, int] = {}
        self.triple_frequency: Dict[Tuple[int, int, int], int] = {}

        if data_adapter is not None:
            self.num_entities = data_adapter.get_num_entities()
            self.num_relations = data_adapter.get_num_relations()
            logger.info(f"Initialized with {self.num_entities} entities, "
                       f"{self.num_relations} relations")
        else:
            self.num_entities = 0
            self.num_relations = 0

        # Try to initialize RE-GCN model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize RE-GCN model using CPU implementation.

        Attempts to import our regcn_cpu module. Falls back to
        frequency-based baseline if import fails.
        """
        try:
            from src.training.models.regcn_cpu import REGCN
            logger.info("CPU RE-GCN implementation available")

            if self.num_entities > 0 and self.num_relations > 0:
                self.model = REGCN(
                    num_entities=self.num_entities,
                    num_relations=self.num_relations,
                    embedding_dim=self.embedding_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                )
                self.model.to(self.device)
                self.use_baseline = False
                logger.info("RE-GCN model initialized successfully")
            else:
                logger.info("Deferring model initialization until fit() is called")
                self.use_baseline = False

        except ImportError as e:
            logger.warning(f"CPU RE-GCN not available: {e}")
            logger.info("Falling back to frequency-based baseline model")
            self.use_baseline = True

    def _ensure_model_initialized(self) -> bool:
        """
        Ensure model is initialized with current entity/relation counts.

        Returns:
            True if model is ready, False if using baseline
        """
        if self.use_baseline:
            return False

        if self.model is not None:
            return True

        if self.num_entities == 0 or self.num_relations == 0:
            logger.warning("Cannot initialize model: no entities/relations")
            return False

        try:
            from src.training.models.regcn_cpu import REGCN
            self.model = REGCN(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                embedding_dim=self.embedding_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self.model.to(self.device)
            logger.info(f"RE-GCN model initialized: {self.num_entities} entities, "
                       f"{self.num_relations} relations")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RE-GCN model: {e}")
            self.use_baseline = True
            return False

    def _quadruples_to_snapshots(
        self,
        quadruples: np.ndarray,
        num_snapshots: int = 30,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert quadruples array to graph snapshot format.

        Args:
            quadruples: (N, 4) array [subject, relation, object, timestep]
            num_snapshots: Number of temporal buckets

        Returns:
            List of (edge_index, edge_type) tuples per snapshot
        """
        if len(quadruples) == 0:
            return []

        # Group by timestep
        timesteps = quadruples[:, 3]
        unique_steps = np.unique(timesteps)
        num_steps = len(unique_steps)

        # Determine bucket size
        if num_steps <= num_snapshots:
            # Few timesteps - one snapshot per timestep
            step_to_bucket = {step: i for i, step in enumerate(unique_steps)}
            actual_snapshots = num_steps
        else:
            # Many timesteps - bucket them
            min_step = timesteps.min()
            max_step = timesteps.max()
            bucket_size = (max_step - min_step + 1) / num_snapshots
            step_to_bucket = {
                step: min(int((step - min_step) / bucket_size), num_snapshots - 1)
                for step in unique_steps
            }
            actual_snapshots = num_snapshots

        # Build snapshots
        snapshots = []
        for bucket_idx in range(actual_snapshots):
            bucket_mask = np.array([step_to_bucket[t] == bucket_idx for t in timesteps])
            bucket_quads = quadruples[bucket_mask]

            if len(bucket_quads) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_type = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                # edge_index: [source, target] = [subject, object]
                edge_index = torch.tensor(
                    bucket_quads[:, [0, 2]].T,
                    dtype=torch.long,
                    device=self.device,
                )
                edge_type = torch.tensor(
                    bucket_quads[:, 1],
                    dtype=torch.long,
                    device=self.device,
                )
            snapshots.append((edge_index, edge_type))

        return snapshots

    def fit(
        self,
        data_adapter: DataAdapter,
        quadruples: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        num_snapshots: int = 30,
        num_negatives: int = 10,
        margin: float = 1.0,
        verbose: bool = True,
    ) -> None:
        """
        Fit model on training data.

        For RE-GCN: trains the model using margin-based ranking loss.
        For baseline: builds frequency statistics.

        Args:
            data_adapter: DataAdapter with entity/relation mappings
            quadruples: Training quadruples (N, 4) array
            epochs: Training epochs (default: 100)
            learning_rate: Adam learning rate (default: 0.001)
            batch_size: Training batch size (default: 1024)
            num_snapshots: Number of temporal snapshots (default: 30)
            num_negatives: Negatives per positive (default: 10)
            margin: Margin for ranking loss (default: 1.0)
            verbose: Show training progress (default: True)
        """
        self.data_adapter = data_adapter
        self.num_entities = data_adapter.get_num_entities()
        self.num_relations = data_adapter.get_num_relations()

        # Always fit baseline statistics (useful for fallback)
        self._fit_baseline(quadruples)

        # Convert to snapshots
        self.snapshots = self._quadruples_to_snapshots(quadruples, num_snapshots)

        if not self._ensure_model_initialized():
            logger.info("Using baseline model (RE-GCN initialization failed)")
            return

        logger.info(f"Training RE-GCN model for {epochs} epochs")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()

        # Extract triples for training (ignore timestep for loss computation)
        triples = quadruples[:, :3].astype(np.int64)
        num_triples = len(triples)

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(num_triples)
            triples_shuffled = triples[perm]

            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, num_triples, batch_size):
                batch_end = min(batch_start + batch_size, num_triples)
                batch_triples = triples_shuffled[batch_start:batch_end]

                # Generate negative samples
                negatives = self._negative_sampling(batch_triples, num_negatives)

                # Convert to tensors
                pos_tensor = torch.tensor(batch_triples, dtype=torch.long, device=self.device)
                neg_tensor = torch.tensor(negatives, dtype=torch.long, device=self.device)

                # Forward pass and loss
                optimizer.zero_grad()
                loss = self.model.compute_loss(
                    self.snapshots,
                    pos_tensor,
                    neg_tensor,
                    margin=margin,
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

        self.model.eval()
        logger.info("RE-GCN training complete")

    def _negative_sampling(
        self,
        positive_triples: np.ndarray,
        num_negatives: int,
    ) -> np.ndarray:
        """
        Generate negative samples by corrupting positive triples.

        Args:
            positive_triples: (batch, 3) array of [s, r, o]
            num_negatives: Negatives per positive

        Returns:
            (batch, num_negatives, 3) array of corrupted triples
        """
        batch_size = len(positive_triples)
        negatives = np.zeros((batch_size, num_negatives, 3), dtype=np.int64)

        for i, (s, r, o) in enumerate(positive_triples):
            for j in range(num_negatives):
                if np.random.random() < 0.5:
                    # Corrupt tail
                    new_o = np.random.randint(self.num_entities)
                    negatives[i, j] = [s, r, new_o]
                else:
                    # Corrupt head
                    new_s = np.random.randint(self.num_entities)
                    negatives[i, j] = [new_s, r, o]

        return negatives

    def _fit_baseline(self, quadruples: np.ndarray) -> None:
        """
        Fit frequency-based baseline model.

        Counts occurrences of relations, entities, and triples.

        Args:
            quadruples: Training data (N, 4) array
        """
        logger.info("Building baseline frequency statistics")

        self.relation_frequency.clear()
        self.entity_frequency.clear()
        self.triple_frequency.clear()

        for quad in quadruples:
            subject_id, relation_id, object_id, timestep = quad

            # Count relation frequency
            self.relation_frequency[relation_id] = \
                self.relation_frequency.get(relation_id, 0) + 1

            # Count entity frequency
            self.entity_frequency[subject_id] = \
                self.entity_frequency.get(subject_id, 0) + 1
            self.entity_frequency[object_id] = \
                self.entity_frequency.get(object_id, 0) + 1

            # Count triple frequency (ignoring time)
            triple = (subject_id, relation_id, object_id)
            self.triple_frequency[triple] = \
                self.triple_frequency.get(triple, 0) + 1

        logger.info(f"Baseline statistics: {len(quadruples)} quadruples, "
                   f"{len(self.entity_frequency)} entities, "
                   f"{len(self.relation_frequency)} relations")

    def predict_relation(
        self,
        subject_id: int,
        object_id: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Predict relation type for (subject, ?, object).

        Args:
            subject_id: Subject entity ID
            object_id: Object entity ID
            k: Number of top predictions to return

        Returns:
            List of (relation_id, confidence) tuples, sorted by confidence
        """
        if self.use_baseline or self.model is None:
            return self._predict_relation_baseline(subject_id, object_id, k)

        # RE-GCN relation prediction
        # Score each relation type with the ConvTransE decoder
        self.model.eval()
        with torch.no_grad():
            entity_emb = self.model.evolve_embeddings(self.snapshots)
            subject_emb = entity_emb[subject_id].unsqueeze(0)
            object_emb = entity_emb[object_id].unsqueeze(0)

            scores = []
            for rel_id in range(self.num_relations):
                rel_emb = self.model.relation_embeddings(
                    torch.tensor([rel_id], device=self.device)
                )
                score = self.model.decoder.score_triple(
                    subject_emb, rel_emb, object_emb
                )
                scores.append((rel_id, score.item()))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Normalize to confidences
            total = sum(max(s, 0) for _, s in scores) if scores else 1.0
            total = max(total, 1.0)
            predictions = [(r, max(s, 0) / total) for r, s in scores[:k]]

        return predictions

    def _predict_relation_baseline(
        self,
        subject_id: int,
        object_id: int,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Baseline relation prediction using frequency statistics.

        Returns most frequent relations seen with this subject or object.

        Args:
            subject_id: Subject entity ID
            object_id: Object entity ID
            k: Number of top predictions

        Returns:
            List of (relation_id, confidence) tuples
        """
        scores = {}

        # Score based on historical triples
        for (s, r, o), count in self.triple_frequency.items():
            if s == subject_id or o == object_id:
                scores[r] = scores.get(r, 0) + count

        # If no specific history, use global relation frequency
        if not scores:
            scores = self.relation_frequency.copy()

        # Normalize to confidences
        total = sum(scores.values()) if scores else 1.0
        predictions = [(r, count / total) for r, count in scores.items()]
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:k]

    def predict_object(
        self,
        subject_id: int,
        relation_id: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Predict object entity for (subject, relation, ?).

        Args:
            subject_id: Subject entity ID
            relation_id: Relation type ID
            k: Number of top predictions to return

        Returns:
            List of (object_id, confidence) tuples, sorted by confidence
        """
        if self.use_baseline or self.model is None:
            return self._predict_object_baseline(subject_id, relation_id, k)

        # RE-GCN object prediction using trained model
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(
                self.snapshots,
                subject_id,
                relation_id,
                k=k,
            )

            # Normalize scores to confidences
            total = sum(max(s, 0) for _, s in predictions) if predictions else 1.0
            total = max(total, 1.0)
            predictions = [(oid, max(s, 0) / total) for oid, s in predictions]

        return predictions

    def _predict_object_baseline(
        self,
        subject_id: int,
        relation_id: int,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Baseline object prediction using frequency statistics.

        Returns most frequent objects seen with this (subject, relation).

        Args:
            subject_id: Subject entity ID
            relation_id: Relation type ID
            k: Number of top predictions

        Returns:
            List of (object_id, confidence) tuples
        """
        scores = {}

        # Find triples with matching subject and relation
        for (s, r, o), count in self.triple_frequency.items():
            if s == subject_id and r == relation_id:
                scores[o] = scores.get(o, 0) + count

        # If no exact matches, use objects seen with this relation
        if not scores:
            for (s, r, o), count in self.triple_frequency.items():
                if r == relation_id:
                    scores[o] = scores.get(o, 0) + count

        # If still no matches, use global entity frequency
        if not scores:
            scores = self.entity_frequency.copy()

        # Normalize to confidences
        total = sum(scores.values()) if scores else 1.0
        predictions = [(o, count / total) for o, count in scores.items()]
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:k]

    def score_triple(
        self,
        subject_id: int,
        relation_id: int,
        object_id: int
    ) -> float:
        """
        Compute plausibility score for triple.

        Args:
            subject_id: Subject entity ID
            relation_id: Relation type ID
            object_id: Object entity ID

        Returns:
            Confidence score in [0, 1]
        """
        if self.use_baseline or self.model is None:
            return self._score_triple_baseline(subject_id, relation_id, object_id)

        # RE-GCN scoring
        self.model.eval()
        with torch.no_grad():
            entity_emb = self.model.evolve_embeddings(self.snapshots)

            subject_emb = entity_emb[subject_id].unsqueeze(0)
            object_emb = entity_emb[object_id].unsqueeze(0)
            relation_emb = self.model.relation_embeddings(
                torch.tensor([relation_id], device=self.device)
            )

            score = self.model.decoder.score_triple(
                subject_emb, relation_emb, object_emb
            )

            # Sigmoid to normalize to [0, 1]
            confidence = torch.sigmoid(score).item()

        return confidence

    def _score_triple_baseline(
        self,
        subject_id: int,
        relation_id: int,
        object_id: int
    ) -> float:
        """
        Baseline triple scoring using frequency statistics.

        Args:
            subject_id: Subject entity ID
            relation_id: Relation type ID
            object_id: Object entity ID

        Returns:
            Normalized frequency score
        """
        triple = (subject_id, relation_id, object_id)
        count = self.triple_frequency.get(triple, 0)

        # Normalize by relation frequency
        rel_count = self.relation_frequency.get(relation_id, 1)
        score = count / rel_count if rel_count > 0 else 0.0

        return min(score, 1.0)  # Cap at 1.0

    def get_embedding(self, entity_id: int) -> Optional[np.ndarray]:
        """
        Get entity embedding vector.

        Args:
            entity_id: Entity ID

        Returns:
            Embedding vector or None if unavailable
        """
        if self.use_baseline or self.model is None:
            return None

        self.model.eval()
        with torch.no_grad():
            entity_emb = self.model.evolve_embeddings(self.snapshots)
            return entity_emb[entity_id].cpu().numpy()

    def save_model(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'embedding_dim': self.embedding_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'use_baseline': self.use_baseline,
            'relation_frequency': self.relation_frequency,
            'entity_frequency': self.entity_frequency,
            'triple_frequency': self.triple_frequency,
        }

        if not self.use_baseline and self.model is not None:
            checkpoint['model_state_dict'] = self.model.state_dict()

        # Save data adapter mappings if available
        if self.data_adapter is not None:
            checkpoint['entity_to_id'] = self.data_adapter.entity_to_id
            checkpoint['id_to_entity'] = self.data_adapter.id_to_entity
            checkpoint['relation_to_id'] = self.data_adapter.relation_to_id
            checkpoint['id_to_relation'] = self.data_adapter.id_to_relation

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.embedding_dim = checkpoint['embedding_dim']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.num_entities = checkpoint['num_entities']
        self.num_relations = checkpoint['num_relations']
        self.use_baseline = checkpoint['use_baseline']
        self.relation_frequency = checkpoint['relation_frequency']
        self.entity_frequency = checkpoint['entity_frequency']
        self.triple_frequency = checkpoint['triple_frequency']

        # Load RE-GCN model state if available
        if not self.use_baseline and 'model_state_dict' in checkpoint:
            try:
                from src.training.models.regcn_cpu import REGCN
                self.model = REGCN(
                    num_entities=self.num_entities,
                    num_relations=self.num_relations,
                    embedding_dim=self.embedding_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                logger.info("RE-GCN model weights loaded")
            except Exception as e:
                logger.warning(f"Could not load RE-GCN model: {e}")
                self.use_baseline = True

        # Restore data adapter mappings if available
        if self.data_adapter is not None and 'entity_to_id' in checkpoint:
            self.data_adapter.entity_to_id = checkpoint['entity_to_id']
            self.data_adapter.id_to_entity = checkpoint['id_to_entity']
            self.data_adapter.relation_to_id = checkpoint['relation_to_id']
            self.data_adapter.id_to_relation = checkpoint['id_to_relation']

        logger.info(f"Model loaded from {path} (baseline={self.use_baseline})")
