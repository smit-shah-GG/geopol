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
- CPU-compatible inference (DGL dependency optional)
- Fallback to simple baseline when RE-GCN unavailable

Reference:
    Li et al. (2021). Temporal Knowledge Graph Reasoning Based on
    Evolutional Representation Learning. SIGIR 2021.
"""

import logging
import os
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

    If RE-GCN dependencies (DGL) are unavailable, falls back to
    simple frequency-based baseline.

    Attributes:
        model: RE-GCN model instance (or None if using baseline)
        data_adapter: DataAdapter instance for format conversion
        num_entities: Total number of entities
        num_relations: Total number of relation types
        use_baseline: Whether using baseline instead of RE-GCN
        device: torch device (CPU)
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
        Initialize RE-GCN model or fall back to baseline.

        Attempts to import DGL and RE-GCN dependencies.
        If unavailable, sets use_baseline=True.
        """
        try:
            # Try importing DGL
            import dgl
            logger.info("DGL available - attempting RE-GCN initialization")

            # Try importing RE-GCN modules
            # Note: This requires adding RE-GCN/src to Python path
            regcn_path = Path(__file__).parent / 'RE-GCN' / 'src'
            if not regcn_path.exists():
                raise ImportError("RE-GCN source not found")

            # For now, use baseline until RE-GCN is fully integrated
            # TODO: Complete RE-GCN model initialization
            logger.warning("RE-GCN model initialization not yet implemented, using baseline")
            self.use_baseline = True

        except ImportError as e:
            logger.warning(f"DGL/RE-GCN not available: {e}")
            logger.info("Falling back to frequency-based baseline model")
            self.use_baseline = True

    def fit(self, data_adapter: DataAdapter, quadruples: np.ndarray) -> None:
        """
        Fit model on training data.

        For baseline: builds frequency statistics.
        For RE-GCN: would train the model (not implemented).

        Args:
            data_adapter: DataAdapter with entity/relation mappings
            quadruples: Training quadruples (N, 4) array
        """
        self.data_adapter = data_adapter
        self.num_entities = data_adapter.get_num_entities()
        self.num_relations = data_adapter.get_num_relations()

        if self.use_baseline:
            self._fit_baseline(quadruples)
        else:
            # TODO: Implement RE-GCN training
            logger.warning("RE-GCN training not implemented")

    def _fit_baseline(self, quadruples: np.ndarray) -> None:
        """
        Fit frequency-based baseline model.

        Counts occurrences of relations, entities, and triples.

        Args:
            quadruples: Training data (N, 4) array
        """
        logger.info("Fitting baseline frequency model")

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

        logger.info(f"Baseline model fitted on {len(quadruples)} quadruples")

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
        if self.use_baseline:
            return self._predict_relation_baseline(subject_id, object_id, k)
        else:
            # TODO: Implement RE-GCN relation prediction
            logger.warning("RE-GCN prediction not implemented, using baseline")
            return self._predict_relation_baseline(subject_id, object_id, k)

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
        if self.use_baseline:
            return self._predict_object_baseline(subject_id, relation_id, k)
        else:
            # TODO: Implement RE-GCN object prediction
            logger.warning("RE-GCN prediction not implemented, using baseline")
            return self._predict_object_baseline(subject_id, relation_id, k)

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
        if self.use_baseline:
            return self._score_triple_baseline(subject_id, relation_id, object_id)
        else:
            # TODO: Implement RE-GCN scoring
            logger.warning("RE-GCN scoring not implemented, using baseline")
            return self._score_triple_baseline(subject_id, relation_id, object_id)

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
        if self.use_baseline:
            # Baseline has no embeddings
            return None
        else:
            # TODO: Extract embeddings from RE-GCN model
            return None

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

        if not self.use_baseline and 'model_state_dict' in checkpoint:
            # TODO: Load RE-GCN model state
            pass

        logger.info(f"Model loaded from {path}")
