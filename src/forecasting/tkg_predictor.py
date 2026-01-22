"""
Temporal Knowledge Graph Predictor for future event forecasting.

This module provides high-level interface for TKG-based predictions:
1. Load/train RE-GCN model on historical graph data
2. Predict future events via link prediction
3. Apply temporal decay weighting for older events
4. Return ranked predictions with confidence scores

The predictor integrates with NetworkX graphs via DataAdapter and provides
predictions that can be used for:
- Scenario validation (check if LLM scenarios align with graph patterns)
- Future event generation (what events are likely next?)
- Confidence scoring (how plausible is a given event?)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from src.forecasting.tkg_models.data_adapter import DataAdapter
from src.forecasting.tkg_models.regcn_wrapper import REGCNWrapper

logger = logging.getLogger(__name__)


class TKGPredictor:
    """
    High-level interface for temporal knowledge graph predictions.

    Manages:
    - Model training/loading from NetworkX graphs
    - Future event prediction with temporal decay
    - Multi-hop reasoning over graph patterns
    - Confidence calibration

    Attributes:
        model: REGCNWrapper instance
        adapter: DataAdapter for format conversion
        history_length: Number of recent time steps to consider (default: 30 days)
        decay_rate: Temporal decay factor for older events (default: 0.95 per day)
        trained: Whether model has been fitted
    """

    # Default path for pretrained model checkpoint
    DEFAULT_MODEL_PATH = Path("models/tkg/regcn_trained.pt")

    def __init__(
        self,
        model: Optional[REGCNWrapper] = None,
        adapter: Optional[DataAdapter] = None,
        history_length: int = 30,
        decay_rate: float = 0.95,
        embedding_dim: int = 200,
        auto_load: bool = True,
    ):
        """
        Initialize TKG predictor.

        Args:
            model: Pre-initialized REGCNWrapper (created if None)
            adapter: Pre-fitted DataAdapter (created if None)
            history_length: Number of recent days to use for training
            decay_rate: Temporal decay per day (0.95 = 5% decay per day)
            embedding_dim: Dimension for embeddings (default: 200)
            auto_load: If True, automatically load pretrained model from
                      models/tkg/regcn_trained.pt if it exists
        """
        self.model = model or REGCNWrapper(embedding_dim=embedding_dim)
        self.adapter = adapter or DataAdapter()
        self.history_length = history_length
        self.decay_rate = decay_rate
        self.trained = False

        logger.info(f"Initialized TKG predictor with {history_length}-day history "
                   f"and {decay_rate} daily decay rate")

        # Auto-load pretrained model if available
        if auto_load and model is None:
            self._try_load_pretrained()

    def _try_load_pretrained(self) -> bool:
        """
        Attempt to load pretrained model from default path.

        Returns:
            True if model was loaded, False otherwise
        """
        if self.DEFAULT_MODEL_PATH.exists():
            try:
                self.load_pretrained(self.DEFAULT_MODEL_PATH)
                return True
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {e}")
        return False

    def load_pretrained(self, checkpoint_path: Path) -> None:
        """
        Load pretrained RE-GCN model from checkpoint.

        The checkpoint should contain:
        - model_state_dict: Model weights
        - model_config: num_entities, num_relations, embedding_dim, num_layers
        - entity_to_id: Entity string to ID mapping
        - relation_to_id: Relation string to ID mapping

        Args:
            checkpoint_path: Path to checkpoint file (.pt)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible
        """
        import torch

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading pretrained model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract model config
        config = checkpoint.get("model_config", {})
        num_entities = config.get("num_entities", checkpoint.get("num_entities", 0))
        num_relations = config.get("num_relations", checkpoint.get("num_relations", 0))
        embedding_dim = config.get("embedding_dim", checkpoint.get("embedding_dim", 200))
        num_layers = config.get("num_layers", checkpoint.get("num_layers", 2))

        if num_entities == 0 or num_relations == 0:
            raise RuntimeError("Checkpoint missing entity/relation counts")

        # Restore adapter mappings
        entity_to_id = checkpoint.get("entity_to_id")
        relation_to_id = checkpoint.get("relation_to_id")

        if entity_to_id and relation_to_id:
            self.adapter.entity_to_id = entity_to_id
            self.adapter.id_to_entity = {v: k for k, v in entity_to_id.items()}
            self.adapter.relation_to_id = relation_to_id
            self.adapter.id_to_relation = {v: k for k, v in relation_to_id.items()}
            logger.info(f"Restored mappings: {len(entity_to_id)} entities, "
                       f"{len(relation_to_id)} relations")

        # Initialize model with correct dimensions
        self.model = REGCNWrapper(
            data_adapter=self.adapter,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
        )
        self.model.num_entities = num_entities
        self.model.num_relations = num_relations

        # Load model weights
        if "model_state_dict" in checkpoint:
            try:
                from src.training.models.regcn import REGCN

                self.model.model = REGCN(
                    num_entities=num_entities,
                    num_relations=num_relations,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                )
                self.model.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.model.eval()
                self.model.use_baseline = False
                logger.info("Loaded RE-GCN model weights")
            except Exception as e:
                logger.warning(f"Could not load RE-GCN weights: {e}")
                self.model.use_baseline = True

        # Restore baseline statistics if available
        if "relation_frequency" in checkpoint:
            self.model.relation_frequency = checkpoint["relation_frequency"]
        if "entity_frequency" in checkpoint:
            self.model.entity_frequency = checkpoint["entity_frequency"]
        if "triple_frequency" in checkpoint:
            self.model.triple_frequency = checkpoint["triple_frequency"]

        # Mark as trained
        self.trained = True

        # Log training info from checkpoint
        epoch = checkpoint.get("epoch", "unknown")
        metrics = checkpoint.get("metrics", {})
        mrr = metrics.get("mrr", metrics.get("best_metric", "N/A"))

        logger.info(f"Pretrained model loaded successfully")
        logger.info(f"  Trained for: {epoch} epochs")
        logger.info(f"  MRR: {mrr}")
        logger.info(f"  Entities: {num_entities:,}")
        logger.info(f"  Relations: {num_relations}")

    def fit(self, graph: nx.MultiDiGraph, recent_days: Optional[int] = None) -> None:
        """
        Train predictor on recent graph history.

        Extracts recent events (last N days), converts to RE-GCN format,
        and fits the model.

        Args:
            graph: NetworkX MultiDiGraph with temporal edges
            recent_days: Number of recent days to use (defaults to history_length)

        Raises:
            ValueError: If graph is empty or has no temporal edges
        """
        if recent_days is None:
            recent_days = self.history_length

        logger.info(f"Fitting TKG predictor on last {recent_days} days of data")

        # Filter graph to recent history
        recent_graph = self._filter_recent_events(graph, recent_days)

        if recent_graph.number_of_edges() == 0:
            raise ValueError(f"No events found in recent {recent_days} days")

        logger.info(f"Using {recent_graph.number_of_nodes()} entities, "
                   f"{recent_graph.number_of_edges()} edges")

        # Convert to RE-GCN format
        quadruples = self.adapter.fit_convert(recent_graph)

        if len(quadruples) == 0:
            raise ValueError("No valid quadruples generated from graph")

        # Fit model
        self.model.fit(self.adapter, quadruples)
        self.trained = True

        logger.info("TKG predictor fitted successfully")

    def _filter_recent_events(
        self,
        graph: nx.MultiDiGraph,
        days: int
    ) -> nx.MultiDiGraph:
        """
        Extract subgraph with events from recent N days.

        Args:
            graph: Full temporal graph
            days: Number of recent days to keep

        Returns:
            Subgraph with only recent edges
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_graph = nx.MultiDiGraph()

        # Copy graph metadata
        recent_graph.graph.update(graph.graph)

        for u, v, key, data in graph.edges(keys=True, data=True):
            ts_str = data.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if ts >= cutoff:
                        recent_graph.add_edge(u, v, key=key, **data)
                except (ValueError, AttributeError):
                    # Skip malformed timestamps
                    continue

        # Add all nodes (even isolated ones) to maintain entity space
        recent_graph.add_nodes_from(graph.nodes(data=True))

        return recent_graph

    def predict_future_events(
        self,
        entity1: Optional[str] = None,
        relation: Optional[str] = None,
        entity2: Optional[str] = None,
        k: int = 10,
        apply_decay: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict future events based on query pattern.

        Supports three query types:
        1. (entity1, ?, entity2): Predict relation between two entities
        2. (entity1, relation, ?): Predict target entity
        3. (?, relation, entity2): Predict source entity

        Args:
            entity1: Source entity (None for wildcard)
            relation: Relation type (None for wildcard)
            entity2: Target entity (None for wildcard)
            k: Number of top predictions to return
            apply_decay: Whether to apply temporal decay weighting

        Returns:
            List of prediction dictionaries with keys:
            - 'entity1': Source entity string
            - 'relation': Relation type string
            - 'entity2': Target entity string
            - 'confidence': Confidence score in [0, 1]

        Raises:
            ValueError: If query is invalid (too many wildcards or model not trained)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Validate query pattern
        wildcards = sum([x is None for x in [entity1, relation, entity2]])
        if wildcards > 1:
            raise ValueError("Query must have at most one wildcard (?)")

        # Map to IDs
        entity1_id = self._entity_to_id(entity1) if entity1 else None
        relation_id = self._relation_to_id(relation) if relation else None
        entity2_id = self._entity_to_id(entity2) if entity2 else None

        # Route to appropriate prediction method
        if relation is None:
            # Query: (entity1, ?, entity2) - predict relation
            predictions = self._predict_relation(entity1_id, entity2_id, k)
        elif entity2 is None:
            # Query: (entity1, relation, ?) - predict object
            predictions = self._predict_object(entity1_id, relation_id, k)
        elif entity1 is None:
            # Query: (?, relation, entity2) - predict subject
            predictions = self._predict_subject(relation_id, entity2_id, k)
        else:
            # Query: (entity1, relation, entity2) - score triple
            score = self.model.score_triple(entity1_id, relation_id, entity2_id)
            return [{
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2,
                'confidence': score
            }]

        # Apply temporal decay if requested
        if apply_decay:
            predictions = self._apply_temporal_decay(predictions)

        return predictions

    def _predict_relation(
        self,
        entity1_id: int,
        entity2_id: int,
        k: int
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict relation between two entities.

        Args:
            entity1_id: Source entity ID
            entity2_id: Target entity ID
            k: Number of predictions

        Returns:
            List of prediction dictionaries
        """
        # Get top-k relation predictions
        predictions = self.model.predict_relation(entity1_id, entity2_id, k)

        results = []
        entity1 = self.adapter.entity_id_to_string(entity1_id)
        entity2 = self.adapter.entity_id_to_string(entity2_id)

        for relation_id, confidence in predictions:
            relation = self.adapter.relation_id_to_string(relation_id)
            results.append({
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2,
                'confidence': float(confidence)
            })

        return results

    def _predict_object(
        self,
        entity1_id: int,
        relation_id: int,
        k: int
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict target entity for (subject, relation, ?).

        Args:
            entity1_id: Source entity ID
            relation_id: Relation type ID
            k: Number of predictions

        Returns:
            List of prediction dictionaries
        """
        predictions = self.model.predict_object(entity1_id, relation_id, k)

        results = []
        entity1 = self.adapter.entity_id_to_string(entity1_id)
        relation = self.adapter.relation_id_to_string(relation_id)

        for entity2_id, confidence in predictions:
            entity2 = self.adapter.entity_id_to_string(entity2_id)
            results.append({
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2,
                'confidence': float(confidence)
            })

        return results

    def _predict_subject(
        self,
        relation_id: int,
        entity2_id: int,
        k: int
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict source entity for (?, relation, object).

        This is less common but useful for "who will act on X?" queries.

        Args:
            relation_id: Relation type ID
            entity2_id: Target entity ID
            k: Number of predictions

        Returns:
            List of prediction dictionaries
        """
        # For baseline: reverse the direction
        # TODO: Implement proper subject prediction in RE-GCN
        predictions = self.model.predict_object(entity2_id, relation_id, k)

        results = []
        entity2 = self.adapter.entity_id_to_string(entity2_id)
        relation = self.adapter.relation_id_to_string(relation_id)

        for entity1_id, confidence in predictions:
            entity1 = self.adapter.entity_id_to_string(entity1_id)
            results.append({
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2,
                'confidence': float(confidence * 0.8)  # Penalty for reversed query
            })

        return results

    def _apply_temporal_decay(
        self,
        predictions: List[Dict[str, Union[str, float]]]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Apply temporal decay to predictions based on recency.

        Assumes predictions are based on historical patterns, so older
        patterns get lower weights.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Predictions with decayed confidence scores
        """
        # For now, apply uniform decay based on assumption that
        # model uses all historical data uniformly
        # TODO: Track pattern recency for more accurate decay

        decay_factor = self.decay_rate ** (self.history_length / 2.0)

        for pred in predictions:
            pred['confidence'] = float(pred['confidence'] * decay_factor)

        return predictions

    def _entity_to_id(self, entity: str) -> int:
        """
        Map entity string to ID.

        Args:
            entity: Entity string

        Returns:
            Entity ID

        Raises:
            ValueError: If entity not found
        """
        entity_id = self.adapter.entity_to_id.get(entity)
        if entity_id is None:
            raise ValueError(f"Entity not found: {entity}")
        return entity_id

    def _relation_to_id(self, relation: str) -> int:
        """
        Map relation string to ID.

        Args:
            relation: Relation type

        Returns:
            Relation ID

        Raises:
            ValueError: If relation not found
        """
        relation_id = self.adapter.relation_to_id.get(relation)
        if relation_id is None:
            raise ValueError(f"Relation type not found: {relation}")
        return relation_id

    def validate_scenario_event(
        self,
        event: Dict[str, str]
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Validate a single scenario event against graph patterns.

        Args:
            event: Event dictionary with keys: entity1, relation, entity2

        Returns:
            Validation result with:
            - 'valid': bool - whether event is plausible
            - 'confidence': float - plausibility score
            - 'similar_events': List[Dict] - historical precedents
        """
        entity1 = event.get('entity1')
        relation = event.get('relation')
        entity2 = event.get('entity2')

        if not all([entity1, relation, entity2]):
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': 'Incomplete event specification',
                'similar_events': []
            }

        try:
            # Query for this specific triple
            predictions = self.predict_future_events(
                entity1=entity1,
                relation=relation,
                entity2=entity2,
                k=1,
                apply_decay=True
            )

            if predictions:
                confidence = predictions[0]['confidence']
                valid = confidence > 0.1  # Threshold for plausibility

                # Find similar patterns
                similar = self.predict_future_events(
                    entity1=entity1,
                    relation=None,
                    entity2=entity2,
                    k=5,
                    apply_decay=True
                )

                return {
                    'valid': valid,
                    'confidence': confidence,
                    'reason': f"Pattern confidence: {confidence:.3f}",
                    'similar_events': similar
                }
            else:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'reason': 'No historical pattern found',
                    'similar_events': []
                }

        except ValueError as e:
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': f"Validation error: {str(e)}",
                'similar_events': []
            }

    def save(self, path: Path) -> None:
        """
        Save predictor state.

        Args:
            path: Path to save checkpoint
        """
        # Save model
        model_path = path / 'model.pt'
        self.model.save_model(model_path)

        # Save adapter state
        adapter_state = {
            'entity_to_id': self.adapter.entity_to_id,
            'id_to_entity': self.adapter.id_to_entity,
            'relation_to_id': self.adapter.relation_to_id,
            'id_to_relation': self.adapter.id_to_relation,
            'time_granularity': self.adapter.time_granularity,
            'min_timestamp': self.adapter.min_timestamp,
        }

        import pickle
        adapter_path = path / 'adapter.pkl'
        with open(adapter_path, 'wb') as f:
            pickle.dump(adapter_state, f)

        logger.info(f"Predictor saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load predictor state.

        Args:
            path: Path to checkpoint directory
        """
        # Load model
        model_path = path / 'model.pt'
        self.model.load_model(model_path)

        # Load adapter state
        import pickle
        adapter_path = path / 'adapter.pkl'
        with open(adapter_path, 'rb') as f:
            adapter_state = pickle.load(f)

        self.adapter.entity_to_id = adapter_state['entity_to_id']
        self.adapter.id_to_entity = adapter_state['id_to_entity']
        self.adapter.relation_to_id = adapter_state['relation_to_id']
        self.adapter.id_to_relation = adapter_state['id_to_relation']
        self.adapter.time_granularity = adapter_state['time_granularity']
        self.adapter.min_timestamp = adapter_state['min_timestamp']

        self.trained = True
        logger.info(f"Predictor loaded from {path}")
