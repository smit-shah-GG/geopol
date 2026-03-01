"""
Temporal Knowledge Graph Predictor for future event forecasting.

This module provides high-level interface for TKG-based predictions:
1. Load/train RE-GCN or TiRGN model on historical graph data
2. Predict future events via link prediction
3. Apply temporal decay weighting for older events
4. Return ranked predictions with confidence scores

Backend selection is controlled by the ``TKG_BACKEND`` environment variable
(via ``Settings.tkg_backend``).  Default is ``regcn`` for backward
compatibility; setting ``TKG_BACKEND=tirgn`` loads the TiRGN model instead.
Switching backend requires a process restart -- there is no per-request model
selection.

The predictor integrates with NetworkX graphs via DataAdapter and provides
predictions that can be used for:
- Scenario validation (check if LLM scenarios align with graph patterns)
- Future event generation (what events are likely next?)
- Confidence scoring (how plausible is a given event?)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from src.forecasting.tkg_models.data_adapter import DataAdapter
from src.forecasting.tkg_models.regcn_wrapper import REGCNWrapper
from src.settings import get_settings

logger = logging.getLogger(__name__)


class TKGPredictor:
    """
    High-level interface for temporal knowledge graph predictions.

    Manages:
    - Model training/loading from NetworkX graphs
    - Future event prediction with temporal decay
    - Multi-hop reasoning over graph patterns
    - Confidence calibration

    The backend (RE-GCN or TiRGN) is selected at construction time via
    ``Settings.tkg_backend``.  The public API (``predict_future_events``,
    ``validate_scenario_event``) is identical regardless of backend.

    Attributes:
        model: REGCNWrapper instance (regcn backend) or None (tirgn backend)
        adapter: DataAdapter for format conversion
        history_length: Number of recent time steps to consider (default: 30 days)
        decay_rate: Temporal decay factor for older events (default: 0.95 per day)
        trained: Whether model has been fitted
    """

    # Mapping from semantic relation labels to CAMEO codes
    # CAMEO QuadClass: Q1=Verbal Coop, Q2=Material Coop, Q3=Verbal Conflict, Q4=Material Conflict
    # Ordered by frequency in training data (most common first)
    SEMANTIC_TO_CAMEO = {
        "CONFLICT": [
            "190_Q4",   # Use of conventional military force (72,921)
            "173_Q4",   # Arrest/detention (61,251)
            "111_Q3",   # Criticize (46,595)
            "112_Q3",   # Accuse (39,187)
            "130_Q3",   # Threaten (32,886)
            "120_Q3",   # Reject (31,600)
        ],
        "COOPERATION": [
            "10_Q1",    # Make statement (149,076)
            "42_Q1",    # Make visit (129,128)
            "43_Q1",    # Host visit (120,717)
            "40_Q1",    # Consult (115,699)
            "51_Q1",    # Praise/endorse (110,491)
            "20_Q1",    # Appeal (99,629)
        ],
        "SANCTION": [
            "163_Q4",   # Impose embargo/sanctions
            "162_Q4",   # Restrict economic activity
            "161_Q4",   # Halt negotiations
        ],
        "DIPLOMATIC": [
            "36_Q1",    # Express intent to meet/negotiate
            "30_Q1",    # Express intent to cooperate
            "46_Q1",    # Engage in diplomatic exchange
        ],
        "INTERACT": [
            "10_Q1",    # Make statement (most generic)
        ],
    }

    # Entity name aliases (normalized names -> GDELT actor names)
    # GDELT uses uppercase actor names from news articles
    ENTITY_ALIASES = {
        # Countries - full name -> GDELT name
        "russian federation": "RUSSIA",
        "russia": "RUSSIA",
        "united states": "UNITED STATES",
        "united states of america": "UNITED STATES",
        "usa": "UNITED STATES",
        "u.s.": "UNITED STATES",
        "u.s.a.": "UNITED STATES",
        "america": "UNITED STATES",
        "people's republic of china": "CHINA",
        "peoples republic of china": "CHINA",  # Without apostrophe
        "people's republic of china (prc)": "CHINA",
        "peoples republic of china (prc)": "CHINA",
        "prc": "CHINA",
        "china": "CHINA",
        "mainland china": "CHINA",
        "beijing": "CHINA",  # Often used as metonym
        "united kingdom": "UNITED KINGDOM",
        "uk": "UNITED KINGDOM",
        "great britain": "UNITED KINGDOM",
        "britain": "UNITED KINGDOM",
        "ukraine": "UKRAINE",
        "iran": "IRAN",
        "islamic republic of iran": "IRAN",
        "israel": "ISRAEL",
        "north korea": "NORTH KOREA",
        "dprk": "NORTH KOREA",
        "democratic people's republic of korea": "NORTH KOREA",
        "democratic peoples republic of korea": "NORTH KOREA",
        "south korea": "SOUTH KOREA",
        "republic of korea": "SOUTH KOREA",
        "rok": "SOUTH KOREA",
        "taiwan": "TAIWAN",
        "republic of china": "TAIWAN",
        "taiwan (roc)": "TAIWAN",
        "roc": "TAIWAN",
        "germany": "GERMANY",
        "france": "FRANCE",
        "japan": "JAPAN",
        "india": "INDIA",
        "pakistan": "PAKISTAN",
        "turkey": "TURKEY",
        "saudi arabia": "SAUDI ARABIA",
        "poland": "POLAND",
        "european union": "EUROPEAN UNION",
        "european union (eu)": "EUROPEAN UNION",
        "eu": "EUROPEAN UNION",
        # Organizations
        "nato": "NATO",
        "north atlantic treaty organization": "NATO",
        "united nations": "UNITED NATIONS",
        "un": "UNITED NATIONS",
        "european central bank": "EUROPEAN CENTRAL BANK",
        "ecb": "EUROPEAN CENTRAL BANK",
        "imf": "INTERNATIONAL MONETARY FUND",
        "international monetary fund": "INTERNATIONAL MONETARY FUND",
        "world bank": "WORLD BANK",
        "opec": "OPEC",
        # Leaders (common references)
        "putin": "PUTIN",
        "vladimir putin": "PUTIN",
        "xi jinping": "XI JINPING",
        "biden": "BIDEN",
        "joe biden": "BIDEN",
        "zelensky": "ZELENSKY",
        "volodymyr zelensky": "ZELENSKY",
    }

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

        Backend is determined by ``Settings.tkg_backend`` (env var
        ``TKG_BACKEND``).  When ``tirgn`` is selected, REGCNWrapper is NOT
        instantiated; the TiRGN model is loaded lazily from checkpoint.

        Args:
            model: Pre-initialized REGCNWrapper (created if None, regcn only)
            adapter: Pre-fitted DataAdapter (created if None)
            history_length: Number of recent days to use for training
            decay_rate: Temporal decay per day (0.95 = 5% decay per day)
            embedding_dim: Dimension for embeddings (default: 200)
            auto_load: If True, automatically load pretrained model
        """
        settings = get_settings()
        self._backend = settings.tkg_backend
        self.adapter = adapter or DataAdapter()
        self.history_length = history_length
        self.decay_rate = decay_rate
        self.trained = False

        # TiRGN-specific state (only populated when _backend == "tirgn")
        self._tirgn_model = None
        self._tirgn_snapshots: list = []

        if self._backend == "tirgn":
            # TiRGN mode: no REGCNWrapper
            self.model = None
            logger.info(
                "TKG backend: TiRGN (history_length=%d, decay_rate=%.2f)",
                history_length,
                decay_rate,
            )
        else:
            # RE-GCN mode: existing behavior
            self.model = model or REGCNWrapper(embedding_dim=embedding_dim)
            logger.info(
                "TKG backend: RE-GCN (history_length=%d, decay_rate=%.2f)",
                history_length,
                decay_rate,
            )

        # Auto-load pretrained model if available
        if auto_load and model is None:
            self._try_load_pretrained()

    @property
    def default_model_path(self) -> Path:
        """Return the default checkpoint path for the active backend."""
        if self._backend == "tirgn":
            return Path("models/tkg/tirgn_best.npz")
        return Path("models/tkg/regcn_trained.pt")

    def _try_load_pretrained(self) -> bool:
        """
        Attempt to load pretrained model from default path.

        Returns:
            True if model was loaded, False otherwise
        """
        path = self.default_model_path
        if not path.exists():
            logger.info("No pretrained model at %s", path)
            return False

        try:
            if self._backend == "tirgn":
                return self._load_tirgn_checkpoint(path)
            else:
                self.load_pretrained(path)
                return True
        except Exception as e:
            logger.warning("Failed to load pretrained model: %s", e)
        return False

    # ------------------------------------------------------------------
    # TiRGN checkpoint loading
    # ------------------------------------------------------------------

    def _load_tirgn_checkpoint(self, npz_path: Path) -> bool:
        """Load a TiRGN checkpoint (.npz weights + .json metadata).

        Validates that the JSON metadata contains ``model_type == "tirgn"``.
        If the metadata model_type does not match the configured backend,
        logs an error and returns False (falls back to baseline).

        Args:
            npz_path: Path to the ``.npz`` weights file.

        Returns:
            True if the model was loaded successfully.

        Raises:
            ValueError: If weight keys in ``.npz`` do not match the model
                        structure (shape/config mismatch).
        """
        import jax
        import jax.numpy as jnp  # noqa: F401 -- needed for array reconstruction
        from flax import nnx

        from src.training.models.tirgn_jax import create_tirgn_model

        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            logger.error("TiRGN metadata file not found: %s", meta_path)
            return False

        with open(meta_path) as f:
            metadata = json.load(f)

        # Validate model_type discriminator
        checkpoint_type = metadata.get("model_type")
        if checkpoint_type != "tirgn":
            logger.error(
                "Checkpoint model_type mismatch: expected 'tirgn', got '%s' "
                "(checkpoint: %s). Falling back to baseline.",
                checkpoint_type,
                npz_path,
            )
            return False

        config = metadata.get("config", {})
        num_entities = config.get("num_entities", 0)
        num_relations = config.get("num_relations", 0)
        embedding_dim = config.get("embedding_dim", 200)
        num_layers = config.get("num_layers", 2)
        history_rate = config.get("history_rate", 0.3)
        history_window = config.get("history_window", 50)

        if num_entities == 0 or num_relations == 0:
            logger.error("Checkpoint missing entity/relation counts")
            return False

        # Restore adapter mappings from metadata
        entity_to_id = metadata.get("entity_to_id")
        relation_to_id = metadata.get("relation_to_id")
        if entity_to_id and relation_to_id:
            self.adapter.entity_to_id = entity_to_id
            self.adapter.id_to_entity = {v: k for k, v in entity_to_id.items()}
            self.adapter.relation_to_id = relation_to_id
            self.adapter.id_to_relation = {v: k for k, v in relation_to_id.items()}
            logger.info(
                "Restored mappings: %d entities, %d relations",
                len(entity_to_id),
                len(relation_to_id),
            )

        # Create fresh model with random weights (correct shape)
        model = create_tirgn_model(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            history_rate=history_rate,
            history_window=history_window,
            seed=0,
        )

        # Split model to get abstract state structure + graphdef
        state, graphdef = nnx.split(model)

        # Load saved weights from .npz
        loaded = np.load(npz_path)

        # Reconstruct state: iterate the fresh model's leaf paths and
        # replace each leaf with the corresponding array from .npz.
        # The .npz keys are str(path) for each leaf (same format used
        # by save_tirgn_checkpoint in train_tirgn.py).
        leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(state)

        restored_leaves = []
        for key_path, leaf in leaves_with_paths:
            key_str = str(key_path)
            if key_str not in loaded:
                raise ValueError(
                    f"TiRGN checkpoint missing key '{key_str}' -- "
                    f"checkpoint and model config are incompatible"
                )
            restored_leaves.append(jnp.array(loaded[key_str]))

        restored_state = treedef.unflatten(restored_leaves)

        # Merge restored state back with graphdef to get the model
        self._tirgn_model = nnx.merge(graphdef, restored_state)

        # Mark as trained
        self.trained = True

        epoch = metadata.get("epoch", "unknown")
        metrics = metadata.get("metrics", {})
        mrr = metrics.get("mrr", "N/A")

        logger.info("TiRGN model loaded successfully")
        logger.info("  Trained for: %s epochs", epoch)
        logger.info("  MRR: %s", mrr)
        logger.info("  Entities: %s", f"{num_entities:,}")
        logger.info("  Relations: %s", num_relations)

        return True

    # ------------------------------------------------------------------
    # RE-GCN checkpoint loading (existing behavior)
    # ------------------------------------------------------------------

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

        The method dispatches to TiRGN or RE-GCN transparently based on
        the configured backend.

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

        if self._backend == "tirgn":
            predictions = self._predict_tirgn(
                entity1, entity1_id,
                relation, relation_id,
                entity2, entity2_id,
                k,
            )
        else:
            # RE-GCN path (original behavior)
            predictions = self._predict_regcn(
                entity1, entity1_id,
                relation, relation_id,
                entity2, entity2_id,
                k,
            )

        # Apply temporal decay if requested
        if apply_decay:
            predictions = self._apply_temporal_decay(predictions)

        return predictions

    # ------------------------------------------------------------------
    # RE-GCN prediction (existing logic, extracted to method)
    # ------------------------------------------------------------------

    def _predict_regcn(
        self,
        entity1: Optional[str], entity1_id: Optional[int],
        relation: Optional[str], relation_id: Optional[int],
        entity2: Optional[str], entity2_id: Optional[int],
        k: int,
    ) -> List[Dict[str, Union[str, float]]]:
        """Dispatch RE-GCN prediction by query type."""
        if relation is None:
            return self._predict_relation(entity1_id, entity2_id, k)
        elif entity2 is None:
            return self._predict_object(entity1_id, relation_id, k)
        elif entity1 is None:
            return self._predict_subject(relation_id, entity2_id, k)
        else:
            # Score a specific triple
            score = self.model.score_triple(entity1_id, relation_id, entity2_id)
            return [{
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2,
                'confidence': score,
            }]

    # ------------------------------------------------------------------
    # TiRGN prediction
    # ------------------------------------------------------------------

    def _predict_tirgn(
        self,
        entity1: Optional[str], entity1_id: Optional[int],
        relation: Optional[str], relation_id: Optional[int],
        entity2: Optional[str], entity2_id: Optional[int],
        k: int,
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict using TiRGN model via fused distribution scoring.

        For a fully specified triple ``(s, r, o)`` the object-position
        probability is returned as the confidence.  For wildcard queries
        (one of entity1, relation, entity2 is None) we score across all
        candidates in that dimension and return top-k.
        """
        import jax.numpy as jnp

        if self._tirgn_model is None:
            raise ValueError("TiRGN model not loaded. Load checkpoint first.")

        # Evolve entity embeddings through the temporal snapshot sequence
        entity_emb = self._tirgn_model.evolve_embeddings(
            self._tirgn_snapshots, training=False
        )

        if relation is None:
            # (entity1, ?, entity2): predict relation
            return self._predict_tirgn_relation(
                entity_emb, entity1, entity1_id, entity2, entity2_id, k
            )
        elif entity2 is None:
            # (entity1, relation, ?): predict object -- native TiRGN use-case
            return self._predict_tirgn_object(
                entity_emb, entity1, entity1_id, relation, relation_id, k
            )
        elif entity1 is None:
            # (?, relation, entity2): predict subject
            return self._predict_tirgn_subject(
                entity_emb, relation, relation_id, entity2, entity2_id, k
            )
        else:
            # Fully specified triple: score it
            query = jnp.array([[entity1_id, relation_id, entity2_id]], dtype=jnp.int32)
            scores = self._tirgn_model.compute_scores(entity_emb, query)
            confidence = float(1.0 / (1.0 + jnp.exp(-scores[0])))  # sigmoid
            return [{
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2,
                'confidence': confidence,
            }]

    def _predict_tirgn_object(
        self,
        entity_emb,
        entity1: str, entity1_id: int,
        relation: str, relation_id: int,
        k: int,
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict target entity for (subject, relation, ?) via TiRGN."""
        import jax.numpy as jnp

        # Build a dummy triple with object=0 (we want all-entity scores)
        query = jnp.array([[entity1_id, relation_id, 0]], dtype=jnp.int32)
        time_idx = jnp.zeros(1, dtype=jnp.int32)

        # Raw decoder gives (1, num_entities) scores
        all_scores = self._tirgn_model.raw_decoder(
            entity_emb, query, time_idx, training=False
        )
        probs = np.array(all_scores[0])

        # Top-k entities
        top_k_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_k_indices:
            entity2_name = self.adapter.entity_id_to_string(int(idx))
            results.append({
                'entity1': entity1,
                'relation': relation,
                'entity2': entity2_name,
                'confidence': float(probs[idx]),
            })
        return results

    def _predict_tirgn_relation(
        self,
        entity_emb,
        entity1: str, entity1_id: int,
        entity2: str, entity2_id: int,
        k: int,
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict relation for (entity1, ?, entity2) via TiRGN."""
        import jax.numpy as jnp

        num_relations = len(self.adapter.relation_to_id)
        best: list[tuple[int, float]] = []

        for rel_id in range(num_relations):
            query = jnp.array([[entity1_id, rel_id, entity2_id]], dtype=jnp.int32)
            scores = self._tirgn_model.compute_scores(entity_emb, query)
            best.append((rel_id, float(scores[0])))

        # Sort by score descending, take top-k
        best.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rel_id, score in best[:k]:
            rel_name = self.adapter.relation_id_to_string(rel_id)
            confidence = float(1.0 / (1.0 + np.exp(-score)))  # sigmoid
            results.append({
                'entity1': entity1,
                'relation': rel_name,
                'entity2': entity2,
                'confidence': confidence,
            })
        return results

    def _predict_tirgn_subject(
        self,
        entity_emb,
        relation: str, relation_id: int,
        entity2: str, entity2_id: int,
        k: int,
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict source entity for (?, relation, entity2) via TiRGN.

        TiRGN does not natively support subject prediction; we iterate
        over all entities scoring (candidate, relation, entity2) and
        apply a 0.8x penalty (consistent with RE-GCN behaviour).
        """
        import jax.numpy as jnp

        num_entities = len(self.adapter.entity_to_id)

        # Build batch of (candidate, relation, entity2) for all candidates
        candidates = jnp.arange(num_entities, dtype=jnp.int32)
        triples = jnp.stack([
            candidates,
            jnp.full(num_entities, relation_id, dtype=jnp.int32),
            jnp.full(num_entities, entity2_id, dtype=jnp.int32),
        ], axis=-1)

        scores = self._tirgn_model.compute_scores(entity_emb, triples)
        scores_np = np.array(scores)

        top_k_indices = np.argsort(scores_np)[::-1][:k]

        results = []
        for idx in top_k_indices:
            entity1_name = self.adapter.entity_id_to_string(int(idx))
            confidence = float(1.0 / (1.0 + np.exp(-scores_np[idx])))
            results.append({
                'entity1': entity1_name,
                'relation': relation,
                'entity2': entity2,
                'confidence': float(confidence * 0.8),  # Penalty for reversed query
            })
        return results

    # ------------------------------------------------------------------
    # RE-GCN prediction helpers (unchanged from original)
    # ------------------------------------------------------------------

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

    # Minimum fuzzy match score (0-100) to accept a match
    # Higher = stricter matching, fewer false positives
    FUZZY_MATCH_THRESHOLD = 85

    def _entity_to_id(self, entity: str) -> int:
        """
        Map entity string to ID.

        Resolution order:
        1. Exact match
        2. Uppercase match (GDELT convention)
        3. Strip parenthetical suffix and retry
        4. Static alias lookup
        5. Fuzzy string matching (fallback)

        Args:
            entity: Entity string (name or alias)

        Returns:
            Entity ID

        Raises:
            ValueError: If entity not found (even with fuzzy matching)
        """
        import re

        # Try exact match first
        entity_id = self.adapter.entity_to_id.get(entity)
        if entity_id is not None:
            return entity_id

        # Try uppercase (GDELT uses uppercase)
        entity_upper = entity.upper()
        entity_id = self.adapter.entity_to_id.get(entity_upper)
        if entity_id is not None:
            return entity_id

        # Strip parenthetical suffix (e.g., "Taiwan (ROC)" -> "Taiwan")
        entity_stripped = re.sub(r'\s*\([^)]*\)\s*$', '', entity).strip()
        if entity_stripped != entity:
            entity_id = self.adapter.entity_to_id.get(entity_stripped.upper())
            if entity_id is not None:
                logger.debug(f"Matched '{entity}' after stripping parenthetical to '{entity_stripped.upper()}'")
                return entity_id

        # Try alias lookup (case-insensitive) - try both original and stripped
        for variant in [entity.lower(), entity_stripped.lower()]:
            if variant in self.ENTITY_ALIASES:
                gdelt_name = self.ENTITY_ALIASES[variant]
                entity_id = self.adapter.entity_to_id.get(gdelt_name)
                if entity_id is not None:
                    logger.debug(f"Mapped entity '{entity}' via alias to '{gdelt_name}'")
                    return entity_id

        # Fuzzy matching fallback
        match = self._fuzzy_match_entity(entity_stripped or entity)
        if match is not None:
            matched_name, score = match
            entity_id = self.adapter.entity_to_id.get(matched_name)
            if entity_id is not None:
                logger.info(f"Fuzzy matched '{entity}' to '{matched_name}' (score={score:.0f})")
                return entity_id

        raise ValueError(f"Entity not found: {entity}")

    def _fuzzy_match_entity(self, entity: str) -> Optional[Tuple[str, float]]:
        """
        Find closest matching entity using fuzzy string matching.

        Uses WRatio scorer which combines multiple matching strategies
        and penalizes length differences.

        Args:
            entity: Query entity string

        Returns:
            Tuple of (matched_entity_name, score) or None if no good match
        """
        from rapidfuzz import process, fuzz

        if not self.adapter.entity_to_id:
            return None

        # Get all entity names
        entity_names = list(self.adapter.entity_to_id.keys())
        query = entity.upper()

        # Find best match using WRatio (weighted ratio - handles partial matches better)
        result = process.extractOne(
            query,
            entity_names,
            scorer=fuzz.WRatio,
            score_cutoff=self.FUZZY_MATCH_THRESHOLD,
        )

        if result is None:
            return None

        matched_name, score, _ = result

        # Additional check: reject if length ratio is too different
        # This prevents "EU" matching "EUROPEAN UNION" with high score
        len_ratio = len(query) / len(matched_name) if matched_name else 0
        if len_ratio < 0.4 or len_ratio > 2.5:
            logger.debug(f"Rejected fuzzy match '{entity}' -> '{matched_name}' (len_ratio={len_ratio:.2f})")
            return None

        return matched_name, score

    def _relation_to_id(self, relation: str) -> int:
        """
        Map relation string to ID.

        Supports both exact CAMEO codes (e.g., '190_Q4') and semantic labels
        (e.g., 'CONFLICT'). Semantic labels are mapped to the most frequent
        CAMEO code in that category.

        Args:
            relation: Relation type (CAMEO code or semantic label)

        Returns:
            Relation ID

        Raises:
            ValueError: If relation not found and not a known semantic label
        """
        # Try exact lookup first
        relation_id = self.adapter.relation_to_id.get(relation)
        if relation_id is not None:
            return relation_id

        # Check if it's a semantic label
        semantic_upper = relation.upper()
        if semantic_upper in self.SEMANTIC_TO_CAMEO:
            cameo_codes = self.SEMANTIC_TO_CAMEO[semantic_upper]
            # Try each CAMEO code in order (most frequent first)
            for cameo_code in cameo_codes:
                cameo_id = self.adapter.relation_to_id.get(cameo_code)
                if cameo_id is not None:
                    logger.debug(f"Mapped semantic '{relation}' to CAMEO '{cameo_code}'")
                    return cameo_id
            # None of the mapped CAMEO codes exist in the model
            raise ValueError(
                f"Semantic label '{relation}' mapped to CAMEO codes {cameo_codes}, "
                f"but none exist in trained model"
            )

        raise ValueError(f"Relation type not found: {relation}")

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
