"""
Data adapter for converting NetworkX temporal graphs to RE-GCN format.

RE-GCN expects quadruples: (subject, relation, object, timestamp)
with consecutive integer IDs for all entities and relations.

This adapter:
1. Converts NetworkX MultiDiGraph to quadruple lists
2. Maps entity IDs to consecutive integers
3. Maps relation types to relation IDs
4. Converts timestamps to discrete time steps
5. Maintains bidirectional mapping for reconstruction
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class DataAdapter:
    """
    Converts NetworkX temporal knowledge graph to RE-GCN data format.

    RE-GCN requires:
    - Entity IDs: consecutive integers [0, num_entities)
    - Relation IDs: consecutive integers [0, num_relations)
    - Timestamps: discrete time steps (integer)
    - Format: (subject_id, relation_id, object_id, timestep)

    Attributes:
        entity_to_id: Maps entity strings to integer IDs
        id_to_entity: Reverse mapping from IDs to entity strings
        relation_to_id: Maps relation types to integer IDs
        id_to_relation: Reverse mapping from IDs to relation types
        time_granularity: Time granularity in seconds (default: 86400 = 1 day)
        min_timestamp: Reference timestamp for time step calculation
    """

    def __init__(self, time_granularity: int = 86400):
        """
        Initialize data adapter.

        Args:
            time_granularity: Time granularity in seconds.
                            Default: 86400 (1 day). Use 3600 for hourly, etc.
        """
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.id_to_relation: Dict[int, str] = {}
        self.time_granularity = time_granularity
        self.min_timestamp: Optional[float] = None

    def fit_graph(self, graph: nx.MultiDiGraph) -> None:
        """
        Build entity and relation mappings from graph.

        Scans graph to create consecutive ID mappings for all entities
        and relation types. Must be called before convert_graph().

        Args:
            graph: NetworkX MultiDiGraph with temporal edges

        Side Effects:
            Populates entity_to_id, id_to_entity, relation_to_id, id_to_relation
        """
        # Extract unique entities (nodes)
        entities = sorted(graph.nodes())
        self.entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
        self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}

        logger.info(f"Mapped {len(entities)} entities to consecutive IDs")

        # Extract unique relation types
        relation_types: Set[str] = set()
        for u, v, key, data in graph.edges(keys=True, data=True):
            relation = data.get('relation_type', 'UNKNOWN')
            relation_types.add(relation)

        relations = sorted(relation_types)
        self.relation_to_id = {rel: idx for idx, rel in enumerate(relations)}
        self.id_to_relation = {idx: rel for rel, idx in self.relation_to_id.items()}

        logger.info(f"Mapped {len(relations)} relation types to consecutive IDs")

        # Find minimum timestamp for time step calculation
        timestamps = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            ts_str = data.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                    timestamps.append(ts)
                except (ValueError, AttributeError):
                    continue

        if timestamps:
            self.min_timestamp = min(timestamps)
            logger.info(f"Reference timestamp: {self.min_timestamp} "
                       f"({datetime.fromtimestamp(self.min_timestamp).isoformat()})")
        else:
            self.min_timestamp = datetime.now().timestamp()
            logger.warning("No valid timestamps found, using current time as reference")

    def convert_graph(self, graph: nx.MultiDiGraph) -> np.ndarray:
        """
        Convert NetworkX graph to RE-GCN quadruples.

        Args:
            graph: NetworkX MultiDiGraph with temporal edges

        Returns:
            numpy array of shape (num_edges, 4) with columns:
            [subject_id, relation_id, object_id, timestep]

        Raises:
            ValueError: If fit_graph() has not been called first
        """
        if not self.entity_to_id or not self.relation_to_id:
            raise ValueError("Must call fit_graph() before convert_graph()")

        quadruples = []

        for u, v, key, data in graph.edges(keys=True, data=True):
            # Map entities
            subject_id = self.entity_to_id.get(u)
            object_id = self.entity_to_id.get(v)

            if subject_id is None or object_id is None:
                logger.warning(f"Skipping edge with unmapped entity: {u} -> {v}")
                continue

            # Map relation
            relation = data.get('relation_type', 'UNKNOWN')
            relation_id = self.relation_to_id.get(relation)

            if relation_id is None:
                logger.warning(f"Skipping edge with unmapped relation: {relation}")
                continue

            # Convert timestamp to discrete time step
            ts_str = data.get('timestamp')
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                    timestep = int((ts - self.min_timestamp) / self.time_granularity)
                except (ValueError, AttributeError):
                    timestep = 0
            else:
                timestep = 0

            quadruples.append([subject_id, relation_id, object_id, timestep])

        if not quadruples:
            logger.error("No valid quadruples generated from graph")
            return np.array([]).reshape(0, 4)

        result = np.array(quadruples, dtype=np.int64)
        logger.info(f"Converted {len(result)} edges to RE-GCN format")
        logger.info(f"Time steps range: {result[:, 3].min()} to {result[:, 3].max()}")

        return result

    def fit_convert(self, graph: nx.MultiDiGraph) -> np.ndarray:
        """
        Convenience method: fit and convert in one call.

        Args:
            graph: NetworkX MultiDiGraph with temporal edges

        Returns:
            numpy array of RE-GCN quadruples
        """
        self.fit_graph(graph)
        return self.convert_graph(graph)

    def get_num_entities(self) -> int:
        """Get total number of entities."""
        return len(self.entity_to_id)

    def get_num_relations(self) -> int:
        """Get total number of relation types."""
        return len(self.relation_to_id)

    def entity_id_to_string(self, entity_id: int) -> Optional[str]:
        """
        Convert entity ID back to original string.

        Args:
            entity_id: Integer entity ID

        Returns:
            Entity string or None if ID not found
        """
        return self.id_to_entity.get(entity_id)

    def relation_id_to_string(self, relation_id: int) -> Optional[str]:
        """
        Convert relation ID back to original type.

        Args:
            relation_id: Integer relation ID

        Returns:
            Relation type string or None if ID not found
        """
        return self.id_to_relation.get(relation_id)

    def timestep_to_datetime(self, timestep: int) -> datetime:
        """
        Convert discrete time step back to datetime.

        Args:
            timestep: Integer time step

        Returns:
            datetime object
        """
        timestamp = self.min_timestamp + (timestep * self.time_granularity)
        return datetime.fromtimestamp(timestamp)

    def split_by_time(
        self,
        quadruples: np.ndarray,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Split quadruples into train/valid/test by time steps.

        Each split is further divided by time step into snapshots
        for temporal training.

        Args:
            quadruples: Array of shape (num_edges, 4)
            train_ratio: Fraction of time steps for training
            valid_ratio: Fraction of time steps for validation

        Returns:
            Tuple of (train_snapshots, valid_snapshots, test_snapshots)
            where each is a list of arrays, one per time step
        """
        # Group by time step
        time_steps = sorted(np.unique(quadruples[:, 3]))
        num_steps = len(time_steps)

        train_cutoff = int(num_steps * train_ratio)
        valid_cutoff = int(num_steps * (train_ratio + valid_ratio))

        train_steps = time_steps[:train_cutoff]
        valid_steps = time_steps[train_cutoff:valid_cutoff]
        test_steps = time_steps[valid_cutoff:]

        def split_by_steps(steps):
            """Extract quadruples for given time steps."""
            snapshots = []
            for step in steps:
                mask = quadruples[:, 3] == step
                snapshot = quadruples[mask][:, :3]  # Remove time column
                if len(snapshot) > 0:
                    snapshots.append(snapshot)
            return snapshots

        train_snapshots = split_by_steps(train_steps)
        valid_snapshots = split_by_steps(valid_steps)
        test_snapshots = split_by_steps(test_steps)

        logger.info(f"Split into {len(train_snapshots)} train, "
                   f"{len(valid_snapshots)} valid, {len(test_snapshots)} test snapshots")

        return train_snapshots, valid_snapshots, test_snapshots

    def reconstruct_triple(
        self,
        subject_id: int,
        relation_id: int,
        object_id: int,
        timestep: int
    ) -> Optional[Tuple[str, str, str, datetime]]:
        """
        Reconstruct original triple from IDs.

        Args:
            subject_id: Entity ID
            relation_id: Relation ID
            object_id: Entity ID
            timestep: Time step

        Returns:
            Tuple of (subject, relation, object, datetime) or None if invalid
        """
        subject = self.entity_id_to_string(subject_id)
        relation = self.relation_id_to_string(relation_id)
        obj = self.entity_id_to_string(object_id)

        if subject is None or relation is None or obj is None:
            return None

        dt = self.timestep_to_datetime(timestep)
        return (subject, relation, obj, dt)
