"""
NetworkX temporal knowledge graph builder from normalized triples.

This module handles:
1. Initialize NetworkX MultiDiGraph with temporal edge attributes
2. Stream events from SQLite in batches
3. Add nodes with entity metadata
4. Add edges with temporal and confidence attributes
5. Implement time-window based edge filtering
"""

import logging
from typing import Dict, Optional, List, Tuple, Set
from datetime import datetime, timedelta
import sqlite3
import networkx as nx
from pathlib import Path

from entity_normalization import EntityNormalizer
from relation_classification import RelationClassifier

logger = logging.getLogger(__name__)


class TemporalKnowledgeGraph:
    """
    NetworkX-based temporal knowledge graph for geopolitical events.

    Structure: MultiDiGraph where:
    - Nodes: entities (countries, organizations, groups)
    - Edges: relations with temporal and confidence attributes
    - Multiple edge types allowed between same node pair (MultiDiGraph)
    """

    def __init__(self, normalizer: Optional[EntityNormalizer] = None,
                 classifier: Optional[RelationClassifier] = None):
        """
        Initialize temporal knowledge graph.

        Args:
            normalizer: Entity normalizer (created if None)
            classifier: Relation classifier (created if None)
        """
        self.graph = nx.MultiDiGraph()
        self.normalizer = normalizer or EntityNormalizer()
        self.classifier = classifier or RelationClassifier()

        # Metadata
        self.graph.graph['created'] = datetime.utcnow().isoformat()
        self.graph.graph['quad_classes'] = [1, 4]  # Diplomatic and conflicts
        self.graph.graph['event_count'] = 0
        self.graph.graph['unique_relations'] = 0

    def add_event_from_db_row(self, row: Dict) -> Optional[Tuple[str, str]]:
        """
        Add single event to graph from database row.

        Args:
            row: Database row with event data

        Returns:
            Tuple of (source_entity_id, target_entity_id) or None if invalid
        """
        # Extract fields
        actor1_code = row.get('actor1_code')
        actor2_code = row.get('actor2_code')
        event_code = row.get('event_code')
        quad_class = row.get('quad_class')
        event_date = row.get('event_date')
        num_mentions = row.get('num_mentions')
        goldstein_scale = row.get('goldstein_scale')
        tone = row.get('tone')

        # Resolve entities
        source_entity, target_entity = self.normalizer.resolve_entity_pair(
            actor1_code, actor2_code
        )

        if source_entity is None or target_entity is None:
            return None

        # Classify event to relation
        timestamp = f"{event_date}T00:00:00Z" if event_date else datetime.utcnow().isoformat()
        relation = self.classifier.classify_event(
            source_entity=source_entity,
            target_entity=target_entity,
            event_code=event_code,
            quad_class=quad_class,
            timestamp=timestamp,
            num_mentions=num_mentions,
            goldstein_scale=goldstein_scale,
            tone=tone
        )

        if relation is None:
            return None

        # Add to graph
        self._add_relation_to_graph(relation, row)

        return (source_entity, target_entity)

    def _add_relation_to_graph(self, relation, original_row: Dict):
        """
        Add relation as edge to graph, updating node metadata.

        Args:
            relation: Relation object
            original_row: Original database row for metadata
        """
        # Add source node if new
        if relation.source_entity not in self.graph:
            entity = self.normalizer.get_entity(relation.source_entity)
            self.graph.add_node(
                relation.source_entity,
                entity_type=entity.entity_type if entity else 'unknown',
                entity_id=relation.source_entity,
                name=entity.name if entity else relation.source_entity,
                canonical=self.normalizer.is_known_entity(relation.source_entity)
            )

        # Add target node if new
        if relation.target_entity not in self.graph:
            entity = self.normalizer.get_entity(relation.target_entity)
            self.graph.add_node(
                relation.target_entity,
                entity_type=entity.entity_type if entity else 'unknown',
                entity_id=relation.target_entity,
                name=entity.name if entity else relation.target_entity,
                canonical=self.normalizer.is_known_entity(relation.target_entity)
            )

        # Add edge with attributes
        edge_key = self.graph.add_edge(
            relation.source_entity,
            relation.target_entity,
            relation_type=relation.relation_type.value,
            timestamp=relation.timestamp,
            confidence=relation.confidence,
            quad_class=relation.quad_class,
            num_mentions=relation.num_mentions,
            goldstein_scale=relation.goldstein_scale,
            tone=relation.tone,
            event_codes=relation.event_codes,
            source_id=original_row.get('id')
        )

        # Update graph stats
        self.graph.graph['event_count'] += 1
        if self.graph.graph['event_count'] % 1000 == 0:
            self.graph.graph['unique_relations'] = self._count_unique_relations()

    def _count_unique_relations(self) -> int:
        """Count unique relation triples (source, target, type)."""
        unique = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            unique.add((u, v, data.get('relation_type')))
        return len(unique)

    def add_events_batch(self, db_path: str, batch_size: int = 1000,
                        limit: Optional[int] = None) -> Dict:
        """
        Stream events from database in batches and add to graph.

        Args:
            db_path: Path to SQLite database
            batch_size: Events per batch
            limit: Maximum total events (None for all)

        Returns:
            Statistics dict with counts and timing
        """
        import time

        stats = {
            'total_events': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'batches': 0,
            'start_time': datetime.utcnow().isoformat(),
            'duration_seconds': 0,
        }

        start_time = time.time()

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query
            query = """
            SELECT id, actor1_code, actor2_code, event_code, quad_class,
                   event_date, num_mentions, goldstein_scale, tone
            FROM events
            WHERE quad_class IN (1, 4)
            ORDER BY event_date DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)

            batch = []
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    row_dict = dict(row)
                    result = self.add_event_from_db_row(row_dict)

                    stats['total_events'] += 1
                    if result:
                        stats['valid_events'] += 1
                    else:
                        stats['invalid_events'] += 1

                stats['batches'] += 1

                if stats['batches'] % 10 == 0:
                    logger.info(
                        f"Processed {stats['total_events']} events, "
                        f"{stats['valid_events']} valid, "
                        f"{self.graph.number_of_nodes()} nodes, "
                        f"{self.graph.number_of_edges()} edges"
                    )

                if limit and stats['total_events'] >= limit:
                    break

            conn.close()

        except Exception as e:
            logger.error(f"Error reading database: {e}")
            raise

        stats['end_time'] = datetime.utcnow().isoformat()
        stats['duration_seconds'] = time.time() - start_time
        stats['final_nodes'] = self.graph.number_of_nodes()
        stats['final_edges'] = self.graph.number_of_edges()
        stats['unique_relations'] = self._count_unique_relations()

        logger.info(
            f"Batch loading complete: {stats['valid_events']} events, "
            f"{stats['final_nodes']} nodes, {stats['final_edges']} edges, "
            f"{stats['duration_seconds']:.2f}s"
        )

        return stats

    def filter_by_time_window(self, start_date: str, end_date: str) -> 'TemporalKnowledgeGraph':
        """
        Create filtered subgraph for time window.

        Args:
            start_date: ISO format start date (YYYY-MM-DD)
            end_date: ISO format end date (YYYY-MM-DD)

        Returns:
            New TemporalKnowledgeGraph with filtered edges
        """
        subgraph = TemporalKnowledgeGraph(self.normalizer, self.classifier)

        # Copy nodes
        for node, data in self.graph.nodes(data=True):
            subgraph.graph.add_node(node, **data)

        # Filter edges by timestamp
        start_ts = f"{start_date}T00:00:00Z"
        end_ts = f"{end_date}T23:59:59Z"

        edge_count = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            timestamp = data.get('timestamp', '')
            if start_ts <= timestamp <= end_ts:
                subgraph.graph.add_edge(u, v, key, **data)
                edge_count += 1

        subgraph.graph.graph['event_count'] = edge_count
        subgraph.graph.graph['time_filter'] = (start_date, end_date)

        return subgraph

    def filter_by_quadclass(self, quad_class: int) -> 'TemporalKnowledgeGraph':
        """
        Create filtered subgraph for specific QuadClass.

        Args:
            quad_class: QuadClass to filter (1, 2, 3, or 4)

        Returns:
            New TemporalKnowledgeGraph with filtered edges
        """
        subgraph = TemporalKnowledgeGraph(self.normalizer, self.classifier)

        # Copy all nodes
        for node, data in self.graph.nodes(data=True):
            subgraph.graph.add_node(node, **data)

        # Filter edges by quad_class
        edge_count = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('quad_class') == quad_class:
                subgraph.graph.add_edge(u, v, key, **data)
                edge_count += 1

        subgraph.graph.graph['event_count'] = edge_count
        subgraph.graph.graph['quad_class_filter'] = quad_class

        return subgraph

    def filter_by_confidence(self, min_confidence: float) -> 'TemporalKnowledgeGraph':
        """
        Create filtered subgraph for minimum confidence threshold.

        Args:
            min_confidence: Minimum confidence score [0, 1]

        Returns:
            New TemporalKnowledgeGraph with filtered edges
        """
        subgraph = TemporalKnowledgeGraph(self.normalizer, self.classifier)

        # Copy all nodes
        for node, data in self.graph.nodes(data=True):
            subgraph.graph.add_node(node, **data)

        # Filter edges by confidence
        edge_count = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('confidence', 0) >= min_confidence:
                subgraph.graph.add_edge(u, v, key, **data)
                edge_count += 1

        subgraph.graph.graph['event_count'] = edge_count
        subgraph.graph.graph['confidence_filter'] = min_confidence

        return subgraph

    def get_actor_statistics(self) -> Dict:
        """
        Get statistics about actors in graph.

        Returns:
            Dict with actor counts and degree statistics
        """
        if self.graph.number_of_nodes() == 0:
            return {'total_actors': 0}

        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())

        in_deg_values = list(in_degrees.values())
        out_deg_values = list(out_degrees.values())

        return {
            'total_actors': self.graph.number_of_nodes(),
            'avg_in_degree': sum(in_deg_values) / len(in_deg_values) if in_deg_values else 0,
            'avg_out_degree': sum(out_deg_values) / len(out_deg_values) if out_deg_values else 0,
            'max_in_degree': max(in_deg_values) if in_deg_values else 0,
            'max_out_degree': max(out_deg_values) if out_deg_values else 0,
            'actors_by_type': self._count_actors_by_type(),
        }

    def _count_actors_by_type(self) -> Dict[str, int]:
        """Count actors by entity type."""
        counts = {}
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'unknown')
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts

    def get_edge_statistics(self) -> Dict:
        """
        Get statistics about edges in graph.

        Returns:
            Dict with edge counts and confidence statistics
        """
        if self.graph.number_of_edges() == 0:
            return {'total_edges': 0}

        confidences = []
        relation_types = {}
        quad_classes = {}

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            conf = data.get('confidence', 0)
            confidences.append(conf)

            rel_type = data.get('relation_type', 'unknown')
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1

            qc = data.get('quad_class')
            quad_classes[qc] = quad_classes.get(qc, 0) + 1

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        max_conf = max(confidences) if confidences else 0
        min_conf = min(confidences) if confidences else 0

        return {
            'total_edges': self.graph.number_of_edges(),
            'avg_confidence': avg_conf,
            'max_confidence': max_conf,
            'min_confidence': min_conf,
            'relation_types': relation_types,
            'quad_classes': quad_classes,
        }

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        import sys

        # Rough estimate: ~1KB per node, ~2KB per edge
        nodes_kb = self.graph.number_of_nodes() * 1.0
        edges_kb = self.graph.number_of_edges() * 2.0

        return (nodes_kb + edges_kb) / 1024.0

    def get_statistics(self) -> Dict:
        """Get comprehensive graph statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'events': self.graph.graph.get('event_count', 0),
            'unique_relations': self.graph.graph.get('unique_relations', 0),
            'actors': self.get_actor_statistics(),
            'edges': self.get_edge_statistics(),
            'memory_mb': self.memory_usage_mb(),
        }


def create_graph(db_path: str, batch_size: int = 1000,
                 limit: Optional[int] = None) -> Tuple[TemporalKnowledgeGraph, Dict]:
    """
    Create and populate temporal knowledge graph from database.

    Args:
        db_path: Path to SQLite database
        batch_size: Batch size for processing
        limit: Maximum events (None for all)

    Returns:
        Tuple of (graph, stats)
    """
    graph = TemporalKnowledgeGraph()
    stats = graph.add_events_batch(db_path, batch_size, limit)
    return graph, stats
