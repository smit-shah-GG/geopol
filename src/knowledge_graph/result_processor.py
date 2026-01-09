"""
Result Processing for Knowledge Graph Queries.

Combines Tasks 4 and 5:
- Temporal filtering and aggregation (time windows, trends, decay)
- Result formatting and explanation (provenance, reasoning paths, confidence)
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EventAggregation:
    """Aggregated events over time window."""
    start_time: str
    end_time: str
    event_count: int
    total_mentions: int
    avg_confidence: float
    quad_class_distribution: Dict[int, int] = field(default_factory=dict)
    relation_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class ExplanationPath:
    """Reasoning path for a prediction or result."""
    source_entity: int
    target_entity: int
    path_edges: List[Tuple[int, int, str, float]]  # (u, v, relation, confidence)
    path_length: int
    total_confidence: float
    temporal_sequence: List[str]  # Timestamps in order
    explanation_text: str


@dataclass
class FormattedResult:
    """Complete formatted query result with explanations."""
    query_id: str
    query_type: str
    query_params: Dict[str, Any]
    timestamp: str
    execution_time_ms: float

    # Results
    entities: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregations
    aggregations: List[EventAggregation] = field(default_factory=list)

    # Explanations
    explanations: List[ExplanationPath] = field(default_factory=list)

    # Confidence
    overall_confidence: float = 0.0
    confidence_components: Dict[str, float] = field(default_factory=dict)

    # Metadata
    result_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class TemporalFilterAggregator:
    """Temporal filtering and aggregation engine."""

    def __init__(self):
        """Initialize aggregator."""
        pass

    def filter_by_time_window(
        self,
        edges: List[Tuple[int, int, int, Dict]],
        start_time: str,
        end_time: str
    ) -> List[Tuple[int, int, int, Dict]]:
        """Filter edges by time window.

        Args:
            edges: List of (u, v, key, data) tuples
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)

        Returns:
            Filtered edge list
        """
        filtered = []
        for u, v, key, data in edges:
            timestamp = data.get('timestamp', '')
            if start_time <= timestamp <= end_time:
                filtered.append((u, v, key, data))

        return filtered

    def aggregate_by_time_period(
        self,
        edges: List[Tuple[int, int, int, Dict]],
        period: str = "daily"  # daily, weekly, monthly
    ) -> List[EventAggregation]:
        """Aggregate events by time period.

        Args:
            edges: List of (u, v, key, data) tuples
            period: Aggregation period

        Returns:
            List of EventAggregation objects
        """
        # Group by time period
        period_buckets = defaultdict(list)

        for u, v, key, data in edges:
            timestamp = data.get('timestamp', '')
            bucket_key = self._get_time_bucket(timestamp, period)
            period_buckets[bucket_key].append(data)

        # Create aggregations
        aggregations = []
        for bucket_key, bucket_edges in sorted(period_buckets.items()):
            agg = self._aggregate_bucket(bucket_key, bucket_edges, period)
            aggregations.append(agg)

        return aggregations

    def compute_sliding_window_trends(
        self,
        edges: List[Tuple[int, int, int, Dict]],
        window_days: int = 7,
        slide_days: int = 1
    ) -> List[Dict[str, Any]]:
        """Compute trends using sliding window analysis.

        Args:
            edges: List of (u, v, key, data) tuples
            window_days: Window size in days
            slide_days: Slide interval in days

        Returns:
            List of trend metrics per window
        """
        if not edges:
            return []

        # Sort edges by timestamp
        sorted_edges = sorted(edges, key=lambda x: x[3].get('timestamp', ''))

        # Find time range
        first_ts = sorted_edges[0][3].get('timestamp', '')
        last_ts = sorted_edges[-1][3].get('timestamp', '')

        try:
            start_dt = datetime.fromisoformat(first_ts.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
        except Exception:
            return []

        trends = []
        current_dt = start_dt

        while current_dt <= end_dt:
            window_start = current_dt
            window_end = current_dt + timedelta(days=window_days)

            # Filter edges in window
            window_edges = [
                e for e in sorted_edges
                if window_start.isoformat() <= e[3].get('timestamp', '') <= window_end.isoformat()
            ]

            if window_edges:
                trend = {
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'event_count': len(window_edges),
                    'avg_confidence': np.mean([e[3].get('confidence', 0) for e in window_edges]),
                    'avg_mentions': np.mean([e[3].get('num_mentions', 0) for e in window_edges])
                }
                trends.append(trend)

            current_dt += timedelta(days=slide_days)

        return trends

    def apply_temporal_decay(
        self,
        edges: List[Tuple[int, int, int, Dict]],
        reference_time: str,
        half_life_days: float = 30.0
    ) -> List[Tuple[int, int, int, Dict, float]]:
        """Apply exponential temporal decay to edge weights.

        Args:
            edges: List of (u, v, key, data) tuples
            reference_time: Reference timestamp
            half_life_days: Half-life for exponential decay

        Returns:
            List of (u, v, key, data, decay_weight) tuples
        """
        try:
            ref_dt = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))
        except Exception:
            # If invalid, return with weight 1.0
            return [(u, v, k, d, 1.0) for u, v, k, d in edges]

        weighted_edges = []
        decay_constant = np.log(2) / half_life_days

        for u, v, key, data in edges:
            timestamp = data.get('timestamp', reference_time)
            try:
                edge_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                days_diff = (ref_dt - edge_dt).days

                # Exponential decay
                if days_diff >= 0:
                    weight = np.exp(-decay_constant * days_diff)
                else:
                    weight = 1.0  # Future events get full weight

                weighted_edges.append((u, v, key, data, weight))
            except Exception:
                weighted_edges.append((u, v, key, data, 1.0))

        return weighted_edges

    def detect_temporal_cooccurrence(
        self,
        edges: List[Tuple[int, int, int, Dict]],
        time_threshold_hours: int = 24
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """Detect co-occurring events within time threshold.

        Args:
            edges: List of (u, v, key, data) tuples
            time_threshold_hours: Maximum time gap for co-occurrence

        Returns:
            List of ((u1, v1), (u2, v2), time_diff_hours) tuples
        """
        cooccurrences = []
        threshold_delta = timedelta(hours=time_threshold_hours)

        # Compare all pairs
        for i in range(len(edges)):
            u1, v1, key1, data1 = edges[i]
            ts1 = data1.get('timestamp', '')

            try:
                dt1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
            except Exception:
                continue

            for j in range(i + 1, len(edges)):
                u2, v2, key2, data2 = edges[j]
                ts2 = data2.get('timestamp', '')

                try:
                    dt2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
                    time_diff = abs((dt2 - dt1).total_seconds() / 3600)

                    if time_diff <= time_threshold_hours:
                        cooccurrences.append(((u1, v1), (u2, v2), time_diff))
                except Exception:
                    continue

        return cooccurrences

    # Helper methods

    def _get_time_bucket(self, timestamp: str, period: str) -> str:
        """Get time bucket key for timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            if period == "daily":
                return dt.strftime("%Y-%m-%d")
            elif period == "weekly":
                # ISO week
                return dt.strftime("%Y-W%W")
            elif period == "monthly":
                return dt.strftime("%Y-%m")
            else:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            return "unknown"

    def _aggregate_bucket(
        self,
        bucket_key: str,
        edges_data: List[Dict],
        period: str
    ) -> EventAggregation:
        """Aggregate events in time bucket."""
        event_count = len(edges_data)
        total_mentions = sum(e.get('num_mentions', 0) for e in edges_data)
        confidences = [e.get('confidence', 0) for e in edges_data]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # QuadClass distribution
        quad_dist = defaultdict(int)
        for e in edges_data:
            qc = e.get('quad_class')
            if qc:
                quad_dist[qc] += 1

        # Relation type distribution
        rel_dist = defaultdict(int)
        for e in edges_data:
            rt = e.get('relation_type')
            if rt:
                rel_dist[rt] += 1

        # Determine time range for bucket
        if period == "daily":
            start_time = bucket_key + "T00:00:00Z"
            end_time = bucket_key + "T23:59:59Z"
        elif period == "weekly":
            start_time = bucket_key + "-1T00:00:00Z"
            end_time = bucket_key + "-7T23:59:59Z"
        elif period == "monthly":
            start_time = bucket_key + "-01T00:00:00Z"
            end_time = bucket_key + "-31T23:59:59Z"
        else:
            start_time = bucket_key
            end_time = bucket_key

        return EventAggregation(
            start_time=start_time,
            end_time=end_time,
            event_count=event_count,
            total_mentions=total_mentions,
            avg_confidence=avg_confidence,
            quad_class_distribution=dict(quad_dist),
            relation_types=dict(rel_dist)
        )


class ResultFormatter:
    """Format query results with explanations and provenance."""

    def __init__(self, graph=None, id_to_entity: Optional[Dict[int, str]] = None):
        """Initialize result formatter.

        Args:
            graph: NetworkX graph for metadata lookup
            id_to_entity: Mapping from entity IDs to names
        """
        self.graph = graph
        self.id_to_entity = id_to_entity or {}

    def format_query_result(
        self,
        query_id: str,
        query_type: str,
        query_params: Dict[str, Any],
        traversal_result,
        similarity_result=None,
        execution_time_ms: float = 0.0,
        include_explanations: bool = True
    ) -> FormattedResult:
        """Format complete query result.

        Args:
            query_id: Unique query identifier
            query_type: Type of query
            query_params: Query parameters
            traversal_result: TraversalResult from graph traversal
            similarity_result: Optional SimilarityResult from vector search
            execution_time_ms: Query execution time
            include_explanations: Whether to generate explanations

        Returns:
            FormattedResult with all components
        """
        result = FormattedResult(
            query_id=query_id,
            query_type=query_type,
            query_params=query_params,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            execution_time_ms=execution_time_ms
        )

        # Format entities
        for entity_id in traversal_result.nodes:
            entity_data = self._format_entity(entity_id)
            result.entities.append(entity_data)

        # Format edges
        for u, v, key, data in traversal_result.edges:
            edge_data = self._format_edge(u, v, key, data)
            result.edges.append(edge_data)

        # Format paths
        for path in traversal_result.paths:
            path_data = self._format_path(path)
            result.paths.append(path_data)

        # Add similarity results if available
        if similarity_result:
            result.metadata['similarity_results'] = {
                'entities': [
                    {'id': eid, 'score': score}
                    for eid, score in similarity_result.entities[:10]
                ]
            }

        # Generate explanations
        if include_explanations:
            explanations = self._generate_explanations(traversal_result)
            result.explanations.extend(explanations)

        # Calculate overall confidence
        result.overall_confidence = self._calculate_overall_confidence(traversal_result)
        result.confidence_components = self._calculate_confidence_components(traversal_result)

        # Set result count
        result.result_count = len(result.edges)

        return result

    def generate_explanation(
        self,
        path: List[Tuple[int, int, int]],
        graph
    ) -> ExplanationPath:
        """Generate explanation for a reasoning path.

        Args:
            path: List of (u, v, key) tuples
            graph: NetworkX graph

        Returns:
            ExplanationPath with detailed reasoning
        """
        if not path:
            return None

        # Extract path information
        source = path[0][0]
        target = path[-1][1]

        path_edges = []
        temporal_sequence = []
        confidences = []

        for u, v, key in path:
            edge_data = graph[u][v][key]
            confidence = edge_data.get('confidence', 0.0)
            relation = edge_data.get('relation_type', 'unknown')
            timestamp = edge_data.get('timestamp', '')

            path_edges.append((u, v, relation, confidence))
            temporal_sequence.append(timestamp)
            confidences.append(confidence)

        # Calculate total confidence (product)
        total_confidence = np.prod(confidences) if confidences else 0.0

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            source, target, path_edges, temporal_sequence
        )

        return ExplanationPath(
            source_entity=source,
            target_entity=target,
            path_edges=path_edges,
            path_length=len(path),
            total_confidence=total_confidence,
            temporal_sequence=temporal_sequence,
            explanation_text=explanation_text
        )

    def calculate_confidence_score(
        self,
        edges: List[Tuple[int, int, int, Dict]],
        method: str = "harmonic_mean"
    ) -> float:
        """Calculate aggregate confidence score.

        Args:
            edges: List of (u, v, key, data) tuples
            method: Aggregation method (harmonic_mean, geometric_mean, min)

        Returns:
            Aggregate confidence score [0,1]
        """
        if not edges:
            return 0.0

        confidences = [data.get('confidence', 0) for _, _, _, data in edges]
        confidences = [c for c in confidences if c > 0]  # Filter zeros

        if not confidences:
            return 0.0

        if method == "harmonic_mean":
            return len(confidences) / sum(1.0 / c for c in confidences)
        elif method == "geometric_mean":
            return np.prod(confidences) ** (1.0 / len(confidences))
        elif method == "min":
            return min(confidences)
        else:
            return np.mean(confidences)

    # Helper methods

    def _format_entity(self, entity_id: int) -> Dict[str, Any]:
        """Format entity for output."""
        entity_data = {
            'id': entity_id,
            'name': self.id_to_entity.get(entity_id, f'Entity_{entity_id}')
        }

        # Add graph metadata if available
        if self.graph and entity_id in self.graph:
            node_data = self.graph.nodes[entity_id]
            entity_data.update(node_data)

        return entity_data

    def _format_edge(self, u: int, v: int, key: int, data: Dict) -> Dict[str, Any]:
        """Format edge for output."""
        return {
            'source': u,
            'target': v,
            'source_name': self.id_to_entity.get(u, f'Entity_{u}'),
            'target_name': self.id_to_entity.get(v, f'Entity_{v}'),
            'relation_type': data.get('relation_type', 'unknown'),
            'confidence': data.get('confidence', 0.0),
            'timestamp': data.get('timestamp', ''),
            'quad_class': data.get('quad_class'),
            'num_mentions': data.get('num_mentions', 0)
        }

    def _format_path(self, path: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Format path for output."""
        return {
            'length': len(path),
            'edges': [
                {'source': u, 'target': v, 'key': key}
                for u, v, key in path
            ]
        }

    def _generate_explanations(self, traversal_result) -> List[ExplanationPath]:
        """Generate explanations for traversal result."""
        explanations = []

        if not self.graph:
            return explanations

        # Generate explanations for paths
        for path in traversal_result.paths[:5]:  # Top 5 paths
            explanation = self.generate_explanation(path, self.graph)
            if explanation:
                explanations.append(explanation)

        return explanations

    def _calculate_overall_confidence(self, traversal_result) -> float:
        """Calculate overall confidence for result."""
        if not traversal_result.edges:
            return 0.0

        confidences = [data.get('confidence', 0) for _, _, _, data in traversal_result.edges]
        return float(np.mean(confidences))

    def _calculate_confidence_components(self, traversal_result) -> Dict[str, float]:
        """Calculate confidence score components."""
        components = {}

        if traversal_result.edges:
            confidences = [data.get('confidence', 0) for _, _, _, data in traversal_result.edges]
            components['edge_confidence'] = float(np.mean(confidences))

            mentions = [data.get('num_mentions', 0) for _, _, _, data in traversal_result.edges]
            components['mention_strength'] = float(np.mean(mentions)) / 1000.0  # Normalize

        return components

    def _generate_explanation_text(
        self,
        source: int,
        target: int,
        path_edges: List[Tuple[int, int, str, float]],
        temporal_sequence: List[str]
    ) -> str:
        """Generate human-readable explanation text."""
        source_name = self.id_to_entity.get(source, f'Entity_{source}')
        target_name = self.id_to_entity.get(target, f'Entity_{target}')

        explanation = f"Path from {source_name} to {target_name}:\n"

        for i, (u, v, relation, confidence) in enumerate(path_edges):
            u_name = self.id_to_entity.get(u, f'Entity_{u}')
            v_name = self.id_to_entity.get(v, f'Entity_{v}')
            timestamp = temporal_sequence[i] if i < len(temporal_sequence) else 'unknown'

            explanation += f"  {i+1}. {u_name} --[{relation}]--> {v_name} "
            explanation += f"(confidence: {confidence:.2f}, time: {timestamp})\n"

        return explanation


def create_temporal_aggregator() -> TemporalFilterAggregator:
    """Create temporal filter and aggregator instance."""
    return TemporalFilterAggregator()


def create_result_formatter(
    graph=None,
    id_to_entity: Optional[Dict[int, str]] = None
) -> ResultFormatter:
    """Create result formatter instance."""
    return ResultFormatter(graph=graph, id_to_entity=id_to_entity)
