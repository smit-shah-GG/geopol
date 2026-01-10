"""
Graph Pattern Extractor for RAG pipeline.

Converts temporal graph patterns into LlamaIndex documents for retrieval.
Extracts structured patterns like:
- Escalation sequences
- Diplomatic cycles
- Conflict chains
- Actor behavior profiles
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from collections import defaultdict

from llama_index.core import Document
from llama_index.core.schema import TextNode
import networkx as nx

from ..knowledge_graph.graph_builder import TemporalKnowledgeGraph
from ..knowledge_graph.graph_traversal import GraphTraversal

logger = logging.getLogger(__name__)


class GraphPatternExtractor:
    """Extract graph patterns as documents for RAG."""

    def __init__(self, graph: TemporalKnowledgeGraph):
        """
        Initialize pattern extractor.

        Args:
            graph: TemporalKnowledgeGraph instance
        """
        self.graph = graph
        self.traversal = GraphTraversal(graph.graph)

    def extract_all_patterns(self,
                            time_window_days: int = 30,
                            min_pattern_size: int = 3) -> List[Document]:
        """
        Extract all types of patterns from graph.

        Args:
            time_window_days: Days to group events
            min_pattern_size: Minimum edges in pattern

        Returns:
            List of LlamaIndex documents
        """
        documents = []

        # Extract different pattern types
        documents.extend(self.extract_escalation_patterns(time_window_days, min_pattern_size))
        documents.extend(self.extract_actor_profiles())
        documents.extend(self.extract_bilateral_histories())
        documents.extend(self.extract_conflict_chains(time_window_days))

        logger.info(f"Extracted {len(documents)} patterns from graph")
        return documents

    def extract_escalation_patterns(self,
                                   time_window_days: int = 30,
                                   min_pattern_size: int = 3) -> List[Document]:
        """
        Extract escalation sequences (diplomatic -> conflict).

        Args:
            time_window_days: Days to group events
            min_pattern_size: Minimum edges in pattern

        Returns:
            List of escalation pattern documents
        """
        documents = []

        # Find nodes with both diplomatic and conflict edges
        escalation_nodes = set()
        for u, v, key, data in self.graph.graph.edges(keys=True, data=True):
            quad_class = data.get('quad_class')
            if quad_class in [1, 4]:  # Diplomatic or conflict
                escalation_nodes.add(u)
                escalation_nodes.add(v)

        # Extract patterns for each active node
        for node in escalation_nodes:
            patterns = self._extract_node_escalation_patterns(
                node, time_window_days, min_pattern_size
            )

            for pattern in patterns:
                doc = self._pattern_to_document(
                    pattern_type="escalation",
                    pattern_data=pattern,
                    node_id=node
                )
                documents.append(doc)

        return documents

    def _extract_node_escalation_patterns(self,
                                         node: str,
                                         time_window_days: int,
                                         min_pattern_size: int) -> List[Dict]:
        """Extract escalation patterns for specific node."""
        patterns = []

        # Get all edges for node
        edges = []
        for neighbor in self.graph.graph.neighbors(node):
            for key, data in self.graph.graph[node][neighbor].items():
                edges.append({
                    'source': node,
                    'target': neighbor,
                    'timestamp': data.get('timestamp', ''),
                    'quad_class': data.get('quad_class'),
                    'relation_type': data.get('relation_type'),
                    'confidence': data.get('confidence', 0),
                    'goldstein_scale': data.get('goldstein_scale', 0)
                })

        # Sort by timestamp
        edges.sort(key=lambda x: x['timestamp'])

        # Group into time windows
        if not edges:
            return patterns

        current_window = []
        window_start = edges[0]['timestamp']

        for edge in edges:
            edge_time = edge['timestamp']

            # Check if still in window
            if self._days_between(window_start, edge_time) <= time_window_days:
                current_window.append(edge)
            else:
                # Process current window
                if len(current_window) >= min_pattern_size:
                    pattern = self._analyze_window_pattern(current_window)
                    if pattern:
                        patterns.append(pattern)

                # Start new window
                current_window = [edge]
                window_start = edge_time

        # Process last window
        if len(current_window) >= min_pattern_size:
            pattern = self._analyze_window_pattern(current_window)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _analyze_window_pattern(self, edges: List[Dict]) -> Optional[Dict]:
        """Analyze edges in time window for patterns."""
        # Count quad classes
        quad_classes = [e['quad_class'] for e in edges]

        # Look for escalation: diplomatic (1) -> conflict (4)
        has_diplomatic = 1 in quad_classes
        has_conflict = 4 in quad_classes

        if not (has_diplomatic and has_conflict):
            return None

        # Find transition point
        first_conflict_idx = -1
        last_diplomatic_idx = -1

        for i, qc in enumerate(quad_classes):
            if qc == 1:
                last_diplomatic_idx = i
            elif qc == 4 and first_conflict_idx == -1:
                first_conflict_idx = i

        # Check if diplomatic preceded conflict
        if last_diplomatic_idx < first_conflict_idx and last_diplomatic_idx >= 0:
            return {
                'pattern_type': 'escalation',
                'start_time': edges[0]['timestamp'],
                'end_time': edges[-1]['timestamp'],
                'event_count': len(edges),
                'diplomatic_events': sum(1 for qc in quad_classes if qc == 1),
                'conflict_events': sum(1 for qc in quad_classes if qc == 4),
                'transition_index': first_conflict_idx,
                'actors': list(set(e['target'] for e in edges)),
                'avg_goldstein': sum(e['goldstein_scale'] for e in edges) / len(edges),
                'edges': edges
            }

        return None

    def extract_actor_profiles(self) -> List[Document]:
        """
        Extract actor behavior profiles.

        Returns:
            List of actor profile documents
        """
        documents = []

        # Get actor statistics
        for node, node_data in self.graph.graph.nodes(data=True):
            # Get all edges for this actor
            out_edges = []
            in_edges = []

            for neighbor in self.graph.graph.successors(node):
                for key, data in self.graph.graph[node][neighbor].items():
                    out_edges.append(data)

            for predecessor in self.graph.graph.predecessors(node):
                for key, data in self.graph.graph[predecessor][node].items():
                    in_edges.append(data)

            if len(out_edges) + len(in_edges) < 5:
                continue  # Skip inactive actors

            # Analyze behavior
            profile = {
                'actor_id': node,
                'actor_name': node_data.get('name', node),
                'entity_type': node_data.get('entity_type', 'unknown'),
                'total_actions': len(out_edges),
                'total_received': len(in_edges),
                'diplomatic_actions': sum(1 for e in out_edges if e.get('quad_class') == 1),
                'conflict_actions': sum(1 for e in out_edges if e.get('quad_class') == 4),
                'avg_goldstein_out': sum(e.get('goldstein_scale', 0) for e in out_edges) / max(1, len(out_edges)),
                'avg_goldstein_in': sum(e.get('goldstein_scale', 0) for e in in_edges) / max(1, len(in_edges)),
                'top_targets': self._get_top_targets(node),
                'top_sources': self._get_top_sources(node),
                'conflict_ratio': sum(1 for e in out_edges if e.get('quad_class') == 4) / max(1, len(out_edges))
            }

            doc = self._pattern_to_document(
                pattern_type="actor_profile",
                pattern_data=profile,
                node_id=node
            )
            documents.append(doc)

        return documents

    def _get_top_targets(self, node: str, limit: int = 5) -> List[Dict]:
        """Get top target actors for node."""
        target_counts = defaultdict(int)

        for neighbor in self.graph.graph.successors(node):
            edge_count = len(self.graph.graph[node][neighbor])
            target_counts[neighbor] = edge_count

        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [{'target': t, 'interactions': c} for t, c in sorted_targets]

    def _get_top_sources(self, node: str, limit: int = 5) -> List[Dict]:
        """Get top source actors for node."""
        source_counts = defaultdict(int)

        for predecessor in self.graph.graph.predecessors(node):
            edge_count = len(self.graph.graph[predecessor][node])
            source_counts[predecessor] = edge_count

        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [{'source': s, 'interactions': c} for s, c in sorted_sources]

    def extract_bilateral_histories(self, min_interactions: int = 10) -> List[Document]:
        """
        Extract bilateral relationship histories.

        Args:
            min_interactions: Minimum interactions for inclusion

        Returns:
            List of bilateral history documents
        """
        documents = []
        processed_pairs = set()

        for u, v, key, data in self.graph.graph.edges(keys=True, data=True):
            pair = tuple(sorted([u, v]))

            if pair in processed_pairs:
                continue

            # Get all interactions between pair
            interactions = []

            # u -> v edges
            if self.graph.graph.has_edge(u, v):
                for k, d in self.graph.graph[u][v].items():
                    interactions.append({
                        'source': u,
                        'target': v,
                        **d
                    })

            # v -> u edges
            if self.graph.graph.has_edge(v, u):
                for k, d in self.graph.graph[v][u].items():
                    interactions.append({
                        'source': v,
                        'target': u,
                        **d
                    })

            if len(interactions) < min_interactions:
                continue

            # Analyze bilateral relationship
            history = self._analyze_bilateral_history(u, v, interactions)

            doc = self._pattern_to_document(
                pattern_type="bilateral_history",
                pattern_data=history,
                node_id=f"{u}__{v}"
            )
            documents.append(doc)

            processed_pairs.add(pair)

        return documents

    def _analyze_bilateral_history(self, actor1: str, actor2: str, interactions: List[Dict]) -> Dict:
        """Analyze bilateral interaction history."""
        # Sort by timestamp
        interactions.sort(key=lambda x: x.get('timestamp', ''))

        # Calculate statistics
        diplomatic_count = sum(1 for i in interactions if i.get('quad_class') == 1)
        conflict_count = sum(1 for i in interactions if i.get('quad_class') == 4)

        # Direction analysis
        actor1_to_actor2 = sum(1 for i in interactions if i['source'] == actor1)
        actor2_to_actor1 = sum(1 for i in interactions if i['source'] == actor2)

        # Time analysis
        timestamps = [i.get('timestamp', '') for i in interactions if i.get('timestamp')]

        if timestamps:
            first_interaction = timestamps[0]
            last_interaction = timestamps[-1]
            duration_days = self._days_between(first_interaction, last_interaction)
        else:
            first_interaction = ''
            last_interaction = ''
            duration_days = 0

        return {
            'actor1': actor1,
            'actor2': actor2,
            'total_interactions': len(interactions),
            'diplomatic_interactions': diplomatic_count,
            'conflict_interactions': conflict_count,
            'cooperation_ratio': diplomatic_count / max(1, len(interactions)),
            'actor1_initiated': actor1_to_actor2,
            'actor2_initiated': actor2_to_actor1,
            'first_interaction': first_interaction,
            'last_interaction': last_interaction,
            'relationship_duration_days': duration_days,
            'avg_goldstein': sum(i.get('goldstein_scale', 0) for i in interactions) / max(1, len(interactions)),
            'recent_trend': self._calculate_trend(interactions[-10:]) if len(interactions) > 10 else 'stable'
        }

    def _calculate_trend(self, recent_interactions: List[Dict]) -> str:
        """Calculate relationship trend from recent interactions."""
        if not recent_interactions:
            return 'stable'

        recent_goldstein = [i.get('goldstein_scale', 0) for i in recent_interactions]

        # Simple trend: compare first half to second half
        mid = len(recent_goldstein) // 2
        first_half = sum(recent_goldstein[:mid]) / max(1, mid)
        second_half = sum(recent_goldstein[mid:]) / max(1, len(recent_goldstein) - mid)

        if second_half > first_half + 1:
            return 'improving'
        elif second_half < first_half - 1:
            return 'deteriorating'
        else:
            return 'stable'

    def extract_conflict_chains(self, time_window_days: int = 7) -> List[Document]:
        """
        Extract conflict propagation chains.

        Args:
            time_window_days: Days to link conflicts

        Returns:
            List of conflict chain documents
        """
        documents = []

        # Get all conflict edges
        conflict_edges = []
        for u, v, key, data in self.graph.graph.edges(keys=True, data=True):
            if data.get('quad_class') == 4:  # Conflict
                conflict_edges.append({
                    'source': u,
                    'target': v,
                    'timestamp': data.get('timestamp', ''),
                    'goldstein_scale': data.get('goldstein_scale', 0),
                    'relation_type': data.get('relation_type'),
                    'confidence': data.get('confidence', 0)
                })

        # Sort by timestamp
        conflict_edges.sort(key=lambda x: x['timestamp'])

        # Find chains
        chains = self._find_conflict_chains(conflict_edges, time_window_days)

        for chain in chains:
            doc = self._pattern_to_document(
                pattern_type="conflict_chain",
                pattern_data=chain
            )
            documents.append(doc)

        return documents

    def _find_conflict_chains(self, conflict_edges: List[Dict], time_window_days: int) -> List[Dict]:
        """Find chains of related conflicts."""
        chains = []
        used_edges = set()

        for i, edge in enumerate(conflict_edges):
            if i in used_edges:
                continue

            # Start new chain
            chain = [edge]
            chain_actors = {edge['source'], edge['target']}
            used_edges.add(i)

            # Look for related conflicts
            for j in range(i + 1, len(conflict_edges)):
                if j in used_edges:
                    continue

                next_edge = conflict_edges[j]

                # Check time window
                if self._days_between(edge['timestamp'], next_edge['timestamp']) > time_window_days:
                    break  # Too far in future

                # Check if connected (shares actor)
                if next_edge['source'] in chain_actors or next_edge['target'] in chain_actors:
                    chain.append(next_edge)
                    chain_actors.add(next_edge['source'])
                    chain_actors.add(next_edge['target'])
                    used_edges.add(j)

            if len(chain) >= 3:  # Minimum chain length
                chains.append({
                    'events': chain,
                    'actors': list(chain_actors),
                    'duration_days': self._days_between(chain[0]['timestamp'], chain[-1]['timestamp']),
                    'total_events': len(chain),
                    'avg_goldstein': sum(e['goldstein_scale'] for e in chain) / len(chain),
                    'start_time': chain[0]['timestamp'],
                    'end_time': chain[-1]['timestamp']
                })

        return chains

    def _pattern_to_document(self,
                           pattern_type: str,
                           pattern_data: Dict,
                           node_id: Optional[str] = None) -> Document:
        """
        Convert pattern to LlamaIndex document.

        Args:
            pattern_type: Type of pattern
            pattern_data: Pattern data dictionary
            node_id: Optional node identifier

        Returns:
            LlamaIndex Document
        """
        # Create text description
        text = self._generate_pattern_text(pattern_type, pattern_data)

        # Create metadata
        metadata = {
            'pattern_type': pattern_type,
            'extraction_time': datetime.utcnow().isoformat(),
            'node_id': node_id
        }

        # Add key metrics to metadata
        if pattern_type == 'escalation':
            metadata['event_count'] = pattern_data.get('event_count', 0)
            metadata['duration_days'] = self._days_between(
                pattern_data.get('start_time', ''),
                pattern_data.get('end_time', '')
            )
        elif pattern_type == 'actor_profile':
            metadata['actor_name'] = pattern_data.get('actor_name', '')
            metadata['conflict_ratio'] = pattern_data.get('conflict_ratio', 0)
        elif pattern_type == 'bilateral_history':
            metadata['actors'] = f"{pattern_data.get('actor1')}_{pattern_data.get('actor2')}"
            metadata['total_interactions'] = pattern_data.get('total_interactions', 0)
        elif pattern_type == 'conflict_chain':
            metadata['chain_length'] = pattern_data.get('total_events', 0)
            metadata['actors_involved'] = len(pattern_data.get('actors', []))

        # Create document with structured data
        doc = Document(
            text=text,
            metadata=metadata,
            extra_info={
                'pattern_data': json.dumps(pattern_data, default=str)
            }
        )

        return doc

    def _generate_pattern_text(self, pattern_type: str, pattern_data: Dict) -> str:
        """Generate natural language description of pattern."""
        if pattern_type == 'escalation':
            return (
                f"Escalation pattern detected involving {len(pattern_data.get('actors', []))} actors "
                f"over {pattern_data.get('event_count', 0)} events. "
                f"Pattern shows {pattern_data.get('diplomatic_events', 0)} diplomatic events "
                f"followed by {pattern_data.get('conflict_events', 0)} conflict events. "
                f"Average Goldstein scale: {pattern_data.get('avg_goldstein', 0):.2f}. "
                f"Time period: {pattern_data.get('start_time', '')} to {pattern_data.get('end_time', '')}."
            )

        elif pattern_type == 'actor_profile':
            return (
                f"Actor profile for {pattern_data.get('actor_name', 'Unknown')} "
                f"({pattern_data.get('entity_type', 'unknown')}): "
                f"{pattern_data.get('total_actions', 0)} initiated actions, "
                f"{pattern_data.get('total_received', 0)} received actions. "
                f"Conflict ratio: {pattern_data.get('conflict_ratio', 0):.2%}. "
                f"Average outgoing Goldstein: {pattern_data.get('avg_goldstein_out', 0):.2f}. "
                f"Top targets: {', '.join(t['target'] for t in pattern_data.get('top_targets', [])[:3])}."
            )

        elif pattern_type == 'bilateral_history':
            return (
                f"Bilateral relationship between {pattern_data.get('actor1', '')} "
                f"and {pattern_data.get('actor2', '')}: "
                f"{pattern_data.get('total_interactions', 0)} total interactions "
                f"({pattern_data.get('diplomatic_interactions', 0)} diplomatic, "
                f"{pattern_data.get('conflict_interactions', 0)} conflict). "
                f"Cooperation ratio: {pattern_data.get('cooperation_ratio', 0):.2%}. "
                f"Relationship trend: {pattern_data.get('recent_trend', 'unknown')}. "
                f"Duration: {pattern_data.get('relationship_duration_days', 0)} days."
            )

        elif pattern_type == 'conflict_chain':
            return (
                f"Conflict chain involving {len(pattern_data.get('actors', []))} actors "
                f"with {pattern_data.get('total_events', 0)} conflict events. "
                f"Chain duration: {pattern_data.get('duration_days', 0)} days. "
                f"Average Goldstein: {pattern_data.get('avg_goldstein', 0):.2f}. "
                f"Actors: {', '.join(pattern_data.get('actors', [])[:5])}. "
                f"Time period: {pattern_data.get('start_time', '')} to {pattern_data.get('end_time', '')}."
            )

        else:
            return f"Pattern of type {pattern_type}: {json.dumps(pattern_data, default=str)}"

    def _days_between(self, timestamp1: str, timestamp2: str) -> int:
        """Calculate days between two ISO timestamps."""
        try:
            dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))
            return abs((dt2 - dt1).days)
        except:
            return 0