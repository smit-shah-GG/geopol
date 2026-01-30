"""
Graph persistence layer for saving and loading temporal knowledge graphs.

This module handles:
1. Graph serialization to GraphML format with metadata
2. Metadata preservation for nodes and edges
3. Incremental update mechanism
4. Round-trip validation
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional
import networkx as nx
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphPersistence:
    """
    Handles serialization and deserialization of temporal knowledge graphs.

    Format: GraphML for standard compatibility
    Metadata: Preserved as node/edge attributes
    """

    def __init__(self, graph_or_path=None):
        """
        Initialize persistence layer.

        Args:
            graph_or_path: NetworkX graph or path to load from (optional)
        """
        if graph_or_path is None:
            self.graph = nx.MultiDiGraph()
        elif isinstance(graph_or_path, str) or isinstance(graph_or_path, Path):
            self.graph = self.load(graph_or_path)
        else:
            self.graph = graph_or_path

    def save(self, filepath: str, format: str = 'graphml') -> Dict:
        """
        Save graph to file.

        Args:
            filepath: Output file path
            format: 'graphml' (default) or 'json'

        Returns:
            Dict with save statistics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        start_time = datetime.utcnow()

        try:
            if format == 'graphml':
                self._save_graphml(filepath)
            elif format == 'json':
                self._save_json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            stats = {
                'filepath': str(filepath),
                'format': format,
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'save_time_seconds': (datetime.utcnow() - start_time).total_seconds(),
                'timestamp': start_time.isoformat(),
            }

            logger.info(f"Saved graph to {filepath}: {stats['nodes']} nodes, {stats['edges']} edges")
            return stats

        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            raise

    def _save_graphml(self, filepath: Path):
        """Save graph in GraphML format with metadata."""
        # Add graph-level metadata
        self.graph.graph['saved_at'] = datetime.utcnow().isoformat()

        # Serialize graph-level list/dict attributes to JSON
        for key, value in list(self.graph.graph.items()):
            if isinstance(value, (list, dict)):
                self.graph.graph[key] = json.dumps(value)

        # Ensure all node attributes are serializable
        for node, data in self.graph.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, (list, dict)):
                    data[key] = json.dumps(value)

        # Ensure all edge attributes are serializable
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            # Ensure confidence is float
            if 'confidence' in data:
                data['confidence'] = float(data['confidence'])
            # Serialize any list/dict attributes to JSON
            for attr_key, attr_value in list(data.items()):
                if isinstance(attr_value, (list, dict)):
                    data[attr_key] = json.dumps(attr_value)

        nx.write_graphml(self.graph, filepath)

    def _save_json(self, filepath: Path):
        """Save graph in JSON format with full metadata."""
        data = {
            'graph': dict(self.graph.graph),
            'nodes': [],
            'edges': [],
            'metadata': {
                'saved_at': datetime.utcnow().isoformat(),
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
            }
        }

        # Serialize nodes
        for node, node_data in self.graph.nodes(data=True):
            node_entry = {
                'id': str(node),
                'data': {
                    k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                    for k, v in node_data.items()
                }
            }
            data['nodes'].append(node_entry)

        # Serialize edges
        for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
            edge_entry = {
                'source': str(u),
                'target': str(v),
                'key': key,
                'data': {
                    k: (json.dumps(v) if isinstance(v, list) else
                        str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
                    for k, v in edge_data.items()
                }
            }
            data['edges'].append(edge_entry)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str, format: str = 'graphml') -> nx.MultiDiGraph:
        """
        Load graph from file.

        Args:
            filepath: Input file path
            format: 'graphml' (default) or 'json'

        Returns:
            Loaded NetworkX graph
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")

        start_time = datetime.utcnow()

        try:
            if format == 'graphml':
                graph = self._load_graphml(filepath)
            elif format == 'json':
                graph = self._load_json(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            load_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Loaded graph from {filepath}: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges, {load_time:.2f}s"
            )

            self.graph = graph
            return graph

        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise

    def _load_graphml(self, filepath: Path) -> nx.MultiDiGraph:
        """Load graph from GraphML format."""
        graph = nx.read_graphml(filepath, force_multigraph=True)

        # Convert back to MultiDiGraph (read_graphml preserves multiedges)
        if not isinstance(graph, nx.MultiDiGraph):
            # Already MultiDiGraph from read_graphml with force_multigraph=True
            pass

        # Deserialize graph-level JSON attributes
        for key, value in list(graph.graph.items()):
            if isinstance(value, str) and value.startswith(('[', '{')):
                try:
                    graph.graph[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

        # Deserialize node-level JSON attributes
        for node, data in graph.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, str) and value.startswith(('[', '{')):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass

        # Deserialize edge-level JSON attributes
        for u, v, key, data in graph.edges(keys=True, data=True):
            for attr_key, attr_value in list(data.items()):
                if isinstance(attr_value, str) and attr_value.startswith(('[', '{')):
                    try:
                        data[attr_key] = json.loads(attr_value)
                    except json.JSONDecodeError:
                        pass

        return graph

    def _load_json(self, filepath: Path) -> nx.MultiDiGraph:
        """Load graph from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        graph = nx.MultiDiGraph()

        # Restore graph-level metadata
        if 'graph' in data:
            graph.graph.update(data['graph'])

        # Restore nodes
        for node_entry in data.get('nodes', []):
            node_id = node_entry['id']
            node_data = node_entry.get('data', {})
            # Convert string booleans
            if 'canonical' in node_data and isinstance(node_data['canonical'], str):
                node_data['canonical'] = node_data['canonical'].lower() == 'true'
            graph.add_node(node_id, **node_data)

        # Restore edges
        for edge_entry in data.get('edges', []):
            source = edge_entry['source']
            target = edge_entry['target']
            key = edge_entry.get('key')
            edge_data = edge_entry.get('data', {})

            # Deserialize JSON fields
            for k, v in edge_data.items():
                if isinstance(v, str) and v.startswith('['):
                    try:
                        edge_data[k] = json.loads(v)
                    except:
                        pass

            graph.add_edge(source, target, key, **edge_data)

        return graph

    def validate_roundtrip(self, filepath: str) -> bool:
        """
        Validate that graph survives save/load roundtrip.

        Args:
            filepath: Path for validation save

        Returns:
            True if roundtrip valid, False otherwise
        """
        original_nodes = self.graph.number_of_nodes()
        original_edges = self.graph.number_of_edges()

        # Save
        self.save(filepath)

        # Load
        loaded = self.load(filepath)

        # Compare
        if loaded.number_of_nodes() != original_nodes:
            logger.error(
                f"Node count mismatch: {original_nodes} vs {loaded.number_of_nodes()}"
            )
            return False

        if loaded.number_of_edges() != original_edges:
            logger.error(
                f"Edge count mismatch: {original_edges} vs {loaded.number_of_edges()}"
            )
            return False

        # Check sample edge integrity
        for u, v, key, original_data in self.graph.edges(keys=True, data=True):
            try:
                loaded_data = loaded.get_edge_data(u, v, key)
                if loaded_data is None:
                    logger.error(f"Edge lost: {u} -> {v} key {key}")
                    return False

                # Check confidence attribute
                if 'confidence' in original_data:
                    if abs(float(loaded_data.get('confidence', 0)) -
                           float(original_data['confidence'])) > 0.01:
                        logger.error(f"Data mismatch on edge {u} -> {v}")
                        return False
            except Exception as e:
                logger.error(f"Validation error on edge {u} -> {v}: {e}")
                return False

        logger.info("Roundtrip validation passed")
        return True

    def incremental_update(self, new_events, filepath: Optional[str] = None) -> Dict:
        """
        Update graph with new events (incremental).

        Args:
            new_events: List of new event dicts
            filepath: Optional path to save after update

        Returns:
            Update statistics
        """
        original_edges = self.graph.number_of_edges()

        # Add new edges
        new_edge_count = 0
        for event in new_events:
            source = event.get('source_entity')
            target = event.get('target_entity')

            if source and target:
                edge_key = self.graph.add_edge(
                    source, target,
                    timestamp=event.get('timestamp'),
                    confidence=event.get('confidence', 0.5),
                    relation_type=event.get('relation_type'),
                    **{k: v for k, v in event.items()
                       if k not in ['source_entity', 'target_entity']}
                )
                new_edge_count += 1

        # Save if path provided
        if filepath:
            self.save(filepath)

        stats = {
            'original_edges': original_edges,
            'new_edges': new_edge_count,
            'total_edges': self.graph.number_of_edges(),
            'timestamp': datetime.utcnow().isoformat(),
        }

        logger.info(f"Incremental update: +{new_edge_count} edges")
        return stats

    def get_statistics(self) -> Dict:
        """Get persistence statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'graph_metadata': dict(self.graph.graph),
        }


def create_persistence(graph: Optional[nx.MultiDiGraph] = None) -> GraphPersistence:
    """Factory function to create persistence layer."""
    return GraphPersistence(graph)
