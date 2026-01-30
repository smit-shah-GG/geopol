"""
Stage implementations for bootstrap pipeline.

Each stage wraps an existing entry point from the codebase without
duplicating business logic.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Default paths
DEFAULT_RAW_DIR = Path("data/gdelt/raw")
DEFAULT_PROCESSED_DIR = Path("data/gdelt/processed")
DEFAULT_EVENTS_DB = Path("data/events.db")
DEFAULT_GRAPH_PATH = Path("data/graphs/knowledge_graph.graphml")
DEFAULT_CHROMA_DIR = Path("chroma_db")


class GDELTCollectStage:
    """
    Stage 1: Collect GDELT events.

    Wraps GDELTHistoricalCollector.collect_last_n_days().
    """

    def __init__(
        self,
        n_days: int = 30,
        quad_classes: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize collect stage.

        Args:
            n_days: Number of days to collect
            quad_classes: QuadClass values to filter (default: all)
            output_dir: Output directory for raw CSVs
        """
        self.n_days = n_days
        self.quad_classes = quad_classes or [1, 2, 3, 4]
        self.output_dir = output_dir or DEFAULT_RAW_DIR

    @property
    def name(self) -> str:
        return "collect"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GDELT collection.

        Args:
            context: Shared context dict

        Returns:
            Statistics dict
        """
        from src.training.data_collector import GDELTHistoricalCollector

        logger.info(f"Collecting {self.n_days} days of GDELT events")

        collector = GDELTHistoricalCollector(output_dir=self.output_dir)
        df = collector.collect_last_n_days(
            n_days=self.n_days,
            quad_classes=self.quad_classes,
        )

        events_count = len(df) if df is not None else 0

        # Store in context for downstream stages
        context["raw_events_count"] = events_count

        return {
            "events_collected": events_count,
            "n_days": self.n_days,
            "output_dir": str(self.output_dir),
        }

    def validate_output(self) -> Tuple[bool, str]:
        """Check at least one CSV file exists and is non-empty."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_files = list(self.output_dir.glob("gdelt_*.csv"))

        if not csv_files:
            return False, "No gdelt_*.csv files found"

        # Check at least one has content
        for csv_file in csv_files:
            if csv_file.stat().st_size > 100:  # More than just a header
                return True, f"Found {len(csv_files)} CSV files"

        return False, "All CSV files are empty"

    def get_output_path(self) -> Optional[str]:
        return str(self.output_dir)


class ProcessEventsStage:
    """
    Stage 2: Process raw events to TKG format and load into SQLite.

    Wraps GDELTDataProcessor.process_all() and bridges Parquet->SQLite.
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize process stage.

        Args:
            raw_dir: Directory with raw CSV files
            processed_dir: Directory for processed parquet
            db_path: Path to SQLite database
        """
        self.raw_dir = raw_dir or DEFAULT_RAW_DIR
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.db_path = db_path or DEFAULT_EVENTS_DB
        self._output_path = self.processed_dir / "events.parquet"

    @property
    def name(self) -> str:
        return "process"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute event processing.

        Args:
            context: Shared context dict

        Returns:
            Statistics dict
        """
        from src.training.data_processor import GDELTDataProcessor
        from src.database.storage import EventStorage
        from src.database.models import Event

        logger.info("Processing raw GDELT events to TKG format")

        # Run processor
        processor = GDELTDataProcessor(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )
        tkg_df = processor.process_all()

        events_processed = len(tkg_df)

        # Bridge Parquet -> SQLite
        logger.info(f"Loading {events_processed} events into SQLite")
        storage = EventStorage(db_path=str(self.db_path))

        events = []
        for _, row in tkg_df.iterrows():
            # Generate content hash from key fields
            content_str = (
                f"{row.get('timestamp', '')}"
                f"{row.get('entity1', '')}"
                f"{row.get('entity2', '')}"
                f"{row.get('EventCode', '')}"
            )
            content_hash = hashlib.md5(content_str.encode()).hexdigest()

            # Extract event date as string
            timestamp = row.get("timestamp")
            if hasattr(timestamp, "strftime"):
                event_date = timestamp.strftime("%Y-%m-%d")
                time_window = timestamp.strftime("%Y-%m")
            else:
                event_date = str(timestamp)[:10] if timestamp else ""
                time_window = str(timestamp)[:7] if timestamp else ""

            # Create Event from parquet row
            event = Event(
                content_hash=content_hash,
                time_window=time_window,
                event_date=event_date,
                gdelt_id=str(row.get("GLOBALEVENTID", "")),
                actor1_code=row.get("entity1"),
                actor2_code=row.get("entity2"),
                event_code=str(row.get("EventCode", "")),
                quad_class=int(row.get("QuadClass")) if pd.notna(row.get("QuadClass")) else None,
                goldstein_scale=float(row.get("GoldsteinScale")) if pd.notna(row.get("GoldsteinScale")) else None,
                num_mentions=int(row.get("NumMentions")) if pd.notna(row.get("NumMentions")) else None,
                tone=float(row.get("AvgTone")) if pd.notna(row.get("AvgTone")) else None,
            )
            events.append(event)

        inserted = storage.insert_events(events)

        context["events_processed"] = events_processed
        context["events_inserted"] = inserted
        context["db_path"] = str(self.db_path)

        return {
            "events_processed": events_processed,
            "events_inserted": inserted,
            "parquet_path": str(self._output_path),
            "db_path": str(self.db_path),
        }

    def validate_output(self) -> Tuple[bool, str]:
        """Check parquet exists and has rows."""
        if not self._output_path.exists():
            return False, f"Parquet file not found: {self._output_path}"

        try:
            df = pd.read_parquet(self._output_path)
            if len(df) == 0:
                return False, "Parquet file is empty"
            return True, f"Parquet has {len(df)} rows"
        except Exception as e:
            return False, f"Failed to read parquet: {e}"

    def get_output_path(self) -> Optional[str]:
        return str(self._output_path)


class BuildGraphStage:
    """
    Stage 3: Build temporal knowledge graph from SQLite events.

    Wraps TemporalKnowledgeGraph.add_events_batch().
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        batch_size: int = 1000,
        limit: Optional[int] = None,
    ):
        """
        Initialize graph build stage.

        Args:
            db_path: Path to SQLite database
            batch_size: Batch size for processing
            limit: Maximum events to process (None for all)
        """
        self.db_path = db_path or DEFAULT_EVENTS_DB
        self.batch_size = batch_size
        self.limit = limit
        self._graph = None

    @property
    def name(self) -> str:
        return "graph"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute graph building.

        Args:
            context: Shared context dict

        Returns:
            Statistics dict
        """
        from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph

        # Use db_path from context if available (from previous stage)
        db_path = context.get("db_path", str(self.db_path))

        logger.info(f"Building temporal knowledge graph from {db_path}")

        graph = TemporalKnowledgeGraph()
        stats = graph.add_events_batch(
            db_path=db_path,
            batch_size=self.batch_size,
            limit=self.limit,
        )

        self._graph = graph

        # Store graph in context for subsequent stages
        context["graph"] = graph

        return {
            "nodes": graph.graph.number_of_nodes(),
            "edges": graph.graph.number_of_edges(),
            "valid_events": stats.get("valid_events", 0),
            "invalid_events": stats.get("invalid_events", 0),
        }

    def validate_output(self) -> Tuple[bool, str]:
        """Check graph has nodes and edges."""
        if self._graph is None:
            return False, "Graph not built"

        nodes = self._graph.graph.number_of_nodes()
        edges = self._graph.graph.number_of_edges()

        if nodes == 0:
            return False, "Graph has no nodes"
        if edges == 0:
            return False, "Graph has no edges"

        return True, f"Graph has {nodes} nodes, {edges} edges"

    def get_output_path(self) -> Optional[str]:
        # In-memory graph, no persistent path yet
        return None


class PersistGraphStage:
    """
    Stage 4: Persist graph to GraphML file.

    Wraps GraphPersistence.save().
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        format: str = "graphml",
    ):
        """
        Initialize persist stage.

        Args:
            output_path: Path for GraphML output
            format: Output format ('graphml' or 'json')
        """
        self.output_path = output_path or DEFAULT_GRAPH_PATH
        self.format = format

    @property
    def name(self) -> str:
        return "persist"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute graph persistence.

        Args:
            context: Shared context dict

        Returns:
            Statistics dict
        """
        from src.knowledge_graph.persistence import GraphPersistence

        graph = context.get("graph")
        if graph is None:
            raise RuntimeError("No graph in context - BuildGraphStage must run first")

        logger.info(f"Persisting graph to {self.output_path}")

        persistence = GraphPersistence(graph.graph)
        stats = persistence.save(str(self.output_path), format=self.format)

        return {
            "nodes": stats.get("nodes", 0),
            "edges": stats.get("edges", 0),
            "file_size_mb": round(stats.get("file_size_mb", 0), 2),
            "output_path": str(self.output_path),
        }

    def validate_output(self) -> Tuple[bool, str]:
        """Check GraphML file exists and has content."""
        path = Path(self.output_path)
        if not path.exists():
            return False, f"GraphML file not found: {path}"

        if path.stat().st_size == 0:
            return False, "GraphML file is empty"

        # Try to count nodes in file
        try:
            import networkx as nx
            graph = nx.read_graphml(str(path), force_multigraph=True)
            if graph.number_of_nodes() == 0:
                return False, "GraphML has no nodes"
            return True, f"GraphML has {graph.number_of_nodes()} nodes"
        except Exception as e:
            return False, f"Failed to read GraphML: {e}"

    def get_output_path(self) -> Optional[str]:
        return str(self.output_path)


class IndexRAGStage:
    """
    Stage 5: Index graph patterns in RAG store.

    Wraps RAGPipeline.index_graph_patterns().
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        rebuild: bool = True,
        time_window_days: int = 30,
        min_pattern_size: int = 3,
    ):
        """
        Initialize RAG index stage.

        Args:
            persist_dir: ChromaDB persistence directory
            rebuild: Whether to rebuild index from scratch
            time_window_days: Days to group events in patterns
            min_pattern_size: Minimum edges in pattern
        """
        self.persist_dir = persist_dir or DEFAULT_CHROMA_DIR
        self.rebuild = rebuild
        self.time_window_days = time_window_days
        self.min_pattern_size = min_pattern_size

    @property
    def name(self) -> str:
        return "index"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute RAG indexing.

        Args:
            context: Shared context dict

        Returns:
            Statistics dict
        """
        from src.forecasting.rag_pipeline import RAGPipeline

        graph = context.get("graph")
        if graph is None:
            raise RuntimeError("No graph in context - BuildGraphStage must run first")

        logger.info(f"Indexing graph patterns to {self.persist_dir}")

        pipeline = RAGPipeline(persist_dir=str(self.persist_dir))
        stats = pipeline.index_graph_patterns(
            graph=graph,
            time_window_days=self.time_window_days,
            min_pattern_size=self.min_pattern_size,
            rebuild=self.rebuild,
        )

        return {
            "patterns_extracted": stats.get("patterns_extracted", 0),
            "documents_indexed": stats.get("documents_indexed", 0),
            "persist_dir": str(self.persist_dir),
        }

    def validate_output(self) -> Tuple[bool, str]:
        """Check ChromaDB collection exists with documents."""
        persist_path = Path(self.persist_dir)
        if not persist_path.exists():
            return False, f"ChromaDB directory not found: {persist_path}"

        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(persist_path))
            collection = client.get_collection("graph_patterns")
            count = collection.count()
            if count == 0:
                return False, "ChromaDB collection is empty"
            return True, f"ChromaDB has {count} documents"
        except Exception as e:
            return False, f"Failed to check ChromaDB: {e}"

    def get_output_path(self) -> Optional[str]:
        return str(self.persist_dir)


def create_all_stages(
    n_days: int = 30,
    quad_classes: Optional[List[int]] = None,
) -> List:
    """
    Create all bootstrap stages with default configuration.

    Args:
        n_days: Days of GDELT data to collect
        quad_classes: QuadClass filter for collection

    Returns:
        List of stage instances in execution order
    """
    return [
        GDELTCollectStage(n_days=n_days, quad_classes=quad_classes),
        ProcessEventsStage(),
        BuildGraphStage(),
        PersistGraphStage(),
        IndexRAGStage(),
    ]
