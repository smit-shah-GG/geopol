"""
Output validation functions for bootstrap pipeline stages.

Each validator implements dual idempotency: check output exists AND is valid
(not empty/corrupted). Validators return (is_valid, reason) tuples and never
raise exceptions - failures are communicated via the return value.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def validate_gdelt_output(output_dir: Path) -> Tuple[bool, str]:
    """
    Validate GDELT collection output.

    Checks:
    1. Directory exists
    2. At least one gdelt_*.csv file exists
    3. At least one CSV has > 0 bytes

    Args:
        output_dir: Directory containing GDELT CSV files

    Returns:
        Tuple of (is_valid, reason_message)
    """
    try:
        output_dir = Path(output_dir)

        if not output_dir.exists():
            return False, f"Directory does not exist: {output_dir}"

        if not output_dir.is_dir():
            return False, f"Path is not a directory: {output_dir}"

        csv_files = list(output_dir.glob("gdelt_*.csv"))

        if not csv_files:
            return False, f"No gdelt_*.csv files found in {output_dir}"

        # Check at least one CSV has content (more than just a header)
        for csv_file in csv_files:
            try:
                size = csv_file.stat().st_size
                if size > 100:  # More than just a header line
                    logger.debug(f"GDELT validation passed: {len(csv_files)} CSV files found")
                    return True, f"Found {len(csv_files)} CSV files with content"
            except OSError as e:
                logger.debug(f"Could not stat {csv_file}: {e}")
                continue

        return False, "All CSV files are empty or contain only headers"

    except Exception as e:
        logger.debug(f"GDELT validation failed with exception: {e}")
        return False, f"Validation error: {e}"


def validate_processed_output(parquet_path: Path, sqlite_path: Path) -> Tuple[bool, str]:
    """
    Validate processed events output.

    Checks:
    1. Parquet file exists and is readable
    2. Parquet has > 0 rows
    3. SQLite database exists
    4. SQLite events table has > 0 rows

    Args:
        parquet_path: Path to events.parquet file
        sqlite_path: Path to SQLite database

    Returns:
        Tuple of (is_valid, reason_message)
    """
    try:
        parquet_path = Path(parquet_path)
        sqlite_path = Path(sqlite_path)

        # Check parquet file
        if not parquet_path.exists():
            return False, f"Parquet file not found: {parquet_path}"

        # Lazy import to avoid import-time failures
        import pandas as pd

        try:
            df = pd.read_parquet(parquet_path)
            parquet_rows = len(df)
            if parquet_rows == 0:
                return False, "Parquet file is empty (0 rows)"
        except Exception as e:
            return False, f"Failed to read parquet file: {e}"

        # Check SQLite database
        if not sqlite_path.exists():
            return False, f"SQLite database not found: {sqlite_path}"

        import sqlite3

        try:
            # Use short timeout to avoid blocking on locked database
            conn = sqlite3.connect(str(sqlite_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            sqlite_rows = cursor.fetchone()[0]
            conn.close()

            if sqlite_rows == 0:
                return False, "SQLite events table is empty (0 rows)"

        except sqlite3.OperationalError as e:
            return False, f"SQLite error (possibly locked or missing table): {e}"
        except Exception as e:
            return False, f"Failed to query SQLite: {e}"

        logger.debug(
            f"Processed output validation passed: "
            f"parquet={parquet_rows} rows, sqlite={sqlite_rows} rows"
        )
        return True, f"Parquet has {parquet_rows} rows, SQLite has {sqlite_rows} rows"

    except Exception as e:
        logger.debug(f"Processed output validation failed with exception: {e}")
        return False, f"Validation error: {e}"


def validate_graph_output(graphml_path: Path) -> Tuple[bool, str]:
    """
    Validate persisted graph output.

    Checks:
    1. GraphML file exists
    2. File size > 0
    3. Can be loaded by networkx (basic parse test)
    4. Loaded graph has > 0 nodes

    Args:
        graphml_path: Path to GraphML file

    Returns:
        Tuple of (is_valid, reason_message)
    """
    try:
        graphml_path = Path(graphml_path)

        if not graphml_path.exists():
            return False, f"GraphML file not found: {graphml_path}"

        try:
            file_size = graphml_path.stat().st_size
        except OSError as e:
            return False, f"Cannot stat file: {e}"

        if file_size == 0:
            return False, "GraphML file is empty (0 bytes)"

        # Lazy import
        import networkx as nx

        try:
            # Use force_multigraph=True for compatibility with multigraphs
            graph = nx.read_graphml(str(graphml_path), force_multigraph=True)
        except Exception as e:
            return False, f"Failed to parse GraphML: {e}"

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        if node_count == 0:
            return False, "GraphML file contains a graph with 0 nodes"

        logger.debug(
            f"Graph output validation passed: {node_count} nodes, {edge_count} edges"
        )
        return True, f"Graph has {node_count} nodes, {edge_count} edges"

    except Exception as e:
        logger.debug(f"Graph output validation failed with exception: {e}")
        return False, f"Validation error: {e}"


def validate_rag_output(
    chroma_dir: Path, collection_name: str = "graph_patterns"
) -> Tuple[bool, str]:
    """
    Validate RAG index output.

    Checks:
    1. ChromaDB directory exists
    2. Collection exists
    3. Collection has > 0 documents

    Args:
        chroma_dir: Path to ChromaDB persistence directory
        collection_name: Name of the collection to check

    Returns:
        Tuple of (is_valid, reason_message)
    """
    try:
        chroma_dir = Path(chroma_dir)

        if not chroma_dir.exists():
            return False, f"ChromaDB directory not found: {chroma_dir}"

        if not chroma_dir.is_dir():
            return False, f"ChromaDB path is not a directory: {chroma_dir}"

        # Lazy import
        import chromadb

        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
        except Exception as e:
            return False, f"Failed to open ChromaDB: {e}"

        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            return False, f"Collection '{collection_name}' not found: {e}"

        try:
            doc_count = collection.count()
        except Exception as e:
            return False, f"Failed to count documents in collection: {e}"

        if doc_count == 0:
            return False, f"Collection '{collection_name}' is empty (0 documents)"

        logger.debug(
            f"RAG output validation passed: {doc_count} documents in '{collection_name}'"
        )
        return True, f"Collection '{collection_name}' has {doc_count} documents"

    except Exception as e:
        logger.debug(f"RAG output validation failed with exception: {e}")
        return False, f"Validation error: {e}"
