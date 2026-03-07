"""
Ephemeral ChromaDB temporal index builder for look-ahead bias prevention.

Creates an in-memory ChromaDB collection containing only article chunks
published on or before a given cutoff date. This guarantees the RAG
pipeline cannot access future information during backtesting.

The source collection (persistent, on-disk) is opened read-only. All
qualifying chunks are copied to an ephemeral in-memory collection via
paginated get() calls with Python-side date filtering (not relying on
ChromaDB string $lte, which breaks on inconsistent date formats).

No disk caching: each call builds a fresh index. This is correct for
an on-demand investigative tool where stale indexes are unacceptable.
"""

from __future__ import annotations

import logging
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# Must match the embedding model used by ArticleIndexer and RAGPipeline.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ChromaDB get() batch size for paginated reads.
BATCH_SIZE = 500


def _parse_published_at(raw: str) -> datetime | None:
    """Parse a published_at metadata string into a datetime.

    Handles ISO 8601 variants:
    - "2026-03-01T14:22:33Z"
    - "2026-03-01T14:22:33+00:00"
    - "2026-03-01"  (date only)

    Returns None on parse failure.
    """
    if not raw:
        return None

    # Normalize the common "Z" suffix to "+00:00" for fromisoformat.
    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        pass

    # Try date-only format.
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def build_temporal_chromadb_index(
    source_persist_dir: str,
    source_collection_name: str,
    cutoff_date: datetime,
) -> chromadb.Collection:
    """Build an ephemeral in-memory ChromaDB collection with articles <= cutoff_date.

    Opens the persistent ChromaDB source, creates an ephemeral in-memory
    client, and copies all chunks whose published_at metadata parses to
    a datetime on or before cutoff_date. Chunks with empty, missing, or
    unparseable published_at are excluded (cannot verify temporal validity).

    Date comparison is done in Python via datetime.fromisoformat() rather
    than relying on ChromaDB's string $lte operator, which fails on
    inconsistent date formats across RSS feeds.

    Args:
        source_persist_dir: Path to the persistent ChromaDB storage directory.
        source_collection_name: Name of the source collection (e.g. "rss_articles").
        cutoff_date: Only include articles published on or before this date.
            If timezone-naive, treated as UTC for comparison purposes.

    Returns:
        An in-memory ChromaDB Collection ready for RAG queries, with the
        same embedding function as the source.

    Raises:
        ValueError: If the source collection doesn't exist.
    """
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Open source (persistent, read-only access).
    source_client = chromadb.PersistentClient(path=source_persist_dir)
    source_col = source_client.get_collection(
        name=source_collection_name,
        embedding_function=ef,
    )

    # Create ephemeral target (in-memory, no disk persistence).
    temp_client = chromadb.Client()
    collection_name = f"temporal_{cutoff_date.strftime('%Y%m%d_%H%M%S')}"
    temp_col = temp_client.create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    # Strip timezone from cutoff for comparison if needed. We compare
    # as naive datetimes to handle the mixed-timezone metadata case.
    cutoff_naive = cutoff_date.replace(tzinfo=None) if cutoff_date.tzinfo else cutoff_date

    offset = 0
    total_copied = 0
    total_skipped_no_date = 0
    total_skipped_future = 0

    while True:
        # Paginated fetch of ALL chunks (no where filter -- we filter in Python).
        batch = source_col.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        if not batch["ids"]:
            break

        # Filter by published_at in Python.
        valid_ids: list[str] = []
        valid_docs: list[str] = []
        valid_metas: list[dict] = []
        valid_embeds: list[list[float]] = []

        for i, meta in enumerate(batch["metadatas"]):
            raw_date = meta.get("published_at", "")
            if not raw_date:
                total_skipped_no_date += 1
                continue

            parsed = _parse_published_at(raw_date)
            if parsed is None:
                total_skipped_no_date += 1
                continue

            # Compare as naive datetimes.
            parsed_naive = parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
            if parsed_naive > cutoff_naive:
                total_skipped_future += 1
                continue

            valid_ids.append(batch["ids"][i])
            if batch["documents"]:
                valid_docs.append(batch["documents"][i])
            valid_metas.append(batch["metadatas"][i])
            if batch["embeddings"]:
                valid_embeds.append(batch["embeddings"][i])

        if valid_ids:
            add_kwargs: dict = {
                "ids": valid_ids,
                "metadatas": valid_metas,
            }
            if valid_docs:
                add_kwargs["documents"] = valid_docs
            if valid_embeds:
                add_kwargs["embeddings"] = valid_embeds

            temp_col.add(**add_kwargs)
            total_copied += len(valid_ids)

        offset += BATCH_SIZE

    logger.info(
        "Temporal index built: %d chunks copied (cutoff=%s, skipped_no_date=%d, "
        "skipped_future=%d)",
        total_copied,
        cutoff_date.isoformat(),
        total_skipped_no_date,
        total_skipped_future,
    )

    return temp_col
