"""
Article text extraction, semantic paragraph chunking, and ChromaDB indexing.

Pipeline: URL -> trafilatura extraction -> paragraph-boundary chunking
-> embedding via all-mpnet-base-v2 -> ChromaDB "rss_articles" collection.

Uses the same embedding model as the existing graph_patterns collection
so that RAG queries at forecast time can search both collections with
a single embedding call.
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# Embedding model -- must match graph_patterns collection in rag_pipeline.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Chunking parameters
MIN_CHUNK_LENGTH = 80  # characters -- skip trivial fragments
MAX_CHUNK_LENGTH = 2000  # characters -- split oversized paragraphs on sentence boundary
TARGET_CHUNK_LENGTH = 800  # characters -- preferred chunk size for embedding quality

# Collection name in ChromaDB
RSS_COLLECTION_NAME = "rss_articles"


@dataclass
class ArticleChunk:
    """A single chunk of article text with metadata."""

    text: str
    chunk_index: int
    source_url: str
    source_name: str
    published_at: Optional[str] = None  # ISO 8601
    title: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of article text extraction."""

    url: str
    title: Optional[str]
    text: Optional[str]
    published_at: Optional[str]
    success: bool
    error: Optional[str] = None


def extract_article_text(url: str, *, timeout: int = 30) -> ExtractionResult:
    """
    Extract article text via trafilatura.

    Uses favor_precision=True to reduce boilerplate extraction,
    and deduplicate=True to collapse repeated paragraphs.
    """
    try:
        import trafilatura
        from trafilatura.settings import use_config

        config = use_config()
        config.set("DEFAULT", "MIN_OUTPUT_SIZE", "200")

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return ExtractionResult(
                url=url, title=None, text=None, published_at=None,
                success=False, error="fetch_url returned None",
            )

        text = trafilatura.extract(
            downloaded,
            favor_precision=True,
            deduplicate=True,
            include_comments=False,
            include_tables=False,
            config=config,
        )

        metadata = trafilatura.extract(
            downloaded,
            output_format="json",
            favor_precision=True,
            deduplicate=True,
            config=config,
        )

        title: Optional[str] = None
        published_at: Optional[str] = None
        if metadata:
            import json
            try:
                meta_dict = json.loads(metadata)
                title = meta_dict.get("title")
                published_at = meta_dict.get("date")
            except (json.JSONDecodeError, TypeError):
                pass

        if not text or len(text.strip()) < 100:
            return ExtractionResult(
                url=url, title=title, text=None, published_at=published_at,
                success=False, error="Extracted text too short or empty",
            )

        return ExtractionResult(
            url=url, title=title, text=text.strip(),
            published_at=published_at, success=True,
        )

    except Exception as exc:
        logger.warning("trafilatura extraction failed for %s: %s", url, exc)
        return ExtractionResult(
            url=url, title=None, text=None, published_at=None,
            success=False, error=str(exc),
        )


def _split_on_sentences(text: str, max_length: int) -> list[str]:
    """Split text on sentence boundaries, respecting max_length."""
    # Sentence-ending pattern: period/question/exclamation followed by space or end
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If single sentence exceeds max, hard-wrap it
        if len(sentence) > max_length:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for wrapped in textwrap.wrap(sentence, width=max_length):
                chunks.append(wrapped)
            continue

        if current_len + len(sentence) + 1 > max_length and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_article(
    text: str,
    *,
    min_length: int = MIN_CHUNK_LENGTH,
    max_length: int = MAX_CHUNK_LENGTH,
) -> list[str]:
    """
    Chunk article text on paragraph boundaries.

    Strategy:
      1. Split on double-newline (paragraph break).
      2. Merge short consecutive paragraphs up to target size.
      3. Split oversized paragraphs on sentence boundaries.
      4. Drop chunks below min_length threshold.
    """
    # Normalize whitespace: collapse multiple newlines to double
    normalized = re.sub(r'\n{3,}', '\n\n', text.strip())
    paragraphs = [p.strip() for p in normalized.split('\n\n') if p.strip()]

    if not paragraphs:
        return []

    # Phase 1: merge small paragraphs, split large ones
    processed: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for para in paragraphs:
        if len(para) > max_length:
            # Flush buffer first
            if buffer:
                processed.append(" ".join(buffer))
                buffer = []
                buffer_len = 0
            # Split oversized paragraph on sentences
            processed.extend(_split_on_sentences(para, max_length))
        elif buffer_len + len(para) + 1 > TARGET_CHUNK_LENGTH and buffer:
            # Buffer would exceed target -- flush it
            processed.append(" ".join(buffer))
            buffer = [para]
            buffer_len = len(para)
        else:
            buffer.append(para)
            buffer_len += len(para) + 1

    if buffer:
        processed.append(" ".join(buffer))

    # Phase 2: filter by minimum length
    return [c for c in processed if len(c) >= min_length]


def _chunk_id(url: str, chunk_index: int) -> str:
    """Deterministic chunk ID for idempotent upserts."""
    raw = f"{url}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class IndexingStats:
    """Per-article indexing statistics."""

    url: str
    chunks_indexed: int = 0
    skipped_duplicate: bool = False
    error: Optional[str] = None


class ArticleIndexer:
    """
    Manages a ChromaDB collection for RSS article chunks.

    Uses ChromaDB's built-in sentence-transformers embedding function
    (all-mpnet-base-v2) so embeddings are computed at add() time.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = RSS_COLLECTION_NAME,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            # ChromaDB's default embedding function uses all-MiniLM-L6-v2.
            # We override to use all-mpnet-base-v2 for consistency with
            # the existing graph_patterns collection.
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            ef = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
            )
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ef,
                metadata={"description": "RSS article chunks for RAG enrichment"},
            )
        return self._collection

    def is_url_indexed(self, url: str) -> bool:
        """Check if any chunks from this URL already exist."""
        try:
            results = self.collection.get(
                where={"source_url": url},
                limit=1,
            )
            return bool(results and results["ids"])
        except Exception:
            return False

    def index_article(
        self,
        url: str,
        source_name: str,
        text: str,
        title: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> IndexingStats:
        """
        Chunk and index an article. Skips if URL already indexed (dedup).

        Returns indexing statistics for the caller to aggregate.
        """
        stats = IndexingStats(url=url)

        # Dedup check: skip if URL already present
        if self.is_url_indexed(url):
            stats.skipped_duplicate = True
            return stats

        chunks = chunk_article(text)
        if not chunks:
            stats.error = "No chunks produced after splitting"
            return stats

        indexed_at = datetime.now(timezone.utc).isoformat()

        try:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, str]] = []

            for i, chunk_text in enumerate(chunks):
                chunk_id = _chunk_id(url, i)
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append({
                    "source_url": url,
                    "source_name": source_name,
                    "title": title or "",
                    "published_at": published_at or "",
                    "indexed_at": indexed_at,
                    "chunk_index": str(i),
                })

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            stats.chunks_indexed = len(chunks)

        except Exception as exc:
            stats.error = str(exc)
            logger.error("Failed to index article %s: %s", url, exc)

        return stats

    def prune_old_articles(self, retention_days: int = 90) -> int:
        """
        Remove article chunks older than retention_days.

        Returns the number of chunks deleted.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat()

        try:
            # Query for old chunks via indexed_at metadata
            old_results = self.collection.get(
                where={"indexed_at": {"$lt": cutoff}},
            )

            if not old_results or not old_results["ids"]:
                return 0

            old_ids = old_results["ids"]
            # ChromaDB delete supports batch by IDs
            self.collection.delete(ids=old_ids)
            logger.info("Pruned %d article chunks older than %d days", len(old_ids), retention_days)
            return len(old_ids)

        except Exception as exc:
            logger.error("Pruning failed: %s", exc)
            return 0

    def get_collection_stats(self) -> dict[str, int]:
        """Return basic collection statistics."""
        try:
            count = self.collection.count()
            return {"total_chunks": count}
        except Exception:
            return {"total_chunks": 0}
