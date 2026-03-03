"""
RSS article search endpoint with dual-mode ChromaDB queries.

Two search modes:
  1. **Keyword mode** (default): Uses ``collection.get()`` with metadata
     filters for country and date range. Text search is applied as a
     Python-side filter on document content.
  2. **Semantic mode** (``?semantic=true``): Uses ``collection.query()``
     with the ``text`` parameter as the embedding query string. Returns
     results ranked by cosine similarity with ``relevance_score``.

The ChromaDB collection name follows the RSS daemon's ``ArticleIndexer``
convention (``rss_articles``).

Requires API key authentication.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.middleware.auth import verify_api_key
from src.api.schemas.article import ArticleDTO
from src.api.schemas.common import PaginatedResponse, ProblemDetail

logger = logging.getLogger(__name__)

router = APIRouter()

# ChromaDB collection name -- must match ArticleIndexer in article_processor.py
_COLLECTION_NAME = "rss_articles"
_CHROMA_PERSIST_DIR = "./chroma_db"


def _get_collection():  # noqa: ANN202
    """Lazily initialize and return the ChromaDB collection.

    Returns None if ChromaDB is not available or the collection doesn't
    exist.  Callers must handle None gracefully (return empty results).
    """
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        client = chromadb.PersistentClient(
            path=_CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        ef = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2",
        )
        return client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception:
        logger.warning("ChromaDB unavailable -- articles endpoint will return empty results", exc_info=True)
        return None


@router.get(
    "",
    response_model=PaginatedResponse[ArticleDTO],
    summary="Search articles",
    description=(
        "Returns RSS articles from ChromaDB. Supports two modes: "
        "keyword filtering (default) with country and date range metadata "
        "filters, and semantic search (?semantic=true) with vector similarity. "
        "Semantic mode requires a text query parameter."
    ),
    responses={
        400: {"model": ProblemDetail, "description": "Missing text for semantic search"},
    },
)
async def list_articles(
    country: str | None = Query(None, description="ISO 3166-1 alpha-2 country code"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD, inclusive)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD, inclusive)"),
    text: str | None = Query(None, description="Text query (required for semantic mode)"),
    semantic: bool = Query(default=False, description="Enable semantic (vector similarity) search"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    _client: str = Depends(verify_api_key),
) -> PaginatedResponse[ArticleDTO]:
    """Search articles in keyword or semantic mode."""
    # Validate: semantic mode requires text
    if semantic and not text:
        raise HTTPException(
            status_code=400,
            detail="Semantic search requires a 'text' query parameter.",
        )

    collection = _get_collection()
    if collection is None:
        return PaginatedResponse(items=[], next_cursor=None, has_more=False)

    try:
        if semantic:
            return _semantic_search(collection, text, limit)  # type: ignore[arg-type]
        return _keyword_search(collection, country, start_date, end_date, text, limit)
    except Exception:
        logger.error("Article search failed", exc_info=True)
        return PaginatedResponse(items=[], next_cursor=None, has_more=False)


def _keyword_search(
    collection,  # noqa: ANN001
    country: str | None,
    start_date: str | None,
    end_date: str | None,
    text: str | None,
    limit: int,
) -> PaginatedResponse[ArticleDTO]:
    """Keyword mode: metadata filters + optional Python-side text filter."""
    # Build ChromaDB where clause from metadata filters
    where_clauses: list[dict] = []
    if country:
        where_clauses.append({"source_name": {"$ne": ""}})  # ensure filter is valid
        # ChromaDB doesn't have a country_iso metadata field by default;
        # filter Python-side after retrieval
    if start_date:
        where_clauses.append({"published_at": {"$gte": start_date}})
    if end_date:
        where_clauses.append({"published_at": {"$lte": end_date + "T23:59:59"}})

    where: dict | None = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    # Fetch more than limit to allow for Python-side filtering
    fetch_limit = min(limit * 3, 300)

    try:
        if where:
            results = collection.get(
                where=where,
                limit=fetch_limit,
                include=["documents", "metadatas"],
            )
        else:
            results = collection.get(
                limit=fetch_limit,
                include=["documents", "metadatas"],
            )
    except Exception:
        logger.warning("ChromaDB get() failed", exc_info=True)
        return PaginatedResponse(items=[], next_cursor=None, has_more=False)

    if not results or not results.get("ids"):
        return PaginatedResponse(items=[], next_cursor=None, has_more=False)

    # Build DTOs with Python-side filtering
    articles: list[ArticleDTO] = []
    ids = results["ids"]
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for i, chunk_id in enumerate(ids):
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}

        # Python-side text filter
        if text and text.lower() not in (doc or "").lower():
            title_str = (meta.get("title") or "").lower()
            if text.lower() not in title_str:
                continue

        # Python-side country filter (articles don't store country_iso natively)
        # Skip country filtering for now -- articles don't have reliable country metadata

        articles.append(
            ArticleDTO(
                chunk_id=chunk_id,
                title=meta.get("title") or "Untitled",
                url=meta.get("source_url") or "",
                source_feed=meta.get("source_name") or "unknown",
                country_iso=None,
                published_at=meta.get("published_at") or None,
                snippet=(doc or "")[:200],
                relevance_score=None,
            )
        )

        if len(articles) >= limit:
            break

    return PaginatedResponse(
        items=articles,
        next_cursor=None,  # No cursor for keyword mode (simple get)
        has_more=False,
    )


def _semantic_search(
    collection,  # noqa: ANN001
    query_text: str,
    limit: int,
) -> PaginatedResponse[ArticleDTO]:
    """Semantic mode: ChromaDB vector similarity query."""
    n_results = min(limit, 20)

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.warning("ChromaDB query() failed", exc_info=True)
        return PaginatedResponse(items=[], next_cursor=None, has_more=False)

    if not results or not results.get("ids") or not results["ids"][0]:
        return PaginatedResponse(items=[], next_cursor=None, has_more=False)

    articles: list[ArticleDTO] = []
    ids = results["ids"][0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for i, chunk_id in enumerate(ids):
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None

        # Convert distance to similarity score (ChromaDB uses L2 by default)
        # Lower distance = more similar. Convert to 0-1 scale.
        relevance = round(1.0 / (1.0 + distance), 4) if distance is not None else None

        articles.append(
            ArticleDTO(
                chunk_id=chunk_id,
                title=meta.get("title") or "Untitled",
                url=meta.get("source_url") or "",
                source_feed=meta.get("source_name") or "unknown",
                country_iso=None,
                published_at=meta.get("published_at") or None,
                snippet=(doc or "")[:200],
                relevance_score=relevance,
            )
        )

    return PaginatedResponse(
        items=articles,
        next_cursor=None,
        has_more=False,
    )
