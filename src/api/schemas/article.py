"""
Article-related Pydantic V2 DTOs.

These DTOs define the API contract for RSS article data exposed through
GET /api/v1/articles.  Articles are sourced from ChromaDB (vector store)
and represent ingested RSS feed content with optional semantic search scores.
"""

from pydantic import BaseModel, Field


class ArticleDTO(BaseModel):
    """Single article record from the RSS ChromaDB collection."""

    chunk_id: str = Field(..., description="ChromaDB chunk identifier")
    title: str = Field(..., description="Article headline")
    url: str = Field(..., description="Original article URL")
    source_feed: str = Field(..., description="RSS feed name that sourced this article")
    country_iso: str | None = Field(
        None, description="ISO 3166-1 alpha-2 country code (if geo-tagged)"
    )
    published_at: str | None = Field(
        None, description="Publication timestamp (ISO 8601)"
    )
    snippet: str = Field(
        ..., description="First ~200 characters of article content"
    )
    relevance_score: float | None = Field(
        None,
        description="Cosine similarity score (only set when semantic search used)",
    )
