"""
Full-text search DTOs for the forecast search endpoint.

SearchResult wraps ForecastResponse with a ts_rank relevance score.
SearchResponse is the top-level response for GET /api/v1/forecasts/search,
including total count and a reserved nullable ``suggestions`` field for
future LLM-powered search suggestions (always None for now).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas.forecast import ForecastResponse


class SearchResult(BaseModel):
    """A forecast search result with relevance ranking."""

    model_config = ConfigDict(from_attributes=True)

    forecast: ForecastResponse = Field(
        ..., description="The matched forecast"
    )
    relevance: float = Field(
        ..., ge=0.0, description="ts_rank relevance score"
    )


class SearchResponse(BaseModel):
    """Full-text search response with result metadata."""

    model_config = ConfigDict(from_attributes=True)

    results: list[SearchResult] = Field(
        default_factory=list, description="Ranked search results"
    )
    total: int = Field(
        ..., ge=0, description="Total matching results (before limit)"
    )
    query: str = Field(
        ..., description="The search query that was executed"
    )
    suggestions: Optional[list[str]] = Field(
        default=None,
        description=(
            "LLM-generated search suggestions on empty results "
            "(not yet implemented, reserved for future use)"
        ),
    )
