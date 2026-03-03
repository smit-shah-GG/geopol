"""
Data source health status Pydantic V2 DTOs.

These DTOs define the API contract for GET /api/v1/sources, which returns
the operational status of each ingestion pipeline.  The SourcesPanel on
the frontend auto-discovers sources from this endpoint -- adding a new
backend source automatically appears in the UI without frontend changes.
"""

from pydantic import BaseModel, Field


class SourceStatusDTO(BaseModel):
    """Operational status of a single data source ingestion pipeline."""

    name: str = Field(
        ...,
        description="Source identifier: 'gdelt', 'rss', 'acled', 'advisory'",
    )
    healthy: bool = Field(
        ..., description="Whether the last run completed successfully"
    )
    last_update: str | None = Field(
        None, description="ISO 8601 timestamp of last successful update"
    )
    events_last_run: int = Field(
        ..., description="Number of new events/items ingested in the last run"
    )
    detail: str = Field(
        ..., description="Human-readable status detail string"
    )
