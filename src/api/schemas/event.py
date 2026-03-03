"""
Event-related Pydantic V2 DTOs.

These DTOs define the API contract for GDELT and ACLED event data exposed
through GET /api/v1/events.  The EventDTO maps directly from the SQLite
Event dataclass via ``from_attributes=True``.
"""

from pydantic import BaseModel, ConfigDict, Field


class EventDTO(BaseModel):
    """Single event record from the unified events table (GDELT or ACLED)."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Auto-increment row ID")
    gdelt_id: str | None = Field(
        None, description="GDELT GlobalEventID or ACLED-prefixed event ID"
    )
    event_date: str = Field(..., description="Event date (YYYY-MM-DD)")
    actor1_code: str | None = Field(None, description="Primary actor CAMEO code")
    actor2_code: str | None = Field(None, description="Secondary actor CAMEO code")
    event_code: str | None = Field(None, description="CAMEO event type code")
    quad_class: int | None = Field(
        None,
        description="QuadClass: 1=Verbal Coop, 2=Material Coop, 3=Verbal Conflict, 4=Material Conflict",
    )
    goldstein_scale: float | None = Field(
        None, description="Goldstein conflict/cooperation scale (-10 to +10)"
    )
    num_mentions: int | None = Field(
        None, description="Number of source mentions (GDELT only)"
    )
    num_sources: int | None = Field(
        None, description="Number of unique sources (GDELT only)"
    )
    tone: float | None = Field(None, description="Average tone of coverage")
    url: str | None = Field(None, description="Source article URL")
    title: str | None = Field(None, description="Article headline or event description")
    country_iso: str | None = Field(
        None, description="ISO 3166-1 alpha-2 country code (event location)"
    )
    source: str = Field(
        "gdelt", description="Data source discriminator: 'gdelt' or 'acled'"
    )
