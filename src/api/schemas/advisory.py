"""
Government travel advisory Pydantic V2 DTOs.

These DTOs define the API contract for travel advisory data exposed through
GET /api/v1/advisories.  Advisories originate from the US State Department
and UK FCDO, normalised to a common 1-4 risk level scale.
"""

from pydantic import BaseModel, Field


class AdvisoryDTO(BaseModel):
    """Single government travel advisory record."""

    source: str = Field(
        ...,
        description="Advisory source: 'us_state_dept' or 'uk_fcdo'",
    )
    country_iso: str | None = Field(
        None, description="ISO 3166-1 alpha-2 country code"
    )
    level: int = Field(
        ...,
        ge=1,
        le=4,
        description="Risk level (normalised): 1=Normal, 2=Increased Caution, 3=Reconsider, 4=Do Not Travel",
    )
    level_description: str = Field(
        ..., description="Human-readable risk level label"
    )
    title: str = Field(..., description="Advisory title or country name")
    summary: str = Field(..., description="Advisory summary text")
    published_at: str | None = Field(
        None, description="Original publication date (ISO 8601)"
    )
    updated_at: str | None = Field(
        None, description="Last update date (ISO 8601)"
    )
    url: str | None = Field(None, description="Link to full advisory")
