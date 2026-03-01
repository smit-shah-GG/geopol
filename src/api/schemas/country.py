"""
Country-level risk summary DTO.

Provides aggregate risk metrics per country for the globe choropleth
and country listing pages. Derived from WORLDMONITOR_INTEGRATION.md
CountryRiskSummary spec.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CountryRiskSummary(BaseModel):
    """Aggregate risk summary for a single country.

    Used by the deck.gl globe choropleth layer and the country listing
    sidebar. The risk_score drives color intensity, trend drives the
    arrow indicator, and top_question provides hover context.
    """

    model_config = ConfigDict(from_attributes=True)

    iso_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate forecast risk score (0.0 = low, 1.0 = critical)",
    )
    forecast_count: int = Field(
        ..., ge=0, description="Number of active forecasts for this country"
    )
    top_question: str = Field(
        ..., description="Most consequential active forecast question"
    )
    top_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of the top forecast"
    )
    trend: Literal["rising", "stable", "falling"] = Field(
        ..., description="Risk trend direction over the last 7 days"
    )
    last_updated: datetime = Field(
        ..., description="When this country summary was last recalculated"
    )
