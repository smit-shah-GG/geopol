"""
Country-level risk summary DTO.

Provides aggregate risk metrics per country for the globe choropleth
and country listing pages. Risk scores are computed from PostgreSQL
predictions table via CTE-based aggregation (count + probability +
Goldstein severity with exponential time decay).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CountryRiskSummary(BaseModel):
    """Aggregate risk summary for a single country.

    All fields are computed from the ``predictions`` table via SQL
    aggregation. The composite risk_score (0-100) combines forecast count,
    average probability, and CAMEO-derived Goldstein severity with
    exponential time decay (7-day half-life). Trend is derived from
    7-day delta comparison.

    Used by the deck.gl globe choropleth layer (risk_score drives color
    intensity), the country listing sidebar, and hover tooltips.
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
        le=100.0,
        description=(
            "Composite risk index 0-100 (count + probability + "
            "Goldstein severity with exponential time decay)"
        ),
    )
    forecast_count: int = Field(
        ..., ge=0, description="Number of active predictions for this country"
    )
    top_forecast: str = Field(
        ..., description="Most recent forecast question for this country"
    )
    top_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of the most recent forecast"
    )
    trend: Literal["rising", "stable", "falling"] = Field(
        ..., description="Risk trend direction over the last 7 days"
    )
    last_updated: datetime = Field(
        ..., description="Timestamp of the most recent prediction for this country"
    )
