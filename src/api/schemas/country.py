"""
Country-level risk summary DTO.

Provides aggregate risk metrics per country for the globe choropleth
and country listing pages. Dual-score model: baseline_risk (universal,
all ~195 countries from GDELT/ACLED/advisory inputs) and forecast_risk
(only where active predictions exist). The blended_risk field combines
70% forecast + 30% baseline when both exist, falling back to baseline
alone otherwise. risk_score is a backward-compat alias for blended_risk.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CountryRiskSummary(BaseModel):
    """Aggregate risk summary for a single country.

    Dual-score model:
      - ``baseline_risk`` is always present (computed hourly from GDELT event
        density, ACLED conflict intensity, travel advisory level, and Goldstein
        severity via the seeding heavy job).
      - ``forecast_risk`` is only present when active predictions exist for the
        country (CTE aggregation over the ``predictions`` table).
      - ``blended_risk`` = 0.7 * forecast_risk + 0.3 * baseline_risk when both
        exist, otherwise equals baseline_risk.
      - ``risk_score`` is a backward-compat alias for blended_risk.

    Used by the deck.gl globe choropleth (risk_score drives color intensity),
    the country listing sidebar, hover tooltips, and drill-down panels.
    """

    model_config = ConfigDict(from_attributes=True)

    iso_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    baseline_risk: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Baseline risk score (0-100) computed from GDELT event density, "
            "ACLED conflict intensity, travel advisory level, and Goldstein "
            "severity with 90-day exponential decay. Always present."
        ),
    )
    forecast_risk: float | None = Field(
        default=None,
        description=(
            "Forecast-derived risk score (0-100) from active predictions. "
            "None for countries with no active forecasts."
        ),
    )
    blended_risk: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Blended risk score: 0.7*forecast + 0.3*baseline when both "
            "exist, otherwise equals baseline_risk."
        ),
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Backward-compatible alias for blended_risk. Used by legacy "
            "frontend consumers and the choropleth color mapping."
        ),
    )
    forecast_count: int = Field(
        ..., ge=0, description="Number of active predictions for this country"
    )
    top_forecast: str | None = Field(
        default=None,
        description=(
            "Most recent forecast question for this country. "
            "None for baseline-only countries."
        ),
    )
    top_probability: float | None = Field(
        default=None,
        description=(
            "Probability of the most recent forecast. "
            "None for baseline-only countries."
        ),
    )
    trend: Literal["rising", "stable", "falling"] = Field(
        ..., description="Risk trend direction over the last 7 days"
    )
    last_updated: datetime = Field(
        ..., description="Timestamp of the most recent data update for this country"
    )
    disputed: bool = Field(
        default=False,
        description="True for disputed territories (XK, TW, PS, EH)",
    )
