"""
Country risk endpoints.

Serves mock country risk data until Phase 10 replaces with real aggregations.
All endpoints require API key authentication.

Endpoints:
    GET /countries              -- List all countries with risk summaries
    GET /countries/{iso_code}   -- Single country risk summary
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.fixtures.factory import create_mock_country_risk
from src.api.middleware.auth import verify_api_key
from src.api.schemas.country import CountryRiskSummary

logger = logging.getLogger(__name__)

router = APIRouter()

# Countries with mock data — matches the fixture set from Plan 02 plus extras
_MOCK_COUNTRIES: list[dict] = [
    {"iso": "SY", "risk": 0.87, "count": 3},
    {"iso": "UA", "risk": 0.82, "count": 4},
    {"iso": "MM", "risk": 0.71, "count": 2},
    {"iso": "IR", "risk": 0.65, "count": 2},
    {"iso": "TW", "risk": 0.58, "count": 2},
    {"iso": "SD", "risk": 0.76, "count": 2},
    {"iso": "KP", "risk": 0.62, "count": 1},
    {"iso": "VE", "risk": 0.45, "count": 1},
]

# Lazy-initialized country risk cache
_country_cache: dict[str, CountryRiskSummary] | None = None


def _get_country_cache() -> dict[str, CountryRiskSummary]:
    """Build the mock country risk cache on first access."""
    global _country_cache  # noqa: PLW0603
    if _country_cache is not None:
        return _country_cache

    _country_cache = {}
    for entry in _MOCK_COUNTRIES:
        summary = create_mock_country_risk(
            iso_code=entry["iso"],
            risk_score=entry["risk"],
            forecast_count=entry["count"],
        )
        _country_cache[entry["iso"]] = summary

    logger.debug("Country cache initialized with %d entries", len(_country_cache))
    return _country_cache


@router.get(
    "",
    response_model=list[CountryRiskSummary],
    summary="List countries with risk summaries",
    description=(
        "Returns all countries with active forecasts, sorted by risk score "
        "descending. Used by the globe choropleth and country listing sidebar."
    ),
)
async def list_countries(
    _client: str = Depends(verify_api_key),
) -> list[CountryRiskSummary]:
    """Return all mock country risk summaries, sorted by risk descending."""
    cache = _get_country_cache()
    return sorted(cache.values(), key=lambda c: c.risk_score, reverse=True)


@router.get(
    "/{iso_code}",
    response_model=CountryRiskSummary,
    summary="Get country risk summary",
    description="Returns the risk summary for a single country by ISO code.",
)
async def get_country_risk(
    iso_code: str,
    _client: str = Depends(verify_api_key),
) -> CountryRiskSummary:
    """Return the risk summary for a single country."""
    iso_upper = iso_code.upper()
    cache = _get_country_cache()

    summary = cache.get(iso_upper)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data for country '{iso_upper}'",
        )

    return summary
