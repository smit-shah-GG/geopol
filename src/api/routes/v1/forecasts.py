"""
Forecast CRUD endpoints.

Serves mock fixture data until Phase 10 replaces with real database queries.
All endpoints require API key authentication via the ``verify_api_key``
dependency.

Endpoints:
    GET  /forecasts/{forecast_id}        -- Single forecast by ID
    GET  /forecasts/country/{iso_code}   -- Forecasts by country ISO code
    GET  /forecasts/top                  -- Top risk forecasts
    POST /forecasts                      -- Create forecast (mock)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.fixtures.factory import (
    create_mock_forecast,
    load_all_fixtures,
    load_fixture,
)
from src.api.middleware.auth import verify_api_key
from src.api.schemas.common import PaginatedResponse
from src.api.schemas.forecast import ForecastResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory mock forecast registry, populated from fixtures on first access.
# Phase 10 replaces this with real database queries.
_forecast_cache: dict[str, ForecastResponse] | None = None


def _get_forecast_cache() -> dict[str, ForecastResponse]:
    """Lazy-load the mock forecast cache from fixtures + generated mocks."""
    global _forecast_cache  # noqa: PLW0603
    if _forecast_cache is not None:
        return _forecast_cache

    _forecast_cache = {}

    # Load hand-crafted fixtures (SY, UA, MM)
    try:
        fixtures = load_all_fixtures()
        for _code, forecast in fixtures.items():
            _forecast_cache[forecast.forecast_id] = forecast
    except Exception as exc:
        logger.warning("Could not load fixtures: %s", exc)

    # Generate additional mocks for countries without fixtures
    for iso in ["IR", "TW", "SD"]:
        mock = create_mock_forecast(country_iso=iso, horizon_days=30)
        _forecast_cache[mock.forecast_id] = mock

    logger.debug("Forecast cache initialized with %d entries", len(_forecast_cache))
    return _forecast_cache


class CreateForecastRequest(BaseModel):
    """Request body for POST /forecasts (mock implementation)."""

    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The forecasting question to answer",
    )
    country_iso: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO country code",
    )
    horizon_days: int = Field(
        default=30,
        gt=0,
        le=365,
        description="Forecast horizon in days",
    )


@router.get(
    "/top",
    response_model=list[ForecastResponse],
    summary="Top risk forecasts",
    description="Returns the highest-probability active forecasts across all countries.",
)
async def get_top_forecasts(
    limit: int = Query(default=5, ge=1, le=20, description="Number of top forecasts"),
    _client: str = Depends(verify_api_key),
) -> list[ForecastResponse]:
    """Return the top N forecasts sorted by probability descending."""
    cache = _get_forecast_cache()
    forecasts = sorted(
        cache.values(), key=lambda f: f.probability, reverse=True
    )
    return forecasts[:limit]


@router.get(
    "/country/{iso_code}",
    response_model=PaginatedResponse[ForecastResponse],
    summary="Forecasts by country",
    description="Returns paginated forecasts for a specific country ISO code.",
)
async def get_forecasts_by_country(
    iso_code: str,
    cursor: Optional[str] = Query(default=None, description="Pagination cursor"),
    limit: int = Query(default=10, ge=1, le=50, description="Page size"),
    _client: str = Depends(verify_api_key),
) -> PaginatedResponse[ForecastResponse]:
    """Return forecasts for a country. Mock data for now."""
    iso_upper = iso_code.upper()
    cache = _get_forecast_cache()

    # Filter by country
    country_forecasts = [
        f for f in cache.values()
        if f.forecast_id.startswith(f"fc-{iso_upper.lower()}-")
        or _guess_country_iso(f) == iso_upper
    ]

    # If no fixtures/mocks exist for this country, try loading fixture directly
    if not country_forecasts:
        try:
            fixture = load_fixture(iso_upper)
            cache[fixture.forecast_id] = fixture
            country_forecasts = [fixture]
        except FileNotFoundError:
            pass  # Country has no forecasts — return empty

    # Simple pagination (mock — no real cursor logic needed for mock data)
    items = country_forecasts[:limit]
    has_more = len(country_forecasts) > limit

    return PaginatedResponse[ForecastResponse](
        items=items,
        next_cursor=None,  # Mock: no real pagination needed
        has_more=has_more,
    )


@router.get(
    "/{forecast_id}",
    response_model=ForecastResponse,
    summary="Get forecast by ID",
    description="Returns a single forecast with full scenario tree, evidence, and calibration data.",
)
async def get_forecast_by_id(
    forecast_id: str,
    _client: str = Depends(verify_api_key),
) -> ForecastResponse:
    """Return a single forecast by its ID."""
    cache = _get_forecast_cache()

    forecast = cache.get(forecast_id)
    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail=f"Forecast '{forecast_id}' not found",
        )

    return forecast


@router.post(
    "",
    response_model=ForecastResponse,
    status_code=201,
    summary="Create forecast",
    description=(
        "Submit a forecasting question. Returns a mock forecast immediately. "
        "Phase 10 replaces this with real prediction pipeline invocation."
    ),
)
async def create_forecast(
    body: CreateForecastRequest,
    _client: str = Depends(verify_api_key),
) -> ForecastResponse:
    """Create a mock forecast for the given question."""
    logger.info(
        "Create forecast request: country=%s, question=%s...",
        body.country_iso,
        body.question[:60],
    )

    forecast = create_mock_forecast(
        country_iso=body.country_iso,
        question=body.question,
        horizon_days=body.horizon_days,
    )

    # Cache the new forecast so it's retrievable by ID
    cache = _get_forecast_cache()
    cache[forecast.forecast_id] = forecast

    return forecast


def _guess_country_iso(forecast: ForecastResponse) -> str | None:
    """Extract country ISO from forecast_id convention: fc-{iso}-{hash}."""
    parts = forecast.forecast_id.split("-")
    if len(parts) >= 2:
        return parts[1].upper()
    return None
