"""
Forecast CRUD endpoints with real data, caching, rate limiting, and sanitization.

GET endpoints query PostgreSQL via ForecastService with ForecastCache
(3-tier: memory -> Redis -> PostgreSQL). Mock fixtures serve as fallback
when PostgreSQL has no data (preserving Phase 9 development behavior).

POST invokes live EnsemblePredictor with rate limiting (per-API-key daily
quota), input sanitization (prompt injection blocklist + geopolitical
keyword filter), and Gemini budget enforcement.

All endpoints require API key authentication via ``verify_api_key``.

Endpoints:
    GET  /forecasts/{forecast_id}        -- Single forecast by ID (cache + DB + fixture)
    GET  /forecasts/country/{iso_code}   -- Forecasts by country (cache + DB + fixture)
    GET  /forecasts/top                  -- Top risk forecasts (cache + DB + fixture)
    POST /forecasts                      -- Live EnsemblePredictor + persist + cache
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_cache, get_db, get_redis
from src.api.fixtures.factory import (
    create_mock_forecast,
    load_all_fixtures,
    load_fixture,
)
from src.api.middleware.auth import verify_api_key
from src.api.middleware.rate_limit import (
    gemini_budget_remaining,
    get_rate_limiter,
    increment_gemini_usage,
)
from src.api.middleware.sanitize import (
    sanitize_error_response,
    validate_forecast_question,
)
from src.api.schemas.common import PaginatedResponse
from src.api.schemas.forecast import ForecastResponse
from src.api.services.cache_service import (
    FULL_FORECAST_TTL,
    SUMMARY_TTL,
    ForecastCache,
    cache_key_for_country,
    cache_key_for_forecast,
    cache_key_for_top,
)
from src.api.services.forecast_service import ForecastService

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory mock forecast registry, populated from fixtures on first access.
# Kept as development fallback when PostgreSQL has no real data.
_fixture_cache: dict[str, ForecastResponse] | None = None


def _get_fixture_cache() -> dict[str, ForecastResponse]:
    """Lazy-load the mock forecast cache from fixtures + generated mocks."""
    global _fixture_cache  # noqa: PLW0603
    if _fixture_cache is not None:
        return _fixture_cache

    _fixture_cache = {}

    # Load hand-crafted fixtures (SY, UA, MM)
    try:
        fixtures = load_all_fixtures()
        for _code, forecast in fixtures.items():
            _fixture_cache[forecast.forecast_id] = forecast
    except Exception as exc:
        logger.warning("Could not load fixtures: %s", exc)

    # Generate additional mocks for countries without fixtures
    for iso in ["IR", "TW", "SD"]:
        mock = create_mock_forecast(country_iso=iso, horizon_days=30)
        _fixture_cache[mock.forecast_id] = mock

    logger.debug("Fixture cache initialized with %d entries", len(_fixture_cache))
    return _fixture_cache


class CreateForecastRequest(BaseModel):
    """Request body for POST /forecasts."""

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


# Rate limiter dependency for POST endpoint
_post_rate_limiter = get_rate_limiter(daily_limit=50)


@router.get(
    "/top",
    response_model=list[ForecastResponse],
    summary="Top risk forecasts",
    description="Returns the highest-probability active forecasts across all countries.",
)
async def get_top_forecasts(
    limit: int = Query(default=5, ge=1, le=20, description="Number of top forecasts"),
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> list[ForecastResponse]:
    """Return the top N forecasts sorted by probability descending.

    Query order: cache -> PostgreSQL -> mock fixtures.
    """
    # Check cache
    key = cache_key_for_top(limit)
    cached = await cache.get(key)
    if cached is not None:
        return [ForecastResponse(**item) for item in cached]

    # Query PostgreSQL
    try:
        service = ForecastService(db)
        result = await service.get_top_forecasts(limit=limit)
        if result:
            data = [item.model_dump(mode="json") for item in result]
            await cache.set(key, data, ttl=SUMMARY_TTL)
            return result
    except Exception as exc:
        logger.warning("PostgreSQL top forecasts query failed: %s", exc)

    # Fall back to fixtures
    fixture_cache = _get_fixture_cache()
    forecasts = sorted(
        fixture_cache.values(), key=lambda f: f.probability, reverse=True
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
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ForecastResponse]:
    """Return forecasts for a country.

    Query order: cache (only for first page) -> PostgreSQL -> mock fixtures.
    """
    iso_upper = iso_code.upper()

    # Check cache (first page only -- cursor pagination bypasses cache)
    if cursor is None:
        key = cache_key_for_country(iso_upper)
        cached = await cache.get(key)
        if cached is not None:
            items = [ForecastResponse(**item) for item in cached.get("items", [])]
            return PaginatedResponse[ForecastResponse](
                items=items[:limit],
                next_cursor=cached.get("next_cursor"),
                has_more=cached.get("has_more", False),
            )

    # Query PostgreSQL
    try:
        service = ForecastService(db)
        result = await service.get_forecasts_by_country(
            country_iso=iso_upper, cursor=cursor, limit=limit
        )
        if result.items:
            # Cache first page only
            if cursor is None:
                data = {
                    "items": [item.model_dump(mode="json") for item in result.items],
                    "next_cursor": result.next_cursor,
                    "has_more": result.has_more,
                }
                await cache.set(cache_key_for_country(iso_upper), data, ttl=SUMMARY_TTL)
            return result
    except Exception as exc:
        logger.warning("PostgreSQL country query failed for %s: %s", iso_upper, exc)

    # Fall back to fixtures
    fixture_cache = _get_fixture_cache()
    country_forecasts = [
        f for f in fixture_cache.values()
        if f.forecast_id.startswith(f"fc-{iso_upper.lower()}-")
        or _guess_country_iso(f) == iso_upper
    ]

    if not country_forecasts:
        try:
            fixture = load_fixture(iso_upper)
            fixture_cache[fixture.forecast_id] = fixture
            country_forecasts = [fixture]
        except FileNotFoundError:
            pass

    items = country_forecasts[:limit]
    has_more = len(country_forecasts) > limit

    return PaginatedResponse[ForecastResponse](
        items=items,
        next_cursor=None,
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
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> ForecastResponse:
    """Return a single forecast by its ID.

    Query order: cache -> PostgreSQL -> mock fixtures -> 404.
    """
    # 1. Check cache
    key = cache_key_for_forecast(forecast_id)
    cached = await cache.get(key)
    if cached is not None:
        return ForecastResponse(**cached)

    # 2. Try PostgreSQL
    try:
        service = ForecastService(db)
        result = await service.get_forecast_by_id(forecast_id)
        if result is not None:
            await cache.set(key, result.model_dump(mode="json"), ttl=FULL_FORECAST_TTL)
            return result
    except Exception as exc:
        logger.warning("PostgreSQL lookup failed for %s: %s", forecast_id, exc)

    # 3. Fall back to mock fixture cache
    fixture_cache = _get_fixture_cache()
    forecast = fixture_cache.get(forecast_id)
    if forecast is not None:
        return forecast

    # 4. Not found
    raise HTTPException(
        status_code=404,
        detail=f"Forecast '{forecast_id}' not found",
    )


@router.post(
    "",
    response_model=ForecastResponse,
    status_code=201,
    summary="Create forecast",
    description=(
        "Submit a forecasting question. Runs the live EnsemblePredictor, "
        "persists the result, and returns the full forecast response. "
        "Rate limited per API key. Input is sanitized against prompt injection."
    ),
)
async def create_forecast(
    body: CreateForecastRequest,
    _client: str = Depends(verify_api_key),
    _rate_limit: None = Depends(_post_rate_limiter),
    cache: ForecastCache = Depends(get_cache),
    redis_client: aioredis.Redis = Depends(get_redis),
    db: AsyncSession = Depends(get_db),
) -> ForecastResponse:
    """Create a forecast via live EnsemblePredictor with rate limiting and sanitization."""
    # Sanitize input
    cleaned_question = validate_forecast_question(body.question)

    # Check Gemini budget
    remaining = await gemini_budget_remaining(redis_client)
    if remaining <= 0:
        raise HTTPException(
            status_code=429,
            detail="Forecast generation budget exhausted for today.",
        )

    logger.info(
        "Create forecast request: country=%s, question=%s...",
        body.country_iso,
        cleaned_question[:60],
    )

    try:
        # Import EnsemblePredictor lazily to avoid circular imports at module load
        from src.forecasting.ensemble_predictor import EnsemblePredictor

        predictor = EnsemblePredictor()

        # Run prediction (synchronous) via asyncio.to_thread
        ensemble_pred, forecast_output = await asyncio.to_thread(
            predictor.predict,
            question=cleaned_question,
        )

        # Persist via ForecastService
        service = ForecastService(db)
        prediction = await service.persist_forecast(
            forecast_output=forecast_output,
            ensemble_prediction=ensemble_pred,
            country_iso=body.country_iso.upper(),
            horizon_days=body.horizon_days,
        )
        await db.commit()

        # Increment Gemini usage
        await increment_gemini_usage(redis_client)

        # Build response DTO
        response = ForecastService.prediction_to_dto(prediction)

        # Cache the new forecast
        key = cache_key_for_forecast(prediction.id)
        await cache.set(key, response.model_dump(mode="json"), ttl=FULL_FORECAST_TTL)

        return response

    except HTTPException:
        raise  # Re-raise rate limit / validation errors
    except Exception as exc:
        logger.error("Forecast creation failed: %s", exc, exc_info=True)
        error_resp = sanitize_error_response(exc)
        raise HTTPException(status_code=500, detail=error_resp["detail"]) from exc


def _guess_country_iso(forecast: ForecastResponse) -> str | None:
    """Extract country ISO from forecast_id convention: fc-{iso}-{hash}."""
    parts = forecast.forecast_id.split("-")
    if len(parts) >= 2:
        return parts[1].upper()
    return None
