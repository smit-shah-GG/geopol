"""
Forecast CRUD endpoints with real data, caching, rate limiting, and sanitization.

GET endpoints query PostgreSQL via ForecastService with ForecastCache
(3-tier: memory -> Redis -> PostgreSQL). Mock fixtures are available only
when USE_FIXTURES=1 is set (development convenience). Production returns
PostgreSQL results only -- empty results yield empty responses, not fixtures.

POST invokes live EnsemblePredictor with rate limiting (per-API-key daily
quota), input sanitization (prompt injection blocklist + geopolitical
keyword filter), and Gemini budget enforcement.

All endpoints require API key authentication via ``verify_api_key``.

Endpoints:
    GET  /forecasts/top                  -- Top risk forecasts (cache + DB)
    GET  /forecasts/search               -- Full-text search (tsvector + GIN)
    GET  /forecasts/country/{iso_code}   -- Forecasts by country (cache + DB)
    GET  /forecasts/{forecast_id}        -- Single forecast by ID (cache + DB)
    POST /forecasts                      -- Live EnsemblePredictor + persist + cache
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
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
from src.api.schemas.search import SearchResponse, SearchResult
from src.db.models import Prediction
from src.api.services.cache_service import (
    FULL_FORECAST_TTL,
    SUMMARY_TTL,
    ForecastCache,
    cache_key_for_country,
    cache_key_for_forecast,
    cache_key_for_top,
)
from src.api.services.forecast_service import ForecastService
from src.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Dev-only fixture infrastructure (USE_FIXTURES=1) ---
# Preserved for local development when PostgreSQL has no real data.
# Never active in production (use_fixtures defaults to False).
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
    limit: int = Query(default=5, ge=1, le=50, description="Number of top forecasts"),
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> list[ForecastResponse]:
    """Return the top N forecasts sorted by probability descending.

    Query order: cache -> PostgreSQL. Fixture fallback only when USE_FIXTURES=1.
    """
    # Check cache
    key = cache_key_for_top(limit)
    cached = await cache.get(key)
    if cached is not None:
        return [ForecastResponse(**item) for item in cached]

    # Query PostgreSQL
    service = ForecastService(db)
    result = await service.get_top_forecasts(limit=limit)
    if result:
        # Enrich with Polymarket comparison data (batch query, no N+1)
        result = await service.enrich_with_comparisons(result)
        data = [item.model_dump(mode="json") for item in result]
        await cache.set(key, data, ttl=SUMMARY_TTL)
        return result

    # Fixture fallback (dev only)
    settings = get_settings()
    if settings.use_fixtures:
        fixture_cache = _get_fixture_cache()
        forecasts = sorted(
            fixture_cache.values(), key=lambda f: f.probability, reverse=True
        )
        return forecasts[:limit]

    return []


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search forecasts",
    description=(
        "Full-text search over forecast questions using PostgreSQL tsvector + GIN index. "
        "Supports optional country and category filters. Results ranked by ts_rank relevance."
    ),
)
async def search_forecasts(
    q: str = Query(
        ..., min_length=2, max_length=200, description="Search query text"
    ),
    country: Optional[str] = Query(
        default=None, description="Filter by country ISO code"
    ),
    category: Optional[str] = Query(
        default=None, description="Filter by forecast category"
    ),
    limit: int = Query(default=20, ge=1, le=50, description="Maximum results"),
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Full-text search over forecast questions with optional filters.

    Uses PostgreSQL ``plainto_tsquery`` for safe natural-language input parsing
    (no injection risk from raw tsquery syntax) and ``ts_rank`` for relevance
    ordering. The GIN index on ``question_tsv`` (migration 004) ensures sub-200ms
    queries even at thousands of predictions.

    Returns empty results (HTTP 200) for queries that match nothing or produce
    empty tsqueries (e.g., all stop-words).
    """
    query_ts = func.plainto_tsquery("english", q)

    # Build base query: match tsvector and compute relevance
    rank_expr = func.ts_rank(Prediction.question_tsv, query_ts)
    stmt = select(Prediction, rank_expr.label("relevance")).where(
        Prediction.question_tsv.op("@@")(query_ts)
    )

    # Optional filters
    if country:
        stmt = stmt.where(Prediction.country_iso == country.upper())
    if category:
        stmt = stmt.where(Prediction.category == category)

    # Total count before pagination
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # Order by relevance descending, apply limit
    stmt = stmt.order_by(rank_expr.desc()).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    results: list[SearchResult] = []
    for prediction, relevance in rows:
        dto = ForecastService.prediction_to_dto(prediction)
        results.append(SearchResult(forecast=dto, relevance=float(relevance)))

    return SearchResponse(results=results, total=total, query=q)


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

    Query order: cache (only for first page) -> PostgreSQL. Fixture fallback
    only when USE_FIXTURES=1.
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
    service = ForecastService(db)
    result = await service.get_forecasts_by_country(
        country_iso=iso_upper, cursor=cursor, limit=limit
    )
    if result.items:
        # Enrich with Polymarket comparison data (batch query, no N+1)
        enriched_items = await service.enrich_with_comparisons(result.items)
        result = PaginatedResponse[ForecastResponse](
            items=enriched_items,
            next_cursor=result.next_cursor,
            has_more=result.has_more,
        )
        # Cache first page only
        if cursor is None:
            data = {
                "items": [item.model_dump(mode="json") for item in result.items],
                "next_cursor": result.next_cursor,
                "has_more": result.has_more,
            }
            await cache.set(cache_key_for_country(iso_upper), data, ttl=SUMMARY_TTL)
        return result

    # Fixture fallback (dev only)
    settings = get_settings()
    if settings.use_fixtures:
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

    return PaginatedResponse[ForecastResponse](
        items=[], next_cursor=None, has_more=False
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

    Query order: cache -> PostgreSQL -> 404. Fixture fallback only when
    USE_FIXTURES=1.
    """
    # 1. Check cache
    key = cache_key_for_forecast(forecast_id)
    cached = await cache.get(key)
    if cached is not None:
        return ForecastResponse(**cached)

    # 2. Try PostgreSQL
    service = ForecastService(db)
    result = await service.get_forecast_by_id(forecast_id)
    if result is not None:
        # Enrich with Polymarket comparison data
        enriched = await service.enrich_with_comparisons([result])
        result = enriched[0]
        await cache.set(key, result.model_dump(mode="json"), ttl=FULL_FORECAST_TTL)
        return result

    # 3. Fixture fallback (dev only)
    settings = get_settings()
    if settings.use_fixtures:
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
    """Extract country ISO from forecast_id convention: fc-{iso}-{hash}.

    Dev-only: only reachable when USE_FIXTURES=1 is set.
    """
    parts = forecast.forecast_id.split("-")
    if len(parts) >= 2:
        return parts[1].upper()
    return None
