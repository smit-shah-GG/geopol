"""
Country risk aggregation endpoints.

Computes composite risk scores from PostgreSQL predictions table via
CTE-based SQL aggregation. Risk score (0-100) combines forecast count,
average probability, and CAMEO-derived Goldstein severity with exponential
time decay (7-day half-life). Trend is derived from 7-day delta comparison.

All endpoints require API key authentication.

Endpoints:
    GET /countries              -- List all countries with risk summaries
    GET /countries/{iso_code}   -- Single country risk summary
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_cache, get_db
from src.api.middleware.auth import verify_api_key
from src.api.schemas.country import CountryRiskSummary
from src.api.services.cache_service import SUMMARY_TTL, ForecastCache
from src.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Cache key helpers (countries namespace)
# ---------------------------------------------------------------------------

_CACHE_KEY_ALL = "countries:all"


def _cache_key_single(iso_code: str) -> str:
    return f"countries:{iso_code.upper()}"


# ---------------------------------------------------------------------------
# SQL: CTE-based country risk aggregation
# ---------------------------------------------------------------------------
#
# The query performs a single-pass aggregation over active predictions:
#   1. Filters to rows where country_iso IS NOT NULL AND expires_at > NOW()
#   2. Computes per-country metrics with exponential time decay (7-day half-life)
#   3. Maps CAMEO root codes to Goldstein severity via SQL CASE
#   4. Computes composite risk score (0-100) from count + probability + severity
#   5. Computes trend from 7-day delta comparison
#
# Covered by composite index: ix_predictions_country_created (country_iso, created_at)

_COUNTRY_RISK_SQL = text("""
WITH active_predictions AS (
    SELECT
        country_iso,
        probability,
        cameo_root_code,
        created_at,
        question,
        -- Exponential decay: half-life = 7 days (604800 seconds)
        -- weight = exp(-0.693 * age_seconds / 604800)
        EXP(-0.693 * EXTRACT(EPOCH FROM (NOW() - created_at)) / 604800.0)
            AS decay_weight,
        -- CAMEO root code -> Goldstein severity mapping
        CASE
            WHEN cameo_root_code IN ('15','16','17','18','19','20')
                THEN 1.0   -- Material conflict
            WHEN cameo_root_code IN ('10','11','12','13','14')
                THEN 0.6   -- Verbal conflict
            WHEN cameo_root_code IN ('06','07','08','09')
                THEN 0.3   -- Material cooperation
            WHEN cameo_root_code IN ('01','02','03','04','05')
                THEN 0.1   -- Verbal cooperation
            ELSE 0.5       -- Unknown / NULL CAMEO
        END AS severity
    FROM predictions
    WHERE country_iso IS NOT NULL
      AND expires_at > NOW()
),
country_agg AS (
    SELECT
        country_iso,
        COUNT(*) AS forecast_count,
        -- Weighted average probability (decay-weighted)
        SUM(probability * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_probability,
        -- Weighted average severity (decay-weighted)
        SUM(severity * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_severity,
        MAX(created_at) AS last_updated
    FROM active_predictions
    GROUP BY country_iso
),
top_forecasts AS (
    -- Most recent prediction per country (window function)
    SELECT DISTINCT ON (country_iso)
        country_iso,
        question AS top_forecast,
        probability AS top_probability
    FROM active_predictions
    ORDER BY country_iso, created_at DESC
),
-- Past scores for trend computation (predictions active 7 days ago)
past_predictions AS (
    SELECT
        country_iso,
        probability,
        cameo_root_code,
        created_at,
        EXP(-0.693 * EXTRACT(EPOCH FROM ((NOW() - INTERVAL '7 days') - created_at)) / 604800.0)
            AS decay_weight,
        CASE
            WHEN cameo_root_code IN ('15','16','17','18','19','20')
                THEN 1.0
            WHEN cameo_root_code IN ('10','11','12','13','14')
                THEN 0.6
            WHEN cameo_root_code IN ('06','07','08','09')
                THEN 0.3
            WHEN cameo_root_code IN ('01','02','03','04','05')
                THEN 0.1
            ELSE 0.5
        END AS severity
    FROM predictions
    WHERE country_iso IS NOT NULL
      AND created_at <= NOW() - INTERVAL '7 days'
      AND expires_at > NOW() - INTERVAL '7 days'
),
past_agg AS (
    SELECT
        country_iso,
        COUNT(*) AS forecast_count,
        SUM(probability * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_probability,
        SUM(severity * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_severity
    FROM past_predictions
    GROUP BY country_iso
)
SELECT
    ca.country_iso,
    ca.forecast_count,
    -- Composite risk score (0-100)
    LEAST(100.0,
        LEAST(20.0, ca.forecast_count * 4.0) +
        COALESCE(ca.avg_probability, 0) * 50.0 +
        COALESCE(ca.avg_severity, 0.5) * 30.0
    ) AS risk_score,
    tf.top_forecast,
    tf.top_probability,
    ca.last_updated,
    -- Trend: compare current score to past score
    CASE
        WHEN pa.country_iso IS NULL THEN 'stable'
        WHEN (
            LEAST(20.0, ca.forecast_count * 4.0) +
            COALESCE(ca.avg_probability, 0) * 50.0 +
            COALESCE(ca.avg_severity, 0.5) * 30.0
        ) - (
            LEAST(20.0, pa.forecast_count * 4.0) +
            COALESCE(pa.avg_probability, 0) * 50.0 +
            COALESCE(pa.avg_severity, 0.5) * 30.0
        ) > 5.0 THEN 'rising'
        WHEN (
            LEAST(20.0, pa.forecast_count * 4.0) +
            COALESCE(pa.avg_probability, 0) * 50.0 +
            COALESCE(pa.avg_severity, 0.5) * 30.0
        ) - (
            LEAST(20.0, ca.forecast_count * 4.0) +
            COALESCE(ca.avg_probability, 0) * 50.0 +
            COALESCE(ca.avg_severity, 0.5) * 30.0
        ) > 5.0 THEN 'falling'
        ELSE 'stable'
    END AS trend
FROM country_agg ca
JOIN top_forecasts tf ON ca.country_iso = tf.country_iso
LEFT JOIN past_agg pa ON ca.country_iso = pa.country_iso
ORDER BY risk_score DESC
""")

# Single-country variant: same logic, filtered by :iso_code parameter
_SINGLE_COUNTRY_SQL = text("""
WITH active_predictions AS (
    SELECT
        country_iso,
        probability,
        cameo_root_code,
        created_at,
        question,
        EXP(-0.693 * EXTRACT(EPOCH FROM (NOW() - created_at)) / 604800.0)
            AS decay_weight,
        CASE
            WHEN cameo_root_code IN ('15','16','17','18','19','20')
                THEN 1.0
            WHEN cameo_root_code IN ('10','11','12','13','14')
                THEN 0.6
            WHEN cameo_root_code IN ('06','07','08','09')
                THEN 0.3
            WHEN cameo_root_code IN ('01','02','03','04','05')
                THEN 0.1
            ELSE 0.5
        END AS severity
    FROM predictions
    WHERE country_iso = :iso_code
      AND expires_at > NOW()
),
country_agg AS (
    SELECT
        country_iso,
        COUNT(*) AS forecast_count,
        SUM(probability * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_probability,
        SUM(severity * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_severity,
        MAX(created_at) AS last_updated
    FROM active_predictions
    GROUP BY country_iso
),
top_forecast_cte AS (
    SELECT DISTINCT ON (country_iso)
        country_iso,
        question AS top_forecast,
        probability AS top_probability
    FROM active_predictions
    ORDER BY country_iso, created_at DESC
),
past_predictions AS (
    SELECT
        country_iso,
        probability,
        cameo_root_code,
        created_at,
        EXP(-0.693 * EXTRACT(EPOCH FROM ((NOW() - INTERVAL '7 days') - created_at)) / 604800.0)
            AS decay_weight,
        CASE
            WHEN cameo_root_code IN ('15','16','17','18','19','20')
                THEN 1.0
            WHEN cameo_root_code IN ('10','11','12','13','14')
                THEN 0.6
            WHEN cameo_root_code IN ('06','07','08','09')
                THEN 0.3
            WHEN cameo_root_code IN ('01','02','03','04','05')
                THEN 0.1
            ELSE 0.5
        END AS severity
    FROM predictions
    WHERE country_iso = :iso_code
      AND created_at <= NOW() - INTERVAL '7 days'
      AND expires_at > NOW() - INTERVAL '7 days'
),
past_agg AS (
    SELECT
        country_iso,
        COUNT(*) AS forecast_count,
        SUM(probability * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_probability,
        SUM(severity * decay_weight) / NULLIF(SUM(decay_weight), 0)
            AS avg_severity
    FROM past_predictions
    GROUP BY country_iso
)
SELECT
    ca.country_iso,
    ca.forecast_count,
    LEAST(100.0,
        LEAST(20.0, ca.forecast_count * 4.0) +
        COALESCE(ca.avg_probability, 0) * 50.0 +
        COALESCE(ca.avg_severity, 0.5) * 30.0
    ) AS risk_score,
    tf.top_forecast,
    tf.top_probability,
    ca.last_updated,
    CASE
        WHEN pa.country_iso IS NULL THEN 'stable'
        WHEN (
            LEAST(20.0, ca.forecast_count * 4.0) +
            COALESCE(ca.avg_probability, 0) * 50.0 +
            COALESCE(ca.avg_severity, 0.5) * 30.0
        ) - (
            LEAST(20.0, pa.forecast_count * 4.0) +
            COALESCE(pa.avg_probability, 0) * 50.0 +
            COALESCE(pa.avg_severity, 0.5) * 30.0
        ) > 5.0 THEN 'rising'
        WHEN (
            LEAST(20.0, pa.forecast_count * 4.0) +
            COALESCE(pa.avg_probability, 0) * 50.0 +
            COALESCE(pa.avg_severity, 0.5) * 30.0
        ) - (
            LEAST(20.0, ca.forecast_count * 4.0) +
            COALESCE(ca.avg_probability, 0) * 50.0 +
            COALESCE(ca.avg_severity, 0.5) * 30.0
        ) > 5.0 THEN 'falling'
        ELSE 'stable'
    END AS trend
FROM country_agg ca
JOIN top_forecast_cte tf ON ca.country_iso = tf.country_iso
LEFT JOIN past_agg pa ON ca.country_iso = pa.country_iso
""")


# ---------------------------------------------------------------------------
# Dev-only fixture infrastructure (USE_FIXTURES=1)
# ---------------------------------------------------------------------------
# Preserved for local development when PostgreSQL has no predictions.
# Never active in production (use_fixtures defaults to False).

_fixture_cache: dict[str, CountryRiskSummary] | None = None

_FIXTURE_COUNTRIES: list[dict] = [
    {"iso": "SY", "risk": 87.0, "count": 3},
    {"iso": "UA", "risk": 82.0, "count": 4},
    {"iso": "MM", "risk": 71.0, "count": 2},
    {"iso": "IR", "risk": 65.0, "count": 2},
    {"iso": "TW", "risk": 58.0, "count": 2},
    {"iso": "SD", "risk": 76.0, "count": 2},
    {"iso": "KP", "risk": 62.0, "count": 1},
    {"iso": "VE", "risk": 45.0, "count": 1},
]


def _get_fixture_cache() -> dict[str, CountryRiskSummary]:
    """Build the fixture country risk cache on first access (dev only)."""
    global _fixture_cache  # noqa: PLW0603
    if _fixture_cache is not None:
        return _fixture_cache

    now = datetime.now(timezone.utc)
    _fixture_cache = {}
    for entry in _FIXTURE_COUNTRIES:
        iso = entry["iso"]
        _fixture_cache[iso] = CountryRiskSummary(
            iso_code=iso,
            risk_score=entry["risk"],
            forecast_count=entry["count"],
            top_forecast=f"Will a significant event occur in {iso} within 30 days?",
            top_probability=round(entry["risk"] / 100.0 * 0.9, 2),
            trend="rising" if entry["risk"] > 70 else "stable",
            last_updated=now,
        )

    logger.debug("Fixture country cache initialized with %d entries", len(_fixture_cache))
    return _fixture_cache


# ---------------------------------------------------------------------------
# Row -> DTO mapping
# ---------------------------------------------------------------------------


def _row_to_dto(row) -> CountryRiskSummary:  # noqa: ANN001
    """Map a SQLAlchemy Row to a CountryRiskSummary DTO."""
    return CountryRiskSummary(
        iso_code=row.country_iso,
        risk_score=round(float(row.risk_score), 1),
        forecast_count=int(row.forecast_count),
        top_forecast=row.top_forecast,
        top_probability=round(float(row.top_probability), 4),
        trend=row.trend,
        last_updated=row.last_updated,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[CountryRiskSummary],
    summary="List countries with risk summaries",
    description=(
        "Returns all countries with active predictions, sorted by composite "
        "risk score (0-100) descending. Risk score combines forecast count, "
        "decay-weighted probability, and CAMEO Goldstein severity. "
        "Used by the globe choropleth and country listing sidebar."
    ),
)
async def list_countries(
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> list[CountryRiskSummary]:
    """Return all country risk summaries from PostgreSQL predictions table.

    Query flow: cache -> PostgreSQL CTE aggregation -> fixture fallback (dev).
    Results cached at SUMMARY_TTL (1 hour).
    """
    # 1. Check cache
    cached = await cache.get(_CACHE_KEY_ALL)
    if cached is not None:
        return [CountryRiskSummary(**item) for item in cached]

    # 2. Query PostgreSQL
    result = await db.execute(_COUNTRY_RISK_SQL)
    rows = result.fetchall()

    if rows:
        summaries = [_row_to_dto(row) for row in rows]
        data = [s.model_dump(mode="json") for s in summaries]
        await cache.set(_CACHE_KEY_ALL, data, ttl=SUMMARY_TTL)
        return summaries

    # 3. Fixture fallback (dev only, USE_FIXTURES=1)
    settings = get_settings()
    if settings.use_fixtures:
        fixture_cache = _get_fixture_cache()
        return sorted(fixture_cache.values(), key=lambda c: c.risk_score, reverse=True)

    return []


@router.get(
    "/{iso_code}",
    response_model=CountryRiskSummary,
    summary="Get country risk summary",
    description=(
        "Returns the composite risk summary for a single country by ISO code. "
        "Risk score (0-100) is computed from the predictions table with "
        "exponential time decay and CAMEO severity weighting."
    ),
)
async def get_country_risk(
    iso_code: str,
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> CountryRiskSummary:
    """Return the risk summary for a single country.

    Query flow: cache -> PostgreSQL CTE aggregation -> fixture fallback (dev) -> 404.
    """
    iso_upper = iso_code.upper()
    cache_key = _cache_key_single(iso_upper)

    # 1. Check cache
    cached = await cache.get(cache_key)
    if cached is not None:
        return CountryRiskSummary(**cached)

    # 2. Query PostgreSQL
    result = await db.execute(_SINGLE_COUNTRY_SQL, {"iso_code": iso_upper})
    row = result.fetchone()

    if row is not None:
        summary = _row_to_dto(row)
        await cache.set(cache_key, summary.model_dump(mode="json"), ttl=SUMMARY_TTL)
        return summary

    # 3. Fixture fallback (dev only)
    settings = get_settings()
    if settings.use_fixtures:
        fixture_cache = _get_fixture_cache()
        fixture = fixture_cache.get(iso_upper)
        if fixture is not None:
            return fixture

    # 4. Not found
    raise HTTPException(
        status_code=404,
        detail=f"No predictions found for country '{iso_upper}'",
    )
