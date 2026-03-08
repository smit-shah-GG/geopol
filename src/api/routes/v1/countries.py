"""
Country risk aggregation endpoints.

Dual-score model: every country gets a baseline_risk from the
``baseline_country_risk`` table (pre-computed hourly from GDELT, ACLED,
advisories, Goldstein). Countries with active predictions additionally
get a forecast_risk from the ``predictions`` table. The blended_risk
is 0.7*forecast + 0.3*baseline when both exist, otherwise equals
baseline.  ``risk_score`` is a backward-compat alias for blended_risk.

All endpoints require API key authentication.

Endpoints:
    GET /countries              -- List all countries with risk summaries
    GET /countries/{iso_code}   -- Single country risk summary
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_cache, get_db
from src.api.middleware.auth import verify_api_key
from src.api.schemas.country import CountryRiskSummary
from src.api.services.cache_service import SUMMARY_TTL, ForecastCache
from src.db.models import BaselineCountryRisk
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
# Forecast risk SQL: CTE-based aggregation over active predictions
# ---------------------------------------------------------------------------
# Returns one row per country_iso with forecast-derived metrics.
# Same proven CTE logic from the original endpoint, unchanged.

_FORECAST_RISK_SQL = text("""
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
    WHERE country_iso IS NOT NULL
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
top_forecasts AS (
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
    LEAST(100.0,
        LEAST(20.0, ca.forecast_count * 4.0) +
        COALESCE(ca.avg_probability, 0) * 50.0 +
        COALESCE(ca.avg_severity, 0.5) * 30.0
    ) AS forecast_risk,
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
JOIN top_forecasts tf ON ca.country_iso = tf.country_iso
LEFT JOIN past_agg pa ON ca.country_iso = pa.country_iso
ORDER BY forecast_risk DESC
""")


# ---------------------------------------------------------------------------
# Blending logic
# ---------------------------------------------------------------------------

_BLEND_FORECAST_WEIGHT = 0.7
_BLEND_BASELINE_WEIGHT = 0.3


def _blend_risk(baseline: float, forecast: float | None) -> float:
    """Compute blended risk: 70% forecast + 30% baseline, or baseline alone."""
    if forecast is not None:
        return round(
            _BLEND_FORECAST_WEIGHT * forecast + _BLEND_BASELINE_WEIGHT * baseline,
            1,
        )
    return round(baseline, 1)


# ---------------------------------------------------------------------------
# Dev-only fixture infrastructure (USE_FIXTURES=1)
# ---------------------------------------------------------------------------
# Preserved for local development when PostgreSQL has no data.
# Never active in production (use_fixtures defaults to False).

_fixture_cache: dict[str, CountryRiskSummary] | None = None

_FIXTURE_COUNTRIES: list[dict] = [
    {"iso": "SY", "risk": 87.0, "count": 3},
    {"iso": "UA", "risk": 82.0, "count": 4},
    {"iso": "MM", "risk": 71.0, "count": 2},
    {"iso": "IR", "risk": 65.0, "count": 2},
    {"iso": "TW", "risk": 58.0, "count": 2, "disputed": True},
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
        risk = entry["risk"]
        _fixture_cache[iso] = CountryRiskSummary(
            iso_code=iso,
            baseline_risk=risk,
            forecast_risk=risk,
            blended_risk=risk,
            risk_score=risk,
            forecast_count=entry["count"],
            top_forecast=f"Will a significant event occur in {iso} within 30 days?",
            top_probability=round(risk / 100.0 * 0.9, 2),
            trend="rising" if risk > 70 else "stable",
            last_updated=now,
            disputed=entry.get("disputed", False),
        )

    logger.debug("Fixture country cache initialized with %d entries", len(_fixture_cache))
    return _fixture_cache


# ---------------------------------------------------------------------------
# Merge helpers: baseline + forecast -> CountryRiskSummary
# ---------------------------------------------------------------------------


def _baseline_to_dto(row: BaselineCountryRisk) -> CountryRiskSummary:
    """Convert a BaselineCountryRisk ORM row to a baseline-only summary."""
    return CountryRiskSummary(
        iso_code=row.country_iso,
        baseline_risk=round(float(row.baseline_risk), 1),
        forecast_risk=None,
        blended_risk=round(float(row.baseline_risk), 1),
        risk_score=round(float(row.baseline_risk), 1),
        forecast_count=0,
        top_forecast=None,
        top_probability=None,
        trend="stable",
        last_updated=row.computed_at,
        disputed=row.disputed,
    )


def _merge_forecast_into_dto(
    baseline_row: BaselineCountryRisk,
    forecast_data: dict,
) -> CountryRiskSummary:
    """Merge baseline + forecast data into a dual-score summary."""
    baseline = round(float(baseline_row.baseline_risk), 1)
    forecast = round(float(forecast_data["forecast_risk"]), 1)
    blended = _blend_risk(baseline, forecast)

    return CountryRiskSummary(
        iso_code=baseline_row.country_iso,
        baseline_risk=baseline,
        forecast_risk=forecast,
        blended_risk=blended,
        risk_score=blended,
        forecast_count=int(forecast_data["forecast_count"]),
        top_forecast=forecast_data["top_forecast"],
        top_probability=round(float(forecast_data["top_probability"]), 4),
        trend=forecast_data["trend"],
        last_updated=forecast_data["last_updated"],
        disputed=baseline_row.disputed,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[CountryRiskSummary],
    summary="List countries with risk summaries",
    description=(
        "Returns all ~195 countries with dual-score risk data, sorted by "
        "blended risk score (0-100) descending. Countries with active "
        "predictions show 70/30 forecast/baseline blend; baseline-only "
        "countries show their baseline risk. Used by the globe choropleth "
        "and country listing sidebar."
    ),
)
async def list_countries(
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> list[CountryRiskSummary]:
    """Return all country risk summaries with dual-score data.

    Query flow: cache -> (baseline_country_risk + predictions CTE) -> fixture fallback (dev).
    Results cached at SUMMARY_TTL (1 hour).
    """
    # 1. Check cache
    cached = await cache.get(_CACHE_KEY_ALL)
    if cached is not None:
        return [CountryRiskSummary(**item) for item in cached]

    # 2. Query baseline_country_risk (all ~195 countries)
    # Table may not exist if migration 010 hasn't been applied yet
    try:
        baseline_result = await db.execute(
            select(BaselineCountryRisk).order_by(BaselineCountryRisk.baseline_risk.desc())
        )
        baseline_rows = baseline_result.scalars().all()
    except Exception:
        await db.rollback()
        baseline_rows = []

    if not baseline_rows:
        # No baseline data yet (first run hasn't happened or table missing) -- try fixtures
        settings = get_settings()
        if settings.use_fixtures:
            fixture_cache = _get_fixture_cache()
            return sorted(fixture_cache.values(), key=lambda c: c.risk_score, reverse=True)
        return []

    # 3. Query forecast risk for countries with active predictions
    forecast_result = await db.execute(_FORECAST_RISK_SQL)
    forecast_rows = forecast_result.fetchall()

    # Build forecast lookup: country_iso -> forecast data dict
    forecast_map: dict[str, dict] = {}
    for row in forecast_rows:
        forecast_map[row.country_iso] = {
            "forecast_risk": row.forecast_risk,
            "forecast_count": row.forecast_count,
            "top_forecast": row.top_forecast,
            "top_probability": row.top_probability,
            "last_updated": row.last_updated,
            "trend": row.trend,
        }

    # 4. Merge: for each baseline country, check if forecast exists
    summaries: list[CountryRiskSummary] = []
    for baseline_row in baseline_rows:
        iso = baseline_row.country_iso
        forecast_data = forecast_map.get(iso)
        if forecast_data is not None:
            summaries.append(_merge_forecast_into_dto(baseline_row, forecast_data))
        else:
            summaries.append(_baseline_to_dto(baseline_row))

    # Sort by blended/risk_score descending
    summaries.sort(key=lambda s: s.risk_score, reverse=True)

    # 5. Cache and return
    data = [s.model_dump(mode="json") for s in summaries]
    await cache.set(_CACHE_KEY_ALL, data, ttl=SUMMARY_TTL)
    return summaries


@router.get(
    "/{iso_code}",
    response_model=CountryRiskSummary,
    summary="Get country risk summary",
    description=(
        "Returns the dual-score risk summary for a single country by ISO code. "
        "Countries with no active forecasts return baseline risk only (no 404). "
        "Only returns 404 if the country has no baseline data at all."
    ),
)
async def get_country_risk(
    iso_code: str,
    _client: str = Depends(verify_api_key),
    cache: ForecastCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db),
) -> CountryRiskSummary:
    """Return the dual-score risk summary for a single country.

    Query flow: cache -> (baseline + forecast) -> fixture fallback (dev) -> 404.
    Countries without forecasts return baseline-only data (NOT a 404).
    """
    iso_upper = iso_code.upper()
    cache_key = _cache_key_single(iso_upper)

    # 1. Check cache
    cached = await cache.get(cache_key)
    if cached is not None:
        return CountryRiskSummary(**cached)

    # 2. Query baseline_country_risk for this country
    # Table may not exist if migration 010 hasn't been applied yet
    try:
        baseline_result = await db.execute(
            select(BaselineCountryRisk).where(
                BaselineCountryRisk.country_iso == iso_upper
            )
        )
        baseline_row = baseline_result.scalar_one_or_none()
    except Exception:
        await db.rollback()
        baseline_row = None

    if baseline_row is not None:
        # 3. Check for active forecast data
        # Use a parameterized single-country forecast query
        forecast_result = await db.execute(
            _SINGLE_COUNTRY_FORECAST_SQL, {"iso_code": iso_upper}
        )
        forecast_row = forecast_result.fetchone()

        if forecast_row is not None:
            forecast_data = {
                "forecast_risk": forecast_row.forecast_risk,
                "forecast_count": forecast_row.forecast_count,
                "top_forecast": forecast_row.top_forecast,
                "top_probability": forecast_row.top_probability,
                "last_updated": forecast_row.last_updated,
                "trend": forecast_row.trend,
            }
            summary = _merge_forecast_into_dto(baseline_row, forecast_data)
        else:
            summary = _baseline_to_dto(baseline_row)

        await cache.set(cache_key, summary.model_dump(mode="json"), ttl=SUMMARY_TTL)
        return summary

    # 4. Fixture fallback (dev only)
    settings = get_settings()
    if settings.use_fixtures:
        fixture_cache = _get_fixture_cache()
        fixture = fixture_cache.get(iso_upper)
        if fixture is not None:
            return fixture

    # 5. Not found -- country has no baseline data at all
    raise HTTPException(
        status_code=404,
        detail=f"No data found for country '{iso_upper}'",
    )


# ---------------------------------------------------------------------------
# Single-country forecast CTE (parameterized variant)
# ---------------------------------------------------------------------------

_SINGLE_COUNTRY_FORECAST_SQL = text("""
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
    ) AS forecast_risk,
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
