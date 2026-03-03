"""
Calibration API endpoints: Polymarket comparison and weight management.

Endpoints:
    GET /calibration/polymarket -- Active/resolved comparisons + summary stats
    GET /calibration/weights    -- Current per-CAMEO calibration weights
    GET /calibration/weights/history -- Weight version history with filters
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.middleware.auth import verify_api_key
from src.db.models import CalibrationWeight, CalibrationWeightHistory

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Response models (small, route-specific -- kept inline per plan)
# ---------------------------------------------------------------------------


class PolymarketComparisonItem(BaseModel):
    """Single active or resolved Polymarket comparison."""

    id: int
    polymarket_event_id: str
    polymarket_slug: str
    polymarket_title: str
    geopol_prediction_id: str
    match_confidence: float
    polymarket_price: float | None
    geopol_probability: float | None
    last_snapshot_at: str | None
    status: str
    geopol_brier: float | None = None
    polymarket_brier: float | None = None
    resolved_at: str | None = None
    created_at: str


class PolymarketSummary(BaseModel):
    """Aggregate comparison statistics."""

    active_count: int
    resolved_count: int
    geopol_avg_brier: float | None
    polymarket_avg_brier: float | None
    geopol_wins: int


class PolymarketComparisonResponse(BaseModel):
    """Full response for GET /calibration/polymarket."""

    active: list[PolymarketComparisonItem]
    resolved: list[PolymarketComparisonItem]
    summary: PolymarketSummary
    seeking_more_matches: bool


class PolymarketTopEventItem(BaseModel):
    """Single event from GET /calibration/polymarket/top."""

    event_id: str
    title: str
    slug: str
    volume: float
    liquidity: float
    # Optional Geopol match data (None when no comparison row exists)
    geopol_prediction_id: str | None = None
    geopol_probability: float | None = None
    geopol_question: str | None = None
    match_confidence: float | None = None


class PolymarketTopResponse(BaseModel):
    """Full response for GET /calibration/polymarket/top."""

    events: list[PolymarketTopEventItem]
    total_geo_markets: int  # Geo events before top-N cut


class CalibrationWeightItem(BaseModel):
    """Current calibration weight for a CAMEO code."""

    cameo_code: str
    alpha: float
    sample_size: int
    brier_score: float | None
    updated_at: str


class CalibrationWeightHistoryItem(BaseModel):
    """Versioned weight history entry."""

    id: int
    cameo_code: str
    alpha: float
    sample_size: int
    brier_score: float | None
    computed_at: str
    auto_applied: bool
    flagged: bool
    flag_reason: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/polymarket",
    response_model=PolymarketComparisonResponse,
    summary="Polymarket vs Geopol comparisons",
    description=(
        "Returns active and resolved Polymarket-vs-Geopol forecast comparisons "
        "with aggregate summary statistics. seeking_more_matches is True when "
        "fewer than 5 active overlaps exist."
    ),
)
async def get_polymarket_comparisons(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> PolymarketComparisonResponse:
    """Fetch Polymarket comparison data via PolymarketComparisonService."""
    from src.polymarket.comparison import PolymarketComparisonService

    # Build a minimal service with only the session factory (querying methods
    # don't use client/matcher/settings -- they use the session factory directly).
    # We wrap the db session in a factory that yields the existing session.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _session_wrapper():
        yield db

    service = PolymarketComparisonService.__new__(PolymarketComparisonService)
    service._session_factory = _session_wrapper

    active_items = await service.get_active_comparisons()
    resolved_items = await service.get_resolved_comparisons()
    summary_data = await service.get_comparison_summary()

    active = [PolymarketComparisonItem(status="active", **item) for item in active_items]
    resolved = [PolymarketComparisonItem(status="resolved", **item) for item in resolved_items]

    summary = PolymarketSummary(**summary_data)
    seeking = summary.active_count < 5

    return PolymarketComparisonResponse(
        active=active,
        resolved=resolved,
        summary=summary,
        seeking_more_matches=seeking,
    )


@router.get(
    "/polymarket/top",
    response_model=PolymarketTopResponse,
    summary="Top geopolitical Polymarket events by volume",
    description=(
        "Returns the top 10 geopolitical Polymarket events sorted by trading "
        "volume, with optional Geopol forecast match data for cross-referencing."
    ),
)
async def get_polymarket_top(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> PolymarketTopResponse:
    """Fetch top geopolitical Polymarket events and cross-reference with Geopol predictions."""
    from src.db.models import PolymarketComparison, Prediction
    from src.polymarket.client import PolymarketClient

    client = PolymarketClient()
    try:
        all_geo = await client.fetch_geopolitical_markets(limit=200)
        top_events = await client.fetch_top_geopolitical(limit=10)
    finally:
        await client.close()

    total_geo = len(all_geo)

    # Gather event IDs from top events for DB lookup
    top_event_ids = [str(e.get("id", "")) for e in top_events if e.get("id")]

    # Cross-reference: find active comparisons for these event IDs
    comparison_map: dict[str, tuple[str, float, float | None, str | None]] = {}
    if top_event_ids:
        stmt = (
            select(
                PolymarketComparison.polymarket_event_id,
                PolymarketComparison.geopol_prediction_id,
                PolymarketComparison.match_confidence,
                PolymarketComparison.geopol_probability,
            )
            .where(PolymarketComparison.polymarket_event_id.in_(top_event_ids))
            .where(PolymarketComparison.status == "active")
        )
        result = await db.execute(stmt)
        rows = result.all()

        # For matched rows, also fetch prediction question text
        pred_ids = [r[1] for r in rows]
        question_map: dict[str, str] = {}
        if pred_ids:
            pred_stmt = select(Prediction.id, Prediction.question).where(
                Prediction.id.in_(pred_ids)
            )
            pred_result = await db.execute(pred_stmt)
            question_map = {r[0]: r[1] for r in pred_result.all()}

        for event_id, pred_id, confidence, geopol_prob in rows:
            comparison_map[event_id] = (
                pred_id,
                confidence,
                geopol_prob,
                question_map.get(pred_id),
            )

    # Build response items
    items: list[PolymarketTopEventItem] = []
    for event in top_events:
        event_id = str(event.get("id", ""))
        try:
            volume = float(event.get("volume", "0") or "0")
        except (ValueError, TypeError):
            volume = 0.0
        try:
            liquidity = float(event.get("liquidity", "0") or "0")
        except (ValueError, TypeError):
            liquidity = 0.0

        match = comparison_map.get(event_id)
        items.append(
            PolymarketTopEventItem(
                event_id=event_id,
                title=event.get("title", ""),
                slug=event.get("slug", ""),
                volume=volume,
                liquidity=liquidity,
                geopol_prediction_id=match[0] if match else None,
                geopol_probability=match[2] if match else None,
                geopol_question=match[3] if match else None,
                match_confidence=match[1] if match else None,
            )
        )

    return PolymarketTopResponse(events=items, total_geo_markets=total_geo)


@router.get(
    "/weights",
    response_model=list[CalibrationWeightItem],
    summary="Current calibration weights",
    description="Returns all current per-CAMEO calibration weights with alpha, sample size, and Brier score.",
)
async def get_calibration_weights(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[CalibrationWeightItem]:
    """Query current calibration weights from the weights table."""
    result = await db.execute(
        select(CalibrationWeight).order_by(CalibrationWeight.cameo_code)
    )
    weights = result.scalars().all()

    return [
        CalibrationWeightItem(
            cameo_code=w.cameo_code,
            alpha=w.alpha,
            sample_size=w.sample_size,
            brier_score=w.brier_score,
            updated_at=w.updated_at.isoformat(),
        )
        for w in weights
    ]


@router.get(
    "/weights/history",
    response_model=list[CalibrationWeightHistoryItem],
    summary="Weight version history",
    description="Returns versioned history of calibration weight computations, with optional CAMEO code and limit filters.",
)
async def get_weight_history(
    cameo_code: Optional[str] = Query(
        default=None,
        description="Filter by CAMEO code (e.g. '14', 'material_conflict', 'global')",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum entries to return"),
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[CalibrationWeightHistoryItem]:
    """Query calibration weight history with optional filters."""
    stmt = select(CalibrationWeightHistory).order_by(
        CalibrationWeightHistory.computed_at.desc()
    )

    if cameo_code is not None:
        stmt = stmt.where(CalibrationWeightHistory.cameo_code == cameo_code)

    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        CalibrationWeightHistoryItem(
            id=row.id,
            cameo_code=row.cameo_code,
            alpha=row.alpha,
            sample_size=row.sample_size,
            brier_score=row.brier_score,
            computed_at=row.computed_at.isoformat(),
            auto_applied=row.auto_applied,
            flagged=row.flagged,
            flag_reason=row.flag_reason,
        )
        for row in rows
    ]
