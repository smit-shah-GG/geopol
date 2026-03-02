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
