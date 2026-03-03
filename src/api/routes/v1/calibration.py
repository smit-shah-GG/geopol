"""
Calibration API endpoints: Polymarket comparison and weight management.

Endpoints:
    GET /calibration/polymarket -- Active/resolved comparisons + summary stats
    GET /calibration/polymarket/top -- Top geopolitical Polymarket events by volume
    GET /calibration/polymarket/comparisons -- All comparisons for ComparisonPanel
    GET /calibration/polymarket/comparisons/{id}/snapshots -- Sparkline time-series
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


class ComparisonPanelItem(BaseModel):
    """Single comparison for the ComparisonPanel (active or resolved)."""

    id: int
    polymarket_event_id: str
    polymarket_slug: str
    polymarket_title: str
    geopol_prediction_id: str
    match_confidence: float
    polymarket_price: float | None
    geopol_probability: float | None
    divergence: float | None  # computed: geopol - polymarket
    status: str  # active | resolved
    provenance: str  # polymarket_driven | polymarket_tracked
    geopol_brier: float | None = None
    polymarket_brier: float | None = None
    polymarket_outcome: float | None = None
    resolved_at: str | None = None
    created_at: str


class ComparisonPanelResponse(BaseModel):
    """Full response for GET /calibration/polymarket/comparisons."""

    comparisons: list[ComparisonPanelItem]
    total: int


class SnapshotPoint(BaseModel):
    """Single time-series data point for sparkline rendering."""

    polymarket_price: float
    geopol_probability: float
    captured_at: str


class SnapshotResponse(BaseModel):
    """Time-series snapshot data for a comparison pair."""

    comparison_id: int
    snapshots: list[SnapshotPoint]
    total_available: int  # total snapshots before sampling


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
        top_events, total_geo = await client.fetch_top_geopolitical(
            limit=10, fetch_limit=100,
        )
    finally:
        await client.close()

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

    # Build response items — volume/liquidity are already floats from the API
    items: list[PolymarketTopEventItem] = []
    for event in top_events:
        event_id = str(event.get("id", ""))
        try:
            volume = float(event.get("volume", 0) or 0)
        except (ValueError, TypeError):
            volume = 0.0
        try:
            liquidity = float(event.get("liquidity", 0) or 0)
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
    "/polymarket/comparisons",
    response_model=ComparisonPanelResponse,
    summary="All comparisons for ComparisonPanel",
    description=(
        "Returns all active and resolved Polymarket-vs-Geopol comparisons "
        "in a single list ordered by created_at DESC, with divergence and "
        "provenance computed per item. Designed for the ComparisonPanel UI."
    ),
)
async def get_comparison_panel(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> ComparisonPanelResponse:
    """Fetch all comparisons with provenance for the ComparisonPanel."""
    from src.polymarket.comparison import PolymarketComparisonService
    from contextlib import asynccontextmanager
    from src.db.models import Prediction

    @asynccontextmanager
    async def _session_wrapper():
        yield db

    service = PolymarketComparisonService.__new__(PolymarketComparisonService)
    service._session_factory = _session_wrapper

    all_items = await service.get_all_comparisons()

    if not all_items:
        return ComparisonPanelResponse(comparisons=[], total=0)

    # Batch-fetch provenance from Prediction rows for all comparison items
    pred_ids = [item["geopol_prediction_id"] for item in all_items]
    prov_stmt = select(Prediction.id, Prediction.provenance).where(
        Prediction.id.in_(pred_ids)
    )
    prov_result = await db.execute(prov_stmt)
    provenance_map: dict[str, str | None] = {
        pid: prov for pid, prov in prov_result.all()
    }

    comparisons: list[ComparisonPanelItem] = []
    for item in all_items:
        pred_prov = provenance_map.get(item["geopol_prediction_id"])
        if pred_prov and pred_prov.startswith("polymarket"):
            provenance = pred_prov
        else:
            provenance = "polymarket_tracked"

        # Compute divergence
        gp = item.get("geopol_probability")
        pp = item.get("polymarket_price")
        divergence: float | None = None
        if gp is not None and pp is not None:
            divergence = round(gp - pp, 4)

        comparisons.append(
            ComparisonPanelItem(
                id=item["id"],
                polymarket_event_id=item["polymarket_event_id"],
                polymarket_slug=item.get("polymarket_slug", ""),
                polymarket_title=item["polymarket_title"],
                geopol_prediction_id=item["geopol_prediction_id"],
                match_confidence=item["match_confidence"],
                polymarket_price=pp,
                geopol_probability=gp,
                divergence=divergence,
                status=item["status"],
                provenance=provenance,
                geopol_brier=item.get("geopol_brier"),
                polymarket_brier=item.get("polymarket_brier"),
                polymarket_outcome=item.get("polymarket_outcome"),
                resolved_at=item.get("resolved_at"),
                created_at=item["created_at"],
            )
        )

    return ComparisonPanelResponse(comparisons=comparisons, total=len(comparisons))


@router.get(
    "/polymarket/comparisons/{comparison_id}/snapshots",
    response_model=SnapshotResponse,
    summary="Sparkline snapshot data for a comparison",
    description=(
        "Returns sampled time-series snapshot data for a single comparison pair. "
        "If more than `limit` snapshots exist, evenly samples to get exactly "
        "`limit` points spanning the full time range."
    ),
)
async def get_comparison_snapshots(
    comparison_id: int,
    limit: int = Query(default=30, ge=1, le=100, description="Maximum data points"),
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> SnapshotResponse:
    """Fetch sampled snapshots for sparkline rendering."""
    from src.polymarket.comparison import PolymarketComparisonService
    from contextlib import asynccontextmanager
    from sqlalchemy import func as sa_func
    from src.db.models import PolymarketSnapshot

    @asynccontextmanager
    async def _session_wrapper():
        yield db

    service = PolymarketComparisonService.__new__(PolymarketComparisonService)
    service._session_factory = _session_wrapper

    # Get total count for response metadata
    count_stmt = (
        select(sa_func.count())
        .where(PolymarketSnapshot.comparison_id == comparison_id)
        .select_from(PolymarketSnapshot)
    )
    count_result = await db.execute(count_stmt)
    total_available = count_result.scalar() or 0

    snapshots_data = await service.get_snapshots_for_comparison(comparison_id, limit)

    snapshots = [
        SnapshotPoint(
            polymarket_price=s["polymarket_price"],
            geopol_probability=s["geopol_probability"],
            captured_at=s["captured_at"],
        )
        for s in snapshots_data
    ]

    return SnapshotResponse(
        comparison_id=comparison_id,
        snapshots=snapshots,
        total_available=total_available,
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
