"""
Globe layer data endpoints.

Serves pre-computed layer data from PostgreSQL for the three globe
overlays: heatmap (H3 hexbins), arcs (bilateral relationships), and
deltas (risk change zones). All data is computed hourly by the seeding
heavy job -- these endpoints just read the latest rows.

All endpoints require API key authentication.

Endpoints:
    GET /globe/heatmap  -- H3 hexbin event density for heatmap layer
    GET /globe/arcs     -- Bilateral country arcs with sentiment
    GET /globe/deltas   -- Countries with significant risk changes
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.middleware.auth import verify_api_key
from src.db.models import CountryArc, HeatmapHexbin, RiskDelta

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Response models (simple DTOs, not worth separate schema files)
# ---------------------------------------------------------------------------


class HexbinResponse(BaseModel):
    """Single H3 hex cell with aggregated event weight."""

    h3_index: str = Field(..., description="H3 cell index string")
    weight: float = Field(..., description="Aggregated decay-weighted event intensity")
    event_count: int = Field(..., ge=0, description="Number of events in this cell")


class ArcResponse(BaseModel):
    """Bilateral country relationship arc."""

    source_iso: str = Field(..., min_length=2, max_length=2, description="Source country ISO alpha-2")
    target_iso: str = Field(..., min_length=2, max_length=2, description="Target country ISO alpha-2")
    event_count: int = Field(..., ge=0, description="Number of bilateral events")
    avg_goldstein: float = Field(
        ..., description="Average Goldstein score (negative=conflictual, positive=cooperative)"
    )


class RiskDeltaResponse(BaseModel):
    """Country with significant risk change in the last 7 days."""

    country_iso: str = Field(..., min_length=2, max_length=2, description="Country ISO alpha-2")
    current_risk: float = Field(..., ge=0.0, le=100.0, description="Current baseline risk score")
    previous_risk: float = Field(..., ge=0.0, le=100.0, description="Previous baseline risk score")
    delta: float = Field(..., description="Risk change (positive = deteriorating)")


class LayerEnvelope(BaseModel):
    """Wrapper with computed_at timestamp for staleness display."""

    computed_at: datetime | None = Field(
        None, description="Timestamp when this layer data was last computed"
    )


class HeatmapEnvelope(LayerEnvelope):
    """Heatmap layer response."""

    hexbins: list[HexbinResponse] = Field(default_factory=list)


class ArcsEnvelope(LayerEnvelope):
    """Arcs layer response."""

    arcs: list[ArcResponse] = Field(default_factory=list)


class DeltasEnvelope(LayerEnvelope):
    """Risk deltas layer response."""

    deltas: list[RiskDeltaResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/heatmap",
    response_model=HeatmapEnvelope,
    summary="Globe heatmap layer data",
    description=(
        "Returns pre-computed H3 hexbin data for the globe heatmap layer. "
        "Each hexbin represents aggregated, decay-weighted event intensity. "
        "Empty list if seeding hasn't run yet."
    ),
)
async def get_heatmap(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> HeatmapEnvelope:
    """Return all H3 hexbin data for the heatmap layer."""
    result = await db.execute(
        select(HeatmapHexbin).order_by(HeatmapHexbin.weight.desc())
    )
    rows = result.scalars().all()

    if not rows:
        return HeatmapEnvelope(computed_at=None, hexbins=[])

    hexbins = [
        HexbinResponse(
            h3_index=row.h3_index,
            weight=round(float(row.weight), 4),
            event_count=row.event_count,
        )
        for row in rows
    ]

    return HeatmapEnvelope(
        computed_at=rows[0].computed_at,
        hexbins=hexbins,
    )


@router.get(
    "/arcs",
    response_model=ArcsEnvelope,
    summary="Globe arc layer data",
    description=(
        "Returns pre-computed bilateral country relationship arcs. "
        "Color encodes sentiment: negative avg_goldstein = conflictual (red), "
        "positive = cooperative (blue). Width = event volume."
    ),
)
async def get_arcs(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> ArcsEnvelope:
    """Return all bilateral arcs for the globe arc layer."""
    result = await db.execute(
        select(CountryArc).order_by(CountryArc.event_count.desc())
    )
    rows = result.scalars().all()

    if not rows:
        return ArcsEnvelope(computed_at=None, arcs=[])

    arcs = [
        ArcResponse(
            source_iso=row.source_iso,
            target_iso=row.target_iso,
            event_count=row.event_count,
            avg_goldstein=round(float(row.avg_goldstein), 2),
        )
        for row in rows
    ]

    return ArcsEnvelope(
        computed_at=rows[0].computed_at,
        arcs=arcs,
    )


@router.get(
    "/deltas",
    response_model=DeltasEnvelope,
    summary="Risk change zones",
    description=(
        "Returns countries with significant risk changes (>10 points) "
        "in the last computation cycle. Positive delta = deteriorating, "
        "negative = improving. Used by the risk-change overlay layer."
    ),
)
async def get_deltas(
    _client: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
) -> DeltasEnvelope:
    """Return countries with significant risk deltas."""
    result = await db.execute(
        select(RiskDelta).order_by(RiskDelta.delta.desc())
    )
    rows = result.scalars().all()

    if not rows:
        return DeltasEnvelope(computed_at=None, deltas=[])

    deltas = [
        RiskDeltaResponse(
            country_iso=row.country_iso,
            current_risk=round(float(row.current_risk), 1),
            previous_risk=round(float(row.previous_risk), 1),
            delta=round(float(row.delta), 1),
        )
        for row in rows
    ]

    return DeltasEnvelope(
        computed_at=rows[0].computed_at,
        deltas=deltas,
    )
