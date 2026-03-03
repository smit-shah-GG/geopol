"""
Data source health auto-discovery endpoint.

Queries the PostgreSQL ``ingest_runs`` table to find the last run for each
known daemon type (gdelt, rss, acled, advisory).  Returns operational
status per source.  Sources that have never run are reported as unhealthy
with ``detail="Never run"``.

This endpoint is public (no API key required) -- it backs the SourcesPanel
health display which is visible before authentication.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas.source import SourceStatusDTO
from src.db.models import IngestRun

logger = logging.getLogger(__name__)

router = APIRouter()

# All known daemon types -- sources that should exist even if never run.
_KNOWN_DAEMON_TYPES = ("gdelt", "rss", "acled", "advisory")


@router.get(
    "",
    response_model=list[SourceStatusDTO],
    summary="List data source health",
    description=(
        "Auto-discovers all ingestion pipelines from the IngestRun audit table. "
        "Returns operational status for each known source: gdelt, rss, acled, "
        "advisory. Sources without any runs are reported as unhealthy. "
        "Adding a new backend source automatically appears here without "
        "frontend changes."
    ),
)
async def list_sources(
    db: AsyncSession = Depends(get_db),
) -> list[SourceStatusDTO]:
    """Return health status for all known data sources."""
    sources: list[SourceStatusDTO] = []

    for daemon_type in _KNOWN_DAEMON_TYPES:
        result = await db.execute(
            select(IngestRun)
            .where(IngestRun.daemon_type == daemon_type)
            .order_by(IngestRun.completed_at.desc())
            .limit(1)
        )
        last_run = result.scalar_one_or_none()

        if last_run is not None:
            sources.append(
                SourceStatusDTO(
                    name=daemon_type,
                    healthy=last_run.status == "success",
                    last_update=(
                        last_run.completed_at.isoformat()
                        if last_run.completed_at
                        else None
                    ),
                    events_last_run=last_run.events_new or 0,
                    detail=f"{last_run.status}: {last_run.events_new} new",
                )
            )
        else:
            sources.append(
                SourceStatusDTO(
                    name=daemon_type,
                    healthy=False,
                    last_update=None,
                    events_last_run=0,
                    detail="Never run",
                )
            )

    return sources
