"""
Full subsystem inventory health endpoint.

Reports status of all 8 canonical subsystems. Each check is wrapped
in try/except -- the health endpoint NEVER crashes regardless of
backend availability. A down subsystem is reported as unhealthy, not
as a 500 error.

Subsystems:
    1. database       -- async SELECT 1 on PostgreSQL
    2. redis          -- PING on Redis
    3. gdelt_store    -- GDELT SQLite file existence check
    4. graph_partitions -- partition count from SQLite partition index
    5. tkg_model      -- model checkpoint file existence
    6. last_ingest    -- most recent ingest_runs row
    7. last_prediction -- most recent predictions row
    8. api_budget     -- placeholder stub (always healthy)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas.health import HealthResponse, SubsystemStatus
from src.db.models import IngestRun, Prediction
from src.settings import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_database(db: AsyncSession) -> SubsystemStatus:
    """Attempt async SELECT 1 on PostgreSQL."""
    now = datetime.now(timezone.utc)
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return SubsystemStatus(
            name="database", healthy=True, detail="PostgreSQL OK", checked_at=now
        )
    except Exception as exc:
        logger.warning("Health check: database unhealthy: %s", exc)
        return SubsystemStatus(
            name="database", healthy=False, detail=str(exc)[:200], checked_at=now
        )


async def _check_redis(settings: Settings) -> SubsystemStatus:
    """Attempt PING on Redis."""
    now = datetime.now(timezone.utc)
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        try:
            pong = await client.ping()
            if pong:
                return SubsystemStatus(
                    name="redis", healthy=True, detail="Redis PONG", checked_at=now
                )
            return SubsystemStatus(
                name="redis", healthy=False, detail="PING returned False", checked_at=now
            )
        finally:
            await client.aclose()
    except Exception as exc:
        logger.warning("Health check: redis unhealthy: %s", exc)
        return SubsystemStatus(
            name="redis", healthy=False, detail=str(exc)[:200], checked_at=now
        )


def _check_gdelt_store(settings: Settings) -> SubsystemStatus:
    """Check if the GDELT SQLite file exists on disk."""
    now = datetime.now(timezone.utc)
    try:
        db_path = Path(settings.gdelt_db_path)
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            return SubsystemStatus(
                name="gdelt_store",
                healthy=True,
                detail=f"SQLite exists ({size_mb:.1f} MB)",
                checked_at=now,
            )
        return SubsystemStatus(
            name="gdelt_store",
            healthy=False,
            detail=f"File not found: {settings.gdelt_db_path}",
            checked_at=now,
        )
    except Exception as exc:
        return SubsystemStatus(
            name="gdelt_store", healthy=False, detail=str(exc)[:200], checked_at=now
        )


def _check_graph_partitions(settings: Settings) -> SubsystemStatus:
    """Query partition count from the SQLite partition index."""
    now = datetime.now(timezone.utc)
    try:
        # The partition index uses the same SQLite base path with a different file
        # Convention: data/partition_index.db alongside data/events.db
        index_path = Path(settings.gdelt_db_path).parent / "partition_index.db"
        if not index_path.exists():
            return SubsystemStatus(
                name="graph_partitions",
                healthy=False,
                detail="Partition index DB not found",
                checked_at=now,
            )

        import sqlite3

        conn = sqlite3.connect(str(index_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM partition_meta"
            )
            count = cursor.fetchone()[0]
            return SubsystemStatus(
                name="graph_partitions",
                healthy=count > 0,
                detail=f"{count} partitions indexed",
                checked_at=now,
            )
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Health check: graph_partitions unhealthy: %s", exc)
        return SubsystemStatus(
            name="graph_partitions",
            healthy=False,
            detail=str(exc)[:200],
            checked_at=now,
        )


def _check_tkg_model() -> SubsystemStatus:
    """Check if a TKG model checkpoint file exists."""
    now = datetime.now(timezone.utc)
    try:
        # Convention: model checkpoints in data/models/
        model_dir = Path("data/models")
        if model_dir.exists():
            checkpoints = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.params"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                return SubsystemStatus(
                    name="tkg_model",
                    healthy=True,
                    detail=f"Latest: {latest.name}",
                    checked_at=now,
                )
        return SubsystemStatus(
            name="tkg_model",
            healthy=False,
            detail="No model checkpoint found",
            checked_at=now,
        )
    except Exception as exc:
        return SubsystemStatus(
            name="tkg_model", healthy=False, detail=str(exc)[:200], checked_at=now
        )


async def _check_last_ingest(db: AsyncSession) -> SubsystemStatus:
    """Query the most recent ingest_runs row."""
    now = datetime.now(timezone.utc)
    try:
        result = await db.execute(
            select(IngestRun).order_by(IngestRun.started_at.desc()).limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return SubsystemStatus(
                name="last_ingest",
                healthy=False,
                detail="No ingest runs recorded",
                checked_at=now,
            )
        age_hours = (now - row.started_at.replace(tzinfo=timezone.utc if row.started_at.tzinfo is None else row.started_at.tzinfo)).total_seconds() / 3600
        healthy = age_hours < 24  # Stale if older than 24h
        return SubsystemStatus(
            name="last_ingest",
            healthy=healthy,
            detail=f"Last run: {row.started_at.isoformat()} ({row.status}, {age_hours:.1f}h ago)",
            checked_at=now,
        )
    except Exception as exc:
        logger.warning("Health check: last_ingest unhealthy: %s", exc)
        return SubsystemStatus(
            name="last_ingest", healthy=False, detail=str(exc)[:200], checked_at=now
        )


async def _check_last_prediction(db: AsyncSession) -> SubsystemStatus:
    """Query the most recent predictions row."""
    now = datetime.now(timezone.utc)
    try:
        result = await db.execute(
            select(Prediction).order_by(Prediction.created_at.desc()).limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return SubsystemStatus(
                name="last_prediction",
                healthy=False,
                detail="No predictions recorded",
                checked_at=now,
            )
        age_hours = (now - row.created_at.replace(tzinfo=timezone.utc if row.created_at.tzinfo is None else row.created_at.tzinfo)).total_seconds() / 3600
        healthy = age_hours < 48  # Stale if older than 48h
        return SubsystemStatus(
            name="last_prediction",
            healthy=healthy,
            detail=f"Last: {row.created_at.isoformat()} ({age_hours:.1f}h ago)",
            checked_at=now,
        )
    except Exception as exc:
        logger.warning("Health check: last_prediction unhealthy: %s", exc)
        return SubsystemStatus(
            name="last_prediction",
            healthy=False,
            detail=str(exc)[:200],
            checked_at=now,
        )


def _check_api_budget() -> SubsystemStatus:
    """Placeholder stub for API budget monitoring.

    Always reports healthy until Gemini API usage tracking is implemented.
    """
    now = datetime.now(timezone.utc)
    return SubsystemStatus(
        name="api_budget",
        healthy=True,
        detail="Budget tracking not yet implemented",
        checked_at=now,
    )


def _derive_status(subsystems: list[SubsystemStatus]) -> str:
    """Derive aggregate status from individual subsystem statuses.

    - "healthy": all subsystems healthy
    - "degraded": some unhealthy but core (database) still up
    - "unhealthy": database is down
    """
    unhealthy = [s for s in subsystems if not s.healthy]
    if not unhealthy:
        return "healthy"

    # If database itself is down, system is unhealthy
    db_status = next((s for s in subsystems if s.name == "database"), None)
    if db_status and not db_status.healthy:
        return "unhealthy"

    return "degraded"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Full subsystem health inventory",
    description=(
        "Reports status of all 8 subsystems. No authentication required. "
        "Used by load balancers, uptime monitors, and the frontend "
        "SystemHealthPanel."
    ),
)
async def health_check(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Run all 8 subsystem checks and return aggregate health."""
    subsystems: list[SubsystemStatus] = []

    # Run checks — order matches canonical subsystem list
    subsystems.append(await _check_database(db))
    subsystems.append(await _check_redis(settings))
    subsystems.append(_check_gdelt_store(settings))
    subsystems.append(_check_graph_partitions(settings))
    subsystems.append(_check_tkg_model())
    subsystems.append(await _check_last_ingest(db))
    subsystems.append(await _check_last_prediction(db))
    subsystems.append(_check_api_budget())

    status = _derive_status(subsystems)

    return HealthResponse(
        status=status,
        subsystems=subsystems,
        timestamp=datetime.now(timezone.utc),
        version="2.0.0-dev",
    )
