"""
Full subsystem inventory health endpoint.

Reports status of all 10 canonical subsystems. Each check is wrapped
in try/except -- the health endpoint NEVER crashes regardless of
backend availability. A down subsystem is reported as unhealthy, not
as a 500 error.

Subsystems:
    1.  database           -- async SELECT 1 on PostgreSQL
    2.  redis              -- PING on Redis
    3.  gdelt_store        -- GDELT SQLite file existence check
    4.  graph_partitions   -- partition count from SQLite partition index
    5.  tkg_model          -- model checkpoint file existence
    6.  last_ingest        -- most recent ingest_runs row (stale >24h)
    7.  last_prediction    -- most recent predictions row (stale >48h)
    8.  api_budget         -- real Gemini budget via BudgetMonitor
    9.  disk_usage         -- root partition usage via DiskMonitor
    10. calibration_freshness -- recency of calibration_weight_history
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas.health import HealthResponse, SubsystemStatus
from src.db.models import CalibrationWeightHistory, IngestRun, Prediction
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
    """Query the most recent ingest_runs row with staleness check."""
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
        started = row.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_hours = (now - started).total_seconds() / 3600
        healthy = age_hours < 24  # Stale if older than 24h
        return SubsystemStatus(
            name="last_ingest",
            healthy=healthy,
            detail=f"Last run: {started.isoformat()} ({row.status}, {age_hours:.1f}h ago)",
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
        created = row.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_hours = (now - created).total_seconds() / 3600
        healthy = age_hours < 48  # Stale if older than 48h
        return SubsystemStatus(
            name="last_prediction",
            healthy=healthy,
            detail=f"Last: {created.isoformat()} ({age_hours:.1f}h ago)",
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


async def _check_api_budget(db: AsyncSession, settings: Settings) -> SubsystemStatus:
    """Check real Gemini API budget via prediction count for today.

    Uses the same counting logic as BudgetMonitor.get_budget_status():
    counts predictions created since midnight UTC against the daily budget.
    Healthy if budget_remaining > 0.
    """
    now = datetime.now(timezone.utc)
    try:
        from sqlalchemy import func

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        result = await db.execute(
            select(func.count(Prediction.id)).where(
                Prediction.created_at >= today_start
            )
        )
        used = result.scalar_one()
        total = settings.gemini_daily_budget
        remaining = max(0, total - used)
        pct_used = (used / total * 100) if total > 0 else 0.0
        healthy = remaining > 0
        return SubsystemStatus(
            name="api_budget",
            healthy=healthy,
            detail=f"{used}/{total} used ({pct_used:.0f}%), {remaining} remaining",
            checked_at=now,
        )
    except Exception as exc:
        logger.warning("Health check: api_budget unhealthy: %s", exc)
        return SubsystemStatus(
            name="api_budget",
            healthy=False,
            detail=str(exc)[:200],
            checked_at=now,
        )


def _check_disk_usage(settings: Settings) -> SubsystemStatus:
    """Check root partition disk usage via psutil.

    Status mapping for health derivation:
    - ok:       healthy=True
    - warning:  healthy=True (degraded, not critical)
    - critical: healthy=False (contributes to "degraded" not "unhealthy")
    """
    now = datetime.now(timezone.utc)
    try:
        usage = psutil.disk_usage("/")
        pct = usage.percent
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)

        if pct >= settings.disk_critical_pct:
            disk_status = "critical"
            healthy = False
        elif pct >= settings.disk_warning_pct:
            disk_status = "warning"
            # Warning is not critical -- still healthy from system perspective
            healthy = True
        else:
            disk_status = "ok"
            healthy = True

        return SubsystemStatus(
            name="disk_usage",
            healthy=healthy,
            detail=f"{pct:.1f}% used ({free_gb:.1f} GB free / {total_gb:.0f} GB), status={disk_status}",
            checked_at=now,
        )
    except Exception as exc:
        logger.warning("Health check: disk_usage error: %s", exc)
        return SubsystemStatus(
            name="disk_usage",
            healthy=False,
            detail=str(exc)[:200],
            checked_at=now,
        )


async def _check_calibration_freshness(db: AsyncSession) -> SubsystemStatus:
    """Check recency of calibration weight computations.

    Queries calibration_weight_history for the most recent row.
    Stale if the most recent computation is older than 14 days.
    """
    now = datetime.now(timezone.utc)
    try:
        result = await db.execute(
            select(CalibrationWeightHistory)
            .order_by(CalibrationWeightHistory.computed_at.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return SubsystemStatus(
                name="calibration_freshness",
                healthy=False,
                detail="No calibration history (cold start)",
                checked_at=now,
            )
        computed = row.computed_at
        if computed.tzinfo is None:
            computed = computed.replace(tzinfo=timezone.utc)
        age_days = (now - computed).total_seconds() / 86400
        healthy = age_days <= 14.0
        return SubsystemStatus(
            name="calibration_freshness",
            healthy=healthy,
            detail=f"Last calibration: {computed.isoformat()} ({age_days:.1f}d ago)",
            checked_at=now,
        )
    except Exception as exc:
        logger.warning("Health check: calibration_freshness error: %s", exc)
        return SubsystemStatus(
            name="calibration_freshness",
            healthy=False,
            detail=str(exc)[:200],
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
        "Reports status of all 10 subsystems. No authentication required. "
        "Used by load balancers, uptime monitors, and the frontend "
        "SystemHealthPanel."
    ),
)
async def health_check(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Run all 10 subsystem checks and return aggregate health."""
    subsystems: list[SubsystemStatus] = []

    # Run checks -- order matches canonical subsystem list
    subsystems.append(await _check_database(db))
    subsystems.append(await _check_redis(settings))
    subsystems.append(_check_gdelt_store(settings))
    subsystems.append(_check_graph_partitions(settings))
    subsystems.append(_check_tkg_model())
    subsystems.append(await _check_last_ingest(db))
    subsystems.append(await _check_last_prediction(db))
    subsystems.append(await _check_api_budget(db, settings))
    subsystems.append(_check_disk_usage(settings))
    subsystems.append(await _check_calibration_freshness(db))

    status = _derive_status(subsystems)

    return HealthResponse(
        status=status,
        subsystems=subsystems,
        timestamp=datetime.now(timezone.utc),
        version="2.0.0-dev",
    )
