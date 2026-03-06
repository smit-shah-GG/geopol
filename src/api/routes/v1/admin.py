"""Admin API router -- operational control plane for the geopol system.

All endpoints require the ``X-Admin-Key`` header matching the ``ADMIN_KEY``
environment variable. This is intentionally separate from the per-client
API key system (``X-API-Key`` + ``api_keys`` table).

Endpoints:
    POST /verify                         -- auth check
    GET  /processes                      -- daemon status table
    POST /processes/{daemon_type}/trigger -- fire job immediately
    POST /processes/{daemon_type}/pause  -- pause job (sets next_run=None)
    POST /processes/{daemon_type}/resume -- resume paused job
    GET  /jobs                           -- raw APScheduler job list
    GET  /config                         -- runtime settings
    PUT  /config                         -- batch update settings
    DELETE /config                       -- reset all overrides
    GET  /logs                           -- ring buffer entries
    GET  /sources                        -- per-source health
    PUT  /sources/{source_name}/toggle   -- enable/disable source
    GET  /feeds                          -- list all RSS feeds with health
    POST /feeds                          -- add a new RSS feed
    PUT  /feeds/{feed_id}                -- update feed properties
    DELETE /feeds/{feed_id}              -- soft/hard delete a feed
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.log_buffer import get_ring_buffer
from src.api.schemas.admin import (
    AddFeedRequest,
    ConfigEntry,
    ConfigUpdate,
    FeedInfo,
    LogEntryDTO,
    ProcessInfo,
    SourceInfo,
    UpdateFeedRequest,
)
from src.api.services.admin_service import AdminService
from src.settings import get_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Auth dependency
# -----------------------------------------------------------------------

async def verify_admin(
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
) -> None:
    """Validate the admin key header against Settings.admin_key.

    Raises:
        HTTPException 503: ADMIN_KEY not configured.
        HTTPException 401: Key missing or invalid.
    """
    settings = get_settings()
    if not settings.admin_key:
        raise HTTPException(
            503,
            "Admin access not configured -- set ADMIN_KEY env var",
        )
    if x_admin_key != settings.admin_key:
        raise HTTPException(401, "Invalid admin key")


# -----------------------------------------------------------------------
# Router (auth applied at router level)
# -----------------------------------------------------------------------

router = APIRouter(dependencies=[Depends(verify_admin)])


def _get_service(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AdminService:
    """FastAPI dependency: scoped AdminService per request.

    Injects the failure tracker from app.state if available (set during
    lifespan startup when APScheduler is initialized).
    """
    failure_tracker = getattr(request.app.state, "failure_tracker", None)
    return AdminService(db, failure_tracker=failure_tracker)


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------


@router.post("/verify")
async def verify() -> dict[str, str]:
    """Lightweight auth check -- returns 200 if admin key is valid."""
    return {"status": "authenticated"}


@router.get("/processes", response_model=list[ProcessInfo])
async def list_processes(
    svc: AdminService = Depends(_get_service),
) -> list[ProcessInfo]:
    """Return status summary for all background daemons."""
    return await svc.get_processes()


@router.post("/processes/{daemon_type}/trigger")
async def trigger_process(
    daemon_type: str,
    svc: AdminService = Depends(_get_service),
) -> dict[str, str]:
    """Fire a daemon's job(s) immediately via APScheduler."""
    await svc.trigger_job(daemon_type)
    return {"status": "triggered", "daemon_type": daemon_type}


@router.post("/processes/{daemon_type}/pause")
async def pause_process(
    daemon_type: str,
    svc: AdminService = Depends(_get_service),
) -> dict[str, str]:
    """Pause all APScheduler jobs for the specified daemon type."""
    await svc.pause_job(daemon_type)
    return {"status": "paused", "daemon_type": daemon_type}


@router.post("/processes/{daemon_type}/resume")
async def resume_process(
    daemon_type: str,
    svc: AdminService = Depends(_get_service),
) -> dict[str, str]:
    """Resume all paused APScheduler jobs for the specified daemon type."""
    await svc.resume_job(daemon_type)
    return {"status": "resumed", "daemon_type": daemon_type}


@router.get("/jobs")
async def list_jobs(
    svc: AdminService = Depends(_get_service),
) -> list[dict]:
    """Return all APScheduler jobs with their live state.

    Complement to /processes which groups by daemon_type -- this endpoint
    returns per-APScheduler-job info for debugging and monitoring.
    """
    return await svc.get_jobs()


@router.get("/config", response_model=list[ConfigEntry])
async def get_config(
    svc: AdminService = Depends(_get_service),
) -> list[ConfigEntry]:
    """Return all runtime settings with current effective values."""
    return await svc.get_config()


@router.put("/config")
async def update_config(
    body: ConfigUpdate,
    svc: AdminService = Depends(_get_service),
) -> dict[str, object]:
    """Persist config changes to system_config table."""
    await svc.update_config(body.updates)
    return {"status": "updated", "keys": list(body.updates.keys())}


@router.delete("/config")
async def reset_config(
    svc: AdminService = Depends(_get_service),
) -> dict[str, str]:
    """Delete all system_config rows, reverting to Settings defaults."""
    await svc.reset_config()
    return {"status": "reset"}


@router.get("/logs", response_model=list[LogEntryDTO])
async def get_logs(
    severity: str | None = Query(None, description="Filter by log level (e.g. ERROR)"),
    subsystem: str | None = Query(None, description="Substring match on module name"),
) -> list[LogEntryDTO]:
    """Return structured log entries from the in-memory ring buffer."""
    ring = get_ring_buffer()
    entries = ring.get_entries(severity=severity, subsystem=subsystem)
    return [
        LogEntryDTO(
            timestamp=e.timestamp,
            severity=e.severity,
            module=e.module,
            message=e.message,
        )
        for e in entries
    ]


@router.get("/sources", response_model=list[SourceInfo])
async def list_sources(
    svc: AdminService = Depends(_get_service),
) -> list[SourceInfo]:
    """Return per-source health with enable/disable state."""
    return await svc.get_sources()


@router.put("/sources/{source_name}/toggle")
async def toggle_source(
    source_name: str,
    body: dict[str, bool],
    svc: AdminService = Depends(_get_service),
) -> dict[str, object]:
    """Toggle a data source on or off."""
    enabled = body.get("enabled")
    if enabled is None:
        raise HTTPException(400, "Request body must include 'enabled' (bool)")
    await svc.toggle_source(source_name, enabled)
    return {"status": "toggled", "source": source_name, "enabled": enabled}


# -----------------------------------------------------------------------
# Feed CRUD (21-01)
# -----------------------------------------------------------------------


@router.get("/feeds", response_model=list[FeedInfo])
async def list_feeds(
    svc: AdminService = Depends(_get_service),
) -> list[FeedInfo]:
    """Return all non-deleted RSS feeds with per-feed health metrics."""
    return await svc.get_feeds()


@router.post("/feeds", response_model=FeedInfo, status_code=201)
async def add_feed(
    body: AddFeedRequest,
    svc: AdminService = Depends(_get_service),
) -> FeedInfo:
    """Add a new RSS feed to the registry."""
    return await svc.add_feed(body)


@router.put("/feeds/{feed_id}", response_model=FeedInfo)
async def update_feed(
    feed_id: int,
    body: UpdateFeedRequest,
    svc: AdminService = Depends(_get_service),
) -> FeedInfo:
    """Update feed properties (tier, enabled, category, etc.)."""
    return await svc.update_feed(feed_id, body)


@router.delete("/feeds/{feed_id}", status_code=204)
async def delete_feed(
    feed_id: int,
    purge: bool = Query(False, description="Hard-delete instead of soft-delete"),
    svc: AdminService = Depends(_get_service),
) -> Response:
    """Soft-delete a feed (or hard-delete with ?purge=true)."""
    await svc.delete_feed(feed_id, purge=purge)
    return Response(status_code=204)
