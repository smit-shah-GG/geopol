"""Admin API router -- operational control plane for the geopol system.

All endpoints require the ``X-Admin-Key`` header matching the ``ADMIN_KEY``
environment variable. This is intentionally separate from the per-client
API key system (``X-API-Key`` + ``api_keys`` table).

Endpoints:
    POST /verify                         -- auth check
    GET  /processes                      -- daemon status table
    POST /processes/{daemon_type}/trigger -- spawn one-shot job
    GET  /config                         -- runtime settings
    PUT  /config                         -- batch update settings
    DELETE /config                       -- reset all overrides
    GET  /logs                           -- ring buffer entries
    GET  /sources                        -- per-source health
    PUT  /sources/{source_name}/toggle   -- enable/disable source
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.log_buffer import get_ring_buffer
from src.api.schemas.admin import (
    ConfigEntry,
    ConfigUpdate,
    LogEntryDTO,
    ProcessInfo,
    SourceInfo,
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


def _get_service(db: AsyncSession = Depends(get_db)) -> AdminService:
    """FastAPI dependency: scoped AdminService per request."""
    return AdminService(db)


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
    """Spawn a one-shot asyncio.Task for the specified daemon."""
    await svc.trigger_job(daemon_type)
    return {"status": "triggered", "daemon_type": daemon_type}


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
