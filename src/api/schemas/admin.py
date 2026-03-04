"""Pydantic DTOs for the admin API endpoints.

All schemas use ``model_config = ConfigDict(from_attributes=True)`` where
applicable, enabling direct ORM-to-DTO conversion via ``model_validate``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ProcessInfo(BaseModel):
    """Status summary for a background daemon / scheduled job."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    daemon_type: str
    status: str  # running | success | failed | unknown
    last_run: datetime | None = None
    next_run: datetime | None = None
    success_count: int = 0
    fail_count: int = 0


class ConfigEntry(BaseModel):
    """Single runtime-adjustable configuration value."""

    key: str
    value: Any
    type: str  # int | float | str | bool | list
    editable: bool = True
    dangerous: bool = False
    description: str = ""


class ConfigUpdate(BaseModel):
    """Batch update payload for PUT /admin/config."""

    updates: dict[str, Any]


class LogEntryDTO(BaseModel):
    """Structured log entry from the in-memory ring buffer."""

    timestamp: str
    severity: str
    module: str
    message: str


class SourceInfo(BaseModel):
    """Per-source health and enable/disable state."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    daemon_type: str
    enabled: bool = True
    healthy: bool = True
    last_run: datetime | None = None
    events_count: int = 0
    tier: str | None = None  # RSS feed tier classification
