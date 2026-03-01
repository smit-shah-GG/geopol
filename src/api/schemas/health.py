"""
Health check DTOs for GET /api/v1/health.

Reports subsystem status for load balancers, uptime monitors, and the
frontend health dashboard. The health endpoint is public (no auth required).
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Canonical subsystem names — the health endpoint always reports exactly
# these 8 subsystems. Adding a new subsystem is an API contract change.
SUBSYSTEM_NAMES: frozenset[str] = frozenset(
    {
        "database",
        "redis",
        "gdelt_store",
        "graph_partitions",
        "tkg_model",
        "last_ingest",
        "last_prediction",
        "api_budget",
    }
)


class SubsystemStatus(BaseModel):
    """Status of a single infrastructure subsystem.

    Each subsystem is checked independently. A failed check marks
    the subsystem as unhealthy with an optional detail string
    explaining the failure (e.g., "connection refused", "stale data:
    last ingest 48h ago").
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        ...,
        description="Subsystem identifier — must be one of the 8 canonical names",
    )
    healthy: bool = Field(..., description="Whether this subsystem is operational")
    detail: Optional[str] = Field(
        None,
        description="Human-readable status detail (error message, latency, etc.)",
    )
    checked_at: datetime = Field(
        ..., description="When this subsystem was last checked"
    )


class HealthResponse(BaseModel):
    """Aggregate health status of the forecasting system.

    status is derived from subsystem states:
    - "healthy": all subsystems healthy
    - "degraded": some subsystems unhealthy but core functionality works
    - "unhealthy": critical subsystems down (database, api_budget)

    The subsystems list always contains exactly 8 entries, one per
    canonical subsystem name.
    """

    model_config = ConfigDict(from_attributes=True)

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Aggregate system health status"
    )
    subsystems: list[SubsystemStatus] = Field(
        ..., description="Per-subsystem health status (exactly 8 entries)"
    )
    timestamp: datetime = Field(
        ..., description="When this health check was performed"
    )
    version: str = Field(..., description="API server version string")
