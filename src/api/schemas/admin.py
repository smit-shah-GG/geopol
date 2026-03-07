"""Pydantic DTOs for the admin API endpoints.

All schemas use ``model_config = ConfigDict(from_attributes=True)`` where
applicable, enabling direct ORM-to-DTO conversion via ``model_validate``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProcessInfo(BaseModel):
    """Status summary for a background daemon / scheduled job."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    daemon_type: str
    status: str  # running | scheduled | paused | success | failed | unknown
    last_run: datetime | None = None
    next_run: datetime | None = None
    success_count: int = 0
    fail_count: int = 0
    last_duration: float | None = None  # seconds, from failure tracker
    last_error: str | None = None  # last exception message
    consecutive_failures: int = 0  # from failure tracker
    paused: bool = False  # True when next_run_time is None (APScheduler paused)


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


# -----------------------------------------------------------------------
# Feed CRUD schemas (21-01)
# -----------------------------------------------------------------------


class FeedInfo(BaseModel):
    """Admin-facing RSS feed with health metrics."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    url: str
    tier: int
    category: str
    lang: str
    enabled: bool
    last_poll_at: str | None = None
    last_error: str | None = None
    error_count: int = 0
    articles_24h: int = 0
    articles_total: int = 0
    avg_articles_per_poll: float = 0.0
    created_at: str


class AddFeedRequest(BaseModel):
    """Payload for POST /admin/feeds."""

    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(..., min_length=1)
    tier: Literal[1, 2] = 2
    category: str = "regional"
    lang: str = "en"


class UpdateFeedRequest(BaseModel):
    """Payload for PUT /admin/feeds/{feed_id}. All fields optional."""

    name: str | None = None
    url: str | None = None
    tier: Literal[1, 2] | None = None
    category: str | None = None
    lang: str | None = None
    enabled: bool | None = None


# -----------------------------------------------------------------------
# Accuracy DTOs (22-03)
# -----------------------------------------------------------------------


class ResolvedComparisonDTO(BaseModel):
    """Single resolved or voided Polymarket comparison for the accuracy table."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    polymarket_title: str
    polymarket_event_id: str
    geopol_probability: float | None = None
    polymarket_price: float | None = None
    polymarket_outcome: float | None = None  # 1.0=Yes, 0.0=No, None=voided
    geopol_brier: float | None = None
    polymarket_brier: float | None = None
    winner: str | None = None  # "geopol" | "polymarket" | "draw" | None (voided)
    status: str  # "resolved" | "voided"
    resolved_at: str | None = None
    created_at: str
    country_iso: str | None = None
    category: str | None = None


class AccuracySummary(BaseModel):
    """Aggregate accuracy stats."""

    total_resolved: int = 0
    total_voided: int = 0
    geopol_wins: int = 0
    polymarket_wins: int = 0
    draws: int = 0
    geopol_cumulative_brier: float | None = None
    polymarket_cumulative_brier: float | None = None
    rolling_30d_geopol_brier: float | None = None
    rolling_30d_polymarket_brier: float | None = None
    rolling_30d_count: int = 0


class AccuracyResponse(BaseModel):
    """Full accuracy endpoint response: summary + resolved comparisons."""

    summary: AccuracySummary
    comparisons: list[ResolvedComparisonDTO]


# -----------------------------------------------------------------------
# Backtesting DTOs (23-02)
# -----------------------------------------------------------------------


class CheckpointInfo(BaseModel):
    """Model checkpoint metadata for TiRGN / RE-GCN comparison."""

    name: str
    model_type: str  # "tirgn" | "regcn"
    path: str
    metrics: dict[str, float] | None = None  # mrr, hits_at_1, etc.
    created_at: str | None = None


class StartBacktestRequest(BaseModel):
    """Payload for POST /admin/backtesting/runs."""

    label: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    window_size_days: int = Field(14, ge=7, le=90)
    slide_step_days: int = Field(7, ge=1, le=30)
    min_predictions_per_window: int = Field(3, ge=1, le=20)
    checkpoints: dict[str, str] = Field(
        ..., min_length=1,
        description="model_name -> checkpoint_filename, at least 1 entry",
    )


class BacktestRunDTO(BaseModel):
    """Backtest run metadata for admin listing and detail views."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    label: str
    description: str | None = None
    window_size_days: int
    slide_step_days: int
    min_predictions_per_window: int
    checkpoints_json: dict[str, Any] | None = None
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    total_windows: int = 0
    completed_windows: int = 0
    total_predictions: int = 0
    aggregate_brier: float | None = None
    aggregate_mrr: float | None = None
    vs_polymarket_record_json: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: str | None = None


class BacktestResultDTO(BaseModel):
    """Per-window evaluation metrics for a backtest run."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    window_start: str | None = None
    window_end: str | None = None
    prediction_start: str | None = None
    prediction_end: str | None = None
    checkpoint_name: str
    num_predictions: int
    brier_score: float | None = None
    mrr: float | None = None
    hits_at_1: float | None = None
    hits_at_10: float | None = None
    calibration_bins_json: dict[str, Any] | None = None
    prediction_details_json: list[dict[str, Any]] | None = None
    polymarket_brier: float | None = None
    geopol_vs_pm_wins: int | None = None
    pm_vs_geopol_wins: int | None = None
    weight_snapshot_json: dict[str, Any] | None = None
    created_at: str | None = None


class BacktestRunDetailDTO(BaseModel):
    """Drill-down response: run metadata + all window results."""

    run: BacktestRunDTO
    results: list[BacktestResultDTO]


class BacktestExportDTO(BaseModel):
    """Export format selection for backtest results."""

    format: Literal["csv", "json"]
    run_ids: list[str] | None = None
