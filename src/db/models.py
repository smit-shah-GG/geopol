"""
SQLAlchemy 2.0 ORM models for PostgreSQL forecast persistence.

Tables:
    predictions                -- Forecast outputs with nested DTO JSON blobs
    outcome_records            -- Ground-truth resolution for calibration feedback
    calibration_weights        -- Per-CAMEO ensemble alpha weights (hierarchical keys)
    calibration_weight_history -- Versioned audit trail of weight computations
    ingest_runs                -- Micro-batch GDELT ingest audit trail
    api_keys                   -- Per-client API key authentication
    pending_questions          -- Queued questions for budget-exhaustion carryover
    forecast_requests          -- User-submitted forecast request queue + tracking
    polymarket_comparisons     -- Paired geopol-vs-Polymarket forecast tracking
    polymarket_snapshots       -- Time-series price/probability snapshots per comparison
    polymarket_accuracy        -- Cumulative Brier score snapshots (Geopol vs Polymarket)
    rss_feeds                  -- Admin-managed RSS feed registry with health metrics
    backtest_runs              -- Walk-forward evaluation run metadata and lifecycle
    backtest_results           -- Per-window evaluation metrics for a backtest run
    baseline_country_risk      -- Pre-computed baseline risk scores for all countries
    heatmap_hexbins            -- H3 hex-binned event density for globe heatmap layer
    country_arcs               -- Bilateral country relationship arcs for globe layer
    risk_deltas                -- 7-day risk change deltas for scenario/change overlay
    travel_advisories          -- Persisted travel advisory levels for cross-process access
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Timezone-aware UTC now -- avoids deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for all PostgreSQL ORM models."""

    pass


class Prediction(Base):
    """Persisted forecast output with full DTO reconstruction data."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    prediction: Mapped[str] = mapped_column(Text, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reasoning_summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)

    # LLM-generated 2-3 sentence analytical narrative of the forecast situation
    narrative_summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, default=None
    )

    # Full nested DTO blobs for ForecastResponse reconstruction
    scenarios_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=list
    )
    ensemble_info_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    calibration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    entities: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)

    country_iso: Mapped[Optional[str]] = mapped_column(
        String(3), nullable=True, index=True
    )
    cameo_root_code: Mapped[Optional[str]] = mapped_column(
        String(4), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Provenance: None (organic), "polymarket_driven", "polymarket_tracked"
    provenance: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True, index=True
    )
    # Direct dedup lookup for Polymarket-driven forecasts
    polymarket_event_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )
    # Timestamp of the most recent re-forecast (None if never re-forecasted).
    # Immutable created_at tracks original creation; this tracks reforecast activity.
    reforecasted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Full-text search on question -- generated column, GIN-indexed in migration 004
    question_tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', question)", persisted=True),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_predictions_country_created", "country_iso", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id!r}, question={self.question[:50]!r}..., "
            f"p={self.probability:.3f}, country={self.country_iso})>"
        )


class OutcomeRecord(Base):
    """Ground-truth resolution for a prediction, used in calibration feedback."""

    __tablename__ = "outcome_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    outcome: Mapped[float] = mapped_column(Float, nullable=False)
    resolution_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    resolution_method: Mapped[str] = mapped_column(String(50), nullable=False)
    evidence_gdelt_ids: Mapped[list[Any]] = mapped_column(
        JSON, default=list
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<OutcomeRecord(id={self.id}, prediction_id={self.prediction_id!r}, "
            f"outcome={self.outcome})>"
        )


class CalibrationWeight(Base):
    """Per-CAMEO-category ensemble alpha weight from outcome feedback."""

    __tablename__ = "calibration_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cameo_code: Mapped[str] = mapped_column(
        String(30), nullable=False, unique=True
    )
    alpha: Mapped[float] = mapped_column(Float, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<CalibrationWeight(cameo={self.cameo_code!r}, "
            f"alpha={self.alpha:.3f}, n={self.sample_size})>"
        )


class IngestRun(Base):
    """Audit record for a micro-batch ingest cycle (GDELT or RSS)."""

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # running | success | failed | interrupted
    daemon_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="gdelt"
    )  # gdelt | rss
    events_fetched: Mapped[int] = mapped_column(Integer, default=0)
    events_new: Mapped[int] = mapped_column(Integer, default=0)
    events_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<IngestRun(id={self.id}, daemon={self.daemon_type!r}, "
            f"status={self.status!r}, new={self.events_new})>"
        )


class PendingQuestion(Base):
    """Questions queued when Gemini budget is exhausted mid-pipeline.

    The daily forecast pipeline generates questions from GDELT events.
    When the Gemini API budget ceiling is hit, remaining questions are
    persisted here and prioritised in the next day's run.
    """

    __tablename__ = "pending_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    country_iso: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=21)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | processing | completed

    def __repr__(self) -> str:
        return (
            f"<PendingQuestion(id={self.id}, category={self.category!r}, "
            f"status={self.status!r})>"
        )


class ForecastRequest(Base):
    """User-submitted forecast request tracking through the generation pipeline.

    Lifecycle: pending -> confirmed -> processing -> complete | failed.
    Supports multi-country questions (country_iso_list is a JSON array of ISO codes).
    Each country generates a separate Prediction row; prediction_ids links back.
    """

    __tablename__ = "forecast_requests"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    country_iso_list: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    category: Mapped[str] = mapped_column(
        String(32), nullable=False, default="GENERAL"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | confirmed | processing | complete | failed
    submitted_by: Mapped[str] = mapped_column(String(100), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    prediction_ids: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    parsed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_forecast_requests_submitted_by", "submitted_by"),
        Index("ix_forecast_requests_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastRequest(id={self.id!r}, status={self.status!r}, "
            f"submitted_by={self.submitted_by!r})>"
        )


class ApiKey(Base):
    """Per-client API key for authentication."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    client_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    def __repr__(self) -> str:
        return (
            f"<ApiKey(id={self.id}, client={self.client_name!r}, "
            f"revoked={self.revoked})>"
        )


class CalibrationWeightHistory(Base):
    """Versioned audit trail of calibration weight computations.

    Every recompute writes a row here before updating calibration_weights.
    Rows with flagged=True exceeded deviation thresholds and may not have
    been auto-applied (check auto_applied).
    """

    __tablename__ = "calibration_weight_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cameo_code: Mapped[str] = mapped_column(String(30), nullable=False)
    alpha: Mapped[float] = mapped_column(Float, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    auto_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    flag_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_cwh_computed", "computed_at"),
        Index("ix_cwh_cameo", "cameo_code"),
    )

    def __repr__(self) -> str:
        return (
            f"<CalibrationWeightHistory(id={self.id}, cameo={self.cameo_code!r}, "
            f"alpha={self.alpha:.3f}, flagged={self.flagged})>"
        )


class PolymarketComparison(Base):
    """Paired tracking of a geopol prediction against a Polymarket event.

    Lifecycle: created as 'active' when matched, updated with snapshots,
    resolved when either market or prediction resolves. Brier scores
    computed on resolution for head-to-head comparison.
    """

    __tablename__ = "polymarket_comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    polymarket_event_id: Mapped[str] = mapped_column(String(100), nullable=False)
    polymarket_slug: Mapped[str] = mapped_column(String(200), nullable=False)
    polymarket_title: Mapped[str] = mapped_column(Text, nullable=False)
    geopol_prediction_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    match_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    polymarket_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    geopol_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_snapshot_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active"
    )  # active | resolved | voided
    polymarket_outcome: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    geopol_brier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    polymarket_brier: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("ix_polymarket_comparisons_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<PolymarketComparison(id={self.id}, "
            f"slug={self.polymarket_slug!r}, status={self.status!r})>"
        )


class SystemConfig(Base):
    """Runtime configuration overrides persisted to PostgreSQL.

    The admin dashboard reads/writes this table to adjust system behavior
    without redeployment. Keys map to Settings field names; values are
    JSON-encoded. When a key exists here it overrides the env-var default.
    DELETE all rows to revert to Settings defaults.
    """

    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    updated_by: Mapped[str] = mapped_column(String(100), default="system")

    def __repr__(self) -> str:
        return f"<SystemConfig(key={self.key!r}, updated_by={self.updated_by!r})>"


class PolymarketSnapshot(Base):
    """Point-in-time price/probability capture for a comparison pair.

    Collected at polymarket_poll_interval. Enables time-series divergence
    charts between geopol ensemble output and Polymarket market price.
    """

    __tablename__ = "polymarket_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    comparison_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("polymarket_comparisons.id"), nullable=False, index=True
    )
    polymarket_price: Mapped[float] = mapped_column(Float, nullable=False)
    geopol_probability: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<PolymarketSnapshot(id={self.id}, comparison={self.comparison_id}, "
            f"pm={self.polymarket_price:.3f}, gp={self.geopol_probability:.3f})>"
        )


class PolymarketAccuracy(Base):
    """Cumulative accuracy snapshot: Geopol vs Polymarket Brier scores.

    Append-only ledger. A new row is written each time a PolymarketComparison
    resolves, recording the running totals at that moment. Enables time-series
    accuracy curves without recomputing from raw comparison rows.
    """

    __tablename__ = "polymarket_accuracy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_resolved: Mapped[int] = mapped_column(Integer, nullable=False)
    geopol_cumulative_brier: Mapped[float] = mapped_column(Float, nullable=False)
    polymarket_cumulative_brier: Mapped[float] = mapped_column(Float, nullable=False)
    geopol_wins: Mapped[int] = mapped_column(Integer, nullable=False)
    polymarket_wins: Mapped[int] = mapped_column(Integer, nullable=False)
    draws: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rolling_30d_geopol_brier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    rolling_30d_polymarket_brier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    rolling_30d_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    triggered_by_comparison_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<PolymarketAccuracy(id={self.id}, resolved={self.total_resolved}, "
            f"geopol_brier={self.geopol_cumulative_brier:.4f}, "
            f"pm_brier={self.polymarket_cumulative_brier:.4f})>"
        )


class RSSFeed(Base):
    """Admin-managed RSS feed with health metrics and soft-delete.

    Feeds are seeded from ``feed_config.py`` via the 007 Alembic migration.
    The RSS daemon reads enabled, non-deleted feeds from this table at each
    poll cycle, falling back to ``feed_config.py`` constants if the DB is
    unreachable.

    Health metrics (error_count, articles_24h, last_poll_at, last_error)
    are updated after every poll cycle. Feeds auto-disable after 5
    consecutive poll failures.
    """

    __tablename__ = "rss_feeds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="regional")
    lang: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_poll_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    articles_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    articles_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_articles_per_poll: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint("tier IN (1, 2)", name="ck_rss_feeds_tier"),
    )

    def __repr__(self) -> str:
        return (
            f"<RSSFeed(id={self.id}, name={self.name!r}, "
            f"tier={self.tier}, enabled={self.enabled})>"
        )


class BacktestRun(Base):
    """A single walk-forward backtesting evaluation run.

    Lifecycle: pending -> running -> completed | cancelled | failed.
    Cancellation is DB-based: admin sets status='cancelling'; the runner
    polls between windows and transitions to 'cancelled' with partial
    results preserved.
    """

    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Run configuration (frozen at start)
    window_size_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=14
    )
    slide_step_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=7
    )
    min_predictions_per_window: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3
    )
    checkpoints_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    # e.g. {"tirgn": "tirgn_best.npz", "regcn": "regcn_jraph_best.npz"}

    # Run lifecycle
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | running | completed | cancelled | failed
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Progress tracking
    total_windows: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    completed_windows: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_predictions: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # Aggregate summary (computed on completion)
    aggregate_brier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    aggregate_mrr: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    vs_polymarket_record_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    # e.g. {"geopol_wins": 5, "polymarket_wins": 3, "draws": 1}

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<BacktestRun(id={self.id!r}, label={self.label!r}, "
            f"status={self.status!r}, windows={self.completed_windows}/{self.total_windows})>"
        )


class BacktestResult(Base):
    """Per-window evaluation metrics for a backtest run.

    Each row captures the metrics computed from re-predicting resolved
    predictions within a single evaluation window using a specific model
    checkpoint. Calibration weight snapshot and per-prediction details
    are stored as JSON for drill-down analysis.
    """

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("backtest_runs.id"), nullable=False, index=True
    )

    # Window definition
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    prediction_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    prediction_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Model identification
    checkpoint_name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    # "tirgn_best" or "regcn_jraph_best" etc.

    # Metrics
    num_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hits_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hits_at_10: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Calibration data (for reliability diagrams)
    calibration_bins_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    # {"bins": [...], "predicted_avg": [...], "observed_freq": [...], "counts": [...]}

    # Per-prediction details (for drill-down)
    prediction_details_json: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    # [{"prediction_id": "...", "question": "...", "predicted_prob": 0.7, ...}]

    # Polymarket comparison (for Geopol vs PM head-to-head)
    polymarket_brier: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    geopol_vs_pm_wins: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    pm_vs_geopol_wins: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    # Calibration weight state used for this window
    weight_snapshot_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    # {"global": 0.58, "super:verbal_conflict": 0.62, "14": 0.71, ...}

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    __table_args__ = (
        Index("ix_backtest_results_run_id_window", "run_id", "window_start"),
    )

    def __repr__(self) -> str:
        return (
            f"<BacktestResult(id={self.id}, run_id={self.run_id!r}, "
            f"checkpoint={self.checkpoint_name!r}, "
            f"brier={self.brier_score}, n={self.num_predictions})>"
        )


class BaselineCountryRisk(Base):
    """Pre-computed baseline risk score for a country.

    Scores are recomputed hourly by the seeding heavy job from 4 inputs:
    GDELT event density (per-capita), ACLED conflict intensity, travel
    advisory level, and Goldstein severity. UPSERT semantics on country_iso
    -- each recompute overwrites the previous row.
    """

    __tablename__ = "baseline_country_risk"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_iso: Mapped[str] = mapped_column(
        String(2), unique=True, nullable=False, index=True
    )
    baseline_risk: Mapped[float] = mapped_column(Float, nullable=False)
    gdelt_score: Mapped[float] = mapped_column(Float, nullable=False)
    acled_score: Mapped[float] = mapped_column(Float, nullable=False)
    advisory_score: Mapped[float] = mapped_column(Float, nullable=False)
    goldstein_score: Mapped[float] = mapped_column(Float, nullable=False)
    advisory_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    gdelt_event_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    acled_event_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    disputed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<BaselineCountryRisk(iso={self.country_iso!r}, "
            f"risk={self.baseline_risk:.1f}, advisory_level={self.advisory_level})>"
        )


class HeatmapHexbin(Base):
    """Pre-computed H3 hex bin for the globe heatmap layer.

    Each row represents an H3 cell with aggregated, time-decayed event
    weight. Recomputed hourly by the seeding heavy job.
    """

    __tablename__ = "heatmap_hexbins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    h3_index: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("ix_heatmap_hexbins_computed_at", "computed_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<HeatmapHexbin(h3={self.h3_index!r}, "
            f"weight={self.weight:.3f}, events={self.event_count})>"
        )


class CountryArc(Base):
    """Pre-computed bilateral country relationship for the globe arc layer.

    Derived from event actor pairs grouped by country ISO. Sentiment
    encoded via avg_goldstein (negative = conflictual, positive = cooperative).
    Recomputed hourly by the seeding heavy job.
    """

    __tablename__ = "country_arcs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_iso: Mapped[str] = mapped_column(String(2), nullable=False)
    target_iso: Mapped[str] = mapped_column(String(2), nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_goldstein: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("ix_country_arcs_pair", "source_iso", "target_iso"),
    )

    def __repr__(self) -> str:
        return (
            f"<CountryArc(src={self.source_iso!r}, tgt={self.target_iso!r}, "
            f"events={self.event_count}, goldstein={self.avg_goldstein:.2f})>"
        )


class RiskDelta(Base):
    """7-day risk change delta for a country.

    Shows where risk is increasing (positive delta) or decreasing
    (negative delta). Used by the scenario/risk-change globe overlay.
    Recomputed hourly by the seeding heavy job.
    """

    __tablename__ = "risk_deltas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_iso: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    current_risk: Mapped[float] = mapped_column(Float, nullable=False)
    previous_risk: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<RiskDelta(iso={self.country_iso!r}, "
            f"current={self.current_risk:.1f}, prev={self.previous_risk:.1f}, "
            f"delta={self.delta:+.1f})>"
        )


class TravelAdvisory(Base):
    """Persisted travel advisory for cross-process access.

    The advisory poller writes here AND to the in-memory AdvisoryStore.
    The baseline risk heavy job (in ProcessPoolExecutor) reads from this
    table since it cannot access main process memory.
    UPSERT on (country_iso, source) -- latest advisory level wins.
    """

    __tablename__ = "travel_advisories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_iso: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("country_iso", "source", name="uq_travel_advisory_country_source"),
    )

    def __repr__(self) -> str:
        return (
            f"<TravelAdvisory(iso={self.country_iso!r}, "
            f"source={self.source!r}, level={self.level})>"
        )
