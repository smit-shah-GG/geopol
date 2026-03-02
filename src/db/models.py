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
    polymarket_comparisons     -- Paired geopol-vs-Polymarket forecast tracking
    polymarket_snapshots       -- Time-series price/probability snapshots per comparison
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON
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
    )  # active | resolved
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
