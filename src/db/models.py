"""
SQLAlchemy 2.0 ORM models for PostgreSQL forecast persistence.

Tables:
    predictions        -- Forecast outputs with nested DTO JSON blobs
    outcome_records    -- Ground-truth resolution for calibration feedback
    calibration_weights-- Per-CAMEO ensemble alpha weights
    ingest_runs        -- Micro-batch GDELT ingest audit trail
    api_keys           -- Per-client API key authentication
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
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
        String(10), nullable=False, unique=True
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
    """Audit record for a micro-batch GDELT ingest cycle."""

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
    )  # running | success | failed
    events_fetched: Mapped[int] = mapped_column(Integer, default=0)
    events_new: Mapped[int] = mapped_column(Integer, default=0)
    events_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<IngestRun(id={self.id}, status={self.status!r}, "
            f"new={self.events_new})>"
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
