"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-03-01 14:43:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- predictions --
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("prediction", sa.Text(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("reasoning_summary", sa.Text(), nullable=False),
        sa.Column("evidence_count", sa.Integer(), server_default="0"),
        sa.Column(
            "scenarios_json",
            JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "ensemble_info_json",
            JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "calibration_json",
            JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "entities",
            JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("country_iso", sa.String(3), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_predictions_category", "predictions", ["category"])
    op.create_index("ix_predictions_country_iso", "predictions", ["country_iso"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index(
        "ix_predictions_country_created",
        "predictions",
        ["country_iso", "created_at"],
    )

    # -- outcome_records --
    op.create_table(
        "outcome_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("prediction_id", sa.String(36), nullable=False),
        sa.Column("outcome", sa.Float(), nullable=False),
        sa.Column("resolution_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolution_method", sa.String(50), nullable=False),
        sa.Column(
            "evidence_gdelt_ids",
            JSON(),
            server_default="[]",
        ),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_outcome_records_prediction_id",
        "outcome_records",
        ["prediction_id"],
    )

    # -- calibration_weights --
    op.create_table(
        "calibration_weights",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cameo_code", sa.String(10), nullable=False, unique=True),
        sa.Column("alpha", sa.Float(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    # -- ingest_runs --
    op.create_table(
        "ingest_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("events_fetched", sa.Integer(), server_default="0"),
        sa.Column("events_new", sa.Integer(), server_default="0"),
        sa.Column("events_duplicate", sa.Integer(), server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
    )

    # -- api_keys --
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(64), nullable=False, unique=True),
        sa.Column("client_name", sa.String(100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("revoked", sa.Boolean(), server_default=sa.text("false")),
    )
    op.create_index("ix_api_keys_key", "api_keys", ["key"])


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("ingest_runs")
    op.drop_table("calibration_weights")
    op.drop_table("outcome_records")
    op.drop_table("predictions")
