"""forecast_requests table and predictions.question_tsv tsvector column

Revision ID: 004
Revises: 003
Create Date: 2026-03-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- forecast_requests --
    op.create_table(
        "forecast_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column(
            "country_iso_list",
            JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="30"),
        sa.Column(
            "category",
            sa.String(32),
            nullable=False,
            server_default="GENERAL",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("submitted_by", sa.String(100), nullable=False),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "prediction_ids",
            JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0"),
        sa.Column("parsed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_forecast_requests_submitted_by",
        "forecast_requests",
        ["submitted_by"],
    )
    op.create_index(
        "ix_forecast_requests_status",
        "forecast_requests",
        ["status"],
    )

    # -- predictions.question_tsv generated tsvector column --
    # PostgreSQL GENERATED ALWAYS AS ... STORED is not supported by Alembic's
    # cross-database add_column, so we use raw SQL.
    op.execute(
        """
        ALTER TABLE predictions
        ADD COLUMN question_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', question)) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX ix_predictions_question_tsv
        ON predictions USING GIN (question_tsv)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_predictions_question_tsv")
    op.execute("ALTER TABLE predictions DROP COLUMN IF EXISTS question_tsv")
    op.drop_index("ix_forecast_requests_status", table_name="forecast_requests")
    op.drop_index("ix_forecast_requests_submitted_by", table_name="forecast_requests")
    op.drop_table("forecast_requests")
