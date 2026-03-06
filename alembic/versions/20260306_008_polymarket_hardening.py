"""Add reforecasted_at column to predictions, create polymarket_accuracy table

Revision ID: 008
Revises: 007
Create Date: 2026-03-06

Schema changes for Phase 22 Polymarket Hardening:
- predictions.reforecasted_at: nullable timestamp tracking last re-forecast
  (fixes the created_at overwrite bug -- created_at is now immutable)
- polymarket_accuracy: append-only cumulative Brier score ledger
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- predictions: add reforecasted_at column --
    op.add_column(
        "predictions",
        sa.Column("reforecasted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_predictions_reforecasted_at", "predictions", ["reforecasted_at"]
    )

    # -- polymarket_accuracy table --
    op.create_table(
        "polymarket_accuracy",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("total_resolved", sa.Integer(), nullable=False),
        sa.Column("geopol_cumulative_brier", sa.Float(), nullable=False),
        sa.Column("polymarket_cumulative_brier", sa.Float(), nullable=False),
        sa.Column("geopol_wins", sa.Integer(), nullable=False),
        sa.Column("polymarket_wins", sa.Integer(), nullable=False),
        sa.Column("draws", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rolling_30d_geopol_brier", sa.Float(), nullable=True),
        sa.Column("rolling_30d_polymarket_brier", sa.Float(), nullable=True),
        sa.Column("rolling_30d_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("triggered_by_comparison_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("polymarket_accuracy")
    op.drop_index("ix_predictions_reforecasted_at", table_name="predictions")
    op.drop_column("predictions", "reforecasted_at")
