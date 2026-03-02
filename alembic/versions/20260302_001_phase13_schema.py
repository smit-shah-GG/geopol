"""Phase 13 schema: calibration history, polymarket tables, prediction cameo_root_code

Revision ID: 003
Revises: 002
Create Date: 2026-03-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- Widen calibration_weights.cameo_code from VARCHAR(10) to VARCHAR(30) --
    # Supports hierarchical keys: "01"-"20" root codes, "super:verbal_coop", "global"
    op.alter_column(
        "calibration_weights",
        "cameo_code",
        type_=sa.String(30),
        existing_type=sa.String(10),
        existing_nullable=False,
    )

    # -- Add cameo_root_code to predictions --
    op.add_column(
        "predictions",
        sa.Column("cameo_root_code", sa.String(4), nullable=True),
    )
    op.create_index(
        "ix_predictions_cameo_root",
        "predictions",
        ["cameo_root_code"],
    )

    # -- calibration_weight_history --
    op.create_table(
        "calibration_weight_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cameo_code", sa.String(30), nullable=False),
        sa.Column("alpha", sa.Float(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "auto_applied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "flagged",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("flag_reason", sa.Text(), nullable=True),
    )
    op.create_index("ix_cwh_computed", "calibration_weight_history", ["computed_at"])
    op.create_index("ix_cwh_cameo", "calibration_weight_history", ["cameo_code"])

    # -- polymarket_comparisons --
    op.create_table(
        "polymarket_comparisons",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("polymarket_event_id", sa.String(100), nullable=False),
        sa.Column("polymarket_slug", sa.String(200), nullable=False),
        sa.Column("polymarket_title", sa.Text(), nullable=False),
        sa.Column("geopol_prediction_id", sa.String(36), nullable=False),
        sa.Column("match_confidence", sa.Float(), nullable=False),
        sa.Column("polymarket_price", sa.Float(), nullable=True),
        sa.Column("geopol_probability", sa.Float(), nullable=True),
        sa.Column("last_snapshot_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="active",
        ),
        sa.Column("polymarket_outcome", sa.Float(), nullable=True),
        sa.Column("geopol_brier", sa.Float(), nullable=True),
        sa.Column("polymarket_brier", sa.Float(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_polymarket_comparisons_geopol_prediction_id",
        "polymarket_comparisons",
        ["geopol_prediction_id"],
    )
    op.create_index(
        "ix_polymarket_comparisons_status",
        "polymarket_comparisons",
        ["status"],
    )

    # -- polymarket_snapshots --
    op.create_table(
        "polymarket_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "comparison_id",
            sa.Integer(),
            sa.ForeignKey("polymarket_comparisons.id"),
            nullable=False,
        ),
        sa.Column("polymarket_price", sa.Float(), nullable=False),
        sa.Column("geopol_probability", sa.Float(), nullable=False),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_polymarket_snapshots_comparison_id",
        "polymarket_snapshots",
        ["comparison_id"],
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("polymarket_snapshots")
    op.drop_table("polymarket_comparisons")
    op.drop_table("calibration_weight_history")

    op.drop_index("ix_predictions_cameo_root", table_name="predictions")
    op.drop_column("predictions", "cameo_root_code")

    op.alter_column(
        "calibration_weights",
        "cameo_code",
        type_=sa.String(10),
        existing_type=sa.String(30),
        existing_nullable=False,
    )
