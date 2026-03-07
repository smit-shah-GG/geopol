"""Create backtest_runs and backtest_results tables

Revision ID: 009
Revises: 008
Create Date: 2026-03-07

Schema for Phase 23 Historical Backtesting:
- backtest_runs: Walk-forward evaluation run metadata and lifecycle
- backtest_results: Per-window evaluation metrics (Brier, MRR, Hits@k,
  calibration bins, prediction details, Polymarket comparison)
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- backtest_runs --
    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("window_size_days", sa.Integer(), nullable=False, server_default="14"),
        sa.Column("slide_step_days", sa.Integer(), nullable=False, server_default="7"),
        sa.Column(
            "min_predictions_per_window",
            sa.Integer(),
            nullable=False,
            server_default="3",
        ),
        sa.Column(
            "checkpoints_json",
            sa.dialects.postgresql.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_windows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_windows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_predictions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("aggregate_brier", sa.Float(), nullable=True),
        sa.Column("aggregate_mrr", sa.Float(), nullable=True),
        sa.Column(
            "vs_polymarket_record_json",
            sa.dialects.postgresql.JSON(),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # -- backtest_results --
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("backtest_runs.id"),
            nullable=False,
        ),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checkpoint_name", sa.String(100), nullable=False),
        sa.Column("num_predictions", sa.Integer(), nullable=False),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column("mrr", sa.Float(), nullable=True),
        sa.Column("hits_at_1", sa.Float(), nullable=True),
        sa.Column("hits_at_10", sa.Float(), nullable=True),
        sa.Column(
            "calibration_bins_json",
            sa.dialects.postgresql.JSON(),
            nullable=True,
        ),
        sa.Column(
            "prediction_details_json",
            sa.dialects.postgresql.JSON(),
            nullable=True,
        ),
        sa.Column("polymarket_brier", sa.Float(), nullable=True),
        sa.Column("geopol_vs_pm_wins", sa.Integer(), nullable=True),
        sa.Column("pm_vs_geopol_wins", sa.Integer(), nullable=True),
        sa.Column(
            "weight_snapshot_json",
            sa.dialects.postgresql.JSON(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Indexes
    op.create_index(
        "ix_backtest_results_run_id", "backtest_results", ["run_id"]
    )
    op.create_index(
        "ix_backtest_results_run_id_window",
        "backtest_results",
        ["run_id", "window_start"],
    )


def downgrade() -> None:
    op.drop_index("ix_backtest_results_run_id_window", table_name="backtest_results")
    op.drop_index("ix_backtest_results_run_id", table_name="backtest_results")
    op.drop_table("backtest_results")
    op.drop_table("backtest_runs")
