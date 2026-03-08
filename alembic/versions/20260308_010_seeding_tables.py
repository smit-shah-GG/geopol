"""Create seeding tables for Phase 24 global baseline risk and globe layers

Revision ID: 010
Revises: 009
Create Date: 2026-03-08

Schema for Phase 24 Global Seeding & Globe Layers:
- baseline_country_risk: Pre-computed baseline risk scores for all ~195 countries
- heatmap_hexbins: H3 hex-binned event density for globe heatmap layer
- country_arcs: Bilateral country relationship arcs with sentiment
- risk_deltas: 7-day risk change deltas for scenario/change overlay
- travel_advisories: Persisted advisory levels for cross-process access
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- baseline_country_risk --
    op.create_table(
        "baseline_country_risk",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("country_iso", sa.String(2), unique=True, nullable=False),
        sa.Column("baseline_risk", sa.Float(), nullable=False),
        sa.Column("gdelt_score", sa.Float(), nullable=False),
        sa.Column("acled_score", sa.Float(), nullable=False),
        sa.Column("advisory_score", sa.Float(), nullable=False),
        sa.Column("goldstein_score", sa.Float(), nullable=False),
        sa.Column(
            "advisory_level", sa.Integer(), nullable=False, server_default="1"
        ),
        sa.Column(
            "gdelt_event_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "acled_event_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "disputed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_baseline_country_risk_country_iso",
        "baseline_country_risk",
        ["country_iso"],
    )

    # -- heatmap_hexbins --
    op.create_table(
        "heatmap_hexbins",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("h3_index", sa.String(20), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_heatmap_hexbins_h3_index", "heatmap_hexbins", ["h3_index"]
    )
    op.create_index(
        "ix_heatmap_hexbins_computed_at", "heatmap_hexbins", ["computed_at"]
    )

    # -- country_arcs --
    op.create_table(
        "country_arcs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_iso", sa.String(2), nullable=False),
        sa.Column("target_iso", sa.String(2), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("avg_goldstein", sa.Float(), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_country_arcs_pair", "country_arcs", ["source_iso", "target_iso"]
    )

    # -- risk_deltas --
    op.create_table(
        "risk_deltas",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("country_iso", sa.String(2), nullable=False),
        sa.Column("current_risk", sa.Float(), nullable=False),
        sa.Column("previous_risk", sa.Float(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_risk_deltas_country_iso", "risk_deltas", ["country_iso"]
    )

    # -- travel_advisories --
    op.create_table(
        "travel_advisories",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("country_iso", sa.String(2), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "country_iso", "source", name="uq_travel_advisory_country_source"
        ),
    )
    op.create_index(
        "ix_travel_advisories_country_iso", "travel_advisories", ["country_iso"]
    )


def downgrade() -> None:
    op.drop_index("ix_travel_advisories_country_iso", table_name="travel_advisories")
    op.drop_table("travel_advisories")

    op.drop_index("ix_risk_deltas_country_iso", table_name="risk_deltas")
    op.drop_table("risk_deltas")

    op.drop_index("ix_country_arcs_pair", table_name="country_arcs")
    op.drop_table("country_arcs")

    op.drop_index("ix_heatmap_hexbins_computed_at", table_name="heatmap_hexbins")
    op.drop_index("ix_heatmap_hexbins_h3_index", table_name="heatmap_hexbins")
    op.drop_table("heatmap_hexbins")

    op.drop_index(
        "ix_baseline_country_risk_country_iso", table_name="baseline_country_risk"
    )
    op.drop_table("baseline_country_risk")
