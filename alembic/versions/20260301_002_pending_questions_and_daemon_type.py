"""pending_questions table and daemon_type column on ingest_runs

Revision ID: 002
Revises: 001
Create Date: 2026-03-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- pending_questions --
    op.create_table(
        "pending_questions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("country_iso", sa.String(3), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="21"),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("priority", sa.Integer(), server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("status", sa.String(20), server_default="'pending'"),
    )

    # -- daemon_type on ingest_runs --
    op.add_column(
        "ingest_runs",
        sa.Column(
            "daemon_type",
            sa.String(20),
            nullable=False,
            server_default="gdelt",
        ),
    )


def downgrade() -> None:
    op.drop_column("ingest_runs", "daemon_type")
    op.drop_table("pending_questions")
