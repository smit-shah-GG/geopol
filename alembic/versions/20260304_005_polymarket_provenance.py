"""Add provenance and polymarket_event_id columns to predictions

Revision ID: 005
Revises: 004
Create Date: 2026-03-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- predictions.provenance --
    op.add_column(
        "predictions",
        sa.Column("provenance", sa.String(30), nullable=True),
    )
    op.create_index(
        "ix_predictions_provenance",
        "predictions",
        ["provenance"],
    )

    # -- predictions.polymarket_event_id --
    op.add_column(
        "predictions",
        sa.Column("polymarket_event_id", sa.String(100), nullable=True),
    )
    op.create_index(
        "ix_predictions_polymarket_event_id",
        "predictions",
        ["polymarket_event_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_polymarket_event_id", table_name="predictions")
    op.drop_column("predictions", "polymarket_event_id")
    op.drop_index("ix_predictions_provenance", table_name="predictions")
    op.drop_column("predictions", "provenance")
