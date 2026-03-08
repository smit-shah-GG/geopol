"""Add narrative_summary column to predictions table

Revision ID: 011
Revises: 010
Create Date: 2026-03-09

Adds a nullable Text column for LLM-generated analytical narratives
that summarize the forecast situation and key factors. Used by the
ScenarioExplorer root node for at-a-glance context.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("narrative_summary", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("predictions", "narrative_summary")
