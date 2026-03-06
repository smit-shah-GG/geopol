"""Create rss_feeds table with seed data from feed_config.py

Revision ID: 007
Revises: 006
Create Date: 2026-03-06

Seeds the table with all feeds from src.ingest.feed_config.ALL_FEEDS,
preserving name, url, tier, category, and lang. This ensures the RSS
daemon has feeds to poll immediately after migration without requiring
manual admin intervention.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    rss_feeds = op.create_table(
        "rss_feeds",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("tier", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("category", sa.String(50), nullable=False, server_default="regional"),
        sa.Column("lang", sa.String(10), nullable=False, server_default="en"),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column("last_poll_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("articles_24h", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("articles_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "avg_articles_per_poll", sa.Float(), nullable=False, server_default="0.0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("tier IN (1, 2)", name="ck_rss_feeds_tier"),
    )

    # Seed all feeds from feed_config.py
    from src.ingest.feed_config import ALL_FEEDS

    seed_rows = [
        {
            "name": feed.name,
            "url": feed.url,
            "tier": int(feed.tier),
            "category": feed.category,
            "lang": feed.lang,
        }
        for feed in ALL_FEEDS
    ]

    if seed_rows:
        op.bulk_insert(rss_feeds, seed_rows)


def downgrade() -> None:
    op.drop_table("rss_feeds")
