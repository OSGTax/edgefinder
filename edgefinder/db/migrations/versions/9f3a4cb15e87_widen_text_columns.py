"""widen text columns to TEXT (unbounded)

Revision ID: 9f3a4cb15e87
Revises: 3771022853b0
Create Date: 2026-05-28 15:20:00.000000

ticker_news.title, ticker_news.article_url, and agent_actions.summary were
declared as VARCHAR(500), but real-world news URLs and titles in 2026
regularly exceed 500 chars (tracking-laden URLs especially). That produced
repeated `value too long for type character varying(500)` insert errors
in prod every time the news accumulator job ran. TEXT and VARCHAR have
identical storage/performance characteristics in Postgres; this just
removes the length cap.

batch_alter_table keeps the migration runnable on both Postgres (online)
and SQLite (--sql / dev) — SQLite handles ALTER COLUMN via table copy
under the hood.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "9f3a4cb15e87"
down_revision: Union[str, Sequence[str], None] = "3771022853b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("ticker_news") as batch:
        batch.alter_column(
            "title",
            existing_type=sa.String(length=500),
            type_=sa.Text(),
            existing_nullable=False,
        )
        batch.alter_column(
            "article_url",
            existing_type=sa.String(length=500),
            type_=sa.Text(),
            existing_nullable=True,
        )
    with op.batch_alter_table("agent_actions") as batch:
        batch.alter_column(
            "summary",
            existing_type=sa.String(length=500),
            type_=sa.Text(),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("ticker_news") as batch:
        batch.alter_column(
            "title",
            existing_type=sa.Text(),
            type_=sa.String(length=500),
            existing_nullable=False,
        )
        batch.alter_column(
            "article_url",
            existing_type=sa.Text(),
            type_=sa.String(length=500),
            existing_nullable=True,
        )
    with op.batch_alter_table("agent_actions") as batch:
        batch.alter_column(
            "summary",
            existing_type=sa.Text(),
            type_=sa.String(length=500),
            existing_nullable=False,
        )
