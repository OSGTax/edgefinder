"""drop sentiment_score column and trade_context table

Revision ID: b7e2f8a3c5d1
Revises: a1d9f3c0b2e4
Create Date: 2026-04-23 19:45:00.000000

- `trades.sentiment_score` was always NULL in production — no code path
  populated it. `trades.sentiment_data` is preserved and now gets wired
  from `fundamentals.news_sentiment` at trade-open time.
- `trade_context` was a write-only table: the capture logic was run on
  every trade.opened event but nothing ever read the rows. Dropping.

Idempotent.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect


revision: str = "b7e2f8a3c5d1"
down_revision: Union[str, Sequence[str], None] = "a1d9f3c0b2e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    if table not in inspector.get_table_names():
        return False
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    if _table_exists("trade_context"):
        op.drop_table("trade_context")

    if _column_exists("trades", "sentiment_score"):
        with op.batch_alter_table("trades") as batch_op:
            batch_op.drop_column("sentiment_score")


def downgrade() -> None:
    if not _column_exists("trades", "sentiment_score"):
        with op.batch_alter_table("trades") as batch_op:
            batch_op.add_column(sa.Column("sentiment_score", sa.Float(), nullable=True))

    if not _table_exists("trade_context"):
        op.create_table(
            "trade_context",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "trade_id",
                sa.String(length=36),
                sa.ForeignKey("trades.trade_id"),
                unique=True,
            ),
            sa.Column("recent_news", sa.JSON(), nullable=True),
            sa.Column("sector_prices", sa.JSON(), nullable=True),
            sa.Column("related_tickers", sa.JSON(), nullable=True),
            sa.Column("short_interest", sa.JSON(), nullable=True),
            sa.Column("dividends", sa.JSON(), nullable=True),
            sa.Column("indicators", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
