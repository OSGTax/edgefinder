"""add daily_bars table

Revision ID: a1b2c3d4e5f6
Revises: e7a2f91b4d08
Create Date: 2026-05-27 00:00:00.000000

Adds the daily_bars table that holds OHLCV bars backfilled from the
Massive flat-files day_aggs dumps. Idempotent: skips creation if the
table already exists (e.g. created by the backfill script's
checkfirst=True bootstrap).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "e7a2f91b4d08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    inspector = sa_inspect(op.get_bind())
    return table in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("daily_bars"):
        return
    op.create_table(
        "daily_bars",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("transactions", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=20), server_default="flatfiles"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("symbol", "date", name="uq_daily_bars_symbol_date"),
    )
    op.create_index("idx_daily_bars_symbol_date", "daily_bars", ["symbol", "date"])
    op.create_index("ix_daily_bars_symbol", "daily_bars", ["symbol"])
    op.create_index("ix_daily_bars_date", "daily_bars", ["date"])


def downgrade() -> None:
    if _table_exists("daily_bars"):
        op.drop_table("daily_bars")
