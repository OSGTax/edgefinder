"""sync schema with ORM — add new columns and tables

Revision ID: 7b2bc613b751
Revises: b326fded4523
Create Date: 2026-04-09 21:13:03.539879

Brings the schema in sync with the v4.x ORM models after multiple
code-level changes that never got migrations. Handles both fresh DBs
(columns/tables may not exist) and existing DBs (columns may already
exist from render_start.py ALTER TABLE).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects import sqlite


# revision identifiers, used by Alembic.
revision: str = '7b2bc613b751'
down_revision: Union[str, Sequence[str], None] = 'b326fded4523'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    """Check if a column already exists (may have been added by render_start.py)."""
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table)]
    return column in columns


def _table_exists(table: str) -> bool:
    """Check if a table already exists."""
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def _add_column_if_missing(table: str, column: sa.Column) -> None:
    """Add a column only if it doesn't already exist."""
    if not _column_exists(table, column.name):
        op.add_column(table, column)


def upgrade() -> None:
    """Upgrade schema."""

    # ── tickers: add scan_batch ──
    _add_column_if_missing("tickers", sa.Column("scan_batch", sa.Integer(), nullable=True))

    # ── strategy_accounts: add realized_pnl ──
    _add_column_if_missing(
        "strategy_accounts",
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0.0"),
    )

    # ── fundamentals: add extended columns ──
    fund_columns = [
        # Extended ratios
        sa.Column("price_to_earnings", sa.Float(), nullable=True),
        sa.Column("price_to_book", sa.Float(), nullable=True),
        sa.Column("return_on_equity", sa.Float(), nullable=True),
        sa.Column("return_on_assets", sa.Float(), nullable=True),
        sa.Column("dividend_yield", sa.Float(), nullable=True),
        sa.Column("free_cash_flow", sa.Float(), nullable=True),
        sa.Column("quick_ratio", sa.Float(), nullable=True),
        # Short interest details
        sa.Column("short_shares", sa.Integer(), nullable=True),
        sa.Column("days_to_cover", sa.Float(), nullable=True),
        # Dividends
        sa.Column("dividend_amount", sa.Float(), nullable=True),
        sa.Column("ex_dividend_date", sa.String(20), nullable=True),
        # News sentiment
        sa.Column("news_sentiment", sa.String(20), nullable=True),
        # Technical indicators (from Massive API)
        sa.Column("rsi_14", sa.Float(), nullable=True),
        sa.Column("ema_21", sa.Float(), nullable=True),
        sa.Column("sma_50", sa.Float(), nullable=True),
        sa.Column("macd_value", sa.Float(), nullable=True),
        sa.Column("macd_signal", sa.Float(), nullable=True),
        sa.Column("macd_histogram", sa.Float(), nullable=True),
    ]
    for col in fund_columns:
        _add_column_if_missing("fundamentals", col)

    # ── ticker_strategy_qualifications: create table ──
    if not _table_exists("ticker_strategy_qualifications"):
        op.create_table(
            "ticker_strategy_qualifications",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("ticker_id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(10), nullable=False),
            sa.Column("strategy_name", sa.String(50), nullable=False),
            sa.Column("qualified", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("score", sa.Float(), nullable=True),
            sa.Column("scan_date", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(["ticker_id"], ["tickers.id"]),
            sa.UniqueConstraint("ticker_id", "strategy_name", name="uq_ticker_strategy"),
        )
        op.create_index("idx_tsq_strategy_qualified", "ticker_strategy_qualifications", ["strategy_name", "qualified"])
        op.create_index(op.f("ix_ticker_strategy_qualifications_symbol"), "ticker_strategy_qualifications", ["symbol"])
        op.create_index(op.f("ix_ticker_strategy_qualifications_strategy_name"), "ticker_strategy_qualifications", ["strategy_name"])

    # ── Drop stale sentiment_readings table ──
    if _table_exists("sentiment_readings"):
        op.drop_table("sentiment_readings")


def downgrade() -> None:
    """Downgrade schema — restore to initial schema state."""
    # Drop new table
    if _table_exists("ticker_strategy_qualifications"):
        op.drop_table("ticker_strategy_qualifications")

    # Remove added columns from fundamentals
    for col_name in [
        "price_to_earnings", "price_to_book", "return_on_equity", "return_on_assets",
        "dividend_yield", "free_cash_flow", "quick_ratio",
        "short_shares", "days_to_cover", "dividend_amount", "ex_dividend_date",
        "news_sentiment", "rsi_14", "ema_21", "sma_50",
        "macd_value", "macd_signal", "macd_histogram",
    ]:
        if _column_exists("fundamentals", col_name):
            op.drop_column("fundamentals", col_name)

    if _column_exists("tickers", "scan_batch"):
        op.drop_column("tickers", "scan_batch")

    if _column_exists("strategy_accounts", "realized_pnl"):
        op.drop_column("strategy_accounts", "realized_pnl")

    # Recreate sentiment_readings
    op.create_table(
        "sentiment_readings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("mention_count", sa.Integer(), nullable=False),
        sa.Column("is_trending", sa.Boolean(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
