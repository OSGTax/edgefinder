"""add real-money execution columns

Revision ID: a7c1e9f3d602
Revises: d4f8a1c93b27
Create Date: 2026-06-15

The real-money book (engine/live_ticket — a Claude-mediated, manual-approve
monthly rebalance on Robinhood via its MCP server) needs to be tagged so it
NEVER mixes with the paper fleet:

- promoted_strategies.execution_mode  "paper" (default) | "live_manual"
- trades.broker                       broker name on a real-money fill (NULL = paper)
- trades.broker_order_id              the broker order id (audit + idempotency)

All idempotent (column-exists-guarded) and nullable / defaulted, so paper
promotions and simulated trades are untouched, and the migration no-ops on a
fresh DB where create_all already built the columns from the ORM.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

# revision identifiers, used by Alembic.
revision: str = "a7c1e9f3d602"
down_revision: Union[str, Sequence[str], None] = "d4f8a1c93b27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    inspector = sa_inspect(op.get_bind())
    return table in inspector.get_table_names()


def _column_exists(table: str, column: str) -> bool:
    inspector = sa_inspect(op.get_bind())
    return column in [c["name"] for c in inspector.get_columns(table)]


def _add_column_if_missing(table: str, column: sa.Column) -> None:
    if _table_exists(table) and not _column_exists(table, column.name):
        op.add_column(table, column)


def upgrade() -> None:
    _add_column_if_missing(
        "promoted_strategies",
        sa.Column("execution_mode", sa.String(20), nullable=False,
                  server_default="paper"))
    _add_column_if_missing(
        "trades", sa.Column("broker", sa.String(30), nullable=True))
    _add_column_if_missing(
        "trades", sa.Column("broker_order_id", sa.String(64), nullable=True))


def downgrade() -> None:
    for table, col in (("trades", "broker_order_id"),
                       ("trades", "broker"),
                       ("promoted_strategies", "execution_mode")):
        if _table_exists(table) and _column_exists(table, col):
            op.drop_column(table, col)
