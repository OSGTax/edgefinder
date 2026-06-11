"""add universe columns to promoted_strategies

Revision ID: f1a9c4d27e55
Revises: 9f3a4cb15e87
Create Date: 2026-06-11

Cross-sectional universe promotions (the 12 hunt finalists trade
"top:500"-style point-in-time universes, not fixed symbol lists):

- universe_spec     "top:N[+OFFSET]" — mutually exclusive with symbols
- rank_window       trailing trading days for the dollar-volume ranking
- resolved_symbols  the last good universe resolution (shrink-guard fallback)
- resolved_at       date of that resolution

All nullable; fixed-symbol promotions are untouched. The table itself is
created by Base.metadata.create_all (scripts/setup_db.py) with the new
columns already in the ORM, so this migration no-ops on a DB where the
table does not exist yet.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

# revision identifiers, used by Alembic.
revision: str = "f1a9c4d27e55"
down_revision: Union[str, Sequence[str], None] = "9f3a4cb15e87"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "promoted_strategies"


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return column in [c["name"] for c in inspector.get_columns(table)]


def _add_column_if_missing(table: str, column: sa.Column) -> None:
    if not _column_exists(table, column.name):
        op.add_column(table, column)


def upgrade() -> None:
    """Upgrade schema."""
    if not _table_exists(TABLE):
        return   # fresh DB: create_all builds the table with these columns
    _add_column_if_missing(TABLE, sa.Column("universe_spec", sa.String(30), nullable=True))
    _add_column_if_missing(TABLE, sa.Column("rank_window", sa.Integer(), nullable=True))
    _add_column_if_missing(TABLE, sa.Column("resolved_symbols", sa.JSON(), nullable=True))
    _add_column_if_missing(TABLE, sa.Column("resolved_at", sa.Date(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    if not _table_exists(TABLE):
        return
    for col in ("resolved_at", "resolved_symbols", "rank_window", "universe_spec"):
        if _column_exists(TABLE, col):
            op.drop_column(TABLE, col)
