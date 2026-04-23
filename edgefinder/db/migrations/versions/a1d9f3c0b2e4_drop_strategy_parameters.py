"""drop strategy_parameters table

Revision ID: a1d9f3c0b2e4
Revises: e7a2f91b4d08
Create Date: 2026-04-23 19:30:00.000000

The StrategyParameterLog ORM model had no live writers in the codebase —
it was leftover audit-log scaffolding from a previous rewrite that never
saw production use. Dropping the table + index.

Idempotent so it coexists with Base.metadata.create_all.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect


revision: str = "a1d9f3c0b2e4"
down_revision: Union[str, Sequence[str], None] = "e7a2f91b4d08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("strategy_parameters"):
        with op.batch_alter_table("strategy_parameters") as batch_op:
            # drop_index is idempotent-ish; guard anyway
            try:
                batch_op.drop_index("ix_strategy_parameters_strategy_name")
            except Exception:
                pass
        op.drop_table("strategy_parameters")


def downgrade() -> None:
    if _table_exists("strategy_parameters"):
        return
    op.create_table(
        "strategy_parameters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("strategy_name", sa.String(length=50), nullable=False),
        sa.Column("param_name", sa.String(length=100), nullable=False),
        sa.Column("old_value", sa.String(length=200), nullable=True),
        sa.Column("new_value", sa.String(length=200), nullable=True),
        sa.Column("changed_by", sa.String(length=30), nullable=False),
        sa.Column(
            "changed_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_strategy_parameters_strategy_name",
        "strategy_parameters",
        ["strategy_name"],
        unique=False,
    )
