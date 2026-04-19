"""add agent_memory table

Revision ID: e7a2f91b4d08
Revises: c8e4d2a91f30
Create Date: 2026-04-17 19:00:00.000000

Persistent memory for management agents — the content the reasoning
step reads before each LLM call and may rewrite after. Idempotent so
it coexists with Base.metadata.create_all.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect


revision: str = "e7a2f91b4d08"
down_revision: Union[str, Sequence[str], None] = "c8e4d2a91f30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("agent_memory"):
        return
    op.create_table(
        "agent_memory",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_name", name="uq_agent_memory_name"),
    )
    op.create_index(
        op.f("ix_agent_memory_agent_name"),
        "agent_memory",
        ["agent_name"],
    )


def downgrade() -> None:
    if _table_exists("agent_memory"):
        op.drop_table("agent_memory")
