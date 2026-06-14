"""add llm decision tables — llm_decision_cache and llm_decision_log

Revision ID: d4f8a1c93b27
Revises: f1a9c4d27e55
Create Date: 2026-06-14 12:00:00.000000

Backing store for the blind LLM judgment strategy (engine/llm_strategy.py):
- llm_decision_cache memoizes decisions by anonymized-context hash so a
  re-run reproduces identical trades (and avoids paid re-calls);
- llm_decision_log is an append-only audit of every prompt + raw response.

Idempotent (table-exists-guarded) so it can run against a DB where
Base.metadata.create_all() already created the tables at startup.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect


revision: str = "d4f8a1c93b27"
down_revision: Union[str, Sequence[str], None] = "f1a9c4d27e55"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("llm_decision_cache"):
        op.create_table(
            "llm_decision_cache",
            sa.Column("context_hash", sa.String(64), nullable=False),
            sa.Column("weights_json", sa.JSON(), nullable=False),
            sa.Column("model", sa.String(60), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("context_hash"),
        )

    if not _table_exists("llm_decision_log"):
        op.create_table(
            "llm_decision_log",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("context_hash", sa.String(64), nullable=False),
            sa.Column("model", sa.String(60), nullable=False),
            sa.Column("prompt", sa.Text(), nullable=False),
            sa.Column("response", sa.Text(), nullable=True),
            sa.Column("weights_json", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "idx_llm_decision_log_hash",
            "llm_decision_log",
            ["context_hash"],
        )


def downgrade() -> None:
    if _table_exists("llm_decision_log"):
        op.drop_table("llm_decision_log")
    if _table_exists("llm_decision_cache"):
        op.drop_table("llm_decision_cache")
