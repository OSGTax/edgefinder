"""add agent_decisions table

Revision ID: b8d2f4e16a09
Revises: a7c1e9f3d602
Create Date: 2026-06-16

Backing store for the research-agent paper account (the "AI analyst"):
one row per (strategy_name, decision_date) holding the agent's executable
target_weights plus the per-pick evidence dossier (picks) and a written
summary. The analyst job writes it; engine/analyst_strategy.AnalystStrategy
reads target_weights during the live cycle; the dashboard renders picks.

Idempotent (table-exists-guarded) so it no-ops on a fresh DB where
Base.metadata.create_all() already built the table from the ORM.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

# revision identifiers, used by Alembic.
revision: str = "b8d2f4e16a09"
down_revision: Union[str, Sequence[str], None] = "a7c1e9f3d602"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    return table in sa_inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _table_exists("agent_decisions"):
        return
    op.create_table(
        "agent_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("decision_date", sa.Date(), nullable=False),
        sa.Column("target_weights", sa.JSON(), nullable=False),
        sa.Column("picks", sa.JSON(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("model", sa.String(60), nullable=True),
        sa.Column("created_at", sa.DateTime(),
                  server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("strategy_name", "decision_date",
                            name="uq_agent_decision_strategy_date"),
    )
    op.create_index("idx_agent_decision_strategy_date", "agent_decisions",
                    ["strategy_name", "decision_date"])


def downgrade() -> None:
    if _table_exists("agent_decisions"):
        op.drop_index("idx_agent_decision_strategy_date", table_name="agent_decisions")
        op.drop_table("agent_decisions")
