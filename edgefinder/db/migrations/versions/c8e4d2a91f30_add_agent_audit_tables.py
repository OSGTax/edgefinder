"""add agent audit tables — agent_observations and agent_actions

Revision ID: c8e4d2a91f30
Revises: 7b2bc613b751
Create Date: 2026-04-17 18:00:00.000000

Foundation for management agents (watchdog, strategist). Observations
capture what the agent sees; actions capture what the agent does. Idempotent
so it can run against a DB where Base.metadata.create_all() already created
the tables at startup.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect


revision: str = "c8e4d2a91f30"
down_revision: Union[str, Sequence[str], None] = "7b2bc613b751"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("agent_observations"):
        op.create_table(
            "agent_observations",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("agent_name", sa.String(50), nullable=False),
            sa.Column(
                "timestamp",
                sa.DateTime(),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.Column("severity", sa.String(10), nullable=False),
            sa.Column("category", sa.String(50), nullable=False),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("obs_metadata", sa.JSON(), nullable=True),
            sa.Column("resolved_at", sa.DateTime(), nullable=True),
            sa.Column("resolved_by", sa.String(50), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "idx_agent_obs_agent_ts",
            "agent_observations",
            ["agent_name", "timestamp"],
        )
        op.create_index(
            "idx_agent_obs_unresolved",
            "agent_observations",
            ["resolved_at"],
        )
        op.create_index(
            op.f("ix_agent_observations_agent_name"),
            "agent_observations",
            ["agent_name"],
        )
        op.create_index(
            op.f("ix_agent_observations_timestamp"),
            "agent_observations",
            ["timestamp"],
        )
        op.create_index(
            op.f("ix_agent_observations_category"),
            "agent_observations",
            ["category"],
        )

    if not _table_exists("agent_actions"):
        op.create_table(
            "agent_actions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("agent_name", sa.String(50), nullable=False),
            sa.Column(
                "timestamp",
                sa.DateTime(),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.Column("action_type", sa.String(30), nullable=False),
            sa.Column("summary", sa.String(500), nullable=False),
            sa.Column("files_touched", sa.JSON(), nullable=True),
            sa.Column("commit_sha", sa.String(64), nullable=True),
            sa.Column("pr_url", sa.String(200), nullable=True),
            sa.Column(
                "status",
                sa.String(20),
                nullable=False,
                server_default="pending",
            ),
            sa.Column("observation_id", sa.Integer(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(
                ["observation_id"], ["agent_observations.id"]
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "idx_agent_act_agent_ts",
            "agent_actions",
            ["agent_name", "timestamp"],
        )
        op.create_index(
            op.f("ix_agent_actions_agent_name"),
            "agent_actions",
            ["agent_name"],
        )
        op.create_index(
            op.f("ix_agent_actions_timestamp"),
            "agent_actions",
            ["timestamp"],
        )


def downgrade() -> None:
    if _table_exists("agent_actions"):
        op.drop_table("agent_actions")
    if _table_exists("agent_observations"):
        op.drop_table("agent_observations")
