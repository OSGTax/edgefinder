"""Persistent memory for management agents.

One row per agent in the agent_memory table. The reasoning step calls
`load_memory()` before invoking the LLM, passes the content as context,
then calls `save_memory()` with the LLM's updated content if any.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from edgefinder.db.models import AgentMemory

DEFAULT_MEMORY = """# Agent Memory

This file is the agent's persistent memory across ticks. The agent reads
it before each tick for context and may rewrite it to capture new
patterns, known false positives, or recent resolutions.

## Patterns Observed

(none yet)

## Known False Positives

(none yet)

## Recent Resolutions

(none yet)
"""


def load_memory(session: Session, agent_name: str) -> str:
    """Return the agent's memory content, creating a default row if absent."""
    row = (
        session.query(AgentMemory)
        .filter(AgentMemory.agent_name == agent_name)
        .one_or_none()
    )
    if row is not None:
        return row.content

    row = AgentMemory(agent_name=agent_name, content=DEFAULT_MEMORY)
    session.add(row)
    session.commit()
    return row.content


def save_memory(session: Session, agent_name: str, content: str) -> None:
    """Upsert the agent's memory content."""
    row = (
        session.query(AgentMemory)
        .filter(AgentMemory.agent_name == agent_name)
        .one_or_none()
    )
    if row is None:
        session.add(AgentMemory(agent_name=agent_name, content=content))
    else:
        row.content = content
    session.commit()
