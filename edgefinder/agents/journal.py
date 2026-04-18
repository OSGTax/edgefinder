"""Persistence helpers for agent observations and actions.

Agents call `record_observation` when they see something worth logging
and `record_action` when they do something (open PR, write diagnostic,
comment). Both return the DB id so callers can link actions back to the
observation that prompted them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy.orm import Session

from edgefinder.db.models import AgentAction, AgentObservation

logger = logging.getLogger(__name__)


VALID_SEVERITIES = {"INFO", "WARN", "ERROR", "CRITICAL"}
VALID_ACTION_STATUSES = {"pending", "submitted", "merged", "rejected"}


def record_observation(
    session: Session,
    agent_name: str,
    severity: str,
    category: str,
    message: str,
    metadata: dict[str, Any] | None = None,
    commit: bool = True,
) -> int:
    """Persist a new observation and return its id."""
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"severity must be one of {sorted(VALID_SEVERITIES)}, got {severity!r}"
        )

    obs = AgentObservation(
        agent_name=agent_name,
        severity=severity,
        category=category,
        message=message,
        obs_metadata=metadata,
    )
    session.add(obs)
    if commit:
        session.commit()
    else:
        session.flush()
    return obs.id


def record_action(
    session: Session,
    agent_name: str,
    action_type: str,
    summary: str,
    files_touched: Iterable[str] | None = None,
    commit_sha: str | None = None,
    pr_url: str | None = None,
    status: str = "pending",
    observation_id: int | None = None,
    notes: str | None = None,
    commit: bool = True,
) -> int:
    """Persist a new action and return its id."""
    if status not in VALID_ACTION_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(VALID_ACTION_STATUSES)}, got {status!r}"
        )

    action = AgentAction(
        agent_name=agent_name,
        action_type=action_type,
        summary=summary,
        files_touched=list(files_touched) if files_touched is not None else None,
        commit_sha=commit_sha,
        pr_url=pr_url,
        status=status,
        observation_id=observation_id,
        notes=notes,
    )
    session.add(action)
    if commit:
        session.commit()
    else:
        session.flush()
    return action.id


def resolve_observation(
    session: Session,
    observation_id: int,
    resolved_by: str,
    commit: bool = True,
) -> bool:
    """Mark an observation resolved. Returns False if id not found."""
    obs = session.get(AgentObservation, observation_id)
    if obs is None:
        return False
    obs.resolved_at = datetime.now(timezone.utc)
    obs.resolved_by = resolved_by
    if commit:
        session.commit()
    return True
