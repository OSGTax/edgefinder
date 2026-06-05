"""Ops API — system health surfaced from the DB onto the dashboard.

The heartbeat, watchdog observations, and scheduler state all exist in the
DB / process already; this endpoint projects them for a dashboard panel so
the owner sees loop health where they already look, not just via GitHub
issues.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.db.models import AgentObservation, SystemHeartbeat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def ops_health(db: Session = Depends(get_db)):
    """Heartbeats + unresolved watchdog observations + scheduler state."""
    now = datetime.now(timezone.utc)

    heartbeats = []
    for hb in db.query(SystemHeartbeat).order_by(SystemHeartbeat.component).all():
        last = hb.last_run_at
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        heartbeats.append({
            "component": hb.component,
            "ok": bool(hb.ok),
            "last_run_at": last.isoformat() if last else None,
            "age_minutes": round((now - last).total_seconds() / 60, 1) if last else None,
            "detail": hb.detail,
        })

    unresolved = (
        db.query(AgentObservation)
        .filter(AgentObservation.resolved_at.is_(None))
        .order_by(AgentObservation.timestamp.desc())
        .limit(20)
        .all()
    )
    observations = [
        {
            "severity": o.severity,
            "category": o.category,
            "message": o.message,
            "agent": o.agent_name,
            "timestamp": o.timestamp.isoformat() if o.timestamp else None,
        }
        for o in unresolved
    ]
    counts = {
        "critical": sum(1 for o in unresolved if o.severity == "CRITICAL"),
        "warn": sum(1 for o in unresolved if o.severity == "WARN"),
        "other": sum(1 for o in unresolved if o.severity not in ("CRITICAL", "WARN")),
    }

    from dashboard.services import get_last_signal_check, get_scheduler

    scheduler = get_scheduler()
    sched = {"running": False}
    if scheduler:
        status = scheduler.get_status()
        sched = {
            "running": status.get("running", True),
            "next_runs": status.get("next_runs", {}),
        }
    sched["last_signal_check"] = get_last_signal_check()

    return {
        "heartbeats": heartbeats,
        "observations": observations,
        "observation_counts": counts,
        "scheduler": sched,
    }
