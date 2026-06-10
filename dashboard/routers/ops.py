"""Ops API — system health surfaced from the DB onto the dashboard.

The heartbeat, watchdog observations, and scheduler state all exist in the
DB / process already; this endpoint projects them for a dashboard panel so
the owner sees loop health where they already look, not just via GitHub
issues.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
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

    from dashboard.services import get_scheduler

    scheduler = get_scheduler()
    sched = {"running": False}
    if scheduler:
        status = scheduler.get_status()
        sched = {
            "running": status.get("running", True),
            "next_runs": status.get("next_runs", {}),
        }

    return {
        "heartbeats": heartbeats,
        "observations": observations,
        "observation_counts": counts,
        "scheduler": sched,
    }


# ── redesign additions (v5.40): activity timeline + storage panel ──


@router.get("/activity")
def agent_activity(
    limit: int = Query(100, le=300),
    include_resolved: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Merged agent observations + actions timeline, newest first."""
    from edgefinder.db.models import AgentAction, AgentObservation

    items: list[dict] = []
    obs_q = db.query(AgentObservation).order_by(AgentObservation.timestamp.desc())
    if not include_resolved:
        obs_q = obs_q.filter(AgentObservation.resolved_at.is_(None))
    for o in obs_q.limit(limit).all():
        items.append({
            "kind": "observation",
            "agent": o.agent_name,
            "timestamp": o.timestamp.isoformat() if o.timestamp else None,
            "severity": o.severity,
            "category": o.category,
            "message": o.message,
            "resolved_at": o.resolved_at.isoformat() if o.resolved_at else None,
        })
    for a in (db.query(AgentAction)
              .order_by(AgentAction.timestamp.desc()).limit(limit).all()):
        items.append({
            "kind": "action",
            "agent": a.agent_name,
            "timestamp": a.timestamp.isoformat() if a.timestamp else None,
            "action_type": a.action_type,
            "summary": a.summary,
            "status": a.status,
            "pr_url": a.pr_url,
            "commit_sha": a.commit_sha,
        })
    items.sort(key=lambda x: x["timestamp"] or "", reverse=True)
    return {"items": items[:limit]}


_storage_cache = None


@router.get("/storage")
def storage_panel(db: Session = Depends(get_db)):
    """Two-tier storage status: DB hot-set aggregates + R2 manifest summary.

    R2 reads go through a 15-min TTL cache; {"r2": null} when the R2 env
    isn't configured (local dev) or the manifest read fails.
    """
    global _storage_cache
    from sqlalchemy import func as _f

    from dashboard.ttl_cache import TTLCache
    from edgefinder.db.models import DailyBar

    if _storage_cache is None:
        _storage_cache = TTLCache(maxsize=4, ttl_seconds=900)

    symbols, rows, dmin, dmax = (
        db.query(_f.count(_f.distinct(DailyBar.symbol)), _f.count(DailyBar.id),
                 _f.min(DailyBar.date), _f.max(DailyBar.date)).one())
    out = {
        "db": {"symbols": symbols, "rows": rows,
               "min_date": str(dmin) if dmin else None,
               "max_date": str(dmax) if dmax else None},
        "r2": _storage_cache.get("r2"),
    }
    if out["r2"] is None:
        try:
            from edgefinder.data.barstore import BarStore

            manifest = BarStore().read_manifest()
            r2 = {
                "symbols": len(manifest),
                "rows": sum((m or {}).get("rows", 0) for m in manifest.values()),
                "max_date": max((str((m or {}).get("max_date", "")) for m in manifest.values()),
                                default=None),
            }
            _storage_cache.set("r2", r2)
            out["r2"] = r2
        except Exception:
            out["r2"] = None
    return out
