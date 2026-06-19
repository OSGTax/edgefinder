"""Picks API — the AI analyst account's daily decision + evidence dossier.

Read-only over ``agent_decisions`` (cheap; the heavy research runs in the
analyst job). The page renders these into a chart-forward report. A POST
trigger kicks an on-demand research run in a background thread (202).
"""

from __future__ import annotations

import threading
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.db.models import AgentDecision

router = APIRouter()

DEFAULT_STRATEGY = "ai_analyst"


def _serialize(row: AgentDecision) -> dict:
    picks = row.picks or []
    return {
        "strategy_name": row.strategy_name,
        "decision_date": row.decision_date.isoformat(),
        "summary": row.summary,
        "model": row.model,
        "target_weights": row.target_weights or {},
        "picks": picks,
        "counts": {
            "holdings": sum(1 for p in picks
                            if p.get("action") in ("hold", "trim", "add")),
            "new": sum(1 for p in picks if p.get("action") == "buy"),
            "sells": sum(1 for p in picks if p.get("action") == "sell"),
        },
    }


@router.get("/latest")
def latest(strategy: str = Query(DEFAULT_STRATEGY),
           db: Session = Depends(get_db)):
    """The most recent decision for ``strategy`` (the daily report)."""
    row = (db.query(AgentDecision)
           .filter(AgentDecision.strategy_name == strategy)
           .order_by(AgentDecision.decision_date.desc())
           .first())
    if row is None:
        return {"strategy_name": strategy, "decision_date": None, "picks": [],
                "summary": None, "target_weights": {},
                "counts": {"holdings": 0, "new": 0, "sells": 0}}
    return _serialize(row)


@router.get("/dates")
def dates(strategy: str = Query(DEFAULT_STRATEGY),
          limit: int = Query(30, ge=1, le=180),
          db: Session = Depends(get_db)):
    """Recent decision dates, newest first (for history navigation)."""
    rows = (db.query(AgentDecision.decision_date)
            .filter(AgentDecision.strategy_name == strategy)
            .order_by(AgentDecision.decision_date.desc())
            .limit(limit).all())
    return {"strategy_name": strategy,
            "dates": [r[0].isoformat() for r in rows]}


@router.get("/{decision_date}")
def by_date(decision_date: str, strategy: str = Query(DEFAULT_STRATEGY),
            db: Session = Depends(get_db)):
    """The decision for a specific date (YYYY-MM-DD)."""
    try:
        d = date.fromisoformat(decision_date)
    except ValueError:
        raise HTTPException(400, "decision_date must be YYYY-MM-DD")
    row = (db.query(AgentDecision)
           .filter(AgentDecision.strategy_name == strategy,
                   AgentDecision.decision_date == d)
           .one_or_none())
    if row is None:
        raise HTTPException(404, f"no decision for {strategy} on {decision_date}")
    return _serialize(row)


@router.post("/run", status_code=202)
def run(strategy: str = Query(DEFAULT_STRATEGY)):
    """Kick an on-demand analyst research run in a background thread."""
    from dashboard.services import run_analyst_job

    threading.Thread(target=run_analyst_job, kwargs={"strategy_name": strategy},
                     name="analyst-run", daemon=True).start()
    return {"status": "started", "strategy": strategy}
