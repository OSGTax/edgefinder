"""Hunt/Lab explorer API — the validation_runs browser.

Every walk-forward run the lab ever recorded becomes browsable: filterable
paged list, full-detail view (config disclosure, both criteria bars, fold
detail when recorded, sealed-vs-evaluated holdout), and the scoreboard
tracking progress toward the 10-finalist goal.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.routers._shared import COST_DISCLOSURE
from edgefinder.db.models import PromotedStrategy, ValidationRun

router = APIRouter()


def _holdout_status(row: ValidationRun) -> str:
    h = row.holdout
    if not h or h.get("passes") is None:
        return "sealed"
    return "pass" if h.get("passes") else "fail"


def _cost_label(cfg: dict) -> str:
    if (cfg or {}).get("costed"):
        return "costed (spread+impact)"
    return f"flat {(cfg or {}).get('cost_bps', '?')} bps"


def _summary(row: ValidationRun) -> dict:
    cfg = row.config or {}
    oos = row.oos or {}
    crit = row.criteria or {}
    return {
        "id": row.id,
        "strategy_name": row.strategy_name,
        "run_at": row.run_at.isoformat() if row.run_at else None,
        "git_sha": row.git_sha,
        "universe": row.universe,
        "verdict": row.verdict,
        "criteria_mode": crit.get("mode"),
        "all_met": bool(crit.get("all_met")),
        "holdout_status": _holdout_status(row),
        "schedule": cfg.get("schedule"),
        "num_folds": cfg.get("num_folds"),
        "cost_label": _cost_label(cfg),
        "prices": cfg.get("prices"),
        "mean_excess_vs_spy_pct": oos.get("mean_excess_vs_spy_pct"),
        "folds_beating_spy": oos.get("folds_beating_spy"),
        "mean_excess_sharpe": oos.get("mean_excess_sharpe"),
        "folds_higher_sharpe": oos.get("folds_higher_sharpe"),
        "mean_sharpe": oos.get("mean_sharpe"),
        "total_trades": oos.get("total_trades"),
        "has_folds": bool(oos.get("folds")),
    }


@router.get("/runs")
def list_runs(
    label: str | None = Query(None, description="universe label prefix, e.g. hunt-r1"),
    strategy: str | None = Query(None, description="substring match"),
    verdict: str | None = Query(None, pattern="^(PASS|FAIL)$"),
    holdout: str | None = Query(None, pattern="^(sealed|pass|fail)$"),
    universe: str | None = Query(None, description="exact universe label"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(ValidationRun)
    if label:
        q = q.filter(ValidationRun.universe.like(f"{label}%"))
    if strategy:
        q = q.filter(ValidationRun.strategy_name.contains(strategy))
    if verdict:
        q = q.filter(ValidationRun.verdict == verdict)
    if universe:
        q = q.filter(ValidationRun.universe == universe)
    rows = q.order_by(ValidationRun.run_at.desc(), ValidationRun.id.desc()).all()
    items = [_summary(r) for r in rows]
    if holdout:                                   # JSON-derived; filter in python
        items = [i for i in items if i["holdout_status"] == holdout]
    total = len(items)
    return {"total": total, "runs": items[offset:offset + limit]}


@router.get("/runs/{run_id}")
def run_detail(run_id: int, db: Session = Depends(get_db)):
    row = db.get(ValidationRun, run_id)
    if row is None:
        raise HTTPException(404, f"validation run {run_id} not found")
    oos = dict(row.oos or {})
    folds = oos.pop("folds", None)
    by_regime = oos.pop("by_regime", None)
    return {
        **_summary(row),
        "config": row.config,
        "oos": oos,
        "folds": folds,            # None => "fold detail not recorded"
        "by_regime": by_regime,
        "criteria": row.criteria,
        "holdout": row.holdout,
        "cost_disclosure": COST_DISCLOSURE,
    }


@router.get("/scoreboard")
def scoreboard(db: Session = Depends(get_db)):
    """Progress toward the 10-finalist goal.

    Finalist candidate = newest run per base strategy (the label prefix is
    stripped from grouping; re-checks share the strategy name) with
    criteria.all_met. Holdout status and promotion are joined per name.
    """
    rows = (db.query(ValidationRun)
            .order_by(ValidationRun.run_at.desc(), ValidationRun.id.desc())
            .all())
    newest: dict[str, ValidationRun] = {}
    for r in rows:
        newest.setdefault(r.strategy_name, r)

    promoted = {p.strategy_name: p for p in db.query(PromotedStrategy).all()}

    finalists = []
    counts = {"criteria_passing": 0, "holdout_passed": 0, "promoted": 0}
    for name, r in newest.items():
        crit = r.criteria or {}
        if not crit.get("all_met"):
            continue
        counts["criteria_passing"] += 1
        status = _holdout_status(r)
        if status == "pass":
            counts["holdout_passed"] += 1
        promo = promoted.get(name)
        if promo and promo.active:
            counts["promoted"] += 1
        finalists.append({
            **_summary(r),
            "promoted": bool(promo and promo.active),
            "tier": promo.tier if promo else None,
        })
    finalists.sort(key=lambda f: (f["holdout_status"] != "pass",
                                  -(f["mean_excess_vs_spy_pct"] or 0)))
    return {"target": 10, "finalists": finalists, "counts": counts}


@router.get("/labels")
def labels(db: Session = Depends(get_db)):
    """Distinct label prefixes + universes for the filter dropdowns."""
    universes = [u for (u,) in db.query(ValidationRun.universe).distinct().all() if u]
    prefixes = sorted({u.split(":", 1)[0] for u in universes if ":" in u})
    return {"prefixes": prefixes, "universes": sorted(universes)}
