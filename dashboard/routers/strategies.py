"""Strategies API — per-strategy accounts, equity curves, scheduler status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.routers._shared import COST_DISCLOSURE as _COST_DISCLOSURE
from dashboard.services import get_scheduler
from edgefinder.db.models import PromotedStrategy, StrategyAccount, StrategySnapshot

logger = logging.getLogger(__name__)

router = APIRouter()


def _lane(name: str, v2_names: set[str]) -> str:
    """Everything is v2 now (the old per-ticker arena was retired); the
    field is kept for API-shape stability."""
    return "v2"


def _v2_names(db: Session) -> set[str]:
    return {p.strategy_name for p in
            db.query(PromotedStrategy).filter(PromotedStrategy.active.is_(True)).all()}


@router.get("")
def list_strategies(db: Session = Depends(get_db)):
    """List the v2 promoted strategies (the live registry is the DB now)."""
    rows = (db.query(PromotedStrategy)
            .order_by(PromotedStrategy.strategy_name).all())
    return [
        {"name": p.strategy_name, "spec": p.spec, "tier": p.tier,
         "schedule": p.schedule, "active": p.active}
        for p in rows
    ]


@router.get("/accounts")
def get_accounts(db: Session = Depends(get_db)):
    """Get strategy account states from the DB (the source of truth —
    the v2 engine marks strategy_accounts every cycle/snapshot)."""
    v2 = _v2_names(db)
    accounts = db.query(StrategyAccount).all()
    return [
        {
            "lane": _lane(a.strategy_name, v2),
            "strategy_name": a.strategy_name,
            "starting_capital": a.starting_capital,
            "cash": a.cash_balance,
            "open_positions_value": a.open_positions_value,
            "total_equity": a.total_equity,
            "peak_equity": a.peak_equity,
            "drawdown_pct": a.drawdown_pct,
            "realized_pnl": a.realized_pnl or 0.0,
            "unrealized_pnl": 0.0,
            "pdt_enabled": a.pdt_enabled,
            "is_paused": a.is_paused,
        }
        for a in accounts
    ]


@router.get("/positions")
def get_positions(db: Session = Depends(get_db)):
    """All OPEN trades grouped by strategy (same shape the arena returned)."""
    from edgefinder.db.models import TradeRecord

    out: dict[str, list[dict]] = {}
    rows = (db.query(TradeRecord)
            .filter(TradeRecord.status == "OPEN")
            .order_by(TradeRecord.strategy_name, TradeRecord.entry_time)
            .all())
    for t in rows:
        out.setdefault(t.strategy_name, []).append({
            "symbol": t.symbol,
            "shares": t.shares,
            "entry_price": t.entry_price,
            "direction": t.direction,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        })
    return out


@router.get("/equity-curve")
def equity_curve(
    strategy: str | None = Query(None),
    days: int = Query(90, le=365),
    db: Session = Depends(get_db),
):
    """Get equity curve data for charting."""
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    q = db.query(StrategySnapshot).filter(StrategySnapshot.timestamp >= cutoff)
    if strategy:
        q = q.filter(StrategySnapshot.strategy_name == strategy)
    q = q.order_by(StrategySnapshot.timestamp)
    snapshots = q.all()

    def _epoch(ts) -> int | None:
        # Stored timestamps are naive UTC; chart axis wants UTC epoch seconds
        # so intraday points within a day stay distinct (not collapsed by date).
        if not ts:
            return None
        return int(ts.replace(tzinfo=timezone.utc).timestamp())

    result: dict[str, list] = {}
    for s in snapshots:
        if s.strategy_name not in result:
            result[s.strategy_name] = []
        result[s.strategy_name].append({
            "time": _epoch(s.timestamp),
            "date": s.timestamp.strftime("%Y-%m-%d") if s.timestamp else None,
            "total_equity": s.total_equity,
            "total_return_pct": s.total_return_pct,
        })

    return result


@router.get("/scorecard")
def scorecard(
    strategy: str | None = Query(None),
    days: int = Query(90, le=365),
    db: Session = Depends(get_db),
):
    """Live-vs-SPY scorecard: the offline validation bar applied to live data.

    Per strategy, over the trailing ``days`` window: annualized Sharpe of the
    daily equity series (last strategy_snapshots mark per ET day), excess
    return vs SPY (index_daily closes, inner-joined on common dates), and
    closed-trade stats — with the same criteria block the walk-forward lab
    emits (sharpe_positive / beats_spy / min_trades_met / all_met). Every
    number is recomputable from strategy_snapshots + index_daily + trades.
    """
    from edgefinder.analytics.live_scorecard import (
        compute_all_scorecards,
        compute_scorecard,
    )

    cards = ([compute_scorecard(db, strategy, days=days)] if strategy
             else compute_all_scorecards(db, days=days))
    for c in cards:
        # live numbers embed real paper slippage; lab numbers assume their
        # own cost model — disclose the asymmetry wherever the two meet
        c["cost_disclosure"] = _COST_DISCLOSURE
    return cards


@router.get("/validation")
def validation_runs(db: Session = Depends(get_db)):
    """Latest offline validation verdict per strategy (walk-forward lab).

    ``validated`` is the honest summary: the explicit criteria bar was met
    AND the sealed holdout was evaluated AND passed. A run whose holdout was
    deliberately left sealed (research stages) is criteria-passing but NOT
    validated — the holdout is the only test that counts. Shown on the
    dashboard beside the Live Proof card — offline claim vs live evidence.
    """
    from edgefinder.db.models import ValidationRun

    rows = (
        db.query(ValidationRun)
        .order_by(ValidationRun.run_at.desc(), ValidationRun.id.desc())
        .limit(100)
        .all()
    )
    latest: dict[str, ValidationRun] = {}
    for r in rows:  # newest first → first row per strategy wins
        latest.setdefault(r.strategy_name, r)
    out = []
    for r in latest.values():
        criteria = r.criteria or {}
        holdout = r.holdout
        validated = bool(
            criteria.get("all_met")
            and holdout is not None
            and holdout.get("passes")
        )
        cfg = r.config or {}
        lab_costs = ("realistic cost model (spread + impact + participation)"
                     if cfg.get("costed")
                     else f"flat {cfg.get('cost_bps', '?')} bps per fill")
        out.append({
            "strategy_name": r.strategy_name,
            "run_at": r.run_at.isoformat() if r.run_at else None,
            "git_sha": r.git_sha,
            "universe": r.universe,
            "config": r.config,
            "oos": r.oos,
            "criteria": criteria,
            "holdout": holdout,
            "verdict": r.verdict,
            "validated": validated,
            "cost_disclosure": {**_COST_DISCLOSURE, "this_run_lab_costs": lab_costs},
        })
    return sorted(out, key=lambda x: x["strategy_name"])


@router.get("/scheduler")
def scheduler_status():
    """Get scheduler status and next run times."""
    scheduler = get_scheduler()
    if not scheduler:
        return {"running": False, "jobs": {}, "message": "Pipeline not initialized"}
    return scheduler.get_status()


# ── redesign additions (v5.40): promoted / summary / meta / ledgers ──


@router.get("/promoted")
def promoted_strategies(db: Session = Depends(get_db)):
    """v2 promoted strategies (active AND demoted) joined to their accounts."""
    accounts = {a.strategy_name: a for a in db.query(StrategyAccount).all()}
    out = []
    for p in db.query(PromotedStrategy).order_by(PromotedStrategy.promoted_at.desc()).all():
        a = accounts.get(p.strategy_name)
        out.append({
            "strategy_name": p.strategy_name,
            "spec": p.spec,
            "symbols": p.symbols,
            "schedule": p.schedule,
            "tier": p.tier,
            "active": p.active,
            "validation_run_id": p.validation_run_id,
            "prices_basis": getattr(p, "prices_basis", None),
            "promoted_at": p.promoted_at.isoformat() if p.promoted_at else None,
            "total_equity": a.total_equity if a else None,
            "drawdown_pct": a.drawdown_pct if a else None,
        })
    return out


@router.get("/summary")
def lane_summary(db: Session = Depends(get_db)):
    """Server-computed header rollups per lane (arena / v2 / all).

    The ONLY source for the portfolio hero stats — the client never invents
    capital figures (the old FALLBACK_STARTING_CAPITAL bug class); if this
    endpoint fails the UI shows an error card.
    """
    from datetime import datetime, time, timezone

    from edgefinder.db.models import TradeRecord

    v2 = _v2_names(db)
    accounts = get_accounts(db)
    midnight = datetime.combine(datetime.now(timezone.utc).date(), time.min,
                                tzinfo=timezone.utc)

    lanes: dict[str, dict] = {}
    for lane_name in ("arena", "v2", "all"):
        lanes[lane_name] = {
            "starting_capital": 0.0, "total_equity": 0.0, "day_pnl": None,
            "unrealized_pnl": 0.0, "open_positions": 0, "win_rate": None,
            "strategies": 0,
        }

    day_base: dict[str, float] = {}
    for (name, equity) in (
            db.query(StrategySnapshot.strategy_name, StrategySnapshot.total_equity)
            .filter(StrategySnapshot.timestamp < midnight)
            .order_by(StrategySnapshot.strategy_name, StrategySnapshot.timestamp)
            .all()):
        day_base[name] = equity   # last pre-midnight snapshot wins

    open_counts: dict[str, int] = {}
    wins: dict[str, int] = {}
    closed: dict[str, int] = {}
    for t in db.query(TradeRecord).all():
        if t.status == "OPEN":
            open_counts[t.strategy_name] = open_counts.get(t.strategy_name, 0) + 1
        elif t.status == "CLOSED":
            closed[t.strategy_name] = closed.get(t.strategy_name, 0) + 1
            if (t.pnl_dollars or 0) > 0:
                wins[t.strategy_name] = wins.get(t.strategy_name, 0) + 1

    day_pnl_acc: dict[str, float] = {"arena": 0.0, "v2": 0.0, "all": 0.0}
    day_pnl_known: dict[str, bool] = {"arena": False, "v2": False, "all": False}
    closed_acc = {"arena": 0, "v2": 0, "all": 0}
    wins_acc = {"arena": 0, "v2": 0, "all": 0}

    for a in accounts:
        name = a.get("strategy_name") or a.get("name", "")
        lane = a.get("lane") or _lane(name, v2)
        equity = a.get("total_equity") or 0.0
        for key in (lane, "all"):
            L = lanes[key]
            L["strategies"] += 1
            L["starting_capital"] += a.get("starting_capital") or 0.0
            L["total_equity"] += equity
            L["unrealized_pnl"] += a.get("unrealized_pnl") or 0.0
            L["open_positions"] += open_counts.get(name, 0)
            closed_acc[key] += closed.get(name, 0)
            wins_acc[key] += wins.get(name, 0)
            if name in day_base:
                day_pnl_acc[key] += equity - day_base[name]
                day_pnl_known[key] = True

    for key, L in lanes.items():
        if day_pnl_known[key]:
            L["day_pnl"] = round(day_pnl_acc[key], 2)
        if closed_acc[key]:
            L["win_rate"] = round(wins_acc[key] / closed_acc[key] * 100, 1)
        L["total_pnl"] = round(L["total_equity"] - L["starting_capital"], 2)
        for f in ("starting_capital", "total_equity", "unrealized_pnl"):
            L[f] = round(L[f], 2)
    return lanes


@router.get("/meta")
def strategies_meta(db: Session = Depends(get_db)):
    """DB-driven strategy metadata: lane, display, color slots.

    Replaces every hardcoded name/color table in the JS — adding or renaming
    a promoted strategy needs zero frontend changes. (The old StrategyRegistry
    half was retired with the arena; everything listed here is v2.)
    """
    v2_rows = (db.query(PromotedStrategy)
               .filter(PromotedStrategy.active.is_(True))
               .order_by(PromotedStrategy.strategy_name).all())
    out = []
    slot = 0
    for p in v2_rows:
        out.append({"name": p.strategy_name, "lane": "v2",
                    "display_name": p.strategy_name, "color_slot": slot % 8,
                    "risk": {}, "tier": p.tier, "schedule": p.schedule})
        slot += 1
    return out


@router.get("/dividends")
def dividend_credits(strategy: str | None = Query(None),
                     db: Session = Depends(get_db)):
    """Dividend cash-credit ledger for v2 paper accounts."""
    from edgefinder.db.models import DividendCredit

    q = db.query(DividendCredit).order_by(DividendCredit.ex_date.desc())
    if strategy:
        q = q.filter(DividendCredit.strategy_name == strategy)
    return [{
        "strategy_name": r.strategy_name, "symbol": r.symbol,
        "ex_date": str(r.ex_date), "shares": r.shares, "amount": r.amount,
    } for r in q.limit(500).all()]


@router.get("/params")
def parameter_audit(strategy: str | None = Query(None),
                    db: Session = Depends(get_db)):
    """Strategy parameter change audit log (coach/agent tuning history)."""
    from edgefinder.db.models import StrategyParameterLog

    q = (db.query(StrategyParameterLog)
         .order_by(StrategyParameterLog.changed_at.desc()))
    if strategy:
        q = q.filter(StrategyParameterLog.strategy_name == strategy)
    return [{
        "strategy_name": r.strategy_name, "param_name": r.param_name,
        "old_value": r.old_value, "new_value": r.new_value,
        "changed_by": r.changed_by,
        "changed_at": r.changed_at.isoformat() if r.changed_at else None,
    } for r in q.limit(200).all()]
