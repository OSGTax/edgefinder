"""Strategies API — per-strategy accounts, equity curves, scheduler status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.routers._shared import COST_DISCLOSURE as _COST_DISCLOSURE
from dashboard.services import get_arena, get_scheduler
from edgefinder.db.models import PromotedStrategy, StrategyAccount, StrategySnapshot
from edgefinder.strategies.base import StrategyRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

def _lane(name: str, v2_names: set[str]) -> str:
    from config.settings import settings

    if name in v2_names:
        return "v2"
    if name in settings.live_strategies:
        return "arena"
    return "arena"


def _v2_names(db: Session) -> set[str]:
    return {p.strategy_name for p in
            db.query(PromotedStrategy).filter(PromotedStrategy.active.is_(True)).all()}


def _live_account_states() -> list[dict] | None:
    """Per-strategy accounts marked to the latest available market price.

    Single source of truth for "current value" so the strategy cards and the
    equity-curve tail can never disagree:

        open_positions_value = Σ shares × current price
                               (live price when available, else entry price)
        total_equity         = cash + open_positions_value
        unrealized_pnl       = market value − cost basis (direction-aware)

    NOTE: ``to_dict()``'s ``open_positions_value`` is already mark-to-market,
    so it must NOT be re-added to unrealized P&L (the prior bug double-counted
    the P&L). We recompute the market value directly from current prices here.

    Returns ``None`` when the arena isn't running so callers fall back to DB.
    """
    arena = get_arena()
    if not arena:
        return None
    from dashboard.services import _provider

    accounts = arena.get_all_accounts()
    positions = arena.get_all_open_positions()

    all_symbols = list({
        p["symbol"] for pos_list in positions.values() for p in pos_list
    })
    live_prices: dict[str, float] = {}
    if all_symbols and _provider:
        for sym in all_symbols:
            try:
                price = _provider.get_latest_price(sym)
                if price:
                    live_prices[sym] = price
            except Exception:
                logger.warning("Failed to fetch live price for %s", sym, exc_info=True)

    result = []
    for name, acct in accounts.items():
        market_value = 0.0
        unrealized = 0.0
        for p in positions.get(name, []):
            price = live_prices.get(p["symbol"]) or p["entry_price"]
            market_value += p["shares"] * price
            if p["direction"] == "LONG":
                unrealized += (price - p["entry_price"]) * p["shares"]
            else:
                unrealized += (p["entry_price"] - price) * p["shares"]
        acct["open_positions_value"] = round(market_value, 2)
        acct["unrealized_pnl"] = round(unrealized, 2)
        acct["total_equity"] = round(acct["cash"] + market_value, 2)
        result.append(acct)
    return result


@router.get("")
def list_strategies():
    """List all registered strategies."""
    return [
        {"name": name, "class": cls.__name__}
        for name, cls in StrategyRegistry.get_all().items()
    ]


@router.get("/accounts")
def get_accounts(db: Session = Depends(get_db)):
    """Get strategy account states — live from arena if available, else DB.

    Enriches with unrealized P&L by fetching current prices for open
    positions, and tags each row with its lane (arena $5k vs v2 $100k).
    """
    v2 = _v2_names(db)
    live = _live_account_states()
    if live is not None:
        for a in live:
            a["lane"] = _lane(a.get("strategy_name") or a.get("name", ""), v2)
        return live

    # Fallback to DB
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
def get_positions():
    """Get all open positions across all strategies."""
    arena = get_arena()
    if not arena:
        return {}
    return arena.get_all_open_positions()


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

    # Append a live "now" point so the curve ends at the current market value
    # (cash + securities at current price) rather than the last persisted
    # snapshot. Appended as a distinct timestamp; only replaces the last point
    # if it lands in the same second (so the aggregate isn't doubled).
    live = _live_account_states()
    if live is not None:
        now = datetime.now(timezone.utc)
        now_epoch = int(now.timestamp())
        now_date = now.strftime("%Y-%m-%d")
        for acct in live:
            name = acct["strategy_name"]
            if strategy and name != strategy:
                continue
            starting = acct.get("starting_capital") or 0
            point = {
                "time": now_epoch,
                "date": now_date,
                "total_equity": acct["total_equity"],
                "total_return_pct": (
                    round((acct["total_equity"] - starting) / starting * 100, 4)
                    if starting else None
                ),
            }
            pts = result.setdefault(name, [])
            if pts and pts[-1].get("time") == now_epoch:
                pts[-1] = point
            else:
                pts.append(point)

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
    """Get scheduler status, next run times, and last cycle result."""
    from dashboard.services import get_last_signal_check
    scheduler = get_scheduler()
    if not scheduler:
        return {"running": False, "jobs": {}, "message": "Pipeline not initialized"}
    status = scheduler.get_status()
    status["last_signal_check"] = get_last_signal_check()
    return status


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
    """Registry-driven strategy metadata: lanes, display, color slots, risk.

    Replaces every hardcoded STRATEGY_RISK/name/color table in the old JS —
    adding or renaming a strategy needs zero frontend changes.
    """
    v2_rows = (db.query(PromotedStrategy)
               .filter(PromotedStrategy.active.is_(True))
               .order_by(PromotedStrategy.strategy_name).all())
    out = []
    slot = 0
    try:
        by_name = {i.name: i for i in StrategyRegistry.get_instances()}
    except Exception:
        by_name = {}
    for name in sorted(StrategyRegistry.get_all()):
        inst = by_name.get(name)
        risk = {}
        for attr in ("risk_pct", "target_pct", "stop_pct"):
            v = getattr(inst, attr, None)
            if v is not None:
                risk[attr] = v
        out.append({"name": name, "lane": "arena", "display_name": name,
                    "color_slot": slot % 8, "risk": risk})
        slot += 1
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
