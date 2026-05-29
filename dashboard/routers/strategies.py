"""Strategies API — per-strategy accounts, equity curves, scheduler status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.services import get_arena, get_scheduler
from edgefinder.db.models import StrategyAccount, StrategySnapshot
from edgefinder.strategies.base import StrategyRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


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

    Enriches with unrealized P&L by fetching current prices for open positions.
    """
    live = _live_account_states()
    if live is not None:
        return live

    # Fallback to DB
    accounts = db.query(StrategyAccount).all()
    return [
        {
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
