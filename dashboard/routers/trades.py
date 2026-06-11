"""Trades API — wins/losses/open/closed, strategy-filterable."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.trading.journal import TradeJournal

ET = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)


def _to_et(dt: datetime | None) -> str | None:
    """Convert a datetime to US/Eastern ISO string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).isoformat()


router = APIRouter()


def _enrich_trade(t, live_prices: dict[str, float] | None = None) -> dict:
    """Build trade dict with current price and unrealized P&L for open trades."""
    current_price = None
    unrealized_pnl = None
    unrealized_pct = None

    if t.status == "OPEN" and live_prices:
        current_price = live_prices.get(t.symbol)
        if current_price and t.entry_price and t.shares:
            if t.direction == "LONG":
                unrealized_pnl = round((current_price - t.entry_price) * t.shares, 2)
            else:
                unrealized_pnl = round((t.entry_price - current_price) * t.shares, 2)
            cost = t.entry_price * t.shares
            if cost > 0:
                unrealized_pct = round(unrealized_pnl / cost * 100, 2)

    trade_value = round(t.entry_price * t.shares, 2) if t.entry_price and t.shares else None

    return {
        "trade_id": t.trade_id,
        "strategy_name": t.strategy_name,
        "symbol": t.symbol,
        "direction": t.direction,
        "trade_type": t.trade_type,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "current_price": current_price,
        "shares": t.shares,
        "trade_value": trade_value,
        "stop_loss": t.stop_loss,
        "target": t.target,
        "confidence": t.confidence,
        "status": t.status,
        "pnl_dollars": t.pnl_dollars if t.status == "CLOSED" else unrealized_pnl,
        "pnl_percent": t.pnl_percent if t.status == "CLOSED" else unrealized_pct,
        "r_multiple": t.r_multiple,
        "exit_reason": t.exit_reason,
        "entry_time": _to_et(t.entry_time),
        "exit_time": _to_et(t.exit_time),
        # Timeline fields — the trades-page expandable reasoning view reads
        # these; they were captured in the DB but never serialized (the UI
        # silently showed "No reasoning captured" for every trade).
        "entry_reasoning": t.entry_reasoning,
        "exit_reasoning": t.exit_reasoning,
        "indicators_at_entry": _as_dict(t.indicators_at_entry),
        "indicators_at_exit": _as_dict(t.indicators_at_exit),
        "hold_duration_hours": t.hold_duration_hours,
    }


def _as_dict(raw):
    """JSON columns return dicts on PG; tolerate string-stored values too."""
    if raw is None or isinstance(raw, dict):
        return raw
    try:
        import json

        v = json.loads(raw)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _get_live_prices(db: Session, symbols: list[str]) -> dict[str, float]:
    """Last completed close per symbol, in ONE query.

    This used to fetch a live quote per symbol from the provider — fine
    for two ETF accounts (7 names), a 60s+ hang once the fleet held ~146
    open lots across ~130 names (serial REST calls). The trades LIST marks
    against the last close; the 30-min account snapshots remain the
    live-quote mark.
    """
    if not symbols:
        return {}
    from sqlalchemy import func as sa_func

    from edgefinder.db.models import DailyBar

    latest = (db.query(DailyBar.symbol,
                       sa_func.max(DailyBar.date).label("d"))
              .filter(DailyBar.symbol.in_(symbols))
              .group_by(DailyBar.symbol).subquery())
    rows = (db.query(DailyBar.symbol, DailyBar.close)
            .join(latest, (DailyBar.symbol == latest.c.symbol)
                  & (DailyBar.date == latest.c.d)).all())
    return {sym: float(close) for sym, close in rows if close}


@router.get("")
def list_trades(
    strategy: str | None = Query(None),
    status: str | None = Query(None, description="OPEN, CLOSED, or CANCELLED"),
    symbol: str | None = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    journal = TradeJournal(db)
    trades = journal.get_trades(
        strategy_name=strategy, status=status, symbol=symbol, limit=limit
    )
    # Fetch live prices for open trades
    open_symbols = list({t.symbol for t in trades if t.status == "OPEN"})
    live_prices = _get_live_prices(db, open_symbols) if open_symbols else {}
    return [_enrich_trade(t, live_prices) for t in trades]


@router.get("/stats")
def trade_stats(
    strategy: str | None = Query(None),
    db: Session = Depends(get_db),
):
    journal = TradeJournal(db)
    return journal.compute_stats(strategy_name=strategy)


@router.get("/wins")
def wins(strategy: str | None = Query(None), db: Session = Depends(get_db)):
    journal = TradeJournal(db)
    trades = journal.get_closed_trades(strategy)
    return [
        _enrich_trade(t) for t in trades if t.pnl_dollars and t.pnl_dollars > 0
    ]


@router.get("/losses")
def losses(strategy: str | None = Query(None), db: Session = Depends(get_db)):
    journal = TradeJournal(db)
    trades = journal.get_closed_trades(strategy)
    return [
        _enrich_trade(t) for t in trades if t.pnl_dollars and t.pnl_dollars <= 0
    ]


@router.get("/integrity")
def trade_integrity(strategy: str | None = Query(None), db: Session = Depends(get_db)):
    """Verify the trades hash chain — recomputed from stored rows alone.

    A verified row proves its identity AND that the prior row's hash was
    untouched; any silent edit/delete breaks every later link. Rows written
    before the v2 chain (2026-06-05) are reported as legacy_unverified.
    """
    from edgefinder.trading.integrity import verify_chain

    return verify_chain(db, strategy)
