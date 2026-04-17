"""Trades API — wins/losses/open/closed, strategy-filterable."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.services import get_arena
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
    }


def _get_live_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for a list of symbols from the arena's provider."""
    from dashboard.services import _provider
    if not _provider:
        return {}
    prices = {}
    for sym in symbols:
        try:
            p = _provider.get_latest_price(sym)
            if p:
                prices[sym] = p
        except Exception:
            logger.warning("Failed to fetch live price for %s", sym, exc_info=True)
    return prices


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
    live_prices = _get_live_prices(open_symbols) if open_symbols else {}
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
