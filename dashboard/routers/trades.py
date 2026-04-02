"""Trades API — wins/losses/open/closed, strategy-filterable."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.services import get_arena
from edgefinder.trading.journal import TradeJournal

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
        "stop_loss": t.stop_loss,
        "target": t.target,
        "confidence": t.confidence,
        "status": t.status,
        "pnl_dollars": t.pnl_dollars if t.status == "CLOSED" else unrealized_pnl,
        "pnl_percent": t.pnl_percent if t.status == "CLOSED" else unrealized_pct,
        "r_multiple": t.r_multiple,
        "exit_reason": t.exit_reason,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
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
            pass
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
