"""Trades API — wins/losses/open/closed, strategy-filterable."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.trading.journal import TradeJournal

router = APIRouter()


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
    return [
        {
            "trade_id": t.trade_id,
            "strategy_name": t.strategy_name,
            "symbol": t.symbol,
            "direction": t.direction,
            "trade_type": t.trade_type,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "shares": t.shares,
            "stop_loss": t.stop_loss,
            "target": t.target,
            "confidence": t.confidence,
            "status": t.status,
            "pnl_dollars": t.pnl_dollars,
            "pnl_percent": t.pnl_percent,
            "r_multiple": t.r_multiple,
            "exit_reason": t.exit_reason,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        }
        for t in trades
    ]


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
        _trade_dict(t) for t in trades if t.pnl_dollars and t.pnl_dollars > 0
    ]


@router.get("/losses")
def losses(strategy: str | None = Query(None), db: Session = Depends(get_db)):
    journal = TradeJournal(db)
    trades = journal.get_closed_trades(strategy)
    return [
        _trade_dict(t) for t in trades if t.pnl_dollars and t.pnl_dollars <= 0
    ]


def _trade_dict(t) -> dict:
    return {
        "trade_id": t.trade_id,
        "strategy_name": t.strategy_name,
        "symbol": t.symbol,
        "direction": t.direction,
        "pnl_dollars": t.pnl_dollars,
        "pnl_percent": t.pnl_percent,
        "r_multiple": t.r_multiple,
        "exit_reason": t.exit_reason,
    }
