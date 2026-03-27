"""Strategies API — per-strategy accounts, equity curves, scheduler status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.services import get_arena, get_scheduler
from edgefinder.db.models import StrategyAccount, StrategySnapshot
from edgefinder.strategies.base import StrategyRegistry

router = APIRouter()


@router.get("")
def list_strategies():
    """List all registered strategies."""
    return [
        {"name": name, "class": cls.__name__}
        for name, cls in StrategyRegistry.get_all().items()
    ]


@router.get("/accounts")
def get_accounts(db: Session = Depends(get_db)):
    """Get strategy account states — live from arena if available, else DB."""
    arena = get_arena()
    if arena:
        # Live in-memory account state
        return list(arena.get_all_accounts().values())

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

    result: dict[str, list] = {}
    for s in snapshots:
        if s.strategy_name not in result:
            result[s.strategy_name] = []
        result[s.strategy_name].append({
            "date": s.timestamp.strftime("%Y-%m-%d") if s.timestamp else None,
            "total_equity": s.total_equity,
            "total_return_pct": s.total_return_pct,
        })

    return result


@router.get("/scheduler")
def scheduler_status():
    """Get scheduler status and next run times."""
    scheduler = get_scheduler()
    if not scheduler:
        return {"running": False, "jobs": {}, "message": "Pipeline not initialized"}
    return scheduler.get_status()
