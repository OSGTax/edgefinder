"""Backtest API — replay a strategy over historical daily bars.

Reads OHLC from the flat-file-backfilled ``daily_bars`` table and runs it
through the real trade engine (see ``edgefinder.backtest.daily_backtest``),
so results reflect live entry/exit/sizing/risk logic. Bounded to a handful of
symbols per request to stay responsive (it's CPU-bound per symbol-day).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.backtest.daily_backtest import run_daily_backtest
from edgefinder.db.models import DailyBar
from edgefinder.strategies.base import StrategyRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_SYMBOLS = 25


class BacktestRequest(BaseModel):
    strategy: str
    symbols: list[str] = Field(default_factory=list)
    start: date | None = None
    end: date | None = None
    starting_cash: float = 10_000.0


@router.post("")
def run_backtest(req: BacktestRequest, db: Session = Depends(get_db)):
    if req.strategy not in StrategyRegistry.list_names():
        raise HTTPException(404, f"unknown strategy {req.strategy!r}")

    symbols = sorted({s.strip().upper() for s in req.symbols if s.strip()})
    if not symbols:
        raise HTTPException(422, "provide at least one symbol")
    if len(symbols) > MAX_SYMBOLS:
        raise HTTPException(422, f"max {MAX_SYMBOLS} symbols per backtest")

    q = db.query(DailyBar).filter(DailyBar.symbol.in_(symbols))
    if req.start:
        q = q.filter(DailyBar.date >= req.start)
    if req.end:
        q = q.filter(DailyBar.date <= req.end)
    rows = q.order_by(DailyBar.symbol, DailyBar.date).all()
    if not rows:
        raise HTTPException(
            404,
            "no daily_bars for those symbols/range — run the daily-bar backfill first",
        )

    by_symbol: dict[str, list[dict]] = {}
    for r in rows:
        by_symbol.setdefault(r.symbol, []).append({
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        })
    bars_by_symbol = {s: pd.DataFrame(recs) for s, recs in by_symbol.items()}

    result = run_daily_backtest(
        req.strategy, bars_by_symbol, starting_cash=req.starting_cash
    )
    result["symbols"] = symbols
    result["coverage"] = {
        s: {"bars": len(df), "first": df["date"].min().isoformat(),
            "last": df["date"].max().isoformat()}
        for s, df in bars_by_symbol.items()
    }
    return result
