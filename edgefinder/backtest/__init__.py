"""Minute-bar backtesting on Massive flat files.

A small, self-contained event-driven backtester — a foundation to extend
toward parity with the live arena, not a drop-in replacement for it.
"""

from edgefinder.backtest.engine import (
    Bar,
    BacktestContext,
    BacktestEngine,
    BacktestResult,
    Fill,
    Order,
    Position,
    load_minute_bars,
)

__all__ = [
    "Bar",
    "BacktestContext",
    "BacktestEngine",
    "BacktestResult",
    "Fill",
    "Order",
    "Position",
    "load_minute_bars",
]
