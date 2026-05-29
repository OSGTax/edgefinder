"""Daily-bar backtester that replays history through the *live* trade engine.

Rather than reimplement strategy logic (which would inevitably drift from
production), this drives the real ``Arena.run_intraday_cycle`` one historical
day at a time: each day's daily bar is fed as the "current" snapshot while the
prior days sit in the arena's bar cache, exactly mirroring how the live
intraday cycle sees cached history + a provisional current bar. Entries,
exits, position sizing, the 20% stop, profit targets, concentration and
max-position caps therefore all run through the same ``VirtualAccount`` /
``Executor`` / ``RiskManager`` code the live system uses — so results can't
diverge from live behaviour by construction.

Reads daily bars from the ``daily_bars`` table (no S3 / minute downloads), so
a backtest is a bounded DB read + in-memory compute — it can't time out on
flat-file I/O.

Caveat — point-in-time fundamentals are not stored, so the fundamental
watchlist filter (``qualifies_stock``) is OFF by default to avoid look-ahead
bias; pass ``fundamentals=`` to apply current fundamentals (look-ahead) if you
want to match the live qualified universe.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from edgefinder.core.events import EventBus
from edgefinder.data.market_data import MarketContext
from edgefinder.trading.arena import ArenaEngine

logger = logging.getLogger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]


class _NullProvider:
    """Arena requires a provider, but the backtest seeds bar caches directly
    and overrides the fetch path, so no provider calls are ever made."""

    def get_bars(self, *a, **k):  # pragma: no cover - never called
        return None


class BacktestArena(ArenaEngine):
    """Arena with wall-clock coupling neutralised for historical replay.

    Uses a private, subscriber-less event bus so simulated trades are never
    persisted to the live trades table or seen by any live handler.
    """

    def __init__(self, provider) -> None:
        super().__init__(provider, event_bus_override=EventBus())

    @staticmethod
    def _minutes_since_market_open() -> float:
        # Each backtest day is a full completed session, so volume-ratio
        # normalisation should treat it as the whole 390-minute day.
        return 390.0

    def _fetch_daily_bars(self, ticker: str):  # noqa: D401 - rely on seeded cache
        return None


def _max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.20 = -20%)."""
    peak = float("-inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return round(mdd, 4)


def run_daily_backtest(
    strategy_name: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    starting_cash: float = 10_000.0,
    fundamentals: dict[str, Any] | None = None,
    min_history: int = 30,
) -> dict:
    """Replay ``bars_by_symbol`` through ``strategy_name`` day by day.

    ``bars_by_symbol``: {symbol: DataFrame with columns
    [date, open, high, low, close, volume]}. Returns a dict with the equity
    curve, realized trades, end-of-run open positions, and summary stats.
    """
    arena = BacktestArena(provider=_NullProvider())
    arena.load_strategies()
    if strategy_name not in arena._slots:
        raise ValueError(
            f"unknown strategy {strategy_name!r}; known: {sorted(arena._slots)}"
        )

    symbols = list(bars_by_symbol)
    arena.set_watchlists({strategy_name: symbols})
    slot = arena._slots[strategy_name]
    # Custom starting capital (executor references the same account object).
    slot.account.starting_capital = float(starting_cash)
    slot.account.cash = float(starting_cash)
    slot.account.peak_equity = float(starting_cash)
    if fundamentals:
        arena.set_fundamentals_cache(fundamentals)

    # Pre-sort each symbol once; advance a per-symbol pointer as days progress
    # (avoids O(days^2) boolean filtering).
    per_sym: dict[str, dict] = {}
    all_days: set[date] = set()
    for sym, df in bars_by_symbol.items():
        d = df.sort_values("date").reset_index(drop=True)
        per_sym[sym] = {
            "dates": list(d["date"]),
            "ohlcv": d[_OHLCV].reset_index(drop=True),
            "ptr": 0,
        }
        all_days.update(d["date"])
    days = sorted(all_days)

    context = MarketContext()  # neutral broad-market state
    equity_curve: list[dict] = []
    closed_all: list = []

    for current_day in days:
        snapshot_data: dict[str, dict] = {}
        for sym, s in per_sym.items():
            dates = s["dates"]
            ptr = s["ptr"]
            while ptr < len(dates) and dates[ptr] < current_day:
                ptr += 1
            s["ptr"] = ptr
            # Prior sessions -> arena bar cache (the provisional bar for
            # `current_day` is appended from the snapshot inside the cycle).
            if ptr >= min_history - 1:
                arena._daily_bars_cache[sym] = s["ohlcv"].iloc[:ptr]
            if ptr < len(dates) and dates[ptr] == current_day:
                bar = s["ohlcv"].iloc[ptr]
                snapshot_data[sym] = {
                    "price": float(bar.close), "open": float(bar.open),
                    "high": float(bar.high), "low": float(bar.low),
                    "volume": float(bar.volume),
                }
        if not snapshot_data:
            continue

        _, closed = arena.run_intraday_cycle(snapshot_data, context)
        closed_all.extend(closed)

        # Mark held positions to the day's close and record equity.
        for pos in slot.account.positions:
            px = snapshot_data.get(pos.symbol, {}).get("price")
            if px:
                pos.market_price = px
        equity_curve.append({
            "date": current_day.isoformat(),
            "equity": round(slot.account.total_equity, 2),
        })

    return _summarize(slot, strategy_name, starting_cash, equity_curve, closed_all)


def _summarize(slot, strategy_name, starting_cash, equity_curve, closed_trades) -> dict:
    closed = [
        {
            "symbol": t.symbol,
            "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4) if t.exit_price is not None else None,
            "shares": t.shares,
            "pnl_dollars": round(t.pnl_dollars, 2) if t.pnl_dollars is not None else None,
            "exit_reason": t.exit_reason,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        }
        for t in closed_trades
    ]
    open_positions = [
        {"symbol": p.symbol, "shares": p.shares,
         "entry_price": round(p.entry_price, 4),
         "market_price": round(p.market_price, 4) if p.market_price else None}
        for p in slot.account.positions
    ]

    final_equity = round(slot.account.total_equity, 2)
    wins = [t for t in closed if (t["pnl_dollars"] or 0) > 0]
    eq_vals = [p["equity"] for p in equity_curve] or [starting_cash]
    return {
        "strategy": strategy_name,
        "starting_cash": starting_cash,
        "final_equity": final_equity,
        "equity_curve": equity_curve,
        "trades": closed,
        "open_positions": open_positions,
        "stats": {
            "return_pct": round((final_equity - starting_cash) / starting_cash * 100, 2)
            if starting_cash else 0.0,
            "num_closed_trades": len(closed),
            "num_open_positions": len(open_positions),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
            "max_drawdown_pct": round(_max_drawdown(eq_vals) * 100, 2),
            "realized_pnl": round(slot.account.realized_pnl, 2),
            "days": len(equity_curve),
        },
    }
