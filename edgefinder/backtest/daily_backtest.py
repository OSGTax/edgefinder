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
import math
import statistics
from datetime import date, datetime, time as dtime, timezone
from typing import Any, Callable

import pandas as pd

from config.settings import settings
from edgefinder.core.events import EventBus
from edgefinder.data import indicator_engine as ie
from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.trading.arena import ArenaEngine

_HIST_DAYS = 30

logger = logging.getLogger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]


# Faithful per-row snapshot builder now lives in indicator_engine so the live
# arena can reuse it to seed history from daily_bars. Re-exported under the
# original name to keep the backtester's call sites unchanged.
precompute_snapshots = ie.compute_snapshot_series


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
        # Set per simulated day before each cycle (precomputed indicators).
        self._bt_snaps: dict[str, IndicatorSnapshot] = {}
        self._bt_hist: dict[str, IndicatorHistory] = {}
        # Entry intents decided on the prior day, awaiting the next session's
        # open fill. Keyed by strategy name → list of (ticker, TradeIntent).
        self._pending_entries: dict[str, list] = {}

    def _build_market_data(self, snapshot_data, market_context):
        """Build MarketData from precomputed snapshots instead of recomputing
        indicators each day — same MarketData the live cycle would see, ~100x
        faster, so full-universe replays are feasible."""
        result = {}
        tickers: set[str] = set()
        for slot in self._slots.values():
            tickers.update(slot.watchlist)
            for pos in slot.account.positions:
                tickers.add(pos.symbol)
        for ticker in tickers:
            snap = snapshot_data.get(ticker)
            cur = self._bt_snaps.get(ticker)
            if not snap or cur is None:
                continue
            current_price = snap.get("price", 0.0)
            if not current_price:
                continue
            today_vol = snap.get("volume", 0.0)
            avg_vol = cur.volume_avg or 0.0
            # Full-day bar -> time_factor 1.0, so volume_ratio is raw.
            volume_ratio = (today_vol / avg_vol) if avg_vol > 0 else 0.0
            result[ticker] = MarketData(
                ticker=ticker,
                current=cur,
                history=self._bt_hist.get(ticker) or IndicatorHistory(max_days=_HIST_DAYS),
                fundamentals=self._fundamentals_cache.get(ticker),
                context=market_context,
                current_price=current_price,
                today_volume=today_vol,
                avg_daily_volume=avg_vol,
                volume_ratio=volume_ratio,
            )
        return result

    @staticmethod
    def _minutes_since_market_open() -> float:
        # Each backtest day is a full completed session, so volume-ratio
        # normalisation should treat it as the whole 390-minute day.
        return 390.0

    def _fetch_daily_bars(self, ticker: str):  # noqa: D401 - rely on seeded cache
        return None

    def _check_entries(self, slot, market_data_map, snapshot_data):
        """Fill entries at the NEXT session's open, not the close that produced
        the signal — removing same-bar look-ahead.

        A signal computed from day T's *completed* bar is queued, then filled
        at day T+1's open (the snapshot ``open``). The live arena fills at the
        real-time current price (correct for live); only the backtest must
        simulate the realistic next-open fill, so this override lives here and
        the live ``_check_entries`` is untouched. Stops/targets are recomputed
        off the actual fill price in ``_execute_intent``, so nothing carries a
        stale signal-day level.
        """
        name = slot.strategy.name
        opened: list = []

        # 1. Fill intents decided yesterday at TODAY's open.
        for ticker, intent in self._pending_entries.pop(name, []):
            if slot.account.get_position(ticker):
                continue
            snap = snapshot_data.get(ticker)
            if not snap:
                continue  # symbol not trading today — drop the stale signal
            open_px = snap.get("open")
            if not open_px:
                continue
            trade = self._execute_intent(slot, intent, open_px)
            if trade:
                opened.append(trade)

        # 2. Generate today's signals from the completed bar and queue them for
        #    next session's open. Reuses the real strategy methods (qualify +
        #    evaluate) so strategy decision logic never drifts from live.
        queued: list = []
        for ticker in slot.watchlist:
            if slot.account.get_position(ticker):
                continue
            mdata = market_data_map.get(ticker)
            if mdata is None:
                continue
            cached_fund = self._fundamentals_cache.get(ticker)
            if cached_fund is not None:
                try:
                    if not slot.strategy.qualifies_stock(cached_fund):
                        continue
                except Exception:
                    logger.exception(
                        "[%s] Re-qualification raised for %s", name, ticker
                    )
            try:
                intent = slot.strategy.evaluate(ticker, mdata)
            except Exception:
                logger.exception("[%s] evaluate() failed for %s", name, ticker)
                continue
            if intent is not None:
                queued.append((ticker, intent))
        self._pending_entries[name] = queued

        return opened


def _max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.20 = -20%)."""
    peak = float("-inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return round(mdd, 4)


def prepare_bars(bars_by_symbol: dict, progress_cb=None) -> tuple[dict, list]:
    """Precompute the immutable per-symbol structures (sorted dates, OHLCV,
    indicator snapshots) once. The optimizer reuses this across many param
    configs so the precompute (~2/3 of a backtest's cost) runs once per window
    instead of once per config. Returns ``(prep, sorted_days)``."""
    prep: dict[str, dict] = {}
    all_days: set = set()
    items = list(bars_by_symbol.items())
    for i, (sym, df) in enumerate(items):
        d = df.sort_values("date").reset_index(drop=True)
        ohlcv = d[_OHLCV].reset_index(drop=True)
        prep[sym] = {
            "dates": list(d["date"]),
            "ohlcv": ohlcv,
            "snaps": precompute_snapshots(ohlcv),
        }
        all_days.update(d["date"])
        if progress_cb and (i % 50 == 0 or i == len(items) - 1):
            progress_cb({"phase": "prepare", "done": i + 1, "total": len(items)})
    return prep, sorted(all_days)


def run_daily_backtest(
    strategy_name: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    starting_cash: float = 10_000.0,
    fundamentals: dict[str, Any] | None = None,
    benchmark: dict | None = None,
    progress_cb: Callable[[dict], None] | None = None,
    min_history: int = 30,
    params: dict | None = None,
    prepared: tuple | None = None,
    trade_start=None,
    spy_bars: pd.DataFrame | None = None,
) -> dict:
    """Replay ``bars_by_symbol`` through ``strategy_name`` day by day.

    ``bars_by_symbol``: {symbol: DataFrame with columns
    [date, open, high, low, close, volume]}. Returns a dict with the equity
    curve, realized trades, end-of-run open positions, and summary stats.

    ``prepared``: optional ``(prep, days)`` from ``prepare_bars`` — the
    immutable precomputed snapshots. Pass it to reuse one precompute across
    many param configs (the optimizer's hot loop); the per-run cursors
    (history pointers) are always rebuilt fresh, so reuse is safe.

    ``trade_start``: optional date — bars before it are WARMUP only (indicators
    and history accumulate; no trading, no equity marks). Fold-based tests must
    pass warmup bars + trade_start or long-lookback indicators (ema_200 needs
    200 bars) are None for the whole window — the bug that silently crippled
    every fold result before 2026-06-05. Stats (days, exposure, Sharpe) cover
    only the scored region.

    ``spy_bars``: optional full SPY daily frame (date/close). When given, each
    simulated day gets a REAL MarketContext (spy_price, spy_change_pct,
    spy_sma_200) computed from SPY closes up to THAT day — so regime-gated
    strategies are testable. Without it the context is neutral (all zeros =
    "unknown"), matching the pre-2026-06 behavior.
    """
    arena = BacktestArena(provider=_NullProvider())
    arena.load_strategies()
    if strategy_name not in arena._slots:
        raise ValueError(
            f"unknown strategy {strategy_name!r}; known: {sorted(arena._slots)}"
        )

    symbols = list(bars_by_symbol)
    arena.set_watchlists({strategy_name: symbols})
    # Apply tuned parameters (rebuilds risk manager + account caps) before
    # the run so optimizer/validated configs take effect.
    if params:
        arena.configure_strategy(strategy_name, params)
    slot = arena._slots[strategy_name]
    # Custom starting capital (executor references the same account object).
    slot.account.starting_capital = float(starting_cash)
    slot.account.cash = float(starting_cash)
    slot.account.peak_equity = float(starting_cash)
    if fundamentals:
        arena.set_fundamentals_cache(fundamentals)

    # Immutable precompute (sorted dates, OHLCV, indicator snapshots): reuse a
    # cached copy across param configs when given, else compute it once here.
    if prepared is None:
        prep, days = prepare_bars(bars_by_symbol, progress_cb)
    else:
        prep, days = prepared
    # Per-run cursors (history buffer + pointers) are always fresh — the run
    # mutates these, never the shared immutable prep.
    per_sym: dict[str, dict] = {
        sym: {
            "dates": p["dates"], "ohlcv": p["ohlcv"], "snaps": p["snaps"],
            "hist": IndicatorHistory(max_days=_HIST_DAYS),
            "added": 0,   # rolling-history high-water mark
            "ptr": 0,
        }
        for sym, p in prep.items()
    }

    # Broad-market context: real per-day SPY state when spy_bars is given
    # (price, day change, 200dma — no look-ahead: day T uses closes <= T),
    # else a neutral context (all zeros = unknown).
    neutral_context = MarketContext()
    context_by_day: dict = {}
    if spy_bars is not None and len(spy_bars) > 0:
        sdf = spy_bars.sort_values("date").reset_index(drop=True)
        closes = sdf["close"].astype(float)
        sma200 = closes.rolling(200).mean()
        prev_close = closes.shift(1)
        chg = ((closes - prev_close) / prev_close * 100).fillna(0.0)
        for idx in range(len(sdf)):
            d = sdf["date"].iloc[idx]
            key = d.date() if hasattr(d, "date") else d
            context_by_day[key] = MarketContext(
                spy_price=float(closes.iloc[idx]),
                spy_change_pct=round(float(chg.iloc[idx]), 4),
                spy_sma_200=(
                    float(sma200.iloc[idx]) if pd.notna(sma200.iloc[idx]) else 0.0
                ),
            )

    equity_curve: list[dict] = []
    closed_all: list = []
    exposure_days = 0

    n_days = len(days)
    for day_idx, current_day in enumerate(days):
        if trade_start is not None and current_day < trade_start:
            continue  # warmup: indicators/history only (advanced lazily below)
        if progress_cb and (day_idx % 25 == 0 or day_idx == n_days - 1):
            progress_cb({"phase": "simulate", "done": day_idx + 1, "total": n_days})
        snapshot_data: dict[str, dict] = {}
        bt_snaps: dict[str, IndicatorSnapshot] = {}
        bt_hist: dict[str, IndicatorHistory] = {}
        for sym, s in per_sym.items():
            dates = s["dates"]
            ptr = s["ptr"]
            while ptr < len(dates) and dates[ptr] < current_day:
                ptr += 1
            s["ptr"] = ptr
            # Roll history forward to include every prior day [0, ptr) exactly
            # once — mirrors the live daily cycle (history holds up to yesterday).
            while s["added"] < ptr:
                s["hist"].add(s["snaps"][s["added"]])
                s["added"] += 1
            if ptr < len(dates) and dates[ptr] == current_day:
                bar = s["ohlcv"].iloc[ptr]
                snapshot_data[sym] = {
                    "price": float(bar.close), "open": float(bar.open),
                    "high": float(bar.high), "low": float(bar.low),
                    "volume": float(bar.volume),
                }
                bt_snaps[sym] = s["snaps"][ptr]      # indicators incl. today
                bt_hist[sym] = s["hist"]
        if not snapshot_data:
            continue

        arena._bt_snaps = bt_snaps
        arena._bt_hist = bt_hist
        # Drive the engine's clock off the simulated day so time-based exits
        # (max-hold) replay faithfully instead of keying off wall-clock now().
        cd = current_day.date() if hasattr(current_day, "date") else current_day
        arena._clock = datetime.combine(cd, dtime(16, 0), tzinfo=timezone.utc)
        day_context = context_by_day.get(cd, neutral_context)
        _, closed = arena.run_intraday_cycle(snapshot_data, day_context)
        closed_all.extend(closed)

        # Mark held positions to the day's close and record equity.
        for pos in slot.account.positions:
            px = snapshot_data.get(pos.symbol, {}).get("price")
            if px:
                pos.market_price = px
        if slot.account.positions:
            exposure_days += 1
        equity_curve.append({
            "date": current_day.isoformat(),
            "equity": round(slot.account.total_equity, 2),
        })

    return _summarize(
        slot, strategy_name, starting_cash, equity_curve, closed_all,
        exposure_days=exposure_days, benchmark=benchmark,
    )


def _sharpe(equity: list[float]) -> float | None:
    """Annualised Sharpe of daily equity returns (rf=0), ~252 trading days."""
    rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity)) if equity[i - 1] > 0]
    if len(rets) < 2:
        return None
    sd = statistics.pstdev(rets)
    if sd == 0:
        return None
    return round(statistics.mean(rets) / sd * math.sqrt(252), 2)


def _cagr(start: float, end: float, n_days: int) -> float | None:
    years = n_days / 252.0
    if years <= 0 or start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1 / years) - 1) * 100, 2)


def _summarize(slot, strategy_name, starting_cash, equity_curve, closed_trades,
               *, exposure_days: int = 0, benchmark: dict | None = None) -> dict:
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
    losses = [t for t in closed if (t["pnl_dollars"] or 0) <= 0]
    eq_vals = [p["equity"] for p in equity_curve] or [starting_cash]
    n_days = len(equity_curve)

    gross_profit = sum(t["pnl_dollars"] or 0 for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] or 0 for t in losses))
    return_pct = round((final_equity - starting_cash) / starting_cash * 100, 2) if starting_cash else 0.0

    stats = {
        "return_pct": return_pct,
        "cagr_pct": _cagr(starting_cash, final_equity, n_days),
        "sharpe": _sharpe(eq_vals),
        "max_drawdown_pct": round(_max_drawdown(eq_vals) * 100, 2),
        "num_closed_trades": len(closed),
        "num_open_positions": len(open_positions),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "avg_win": round(gross_profit / len(wins), 2) if wins else None,
        "avg_loss": round(sum(t["pnl_dollars"] or 0 for t in losses) / len(losses), 2) if losses else None,
        "exposure_pct": round(exposure_days / n_days * 100, 1) if n_days else 0.0,
        "realized_pnl": round(slot.account.realized_pnl, 2),
        "days": n_days,
    }

    # Benchmark (e.g., SPY buy-hold over the overlapping window)
    if benchmark and benchmark.get("return_pct") is not None:
        stats["benchmark_symbol"] = benchmark.get("symbol")
        stats["benchmark_return_pct"] = round(benchmark["return_pct"], 2)
        stats["benchmark_period"] = benchmark.get("period")
        stats["excess_return_pct"] = round(return_pct - benchmark["return_pct"], 2)

    return {
        "strategy": strategy_name,
        "starting_cash": starting_cash,
        "final_equity": final_equity,
        "equity_curve": equity_curve,
        "trades": closed,
        "open_positions": open_positions,
        "stats": stats,
    }
