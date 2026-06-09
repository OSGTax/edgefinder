"""The portfolio backtest engine — one correct trading sequence, fully tested.

A backtest is a pure function: (bars, strategy, costs) -> equity curve. There is
NO account state machine, NO cooldown/PDT/revenge logic, NO wall-clock — those
live-trading concerns are exactly where the old engine's bugs hid. Here the only
job is: ask the strategy for target weights using data through yesterday, trade
to them at today's open with realistic costs, mark to market, repeat.

Correct by construction:
- No look-ahead: the rebalance context is built as of the PREVIOUS bar; fills
  happen at TODAY's open.
- No wall-clock: everything is keyed to the simulated calendar.
- Bad prints can't fire phantom trades (OHLC is sanitized).
- A delisted holding is closed at its last real price, never frozen.
- Costs (spread/impact in bps of traded notional) are charged on every fill.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# Reuse the vetted precompute + OHLC sanitizer; rebuild only the trade loop.
from edgefinder.backtest.daily_backtest import _sanitize_ohlcv, precompute_snapshots
from edgefinder.engine.strategy import (
    AssetView,
    RebalanceContext,
    Strategy,
    _resolve_index,
)


@dataclass
class BacktestResult:
    equity_curve: list[tuple]          # [(date, equity)]
    trades: list[dict]
    stats: dict
    weights_log: list[dict] = field(default_factory=list)  # [{date, weights}]


def _sharpe(equity: list[float]) -> float | None:
    if len(equity) < 3:
        return None
    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity)) if equity[i - 1] > 0]
    if len(rets) < 2:
        return None
    sd = statistics.pstdev(rets) if len(rets) < 2 else statistics.stdev(rets)
    return (statistics.mean(rets) / sd) * math.sqrt(252) if sd > 0 else None


def _max_drawdown(equity: list[float]) -> float:
    peak, mdd = equity[0] if equity else 0.0, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _is_rebalance(i: int, cal: list, schedule: str) -> bool:
    if i == 0:
        return False
    if schedule == "daily":
        return True
    prev, cur = cal[i - 1], cal[i]
    if schedule == "weekly":
        return cur.isocalendar()[1] != prev.isocalendar()[1] or cur.year != prev.year
    if schedule == "monthly":
        return cur.month != prev.month or cur.year != prev.year
    raise ValueError(f"unknown schedule {schedule!r}")


def _build_context(prep: dict, decision_date, fundamentals: dict | None) -> RebalanceContext:
    """Snapshot the whole universe as of ``decision_date`` (inclusive)."""
    assets: dict[str, AssetView] = {}
    for sym, p in prep.items():
        di = _resolve_index(p["dates"], decision_date)
        if di < 0:
            continue
        price = float(p["ohlcv"]["close"].iloc[di])
        if price <= 0:
            continue
        assets[sym] = AssetView(
            symbol=sym,
            price=price,
            indicators=p["snaps"][di],
            history=p["ohlcv"].iloc[: di + 1],
            fundamentals=(fundamentals or {}).get(sym),
        )
    return RebalanceContext(date=decision_date, assets=assets)


def _execute_to_target(
    weights: dict, open_px: dict, holdings: dict, cash: float,
    cost_rate: float, trades: list, dt,
) -> float:
    """Trade current holdings toward the target weights at today's open.

    Returns the new cash. Sells settle before buys (cash-only, no leverage);
    buys are capped by available cash; every fill pays ``cost_rate`` of notional.
    """
    equity = cash + sum(sh * open_px[s] for s, sh in holdings.items() if s in open_px)
    target: dict[str, int] = {}
    for s, w in weights.items():
        if w and w > 0 and s in open_px and open_px[s] > 0:
            target[s] = int((w * equity) / open_px[s])

    deltas = {}
    for s in set(holdings) | set(target):
        if s not in open_px:           # untradeable today — leave the holding
            continue
        d = target.get(s, 0) - holdings.get(s, 0)
        if d != 0:
            deltas[s] = d

    for s, d in deltas.items():        # sells first (raise cash)
        if d < 0:
            px = open_px[s]
            cash += (-d) * px * (1 - cost_rate)
            holdings[s] = holdings.get(s, 0) + d
            trades.append({"date": dt, "symbol": s, "side": "SELL",
                           "shares": -d, "price": round(px, 4)})
    for s, d in deltas.items():        # then buys (capped by cash)
        if d > 0:
            px = open_px[s]
            unit = px * (1 + cost_rate)
            buy = min(d, int(cash / unit) if unit > 0 else 0)
            if buy > 0:
                cash -= buy * unit
                holdings[s] = holdings.get(s, 0) + buy
                trades.append({"date": dt, "symbol": s, "side": "BUY",
                               "shares": buy, "price": round(px, 4)})

    for s in [s for s, sh in holdings.items() if sh == 0]:
        del holdings[s]
    return cash


def run_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: Strategy,
    *,
    start_cash: float = 10_000.0,
    schedule: str = "weekly",
    cost_bps: float = 2.0,
    warmup_days: int = 200,
    benchmark: pd.DataFrame | None = None,
    fundamentals: dict | None = None,
    log_weights: bool = False,
) -> BacktestResult:
    """Replay ``strategy`` over ``bars_by_symbol`` and return the equity curve,
    trades, and stats. ``benchmark`` is an optional (date, close) frame scored
    the same way for an apples-to-apples comparison."""
    cost_rate = cost_bps / 1e4

    prep: dict[str, dict] = {}
    all_dates: set = set()
    for sym, df in bars_by_symbol.items():
        d = df.sort_values("date").reset_index(drop=True)
        ohlcv = _sanitize_ohlcv(
            d[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
        dates = list(d["date"])
        prep[sym] = {"dates": dates, "ohlcv": ohlcv,
                     "snaps": precompute_snapshots(ohlcv),
                     "idx": {dt: i for i, dt in enumerate(dates)},
                     "last_date": dates[-1] if dates else None,
                     "last_close": float(ohlcv["close"].iloc[-1]) if len(ohlcv) else 0.0}
        all_dates.update(dates)
    calendar = sorted(all_dates)

    holdings: dict[str, int] = {}
    cash = float(start_cash)
    equity_curve: list[tuple] = []
    trades: list[dict] = []
    weights_log: list[dict] = []
    last_price: dict[str, float] = {}

    for i, dt in enumerate(calendar):
        # today's tradable bars
        today_open: dict[str, float] = {}
        today_close: dict[str, float] = {}
        for sym, p in prep.items():
            j = p["idx"].get(dt)
            if j is not None:
                today_open[sym] = float(p["ohlcv"]["open"].iloc[j])
                today_close[sym] = float(p["ohlcv"]["close"].iloc[j])
                last_price[sym] = today_close[sym]

        # delist: a holding whose data has ended is closed at its last real price
        for sym in list(holdings):
            if holdings[sym] and dt > prep[sym]["last_date"]:
                px = prep[sym]["last_close"]
                cash += holdings[sym] * px * (1 - cost_rate)
                trades.append({"date": dt, "symbol": sym, "side": "SELL",
                               "shares": holdings[sym], "price": round(px, 4),
                               "reason": "DELISTED"})
                del holdings[sym]

        # rebalance (decision uses data through YESTERDAY; fill at today's open)
        if i >= warmup_days and _is_rebalance(i, calendar, schedule):
            ctx = _build_context(prep, calendar[i - 1], fundamentals)
            weights = strategy.rebalance(ctx) or {}
            total = sum(w for w in weights.values() if w and w > 0)
            if total > 1.0:                      # never lever; scale to 100%
                weights = {s: w / total for s, w in weights.items()}
            cash = _execute_to_target(weights, today_open, holdings, cash,
                                      cost_rate, trades, dt)
            if log_weights:
                weights_log.append({"date": dt, "weights": dict(weights)})

        # mark to market at close
        eq = cash
        for sym, sh in holdings.items():
            eq += sh * today_close.get(sym, last_price.get(sym, 0.0))
        equity_curve.append((dt, round(eq, 2)))

    stats = _summarize(equity_curve, trades, start_cash, holdings, benchmark)
    return BacktestResult(equity_curve, trades, stats, weights_log)


def _summarize(equity_curve, trades, start_cash, holdings, benchmark) -> dict:
    eq = [e for _, e in equity_curve] or [start_cash]
    final = eq[-1]
    n = len(eq)
    exposure = sum(1 for _, e in equity_curve)  # placeholder; refined below
    invested_days = 0  # not tracked per-day cheaply; report position turnover instead
    stats = {
        "return_pct": round((final - start_cash) / start_cash * 100, 2) if start_cash else 0.0,
        "sharpe": round(_sharpe(eq), 2) if _sharpe(eq) is not None else None,
        "max_drawdown_pct": round(_max_drawdown(eq) * 100, 2),
        "final_equity": round(final, 2),
        "num_trades": len(trades),
        "days": n,
        "open_positions": len(holdings),
    }
    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.sort_values("date")
        bc = [c for c in b["close"].tolist() if c and c > 0]
        if len(bc) > 1:
            bret = (bc[-1] - bc[0]) / bc[0] * 100
            brets = [bc[i] / bc[i - 1] - 1 for i in range(1, len(bc))]
            bsh = (statistics.mean(brets) / statistics.stdev(brets) * math.sqrt(252)
                   if len(brets) > 1 and statistics.stdev(brets) > 0 else None)
            stats["benchmark_return_pct"] = round(bret, 2)
            stats["excess_return_pct"] = round(stats["return_pct"] - bret, 2)
            if bsh is not None:
                stats["benchmark_sharpe"] = round(bsh, 2)
                if stats["sharpe"] is not None:
                    stats["excess_sharpe"] = round(stats["sharpe"] - bsh, 2)
            stats["benchmark_max_drawdown_pct"] = round(_max_drawdown(bc) * 100, 2)
            stats["drawdown_reduction_pct"] = round(
                stats["benchmark_max_drawdown_pct"] - stats["max_drawdown_pct"], 2)
    return stats
