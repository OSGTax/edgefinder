"""Walk-forward out-of-sample validation harness.

For each rolling fold it optimizes parameters on the in-sample window, then
scores that single config on the *untouched* out-of-sample window. The
aggregate **OOS** scorecard — never in-sample — is the bar for "this strategy
works": positive OOS expectancy and risk-adjusted excess return over SPY,
holding across folds and market regimes.

This is the missing rigor identified in the review: the existing backtester
replays the real engine and computes good metrics, but a single in-sample run
proves nothing. Walk-forward optimize→test is what separates a real edge from
a curve-fit.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import date

import pandas as pd

from edgefinder.analytics.regime import MarketCondition
from edgefinder.backtest.daily_backtest import run_daily_backtest
from edgefinder.backtest.optimize import optimize as optimize_params

logger = logging.getLogger(__name__)

# Defaults tuned to the ~3y of daily_bars currently available. With more
# history, widen the in-sample window. ~252 trading days/year.
DEFAULT_IS_DAYS = 378     # ~1.5y in-sample (optimize)
DEFAULT_OOS_DAYS = 126    # ~6mo out-of-sample (test)
DEFAULT_STEP_DAYS = 126   # non-overlapping OOS windows


def _all_dates(bars_by_symbol: dict) -> list:
    seen: set = set()
    for df in bars_by_symbol.values():
        seen.update(df["date"])
    return sorted(seen)


def _slice(bars_by_symbol: dict, start, end) -> dict:
    out: dict = {}
    for sym, df in bars_by_symbol.items():
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(sub) > 0:
            out[sym] = sub.reset_index(drop=True)
    return out


def _benchmark_window(spy_bars: pd.DataFrame | None, start, end) -> dict | None:
    """SPY buy-hold return over [start, end] (the per-window benchmark)."""
    if spy_bars is None or len(spy_bars) == 0:
        return None
    sub = spy_bars[(spy_bars["date"] >= start) & (spy_bars["date"] <= end)].sort_values("date")
    closes = [c for c in sub["close"].tolist() if c]
    if len(closes) < 2 or closes[0] <= 0:
        return None
    return {
        "symbol": "SPY",
        "return_pct": (closes[-1] - closes[0]) / closes[0] * 100,
        "period": f"{start}..{end}",
    }


def _regime(spy_bars: pd.DataFrame | None, start, end) -> str:
    """Trend + volatility label for a window from SPY bars (VIX-free), mapped
    onto the shared MarketCondition buckets so an edge can be read per regime."""
    if spy_bars is None or len(spy_bars) == 0:
        return "unknown"
    sub = spy_bars[(spy_bars["date"] >= start) & (spy_bars["date"] <= end)].sort_values("date")
    closes = [c for c in sub["close"].tolist() if c]
    if len(closes) < 2 or closes[0] <= 0:
        return "unknown"
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]
    total = (closes[-1] - closes[0]) / closes[0]
    vol_annual = statistics.pstdev(rets) * math.sqrt(252) if len(rets) > 1 else 0.0
    volatile = vol_annual >= 0.20
    if total > 0.02:
        return (MarketCondition.BULL_VOLATILE if volatile else MarketCondition.BULL_CALM).value
    if total < -0.02:
        return (MarketCondition.BEAR_VOLATILE if volatile else MarketCondition.BEAR_CALM).value
    return (MarketCondition.SIDEWAYS_VOLATILE if volatile else MarketCondition.SIDEWAYS_CALM).value


@dataclass
class Fold:
    index: int
    oos_start: date
    oos_end: date
    params: dict
    oos_stats: dict
    regime: str


def run_walkforward(
    strategy_name: str,
    bars_by_symbol: dict,
    *,
    spy_bars: pd.DataFrame | None = None,
    is_days: int = DEFAULT_IS_DAYS,
    oos_days: int = DEFAULT_OOS_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    starting_cash: float = 10_000.0,
    do_optimize: bool = True,
    search_iters: int = 40,
    seed: int = 0,
    min_trades: int = 10,
    progress_cb=None,
) -> dict:
    """Roll IS→OOS folds, optimize on IS, score on OOS, aggregate the OOS
    scorecard. Raises ValueError if there isn't enough history for one fold."""
    days = _all_dates(bars_by_symbol)
    if len(days) < is_days + oos_days:
        raise ValueError(
            f"not enough history: {len(days)} trading days < is+oos "
            f"({is_days}+{oos_days}) — backfill more daily_bars or shrink windows"
        )

    folds: list[Fold] = []
    i = 0
    while i + is_days + oos_days <= len(days):
        is_start, is_end = days[i], days[i + is_days - 1]
        oos_start, oos_end = days[i + is_days], days[i + is_days + oos_days - 1]

        params: dict = {}
        if do_optimize:
            is_bars = _slice(bars_by_symbol, is_start, is_end)
            params, _, _ = optimize_params(
                strategy_name, is_bars,
                benchmark=_benchmark_window(spy_bars, is_start, is_end),
                starting_cash=starting_cash, search_iters=search_iters,
                seed=seed, min_trades=min_trades,
            )

        oos_bars = _slice(bars_by_symbol, oos_start, oos_end)
        res = run_daily_backtest(
            strategy_name, oos_bars, starting_cash=starting_cash,
            benchmark=_benchmark_window(spy_bars, oos_start, oos_end), params=params,
        )
        folds.append(Fold(
            len(folds), oos_start, oos_end, params, res["stats"],
            _regime(spy_bars, oos_start, oos_end),
        ))
        if progress_cb:
            progress_cb({"fold": len(folds), "oos": f"{oos_start}..{oos_end}"})
        i += step_days

    return _aggregate(strategy_name, folds, is_days, oos_days, step_days)


def _aggregate(strategy_name: str, folds: list[Fold], is_days, oos_days, step_days) -> dict:
    rets = [f.oos_stats.get("return_pct") or 0.0 for f in folds]
    sharpes = [f.oos_stats["sharpe"] for f in folds if f.oos_stats.get("sharpe") is not None]
    excess = [f.oos_stats["excess_return_pct"] for f in folds
              if f.oos_stats.get("excess_return_pct") is not None]
    wins = [f.oos_stats["win_rate"] for f in folds if f.oos_stats.get("win_rate") is not None]
    dds = [f.oos_stats.get("max_drawdown_pct") or 0.0 for f in folds]
    trades = sum(f.oos_stats.get("num_closed_trades") or 0 for f in folds)

    # Compounded OOS equity across folds (chain the fold returns).
    comp = 1.0
    for r in rets:
        comp *= (1 + r / 100.0)
    oos_total_return = round((comp - 1) * 100, 2)

    folds_beating_spy = sum(1 for e in excess if e > 0)
    mean_excess = round(statistics.mean(excess), 2) if excess else None
    mean_sharpe = round(statistics.mean(sharpes), 2) if sharpes else None

    # The bar: positive compounded OOS, positive average excess over SPY,
    # beats SPY in a majority of folds, on a meaningful number of trades.
    passed = bool(
        oos_total_return > 0
        and mean_excess is not None and mean_excess > 0
        and excess and folds_beating_spy > len(excess) / 2
        and trades >= 10
    )

    by_regime: dict[str, list] = {}
    for f in folds:
        by_regime.setdefault(f.regime, []).append(f.oos_stats.get("return_pct") or 0.0)
    regime_summary = {
        r: {"folds": len(v), "avg_return_pct": round(statistics.mean(v), 2)}
        for r, v in by_regime.items()
    }

    return {
        "strategy": strategy_name,
        "config": {
            "is_days": is_days, "oos_days": oos_days,
            "step_days": step_days, "num_folds": len(folds),
        },
        "oos": {
            "total_return_pct": oos_total_return,
            "mean_fold_return_pct": round(statistics.mean(rets), 2) if rets else None,
            "mean_sharpe": mean_sharpe,
            "mean_excess_vs_spy_pct": mean_excess,
            "folds_beating_spy": f"{folds_beating_spy}/{len(excess)}" if excess else "n/a",
            "total_trades": trades,
            "mean_win_rate": round(statistics.mean(wins), 1) if wins else None,
            "worst_max_drawdown_pct": round(max(dds), 2) if dds else None,
        },
        "verdict": "PASS" if passed else "FAIL",
        "by_regime": regime_summary,
        "folds": [
            {
                "index": f.index,
                "oos": f"{f.oos_start}..{f.oos_end}",
                "regime": f.regime,
                "params": f.params,
                "stats": f.oos_stats,
            }
            for f in folds
        ],
    }
