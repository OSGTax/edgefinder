"""Walk-forward validation for the portfolio engine — the honest gate.

Rolling IS→OOS folds plus a sealed holdout, every window scored against SPY
over exactly the same dates. Ported from the proven discipline in
edgefinder/backtest/walkforward.py, but driving the pure portfolio engine
(engine/backtest.run_backtest) and supporting only the pre-registered
fixed-parameter path — no per-fold optimizer. A strategy is judged on the
parameters it was committed with; that is the stronger claim, and it removes
the per-fold selection bias the old harness's optimizer path suffered from.

Methodology notes carried over (and disclosed in every scorecard):
- Each fold's bar slice starts ``warmup_days`` calendar entries BEFORE the
  scored window; the engine's ``trade_start`` gates trading and equity marks
  so indicators are warm but stats cover only the scored region (the
  cold-fold ema_200 bug class, fixed 2026-06-05, cannot recur).
- ``trades`` counts FILLS, not round trips — a portfolio engine has no
  natural closed-trade unit. Criteria thresholds are on fills, disclosed.
- Prices are split-adjusted but dividend-UNadjusted (Yahoo 2005-2021 +
  Polygon 2021+ splice). Fine for strategy-vs-SPY on the same data; biased
  against high-yield assets in cross-asset rankings.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from edgefinder.analytics.regime import MarketCondition
from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.strategy import Strategy


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows of a (date, ...) frame with start <= date <= end."""
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def _slice_bars(bars_by_symbol: dict[str, pd.DataFrame], start, end) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in bars_by_symbol.items():
        sub = _slice(df, start, end)
        if len(sub):
            out[sym] = sub
    return out


def _regime(spy_bars: pd.DataFrame | None, start, end) -> str:
    """Trend + volatility label for a window from SPY closes (VIX-free)."""
    if spy_bars is None or len(spy_bars) == 0:
        return "unknown"
    closes = [c for c in _slice(spy_bars, start, end).sort_values("date")["close"] if c]
    if len(closes) < 2 or closes[0] <= 0:
        return "unknown"
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]
    total = (closes[-1] - closes[0]) / closes[0]
    volatile = (statistics.pstdev(rets) * math.sqrt(252) if len(rets) > 1 else 0.0) >= 0.20
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
    stats: dict
    regime: str


def _excess_sharpe(stats: dict) -> float | None:
    """A window's excess Sharpe, imputing Sharpe 0 for a flat (all-cash)
    equity curve. Without this, windows a cash-timer sat out vanish from the
    Sharpe criteria denominator (a bull fold it skipped should count as
    'did not beat SPY') while still collecting drawdown-reduction credit —
    an anti-conservative hole in the gate. The imputation is symmetric: an
    all-cash window in a bear (negative SPY Sharpe) counts positively."""
    if stats.get("excess_sharpe") is not None:
        return stats["excess_sharpe"]
    if stats.get("sharpe") is None and stats.get("benchmark_sharpe") is not None:
        return round(0.0 - stats["benchmark_sharpe"], 2)
    return None


def _all_dates(bars_by_symbol: dict[str, pd.DataFrame]) -> list:
    days: set = set()
    for df in bars_by_symbol.values():
        days.update(df["date"])
    return sorted(days)


def _window(start, end) -> str:
    return f"{start}..{end}"


def plan_folds(
    days: list, *,
    is_days: int = 378, oos_days: int = 126, step_days: int = 126,
    holdout_days: int = 0, holdout_start=None,
) -> tuple[list[tuple], tuple | None]:
    """The fold/holdout geometry, as pure date arithmetic.

    Returns ``(folds, holdout)`` where folds is ``[(oos_start, oos_end), ...]``
    and holdout is ``(start, end)`` or None. Exposed so a caller can resolve
    per-fold point-in-time universes BEFORE loading bars (the CLI does this
    with the SPY calendar, which is the NYSE trading calendar).
    """
    n = len(days)
    if holdout_start is not None:
        wf_n = bisect.bisect_left(days, holdout_start)
        holdout_days = n - wf_n
    else:
        wf_n = n - holdout_days
    if wf_n < is_days + oos_days:
        raise ValueError(
            f"not enough history: {n} trading days minus {holdout_days} holdout "
            f"leaves {wf_n}, need >= is_days+oos_days = {is_days + oos_days}")
    folds = []
    i = 0
    while i + is_days + oos_days <= wf_n:
        folds.append((days[i + is_days], days[i + is_days + oos_days - 1]))
        i += step_days
    holdout = (days[wf_n], days[-1]) if holdout_days > 0 else None
    return folds, holdout


def run_walkforward(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_factory: Callable[[], Strategy],
    *,
    spy_bars: pd.DataFrame | None = None,
    is_days: int = 378,
    oos_days: int = 126,
    step_days: int = 126,
    holdout_days: int = 0,
    holdout_start: date | None = None,
    holdout_eval: bool = True,
    warmup_days: int = 210,
    start_cash: float = 1_000_000.0,
    schedule: str = "monthly",
    cost_bps: float = 2.0,
    rebalance_band: float = 0.0,
    risk_adjusted: bool = True,
    pass_min_trades: int = 30,
    universe_fn: Callable[[date], list[str]] | None = None,
    cost_model=None,
    prices_label: str = "split-adjusted, dividend-unadjusted",
    calendar: list | None = None,
) -> dict:
    """Run rolling out-of-sample folds (and optionally the sealed holdout) and
    return the scorecard dict (same shape the old harness recorded).

    ``strategy_factory`` builds a FRESH strategy instance per window, so
    stateful strategies cannot leak across folds. With ``holdout_days > 0``
    (count-relative) or ``holdout_start`` (a date — pins the sealed boundary
    permanently, so it cannot roll forward as new bars accrue; preferred for
    any multi-run research program) the final stretch of the calendar is
    carved off and untouchable by folds; ``holdout_eval=False`` reserves it
    without burning the one look (the sealed window's dates are disclosed in
    config either way).

    ``start_cash`` defaults high so integer-share flooring is negligible —
    at $10k across 7 ETFs the flooring alone measured ~-1.3pp/fold of phantom
    deficit against the frictionless benchmark.

    ``universe_fn(oos_start) -> [symbols]`` restricts each window to its own
    POINT-IN-TIME universe (resolved as of the day before the window's first
    scored day — without this, a top-N stock universe ranked on the whole
    table selects tomorrow's winners into yesterday's menu, measured at
    ~+3pp/126d of free excess). ``cost_model`` threads the realistic cost
    engine (spread + impact + participation caps) into every fill.

    ``calendar``: the trading calendar to plan fold geometry on. A caller
    that resolved per-window universes against its own calendar (the CLI
    plans on SPY's) MUST pass that same calendar — otherwise geometry is
    re-derived from the union of loaded bars, and one stray off-calendar bar
    would silently shift every window boundary.
    """
    days = calendar if calendar is not None else _all_dates(bars_by_symbol)
    planned, planned_holdout = plan_folds(
        days, is_days=is_days, oos_days=oos_days, step_days=step_days,
        holdout_days=holdout_days, holdout_start=holdout_start)
    if holdout_start is not None:
        holdout_days = len(days) - bisect.bisect_left(days, holdout_start)

    def _window_bars(start, end, window_anchor) -> dict:
        bars = _slice_bars(bars_by_symbol, start, end)
        if universe_fn is not None:
            allowed = universe_fn(window_anchor)
            if allowed is None:
                raise ValueError(
                    f"no PIT universe resolved for window anchor {window_anchor}: "
                    "the universe was planned on a different calendar than the "
                    "fold geometry — pass the planning calendar via calendar=")
            bars = {s: df for s, df in bars.items() if s in set(allowed)}
        return bars

    probe = strategy_factory()
    strategy_name = probe.name

    folds: list[Fold] = []
    for oos_start, oos_end in planned:
        warm_idx = max(0, bisect.bisect_left(days, oos_start) - warmup_days)
        warm_start = days[warm_idx]
        result = run_backtest(
            _window_bars(warm_start, oos_end, oos_start),
            strategy_factory(),
            start_cash=start_cash,
            schedule=schedule,
            cost_bps=cost_bps,
            cost_model=cost_model,
            rebalance_band=rebalance_band,
            trade_start=oos_start,
            benchmark=_slice(spy_bars, oos_start, oos_end) if spy_bars is not None else None,
        )
        folds.append(Fold(
            index=len(folds), oos_start=oos_start, oos_end=oos_end,
            stats=result.stats, regime=_regime(spy_bars, oos_start, oos_end)))

    holdout = None
    if planned_holdout is not None and holdout_eval:
        h_start, h_end = planned_holdout
        warm_idx = max(0, bisect.bisect_left(days, h_start) - warmup_days)
        warm_start = days[warm_idx]
        h = run_backtest(
            _window_bars(warm_start, h_end, h_start),
            strategy_factory(),
            start_cash=start_cash,
            schedule=schedule,
            cost_bps=cost_bps,
            cost_model=cost_model,
            rebalance_band=rebalance_band,
            trade_start=h_start,
            benchmark=_slice(spy_bars, h_start, h_end) if spy_bars is not None else None,
        ).stats
        h_exs = _excess_sharpe(h)
        if risk_adjusted:
            h_passes = ((h_exs or 0) > 0
                        and (h.get("drawdown_reduction_pct") or 0) > 0
                        and h.get("num_trades", 0) >= 3)
        else:
            h_passes = ((h.get("sharpe") or 0) > 0
                        and (h.get("excess_return_pct") or 0) > 0
                        and h.get("num_trades", 0) >= pass_min_trades)
        holdout = {
            "window": _window(h_start, h_end),
            "regime": _regime(spy_bars, h_start, h_end),
            "params": {},          # fixed pre-registered params, by construction
            "return_pct": h.get("return_pct"),
            "sharpe": h.get("sharpe"),
            "excess_vs_spy_pct": h.get("excess_return_pct"),
            "excess_sharpe": h_exs,
            "drawdown_reduction_pct": h.get("drawdown_reduction_pct"),
            "max_drawdown_pct": h.get("max_drawdown_pct"),
            "trades": h.get("num_trades", 0),
            "sharpe_positive": (h.get("sharpe") or 0) > 0,
            "beats_spy": (h.get("excess_return_pct") or 0) > 0,
            "passes": h_passes,
        }

    card = _aggregate(
        strategy_name, folds,
        is_days=is_days, oos_days=oos_days, step_days=step_days,
        warmup_days=warmup_days, schedule=schedule, cost_bps=cost_bps,
        start_cash=start_cash, holdout=holdout, holdout_days=holdout_days,
        holdout_eval=holdout_eval, risk_adjusted=risk_adjusted,
        pass_min_trades=pass_min_trades)
    card["config"]["costed"] = cost_model is not None
    card["config"]["pit_universe"] = universe_fn is not None
    card["config"]["prices"] = prices_label
    card["config"]["rebalance_band"] = rebalance_band
    if planned_holdout is not None:
        # pin the sealed boundary in the record — the carve is count-relative,
        # so without this the "sealed" region would drift as new bars accrue
        card["config"]["holdout_window"] = _window(*planned_holdout)
    return card


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None


def _aggregate(
    strategy_name: str, folds: list[Fold], *,
    is_days: int, oos_days: int, step_days: int, warmup_days: int,
    schedule: str, cost_bps: float, start_cash: float,
    holdout: dict | None, holdout_days: int, holdout_eval: bool,
    risk_adjusted: bool, pass_min_trades: int,
) -> dict:
    """Fold stats -> scorecard with criteria + verdict (engine-agnostic math)."""
    rets = [f.stats.get("return_pct") for f in folds]
    sharpes = [f.stats.get("sharpe") for f in folds]
    excess_rets = [f.stats.get("excess_return_pct") for f in folds]
    excess_sharpes = [_excess_sharpe(f.stats) for f in folds]   # flat folds imputed
    dd_reductions = [f.stats.get("drawdown_reduction_pct") for f in folds]
    total_trades = sum(f.stats.get("num_trades", 0) for f in folds)

    # compounding only makes calendar sense when folds tile exactly
    compounded = 1.0 if step_days == oos_days else None
    if compounded is not None:
        for r in rets:
            if r is not None:
                compounded *= 1 + r / 100

    ex_known = [e for e in excess_rets if e is not None]
    exs_known = [e for e in excess_sharpes if e is not None]
    beats_spy = sum(1 for e in ex_known if e > 0)
    higher_sharpe = sum(1 for e in exs_known if e > 0)

    oos = {
        "total_return_pct": (round((compounded - 1) * 100, 2)
                             if compounded is not None else None),
        "mean_fold_return_pct": _mean(rets),
        "mean_sharpe": _mean(sharpes),
        "mean_excess_vs_spy_pct": _mean(excess_rets),
        "folds_beating_spy": f"{beats_spy}/{len(ex_known)}",
        "mean_excess_sharpe": _mean(excess_sharpes),
        "folds_higher_sharpe": f"{higher_sharpe}/{len(exs_known)}",
        "mean_drawdown_reduction_pct": _mean(dd_reductions),
        "worst_max_drawdown_pct": max(
            (f.stats.get("max_drawdown_pct") or 0) for f in folds) if folds else None,
        "total_trades": total_trades,
    }

    if risk_adjusted:
        mean_exs = _mean(excess_sharpes)
        mean_ddr = _mean(dd_reductions)
        criteria = {
            "mode": "risk_adjusted",
            "sharpe_beats_spy": mean_exs is not None and mean_exs > 0,
            "majority_folds_higher_sharpe": higher_sharpe > len(exs_known) / 2,
            "lower_drawdown_than_spy": mean_ddr is not None and mean_ddr > 0,
            "traded": total_trades >= 3,
        }
    else:
        mean_sh = _mean(sharpes)
        mean_ex = _mean(excess_rets)
        criteria = {
            "mode": "total_return",
            "oos_sharpe_positive": mean_sh is not None and mean_sh > 0,
            "beats_spy_majority_folds": beats_spy > len(ex_known) / 2,
            "mean_excess_positive": mean_ex is not None and mean_ex > 0,
            "min_trades_met": total_trades >= pass_min_trades,
            "min_trades_threshold": pass_min_trades,
        }
    criteria["all_met"] = all(
        v for k, v in criteria.items() if isinstance(v, bool))

    by_regime: dict[str, dict] = {}
    for f in folds:
        slot = by_regime.setdefault(f.regime, {"folds": 0, "excess_sharpes": []})
        slot["folds"] += 1
        exs = _excess_sharpe(f.stats)
        if exs is not None:
            slot["excess_sharpes"].append(exs)
    by_regime = {
        reg: {"folds": s["folds"], "mean_excess_sharpe": _mean(s["excess_sharpes"])}
        for reg, s in by_regime.items()}

    return {
        "strategy": strategy_name,
        "config": {
            "engine": "v2",
            "schedule": schedule,
            "cost_bps": cost_bps,
            "start_cash": start_cash,
            "is_days": is_days,
            "oos_days": oos_days,
            "step_days": step_days,
            "num_folds": len(folds),
            "warmup_days": warmup_days,
            "optimized": False,
            "risk_adjusted": risk_adjusted,
            "holdout_days": holdout_days,
            "holdout_evaluated": holdout is not None,
            "trades_unit": "fills",
            "prices": "split-adjusted, dividend-unadjusted",
            "fundamentals": "none",
        },
        "oos": oos,
        "criteria": criteria,
        "holdout": holdout,
        "by_regime": by_regime,
        "folds": [
            {
                "window": _window(f.oos_start, f.oos_end),
                "regime": f.regime,
                "return_pct": f.stats.get("return_pct"),
                "sharpe": f.stats.get("sharpe"),
                "excess_vs_spy_pct": f.stats.get("excess_return_pct"),
                "excess_sharpe": _excess_sharpe(f.stats),
                "drawdown_reduction_pct": f.stats.get("drawdown_reduction_pct"),
                "max_drawdown_pct": f.stats.get("max_drawdown_pct"),
                "trades": f.stats.get("num_trades", 0),
            }
            for f in folds
        ],
        "verdict": "PASS" if criteria["all_met"] else "FAIL",
    }
