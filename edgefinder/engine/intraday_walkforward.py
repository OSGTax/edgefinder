"""Walk-forward validation for the INTRADAY engine — the honest gate.

The daily sibling (engine/walkforward.py) is the template. Folds are planned on
the trading-DAY calendar (so geometry matches the daily lane and SPY), but each
fold is replayed bar-by-bar by run_intraday_backtest, which emits a DAILY equity
curve. That daily curve feeds the EXISTING _aggregate, so an intraday scorecard
is byte-shape-identical to a daily one — criteria/verdict/holdout/by_regime all
come free and directly comparable to the monthly fleet.

Day-resolution everywhere a fold is sliced: ``warmup_days`` is trailing CALENDAR
DAYS of minute history before a fold (converted to a session-date slice start),
and the fold's minute bars are clipped to whole ET sessions.
"""

from __future__ import annotations

import bisect
from datetime import date
from typing import Callable

import pandas as pd

from edgefinder.data.minutestore import to_et
from edgefinder.engine.intraday_backtest import run_intraday_backtest
from edgefinder.engine.intraday_strategy import IntradayStrategy
from edgefinder.engine.walkforward import (
    Fold,
    _aggregate,
    _excess_sharpe,
    _regime,
    _window,
    plan_folds,
)


def _session_dates(bars_by_symbol: dict[str, pd.DataFrame]) -> list:
    """Sorted unique ET session dates across all minute frames."""
    days: set = set()
    for df in bars_by_symbol.values():
        if len(df):
            days.update(to_et(df["ts"]).dt.date.tolist())
    return sorted(days)


def _slice_minutes(bars_by_symbol: dict[str, pd.DataFrame],
                   start_day: date, end_day: date) -> dict[str, pd.DataFrame]:
    """Minute frames clipped to ET sessions in [start_day, end_day]."""
    out: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        if not len(df):
            continue
        d = to_et(df["ts"]).dt.date
        sub = df.loc[(d >= start_day) & (d <= end_day)].reset_index(drop=True)
        if len(sub):
            out[sym] = sub
    return out


def _spy_daily_from_minute(spy_bars: pd.DataFrame | None) -> pd.DataFrame | None:
    """A daily (date, close) frame from SPY minute session closes — what
    _regime wants when only minute SPY is available."""
    if spy_bars is None or not len(spy_bars):
        return None
    et = to_et(spy_bars["ts"])
    tmp = spy_bars.assign(_d=et.dt.date.to_numpy()).sort_values("ts")
    rows = [{"date": d, "close": float(g["close"].iloc[-1])}
            for d, g in tmp.groupby("_d", sort=True)]
    return pd.DataFrame(rows)


def run_intraday_walkforward(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_factory: Callable[[], IntradayStrategy],
    *,
    spy_bars: pd.DataFrame | None = None,
    is_days: int = 60,
    oos_days: int = 21,
    step_days: int = 21,
    holdout_days: int = 0,
    holdout_start: date | None = None,
    holdout_eval: bool = False,
    warmup_days: int = 5,
    start_cash: float = 1_000_000.0,
    cost_model=None,
    cost_bps: float = 2.0,
    flatten_at_close: bool = True,
    decision_interval: int = 1,
    bar_seconds: int = 60,
    rebalance_band: float = 0.0,
    risk_adjusted: bool = True,
    pass_min_trades: int = 30,
    calendar: list | None = None,
    prices_label: str = "minute RTH, raw",
) -> dict:
    """Plan folds on the trading-DAY calendar and replay each with the intraday
    engine; aggregate via the daily _aggregate. ``spy_bars`` may be a DAILY
    (date, close) frame for regime tagging OR a SPY minute frame (used both for
    the per-fold benchmark and, reduced to session closes, for regimes)."""
    # benchmark is minute; regime wants daily SPY closes.
    spy_minute = spy_bars if (spy_bars is not None and "ts" in getattr(spy_bars, "columns", [])) else None
    if spy_minute is not None:
        spy_daily = _spy_daily_from_minute(spy_minute)
    else:
        spy_daily = spy_bars  # already daily (date, close)

    days = calendar if calendar is not None else _session_dates(bars_by_symbol)
    planned, planned_holdout = plan_folds(
        days, is_days=is_days, oos_days=oos_days, step_days=step_days,
        holdout_days=holdout_days, holdout_start=holdout_start)
    if holdout_start is not None:
        holdout_days = len(days) - bisect.bisect_left(days, holdout_start)

    def _warm_start_day(oos_start: date) -> date:
        i = bisect.bisect_left(days, oos_start)
        return days[max(0, i - warmup_days)]

    probe = strategy_factory()
    strategy_name = probe.name

    folds: list[Fold] = []
    for oos_start, oos_end in planned:
        warm = _warm_start_day(oos_start)
        fold_bars = _slice_minutes(bars_by_symbol, warm, oos_end)
        bench = (_slice_minutes({"SPY": spy_minute}, oos_start, oos_end).get("SPY")
                 if spy_minute is not None else None)
        result = run_intraday_backtest(
            fold_bars, strategy_factory(),
            start_cash=start_cash, decision_interval=decision_interval,
            cost_model=cost_model, cost_bps=cost_bps,
            trade_start_day=oos_start, flatten_at_close=flatten_at_close,
            benchmark=bench, bar_seconds=bar_seconds,
            rebalance_band=rebalance_band)
        folds.append(Fold(
            index=len(folds), oos_start=oos_start, oos_end=oos_end,
            stats=result.stats, regime=_regime(spy_daily, oos_start, oos_end)))

    holdout = None
    if planned_holdout is not None and holdout_eval:
        h_start, h_end = planned_holdout
        warm = _warm_start_day(h_start)
        h_bars = _slice_minutes(bars_by_symbol, warm, h_end)
        bench = (_slice_minutes({"SPY": spy_minute}, h_start, h_end).get("SPY")
                 if spy_minute is not None else None)
        h = run_intraday_backtest(
            h_bars, strategy_factory(),
            start_cash=start_cash, decision_interval=decision_interval,
            cost_model=cost_model, cost_bps=cost_bps,
            trade_start_day=h_start, flatten_at_close=flatten_at_close,
            benchmark=bench, bar_seconds=bar_seconds,
            rebalance_band=rebalance_band).stats
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
            "regime": _regime(spy_daily, h_start, h_end),
            "params": {},
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
        warmup_days=warmup_days, schedule="intraday", cost_bps=cost_bps,
        start_cash=start_cash, holdout=holdout, holdout_days=holdout_days,
        holdout_eval=holdout_eval, risk_adjusted=risk_adjusted,
        pass_min_trades=pass_min_trades)
    card["config"]["engine"] = "intraday"
    card["config"]["bar"] = "1min"
    card["config"]["flatten_at_close"] = flatten_at_close
    card["config"]["decision_interval"] = decision_interval
    card["config"]["rebalance_band"] = rebalance_band
    card["config"]["costed"] = cost_model is not None
    card["config"]["prices"] = prices_label
    # mean intraday drawdown across folds (the honest intra-session figure)
    intras = [f.stats.get("intraday_max_drawdown_pct") for f in folds
              if f.stats.get("intraday_max_drawdown_pct") is not None]
    card["oos"]["mean_intraday_max_drawdown_pct"] = (
        round(sum(intras) / len(intras), 2) if intras else None)
    if planned_holdout is not None:
        card["config"]["holdout_window"] = _window(*planned_holdout)
    return card
