"""In-sample parameter optimization for EdgeFinder strategies.

Searches a small, hand-bounded parameter space per strategy and returns the
best config by an overfit-resistant objective. Designed to be driven by the
walk-forward harness: optimize on the in-sample window, then score the winning
config on the *untouched* out-of-sample window.

Anti-overfit by construction: few knobs, coarse grids around the hand-tuned
defaults, a minimum-trade gate, and an objective that rewards risk-adjusted
return (Sharpe) rather than raw return.
"""

from __future__ import annotations

import logging
import random

from edgefinder.backtest.daily_backtest import prepare_bars, run_daily_backtest

logger = logging.getLogger(__name__)

# Search spaces. Deliberately wider than the original hand-bounded grids — the
# walk-forward + sealed-holdout structure is what guards against curve-fitting,
# so a broad search is the honest way to ask "does *any* config work?". stop_pct
# stays fixed (non-negotiable 20% catastrophic stop), so it is not a knob here.
PARAM_SPACE: dict[str, dict[str, list]] = {
    "coward": {
        "rsi_oversold": [20, 25, 30, 35, 40, 45],
        "rsi_exit": [55, 60, 65, 70, 75, 80],
        "target_pct": [0.06, 0.10, 0.15, 0.20, 0.30],
        "risk_pct": [0.02, 0.03, 0.05, 0.08, 0.10],
        "max_hold_days": [5, 10, 20, 30, 45],
        "trailing_stop_pct": [0.05, 0.08, 0.12, 0.20, 0.30],
    },
    "gambler": {
        "rsi_low": [30, 35, 40, 45, 50],
        "rsi_high": [50, 55, 60, 65, 70],
        "target_pct": [0.10, 0.15, 0.25, 0.35, 0.50],
        "risk_pct": [0.03, 0.05, 0.10, 0.15, 0.20],
        "max_hold_days": [5, 10, 20, 30, 45],
        "trailing_stop_pct": [0.05, 0.08, 0.12, 0.20, 0.30],
    },
    "pullback_rider": {
        "rsi_floor": [35, 40, 45],
        "rsi_exit": [65, 70, 75],
        "target_pct": [0.06, 0.08, 0.10, 0.12],
        "max_hold_days": [10, 15, 20],
        "adx_min": [0, 15, 20],
        "risk_pct": [0.02, 0.03],
    },
    "gap_drift": {
        "gap_min": [0.03, 0.05, 0.07],
        "close_loc": [0.5, 0.65, 0.8],
        "fail_pct": [0.04, 0.06, 0.08],
        "target_pct": [0.10, 0.15, 0.20],
        "max_hold_days": [10, 15, 20],
        "trend_gate": [True, False],
    },
    "gap_drift_v2": {
        "atr_mult": [1.5, 2.0, 2.5, 3.0],
        "close_loc": [0.5, 0.65, 0.8],
        "fail_pct": [0.04, 0.06, 0.08],
        "target_pct": [0.10, 0.15, 0.20],
        "max_hold_days": [10, 15, 20],
        "trend_gate": [True, False],
    },
    "tom_seasonality": {
        # Calendar window edges only; everything else fixed by design.
        "entry_day": [23, 25, 27],
        "exit_day": [3, 5, 8],
    },
    "dual_momentum": {
        "top_k": [2, 3, 4],
        # lookback_ema reserved; score uses ema_200 (fixed) for now.
    },
    "micro_reversal": {
        # Washout definition + recovery exit. risk_pct/target_pct FIXED.
        "lookback": [2, 3, 5],
        "drop_pct": [0.08, 0.12, 0.18],
        "rsi_entry": [20, 30, 40],
        "rsi_exit": [50, 55, 60],
        "max_hold_days": [3, 5, 8],
    },
    "xsec_mom": {
        # Score definition and exit_rank (3x top_k) are FIXED by design —
        # only portfolio breadth and the recycle horizon are searched.
        "top_k": [3, 5, 10],
        "max_hold_days": [21, 42, 63],
    },
    "gap_carry": {
        # Entry is FIXED at gap_drift v1's pre-registered defaults by design
        # (one variable at a time) — only the exit side is searched.
        "fail_pct": [0.04, 0.06, 0.08],
        "max_hold_days": [30, 45, 60],
    },
    "trend_dip": {
        "wr_entry": [-95, -90, -85],
        "down_days_min": [2, 3, 4],
        "rsi_exit": [55, 60, 65],
        "max_hold_days": [4, 6, 10],
        "risk_pct": [0.02, 0.03],
    },
    "turtle_adx": {
        "adx_min": [18, 22, 26],
        "vol_min": [1.0, 1.2, 1.5],
        "target_pct": [0.25, 0.40, 0.60],
        "trailing_stop_pct": [0.08, 0.12, 0.15],
        "max_hold_days": [30, 45, 60],
        "risk_pct": [0.02, 0.03],
    },
    "degenerate": {
        # Lower volume/rsi floors than before so the optimizer can pick
        # higher-frequency configs (raises trade count, same volume-spike
        # + bullish-momentum thesis).
        "volume_spike_mult": [1.25, 1.5, 2.0, 2.5, 3.0],
        "rsi_min": [35, 40, 45, 50, 55],
        "target_pct": [0.20, 0.30, 0.50, 0.75, 1.00],
        "risk_pct": [0.05, 0.10, 0.15, 0.20, 0.25],
        "max_hold_days": [5, 10, 20, 30, 45],
        "trailing_stop_pct": [0.08, 0.10, 0.15, 0.25, 0.35],
    },
}

MIN_TRADES = 10  # ignore configs that barely trade (not statistically meaningful)


def objective(stats: dict, min_trades: int = MIN_TRADES) -> float:
    """Scalar score for a backtest result (higher is better; -inf if too few
    trades). Sharpe dominates (risk-adjusted), total/excess return is a
    tie-breaker, with a small profit-factor nudge. Terms are scaled to O(1)."""
    n = stats.get("num_closed_trades") or 0
    if n < min_trades:
        return float("-inf")
    sharpe = stats.get("sharpe") or 0.0
    ret = stats.get("excess_return_pct")
    if ret is None:
        ret = stats.get("return_pct") or 0.0
    pf = stats.get("profit_factor") or 0.0
    return sharpe + ret / 100.0 + min(pf, 5.0) / 10.0


def sample_configs(space: dict, n: int, seed: int = 0) -> list[dict]:
    """Random sample of up to n distinct configs from the grid. Random search
    beats grid search for several parameters at a fixed evaluation budget."""
    rng = random.Random(seed)
    keys = list(space)
    total = 1
    for v in space.values():
        total *= len(v)
    n = min(n, total)
    seen: set = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        cfg = {k: rng.choice(space[k]) for k in keys}
        sig = tuple(cfg[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def optimize(
    strategy_name: str,
    bars_by_symbol: dict,
    *,
    benchmark: dict | None = None,
    starting_cash: float = 10_000.0,
    search_iters: int = 40,
    seed: int = 0,
    min_trades: int = MIN_TRADES,
    trade_start=None,
    spy_bars=None,
    cost_model=None,
) -> tuple[dict, dict | None, float]:
    """Search ``PARAM_SPACE[strategy]`` on the given (in-sample) bars.

    Returns ``(best_params, best_stats, best_score)``. Falls back to defaults
    (``{}``) when no sampled config clears the min-trade gate.
    """
    space = PARAM_SPACE.get(strategy_name, {})
    if not space:
        return {}, None, float("-inf")

    best_params: dict = {}
    best_stats: dict | None = None
    best_score = float("-inf")

    # Precompute snapshots once for this window; every config reuses them
    # (the precompute is ~2/3 of a backtest, so this ~3x's the search).
    prepared = prepare_bars(bars_by_symbol)

    for cfg in sample_configs(space, search_iters, seed):
        try:
            res = run_daily_backtest(
                strategy_name, bars_by_symbol,
                starting_cash=starting_cash, benchmark=benchmark, params=cfg,
                prepared=prepared, trade_start=trade_start, spy_bars=spy_bars,
                cost_model=cost_model,
            )
        except Exception:
            logger.exception("optimize: backtest failed (%s, %s)", strategy_name, cfg)
            continue
        score = objective(res["stats"], min_trades)
        if score > best_score:
            best_params, best_stats, best_score = cfg, res["stats"], score

    return best_params, best_stats, best_score
