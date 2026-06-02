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

from edgefinder.backtest.daily_backtest import run_daily_backtest

logger = logging.getLogger(__name__)

# Coarse, hand-bounded search spaces (kept small on purpose — more knobs/finer
# grids invite curve-fitting on a ~3y dataset).
PARAM_SPACE: dict[str, dict[str, list]] = {
    "coward": {
        "rsi_oversold": [25, 30, 35, 40],
        "rsi_exit": [65, 70, 75],
        "target_pct": [0.10, 0.15, 0.20],
        "risk_pct": [0.03, 0.05, 0.08],
        "max_hold_days": [10, 20, 30],
        "trailing_stop_pct": [0.08, 0.12, 0.20],
    },
    "gambler": {
        "rsi_low": [35, 40, 45],
        "rsi_high": [55, 60, 65],
        "target_pct": [0.15, 0.25, 0.35],
        "risk_pct": [0.05, 0.10, 0.15],
        "max_hold_days": [10, 20, 30],
        "trailing_stop_pct": [0.08, 0.12, 0.20],
    },
    "degenerate": {
        "volume_spike_mult": [1.5, 2.0, 3.0],
        "rsi_min": [45, 50, 55],
        "target_pct": [0.30, 0.50, 0.75],
        "risk_pct": [0.10, 0.15, 0.20],
        "max_hold_days": [10, 20, 30],
        "trailing_stop_pct": [0.10, 0.15, 0.25],
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

    for cfg in sample_configs(space, search_iters, seed):
        try:
            res = run_daily_backtest(
                strategy_name, bars_by_symbol,
                starting_cash=starting_cash, benchmark=benchmark, params=cfg,
            )
        except Exception:
            logger.exception("optimize: backtest failed (%s, %s)", strategy_name, cfg)
            continue
        score = objective(res["stats"], min_trades)
        if score > best_score:
            best_params, best_stats, best_score = cfg, res["stats"], score

    return best_params, best_stats, best_score
