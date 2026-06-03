"""Phase 2 validation lab: in-sample optimizer + walk-forward OOS harness."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.backtest.optimize import objective, optimize, sample_configs
from edgefinder.backtest.walkforward import run_walkforward


def _osc_series(n: int, base: float = 100.0, amp: float = 20.0,
                period: float = 12.0, start: date = date(2024, 1, 1)) -> pd.DataFrame:
    """Oscillating price path so RSI swings through oversold/overbought and the
    strategies actually generate entries/exits."""
    rows = []
    for i in range(n):
        p = base + amp * math.sin(i / period) + (i * 0.02)  # gentle uptrend + waves
        rows.append({
            "date": start + timedelta(days=i),
            "open": p, "high": p * 1.02, "low": p * 0.98,
            "close": p, "volume": 1_000_000.0 + (i % 7) * 100_000.0,
        })
    return pd.DataFrame(rows)


def _bars(n=200):
    return {"AAA": _osc_series(n), "BBB": _osc_series(n, base=50.0, amp=12.0, period=9.0)}


# ── optimizer ───────────────────────────────────────────────────────────


def test_objective_rejects_too_few_trades():
    assert objective({"num_closed_trades": 2, "sharpe": 5}, min_trades=10) == float("-inf")
    score = objective({"num_closed_trades": 50, "sharpe": 1.0, "return_pct": 10.0,
                       "profit_factor": 2.0})
    assert score > 0


def test_sample_configs_distinct_and_bounded():
    space = {"a": [1, 2, 3], "b": [10, 20]}  # 6 combos
    cfgs = sample_configs(space, n=100, seed=1)
    assert len(cfgs) == 6  # can't exceed the grid
    assert len({tuple(sorted(c.items())) for c in cfgs}) == 6  # all distinct


def test_optimize_returns_a_config():
    params, stats, score = optimize(
        "coward", _bars(120), search_iters=4, seed=0, min_trades=1
    )
    assert isinstance(params, dict)
    # With trades present, a best config should have been scored.
    if stats is not None:
        assert math.isfinite(score)


def test_optimize_unknown_strategy_is_empty():
    params, stats, score = optimize("nope", _bars(60), search_iters=2)
    assert params == {} and stats is None


# ── walk-forward ─────────────────────────────────────────────────────────


def test_walkforward_runs_and_scores_oos():
    bars = _bars(200)
    spy = _osc_series(200, base=400.0, amp=15.0, period=20.0)
    sc = run_walkforward(
        "coward", bars, spy_bars=spy,
        is_days=40, oos_days=40, step_days=40,
        search_iters=4, seed=0, min_trades=1,
    )
    assert sc["strategy"] == "coward"
    assert sc["verdict"] in ("PASS", "FAIL")
    assert sc["config"]["num_folds"] >= 1
    assert len(sc["folds"]) == sc["config"]["num_folds"]
    # OOS scorecard fields present
    for k in ("total_return_pct", "mean_sharpe", "mean_excess_vs_spy_pct", "total_trades"):
        assert k in sc["oos"]
    # Every fold carries the config used and its OOS stats
    for f in sc["folds"]:
        assert isinstance(f["params"], dict)
        assert "num_closed_trades" in f["stats"]
        assert f["regime"]  # a regime label was assigned


def test_walkforward_without_optimization_uses_defaults():
    bars = _bars(160)
    sc = run_walkforward(
        "gambler", bars, is_days=40, oos_days=40, step_days=40,
        do_optimize=False, min_trades=1,
    )
    assert all(f["params"] == {} for f in sc["folds"])


def test_walkforward_needs_enough_history():
    with pytest.raises(ValueError):
        run_walkforward("coward", _bars(50), is_days=378, oos_days=126)


def test_walkforward_no_holdout_by_default():
    sc = run_walkforward(
        "coward", _bars(160), is_days=40, oos_days=40, step_days=40,
        search_iters=2, seed=0, min_trades=1,
    )
    assert sc["holdout"] is None
    # criteria block always present; default trade bar is 30
    assert sc["criteria"]["min_trades_threshold"] == 30
    for k in ("oos_sharpe_positive", "beats_spy_majority_folds",
              "min_trades_met", "all_met"):
        assert isinstance(sc["criteria"][k], bool)


def test_walkforward_holdout_is_sealed_and_scored():
    bars = _bars(200)
    spy = _osc_series(200, base=400.0, amp=15.0, period=20.0)
    sc = run_walkforward(
        "coward", bars, spy_bars=spy,
        is_days=40, oos_days=40, step_days=40, holdout_days=40,
        search_iters=4, seed=0, min_trades=1, pass_min_trades=5,
    )
    h = sc["holdout"]
    assert h is not None
    for k in ("window", "sharpe", "excess_vs_spy_pct", "trades",
              "sharpe_positive", "beats_spy", "passes"):
        assert k in h
    assert isinstance(h["passes"], bool)
    # The holdout window must start strictly after the last rolling fold's OOS
    # end — i.e. it was sealed off from every optimization (ISO dates sort
    # lexicographically = chronologically).
    last_fold_end = sc["folds"][-1]["oos"].split("..")[1]
    holdout_start = h["window"].split("..")[0]
    assert holdout_start > last_fold_end
