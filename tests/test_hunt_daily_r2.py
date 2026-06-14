"""Daily-decision hunt round-2 (STATEFUL) roster — shape, honesty, turnover.

Shape tests prove every roster spec builds, returns engine-legal weights,
never crashes on missing/empty data, and only names symbols in the context.
The turnover test is the round's headline claim: hysteresis materially cuts
trade count vs the stateless daily-r1 twin and HOLDS buffer-zone names.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.data.market_data import IndicatorSnapshot
from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.hunt_daily_r2 import HUNT_DAILY_R2_SPECS
from edgefinder.engine.strategies import make_strategy_factory
from edgefinder.engine.strategy import AssetView, RebalanceContext


def _bars(closes, start=date(2024, 1, 2)):
    rows = []
    d = start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        rows.append({"date": d, "open": float(c) * 1.002, "high": float(c) * 1.01,
                     "low": float(c) * 0.99, "close": float(c),
                     "volume": 1_000_000.0})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _asset(symbol, price=100.0, n=300, trend=1.0005, held=False):
    closes = [price * trend ** i for i in range(n)]
    df = _bars(closes)
    ind = IndicatorSnapshot(
        close=closes[-1], ema_50=closes[-1] * 0.98, ema_200=closes[-1] * 0.9,
        rsi=50.0, bb_width=0.05)
    return AssetView(symbol=symbol, price=closes[-1], indicators=ind,
                     history=df, fundamentals=None)


def _ctx(n_assets=40, holdings=None):
    syms = [f"{chr(65 + i % 26)}{'X' * (i // 26)}SY" for i in range(n_assets)]
    assets = {s: _asset(s, price=50.0 + i) for i, s in enumerate(syms)}
    return RebalanceContext(date=date(2025, 3, 4), assets=assets,
                            holdings=holdings or {})


def _short_ctx():
    df = pd.DataFrame([{"date": date(2024, 1, 2), "open": 100.0, "high": 101.0,
                        "low": 99.0, "close": 100.0, "volume": 1_000_000.0}])
    ind = IndicatorSnapshot(close=100.0, ema_50=95.0, ema_200=90.0)
    a = AssetView(symbol="ABC", price=100.0, indicators=ind, history=df)
    return RebalanceContext(date=date(2024, 1, 2), assets={"ABC": a})


@pytest.mark.parametrize("spec", sorted(HUNT_DAILY_R2_SPECS))
def test_every_spec_builds_and_returns_legal_weights(spec):
    strat = make_strategy_factory(spec)()
    assert strat.name
    # exercise both a flat book and a non-trivial held book (hysteresis path)
    held = {"ASY": 0.5, "BSY": 0.5}
    for ctx in (_ctx(), _ctx(holdings=held), _short_ctx(),
                RebalanceContext(date=date(2025, 3, 4), assets={})):
        w = strat.rebalance(ctx) or {}
        assert all(v >= 0 for v in w.values()), f"{spec}: negative weight"
        assert sum(w.values()) <= 1.0 + 1e-9, f"{spec}: levered weights"
        assert all(s in ctx.assets for s in w), f"{spec}: unknown symbol"


def test_random_controls_are_deterministic_and_seed_dependent():
    ctx = _ctx(n_assets=60)
    a = make_strategy_factory("sh_random_71")()
    b = make_strategy_factory("sh_random_73")()
    assert a.rebalance(ctx) == a.rebalance(ctx)
    assert a.rebalance(ctx) != b.rebalance(ctx)


def test_hysteresis_holds_buffer_zone_name():
    # A held name parked in the buffer (entry rank 10, exit rank 30) must be
    # KEPT. Build a universe where one held name ranks ~rank-15 by ret(20).
    strat = make_strategy_factory("sh_mom_20")()
    # 40 names; rank by ret(20) is driven by their per-name trend (higher
    # trend -> higher 20d return -> better rank).
    syms = [f"S{i:02d}" for i in range(40)]
    assets = {}
    for i, s in enumerate(syms):
        # trend decreasing with i: S00 best momentum, S39 worst
        assets[s] = _asset(s, price=100.0, trend=1.001 - i * 0.00002)
    # hold S15 (rank ~15 -> inside 30 exit band, outside 10 entry band)
    ctx = RebalanceContext(date=date(2025, 3, 4), assets=assets,
                           holdings={"S15": 1.0})
    w = strat.rebalance(ctx)
    assert "S15" in w, "buffer-zone held name must be retained (no churn)"


def _seed_trending_bars(n_syms=30, n_bars=120):
    """A multi-day, multi-symbol panel with PERSISTENT cross-sectional rank
    plus small daily jitter so the stateless top-K reselects but the stateful
    buffer holds. Returns {symbol: bars}."""
    import math
    out = {}
    for i in range(n_syms):
        base = 100.0
        drift = 1.0 + (n_syms - i) * 0.0006        # higher i -> weaker trend
        closes = []
        for t in range(n_bars):
            jitter = 1.0 + 0.01 * math.sin((t + i) * 1.7)   # deterministic wobble
            closes.append(base * (drift ** t) * jitter)
        out[f"S{i:02d}"] = _bars(closes)
    return out


class _StatelessTop10Mom20:
    """The daily-r1 twin of sh_mom_20: stateless top-10 by ret(20), reselected
    every day (no hysteresis)."""

    name = "stateless_top10_mom20"

    def rebalance(self, ctx):
        scored = []
        for s, a in ctx.assets.items():
            r = a.ret(20)
            if r is not None:
                scored.append((r, s))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        pick = scored[:10]
        return {s: 0.1 for _, s in pick} if pick else {}


def test_hysteresis_materially_reduces_turnover_vs_stateless_twin():
    bars = _seed_trending_bars()
    stateless = run_backtest(bars, _StatelessTop10Mom20(), schedule="daily",
                             cost_bps=0.0, warmup_days=25,
                             rebalance_band=0.01)
    stateful = run_backtest(bars, make_strategy_factory("sh_mom_20")(),
                            schedule="daily", cost_bps=0.0, warmup_days=25,
                            rebalance_band=0.01)
    assert stateful.stats["num_trades"] < stateless.stats["num_trades"], (
        f"stateful={stateful.stats['num_trades']} not < "
        f"stateless={stateless.stats['num_trades']}")
    # "materially" — at least a third fewer trades on this panel
    assert stateful.stats["num_trades"] <= stateless.stats["num_trades"] * 0.67
