"""Daily-decision hunt round-1 roster — shape and honesty tests.

These don't judge performance (the validator does); they prove every roster
spec builds, returns engine-legal weights, never crashes on missing/empty
data, and only ever names symbols that exist in the context.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.engine.hunt_daily_r1 import HUNT_DAILY_R1_SPECS
from edgefinder.engine.strategies import make_strategy_factory
from edgefinder.engine.strategy import AssetView, RebalanceContext
from edgefinder.data.market_data import IndicatorSnapshot


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


def _asset(symbol, price=100.0, n=300, trend=1.0005):
    closes = [price * trend ** i for i in range(n)]
    df = _bars(closes)
    ind = IndicatorSnapshot(
        close=closes[-1], ema_50=closes[-1] * 0.98, ema_200=closes[-1] * 0.9,
        rsi=50.0, bb_width=0.05)
    return AssetView(symbol=symbol, price=closes[-1], indicators=ind,
                     history=df, fundamentals=None)


def _ctx(n_assets=40):
    syms = [f"{chr(65 + i % 26)}{'X' * (i // 26)}SY" for i in range(n_assets)]
    assets = {s: _asset(s, price=50.0 + i) for i, s in enumerate(syms)}
    return RebalanceContext(date=date(2025, 3, 4), assets=assets)


def _short_ctx():
    """A 1-bar universe — exercises the len-guards (gap, range expansion)."""
    df = pd.DataFrame([{"date": date(2024, 1, 2), "open": 100.0, "high": 101.0,
                        "low": 99.0, "close": 100.0, "volume": 1_000_000.0}])
    ind = IndicatorSnapshot(close=100.0, ema_200=90.0)
    a = AssetView(symbol="ABC", price=100.0, indicators=ind, history=df)
    return RebalanceContext(date=date(2024, 1, 2), assets={"ABC": a})


@pytest.mark.parametrize("spec", sorted(HUNT_DAILY_R1_SPECS))
def test_every_spec_builds_and_returns_legal_weights(spec):
    strat = make_strategy_factory(spec)()
    assert strat.name  # every candidate is identifiable
    for ctx in (_ctx(), _short_ctx(),
                RebalanceContext(date=date(2025, 3, 4), assets={})):
        w = strat.rebalance(ctx) or {}
        assert all(v >= 0 for v in w.values()), f"{spec}: negative weight"
        assert sum(w.values()) <= 1.0 + 1e-9, f"{spec}: levered weights"
        assert all(s in ctx.assets for s in w), f"{spec}: unknown symbol"


def test_random_controls_are_deterministic_and_seed_dependent():
    ctx = _ctx(n_assets=60)
    a = make_strategy_factory("dr_random_61")()
    b = make_strategy_factory("dr_random_67")()
    assert a.rebalance(ctx) == a.rebalance(ctx)
    assert a.rebalance(ctx) != b.rebalance(ctx)
