"""Hunt round-1 roster + PIT-fundamentals wiring — shape and honesty tests.

These don't judge performance (the validator does); they prove every roster
spec builds, returns engine-legal weights, never crashes on missing data,
and that the PIT fundamentals path feeds strategies dated snapshots.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import FundamentalsSnapshot  # registers the table for db fixtures
from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.hunt_r1 import ETF_MENU, HUNT_SPECS
from edgefinder.engine.hunt_r2 import HUNT_R2_SPECS
from edgefinder.engine.strategies import make_strategy_factory
from edgefinder.engine.strategy import AssetView, RebalanceContext
from edgefinder.data.market_data import IndicatorSnapshot


def _bars(closes, start=date(2024, 1, 2)):
    rows = []
    d = start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        rows.append({"date": d, "open": float(c), "high": float(c) * 1.01,
                     "low": float(c) * 0.99, "close": float(c),
                     "volume": 1_000_000.0})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _asset(symbol, price=100.0, n=300, fundamentals=None, trend=1.0005):
    closes = [price * trend ** i for i in range(n)]
    df = _bars(closes)
    ind = IndicatorSnapshot(
        close=closes[-1], ema_50=closes[-1] * 0.98, ema_200=closes[-1] * 0.9,
        rsi=50.0, bb_width=0.05)
    return AssetView(symbol=symbol, price=closes[-1], indicators=ind,
                     history=df, fundamentals=fundamentals)


GOOD_F = TickerFundamentals(
    symbol="X", earnings_per_share=5.0, earnings_growth=0.25,
    revenue_growth=0.15, debt_to_equity=0.4, current_ratio=2.5,
    return_on_equity=0.22)


def _ctx(n_assets=30, with_fundamentals=False):
    assets = {}
    syms = [f"{chr(65 + i % 26)}{'X' * (i // 26)}SY" for i in range(n_assets)]
    for s in ETF_MENU:
        assets[s] = _asset(s)
    for i, s in enumerate(syms):
        f = GOOD_F if (with_fundamentals and i % 2 == 0) else None
        assets[s] = _asset(s, price=50.0 + i, fundamentals=f)
    return RebalanceContext(date=date(2025, 3, 4), assets=assets)


@pytest.mark.parametrize("spec", sorted(HUNT_SPECS) + sorted(HUNT_R2_SPECS))
def test_every_spec_builds_and_returns_legal_weights(spec):
    strat = make_strategy_factory(spec)()
    assert strat.name  # every candidate is identifiable
    for ctx in (_ctx(), _ctx(with_fundamentals=True),
                RebalanceContext(date=date(2025, 3, 4), assets={})):
        w = strat.rebalance(ctx) or {}
        assert all(v >= 0 for v in w.values()), f"{spec}: negative weight"
        assert sum(w.values()) <= 1.0 + 1e-9, f"{spec}: levered weights"
        assert all(s in ctx.assets for s in w), f"{spec}: unknown symbol"


@pytest.mark.parametrize("spec", ["garp_classic", "garp_quality",
                                  "earnings_growers", "low_debt_value",
                                  "cash_rich_growth", "deep_value_pe10",
                                  "garp_momentum"])
def test_fundamental_strategies_hold_cash_without_fundamentals(spec):
    # an all-None-fundamentals universe must yield no positions, not a crash
    strat = make_strategy_factory(spec)()
    w = strat.rebalance(_ctx(with_fundamentals=False))
    assert w == {}


def test_random_k_is_deterministic_and_seed_dependent():
    ctx = _ctx(n_assets=60)
    a = make_strategy_factory("random_20_s7")()
    b = make_strategy_factory("random_20_s13")()
    assert a.rebalance(ctx) == a.rebalance(ctx)   # same month -> same draw
    assert a.rebalance(ctx) != b.rebalance(ctx)   # seeds differ


def test_pit_fundamentals_flow_through_backtest():
    """run_backtest(fundamentals=<asof source>) hands strategies DATED
    snapshots: the GARP screen must hold cash before coverage begins and
    buy once the snapshot date passes."""
    coverage_start = date(2024, 6, 3)

    class StubPIT:
        def asof(self, symbol, as_of):
            if symbol == "AAA" and as_of >= coverage_start:
                return GOOD_F
            return None

    closes = [100.0] * 300
    bars = {"AAA": _bars(closes, start=date(2024, 1, 2))}
    res = run_backtest(bars, make_strategy_factory("garp_classic")(),
                       schedule="monthly", cost_bps=0.0, warmup_days=5,
                       fundamentals=StubPIT(), log_weights=True)
    dated = {w["date"]: w["weights"] for w in res.weights_log}
    before = [w for d, w in dated.items() if d < coverage_start]
    after = [w for d, w in dated.items() if d > coverage_start + timedelta(days=40)]
    assert all(w == {} for w in before), "bought before PIT coverage existed"
    assert any(w.get("AAA") for w in after), "never bought after coverage began"


def test_walkforward_discloses_fundamentals_source():
    from edgefinder.engine.walkforward import run_walkforward

    closes = list(range(100, 800))
    bars = {"AAA": _bars([float(c) for c in closes], start=date(2022, 1, 3))}

    class StubPIT:
        def asof(self, symbol, as_of):
            return None

    card = run_walkforward(bars, make_strategy_factory("garp_classic"),
                           is_days=200, oos_days=100, step_days=100,
                           fundamentals=StubPIT())
    assert card["config"]["fundamentals"] == "pit"
    card2 = run_walkforward(bars, make_strategy_factory("garp_classic"),
                            is_days=200, oos_days=100, step_days=100)
    assert card2["config"]["fundamentals"] == "none"


def test_pit_preload_serves_asof_without_session(db_session):
    from edgefinder.data.pit_fundamentals import PITFundamentals

    db_session.add(FundamentalsSnapshot(
        symbol="AAA", as_of=date(2024, 3, 1),
        data={"symbol": "AAA", "earnings_per_share": 2.0}))
    db_session.commit()
    pit = PITFundamentals(db_session)
    assert pit.preload(["AAA", "BBB"]) == 1
    pit._session = None   # simulate validate.py closing the session
    assert pit.asof("AAA", date(2024, 6, 1)).earnings_per_share == 2.0
    assert pit.asof("AAA", date(2024, 1, 1)) is None   # before coverage
    assert pit.asof("BBB", date(2024, 6, 1)) is None   # no data, no crash
