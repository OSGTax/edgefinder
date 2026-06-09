"""Tests for the portfolio-interface research strategies (engine v2)."""

import pandas as pd

from edgefinder.data.market_data import IndicatorSnapshot
from edgefinder.engine.strategies import DualMomentum, TrendTimer
from edgefinder.engine.strategy import AssetView, RebalanceContext

_EMPTY = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _asset(sym: str, price: float, ema_200: float | None) -> AssetView:
    return AssetView(symbol=sym, price=price,
                     indicators=IndicatorSnapshot(close=price, ema_200=ema_200),
                     history=_EMPTY)


def _ctx(*assets: AssetView) -> RebalanceContext:
    from datetime import date
    return RebalanceContext(date=date(2024, 1, 2),
                            assets={a.symbol: a for a in assets})


class TestTrendTimer:
    def test_holds_index_above_ema(self):
        assert TrendTimer("SPY").rebalance(
            _ctx(_asset("SPY", 500.0, 450.0))) == {"SPY": 1.0}

    def test_cash_below_ema_or_unwarmed(self):
        t = TrendTimer("SPY")
        assert t.rebalance(_ctx(_asset("SPY", 400.0, 450.0))) == {}
        assert t.rebalance(_ctx(_asset("SPY", 400.0, None))) == {}
        assert t.rebalance(_ctx()) == {}


class TestDualMomentum:
    def test_ranks_by_momentum_and_caps_at_top_k(self):
        dm = DualMomentum(symbols=("AAA", "BBB", "CCC", "DDD"), top_k=2)
        weights = dm.rebalance(_ctx(
            _asset("AAA", 110.0, 100.0),   # +10%
            _asset("BBB", 130.0, 100.0),   # +30%  <- top
            _asset("CCC", 120.0, 100.0),   # +20%  <- second
            _asset("DDD", 105.0, 100.0)))  # +5%
        assert weights == {"BBB": 0.5, "CCC": 0.5}

    def test_absolute_filter_leaves_slots_in_cash(self):
        dm = DualMomentum(symbols=("AAA", "BBB", "CCC"), top_k=3)
        weights = dm.rebalance(_ctx(
            _asset("AAA", 110.0, 100.0),   # eligible
            _asset("BBB", 90.0, 100.0),    # below its 200-EMA — ineligible
            _asset("CCC", 100.0, None)))   # unwarmed — ineligible
        # one eligible name at fixed 1/top_k; the other two slots stay cash
        assert weights == {"AAA": 1.0 / 3}

    def test_all_cash_in_a_broad_downtrend(self):
        dm = DualMomentum(symbols=("AAA", "BBB"), top_k=2)
        assert dm.rebalance(_ctx(
            _asset("AAA", 90.0, 100.0), _asset("BBB", 80.0, 100.0))) == {}
