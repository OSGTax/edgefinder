"""Tests for edgefinder/strategies/."""

import pytest
import numpy as np
import pandas as pd

from edgefinder.core.models import TickerFundamentals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is reset between tests, then reload strategies."""
    StrategyRegistry.clear()
    # Re-import to trigger @register decorators
    import importlib
    from edgefinder.strategies import alpha, bravo, charlie
    importlib.reload(alpha)
    importlib.reload(bravo)
    importlib.reload(charlie)
    yield
    StrategyRegistry.clear()


class TestStrategyRegistry:
    def test_register_and_get(self):
        assert StrategyRegistry.get("alpha") is not None

    def test_get_all(self):
        all_strats = StrategyRegistry.get_all()
        assert "alpha" in all_strats
        assert "bravo" in all_strats
        assert "charlie" in all_strats

    def test_get_instances(self):
        instances = StrategyRegistry.get_instances()
        assert len(instances) == 3
        names = {s.name for s in instances}
        assert names == {"alpha", "bravo", "charlie"}

    def test_list_names(self):
        names = StrategyRegistry.list_names()
        assert "alpha" in names

    def test_get_nonexistent(self):
        assert StrategyRegistry.get("nonexistent") is None

    def test_clear(self):
        StrategyRegistry.clear()
        assert len(StrategyRegistry.get_all()) == 0


class TestAlphaStrategy:
    def test_qualifies_good_stock(self):
        fund = TickerFundamentals(
            symbol="AAPL",
            composite_score=75.0,
            earnings_growth=0.20,
            peg_ratio=1.2,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is True

    def test_rejects_low_composite(self):
        fund = TickerFundamentals(
            symbol="BAD",
            composite_score=40.0,
            earnings_growth=0.20,
            peg_ratio=1.2,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_negative_growth(self):
        fund = TickerFundamentals(
            symbol="BAD",
            composite_score=75.0,
            earnings_growth=-0.10,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_high_peg(self):
        fund = TickerFundamentals(
            symbol="BAD",
            composite_score=75.0,
            earnings_growth=0.20,
            peg_ratio=3.0,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is False

    def test_properties(self):
        s = StrategyRegistry.get("alpha")()
        assert s.name == "alpha"
        assert s.version == "2.0"
        assert "ema_crossover_bullish" in s.preferred_signals


class TestBravoStrategy:
    def test_qualifies_value_stock(self):
        fund = TickerFundamentals(
            symbol="VALUE",
            burry_score=60.0,
            current_ratio=1.5,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is True

    def test_rejects_low_burry(self):
        fund = TickerFundamentals(
            symbol="GROWTH",
            burry_score=30.0,
            current_ratio=1.5,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_low_current_ratio(self):
        fund = TickerFundamentals(
            symbol="RISKY",
            burry_score=60.0,
            current_ratio=0.8,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is False


class TestCharlieStrategy:
    def test_qualifies_deep_value(self):
        fund = TickerFundamentals(
            symbol="DEEP",
            burry_score=80.0,
            short_interest=0.20,
            fcf_yield=0.08,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is True

    def test_rejects_low_short_interest(self):
        fund = TickerFundamentals(
            symbol="NOSQUEEZE",
            burry_score=80.0,
            short_interest=0.05,
            fcf_yield=0.08,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_low_fcf(self):
        fund = TickerFundamentals(
            symbol="NOCASH",
            burry_score=80.0,
            short_interest=0.20,
            fcf_yield=0.02,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is False


class TestOptionalMethods:
    def test_defaults_dont_raise(self):
        s = StrategyRegistry.get("alpha")()
        assert s.get_watchlist() == []
        s.on_market_regime_change("bull")
        s.on_strategy_pause("test")
        assert s.get_state() == {}
        assert s.apply_suggestion({}) is False
