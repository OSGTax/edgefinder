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
            earnings_growth=0.20,
            revenue_growth=0.15,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is True

    def test_rejects_negative_earnings(self):
        fund = TickerFundamentals(
            symbol="BAD",
            earnings_growth=-0.10,
            revenue_growth=0.15,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_negative_revenue(self):
        fund = TickerFundamentals(
            symbol="BAD",
            earnings_growth=0.20,
            revenue_growth=-0.05,
        )
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_missing_growth(self):
        fund = TickerFundamentals(symbol="BAD")
        strategy = StrategyRegistry.get("alpha")()
        assert strategy.qualifies_stock(fund) is False

    def test_properties(self):
        s = StrategyRegistry.get("alpha")()
        assert s.name == "alpha"
        assert s.version == "4.0"
        assert "ema_crossover_bullish" in s.preferred_signals

    def test_exit_signals(self):
        s = StrategyRegistry.get("alpha")()
        assert "ema_crossover_bearish" in s.exit_signals
        assert "volume_spike_bearish" in s.exit_signals


class TestBravoStrategy:
    def test_qualifies_value_stock(self):
        fund = TickerFundamentals(
            symbol="VALUE",
            current_ratio=1.5,
            debt_to_equity=1.0,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is True

    def test_rejects_low_current_ratio(self):
        fund = TickerFundamentals(
            symbol="RISKY",
            current_ratio=0.8,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_high_debt(self):
        fund = TickerFundamentals(
            symbol="LEVERAGED",
            current_ratio=1.5,
            debt_to_equity=3.0,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is False

    def test_qualifies_without_debt_data(self):
        fund = TickerFundamentals(
            symbol="NODBT",
            current_ratio=1.5,
        )
        strategy = StrategyRegistry.get("bravo")()
        assert strategy.qualifies_stock(fund) is True

    def test_exit_signals(self):
        s = StrategyRegistry.get("bravo")()
        assert "rsi_overbought" in s.exit_signals


class TestCharlieStrategy:
    def test_qualifies_deep_value(self):
        fund = TickerFundamentals(
            symbol="DEEP",
            fcf_yield=0.06,
            debt_to_equity=1.5,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is True

    def test_qualifies_without_debt_data(self):
        """Charlie should qualify stocks even when debt_to_equity is unavailable."""
        fund = TickerFundamentals(
            symbol="NODBT",
            fcf_yield=0.05,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is True

    def test_rejects_low_fcf(self):
        fund = TickerFundamentals(
            symbol="NOCASH",
            fcf_yield=0.01,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is False

    def test_rejects_high_debt(self):
        fund = TickerFundamentals(
            symbol="LEVERAGED",
            fcf_yield=0.06,
            debt_to_equity=4.0,
        )
        strategy = StrategyRegistry.get("charlie")()
        assert strategy.qualifies_stock(fund) is False

    def test_exit_signals(self):
        s = StrategyRegistry.get("charlie")()
        assert "macd_bearish_cross" in s.exit_signals


class TestOptionalMethods:
    def test_defaults_dont_raise(self):
        s = StrategyRegistry.get("alpha")()
        assert s.get_watchlist() == []
        s.on_market_regime_change("bull")
        s.on_strategy_pause("test")
        assert s.get_state() == {}
        assert s.apply_suggestion({}) is False
