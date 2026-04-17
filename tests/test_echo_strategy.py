"""Tests for the Echo meta-strategy."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from edgefinder.analytics.conditional_stats import (
    ConditionStats,
    StrategyProfile,
)
from edgefinder.analytics.regime import MarketCondition
from edgefinder.core.models import MarketSnapshot, SignalAction, TickerFundamentals
from edgefinder.strategies.base import StrategyRegistry, TradeNotification
from edgefinder.strategies.echo import EchoStrategy


@pytest.fixture
def echo():
    """Fresh EchoStrategy instance (not from registry to avoid side effects)."""
    return EchoStrategy()


@pytest.fixture
def good_fundamentals():
    return TickerFundamentals(
        symbol="AAPL",
        market_cap=2_000_000_000_000,
        earnings_growth=0.15,
        revenue_growth=0.10,
    )


@pytest.fixture
def bad_fundamentals():
    return TickerFundamentals(
        symbol="PENNY",
        market_cap=50_000_000,  # below $300M threshold
    )


class TestEchoProperties:
    def test_name(self, echo):
        assert echo.name == "echo"

    def test_version(self, echo):
        assert echo.version == "1.0"

    def test_preferred_signals(self, echo):
        assert "ema_crossover_bullish" in echo.preferred_signals
        assert "macd_bullish_cross" in echo.preferred_signals

    def test_exit_signals(self, echo):
        assert "ema_crossover_bearish" in echo.exit_signals

    def test_risk_config(self, echo):
        assert echo.risk_config["max_risk_pct"] == 0.02
        assert echo.risk_config["max_concentration_pct"] == 0.20

    def test_registered_in_registry(self):
        assert StrategyRegistry.get("echo") is not None


class TestEchoQualification:
    def test_qualifies_large_cap_with_data(self, echo, good_fundamentals):
        assert echo.qualifies_stock(good_fundamentals) is True

    def test_rejects_small_cap(self, echo, bad_fundamentals):
        assert echo.qualifies_stock(bad_fundamentals) is False

    def test_rejects_no_data(self, echo):
        f = TickerFundamentals(symbol="EMPTY", market_cap=1_000_000_000)
        assert echo.qualifies_stock(f) is False

    def test_qualifies_with_fcf_only(self, echo):
        f = TickerFundamentals(
            symbol="FCF",
            market_cap=500_000_000,
            fcf_yield=0.05,
        )
        assert echo.qualifies_stock(f) is True


class TestEchoRegimeTracking:
    def test_initial_regime_is_sideways_calm(self, echo):
        assert echo._current_regime == MarketCondition.SIDEWAYS_CALM

    def test_on_market_snapshot_updates_regime(self, echo):
        snapshot = MarketSnapshot(
            vix_level=25.0,
            spy_change_pct=0.8,
        )
        echo.on_market_snapshot(snapshot)
        assert echo._current_regime == MarketCondition.BULL_VOLATILE

    def test_regime_change_updates_active_strategy(self, echo):
        # Set up profiles where alpha wins in bull_calm
        alpha_stats = ConditionStats(
            strategy_name="alpha", condition="bull_calm",
            total_trades=10, wins=8, losses=2,
            win_rate=0.8, expectancy=60.0,
        )
        bravo_stats = ConditionStats(
            strategy_name="bravo", condition="bull_calm",
            total_trades=10, wins=3, losses=7,
            win_rate=0.3, expectancy=-20.0,
        )
        echo._profiles = {
            "alpha": StrategyProfile(
                strategy_name="alpha", by_regime={"bull_calm": alpha_stats},
            ),
            "bravo": StrategyProfile(
                strategy_name="bravo", by_regime={"bull_calm": bravo_stats},
            ),
        }

        snapshot = MarketSnapshot(vix_level=15.0, spy_change_pct=0.5)
        echo.on_market_snapshot(snapshot)
        assert echo._active_strategy == "alpha"


class TestEchoSignalGeneration:
    def test_no_profiles_uses_all_preferred(self, echo):
        """With no history, echo should accept all its preferred signals."""
        patterns = echo._get_allowed_patterns()
        assert patterns == set(echo.preferred_signals)

    def test_with_active_strategy_uses_that_strategys_signals(self, echo):
        """When a clear winner is selected, use its signal preferences."""
        echo._active_strategy = "alpha"
        patterns = echo._get_allowed_patterns()
        # Alpha's preferred signals
        assert "ema_crossover_bullish" in patterns
        assert "macd_bullish_cross" in patterns

    def test_generate_signals_returns_buy_only(self, echo):
        """Echo should only emit BUY signals."""
        # Create mock bars with enough data
        dates = pd.date_range("2024-01-01", periods=200, freq="5min")
        bars = pd.DataFrame({
            "open": [100.0] * 200,
            "high": [101.0] * 200,
            "low": [99.0] * 200,
            "close": [100.5] * 200,
            "volume": [1000000] * 200,
        }, index=dates)

        signals = echo.generate_signals("TEST", bars)
        for sig in signals:
            assert sig.action == SignalAction.BUY
            assert sig.strategy_name == "echo"


class TestEchoGetState:
    def test_get_state_returns_dict(self, echo):
        state = echo.get_state()
        assert "current_regime" in state
        assert "active_strategy" in state
        assert "vix_level" in state
        assert "profiles_loaded" in state

    def test_get_state_with_profiles(self, echo):
        alpha_stats = ConditionStats(
            strategy_name="alpha", condition="bull_calm",
            total_trades=10, wins=8, losses=2,
            win_rate=0.8, expectancy=60.0,
        )
        echo._profiles = {
            "alpha": StrategyProfile(
                strategy_name="alpha", by_regime={"bull_calm": alpha_stats},
            ),
        }
        echo._current_regime = MarketCondition.BULL_CALM

        state = echo.get_state()
        assert state["profiles_loaded"] == 1
        assert len(state["strategy_scores"]) > 0


class TestEchoLearningLoop:
    def test_refresh_profiles_without_session_is_noop(self, echo):
        echo._refresh_profiles()
        assert echo._profiles == {}

    def test_on_trade_closed_triggers_refresh(self, echo):
        echo._db_session = MagicMock()
        with patch.object(echo, "_refresh_profiles") as mock_refresh:
            notification = MagicMock(spec=TradeNotification)
            notification.event = "closed"
            echo.on_trade_executed(notification)
            mock_refresh.assert_called_once()

    def test_on_trade_opened_does_not_refresh(self, echo):
        echo._db_session = MagicMock()
        with patch.object(echo, "_refresh_profiles") as mock_refresh:
            notification = MagicMock(spec=TradeNotification)
            notification.event = "opened"
            echo.on_trade_executed(notification)
            mock_refresh.assert_not_called()
