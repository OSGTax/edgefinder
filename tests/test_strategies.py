"""
Tests for EdgeFinder Strategy Plugin System
=============================================
Tests BaseStrategy ABC, StrategyRegistry, and Signal dataclass.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock

from modules.strategies.base import (
    BaseStrategy,
    StrategyRegistry,
    Signal,
    TradeNotification,
    MarketRegime,
)


# ── FIXTURES ─────────────────────────────────────────────────

class DummyStrategy(BaseStrategy):
    """Minimal concrete strategy for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def init(self) -> None:
        self.initialized = True
        self.trades_received = []

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for ticker, df in bars.items():
            if not df.empty:
                close = float(df.iloc[-1]["close"])
                signals.append(Signal(
                    ticker=ticker,
                    action="BUY",
                    entry_price=close,
                    stop_loss=close * 0.95,
                    target=close * 1.10,
                    confidence=75.0,
                ))
        return signals

    def on_trade_executed(self, notification: TradeNotification) -> None:
        self.trades_received.append(notification)


class BadStrategy:
    """Not a BaseStrategy subclass — should fail registration."""
    pass


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before each test."""
    StrategyRegistry.clear()
    yield
    StrategyRegistry.clear()


@pytest.fixture
def sample_bars():
    """Sample OHLCV data for testing."""
    dates = pd.date_range("2026-03-01", periods=5, freq="B")
    return {
        "AAPL": pd.DataFrame({
            "open": [150, 151, 152, 153, 154],
            "high": [152, 153, 154, 155, 156],
            "low": [149, 150, 151, 152, 153],
            "close": [151, 152, 153, 154, 155],
            "volume": [1000000] * 5,
        }, index=dates),
        "MSFT": pd.DataFrame({
            "open": [300, 301, 302, 303, 304],
            "high": [302, 303, 304, 305, 306],
            "low": [299, 300, 301, 302, 303],
            "close": [301, 302, 303, 304, 305],
            "volume": [800000] * 5,
        }, index=dates),
    }


# ── SIGNAL TESTS ─────────────────────────────────────────────

class TestSignal:

    def test_signal_creation(self):
        s = Signal(
            ticker="AAPL",
            action="BUY",
            entry_price=150.0,
            stop_loss=145.0,
            target=165.0,
        )
        assert s.ticker == "AAPL"
        assert s.action == "BUY"
        assert s.trade_type == "DAY"  # default

    def test_risk_per_share(self):
        s = Signal(ticker="AAPL", action="BUY", entry_price=150.0,
                   stop_loss=145.0, target=165.0)
        assert s.risk_per_share == 5.0

    def test_reward_to_risk(self):
        s = Signal(ticker="AAPL", action="BUY", entry_price=150.0,
                   stop_loss=145.0, target=165.0)
        assert s.reward_to_risk == 3.0  # 15/5

    def test_zero_risk(self):
        s = Signal(ticker="AAPL", action="BUY", entry_price=150.0,
                   stop_loss=150.0, target=165.0)
        assert s.reward_to_risk == 0.0

    def test_signal_metadata(self):
        s = Signal(ticker="AAPL", action="BUY", entry_price=150.0,
                   stop_loss=145.0, target=160.0,
                   metadata={"indicator": "RSI"})
        assert s.metadata["indicator"] == "RSI"

    def test_signal_timestamp(self):
        s = Signal(ticker="AAPL", action="BUY", entry_price=150.0,
                   stop_loss=145.0, target=160.0)
        assert s.timestamp.tzinfo is not None  # timezone-aware


# ── REGISTRY TESTS ───────────────────────────────────────────

class TestStrategyRegistry:

    def test_register_strategy(self):
        StrategyRegistry.register("test_dummy")(DummyStrategy)
        assert "test_dummy" in StrategyRegistry.list_strategies()

    def test_get_strategy(self):
        StrategyRegistry.register("test_dummy")(DummyStrategy)
        cls = StrategyRegistry.get("test_dummy")
        assert cls is DummyStrategy

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            StrategyRegistry.get("nonexistent")

    def test_duplicate_registration_raises(self):
        StrategyRegistry.register("test_dummy")(DummyStrategy)
        with pytest.raises(ValueError, match="already registered"):
            StrategyRegistry.register("test_dummy")(DummyStrategy)

    def test_non_basestrategy_raises(self):
        with pytest.raises(TypeError, match="must inherit"):
            StrategyRegistry.register("bad")(BadStrategy)

    def test_list_strategies(self):
        StrategyRegistry.register("one")(DummyStrategy)

        class AnotherDummy(DummyStrategy):
            @property
            def name(self):
                return "another"

        StrategyRegistry.register("two")(AnotherDummy)
        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) == 2
        assert "one" in strategies
        assert "two" in strategies

    def test_clear_registry(self):
        StrategyRegistry.register("test_dummy")(DummyStrategy)
        StrategyRegistry.clear()
        assert len(StrategyRegistry.list_strategies()) == 0

    def test_create_all(self):
        StrategyRegistry.register("test_dummy")(DummyStrategy)
        instances = StrategyRegistry.create_all()
        assert "test_dummy" in instances
        assert instances["test_dummy"].initialized is True

    def test_create_all_handles_error(self):
        class FailInit(DummyStrategy):
            def init(self):
                raise RuntimeError("init failed")

        StrategyRegistry.register("fail")(FailInit)
        instances = StrategyRegistry.create_all()
        assert "fail" not in instances  # Should be skipped, not crash


# ── BASE STRATEGY TESTS ─────────────────────────────────────

class TestBaseStrategy:

    def test_init_called(self):
        s = DummyStrategy()
        s.init()
        assert s.initialized is True

    def test_generate_signals(self, sample_bars):
        s = DummyStrategy()
        s.init()
        signals = s.generate_signals(sample_bars)
        assert len(signals) == 2
        tickers = {sig.ticker for sig in signals}
        assert tickers == {"AAPL", "MSFT"}

    def test_generate_signals_empty_bars(self):
        s = DummyStrategy()
        s.init()
        signals = s.generate_signals({})
        assert signals == []

    def test_on_trade_executed(self):
        s = DummyStrategy()
        s.init()
        notification = TradeNotification(
            trade_id="T1", ticker="AAPL", action="BUY",
            entry_price=150.0, shares=10,
        )
        s.on_trade_executed(notification)
        assert len(s.trades_received) == 1
        assert s.trades_received[0].ticker == "AAPL"

    def test_optional_hooks_dont_crash(self):
        s = DummyStrategy()
        s.init()
        # These are optional — should not raise
        s.on_market_regime_change(MarketRegime(trend="bull"))
        s.on_strategy_pause("test pause")

    def test_get_watchlist_default_empty(self):
        s = DummyStrategy()
        assert s.get_watchlist() == []

    def test_name_and_version(self):
        s = DummyStrategy()
        assert s.name == "dummy"
        assert s.version == "1.0.0"

    def test_signal_confidence(self, sample_bars):
        s = DummyStrategy()
        s.init()
        signals = s.generate_signals(sample_bars)
        for sig in signals:
            assert sig.confidence == 75.0


# ── MARKET REGIME TESTS ─────────────────────────────────────

class TestMarketRegime:

    def test_defaults(self):
        r = MarketRegime()
        assert r.trend == "unknown"
        assert r.volatility == "normal"
        assert r.vix_level is None

    def test_custom_values(self):
        r = MarketRegime(trend="bull", volatility="low", vix_level=15.5)
        assert r.trend == "bull"
        assert r.volatility == "low"
        assert r.vix_level == 15.5

    def test_timestamp_is_aware(self):
        r = MarketRegime()
        assert r.timestamp.tzinfo is not None


# ── TRADE NOTIFICATION TESTS ────────────────────────────────

class TestTradeNotification:

    def test_open_notification(self):
        n = TradeNotification(
            trade_id="T1", ticker="AAPL", action="BUY",
            entry_price=150.0, shares=10,
        )
        assert n.exit_price is None
        assert n.pnl_dollars is None

    def test_close_notification(self):
        n = TradeNotification(
            trade_id="T1", ticker="AAPL", action="SELL",
            entry_price=150.0, exit_price=160.0,
            shares=10, pnl_dollars=100.0,
            pnl_percent=6.67, r_multiple=2.0,
            exit_reason="TARGET_HIT",
        )
        assert n.pnl_dollars == 100.0
        assert n.exit_reason == "TARGET_HIT"
