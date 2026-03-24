"""
Tests for Lynch and Burry Strategy Plugins
============================================
Tests registration, signal generation, watchlist filtering,
sentiment gate integration, and arena integration.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from modules.strategies.base import StrategyRegistry, Signal, TradeNotification, MarketRegime


# ── HELPERS ──────────────────────────────────────────────────

def make_ohlcv(
    n: int = 250,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.02,
    base_volume: float = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(trend, volatility, n)
    close = start_price * np.cumprod(1 + returns)

    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.normal(base_volume, base_volume * 0.3, n).clip(min=1000)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


def make_scored_stock(
    ticker: str,
    lynch_score: float = 70.0,
    burry_score: float = 65.0,
    composite_score: float = 67.5,
    lynch_category: str = "fast_grower",
    fcf_yield: float = 0.08,
    price_to_tangible_book: float = 1.5,
) -> dict:
    return {
        "ticker": ticker,
        "lynch_score": lynch_score,
        "burry_score": burry_score,
        "composite_score": composite_score,
        "lynch_category": lynch_category,
        "fcf_yield": fcf_yield,
        "price_to_tangible_book": price_to_tangible_book,
    }


@pytest.fixture(autouse=True)
def clean_registry():
    StrategyRegistry.clear()
    # Re-register after clear (decorators only fire on first import)
    from modules.strategies.lynch import LynchStrategy
    from modules.strategies.burry import BurryStrategy
    if "lynch" not in StrategyRegistry.list_strategies():
        StrategyRegistry.register("lynch")(LynchStrategy)
    if "burry" not in StrategyRegistry.list_strategies():
        StrategyRegistry.register("burry")(BurryStrategy)
    yield
    StrategyRegistry.clear()


# ── REGISTRATION TESTS ───────────────────────────────────────

class TestRegistration:

    def test_lynch_registered(self):
        assert "lynch" in StrategyRegistry.list_strategies()

    def test_burry_registered(self):
        assert "burry" in StrategyRegistry.list_strategies()

    def test_both_registered(self):
        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) >= 2
        assert "lynch" in strategies
        assert "burry" in strategies

    def test_lynch_instantiation(self):
        cls = StrategyRegistry.get("lynch")
        instance = cls()
        instance.init()
        assert instance.name == "lynch"
        assert instance.version == "1.0.0"

    def test_burry_instantiation(self):
        cls = StrategyRegistry.get("burry")
        instance = cls()
        instance.init()
        assert instance.name == "burry"
        assert instance.version == "1.0.0"


# ── LYNCH STRATEGY TESTS ────────────────────────────────────

class TestLynchStrategy:

    def _make_strategy(self):
        cls = StrategyRegistry.get("lynch")
        s = cls()
        s.init()
        s._use_sentiment = False  # Disable for unit tests
        return s

    def test_empty_watchlist_by_default(self):
        s = self._make_strategy()
        assert s.get_watchlist() == []

    def test_set_watchlist_filters_by_lynch_score(self):
        s = self._make_strategy()
        stocks = [
            make_scored_stock("AAPL", lynch_score=80),
            make_scored_stock("MSFT", lynch_score=30),  # Below threshold
            make_scored_stock("GOOGL", lynch_score=60),
        ]
        s.set_watchlist(stocks)
        wl = s.get_watchlist()
        assert "AAPL" in wl
        assert "GOOGL" in wl
        assert "MSFT" not in wl

    def test_scores_stored(self):
        s = self._make_strategy()
        stocks = [make_scored_stock("AAPL", lynch_score=80, lynch_category="fast_grower")]
        s.set_watchlist(stocks)
        assert s._scores["AAPL"]["lynch_category"] == "fast_grower"

    def test_generate_signals_empty_bars(self):
        s = self._make_strategy()
        signals = s.generate_signals({})
        assert signals == []

    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    def test_generate_signals_with_mock_technical(self, mock_compute, mock_detect):
        """Test signal generation with mocked technical analysis."""
        s = self._make_strategy()

        # Mock indicator computation
        mock_snapshot = MagicMock()
        mock_compute.return_value = mock_snapshot

        # Mock a buy signal
        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 75.0
        mock_signal.price = 150.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = {"ema_crossover_day": True}
        mock_signal.reason = "BUY: EMA crossover"
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(n=250, start_price=150.0)}
        signals = s.generate_signals(bars)

        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"
        assert signals[0].action == "BUY"
        assert signals[0].confidence == 75.0
        assert signals[0].stop_loss < signals[0].entry_price
        assert signals[0].target > signals[0].entry_price

    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    def test_sell_signals_skipped(self, mock_compute, mock_detect):
        s = self._make_strategy()
        mock_compute.return_value = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal_type = "SELL"
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv()}
        signals = s.generate_signals(bars)
        assert len(signals) == 0

    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    def test_low_confidence_skipped(self, mock_compute, mock_detect):
        s = self._make_strategy()
        mock_compute.return_value = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 20.0  # Below threshold
        mock_signal.price = 100.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv()}
        signals = s.generate_signals(bars)
        assert len(signals) == 0

    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    def test_metadata_includes_lynch_info(self, mock_compute, mock_detect):
        s = self._make_strategy()
        s.set_watchlist([make_scored_stock("AAPL", lynch_score=80, lynch_category="fast_grower")])

        mock_compute.return_value = MagicMock()
        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 75.0
        mock_signal.price = 150.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(start_price=150.0)}
        signals = s.generate_signals(bars)

        assert len(signals) == 1
        assert signals[0].metadata["strategy"] == "lynch"
        assert signals[0].metadata["lynch_score"] == 80
        assert signals[0].metadata["lynch_category"] == "fast_grower"

    def test_on_trade_executed(self):
        s = self._make_strategy()
        n = TradeNotification(
            trade_id="T1", ticker="AAPL", action="BUY",
            entry_price=150.0, shares=10,
        )
        s.on_trade_executed(n)
        assert len(s._trades_log) == 1

    def test_optional_hooks(self):
        s = self._make_strategy()
        # Should not raise
        s.on_market_regime_change(MarketRegime(trend="bull"))
        s.on_strategy_pause("test")

    def test_stop_loss_is_5_percent(self):
        """Lynch uses 5% stop loss."""
        s = self._make_strategy()
        price = 100.0
        expected_stop = 95.0
        # Verify from signal generation
        with patch("modules.strategies.lynch.compute_indicators") as mc, \
             patch("modules.strategies.lynch.detect_signals") as md:
            mc.return_value = MagicMock()
            sig = MagicMock()
            sig.signal_type = "BUY"
            sig.confidence = 80.0
            sig.price = price
            sig.trade_type = "DAY"
            sig.indicators = {}
            sig.reason = ""
            md.return_value = [sig]

            signals = s.generate_signals({"AAPL": make_ohlcv(start_price=price)})
            assert len(signals) == 1
            assert signals[0].stop_loss == expected_stop


# ── BURRY STRATEGY TESTS ────────────────────────────────────

class TestBurryStrategy:

    def _make_strategy(self):
        cls = StrategyRegistry.get("burry")
        s = cls()
        s.init()
        s._use_sentiment = False
        return s

    def test_empty_watchlist_by_default(self):
        s = self._make_strategy()
        assert s.get_watchlist() == []

    def test_set_watchlist_filters_by_burry_score(self):
        s = self._make_strategy()
        stocks = [
            make_scored_stock("AAPL", burry_score=70),
            make_scored_stock("MSFT", burry_score=20),  # Below threshold
            make_scored_stock("GOOGL", burry_score=55),
        ]
        s.set_watchlist(stocks)
        wl = s.get_watchlist()
        assert "AAPL" in wl
        assert "GOOGL" in wl
        assert "MSFT" not in wl

    @patch("modules.strategies.burry.detect_signals")
    @patch("modules.strategies.burry.compute_indicators")
    def test_generate_signals_basic(self, mock_compute, mock_detect):
        s = self._make_strategy()
        mock_compute.return_value = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 70.0
        mock_signal.price = 50.0
        mock_signal.trade_type = "SWING"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(start_price=50.0)}
        signals = s.generate_signals(bars)

        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"
        assert signals[0].action == "BUY"
        assert signals[0].metadata["strategy"] == "burry"

    @patch("modules.strategies.burry.detect_signals")
    @patch("modules.strategies.burry.compute_indicators")
    def test_rsi_oversold_boost(self, mock_compute, mock_detect):
        """Burry boosts confidence when RSI is oversold."""
        s = self._make_strategy()
        mock_compute.return_value = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 65.0
        mock_signal.price = 50.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = [{"name": "rsi_oversold", "rsi": 25}]
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(start_price=50.0)}
        signals = s.generate_signals(bars)

        assert len(signals) == 1
        # Confidence should be boosted by 10 (RSI boost)
        assert signals[0].confidence == 75.0
        assert signals[0].metadata["rsi_boost_applied"] is True

    @patch("modules.strategies.burry.detect_signals")
    @patch("modules.strategies.burry.compute_indicators")
    def test_no_rsi_boost_without_oversold(self, mock_compute, mock_detect):
        s = self._make_strategy()
        mock_compute.return_value = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 65.0
        mock_signal.price = 50.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = [{"name": "ema_crossover_day"}]
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(start_price=50.0)}
        signals = s.generate_signals(bars)

        assert len(signals) == 1
        assert signals[0].confidence == 65.0  # No boost

    @patch("modules.strategies.burry.detect_signals")
    @patch("modules.strategies.burry.compute_indicators")
    def test_wider_stop_loss(self, mock_compute, mock_detect):
        """Burry uses 7% stop loss (wider than Lynch's 5%)."""
        s = self._make_strategy()
        mock_compute.return_value = MagicMock()

        price = 100.0
        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 80.0
        mock_signal.price = price
        mock_signal.trade_type = "SWING"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(start_price=price)}
        signals = s.generate_signals(bars)

        assert len(signals) == 1
        assert signals[0].stop_loss == 93.0  # 100 * 0.93

    @patch("modules.strategies.burry.detect_signals")
    @patch("modules.strategies.burry.compute_indicators")
    def test_metadata_includes_burry_info(self, mock_compute, mock_detect):
        s = self._make_strategy()
        s.set_watchlist([make_scored_stock(
            "AAPL", burry_score=75, fcf_yield=0.09, price_to_tangible_book=0.8
        )])
        mock_compute.return_value = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 75.0
        mock_signal.price = 50.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        bars = {"AAPL": make_ohlcv(start_price=50.0)}
        signals = s.generate_signals(bars)

        assert signals[0].metadata["burry_score"] == 75
        assert signals[0].metadata["fcf_yield"] == 0.09

    def test_on_trade_executed(self):
        s = self._make_strategy()
        n = TradeNotification(
            trade_id="T1", ticker="AAPL", action="BUY",
            entry_price=50.0, shares=20,
        )
        s.on_trade_executed(n)
        assert len(s._trades_log) == 1

    def test_regime_change_logging(self):
        s = self._make_strategy()
        # Should not raise
        s.on_market_regime_change(MarketRegime(trend="bear"))
        s.on_market_regime_change(MarketRegime(trend="bull"))
        s.on_strategy_pause("drawdown")


# ── SENTIMENT GATE INTEGRATION ──────────────────────────────

class TestSentimentIntegration:

    @patch("modules.sentiment.gate_trade")
    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    def test_lynch_sentiment_blocks_trade(self, mock_compute, mock_detect, mock_gate):
        """Lynch respects sentiment BLOCK action."""
        cls = StrategyRegistry.get("lynch")
        s = cls()
        s.init()
        s._use_sentiment = True

        mock_compute.return_value = MagicMock()
        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 80.0
        mock_signal.price = 150.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        mock_gate.return_value = ("BLOCK", 0.0, MagicMock())

        bars = {"AAPL": make_ohlcv(start_price=150.0)}
        signals = s.generate_signals(bars)
        assert len(signals) == 0

    @patch("modules.sentiment.gate_trade")
    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    def test_lynch_sentiment_adjusts_confidence(self, mock_compute, mock_detect, mock_gate):
        cls = StrategyRegistry.get("lynch")
        s = cls()
        s.init()
        s._use_sentiment = True

        mock_compute.return_value = MagicMock()
        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.confidence = 70.0
        mock_signal.price = 150.0
        mock_signal.trade_type = "DAY"
        mock_signal.indicators = {}
        mock_signal.reason = ""
        mock_detect.return_value = [mock_signal]

        # Sentiment boosts confidence
        mock_gate.return_value = ("CONFIDENCE_PLUS_10", 80.0, MagicMock())

        bars = {"AAPL": make_ohlcv(start_price=150.0)}
        signals = s.generate_signals(bars)
        assert len(signals) == 1
        assert signals[0].confidence == 80.0


# ── ARENA INTEGRATION ────────────────────────────────────────

class TestArenaIntegration:

    @patch("modules.strategies.lynch.detect_signals")
    @patch("modules.strategies.lynch.compute_indicators")
    @patch("modules.strategies.burry.detect_signals")
    @patch("modules.strategies.burry.compute_indicators")
    def test_both_strategies_in_arena(
        self, burry_compute, burry_detect, lynch_compute, lynch_detect
    ):
        """Both Lynch and Burry can run simultaneously in the arena."""
        from modules.arena.engine import ArenaEngine

        engine = ArenaEngine(starting_capital=10000)

        lynch_cls = StrategyRegistry.get("lynch")
        burry_cls = StrategyRegistry.get("burry")

        lynch = lynch_cls()
        lynch.init()
        lynch._use_sentiment = False

        burry = burry_cls()
        burry.init()
        burry._use_sentiment = False

        engine.add_strategy("lynch", lynch)
        engine.add_strategy("burry", burry)

        # Both generate signals
        for mock_compute, mock_detect in [(lynch_compute, lynch_detect), (burry_compute, burry_detect)]:
            mock_compute.return_value = MagicMock()
            sig = MagicMock()
            sig.signal_type = "BUY"
            sig.confidence = 75.0
            sig.price = 100.0
            sig.trade_type = "DAY"
            sig.indicators = {}
            sig.reason = ""
            mock_detect.return_value = [sig]

        bars = {"AAPL": make_ohlcv(start_price=100.0)}
        executed = engine.run_signal_check(bars)

        # Both strategies should have opened a position
        assert engine.accounts["lynch"].open_position_count == 1
        assert engine.accounts["burry"].open_position_count == 1

        # Overlap should show AAPL held by both
        overlap = engine.get_overlap_report()
        assert "AAPL" in overlap["overlapping_positions"]

    def test_create_all_includes_both(self):
        """StrategyRegistry.create_all() produces both Lynch and Burry."""
        instances = StrategyRegistry.create_all()
        assert "lynch" in instances
        assert "burry" in instances
        assert instances["lynch"].name == "lynch"
        assert instances["burry"].name == "burry"
