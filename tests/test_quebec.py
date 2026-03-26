"""
Tests for Quebec Strategy — Momentum Swing
============================================
Tests registration, qualification (pure technical), signal generation
with 3+ confirmation requirement, stop/target parameters, confidence
boosts, and market regime adaptation.
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
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


def _mock_buy_signal(indicators: dict, price: float = 100.0, confidence: float = 75.0):
    """Create a mock trade signal with given indicators."""
    sig = MagicMock()
    sig.signal_type = "BUY"
    sig.confidence = confidence
    sig.price = price
    sig.trade_type = "SWING"
    sig.indicators = indicators
    sig.reason = "BUY: momentum swing"
    return sig


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear and re-register Quebec for each test."""
    StrategyRegistry.clear()
    import modules.strategies.quebec
    if "quebec" not in StrategyRegistry.list_strategies():
        StrategyRegistry.register("quebec")(modules.strategies.quebec.QuebecStrategy)
    yield
    StrategyRegistry.clear()


def _make_strategy():
    """Instantiate and init Quebec with sentiment disabled."""
    cls = StrategyRegistry.get("quebec")
    s = cls()
    s.init()
    s._use_sentiment = False
    return s


# ── REGISTRATION TESTS ───────────────────────────────────────

class TestRegistration:

    def test_strategy_registered(self):
        assert "quebec" in StrategyRegistry.list_strategies()

    def test_name(self):
        s = _make_strategy()
        assert s.name == "quebec"

    def test_version(self):
        s = _make_strategy()
        assert s.version == "1.0.0"

    def test_preferred_signals(self):
        s = _make_strategy()
        expected = {"ema_crossover_swing", "macd_crossover", "adx_trend", "volume_spike"}
        assert s.preferred_signals == expected

    def test_empty_watchlist_by_default(self):
        s = _make_strategy()
        assert s.get_watchlist() == []


# ── QUALIFICATION TESTS ──────────────────────────────────────

class TestQualification:

    def test_high_volume_stock_accepted(self):
        s = _make_strategy()
        stock = {"ticker": "QMOM", "avg_volume": 500_000, "price": 50.0}
        assert s.qualifies_stock(stock) is True

    def test_low_volume_rejected(self):
        s = _make_strategy()
        stock = {"ticker": "X", "avg_volume": 50_000, "price": 50.0}
        assert s.qualifies_stock(stock) is False

    def test_penny_stock_rejected(self):
        s = _make_strategy()
        stock = {"ticker": "X", "avg_volume": 500_000, "price": 0.50}
        assert s.qualifies_stock(stock) is False

    def test_empty_data_rejected(self):
        s = _make_strategy()
        assert s.qualifies_stock({}) is False

    def test_none_values_rejected(self):
        s = _make_strategy()
        stock = {"ticker": "X", "avg_volume": None, "price": None}
        assert s.qualifies_stock(stock) is False

    def test_set_watchlist_filters(self):
        s = _make_strategy()
        stocks = [
            {"ticker": "GOOD", "avg_volume": 500_000, "price": 50.0},
            {"ticker": "BAD", "avg_volume": 10_000, "price": 0.50},
        ]
        s.set_watchlist(stocks)
        wl = s.get_watchlist()
        assert "GOOD" in wl
        assert "BAD" not in wl


# ── SIGNAL GENERATION TESTS ─────────────────────────────────

class TestSignalGeneration:

    def test_empty_bars_returns_empty(self):
        s = _make_strategy()
        assert s.generate_signals({}) == []

    def test_none_df_handled(self):
        s = _make_strategy()
        assert s.generate_signals({"AAPL": None}) == []

    def test_empty_df_handled(self):
        s = _make_strategy()
        assert s.generate_signals({"AAPL": pd.DataFrame()}) == []

    def test_generates_signal_with_3_confirmations(self):
        """3 preferred signals + RSI in range = signal generated."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }

        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 55.0  # In the 40-70 range
            snapshot.adx = 26.0
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators)]

            bars = {"AAPL": make_ohlcv(start_price=100.0)}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].ticker == "AAPL"
            assert signals[0].action == "BUY"
            assert signals[0].trade_type == "SWING"
            assert signals[0].metadata["strategy"] == "quebec"
            assert signals[0].metadata["confirmations"] >= 3

    def test_rejects_with_only_2_confirmations(self):
        """Only 2 preferred signals and RSI out of range = no signal."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
        }

        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 75.0  # Out of 40-70 range
            snapshot.adx = 20.0
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0

    def test_rsi_in_range_counts_as_confirmation(self):
        """2 preferred signals + RSI in range = 3 confirmations = signal."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
        }

        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 55.0  # In range — counts as confirmation
            snapshot.adx = 20.0
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 1

    def test_sell_signals_skipped(self):
        s = _make_strategy()
        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            mc.return_value = snapshot
            sig = MagicMock()
            sig.signal_type = "SELL"
            md.return_value = [sig]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0

    def test_low_confidence_skipped(self):
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }

        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 55.0
            snapshot.adx = 20.0
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators, confidence=20.0)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0

    def test_non_preferred_signal_skipped(self):
        s = _make_strategy()
        indicators = {
            "rsi_oversold": {"name": "rsi_oversold"},
        }

        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0


# ── STOP LOSS AND TARGET TESTS ──────────────────────────────

class TestStopLossAndTarget:

    def _generate_signal(self, s, price=100.0):
        """Helper to generate a signal with known price."""
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }
        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None  # Force fallback
            snapshot.rsi = 55.0
            snapshot.adx = 26.0
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators, price=price)]

            bars = {"AAPL": make_ohlcv(start_price=price)}
            return s.generate_signals(bars)

    def test_fallback_stop_loss(self):
        """Fallback stop = price * (1 - 0.07) = 93.0 for price 100."""
        s = _make_strategy()
        signals = self._generate_signal(s, price=100.0)
        assert len(signals) == 1
        assert signals[0].stop_loss == round(100.0 * 0.93, 2)

    def test_target_rr_ratio(self):
        """Target = price + (price - stop) * 3.0."""
        s = _make_strategy()
        signals = self._generate_signal(s, price=100.0)
        assert len(signals) == 1
        risk = 100.0 - signals[0].stop_loss  # 7.0
        expected_target = round(100.0 + risk * 3.0, 2)
        assert signals[0].target == expected_target

    def test_atr_based_stop_loss(self):
        """When ATR is available, stop = price - ATR * 2.5."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }
        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = 2.0  # ATR of $2
            snapshot.rsi = 55.0
            snapshot.adx = 26.0
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators, price=100.0)]

            bars = {"AAPL": make_ohlcv(start_price=100.0)}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].stop_loss == round(100.0 - 2.0 * 2.5, 2)  # 95.0


# ── CONFIDENCE BOOST TESTS ──────────────────────────────────

class TestConfidenceBoosts:

    def test_adx_strong_boost(self):
        """ADX > 30 adds +5 confidence."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }
        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 55.0
            snapshot.adx = 35.0  # > 30 → boost
            snapshot.current_volume = 1_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators, confidence=70.0)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].confidence == 75.0  # 70 + 5

    def test_volume_exceptional_boost(self):
        """Volume > 2x average adds +5 confidence."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }
        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 55.0
            snapshot.adx = 20.0  # Not > 30
            snapshot.current_volume = 3_000_000  # > 2x avg
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators, confidence=70.0)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].confidence == 75.0  # 70 + 5

    def test_both_boosts_stack(self):
        """ADX > 30 + volume > 2x = +10 total."""
        s = _make_strategy()
        indicators = {
            "ema_crossover_swing": {"name": "ema_crossover_swing"},
            "macd_crossover": {"name": "macd_crossover"},
            "adx_trend": {"name": "adx_trend"},
        }
        with patch("modules.strategies.quebec.compute_indicators") as mc, \
             patch("modules.strategies.quebec.detect_signals") as md:
            snapshot = MagicMock()
            snapshot.atr = None
            snapshot.rsi = 55.0
            snapshot.adx = 35.0
            snapshot.current_volume = 3_000_000
            snapshot.avg_volume = 1_000_000
            mc.return_value = snapshot
            md.return_value = [_mock_buy_signal(indicators, confidence=70.0)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].confidence == 80.0  # 70 + 5 + 5


# ── MARKET REGIME TESTS ─────────────────────────────────────

class TestMarketRegime:

    def test_bear_tightens_parameters(self):
        s = _make_strategy()
        s.on_market_regime_change(MarketRegime(trend="bear"))
        assert s._atr_multiplier == 2.0
        assert s._rr_ratio == 2.0

    def test_bull_widens_target(self):
        s = _make_strategy()
        s.on_market_regime_change(MarketRegime(trend="bull"))
        assert s._atr_multiplier == 2.5
        assert s._rr_ratio == 4.0

    def test_sideways_resets_to_defaults(self):
        s = _make_strategy()
        s.on_market_regime_change(MarketRegime(trend="bear"))
        s.on_market_regime_change(MarketRegime(trend="sideways"))
        assert s._atr_multiplier == 2.5
        assert s._rr_ratio == 3.0


# ── TRADE NOTIFICATION TESTS ────────────────────────────────

class TestTradeNotifications:

    def test_on_trade_executed(self):
        s = _make_strategy()
        n = TradeNotification(
            trade_id="T1", ticker="AAPL", action="BUY",
            entry_price=100.0, shares=10,
        )
        s.on_trade_executed(n)
        assert len(s._trades_log) == 1

    def test_on_strategy_pause(self):
        """on_strategy_pause doesn't raise."""
        s = _make_strategy()
        s.on_strategy_pause("drawdown limit")

    def test_regime_hooks_dont_raise(self):
        s = _make_strategy()
        s.on_market_regime_change(MarketRegime(trend="bull"))
        s.on_market_regime_change(MarketRegime(trend="bear"))
        s.on_market_regime_change(MarketRegime(trend="sideways"))
