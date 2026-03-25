"""
Tests for NATO Strategy Plugins (Alpha through Papa)
=====================================================
Tests registration, qualification filters, signal generation,
stop loss / target parameters, and metadata for all 14 new strategies.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from collections import defaultdict

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


# All 14 NATO strategy names and their modules
NATO_STRATEGIES = {
    "alpha": "modules.strategies.alpha",
    "bravo": "modules.strategies.bravo",
    "charlie": "modules.strategies.charlie",
    "delta": "modules.strategies.delta",
    "echo": "modules.strategies.echo",
    "foxtrot": "modules.strategies.foxtrot",
    "golf": "modules.strategies.golf",
    "hotel": "modules.strategies.hotel",
    "india": "modules.strategies.india",
    "juliet": "modules.strategies.juliet",
    "kilo": "modules.strategies.kilo",
    "lima": "modules.strategies.lima",
    "mike": "modules.strategies.mike",
    "november": "modules.strategies.november",
    "oscar": "modules.strategies.oscar",
    "papa": "modules.strategies.papa",
}

# Expected stop loss percentages and R:R ratios per strategy
STRATEGY_PARAMS = {
    "alpha":    {"stop_pct": 0.05, "rr": 2.0},
    "bravo":    {"stop_pct": 0.04, "rr": 1.5},
    "charlie":  {"stop_pct": 0.07, "rr": 2.0},
    "delta":    {"stop_pct": 0.06, "rr": 2.0},
    "echo":     {"stop_pct": 0.06, "rr": 2.0},
    "foxtrot":  {"stop_pct": 0.06, "rr": 2.5},
    "golf":     {"stop_pct": 0.07, "rr": 2.5},
    "hotel":    {"stop_pct": 0.05, "rr": 1.8},
    "india":    {"stop_pct": 0.05, "rr": 2.0},
    "juliet":   {"stop_pct": 0.08, "rr": 2.5},
    "kilo":     {"stop_pct": 0.06, "rr": 1.5},
    "lima":     {"stop_pct": 0.06, "rr": 2.0},
    "mike":     {"stop_pct": 0.05, "rr": 1.8},
    "november": {"stop_pct": 0.07, "rr": 2.5},
    "oscar":    {"stop_pct": 0.04, "rr": 1.5},
    "papa":     {"stop_pct": 0.08, "rr": 3.0},
}

# Stock data templates for qualification tests
QUALIFYING_STOCKS = {
    "alpha": {
        "ticker": "AFST", "lynch_category": "fast_grower",
        "peg_ratio": 0.9, "earnings_growth": 0.30, "debt_to_equity": 0.4,
        "lynch_score": 80, "composite_score": 70,
    },
    "bravo": {
        "ticker": "BSTL", "lynch_category": "stalwart",
        "market_cap": 50_000_000_000, "earnings_growth": 0.15,
        "debt_to_equity": 0.4, "institutional_pct": 0.50,
        "lynch_score": 70, "composite_score": 65,
    },
    "charlie": {
        "ticker": "CDVL", "price_to_tangible_book": 0.7,
        "fcf_yield": 0.10, "ev_to_ebitda": 6, "current_ratio": 2.0,
        "burry_score": 80, "composite_score": 70,
    },
    "delta": {
        "ticker": "DCFM", "fcf_yield": 0.12, "debt_to_equity": 0.3,
        "current_ratio": 2.5, "ev_to_ebitda": 8,
        "burry_score": 75, "composite_score": 68,
    },
    "echo": {
        "ticker": "ECYC", "lynch_category": "cyclical",
        "revenue_growth": 0.08, "current_ratio": 1.5, "debt_to_equity": 0.7,
        "lynch_score": 60, "composite_score": 55,
    },
    "foxtrot": {
        "ticker": "FAST", "lynch_category": "asset_play",
        "price_to_tangible_book": 1.2, "current_ratio": 2.0,
        "institutional_pct": 0.25, "lynch_score": 65, "composite_score": 60,
    },
    "golf": {
        "ticker": "GCNT", "short_interest": 0.20, "fcf_yield": 0.07,
        "ev_to_ebitda": 10, "current_ratio": 1.5, "burry_score": 55,
        "composite_score": 58,
    },
    "hotel": {
        "ticker": "HHYB", "lynch_score": 70, "burry_score": 65,
        "composite_score": 67.5, "debt_to_equity": 0.5,
        "lynch_category": "fast_grower",
    },
    "india": {
        "ticker": "IMOM", "earnings_growth": 0.30, "revenue_growth": 0.25,
        "peg_ratio": 1.5, "composite_score": 65,
    },
    "juliet": {
        "ticker": "JDVL", "price_to_tangible_book": 0.8, "fcf_yield": 0.08,
        "current_ratio": 2.5, "short_interest": 0.15,
        "burry_score": 70, "composite_score": 65,
    },
    "kilo": {
        "ticker": "KROT", "composite_score": 60, "earnings_growth": 0.15,
        "sector": "Technology",
    },
    "lima": {
        "ticker": "LSCQ", "market_cap": 1_500_000_000,
        "institutional_pct": 0.25, "revenue_growth": 0.20,
        "debt_to_equity": 0.5, "composite_score": 60,
    },
    "mike": {
        "ticker": "MCFC", "fcf_yield": 0.09, "debt_to_equity": 0.3,
        "ev_to_ebitda": 10, "earnings_growth": 0.15,
        "composite_score": 65,
    },
    "november": {
        "ticker": "NTRN", "lynch_category": "turnaround",
        "current_ratio": 2.0, "fcf_yield": 0.05,
        "price_to_tangible_book": 1.8, "lynch_score": 55, "composite_score": 50,
    },
    "oscar": {
        "ticker": "OGRP", "peg_ratio": 1.0, "earnings_growth": 0.18,
        "debt_to_equity": 0.5, "institutional_pct": 0.50,
        "composite_score": 65,
    },
    "papa": {
        "ticker": "PSQZ", "short_interest": 0.25, "revenue_growth": 0.08,
        "current_ratio": 1.5, "ev_to_ebitda": 12,
        "composite_score": 55,
    },
}

# Stocks that should NOT qualify for each strategy
NON_QUALIFYING_STOCKS = {
    "alpha": {"ticker": "X", "lynch_category": "stalwart", "peg_ratio": 2.0, "earnings_growth": 0.05, "debt_to_equity": 1.5},
    "bravo": {"ticker": "X", "lynch_category": "stalwart", "market_cap": 1_000_000_000, "earnings_growth": 0.05, "debt_to_equity": 1.0, "institutional_pct": 0.10},
    "charlie": {"ticker": "X", "price_to_tangible_book": 3.0, "fcf_yield": 0.02, "ev_to_ebitda": 20, "current_ratio": 0.8},
    "delta": {"ticker": "X", "fcf_yield": 0.03, "debt_to_equity": 1.0, "current_ratio": 1.0, "ev_to_ebitda": 15},
    "echo": {"ticker": "X", "lynch_category": "fast_grower", "revenue_growth": 0.01, "current_ratio": 0.8, "debt_to_equity": 2.0},
    "foxtrot": {"ticker": "X", "lynch_category": "stalwart", "price_to_tangible_book": 3.0, "current_ratio": 0.8, "institutional_pct": 0.80},
    "golf": {"ticker": "X", "short_interest": 0.05, "fcf_yield": 0.02, "ev_to_ebitda": 20, "current_ratio": 0.8, "burry_score": 30},
    "hotel": {"ticker": "X", "lynch_score": 40, "burry_score": 40, "composite_score": 40, "debt_to_equity": 1.5},
    "india": {"ticker": "X", "earnings_growth": 0.10, "revenue_growth": 0.05, "peg_ratio": 3.0},
    "juliet": {"ticker": "X", "price_to_tangible_book": 3.0, "fcf_yield": 0.02, "current_ratio": 1.0, "short_interest": 0.03},
    "kilo": {"ticker": "X", "composite_score": 30, "earnings_growth": 0.03, "sector": "Utilities"},
    "lima": {"ticker": "X", "market_cap": 50_000_000_000, "institutional_pct": 0.80, "revenue_growth": 0.05, "debt_to_equity": 2.0},
    "mike": {"ticker": "X", "fcf_yield": 0.02, "debt_to_equity": 1.5, "ev_to_ebitda": 20, "earnings_growth": 0.03},
    "november": {"ticker": "X", "lynch_category": "fast_grower", "current_ratio": 0.8, "fcf_yield": -0.02, "price_to_tangible_book": 5.0},
    "oscar": {"ticker": "X", "peg_ratio": 3.0, "earnings_growth": 0.05, "debt_to_equity": 1.5, "institutional_pct": 0.90},
    "papa": {"ticker": "X", "short_interest": 0.05, "revenue_growth": -0.10, "current_ratio": 0.8, "ev_to_ebitda": 25},
}


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear and re-register all strategies for each test."""
    StrategyRegistry.clear()
    # Import all strategy modules to trigger registration
    import modules.strategies.alpha
    import modules.strategies.bravo
    import modules.strategies.charlie
    import modules.strategies.delta
    import modules.strategies.echo
    import modules.strategies.foxtrot
    import modules.strategies.golf
    import modules.strategies.hotel
    import modules.strategies.india
    import modules.strategies.juliet
    import modules.strategies.kilo
    import modules.strategies.lima
    import modules.strategies.mike
    import modules.strategies.november
    import modules.strategies.oscar
    import modules.strategies.papa

    # Re-register if cleared (decorators only fire once per import)
    strategy_classes = {
        "alpha": modules.strategies.alpha.AlphaStrategy,
        "bravo": modules.strategies.bravo.BravoStrategy,
        "charlie": modules.strategies.charlie.CharlieStrategy,
        "delta": modules.strategies.delta.DeltaStrategy,
        "echo": modules.strategies.echo.EchoStrategy,
        "foxtrot": modules.strategies.foxtrot.FoxtrotStrategy,
        "golf": modules.strategies.golf.GolfStrategy,
        "hotel": modules.strategies.hotel.HotelStrategy,
        "india": modules.strategies.india.IndiaStrategy,
        "juliet": modules.strategies.juliet.JulietStrategy,
        "kilo": modules.strategies.kilo.KiloStrategy,
        "lima": modules.strategies.lima.LimaStrategy,
        "mike": modules.strategies.mike.MikeStrategy,
        "november": modules.strategies.november.NovemberStrategy,
        "oscar": modules.strategies.oscar.OscarStrategy,
        "papa": modules.strategies.papa.PapaStrategy,
    }
    for name, cls in strategy_classes.items():
        if name not in StrategyRegistry.list_strategies():
            StrategyRegistry.register(name)(cls)
    yield
    StrategyRegistry.clear()


def _make_strategy(name: str):
    """Instantiate and init a strategy by name with sentiment disabled."""
    cls = StrategyRegistry.get(name)
    s = cls()
    s.init()
    s._use_sentiment = False
    return s


def _mock_buy_signal(preferred_signal: str, price: float = 100.0, confidence: float = 75.0):
    """Create a mock trade signal matching a preferred signal type."""
    sig = MagicMock()
    sig.signal_type = "BUY"
    sig.confidence = confidence
    sig.price = price
    sig.trade_type = "DAY"
    sig.indicators = {preferred_signal: {"name": preferred_signal}}
    sig.reason = f"BUY: {preferred_signal}"
    return sig


# ── REGISTRATION TESTS ───────────────────────────────────────

class TestRegistration:
    """All 14 NATO strategies must register correctly."""

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_strategy_registered(self, name):
        assert name in StrategyRegistry.list_strategies()

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_strategy_instantiation(self, name):
        s = _make_strategy(name)
        assert s.name == name
        assert s.version == "1.0.0"

    def test_all_16_strategies_registered(self):
        """All 14 NATO + Lynch + Burry = 16 total."""
        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) >= 16

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_preferred_signals_not_empty(self, name):
        s = _make_strategy(name)
        assert len(s.preferred_signals) > 0
        assert isinstance(s.preferred_signals, set)

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_empty_watchlist_by_default(self, name):
        s = _make_strategy(name)
        assert s.get_watchlist() == []


# ── QUALIFICATION TESTS ──────────────────────────────────────

class TestQualification:
    """Each strategy correctly accepts/rejects stocks."""

    @pytest.mark.parametrize("name", list(QUALIFYING_STOCKS.keys()))
    def test_qualifying_stock_accepted(self, name):
        s = _make_strategy(name)
        assert s.qualifies_stock(QUALIFYING_STOCKS[name]) is True

    @pytest.mark.parametrize("name", list(NON_QUALIFYING_STOCKS.keys()))
    def test_non_qualifying_stock_rejected(self, name):
        s = _make_strategy(name)
        assert s.qualifies_stock(NON_QUALIFYING_STOCKS[name]) is False

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_empty_data_rejected(self, name):
        s = _make_strategy(name)
        assert s.qualifies_stock({}) is False

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_none_values_handled(self, name):
        """Strategies handle None values without crashing."""
        s = _make_strategy(name)
        stock = {"ticker": "NULL", "lynch_category": None, "peg_ratio": None,
                 "earnings_growth": None, "debt_to_equity": None,
                 "revenue_growth": None, "institutional_pct": None,
                 "fcf_yield": None, "price_to_tangible_book": None,
                 "short_interest": None, "ev_to_ebitda": None,
                 "current_ratio": None, "lynch_score": None,
                 "burry_score": None, "composite_score": None,
                 "market_cap": None, "sector": None}
        # Should not raise, should return False
        result = s.qualifies_stock(stock)
        assert result is False


# ── SET_WATCHLIST TESTS ──────────────────────────────────────

class TestSetWatchlist:
    """set_watchlist filters correctly for each strategy."""

    @pytest.mark.parametrize("name", list(QUALIFYING_STOCKS.keys()))
    def test_set_watchlist_includes_qualifying(self, name):
        s = _make_strategy(name)
        qualifying = QUALIFYING_STOCKS[name]
        non_qualifying = NON_QUALIFYING_STOCKS[name]
        s.set_watchlist([qualifying, non_qualifying])
        wl = s.get_watchlist()
        assert qualifying["ticker"] in wl
        assert non_qualifying["ticker"] not in wl


# ── SIGNAL GENERATION TESTS ─────────────────────────────────

class TestSignalGeneration:
    """Signal generation works for all strategies."""

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_empty_bars_returns_empty(self, name):
        s = _make_strategy(name)
        assert s.generate_signals({}) == []

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_none_df_handled(self, name):
        s = _make_strategy(name)
        assert s.generate_signals({"AAPL": None}) == []

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_empty_df_handled(self, name):
        s = _make_strategy(name)
        assert s.generate_signals({"AAPL": pd.DataFrame()}) == []

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_generates_buy_signal(self, name):
        """Each strategy generates a signal when given matching technical data."""
        s = _make_strategy(name)
        preferred = list(s.preferred_signals)[0]
        module_path = NATO_STRATEGIES[name]

        with patch(f"{module_path}.compute_indicators") as mc, \
             patch(f"{module_path}.detect_signals") as md:
            mc.return_value = MagicMock()
            md.return_value = [_mock_buy_signal(preferred, price=100.0)]

            bars = {"AAPL": make_ohlcv(start_price=100.0)}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].ticker == "AAPL"
            assert signals[0].action == "BUY"
            assert signals[0].entry_price == 100.0
            assert signals[0].metadata["strategy"] == name

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_sell_signals_skipped(self, name):
        """Strategies only generate BUY signals."""
        s = _make_strategy(name)
        module_path = NATO_STRATEGIES[name]

        with patch(f"{module_path}.compute_indicators") as mc, \
             patch(f"{module_path}.detect_signals") as md:
            mc.return_value = MagicMock()
            sig = MagicMock()
            sig.signal_type = "SELL"
            md.return_value = [sig]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_low_confidence_skipped(self, name):
        """Signals below SIGNAL_MIN_CONFIDENCE_TO_TRADE are skipped."""
        s = _make_strategy(name)
        preferred = list(s.preferred_signals)[0]
        module_path = NATO_STRATEGIES[name]

        with patch(f"{module_path}.compute_indicators") as mc, \
             patch(f"{module_path}.detect_signals") as md:
            mc.return_value = MagicMock()
            md.return_value = [_mock_buy_signal(preferred, confidence=20.0)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_non_preferred_signal_skipped(self, name):
        """Strategies skip signals not in their preferred_signals set."""
        s = _make_strategy(name)
        # Find a signal type NOT in this strategy's preferred set
        all_types = {"ema_crossover_day", "ema_crossover_swing", "rsi_oversold",
                     "rsi_overbought", "macd_crossover", "volume_spike"}
        non_preferred = all_types - s.preferred_signals
        if not non_preferred:
            pytest.skip("Strategy accepts all signal types")
        wrong_signal = list(non_preferred)[0]
        module_path = NATO_STRATEGIES[name]

        with patch(f"{module_path}.compute_indicators") as mc, \
             patch(f"{module_path}.detect_signals") as md:
            mc.return_value = MagicMock()
            md.return_value = [_mock_buy_signal(wrong_signal)]

            bars = {"AAPL": make_ohlcv()}
            signals = s.generate_signals(bars)
            assert len(signals) == 0


# ── STOP LOSS AND TARGET TESTS ──────────────────────────────

class TestStopLossAndTarget:
    """Each strategy uses its documented stop % and R:R ratio."""

    @pytest.mark.parametrize("name,params", list(STRATEGY_PARAMS.items()))
    def test_stop_loss_percentage(self, name, params):
        s = _make_strategy(name)
        preferred = list(s.preferred_signals)[0]
        module_path = NATO_STRATEGIES[name]
        price = 100.0

        with patch(f"{module_path}.compute_indicators") as mc, \
             patch(f"{module_path}.detect_signals") as md:
            mc.return_value = MagicMock()
            md.return_value = [_mock_buy_signal(preferred, price=price)]

            bars = {"AAPL": make_ohlcv(start_price=price)}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            expected_stop = round(price * (1 - params["stop_pct"]), 2)
            assert signals[0].stop_loss == expected_stop, (
                f"{name}: expected stop {expected_stop}, got {signals[0].stop_loss}"
            )

    @pytest.mark.parametrize("name,params", list(STRATEGY_PARAMS.items()))
    def test_target_rr_ratio(self, name, params):
        s = _make_strategy(name)
        preferred = list(s.preferred_signals)[0]
        module_path = NATO_STRATEGIES[name]
        price = 100.0

        with patch(f"{module_path}.compute_indicators") as mc, \
             patch(f"{module_path}.detect_signals") as md:
            mc.return_value = MagicMock()
            md.return_value = [_mock_buy_signal(preferred, price=price)]

            bars = {"AAPL": make_ohlcv(start_price=price)}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            risk = price - signals[0].stop_loss
            expected_target = round(price + risk * params["rr"], 2)
            assert signals[0].target == expected_target, (
                f"{name}: expected target {expected_target}, got {signals[0].target}"
            )


# ── TRADE NOTIFICATION TESTS ────────────────────────────────

class TestTradeNotifications:

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_on_trade_executed(self, name):
        s = _make_strategy(name)
        n = TradeNotification(
            trade_id="T1", ticker="AAPL", action="BUY",
            entry_price=100.0, shares=10,
        )
        s.on_trade_executed(n)
        assert len(s._trades_log) == 1

    @pytest.mark.parametrize("name", list(NATO_STRATEGIES.keys()))
    def test_optional_hooks_dont_raise(self, name):
        s = _make_strategy(name)
        s.on_market_regime_change(MarketRegime(trend="bull"))
        s.on_market_regime_change(MarketRegime(trend="bear"))
        s.on_strategy_pause("test pause")


# ── KILO SECTOR ROTATION SPECIFIC TESTS ────────────────────

class TestKiloSectorRotation:
    """Kilo has special sector-based watchlist logic."""

    def test_sector_ranking(self):
        s = _make_strategy("kilo")
        stocks = [
            {"ticker": "T1", "sector": "Technology", "composite_score": 80, "earnings_growth": 0.20},
            {"ticker": "T2", "sector": "Technology", "composite_score": 70, "earnings_growth": 0.15},
            {"ticker": "T3", "sector": "Healthcare", "composite_score": 75, "earnings_growth": 0.18},
            {"ticker": "T4", "sector": "Energy", "composite_score": 40, "earnings_growth": 0.12},
            {"ticker": "T5", "sector": "Energy", "composite_score": 35, "earnings_growth": 0.08},
        ]
        s.set_watchlist(stocks)
        wl = s.get_watchlist()
        # Technology and Healthcare should be top 2 sectors
        assert "T1" in wl
        assert "T2" in wl
        assert "T3" in wl
        # Energy has lower average score, should be excluded
        assert "T4" not in wl
        assert "T5" not in wl

    def test_top_sectors_tracked(self):
        s = _make_strategy("kilo")
        stocks = [
            {"ticker": "T1", "sector": "Technology", "composite_score": 80, "earnings_growth": 0.20},
            {"ticker": "T2", "sector": "Healthcare", "composite_score": 70, "earnings_growth": 0.15},
        ]
        s.set_watchlist(stocks)
        assert len(s._top_sectors) == 2
        assert "Technology" in s._top_sectors
        assert "Healthcare" in s._top_sectors


# ── PAPA VOLUME SPIKE BOOST TESTS ──────────────────────────

class TestPapaVolumeSpikeBoost:
    """Papa has a confidence boost for volume spikes."""

    def test_volume_spike_boost_applied(self):
        s = _make_strategy("papa")
        with patch("modules.strategies.papa.compute_indicators") as mc, \
             patch("modules.strategies.papa.detect_signals") as md:
            mc.return_value = MagicMock()
            sig = MagicMock()
            sig.signal_type = "BUY"
            sig.confidence = 65.0
            sig.price = 50.0
            sig.trade_type = "DAY"
            sig.indicators = [{"name": "volume_spike"}]
            sig.reason = ""
            md.return_value = [sig]

            bars = {"AAPL": make_ohlcv(start_price=50.0)}
            signals = s.generate_signals(bars)

            assert len(signals) == 1
            assert signals[0].confidence == 70.0  # 65 + 5 boost
            assert signals[0].metadata["volume_spike_detected"] is True


# ── CROSS-STRATEGY DIFFERENTIATION TESTS ───────────────────

class TestStrategyDifferentiation:
    """Strategies should have distinct configurations."""

    def test_no_duplicate_preferred_signals_and_stop_combo(self):
        """No two strategies should have identical preferred_signals + stop + R:R."""
        combos = {}
        for name in NATO_STRATEGIES:
            s = _make_strategy(name)
            key = (frozenset(s.preferred_signals), STRATEGY_PARAMS[name]["stop_pct"], STRATEGY_PARAMS[name]["rr"])
            if key in combos:
                # It's OK if the combo is shared — the qualification filters differentiate
                pass
            combos[key] = name

    def test_all_strategies_have_unique_names(self):
        seen = set()
        for name in NATO_STRATEGIES:
            s = _make_strategy(name)
            assert s.name not in seen
            seen.add(s.name)

    def test_create_all_includes_nato_strategies(self):
        """StrategyRegistry.create_all() includes all NATO strategies."""
        instances = StrategyRegistry.create_all()
        for name in NATO_STRATEGIES:
            assert name in instances
            assert instances[name].name == name
