"""
EdgeFinder Module 2 Tests: Technical Signal Engine
====================================================
Tests cover: indicator computation, EMA crossover detection, RSI signals,
MACD crossovers, volume spikes, confidence scoring, trade type classification,
signal aggregation, database persistence, and edge cases.

Run: python -m pytest tests/test_signals.py -v
"""

import numpy as np
import pandas as pd
import pytest

from modules.signals import (
    IndicatorSnapshot,
    TradeSignal,
    compute_indicators,
    detect_ema_crossover_day,
    detect_ema_crossover_swing,
    detect_rsi_signal,
    detect_macd_crossover,
    detect_volume_spike,
    classify_trade_type,
    compute_confidence,
    determine_direction,
    generate_signals,
    scan_ticker,
    scan_watchlist,
    _save_signals,
    _safe_float,
)
from config import settings


# ── TEST HELPERS ─────────────────────────────────────────────

def make_ohlcv(
    n: int = 250,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.02,
    base_volume: float = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        n: Number of bars.
        start_price: Starting close price.
        trend: Daily drift (e.g., 0.001 for uptrend).
        volatility: Daily return std dev.
        base_volume: Average volume.
        seed: Random seed for reproducibility.
    """
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


def make_crossover_df(
    fast_above_slow_at_end: bool = True,
    cross_type: str = "day",
) -> pd.DataFrame:
    """
    Generate OHLCV data that produces a specific EMA crossover pattern.

    Creates a price series that trends one way, then crosses over.
    """
    n = 250
    rng = np.random.RandomState(123)

    if cross_type == "day":
        # For 9/21 EMA: need a relatively quick reversal
        pivot = n - 5
    else:
        # For 50/200 EMA: need a longer trend shift
        pivot = n - 60

    prices = np.zeros(n)
    prices[0] = 100.0

    for i in range(1, n):
        if fast_above_slow_at_end:
            # Downtrend then uptrend (bullish crossover)
            if i < pivot:
                prices[i] = prices[i - 1] * (1 - 0.0005 + rng.normal(0, 0.005))
            else:
                prices[i] = prices[i - 1] * (1 + 0.008 + rng.normal(0, 0.003))
        else:
            # Uptrend then downtrend (bearish crossover)
            if i < pivot:
                prices[i] = prices[i - 1] * (1 + 0.0005 + rng.normal(0, 0.005))
            else:
                prices[i] = prices[i - 1] * (1 - 0.008 + rng.normal(0, 0.003))

    volume = rng.normal(1_000_000, 200_000, n).clip(min=1000)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")

    return pd.DataFrame({
        "Open": prices * (1 + rng.normal(0, 0.003, n)),
        "High": prices * (1 + rng.uniform(0, 0.015, n)),
        "Low": prices * (1 - rng.uniform(0, 0.015, n)),
        "Close": prices,
        "Volume": volume,
    }, index=dates)


# ── INDICATOR SNAPSHOT FIXTURES ──────────────────────────────

@pytest.fixture
def bullish_ema_day_snap() -> IndicatorSnapshot:
    """Snapshot where 9-day EMA just crossed above 21-day."""
    return IndicatorSnapshot(
        ticker="BULL",
        ema_fast_day=51.0, ema_slow_day=50.0,
        prev_ema_fast_day=49.0, prev_ema_slow_day=50.0,
        rsi=45.0,
        current_price=51.5,
    )


@pytest.fixture
def bearish_ema_day_snap() -> IndicatorSnapshot:
    """Snapshot where 9-day EMA just crossed below 21-day."""
    return IndicatorSnapshot(
        ticker="BEAR",
        ema_fast_day=49.0, ema_slow_day=50.0,
        prev_ema_fast_day=51.0, prev_ema_slow_day=50.0,
        rsi=55.0,
        current_price=48.5,
    )


@pytest.fixture
def bullish_swing_snap() -> IndicatorSnapshot:
    """Snapshot where 50-day EMA just crossed above 200-day (golden cross)."""
    return IndicatorSnapshot(
        ticker="GOLD",
        ema_fast_swing=101.0, ema_slow_swing=100.0,
        prev_ema_fast_swing=99.0, prev_ema_slow_swing=100.0,
        rsi=55.0,
        current_price=102.0,
    )


@pytest.fixture
def rsi_oversold_snap() -> IndicatorSnapshot:
    """RSI below 30 — oversold."""
    return IndicatorSnapshot(
        ticker="OVER",
        rsi=25.0,
        current_price=30.0,
    )


@pytest.fixture
def rsi_overbought_snap() -> IndicatorSnapshot:
    """RSI above 70 — overbought."""
    return IndicatorSnapshot(
        ticker="HIGH",
        rsi=78.0,
        current_price=150.0,
    )


@pytest.fixture
def macd_bullish_snap() -> IndicatorSnapshot:
    """MACD line just crossed above signal line."""
    return IndicatorSnapshot(
        ticker="MACD",
        macd_line=0.5, macd_signal=0.3,
        prev_macd_line=0.2, prev_macd_signal=0.3,
        macd_histogram=0.2,
        current_price=50.0,
    )


@pytest.fixture
def volume_spike_snap() -> IndicatorSnapshot:
    """Volume at 2x the 20-day average."""
    return IndicatorSnapshot(
        ticker="VOL",
        current_volume=2_000_000,
        avg_volume=1_000_000,
        current_price=50.0,
    )


@pytest.fixture
def multi_signal_bullish_snap() -> IndicatorSnapshot:
    """Multiple bullish indicators firing at once."""
    return IndicatorSnapshot(
        ticker="MULTI",
        # EMA day crossover (bullish)
        ema_fast_day=51.0, ema_slow_day=50.0,
        prev_ema_fast_day=49.0, prev_ema_slow_day=50.0,
        # RSI oversold
        rsi=28.0,
        # MACD bullish crossover
        macd_line=0.5, macd_signal=0.3,
        prev_macd_line=0.2, prev_macd_signal=0.3,
        macd_histogram=0.2,
        # Volume spike
        current_volume=2_500_000,
        avg_volume=1_000_000,
        current_price=50.0,
    )


@pytest.fixture
def no_signal_snap() -> IndicatorSnapshot:
    """All indicators neutral — no signals should fire."""
    return IndicatorSnapshot(
        ticker="FLAT",
        ema_fast_day=50.0, ema_slow_day=50.0,
        prev_ema_fast_day=50.0, prev_ema_slow_day=50.0,
        ema_fast_swing=100.0, ema_slow_swing=100.0,
        prev_ema_fast_swing=100.0, prev_ema_slow_swing=100.0,
        rsi=50.0,
        macd_line=0.0, macd_signal=0.0,
        prev_macd_line=0.0, prev_macd_signal=0.0,
        macd_histogram=0.0,
        current_volume=1_000_000,
        avg_volume=1_000_000,
        current_price=100.0,
    )


# ════════════════════════════════════════════════════════════
# EMA CROSSOVER — DAY TRADE (9/21)
# ════════════════════════════════════════════════════════════

class TestEMACrossoverDay:
    """Test 9/21 EMA crossover detection for day trades."""

    def test_bullish_crossover_detected(self, bullish_ema_day_snap):
        result = detect_ema_crossover_day(bullish_ema_day_snap)
        assert result is not None
        assert result["direction"] == "BUY"
        assert result["name"] == "ema_crossover_day"

    def test_bearish_crossover_detected(self, bearish_ema_day_snap):
        result = detect_ema_crossover_day(bearish_ema_day_snap)
        assert result is not None
        assert result["direction"] == "SELL"

    def test_no_crossover_when_flat(self, no_signal_snap):
        result = detect_ema_crossover_day(no_signal_snap)
        assert result is None

    def test_no_crossover_when_already_above(self):
        snap = IndicatorSnapshot(
            ticker="X",
            ema_fast_day=55.0, ema_slow_day=50.0,
            prev_ema_fast_day=53.0, prev_ema_slow_day=50.0,
        )
        result = detect_ema_crossover_day(snap)
        assert result is None

    def test_missing_data_returns_none(self):
        snap = IndicatorSnapshot(ticker="X", ema_fast_day=50.0)
        result = detect_ema_crossover_day(snap)
        assert result is None

    def test_exact_touch_no_crossover(self):
        """Fast touching slow without crossing should not trigger."""
        snap = IndicatorSnapshot(
            ticker="X",
            ema_fast_day=50.0, ema_slow_day=50.0,
            prev_ema_fast_day=49.0, prev_ema_slow_day=50.0,
        )
        # prev: fast <= slow (49 <= 50) → True
        # curr: fast > slow (50 > 50) → False
        result = detect_ema_crossover_day(snap)
        assert result is None


# ════════════════════════════════════════════════════════════
# EMA CROSSOVER — SWING (50/200)
# ════════════════════════════════════════════════════════════

class TestEMACrossoverSwing:
    """Test 50/200 EMA crossover (golden/death cross) for swing trades."""

    def test_golden_cross_detected(self, bullish_swing_snap):
        result = detect_ema_crossover_swing(bullish_swing_snap)
        assert result is not None
        assert result["direction"] == "BUY"
        assert result["name"] == "ema_crossover_swing"

    def test_death_cross_detected(self):
        snap = IndicatorSnapshot(
            ticker="DEATH",
            ema_fast_swing=99.0, ema_slow_swing=100.0,
            prev_ema_fast_swing=101.0, prev_ema_slow_swing=100.0,
        )
        result = detect_ema_crossover_swing(snap)
        assert result is not None
        assert result["direction"] == "SELL"

    def test_no_cross_when_parallel(self, no_signal_snap):
        result = detect_ema_crossover_swing(no_signal_snap)
        assert result is None

    def test_missing_data_returns_none(self):
        snap = IndicatorSnapshot(ticker="X")
        result = detect_ema_crossover_swing(snap)
        assert result is None


# ════════════════════════════════════════════════════════════
# RSI SIGNALS
# ════════════════════════════════════════════════════════════

class TestRSISignal:
    """Test RSI oversold/overbought signal detection."""

    def test_oversold_buy_signal(self, rsi_oversold_snap):
        result = detect_rsi_signal(rsi_oversold_snap)
        assert result is not None
        assert result["direction"] == "BUY"
        assert result["name"] == "rsi_oversold"
        assert result["rsi"] == 25.0

    def test_overbought_sell_signal(self, rsi_overbought_snap):
        result = detect_rsi_signal(rsi_overbought_snap)
        assert result is not None
        assert result["direction"] == "SELL"
        assert result["name"] == "rsi_overbought"

    def test_neutral_rsi_no_signal(self):
        snap = IndicatorSnapshot(ticker="X", rsi=50.0)
        result = detect_rsi_signal(snap)
        assert result is None

    def test_boundary_oversold(self):
        """RSI exactly at oversold threshold should trigger."""
        snap = IndicatorSnapshot(ticker="X", rsi=settings.SIGNAL_RSI_OVERSOLD)
        result = detect_rsi_signal(snap)
        assert result is not None
        assert result["direction"] == "BUY"

    def test_boundary_overbought(self):
        """RSI exactly at overbought threshold should trigger."""
        snap = IndicatorSnapshot(ticker="X", rsi=settings.SIGNAL_RSI_OVERBOUGHT)
        result = detect_rsi_signal(snap)
        assert result is not None
        assert result["direction"] == "SELL"

    def test_just_above_oversold_no_signal(self):
        snap = IndicatorSnapshot(ticker="X", rsi=settings.SIGNAL_RSI_OVERSOLD + 1)
        result = detect_rsi_signal(snap)
        assert result is None

    def test_none_rsi_no_signal(self):
        snap = IndicatorSnapshot(ticker="X", rsi=None)
        result = detect_rsi_signal(snap)
        assert result is None


# ════════════════════════════════════════════════════════════
# MACD CROSSOVER
# ════════════════════════════════════════════════════════════

class TestMACDCrossover:
    """Test MACD line / signal line crossover detection."""

    def test_bullish_macd_crossover(self, macd_bullish_snap):
        result = detect_macd_crossover(macd_bullish_snap)
        assert result is not None
        assert result["direction"] == "BUY"
        assert result["name"] == "macd_crossover"

    def test_bearish_macd_crossover(self):
        snap = IndicatorSnapshot(
            ticker="X",
            macd_line=-0.1, macd_signal=0.1,
            prev_macd_line=0.2, prev_macd_signal=0.1,
            macd_histogram=-0.2,
        )
        result = detect_macd_crossover(snap)
        assert result is not None
        assert result["direction"] == "SELL"

    def test_no_crossover_when_parallel(self, no_signal_snap):
        result = detect_macd_crossover(no_signal_snap)
        assert result is None

    def test_missing_prev_returns_none(self):
        snap = IndicatorSnapshot(
            ticker="X",
            macd_line=0.5, macd_signal=0.3,
        )
        result = detect_macd_crossover(snap)
        assert result is None

    def test_histogram_included_in_result(self, macd_bullish_snap):
        result = detect_macd_crossover(macd_bullish_snap)
        assert "histogram" in result
        assert result["histogram"] == 0.2


# ════════════════════════════════════════════════════════════
# VOLUME SPIKE
# ════════════════════════════════════════════════════════════

class TestVolumeSpike:
    """Test volume spike detection."""

    def test_spike_detected(self, volume_spike_snap):
        result = detect_volume_spike(volume_spike_snap)
        assert result is not None
        assert result["name"] == "volume_spike"
        assert result["direction"] == "NEUTRAL"
        assert result["volume_ratio"] == 2.0

    def test_normal_volume_no_spike(self):
        snap = IndicatorSnapshot(
            ticker="X",
            current_volume=1_000_000,
            avg_volume=1_000_000,
        )
        result = detect_volume_spike(snap)
        assert result is None

    def test_boundary_spike(self):
        """Exactly at multiplier threshold should trigger."""
        snap = IndicatorSnapshot(
            ticker="X",
            current_volume=1_500_000,
            avg_volume=1_000_000,
        )
        result = detect_volume_spike(snap)
        assert result is not None
        assert result["volume_ratio"] == 1.5

    def test_zero_avg_volume_no_crash(self):
        snap = IndicatorSnapshot(
            ticker="X",
            current_volume=1_000_000,
            avg_volume=0,
        )
        result = detect_volume_spike(snap)
        assert result is None

    def test_missing_volume_no_crash(self):
        snap = IndicatorSnapshot(ticker="X")
        result = detect_volume_spike(snap)
        assert result is None


# ════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ════════════════════════════════════════════════════════════

class TestConfidenceScoring:
    """Test confidence score computation based on indicator count."""

    def test_single_indicator_low_confidence(self):
        indicators = [{"name": "rsi_oversold", "direction": "BUY", "rsi": 25}]
        score = compute_confidence(indicators, has_volume_spike=False)
        assert score == settings.SIGNAL_CONFIDENCE_LOW  # 40

    def test_two_indicators_moderate_confidence(self):
        indicators = [
            {"name": "rsi_oversold", "direction": "BUY", "rsi": 25},
            {"name": "ema_crossover_day", "direction": "BUY"},
        ]
        score = compute_confidence(indicators, has_volume_spike=False)
        assert score == settings.SIGNAL_CONFIDENCE_MODERATE  # 60

    def test_three_indicators_high_confidence(self):
        indicators = [
            {"name": "rsi_oversold", "direction": "BUY", "rsi": 25},
            {"name": "ema_crossover_day", "direction": "BUY"},
            {"name": "macd_crossover", "direction": "BUY"},
        ]
        score = compute_confidence(indicators, has_volume_spike=False)
        assert score == settings.SIGNAL_CONFIDENCE_HIGH  # 80

    def test_volume_spike_adds_bonus(self):
        indicators = [{"name": "rsi_oversold", "direction": "BUY", "rsi": 25}]
        without_vol = compute_confidence(indicators, has_volume_spike=False)
        with_vol = compute_confidence(indicators, has_volume_spike=True)
        assert with_vol == without_vol + 10

    def test_deeply_oversold_bonus(self):
        """RSI <= 20 should get extra bonus."""
        indicators = [{"name": "rsi_oversold", "direction": "BUY", "rsi": 18}]
        score = compute_confidence(indicators, has_volume_spike=False)
        assert score == settings.SIGNAL_CONFIDENCE_LOW + 5  # 45

    def test_deeply_overbought_bonus(self):
        """RSI >= 80 should get extra bonus."""
        indicators = [{"name": "rsi_overbought", "direction": "SELL", "rsi": 85}]
        score = compute_confidence(indicators, has_volume_spike=False)
        assert score == settings.SIGNAL_CONFIDENCE_LOW + 5  # 45

    def test_confidence_capped_at_100(self):
        indicators = [
            {"name": "rsi_oversold", "direction": "BUY", "rsi": 15},
            {"name": "ema_crossover_day", "direction": "BUY"},
            {"name": "macd_crossover", "direction": "BUY"},
            {"name": "ema_crossover_swing", "direction": "BUY"},
        ]
        score = compute_confidence(indicators, has_volume_spike=True)
        assert score <= 100.0

    def test_empty_indicators_zero(self):
        assert compute_confidence([], has_volume_spike=False) == 0.0

    def test_only_neutral_indicators_zero(self):
        indicators = [{"name": "volume_spike", "direction": "NEUTRAL"}]
        assert compute_confidence(indicators, has_volume_spike=True) == 0.0


# ════════════════════════════════════════════════════════════
# TRADE TYPE CLASSIFICATION
# ════════════════════════════════════════════════════════════

class TestTradeTypeClassification:
    """Test day trade vs swing trade classification."""

    def test_day_trade_default(self):
        indicators = [
            {"name": "ema_crossover_day", "direction": "BUY"},
            {"name": "rsi_oversold", "direction": "BUY"},
        ]
        assert classify_trade_type(indicators) == "DAY"

    def test_swing_when_swing_ema_present(self):
        indicators = [
            {"name": "ema_crossover_swing", "direction": "BUY"},
            {"name": "rsi_oversold", "direction": "BUY"},
        ]
        assert classify_trade_type(indicators) == "SWING"

    def test_swing_dominates_day(self):
        """If both day and swing crossover, classify as swing."""
        indicators = [
            {"name": "ema_crossover_day", "direction": "BUY"},
            {"name": "ema_crossover_swing", "direction": "BUY"},
        ]
        assert classify_trade_type(indicators) == "SWING"

    def test_empty_indicators_day(self):
        assert classify_trade_type([]) == "DAY"


# ════════════════════════════════════════════════════════════
# DIRECTION DETERMINATION
# ════════════════════════════════════════════════════════════

class TestDirectionDetermination:
    """Test consensus direction from indicator list."""

    def test_unanimous_buy(self):
        indicators = [
            {"name": "rsi", "direction": "BUY"},
            {"name": "ema", "direction": "BUY"},
        ]
        assert determine_direction(indicators) == "BUY"

    def test_unanimous_sell(self):
        indicators = [
            {"name": "rsi", "direction": "SELL"},
            {"name": "macd", "direction": "SELL"},
        ]
        assert determine_direction(indicators) == "SELL"

    def test_conflicting_signals_none(self):
        indicators = [
            {"name": "rsi", "direction": "BUY"},
            {"name": "macd", "direction": "SELL"},
        ]
        assert determine_direction(indicators) is None

    def test_majority_buy(self):
        indicators = [
            {"name": "rsi", "direction": "BUY"},
            {"name": "ema", "direction": "BUY"},
            {"name": "macd", "direction": "SELL"},
        ]
        assert determine_direction(indicators) == "BUY"

    def test_neutral_ignored(self):
        indicators = [
            {"name": "rsi", "direction": "BUY"},
            {"name": "volume", "direction": "NEUTRAL"},
        ]
        assert determine_direction(indicators) == "BUY"

    def test_empty_returns_none(self):
        assert determine_direction([]) is None


# ════════════════════════════════════════════════════════════
# SIGNAL GENERATION (AGGREGATION)
# ════════════════════════════════════════════════════════════

class TestSignalGeneration:
    """Test the full signal generation pipeline from a snapshot."""

    def test_multi_indicator_bullish_generates_buy(self, multi_signal_bullish_snap):
        signals = generate_signals(multi_signal_bullish_snap)
        assert len(signals) >= 1
        buy_signals = [s for s in signals if s.signal_type == "BUY"]
        assert len(buy_signals) == 1
        assert buy_signals[0].confidence >= settings.SIGNAL_CONFIDENCE_HIGH

    def test_no_signal_from_flat(self, no_signal_snap):
        signals = generate_signals(no_signal_snap)
        assert len(signals) == 0

    def test_single_ema_crossover_generates_signal(self, bullish_ema_day_snap):
        signals = generate_signals(bullish_ema_day_snap)
        assert len(signals) == 1
        assert signals[0].signal_type == "BUY"
        assert signals[0].trade_type == "DAY"
        assert signals[0].confidence == settings.SIGNAL_CONFIDENCE_LOW  # Only 1 indicator

    def test_swing_crossover_classified_correctly(self, bullish_swing_snap):
        signals = generate_signals(bullish_swing_snap)
        assert len(signals) == 1
        assert signals[0].trade_type == "SWING"

    def test_none_snapshot_returns_empty(self):
        assert generate_signals(None) == []

    def test_signal_includes_price(self, multi_signal_bullish_snap):
        signals = generate_signals(multi_signal_bullish_snap)
        assert signals[0].price == 50.0

    def test_signal_includes_indicator_details(self, multi_signal_bullish_snap):
        signals = generate_signals(multi_signal_bullish_snap)
        buy = [s for s in signals if s.signal_type == "BUY"][0]
        assert "ema_crossover_day" in buy.indicators
        assert "rsi_oversold" in buy.indicators
        assert "macd_crossover" in buy.indicators

    def test_volume_spike_only_no_signal(self):
        """Volume spike alone (no directional indicator) should not generate signal."""
        snap = IndicatorSnapshot(
            ticker="X",
            current_volume=3_000_000,
            avg_volume=1_000_000,
            current_price=50.0,
        )
        signals = generate_signals(snap)
        assert len(signals) == 0


# ════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION FROM OHLCV
# ════════════════════════════════════════════════════════════

class TestComputeIndicators:
    """Test computing indicators from raw OHLCV data."""

    def test_basic_computation(self):
        df = make_ohlcv(n=250)
        snap = compute_indicators(df, ticker="TEST")
        assert snap is not None
        assert snap.ticker == "TEST"
        assert snap.ema_fast_day is not None
        assert snap.ema_slow_day is not None
        assert snap.ema_fast_swing is not None
        assert snap.ema_slow_swing is not None
        assert snap.rsi is not None
        assert snap.macd_line is not None
        assert snap.macd_signal is not None
        assert snap.current_price is not None

    def test_insufficient_data_returns_none(self):
        df = make_ohlcv(n=50)  # Not enough for 200-day EMA
        snap = compute_indicators(df, ticker="SHORT")
        assert snap is None

    def test_empty_df_returns_none(self):
        snap = compute_indicators(pd.DataFrame(), ticker="EMPTY")
        assert snap is None

    def test_none_df_returns_none(self):
        snap = compute_indicators(None, ticker="NULL")
        assert snap is None

    def test_rsi_in_valid_range(self):
        df = make_ohlcv(n=250)
        snap = compute_indicators(df, ticker="RSI")
        assert 0 <= snap.rsi <= 100

    def test_prev_values_populated(self):
        """Previous bar values needed for crossover detection."""
        df = make_ohlcv(n=250)
        snap = compute_indicators(df, ticker="PREV")
        assert snap.prev_ema_fast_day is not None
        assert snap.prev_ema_slow_day is not None
        assert snap.prev_macd_line is not None
        assert snap.prev_macd_signal is not None

    def test_volume_data_populated(self):
        df = make_ohlcv(n=250, base_volume=2_000_000)
        snap = compute_indicators(df, ticker="VOL")
        assert snap.current_volume is not None
        assert snap.avg_volume is not None
        assert snap.current_volume > 0
        assert snap.avg_volume > 0


# ════════════════════════════════════════════════════════════
# SCAN TICKER (FULL PIPELINE WITH MOCK DATA)
# ════════════════════════════════════════════════════════════

class TestScanTicker:
    """Test the single-ticker scanning pipeline."""

    def test_scan_with_provided_df(self):
        df = make_ohlcv(n=250)
        signals = scan_ticker("TEST", df=df)
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, TradeSignal)
            assert sig.ticker == "TEST"

    def test_scan_empty_df_returns_empty(self):
        signals = scan_ticker("EMPTY", df=pd.DataFrame())
        assert signals == []

    def test_scan_short_df_returns_empty(self):
        df = make_ohlcv(n=30)
        signals = scan_ticker("SHORT", df=df)
        assert signals == []


# ════════════════════════════════════════════════════════════
# SCAN WATCHLIST
# ════════════════════════════════════════════════════════════

class TestScanWatchlist:
    """Test scanning a list of watchlist tickers (with mocks)."""

    def test_scan_watchlist_filters_by_confidence(self, monkeypatch):
        """Only signals above min_confidence should be returned."""
        # Monkey-patch scan_ticker to return controlled signals
        def mock_scan_ticker(ticker, **kwargs):
            if ticker == "HIGH":
                return [TradeSignal(
                    ticker="HIGH", signal_type="BUY", trade_type="DAY",
                    confidence=80.0, indicators={}, price=50.0,
                )]
            elif ticker == "LOW":
                return [TradeSignal(
                    ticker="LOW", signal_type="BUY", trade_type="DAY",
                    confidence=30.0, indicators={}, price=20.0,
                )]
            return []

        monkeypatch.setattr("modules.signals.scan_ticker", mock_scan_ticker)
        results = scan_watchlist(["HIGH", "LOW"], save_to_db=False)
        assert len(results) == 1
        assert results[0].ticker == "HIGH"

    def test_scan_watchlist_handles_errors(self, monkeypatch):
        """Errors on individual tickers should not crash the scan."""
        def mock_scan_ticker(ticker, **kwargs):
            if ticker == "FAIL":
                raise ValueError("yfinance exploded")
            return []

        monkeypatch.setattr("modules.signals.scan_ticker", mock_scan_ticker)
        results = scan_watchlist(["FAIL", "OK"], save_to_db=False)
        assert isinstance(results, list)


# ════════════════════════════════════════════════════════════
# DATABASE PERSISTENCE
# ════════════════════════════════════════════════════════════

class TestDatabasePersistence:
    """Test saving signals to the database."""

    def test_save_signals(self, in_memory_db):
        signal = TradeSignal(
            ticker="TEST", signal_type="BUY", trade_type="DAY",
            confidence=75.0, indicators={"rsi_oversold": {"rsi": 25}},
            price=50.0,
        )
        _save_signals([signal], was_traded=False)

        from modules.database import get_session, Signal as SignalRecord
        session = get_session()
        records = session.query(SignalRecord).all()
        assert len(records) == 1
        assert records[0].ticker == "TEST"
        assert records[0].signal_type == "BUY"
        assert records[0].confidence == 75.0
        session.close()

    def test_save_skipped_signal_with_reason(self, in_memory_db):
        signal = TradeSignal(
            ticker="SKIP", signal_type="BUY", trade_type="DAY",
            confidence=35.0, indicators={}, price=30.0,
        )
        _save_signals([signal], was_traded=False, reason_skipped="Below threshold")

        from modules.database import get_session, Signal as SignalRecord
        session = get_session()
        record = session.query(SignalRecord).first()
        assert record.reason_skipped == "Below threshold"
        assert record.was_traded is False
        session.close()

    def test_save_empty_list_no_error(self, in_memory_db):
        _save_signals([], was_traded=False)


# ════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_all_none_snapshot(self):
        snap = IndicatorSnapshot(ticker="NULL")
        signals = generate_signals(snap)
        assert signals == []

    def test_rsi_at_zero(self):
        snap = IndicatorSnapshot(ticker="X", rsi=0.0)
        result = detect_rsi_signal(snap)
        assert result is not None
        assert result["direction"] == "BUY"

    def test_rsi_at_100(self):
        snap = IndicatorSnapshot(ticker="X", rsi=100.0)
        result = detect_rsi_signal(snap)
        assert result is not None
        assert result["direction"] == "SELL"

    def test_negative_volume_handled(self):
        snap = IndicatorSnapshot(
            ticker="X",
            current_volume=-100,
            avg_volume=1_000_000,
        )
        result = detect_volume_spike(snap)
        assert result is None

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) is None

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) is None

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_normal(self):
        assert _safe_float(3.14) == 3.14

    def test_safe_float_string(self):
        assert _safe_float("not a number") is None

    def test_conflicting_buy_and_sell_both_generated(self):
        """Rare case: EMA says BUY but RSI says SELL — both signals generated."""
        snap = IndicatorSnapshot(
            ticker="CONFLICT",
            ema_fast_day=51.0, ema_slow_day=50.0,
            prev_ema_fast_day=49.0, prev_ema_slow_day=50.0,
            rsi=78.0,  # overbought → SELL
            current_price=51.0,
        )
        signals = generate_signals(snap)
        types = {s.signal_type for s in signals}
        assert "BUY" in types
        assert "SELL" in types

    def test_very_large_volume_ratio(self):
        snap = IndicatorSnapshot(
            ticker="X",
            current_volume=100_000_000,
            avg_volume=1_000_000,
        )
        result = detect_volume_spike(snap)
        assert result is not None
        assert result["volume_ratio"] == 100.0


# ════════════════════════════════════════════════════════════
# INTEGRATION TEST (hits real API — skip in CI)
# ════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that hit real yfinance API.
    Run with: python -m pytest tests/test_signals.py -v -m integration
    Skip with: python -m pytest tests/test_signals.py -v -m "not integration"
    """

    def test_fetch_and_compute_real_stock(self):
        """Fetch AAPL history and compute indicators."""
        from modules.signals import fetch_price_history
        df = fetch_price_history("AAPL")
        assert df is not None and not df.empty
        snap = compute_indicators(df, ticker="AAPL")
        assert snap is not None
        assert snap.rsi is not None
        assert snap.ema_fast_day is not None

    def test_scan_real_ticker(self):
        """Full scan pipeline on a real ticker."""
        signals = scan_ticker("MSFT")
        assert isinstance(signals, list)
        for sig in signals:
            assert sig.ticker == "MSFT"
            assert sig.signal_type in ("BUY", "SELL")
            assert 0 <= sig.confidence <= 100

    def test_scanner_to_signals_pipeline(self, in_memory_db):
        """Integration: scanner watchlist → signal scan pipeline."""
        from modules.scanner import get_active_watchlist, _save_watchlist, ScoredStock, FundamentalData

        # Simulate a scanner result
        mock_scored = ScoredStock(
            data=FundamentalData(
                ticker="AAPL", company_name="Apple Inc", sector="Technology",
                market_cap=3e12, price=250,
            ),
            lynch_score=80, burry_score=75, composite_score=77.5,
            lynch_category="stalwart",
        )
        _save_watchlist([mock_scored])

        # Get watchlist and scan for signals
        watchlist = get_active_watchlist()
        assert len(watchlist) >= 1
        tickers = [w["ticker"] for w in watchlist]
        signals = scan_watchlist(tickers, save_to_db=True)
        assert isinstance(signals, list)


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_signals.py -v
#
# Expected results:
#   TestEMACrossoverDay:        6 tests  — all should PASS
#   TestEMACrossoverSwing:      4 tests  — all should PASS
#   TestRSISignal:              7 tests  — all should PASS
#   TestMACDCrossover:          5 tests  — all should PASS
#   TestVolumeSpike:            5 tests  — all should PASS
#   TestConfidenceScoring:     10 tests  — all should PASS
#   TestTradeTypeClassification: 4 tests — all should PASS
#   TestDirectionDetermination:  6 tests — all should PASS
#   TestSignalGeneration:        8 tests — all should PASS
#   TestComputeIndicators:       7 tests — all should PASS
#   TestScanTicker:              3 tests — all should PASS
#   TestScanWatchlist:           2 tests — all should PASS
#   TestDatabasePersistence:     3 tests — all should PASS
#   TestEdgeCases:              12 tests — all should PASS
#   TestIntegration:             3 tests — may skip if no network
#
# TOTAL: 85 tests
#
# If any test in TestEMACrossoverDay, TestRSISignal, TestMACDCrossover,
# TestConfidenceScoring, or TestSignalGeneration fails, DO NOT proceed
# to Module 2.5. Fix the signal logic first.
# ════════════════════════════════════════════════════════════
