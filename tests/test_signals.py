"""Tests for edgefinder/signals/engine.py."""

import numpy as np
import pandas as pd
import pytest

from edgefinder.signals.engine import (
    IndicatorSnapshot,
    _atr,
    _bollinger_bands,
    _ema,
    _macd,
    _rsi,
    compute_indicators,
    detect_signals,
)


def _make_ohlcv(n: int = 250, start_price: float = 100.0, trend: float = 0.0, volatility: float = 1.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    returns = np.random.normal(trend, volatility * 0.01, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.002, n))
    volume = np.random.randint(500_000, 2_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


class TestEMA:
    def test_ema_basic(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = _ema(s, 3)
        assert len(result) == 10
        assert result.iloc[-1] > result.iloc[0]

    def test_ema_fast_reacts_faster(self):
        s = pd.Series(range(1, 51), dtype=float)
        fast = _ema(s, 9)
        slow = _ema(s, 21)
        assert fast.iloc[-1] > slow.iloc[-1]


class TestRSI:
    def test_all_gains(self):
        s = pd.Series(range(100, 150), dtype=float)
        result = _rsi(s, 14)
        assert result.iloc[-1] > 90

    def test_all_losses(self):
        s = pd.Series(range(150, 100, -1), dtype=float)
        result = _rsi(s, 14)
        assert result.iloc[-1] < 10

    def test_mixed_typical_range(self):
        np.random.seed(42)
        s = pd.Series(100 + np.random.normal(0, 1, 100).cumsum())
        result = _rsi(s, 14)
        last = result.iloc[-1]
        assert 10 < last < 90


class TestMACD:
    def test_bullish_histogram(self):
        s = pd.Series(range(1, 51), dtype=float)
        line, signal, hist = _macd(s, 12, 26, 9)
        assert hist.iloc[-1] > 0

    def test_returns_three_series(self):
        s = pd.Series(range(1, 51), dtype=float)
        result = _macd(s, 12, 26, 9)
        assert len(result) == 3


class TestBollingerBands:
    def test_bands_contain_price(self):
        np.random.seed(42)
        close = pd.Series(100 + np.random.normal(0, 1, 200).cumsum())
        upper, middle, lower = _bollinger_bands(close, 20, 2.0)
        valid = close.iloc[20:]
        within = ((close >= lower) & (close <= upper)).iloc[20:]
        pct_within = within.mean()
        assert pct_within > 0.90


class TestATR:
    def test_calculation(self):
        n = 100
        high = pd.Series(np.full(n, 105.0))
        low = pd.Series(np.full(n, 95.0))
        close = pd.Series(np.full(n, 100.0))
        result = _atr(high, low, close, 14)
        assert abs(result.iloc[-1] - 10.0) < 1.0


class TestComputeIndicators:
    def test_returns_snapshot(self):
        df = _make_ohlcv(250)
        result = compute_indicators(df)
        assert result is not None
        assert isinstance(result, IndicatorSnapshot)
        assert result.close > 0
        assert result.rsi is not None
        assert result.atr is not None

    def test_insufficient_data_returns_none(self):
        df = _make_ohlcv(10)
        result = compute_indicators(df)
        assert result is None

    def test_to_dict(self):
        df = _make_ohlcv(250)
        result = compute_indicators(df)
        d = result.to_dict()
        assert "close" in d
        assert "rsi" in d


class TestDetectSignals:
    def test_ema_crossover_bullish(self):
        """Craft data where fast EMA crosses above slow."""
        ind = IndicatorSnapshot(
            close=100.0,
            prev_close=99.0,
            ema_fast_day=50.1,
            ema_slow_day=50.0,
            prev_ema_fast_day=49.9,
            prev_ema_slow_day=50.0,
            atr=2.0,
            recent_low=97.0,
            recent_high=103.0,
        )
        signals = detect_signals(ind, "TEST")
        buy_signals = [s for s in signals if s.metadata.get("pattern") == "ema_crossover_bullish"]
        assert len(buy_signals) == 1
        assert buy_signals[0].action.value == "BUY"

    def test_ema_crossover_bearish(self):
        ind = IndicatorSnapshot(
            close=100.0,
            prev_close=101.0,
            ema_fast_day=49.9,
            ema_slow_day=50.0,
            prev_ema_fast_day=50.1,
            prev_ema_slow_day=50.0,
            atr=2.0,
            recent_low=97.0,
            recent_high=103.0,
        )
        signals = detect_signals(ind, "TEST")
        sell_signals = [s for s in signals if s.metadata.get("pattern") == "ema_crossover_bearish"]
        assert len(sell_signals) == 1

    def test_no_crossover_no_signal(self):
        """Flat EMAs — no crossover detected."""
        ind = IndicatorSnapshot(
            close=100.0,
            prev_close=100.0,
            ema_fast_day=50.5,
            ema_slow_day=50.0,
            prev_ema_fast_day=50.3,
            prev_ema_slow_day=50.0,
            atr=2.0,
        )
        signals = detect_signals(ind, "TEST")
        crossover_signals = [s for s in signals if "crossover" in s.metadata.get("pattern", "")]
        assert len(crossover_signals) == 0

    def test_rsi_oversold(self):
        ind = IndicatorSnapshot(
            close=100.0,
            rsi=25.0,
            atr=2.0,
            recent_low=97.0,
        )
        signals = detect_signals(ind, "TEST")
        rsi_signals = [s for s in signals if s.metadata.get("pattern") == "rsi_oversold"]
        assert len(rsi_signals) == 1

    def test_rsi_overbought(self):
        ind = IndicatorSnapshot(
            close=100.0,
            rsi=75.0,
            atr=2.0,
            recent_high=103.0,
        )
        signals = detect_signals(ind, "TEST")
        rsi_signals = [s for s in signals if s.metadata.get("pattern") == "rsi_overbought"]
        assert len(rsi_signals) == 1

    def test_macd_bullish_cross(self):
        ind = IndicatorSnapshot(
            close=100.0,
            macd_histogram=0.1,
            prev_macd_histogram=-0.1,
            macd_line=0.5,
            atr=2.0,
            recent_low=97.0,
        )
        signals = detect_signals(ind, "TEST")
        macd_signals = [s for s in signals if s.metadata.get("pattern") == "macd_bullish_cross"]
        assert len(macd_signals) == 1

    def test_bb_lower_touch(self):
        ind = IndicatorSnapshot(
            close=95.0,
            bb_lower=96.0,
            bb_middle=100.0,
            bb_upper=104.0,
            atr=2.0,
            recent_low=94.0,
        )
        signals = detect_signals(ind, "TEST")
        bb_signals = [s for s in signals if s.metadata.get("pattern") == "bb_lower_touch"]
        assert len(bb_signals) == 1

    def test_volume_spike_bullish(self):
        ind = IndicatorSnapshot(
            close=101.0,
            prev_close=100.0,
            volume_ratio=2.0,
            atr=2.0,
            recent_low=97.0,
        )
        signals = detect_signals(ind, "TEST")
        vol_signals = [s for s in signals if s.metadata.get("pattern") == "volume_spike_bullish"]
        assert len(vol_signals) == 1

    def test_confidence_boosted_by_volume(self):
        """Signal with volume spike should have higher confidence."""
        base = IndicatorSnapshot(
            close=100.0,
            rsi=25.0,
            atr=2.0,
            recent_low=97.0,
            volume_ratio=0.5,
        )
        boosted = IndicatorSnapshot(
            close=100.0,
            rsi=25.0,
            atr=2.0,
            recent_low=97.0,
            volume_ratio=2.0,
        )
        base_signals = detect_signals(base, "TEST")
        boost_signals = detect_signals(boosted, "TEST")
        base_rsi = [s for s in base_signals if s.metadata.get("pattern") == "rsi_oversold"]
        boost_rsi = [s for s in boost_signals if s.metadata.get("pattern") == "rsi_oversold"]
        assert len(base_rsi) == 1 and len(boost_rsi) == 1
        assert boost_rsi[0].confidence > base_rsi[0].confidence

    def test_signal_entry_stop_target_math(self):
        ind = IndicatorSnapshot(
            close=100.0,
            rsi=25.0,
            atr=2.0,
            recent_low=97.0,
        )
        signals = detect_signals(ind, "TEST")
        for sig in signals:
            if sig.action.value == "BUY":
                assert sig.stop_loss < sig.entry_price
                assert sig.target > sig.entry_price
                risk = sig.entry_price - sig.stop_loss
                if risk > 0:
                    assert sig.reward_to_risk >= 1.5
