"""Tests for the shared indicator computation engine."""

import numpy as np
import pandas as pd
import pytest

from edgefinder.data.indicator_engine import compute_indicators_from_bars
from edgefinder.data.market_data import IndicatorSnapshot


class TestComputeIndicators:
    @pytest.fixture
    def daily_bars(self):
        """Generate 60 days of realistic daily OHLCV bars."""
        np.random.seed(42)
        n = 60
        close = 100 + np.random.normal(0, 1.5, n).cumsum()
        df = pd.DataFrame({
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=pd.date_range("2024-01-01", periods=n, freq="B", name="timestamp"))
        return df

    def test_returns_indicator_snapshot(self, daily_bars):
        result = compute_indicators_from_bars(daily_bars)
        assert isinstance(result, IndicatorSnapshot)
        assert result.close > 0
        assert result.rsi is not None
        assert result.ema_9 is not None
        assert result.ema_21 is not None
        assert result.macd_line is not None
        assert result.bb_upper is not None
        assert result.atr is not None

    def test_returns_none_on_insufficient_data(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.0], "volume": [1000000.0],
        }, index=pd.date_range("2024-01-01", periods=1, freq="B", name="timestamp"))
        result = compute_indicators_from_bars(df)
        assert result is None

    def test_includes_ohlcv_in_snapshot(self, daily_bars):
        result = compute_indicators_from_bars(daily_bars)
        assert result.open > 0
        assert result.high > 0
        assert result.low > 0
        assert result.volume > 0
