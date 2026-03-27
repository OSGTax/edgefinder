"""Tests for edgefinder/data/cache.py."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from edgefinder.core.models import TickerFundamentals
from edgefinder.data.cache import DataCache


@pytest.fixture
def cache(tmp_path):
    return DataCache(cache_dir=tmp_path / "cache")


class TestBarsCache:
    def test_store_and_retrieve(self, cache):
        df = pd.DataFrame(
            {"open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6]},
            index=pd.DatetimeIndex([datetime(2024, 1, 15)], name="timestamp"),
        )
        cache.store_bars("AAPL", "day", df)
        result = cache.get_bars("AAPL", "day", date(2024, 1, 15))
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["close"] == 103.0

    def test_cache_miss(self, cache):
        result = cache.get_bars("AAPL", "day", date(2024, 1, 15))
        assert result is None

    def test_expired_returns_none(self, cache, tmp_path):
        df = pd.DataFrame(
            {"open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6]},
            index=pd.DatetimeIndex([datetime(2024, 1, 15)], name="timestamp"),
        )
        cache.store_bars("AAPL", "day", df)
        # Manually set mtime to far in the past
        path = tmp_path / "cache" / "bars" / "AAPL" / "day.parquet"
        import os
        old_time = (datetime.now() - timedelta(hours=48)).timestamp()
        os.utime(path, (old_time, old_time))
        result = cache.get_bars("AAPL", "day", date(2024, 1, 15))
        assert result is None


class TestFundamentalsCache:
    def test_store_and_retrieve(self, cache):
        fund = TickerFundamentals(symbol="GOOG", company_name="Alphabet", peg_ratio=1.5)
        cache.store_fundamentals("GOOG", fund)
        result = cache.get_fundamentals("GOOG")
        assert result is not None
        assert result.symbol == "GOOG"
        assert result.peg_ratio == 1.5

    def test_cache_miss(self, cache):
        result = cache.get_fundamentals("NONEXIST")
        assert result is None


class TestUniverseCache:
    def test_store_and_retrieve(self, cache):
        tickers = ["AAPL", "MSFT", "GOOG"]
        cache.store_universe(tickers)
        result = cache.get_universe()
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_cache_miss(self, cache):
        result = cache.get_universe()
        assert result is None


class TestClear:
    def test_clear_removes_all(self, cache):
        fund = TickerFundamentals(symbol="AAPL")
        cache.store_fundamentals("AAPL", fund)
        cache.store_universe(["AAPL"])
        cache.clear()
        assert cache.get_fundamentals("AAPL") is None
        assert cache.get_universe() is None
