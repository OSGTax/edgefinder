"""Tests for edgefinder/data/provider.py (CachedDataProvider)."""

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from edgefinder.core.models import TickerFundamentals
from edgefinder.data.cache import DataCache
from edgefinder.data.provider import CachedDataProvider


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def cache(tmp_path):
    return DataCache(cache_dir=tmp_path / "cache")


@pytest.fixture
def cached_provider(mock_provider, cache):
    return CachedDataProvider(provider=mock_provider, cache=cache)


class TestGetBars:
    def test_cache_hit_skips_provider(self, cached_provider, cache, mock_provider):
        # Pre-populate cache
        df = pd.DataFrame(
            {"open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6]},
            index=pd.DatetimeIndex([datetime(2024, 1, 15)], name="timestamp"),
        )
        cache.store_bars("AAPL", "day", df)

        result = cached_provider.get_bars("AAPL", "day", date(2024, 1, 15))
        assert result is not None
        assert len(result) == 1
        mock_provider.get_bars.assert_not_called()

    def test_cache_miss_calls_provider(self, cached_provider, mock_provider):
        df = pd.DataFrame(
            {"open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0], "volume": [1e6]},
            index=pd.DatetimeIndex([datetime(2024, 1, 15)], name="timestamp"),
        )
        mock_provider.get_bars.return_value = df

        result = cached_provider.get_bars("AAPL", "day", date(2024, 1, 15))
        assert result is not None
        mock_provider.get_bars.assert_called_once()

    def test_provider_none_not_cached(self, cached_provider, mock_provider, cache):
        mock_provider.get_bars.return_value = None
        result = cached_provider.get_bars("AAPL", "day", date(2024, 1, 15))
        assert result is None
        # Verify nothing was stored
        assert cache.get_bars("AAPL", "day", date(2024, 1, 15)) is None


class TestGetLatestPrice:
    def test_calls_provider_on_miss(self, cached_provider, mock_provider):
        from edgefinder.data import provider as provider_mod
        provider_mod._PRICE_CACHE.clear()
        mock_provider.get_latest_price.return_value = 155.50
        assert cached_provider.get_latest_price("AAPL") == 155.50
        mock_provider.get_latest_price.assert_called_once_with("AAPL")

    def test_returns_cached_on_hit(self, cached_provider, mock_provider):
        from edgefinder.data import provider as provider_mod
        provider_mod._PRICE_CACHE.clear()
        mock_provider.get_latest_price.return_value = 155.50
        cached_provider.get_latest_price("AAPL")
        cached_provider.get_latest_price("AAPL")
        # Should only call provider once — second call hits cache
        mock_provider.get_latest_price.assert_called_once()


class TestGetFundamentals:
    def test_cache_hit(self, cached_provider, cache, mock_provider):
        fund = TickerFundamentals(symbol="GOOG", peg_ratio=1.2)
        cache.store_fundamentals("GOOG", fund)

        result = cached_provider.get_fundamentals("GOOG")
        assert result.peg_ratio == 1.2
        mock_provider.get_fundamentals.assert_not_called()

    def test_cache_miss(self, cached_provider, mock_provider):
        fund = TickerFundamentals(symbol="GOOG", peg_ratio=1.5)
        mock_provider.get_fundamentals.return_value = fund

        result = cached_provider.get_fundamentals("GOOG")
        assert result.peg_ratio == 1.5
        mock_provider.get_fundamentals.assert_called_once()


class TestGetTickerUniverse:
    def test_cache_hit(self, cached_provider, cache, mock_provider):
        cache.store_universe(["AAPL", "MSFT"])
        result = cached_provider.get_ticker_universe()
        assert result == ["AAPL", "MSFT"]
        mock_provider.get_ticker_universe.assert_not_called()

    def test_cache_miss(self, cached_provider, mock_provider):
        mock_provider.get_ticker_universe.return_value = ["AAPL"]
        result = cached_provider.get_ticker_universe()
        assert result == ["AAPL"]


class TestIsMarketOpen:
    def test_delegates_to_provider(self, cached_provider, mock_provider):
        mock_provider.is_market_open.return_value = True
        assert cached_provider.is_market_open() is True
