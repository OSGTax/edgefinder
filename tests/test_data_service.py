"""
Tests for the Data Service Layer
=================================
All external API calls are mocked — no keys required to run these tests.

Run:
    python -m pytest tests/test_data_service.py -v
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.cache import DataCache, CachedBar, CachedFundamental
from services.alpaca_client import AlpacaClient
from services.fmp_client import FMPClient
from services.data_service import DataService


# ── FIXTURES ────────────────────────────────────────────────

@pytest.fixture
def cache():
    """In-memory cache for testing."""
    return DataCache(cache_path=":memory:")


@pytest.fixture
def sample_bars_df():
    """Sample OHLCV DataFrame."""
    dates = pd.date_range("2026-03-17", periods=5, freq="B")
    return pd.DataFrame({
        "open": [150.0, 151.0, 152.0, 149.0, 153.0],
        "high": [152.0, 153.0, 154.0, 151.0, 155.0],
        "low": [149.0, 150.0, 151.0, 148.0, 152.0],
        "close": [151.0, 152.0, 153.0, 150.0, 154.0],
        "volume": [1000000, 1100000, 900000, 1200000, 1050000],
        "vwap": [150.5, 151.5, 152.5, 149.5, 153.5],
        "trade_count": [5000, 5500, 4500, 6000, 5250],
    }, index=dates)


@pytest.fixture
def sample_alpaca_bars_response():
    """Raw Alpaca API response for bars."""
    return {
        "bars": [
            {
                "t": "2026-03-17T05:00:00Z",
                "o": 150.0, "h": 152.0, "l": 149.0, "c": 151.0,
                "v": 1000000, "vw": 150.5, "n": 5000,
            },
            {
                "t": "2026-03-18T05:00:00Z",
                "o": 151.0, "h": 153.0, "l": 150.0, "c": 152.0,
                "v": 1100000, "vw": 151.5, "n": 5500,
            },
        ],
        "next_page_token": None,
    }


@pytest.fixture
def sample_fmp_profile():
    """Sample FMP company profile response."""
    return [{
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "mktCap": 2800000000000,
        "price": 178.50,
        "volAvg": 55000000,
        "description": "Apple designs and sells consumer electronics.",
        "exchangeShortName": "NASDAQ",
    }]


# ── CACHE TESTS ─────────────────────────────────────────────

class TestDataCache:
    """Tests for the SQLite cache layer."""

    def test_cache_initializes(self, cache):
        """Cache should initialize without errors."""
        stats = cache.get_stats()
        assert stats["cached_bars"] == 0
        assert stats["cached_fundamentals"] == 0

    def test_store_and_retrieve_bars(self, cache, sample_bars_df):
        """Bars should round-trip through the cache."""
        cache.store_bars("AAPL", "1Day", sample_bars_df)
        retrieved = cache.get_bars("AAPL", "1Day")

        assert retrieved is not None
        assert len(retrieved) == 5
        assert retrieved.iloc[0]["close"] == 151.0
        assert retrieved.iloc[-1]["close"] == 154.0

    def test_cache_miss_returns_none(self, cache):
        """Missing data should return None."""
        result = cache.get_bars("MSFT", "1Day")
        assert result is None

    def test_store_and_retrieve_fundamental(self, cache):
        """Fundamental data should round-trip through the cache."""
        data = {"symbol": "AAPL", "sector": "Technology", "mktCap": 2800000000000}
        cache.store_fundamental("AAPL", "profile", data)
        retrieved = cache.get_fundamental("AAPL", "profile")

        assert retrieved is not None
        assert retrieved["symbol"] == "AAPL"
        assert retrieved["mktCap"] == 2800000000000

    def test_fundamental_upsert(self, cache):
        """Storing same ticker/type should overwrite."""
        cache.store_fundamental("AAPL", "profile", {"price": 150})
        cache.store_fundamental("AAPL", "profile", {"price": 175})
        retrieved = cache.get_fundamental("AAPL", "profile")
        assert retrieved["price"] == 175

    def test_bar_upsert(self, cache, sample_bars_df):
        """Storing bars for same ticker/timeframe/timestamp should overwrite."""
        cache.store_bars("AAPL", "1Day", sample_bars_df)

        # Modify one bar
        updated = sample_bars_df.copy()
        updated.iloc[0, updated.columns.get_loc("close")] = 999.0
        cache.store_bars("AAPL", "1Day", updated)

        retrieved = cache.get_bars("AAPL", "1Day")
        assert retrieved.iloc[0]["close"] == 999.0
        # Should still be 5 bars, not 10
        assert len(retrieved) == 5

    def test_clear_ticker(self, cache, sample_bars_df):
        """Clearing a ticker removes all its data."""
        cache.store_bars("AAPL", "1Day", sample_bars_df)
        cache.store_fundamental("AAPL", "profile", {"symbol": "AAPL"})

        deleted = cache.clear_ticker("AAPL")
        assert deleted == 6  # 5 bars + 1 fundamental

        assert cache.get_bars("AAPL", "1Day") is None
        assert cache.get_fundamental("AAPL", "profile") is None

    def test_cache_stats(self, cache, sample_bars_df):
        """Stats should reflect stored data."""
        cache.store_bars("AAPL", "1Day", sample_bars_df)
        cache.store_bars("MSFT", "1Day", sample_bars_df)
        cache.store_fundamental("AAPL", "profile", {"symbol": "AAPL"})

        stats = cache.get_stats()
        assert stats["cached_bars"] == 10
        assert stats["bar_tickers"] == 2
        assert stats["cached_fundamentals"] == 1
        assert stats["fundamental_tickers"] == 1

    def test_clear_all(self, cache, sample_bars_df):
        """Clear all should empty the cache."""
        cache.store_bars("AAPL", "1Day", sample_bars_df)
        cache.store_fundamental("AAPL", "profile", {"symbol": "AAPL"})

        deleted = cache.clear_all()
        assert deleted == 6

        stats = cache.get_stats()
        assert stats["cached_bars"] == 0
        assert stats["cached_fundamentals"] == 0

    def test_date_range_filter(self, cache, sample_bars_df):
        """Bars should filter by date range."""
        sample_bars_df.index.name = "timestamp"
        cache.store_bars("AAPL", "1Day", sample_bars_df)

        # Get only first 3 days
        start = datetime(2026, 3, 17)
        end = datetime(2026, 3, 19)
        retrieved = cache.get_bars("AAPL", "1Day", start=start, end=end)

        assert retrieved is not None
        assert len(retrieved) == 3


# ── ALPACA CLIENT TESTS ─────────────────────────────────────

class TestAlpacaClient:
    """Tests for the Alpaca API client (all mocked)."""

    def test_requires_credentials(self):
        """Should raise ValueError without credentials."""
        with pytest.raises(ValueError):
            AlpacaClient("", "")

    @patch("services.alpaca_client.requests.request")
    def test_get_bars(self, mock_request, sample_alpaca_bars_response):
        """Should parse Alpaca bar response into DataFrame."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_alpaca_bars_response
        mock_resp.headers = {}
        mock_request.return_value = mock_resp

        client = AlpacaClient("test_key", "test_secret")
        df = client.get_bars("AAPL", timeframe="1Day")

        assert df is not None
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns
        assert df.iloc[0]["close"] == 151.0

    @patch("services.alpaca_client.requests.request")
    def test_get_account(self, mock_request):
        """Should return account info dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ACTIVE",
            "buying_power": "100000",
            "equity": "100000",
        }
        mock_resp.headers = {}
        mock_request.return_value = mock_resp

        client = AlpacaClient("test_key", "test_secret")
        account = client.get_account()

        assert account is not None
        assert account["status"] == "ACTIVE"

    @patch("services.alpaca_client.requests.request")
    def test_handles_api_error(self, mock_request):
        """Should return None on API error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_resp.headers = {}
        mock_request.return_value = mock_resp

        client = AlpacaClient("test_key", "test_secret")
        result = client.get_bars("AAPL")

        assert result is None

    @patch("services.alpaca_client.requests.request")
    def test_get_latest_quote(self, mock_request):
        """Should parse latest quote response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "quote": {"bp": 178.40, "ap": 178.50, "bs": 100, "as": 200, "t": "2026-03-24T15:00:00Z"}
        }
        mock_resp.headers = {}
        mock_request.return_value = mock_resp

        client = AlpacaClient("test_key", "test_secret")
        quote = client.get_latest_quote("AAPL")

        assert quote is not None
        assert quote["bid"] == 178.40
        assert quote["ask"] == 178.50


# ── FMP CLIENT TESTS ────────────────────────────────────────

class TestFMPClient:
    """Tests for the FMP API client (all mocked)."""

    def test_requires_api_key(self):
        """Should raise ValueError without API key."""
        with pytest.raises(ValueError):
            FMPClient("")

    @patch("services.fmp_client.requests.get")
    def test_get_profile(self, mock_get, sample_fmp_profile):
        """Should parse FMP profile response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_fmp_profile
        mock_get.return_value = mock_resp

        client = FMPClient("test_key")
        profile = client.get_profile("AAPL")

        assert profile is not None
        assert profile["companyName"] == "Apple Inc."
        assert profile["sector"] == "Technology"

    @patch("services.fmp_client.requests.get")
    def test_requests_remaining(self, mock_get):
        """Should track daily request count."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"symbol": "AAPL"}]
        mock_get.return_value = mock_resp

        client = FMPClient("test_key")
        assert client.requests_remaining == 250

        client.get_profile("AAPL")
        assert client.requests_remaining == 249

    @patch("services.fmp_client.requests.get")
    def test_daily_limit_enforcement(self, mock_get):
        """Should stop making requests at daily limit."""
        client = FMPClient("test_key")
        client._daily_requests = 250  # At limit

        result = client.get_profile("AAPL")
        assert result is None
        mock_get.assert_not_called()  # Should not have made a request


# ── DATA SERVICE TESTS ──────────────────────────────────────

class TestDataService:
    """Tests for the unified data service."""

    @patch("services.data_service.load_dotenv")
    def test_initializes_without_keys(self, mock_dotenv):
        """Should initialize gracefully without API keys (yfinance only)."""
        with patch.dict("os.environ", {}, clear=True):
            ds = DataService(cache_path=":memory:")
            assert ds.alpaca is None
            assert ds.fmp is None
            assert ds.available_sources["yfinance"] is True

    def test_initializes_with_keys(self):
        """Should initialize all clients when keys provided."""
        ds = DataService(
            alpaca_key="test", alpaca_secret="test",
            fmp_key="test", cache_path=":memory:"
        )
        assert ds.alpaca is not None
        assert ds.fmp is not None

    def test_cache_first_lookup(self, sample_bars_df):
        """Should return cached data without hitting APIs."""
        ds = DataService(cache_path=":memory:")
        ds.cache.store_bars("AAPL", "1Day", sample_bars_df)

        result = ds.get_bars("AAPL", timeframe="1Day", days_back=10)
        assert result is not None
        assert len(result) == 5

    @patch("services.data_service.DataService._yfinance_bars")
    def test_yfinance_fallback(self, mock_yf, sample_bars_df):
        """Should fall back to yfinance when Alpaca not configured."""
        mock_yf.return_value = sample_bars_df

        ds = DataService(cache_path=":memory:")
        result = ds.get_bars("AAPL", timeframe="1Day", days_back=5, use_cache=False)

        assert result is not None
        assert len(result) == 5
        mock_yf.assert_called_once()

    @patch("services.data_service.DataService._yfinance_bars")
    def test_bars_cached_after_fetch(self, mock_yf, sample_bars_df):
        """Fetched bars should be stored in cache."""
        mock_yf.return_value = sample_bars_df

        ds = DataService(cache_path=":memory:")
        ds.get_bars("AAPL", timeframe="1Day", days_back=5, use_cache=False)

        # Second call should hit cache
        mock_yf.reset_mock()
        result = ds.get_bars("AAPL", timeframe="1Day", days_back=10)
        assert result is not None
        # yfinance should NOT have been called again
        mock_yf.assert_not_called()

    @patch("services.data_service.DataService._yfinance_profile")
    def test_profile_fallback(self, mock_yf):
        """Should fall back to yfinance for profiles."""
        mock_yf.return_value = {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "sector": "Technology",
        }

        ds = DataService(cache_path=":memory:")
        profile = ds.get_profile("AAPL")

        assert profile is not None
        assert profile["companyName"] == "Apple Inc."

    @patch("services.data_service.DataService._yfinance_fundamentals")
    def test_fundamentals_combined(self, mock_yf):
        """Should return combined fundamental data."""
        mock_yf.return_value = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "market_cap": 2800000000000,
            "peg_ratio": 1.2,
            "debt_to_equity": 0.45,
        }

        ds = DataService(cache_path=":memory:")
        data = ds.get_fundamentals("AAPL")

        assert data is not None
        assert data["ticker"] == "AAPL"
        assert data["peg_ratio"] == 1.2

    def test_cache_management(self, sample_bars_df):
        """Cache management methods should work."""
        ds = DataService(cache_path=":memory:")
        ds.cache.store_bars("AAPL", "1Day", sample_bars_df)

        stats = ds.get_cache_stats()
        assert stats["cached_bars"] == 5

        ds.clear_cache("AAPL")
        stats = ds.get_cache_stats()
        assert stats["cached_bars"] == 0

    def test_get_latest_price_from_cache(self, sample_bars_df):
        """Should get latest price from cached bars as last resort."""
        ds = DataService(cache_path=":memory:")
        ds.alpaca = None  # Ensure no Alpaca client
        ds.cache.store_bars("AAPL", "1Day", sample_bars_df)

        # With no API clients, should fall back to cache
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.fast_info.get.return_value = None
            mock_ticker_cls.return_value = mock_ticker
            price = ds.get_latest_price("AAPL")
            # Should get last cached close (154.0)
            assert price == 154.0


# ── INTEGRATION TEST (hits real APIs — skipped in CI) ───────

@pytest.mark.integration
class TestDataServiceIntegration:
    """Integration tests that hit real APIs. Run with: pytest -m integration"""

    def test_yfinance_bars_real(self):
        """Fetch real bars from yfinance."""
        ds = DataService(cache_path=":memory:")
        bars = ds.get_bars("AAPL", timeframe="1Day", days_back=5)
        assert bars is not None
        assert len(bars) > 0
        assert "close" in bars.columns

    def test_yfinance_profile_real(self):
        """Fetch real profile from yfinance."""
        ds = DataService(cache_path=":memory:")
        profile = ds.get_profile("AAPL")
        assert profile is not None
        assert "AAPL" in str(profile.get("symbol", "")) or "Apple" in str(profile.get("companyName", ""))

    def test_multi_bars_real(self):
        """Fetch bars for multiple tickers."""
        ds = DataService(cache_path=":memory:")
        result = ds.get_multi_bars(["AAPL", "MSFT", "GOOGL"], days_back=5)
        assert len(result) > 0
