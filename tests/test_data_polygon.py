"""Tests for edgefinder/data/polygon.py (mocked HTTP)."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from edgefinder.data.polygon import PolygonDataProvider


@pytest.fixture
def provider():
    """PolygonDataProvider with a mocked RESTClient."""
    with patch("edgefinder.data.polygon.RESTClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        p = PolygonDataProvider(api_key="test_key")
        p._client = mock_client
        yield p, mock_client


class TestGetBars:
    def test_returns_dataframe(self, provider):
        p, client = provider
        mock_agg = MagicMock()
        mock_agg.timestamp = int(datetime(2024, 1, 15, 10, 0).timestamp() * 1000)
        mock_agg.open = 150.0
        mock_agg.high = 155.0
        mock_agg.low = 149.0
        mock_agg.close = 153.0
        mock_agg.volume = 1_000_000
        mock_agg.vwap = 152.0
        mock_agg.transactions = 5000

        client.get_aggs.return_value = [mock_agg]

        df = p.get_bars("AAPL", "day", date(2024, 1, 15))
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["close"] == 153.0
        assert df.iloc[0]["vwap"] == 152.0

    def test_empty_response_returns_none(self, provider):
        p, client = provider
        client.get_aggs.return_value = []
        df = p.get_bars("AAPL", "day", date(2024, 1, 15))
        assert df is None

    def test_api_error_returns_none(self, provider):
        p, client = provider
        client.get_aggs.side_effect = Exception("API error")
        df = p.get_bars("AAPL", "day", date(2024, 1, 15))
        assert df is None


class TestGetLatestPrice:
    def test_returns_day_close(self, provider):
        p, client = provider
        snapshot = MagicMock()
        snapshot.day = MagicMock()
        snapshot.day.close = 155.50
        snapshot.prev_day = None
        client.get_snapshot_ticker.return_value = snapshot

        price = p.get_latest_price("AAPL")
        assert price == 155.50

    def test_falls_back_to_prev_day(self, provider):
        p, client = provider
        snapshot = MagicMock()
        snapshot.day = MagicMock()
        snapshot.day.close = None
        snapshot.prev_day = MagicMock()
        snapshot.prev_day.close = 154.00
        client.get_snapshot_ticker.return_value = snapshot

        price = p.get_latest_price("AAPL")
        assert price == 154.00

    def test_api_error_returns_none(self, provider):
        p, client = provider
        client.get_snapshot_ticker.side_effect = Exception("fail")
        assert p.get_latest_price("AAPL") is None


class TestGetFundamentals:
    def test_returns_fundamentals(self, provider):
        p, client = provider
        details = MagicMock()
        details.name = "Apple Inc."
        details.sic_description = "Technology"
        details.market_cap = 3_000_000_000_000
        details.sic_code = "3571"
        details.total_employees = 160000
        details.list_date = "1980-12-12"
        details.description = "Apple designs..."
        details.homepage_url = "https://apple.com"
        details.locale = "us"
        details.primary_exchange = "XNAS"
        client.get_ticker_details.return_value = details

        # Mock financials
        financials = MagicMock()
        financials.financials = MagicMock()
        financials.financials.income_statement = MagicMock()
        financials.financials.balance_sheet = MagicMock()
        financials.financials.cash_flow_statement = MagicMock()

        revenues = MagicMock()
        revenues.value = 380_000_000_000
        financials.financials.income_statement.revenues = revenues
        financials.financials.income_statement.net_income_loss = MagicMock(value=95_000_000_000)

        financials.financials.balance_sheet.assets = MagicMock(value=350_000_000_000)
        financials.financials.balance_sheet.liabilities = MagicMock(value=280_000_000_000)
        financials.financials.balance_sheet.equity = MagicMock(value=70_000_000_000)
        financials.financials.balance_sheet.current_assets = MagicMock(value=135_000_000_000)
        financials.financials.balance_sheet.current_liabilities = MagicMock(value=145_000_000_000)
        financials.financials.cash_flow_statement.net_cash_flow_from_operating_activities = MagicMock(value=110_000_000_000)

        client.vx.list_stock_financials.return_value = iter([financials])

        result = p.get_fundamentals("AAPL")
        assert result is not None
        assert result.company_name == "Apple Inc."
        assert result.market_cap == 3_000_000_000_000
        assert result.current_ratio is not None
        assert result.debt_to_equity is not None
        assert result.raw_data is not None

    def test_api_error_returns_basic(self, provider):
        p, client = provider
        client.get_ticker_details.side_effect = Exception("fail")
        client.vx.list_stock_financials.return_value = iter([])
        result = p.get_fundamentals("AAPL")
        assert result is not None
        assert result.symbol == "AAPL"


class TestGetTickerUniverse:
    def test_returns_tickers(self, provider):
        p, client = provider
        t1 = MagicMock()
        t1.ticker = "AAPL"
        t2 = MagicMock()
        t2.ticker = "MSFT"
        client.list_tickers.return_value = [t1, t2]

        result = p.get_ticker_universe()
        assert result == ["AAPL", "MSFT"]

    def test_api_error_returns_empty(self, provider):
        p, client = provider
        client.list_tickers.side_effect = Exception("fail")
        result = p.get_ticker_universe()
        assert result == []


class TestIsMarketOpen:
    def test_open(self, provider):
        p, client = provider
        status = MagicMock()
        status.market = "open"
        client.get_market_status.return_value = status
        assert p.is_market_open() is True

    def test_closed(self, provider):
        p, client = provider
        status = MagicMock()
        status.market = "closed"
        client.get_market_status.return_value = status
        assert p.is_market_open() is False

    def test_api_error(self, provider):
        p, client = provider
        client.get_market_status.side_effect = Exception("fail")
        assert p.is_market_open() is False


class TestParseTimeframe:
    def test_minute(self):
        assert PolygonDataProvider._parse_timeframe("1") == (1, "minute")
        assert PolygonDataProvider._parse_timeframe("5") == (5, "minute")
        assert PolygonDataProvider._parse_timeframe("15") == (15, "minute")

    def test_hour(self):
        assert PolygonDataProvider._parse_timeframe("60") == (1, "hour")

    def test_day(self):
        assert PolygonDataProvider._parse_timeframe("day") == (1, "day")

    def test_unknown_defaults_to_day(self):
        assert PolygonDataProvider._parse_timeframe("unknown") == (1, "day")
