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


def _make_details_mock():
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
    return details


def _make_financials_mock(
    revenues=380_000_000_000, net_income=95_000_000_000,
    assets=350_000_000_000, liabilities=280_000_000_000,
    equity=70_000_000_000, current_assets=135_000_000_000,
    current_liabilities=145_000_000_000, operating_cf=110_000_000_000,
):
    fin = MagicMock()
    fin.financials = MagicMock()
    fin.financials.income_statement = MagicMock()
    fin.financials.balance_sheet = MagicMock()
    fin.financials.cash_flow_statement = MagicMock()

    fin.financials.income_statement.revenues = MagicMock(value=revenues)
    fin.financials.income_statement.net_income_loss = MagicMock(value=net_income)

    fin.financials.balance_sheet.assets = MagicMock(value=assets)
    fin.financials.balance_sheet.liabilities = MagicMock(value=liabilities)
    fin.financials.balance_sheet.equity = MagicMock(value=equity)
    fin.financials.balance_sheet.current_assets = MagicMock(value=current_assets)
    fin.financials.balance_sheet.current_liabilities = MagicMock(value=current_liabilities)
    fin.financials.cash_flow_statement.net_cash_flow_from_operating_activities = MagicMock(value=operating_cf)
    return fin


class TestGetFundamentals:
    def test_returns_fundamentals(self, provider):
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()

        current = _make_financials_mock()
        previous = _make_financials_mock(
            revenues=340_000_000_000, net_income=80_000_000_000,
        )
        client.vx.list_stock_financials.return_value = [current, previous]

        result = p.get_fundamentals("AAPL")
        assert result is not None
        assert result.company_name == "Apple Inc."
        assert result.market_cap == 3_000_000_000_000
        assert result.current_ratio is not None
        assert result.debt_to_equity is not None
        assert result.raw_data is not None
        # Derived metrics from 2-period comparison
        assert result.earnings_growth == pytest.approx(
            (95e9 - 80e9) / 80e9, rel=1e-4
        )
        assert result.revenue_growth == pytest.approx(
            (380e9 - 340e9) / 340e9, rel=1e-4
        )
        assert result.fcf_yield == pytest.approx(110e9 / 3e12, rel=1e-4)
        assert result.price_to_tangible_book == pytest.approx(
            3e12 / (350e9 - 280e9), rel=1e-4
        )
        assert result.peg_ratio is not None

    def test_single_period_no_growth(self, provider):
        """With only 1 period, growth metrics are None but others still work."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.vx.list_stock_financials.return_value = [_make_financials_mock()]

        result = p.get_fundamentals("AAPL")
        assert result.earnings_growth is None
        assert result.revenue_growth is None
        # Single-period metrics still derived
        assert result.fcf_yield is not None
        assert result.price_to_tangible_book is not None
        assert result.current_ratio is not None

    def test_zero_denominator_safety(self, provider):
        """Division by zero in previous period should not crash."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()

        current = _make_financials_mock()
        previous = _make_financials_mock(revenues=0, net_income=0)
        client.vx.list_stock_financials.return_value = [current, previous]

        result = p.get_fundamentals("AAPL")
        assert result is not None
        assert result.earnings_growth is None  # prev_ni == 0 → skipped
        assert result.revenue_growth is None   # prev_rev == 0 → skipped

    def test_api_error_returns_basic(self, provider):
        p, client = provider
        client.get_ticker_details.side_effect = Exception("fail")
        client.vx.list_stock_financials.return_value = []
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
