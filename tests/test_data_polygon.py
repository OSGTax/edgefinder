"""Tests for edgefinder/data/polygon.py — Massive (formerly Polygon.io) provider (mocked HTTP)."""

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
        # Default all new endpoints to return empty/None so tests don't break
        mock_client.list_financials_ratios.return_value = []
        mock_client.list_benzinga_earnings.return_value = []
        mock_client.list_benzinga_consensus_ratings.return_value = []
        mock_client.list_short_interest.return_value = []
        mock_client.list_dividends.return_value = []
        mock_client.get_related_companies.return_value = []
        mock_client.list_ticker_news.return_value = []
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
    details.category = "Technology"
    details.cik = "0000320193"
    details.share_class_shares_outstanding = 15_000_000_000
    details.weighted_shares_outstanding = 15_200_000_000
    return details


def _make_ratios_mock(**overrides):
    r = MagicMock()
    defaults = {
        "price": 200.0,
        "market_cap": 3_000_000_000_000,
        "average_volume": 50_000_000,
        "earnings_per_share": 6.50,
        "price_to_earnings": 30.7,
        "price_to_book": 48.5,
        "price_to_sales": 7.8,
        "price_to_cash_flow": 25.2,
        "price_to_free_cash_flow": 28.1,
        "dividend_yield": 0.005,
        "return_on_assets": 0.28,
        "return_on_equity": 1.56,
        "debt_to_equity": 1.87,
        "current": 0.93,  # current ratio
        "quick": 0.85,
        "cash": 0.35,
        "ev_to_sales": 8.1,
        "ev_to_ebitda": 24.5,
        "enterprise_value": 3_100_000_000_000,
        "free_cash_flow": 110_000_000_000,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(r, k, v)
    r.__annotations__ = {k: type(v) for k, v in defaults.items()}
    return r


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


def _make_earnings_mock(ticker="AAPL"):
    past = MagicMock()
    past.date = "2024-01-25"
    past.ticker = ticker
    past.eps_surprise_percent = 5.2
    past.revenue_surprise_percent = 2.1
    past.estimated_eps = 2.10
    past.actual_eps = 2.18
    past.__annotations__ = {"date": str, "ticker": str, "eps_surprise_percent": float,
                            "revenue_surprise_percent": float, "estimated_eps": float, "actual_eps": float}

    future = MagicMock()
    future.date = "2099-04-28"  # far future so it's always "next"
    future.ticker = ticker
    future.estimated_eps = 2.35
    future.eps_surprise_percent = None
    future.revenue_surprise_percent = None
    future.__annotations__ = {"date": str, "ticker": str, "estimated_eps": float}
    return [past, future]


def _make_consensus_mock():
    c = MagicMock()
    c.strong_buy_ratings = 15
    c.buy_ratings = 20
    c.hold_ratings = 8
    c.sell_ratings = 2
    c.strong_sell_ratings = 1
    c.consensus_price_target = 225.0
    c.consensus_rating = "Buy"
    return c


class TestGetFundamentals:
    def test_returns_fundamentals_with_ratios(self, provider):
        """Full fundamentals with pre-computed ratios from Massive."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]

        current = _make_financials_mock()
        previous = _make_financials_mock(
            revenues=340_000_000_000, net_income=80_000_000_000,
        )
        client.vx.list_stock_financials.return_value = [current, previous]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result is not None
        assert result.company_name == "Apple Inc."
        assert result.market_cap == 3_000_000_000_000
        # Pre-computed ratios from Massive
        assert result.current_ratio == 0.93
        assert result.debt_to_equity == 1.87
        assert result.price_to_earnings == 30.7
        assert result.return_on_equity == 1.56
        assert result.ev_to_ebitda == 24.5
        assert result.free_cash_flow == 110_000_000_000
        assert result.dividend_yield == 0.005
        # Growth from raw financials
        assert result.earnings_growth == pytest.approx(
            (95e9 - 80e9) / 80e9, rel=1e-4
        )
        assert result.revenue_growth == pytest.approx(
            (380e9 - 340e9) / 340e9, rel=1e-4
        )

    def test_earnings_data_populated(self, provider):
        """Benzinga earnings calendar data is populated."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]
        client.vx.list_stock_financials.return_value = []
        client.list_benzinga_earnings.return_value = _make_earnings_mock()

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result.last_earnings_date == "2024-01-25"
        assert result.estimated_next_earnings_date is not None
        assert result.eps_surprise_pct == 5.2

    def test_analyst_consensus_populated(self, provider):
        """Benzinga analyst consensus data is populated."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]
        client.vx.list_stock_financials.return_value = []
        client.list_benzinga_consensus_ratings.return_value = [_make_consensus_mock()]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result.analyst_rating == "buy"
        assert result.analyst_target_price == 225.0
        assert result.analyst_buy_count == 35  # 15 strong_buy + 20 buy
        assert result.analyst_sell_count == 3   # 2 sell + 1 strong_sell

    def test_short_interest_populated(self, provider):
        """Short interest data is populated."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]
        client.vx.list_stock_financials.return_value = []

        si = MagicMock()
        si.short_interest = 200_000_000
        si.days_to_cover = 2.5
        client.list_short_interest.return_value = [si]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result.short_shares == 200_000_000
        assert result.days_to_cover == 2.5

    def test_single_period_no_growth(self, provider):
        """With only 1 period, growth metrics are None but ratios still work."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]
        client.vx.list_stock_financials.return_value = [_make_financials_mock()]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result.earnings_growth is None
        assert result.revenue_growth is None
        # Pre-computed ratios still work
        assert result.current_ratio == 0.93
        assert result.price_to_earnings == 30.7
        assert result.free_cash_flow == 110_000_000_000

    def test_zero_denominator_safety(self, provider):
        """Division by zero in previous period should not crash."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]

        current = _make_financials_mock()
        previous = _make_financials_mock(revenues=0, net_income=0)
        client.vx.list_stock_financials.return_value = [current, previous]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result is not None
        assert result.earnings_growth is None  # prev_ni == 0 → skipped
        assert result.revenue_growth is None   # prev_rev == 0 → skipped

    def test_api_error_returns_basic(self, provider):
        p, client = provider
        client.get_ticker_details.side_effect = Exception("fail")
        client.list_financials_ratios.side_effect = Exception("fail")
        client.vx.list_stock_financials.return_value = []
        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_graceful_degradation(self, provider):
        """If some endpoints fail, others still populate data."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]
        client.vx.list_stock_financials.side_effect = Exception("fail")
        client.list_benzinga_earnings.side_effect = Exception("fail")
        # Consensus still works
        client.list_benzinga_consensus_ratings.return_value = [_make_consensus_mock()]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result.company_name == "Apple Inc."
        assert result.current_ratio == 0.93  # from ratios
        assert result.earnings_growth is None  # financials failed
        assert result.analyst_rating == "buy"  # consensus worked

    def test_news_sentiment(self, provider):
        """News with AI insights populates sentiment."""
        p, client = provider
        client.get_ticker_details.return_value = _make_details_mock()
        client.list_financials_ratios.return_value = [_make_ratios_mock()]
        client.vx.list_stock_financials.return_value = []

        article1 = MagicMock()
        article1.title = "Apple beats estimates"
        article1.published_utc = "2024-01-25T16:00:00Z"
        article1.article_url = "https://example.com/1"
        insight1 = MagicMock()
        insight1.ticker = "AAPL"
        insight1.sentiment = "positive"
        article1.insights = [insight1]

        article2 = MagicMock()
        article2.title = "Apple faces headwinds"
        article2.published_utc = "2024-01-24T10:00:00Z"
        article2.article_url = "https://example.com/2"
        insight2 = MagicMock()
        insight2.ticker = "AAPL"
        insight2.sentiment = "positive"
        article2.insights = [insight2]

        client.list_ticker_news.return_value = [article1, article2]

        result = p.get_fundamentals("AAPL", full_refresh=True)
        assert result.news_sentiment == "positive"
        assert result.recent_news_count == 2


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
