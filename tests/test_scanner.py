"""Tests for edgefinder/scanner/scanner.py."""

from unittest.mock import MagicMock

import pytest

from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, Ticker
from edgefinder.scanner.scanner import FundamentalScanner, ScoredStock
from edgefinder.strategies.base import StrategyRegistry


def _make_fund(symbol: str = "AAPL", **overrides) -> TickerFundamentals:
    """Build a TickerFundamentals with sensible defaults."""
    defaults = dict(
        symbol=symbol,
        company_name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=50_000_000_000,
        price=175.0,
        peg_ratio=1.2,
        earnings_growth=0.25,
        debt_to_equity=0.5,
        revenue_growth=0.20,
        institutional_pct=0.60,
        fcf_yield=0.06,
        price_to_tangible_book=2.0,
        short_interest=0.03,
        ev_to_ebitda=6.0,
        current_ratio=1.8,
    )
    defaults.update(overrides)
    return TickerFundamentals(**defaults)


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_ticker_universe.return_value = ["AAPL", "MSFT", "GOOG"]
    provider.get_fundamentals.side_effect = lambda t: _make_fund(t)
    return provider


@pytest.fixture
def scanner(mock_provider, db_session):
    # Ensure strategies are loaded
    import importlib
    from edgefinder.strategies import alpha, bravo, charlie
    StrategyRegistry.clear()
    importlib.reload(alpha)
    importlib.reload(bravo)
    importlib.reload(charlie)
    return FundamentalScanner(provider=mock_provider, session=db_session)


class TestPreScreen:
    def test_filters_low_market_cap(self, scanner):
        fund = _make_fund(market_cap=100_000_000)
        assert scanner._passes_prescreen(fund) is False

    def test_filters_high_market_cap(self, scanner):
        fund = _make_fund(market_cap=300_000_000_000_000)
        assert scanner._passes_prescreen(fund) is False

    def test_filters_low_price(self, scanner):
        fund = _make_fund(price=2.0)
        assert scanner._passes_prescreen(fund) is False

    def test_filters_excluded_sector(self, scanner):
        fund = _make_fund(sector="Utilities")
        assert scanner._passes_prescreen(fund) is False

    def test_passes_qualifying_stock(self, scanner):
        fund = _make_fund()
        assert scanner._passes_prescreen(fund) is True


class TestLynchScoring:
    def test_perfect_stock(self, scanner):
        fund = _make_fund(
            peg_ratio=0.8,
            earnings_growth=0.30,
            debt_to_equity=0.3,
            revenue_growth=0.25,
            institutional_pct=0.45,
        )
        score, cat = scanner._score_lynch(fund)
        assert score > 80
        assert cat == "fast_grower"

    def test_terrible_stock(self, scanner):
        fund = _make_fund(
            peg_ratio=4.0,
            earnings_growth=-0.10,
            debt_to_equity=3.0,
            revenue_growth=-0.05,
            institutional_pct=0.90,
        )
        score, cat = scanner._score_lynch(fund)
        assert score < 25

    def test_missing_data_graceful(self, scanner):
        fund = TickerFundamentals(symbol="SPARSE")
        score, cat = scanner._score_lynch(fund)
        assert 0 <= score <= 100
        assert cat == "slow_grower"

    def test_category_fast_grower(self, scanner):
        fund = _make_fund(earnings_growth=0.25, revenue_growth=0.20)
        _, cat = scanner._score_lynch(fund)
        assert cat == "fast_grower"

    def test_category_turnaround(self, scanner):
        fund = _make_fund(earnings_growth=-0.10, debt_to_equity=1.0, revenue_growth=0.01)
        _, cat = scanner._score_lynch(fund)
        assert cat == "turnaround"

    def test_category_stalwart(self, scanner):
        fund = _make_fund(
            earnings_growth=0.15, revenue_growth=0.08,
            market_cap=50_000_000_000,
        )
        _, cat = scanner._score_lynch(fund)
        assert cat == "stalwart"


class TestBurryScoring:
    def test_perfect_value(self, scanner):
        fund = _make_fund(
            fcf_yield=0.10,
            price_to_tangible_book=0.8,
            short_interest=0.20,
            ev_to_ebitda=3.0,
            current_ratio=2.5,
        )
        score = scanner._score_burry(fund)
        assert score > 85

    def test_no_value(self, scanner):
        fund = _make_fund(
            fcf_yield=-0.02,
            price_to_tangible_book=8.0,
            short_interest=0.01,
            ev_to_ebitda=20.0,
            current_ratio=0.5,
        )
        score = scanner._score_burry(fund)
        assert score < 20

    def test_missing_data_graceful(self, scanner):
        fund = TickerFundamentals(symbol="SPARSE")
        score = scanner._score_burry(fund)
        assert 0 <= score <= 100


class TestComposite:
    def test_50_50_default(self, scanner):
        result = scanner._compute_composite(80.0, 60.0)
        assert result == 70.0


class TestStrategyQualification:
    def test_any_strategy_qualifies(self, scanner):
        fund = _make_fund(composite_score=75.0)
        qualifying = scanner._check_strategy_qualification(fund)
        assert len(qualifying) > 0
        assert "alpha" in qualifying

    def test_no_strategy_qualifies(self, scanner):
        fund = TickerFundamentals(
            symbol="JUNK",
            composite_score=10.0,
            earnings_growth=-0.50,
            burry_score=10.0,
            current_ratio=0.5,
            short_interest=0.01,
            fcf_yield=0.01,
        )
        qualifying = scanner._check_strategy_qualification(fund)
        assert len(qualifying) == 0


class TestDBPersistence:
    def test_saves_ticker(self, scanner, db_session):
        scanner.run(tickers=["AAPL"])
        ticker = db_session.query(Ticker).filter_by(symbol="AAPL").first()
        assert ticker is not None
        assert ticker.company_name == "Apple Inc."

    def test_saves_fundamentals(self, scanner, db_session):
        scanner.run(tickers=["AAPL"])
        ticker = db_session.query(Ticker).filter_by(symbol="AAPL").first()
        fund = db_session.query(Fundamental).filter_by(ticker_id=ticker.id).first()
        assert fund is not None
        assert fund.lynch_score is not None
        assert fund.burry_score is not None

    def test_upsert_updates_existing(self, scanner, db_session):
        scanner.run(tickers=["AAPL"])
        scanner.run(tickers=["AAPL"])
        count = db_session.query(Ticker).filter_by(symbol="AAPL").count()
        assert count == 1


class TestFullPipeline:
    def test_scan_returns_scored_stocks(self, scanner):
        results = scanner.run(tickers=["AAPL", "MSFT"])
        assert len(results) == 2
        for stock in results:
            assert isinstance(stock, ScoredStock)
            assert stock.lynch_score > 0
            assert stock.burry_score > 0
            assert stock.composite_score > 0

    def test_scan_publishes_event(self, scanner):
        events = []
        from edgefinder.core.events import event_bus
        event_bus.subscribe("scan.completed", lambda d: events.append(d))
        scanner.run(tickers=["AAPL"])
        assert len(events) == 1
        assert events[0]["qualified"] >= 0
        event_bus.clear()
