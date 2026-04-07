"""Tests for edgefinder/scanner/scanner.py."""

from unittest.mock import MagicMock

import pytest

from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, Ticker, TickerStrategyQualification
from edgefinder.scanner.scanner import FundamentalScanner, ScannedStock
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


class TestStrategyQualification:
    def test_any_strategy_qualifies(self, scanner):
        fund = _make_fund(earnings_growth=0.20, revenue_growth=0.15)
        qualifying = scanner._check_strategy_qualification(fund)
        assert len(qualifying) > 0

    def test_no_strategy_qualifies(self, scanner):
        fund = TickerFundamentals(
            symbol="JUNK",
            earnings_growth=-0.50,
            revenue_growth=-0.20,
            current_ratio=0.5,
            fcf_yield=0.01,
            debt_to_equity=5.0,
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
        assert fund.earnings_growth is not None
        assert fund.current_ratio is not None

    def test_upsert_updates_existing(self, scanner, db_session):
        scanner.run(tickers=["AAPL"])
        scanner.run(tickers=["AAPL"])
        count = db_session.query(Ticker).filter_by(symbol="AAPL").count()
        assert count == 1

    def test_batch_sets_scan_batch(self, scanner, db_session):
        scanner.run(tickers=["AAPL"], batch_index=2)
        ticker = db_session.query(Ticker).filter_by(symbol="AAPL").first()
        assert ticker.scan_batch == 2

    def test_batch_deactivation_scoped(self, scanner, db_session, mock_provider):
        """Scanning batch 1 should not deactivate tickers from batch 0."""
        # Batch 0: AAPL qualifies
        scanner.run(tickers=["AAPL"], batch_index=0)
        assert db_session.query(Ticker).filter_by(symbol="AAPL").first().is_active is True

        # Batch 1: MSFT qualifies
        scanner.run(tickers=["MSFT"], batch_index=1)
        # AAPL should still be active (different batch)
        assert db_session.query(Ticker).filter_by(symbol="AAPL").first().is_active is True
        assert db_session.query(Ticker).filter_by(symbol="MSFT").first().is_active is True

    def test_batch_deactivates_own_tickers(self, scanner, db_session, mock_provider):
        """Re-scanning a batch deactivates tickers from that batch that no longer qualify."""
        # Batch 0: AAPL qualifies
        scanner.run(tickers=["AAPL", "GOOG"], batch_index=0)
        assert db_session.query(Ticker).filter_by(symbol="AAPL").first().is_active is True
        assert db_session.query(Ticker).filter_by(symbol="GOOG").first().is_active is True

        # Re-scan batch 0 with only AAPL (GOOG dropped from batch)
        scanner.run(tickers=["AAPL"], batch_index=0)
        assert db_session.query(Ticker).filter_by(symbol="AAPL").first().is_active is True
        assert db_session.query(Ticker).filter_by(symbol="GOOG").first().is_active is False


class TestPerStrategyQualifications:
    """Tests for per-strategy qualification persistence (Phase 2 fix)."""

    def test_qualifications_saved_per_strategy(self, scanner, db_session):
        """Each strategy gets its own qualification row per ticker."""
        scanner.run(tickers=["AAPL"])
        quals = db_session.query(TickerStrategyQualification).filter_by(symbol="AAPL").all()
        strategy_names = {q.strategy_name for q in quals}
        # Should have a row for each registered strategy
        assert "alpha" in strategy_names
        assert "bravo" in strategy_names
        assert "charlie" in strategy_names

    def test_qualified_flag_varies_by_strategy(self, scanner, db_session, mock_provider):
        """A stock may qualify for some strategies but not others."""
        # Stock with positive earnings/revenue but bad current_ratio — qualifies for Alpha, not Bravo
        mock_provider.get_fundamentals.side_effect = lambda t: _make_fund(
            t,
            earnings_growth=0.20, revenue_growth=0.15,
            current_ratio=0.5,  # too low for Bravo
            fcf_yield=0.005,    # too low for Charlie
            debt_to_equity=4.0, # too high for Bravo and Charlie
        )
        scanner.run(tickers=["AAPL"])

        alpha_qual = db_session.query(TickerStrategyQualification).filter_by(
            symbol="AAPL", strategy_name="alpha"
        ).first()
        bravo_qual = db_session.query(TickerStrategyQualification).filter_by(
            symbol="AAPL", strategy_name="bravo"
        ).first()

        assert alpha_qual.qualified is True
        assert bravo_qual.qualified is False

    def test_upsert_updates_qualification(self, scanner, db_session):
        """Re-scanning a ticker updates existing qualification rows."""
        scanner.run(tickers=["AAPL"])
        scanner.run(tickers=["AAPL"])
        # Should still have exactly one row per strategy, not duplicates
        count = db_session.query(TickerStrategyQualification).filter_by(symbol="AAPL").count()
        assert count == 3  # one per strategy (alpha, bravo, charlie)


class TestFullPipeline:
    def test_scan_returns_scanned_stocks(self, scanner):
        results = scanner.run(tickers=["AAPL", "MSFT"])
        assert len(results) == 2
        for stock in results:
            assert isinstance(stock, ScannedStock)
            assert stock.fundamentals.earnings_growth is not None

    def test_scan_publishes_event(self, scanner):
        events = []
        from edgefinder.core.events import event_bus
        event_bus.subscribe("scan.completed", lambda d: events.append(d))
        scanner.run(tickers=["AAPL"])
        assert len(events) == 1
        assert events[0]["qualified"] >= 0
        event_bus.clear()
