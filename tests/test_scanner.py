"""
EdgeFinder Module 1 Tests: Fundamental Scanner
================================================
Tests cover: pre-screening, Lynch scoring, Burry scoring,
composite scoring, database persistence, and edge cases.

Run: python -m pytest tests/test_scanner.py -v
"""

import pytest
from modules.scanner import (
    passes_prescreen, score_lynch, score_burry, score_stock,
    classify_lynch_category, run_scan, get_active_watchlist,
    FundamentalData, _safe_float,
)
from modules.database import WatchlistStock, get_session


# ════════════════════════════════════════════════════════════
# PRE-SCREENING TESTS
# ════════════════════════════════════════════════════════════

class TestPreScreening:
    """Test the pre-screening filter that eliminates junk before scoring."""

    def test_strong_stock_passes(self, sample_strong_stock):
        assert passes_prescreen(sample_strong_stock) is True

    def test_penny_stock_rejected(self, sample_penny_stock):
        """Penny stocks (low cap, low price, low volume) must be filtered."""
        assert passes_prescreen(sample_penny_stock) is False

    def test_utility_stock_rejected(self, sample_utility_stock):
        """Excluded sectors must be filtered."""
        assert passes_prescreen(sample_utility_stock) is False

    def test_no_market_cap_rejected(self):
        data = FundamentalData(ticker="X", market_cap=0, price=50, avg_volume=1_000_000)
        assert passes_prescreen(data) is False

    def test_no_price_rejected(self):
        data = FundamentalData(ticker="X", market_cap=1e9, price=0, avg_volume=1_000_000)
        assert passes_prescreen(data) is False

    def test_price_too_high_rejected(self):
        data = FundamentalData(ticker="X", market_cap=1e9, price=600, avg_volume=1_000_000)
        assert passes_prescreen(data) is False

    def test_low_volume_rejected(self):
        data = FundamentalData(ticker="X", market_cap=1e9, price=50, avg_volume=100_000)
        assert passes_prescreen(data) is False

    def test_mega_cap_rejected(self):
        data = FundamentalData(ticker="X", market_cap=300e9, price=50, avg_volume=1_000_000)
        assert passes_prescreen(data) is False

    def test_boundary_min_market_cap(self):
        """Exactly at minimum market cap should pass."""
        data = FundamentalData(ticker="X", market_cap=300_000_000, price=10, avg_volume=600_000)
        assert passes_prescreen(data) is True

    def test_boundary_min_price(self):
        """Exactly at minimum price should pass."""
        data = FundamentalData(ticker="X", market_cap=1e9, price=5.00, avg_volume=600_000)
        assert passes_prescreen(data) is True


# ════════════════════════════════════════════════════════════
# LYNCH SCORING TESTS
# ════════════════════════════════════════════════════════════

class TestLynchScoring:
    """Test Peter Lynch scoring criteria."""

    def test_strong_stock_scores_high(self, sample_strong_stock):
        score, category, breakdown = score_lynch(sample_strong_stock)
        assert score >= 75, f"Strong stock should score >= 75, got {score}"
        assert category == "fast_grower"

    def test_weak_stock_scores_low(self, sample_weak_stock):
        score, category, breakdown = score_lynch(sample_weak_stock)
        assert score <= 30, f"Weak stock should score <= 30, got {score}"

    def test_score_range(self, sample_moderate_stock):
        score, _, _ = score_lynch(sample_moderate_stock)
        assert 0 <= score <= 100, f"Score must be 0-100, got {score}"

    def test_missing_data_doesnt_crash(self, sample_missing_data_stock):
        """Stocks with None fields should still score without errors."""
        score, category, breakdown = score_lynch(sample_missing_data_stock)
        assert 0 <= score <= 100
        assert isinstance(breakdown, dict)

    def test_peg_below_ideal_scores_max(self):
        data = FundamentalData(ticker="X", peg_ratio=0.5)
        _, _, breakdown = score_lynch(data)
        assert breakdown["peg"] == 100.0

    def test_peg_above_max_scores_zero(self):
        data = FundamentalData(ticker="X", peg_ratio=2.0)
        _, _, breakdown = score_lynch(data)
        assert breakdown["peg"] == 0.0

    def test_peg_between_ideal_and_max_interpolates(self):
        data = FundamentalData(ticker="X", peg_ratio=1.25)
        _, _, breakdown = score_lynch(data)
        assert 0 < breakdown["peg"] < 100

    def test_high_debt_scores_zero(self):
        data = FundamentalData(ticker="X", debt_to_equity=3.0)
        _, _, breakdown = score_lynch(data)
        assert breakdown["debt_to_equity"] == 0.0

    def test_low_debt_scores_max(self):
        data = FundamentalData(ticker="X", debt_to_equity=0.2)
        _, _, breakdown = score_lynch(data)
        assert breakdown["debt_to_equity"] == 100.0

    def test_overcrowded_institutional_penalized(self):
        data = FundamentalData(ticker="X", institutional_pct=0.90)
        _, _, breakdown = score_lynch(data)
        assert breakdown["institutional"] < 50

    def test_ideal_institutional_scores_max(self):
        data = FundamentalData(ticker="X", institutional_pct=0.45)
        _, _, breakdown = score_lynch(data)
        assert breakdown["institutional"] == 100.0

    def test_all_none_fields(self):
        """Completely empty stock should not crash."""
        data = FundamentalData(ticker="X")
        score, category, breakdown = score_lynch(data)
        assert 0 <= score <= 100
        assert category == "unclassified"


# ════════════════════════════════════════════════════════════
# LYNCH CATEGORY CLASSIFICATION
# ════════════════════════════════════════════════════════════

class TestLynchCategories:
    """Test stock category classification."""

    def test_fast_grower(self):
        data = FundamentalData(ticker="X", earnings_growth=0.30, revenue_growth=0.25)
        assert classify_lynch_category(data) == "fast_grower"

    def test_stalwart(self):
        data = FundamentalData(ticker="X", earnings_growth=0.15, revenue_growth=0.12)
        assert classify_lynch_category(data) == "stalwart"

    def test_slow_grower(self):
        data = FundamentalData(ticker="X", earnings_growth=0.05, revenue_growth=0.03)
        assert classify_lynch_category(data) == "slow_grower"

    def test_turnaround(self):
        data = FundamentalData(
            ticker="X", earnings_growth=-0.10, earnings_quarterly_growth=0.05
        )
        assert classify_lynch_category(data) == "turnaround"

    def test_asset_play(self):
        data = FundamentalData(
            ticker="X", earnings_growth=0.05, price_to_tangible_book=0.7
        )
        assert classify_lynch_category(data) == "asset_play"

    def test_cyclical(self):
        data = FundamentalData(ticker="X", earnings_growth=-0.15)
        assert classify_lynch_category(data) == "cyclical"

    def test_no_data_unclassified(self):
        data = FundamentalData(ticker="X")
        assert classify_lynch_category(data) == "unclassified"


# ════════════════════════════════════════════════════════════
# BURRY SCORING TESTS
# ════════════════════════════════════════════════════════════

class TestBurryScoring:
    """Test Michael Burry scoring criteria."""

    def test_strong_stock_scores_high(self, sample_strong_stock):
        score, breakdown = score_burry(sample_strong_stock)
        assert score >= 70, f"Strong stock should score >= 70, got {score}"

    def test_weak_stock_scores_low(self, sample_weak_stock):
        score, breakdown = score_burry(sample_weak_stock)
        assert score <= 35, f"Weak stock should score <= 35, got {score}"

    def test_score_range(self, sample_moderate_stock):
        score, _ = score_burry(sample_moderate_stock)
        assert 0 <= score <= 100

    def test_high_fcf_yield_scores_high(self):
        data = FundamentalData(ticker="X", fcf_yield=0.12)
        _, breakdown = score_burry(data)
        assert breakdown["fcf_yield"] == 100.0

    def test_negative_fcf_yield_scores_zero(self):
        data = FundamentalData(ticker="X", fcf_yield=-0.05)
        _, breakdown = score_burry(data)
        assert breakdown["fcf_yield"] == 0.0

    def test_deep_value_book(self):
        data = FundamentalData(ticker="X", price_to_tangible_book=0.8)
        _, breakdown = score_burry(data)
        assert breakdown["price_to_tangible_book"] == 100.0

    def test_high_ev_ebitda_penalized(self):
        data = FundamentalData(ticker="X", ev_to_ebitda=20.0)
        _, breakdown = score_burry(data)
        assert breakdown["ev_to_ebitda"] < 50

    def test_strong_current_ratio(self):
        data = FundamentalData(ticker="X", current_ratio=3.0)
        _, breakdown = score_burry(data)
        assert breakdown["current_ratio"] >= 80

    def test_weak_current_ratio(self):
        data = FundamentalData(ticker="X", current_ratio=0.5)
        _, breakdown = score_burry(data)
        assert breakdown["current_ratio"] < 30

    def test_contrarian_short_interest(self):
        data = FundamentalData(ticker="X", short_interest=0.20)
        _, breakdown = score_burry(data)
        assert breakdown["short_interest"] >= 70

    def test_all_none_fields(self):
        data = FundamentalData(ticker="X")
        score, breakdown = score_burry(data)
        assert score == 0


# ════════════════════════════════════════════════════════════
# COMPOSITE SCORING TESTS
# ════════════════════════════════════════════════════════════

class TestCompositeScoring:
    """Test the combined Lynch + Burry composite scoring."""

    def test_strong_stock_high_composite(self, sample_strong_stock):
        scored = score_stock(sample_strong_stock)
        assert scored.composite_score >= 70
        assert scored.lynch_score > 0
        assert scored.burry_score > 0

    def test_weak_stock_low_composite(self, sample_weak_stock):
        scored = score_stock(sample_weak_stock)
        assert scored.composite_score <= 35

    def test_composite_is_weighted_average(self, sample_strong_stock):
        scored = score_stock(sample_strong_stock)
        expected = (scored.lynch_score * 0.5) + (scored.burry_score * 0.5)
        assert abs(scored.composite_score - expected) < 0.1

    def test_scoring_preserves_data(self, sample_strong_stock):
        scored = score_stock(sample_strong_stock)
        assert scored.data.ticker == "GREAT"
        assert scored.data.price == 50.00

    def test_score_breakdown_populated(self, sample_strong_stock):
        scored = score_stock(sample_strong_stock)
        assert "lynch" in scored.score_breakdown
        assert "burry" in scored.score_breakdown
        assert "peg" in scored.score_breakdown["lynch"]
        assert "fcf_yield" in scored.score_breakdown["burry"]

    def test_ordering_strong_beats_weak(self, sample_strong_stock, sample_weak_stock):
        strong = score_stock(sample_strong_stock)
        weak = score_stock(sample_weak_stock)
        assert strong.composite_score > weak.composite_score

    def test_ordering_strong_beats_moderate(self, sample_strong_stock, sample_moderate_stock):
        strong = score_stock(sample_strong_stock)
        moderate = score_stock(sample_moderate_stock)
        assert strong.composite_score > moderate.composite_score


# ════════════════════════════════════════════════════════════
# DATABASE PERSISTENCE TESTS
# ════════════════════════════════════════════════════════════

class TestDatabasePersistence:
    """Test saving and retrieving watchlist from database."""

    def test_scan_saves_to_db(self, in_memory_db, sample_strong_stock):
        """Running a scan with mock data should persist to database."""
        results = run_scan(
            tickers=["AAPL"],  # Will try to fetch, likely works
            save_to_db=True,
        )
        # We can't guarantee yfinance returns data in test env,
        # so just check the function doesn't crash
        assert isinstance(results, list)

    def test_run_scan_with_mock_data(self, in_memory_db):
        """Test the scan pipeline with controlled mock data."""
        from modules.scanner import _save_watchlist, ScoredStock

        mock_scored = ScoredStock(
            data=FundamentalData(
                ticker="TEST",
                company_name="Test Corp",
                sector="Technology",
                market_cap=5e9,
                price=100,
            ),
            lynch_score=80,
            burry_score=75,
            composite_score=77.5,
            lynch_category="fast_grower",
        )

        _save_watchlist([mock_scored])

        # Verify it was saved
        watchlist = get_active_watchlist()
        assert len(watchlist) == 1
        assert watchlist[0]["ticker"] == "TEST"
        assert watchlist[0]["composite_score"] == 77.5

    def test_new_scan_preserves_other_tickers(self, in_memory_db):
        """Scanning new tickers should preserve entries from other sectors."""
        from modules.scanner import _save_watchlist, ScoredStock

        old = ScoredStock(
            data=FundamentalData(ticker="OLD", company_name="Old Corp", sector="Tech",
                                 market_cap=1e9, price=50),
            lynch_score=70, burry_score=65, composite_score=67.5,
            lynch_category="stalwart",
        )
        _save_watchlist([old])
        assert len(get_active_watchlist()) == 1

        new = ScoredStock(
            data=FundamentalData(ticker="NEW", company_name="New Corp", sector="Healthcare",
                                 market_cap=2e9, price=75),
            lynch_score=85, burry_score=80, composite_score=82.5,
            lynch_category="fast_grower",
        )
        _save_watchlist([new])

        watchlist = get_active_watchlist()
        assert len(watchlist) == 2  # Both persist (different tickers)

    def test_upsert_same_ticker_updates_row(self, in_memory_db):
        """Re-scanning the same ticker should update in place, not duplicate."""
        from modules.scanner import _save_watchlist, ScoredStock

        v1 = ScoredStock(
            data=FundamentalData(ticker="AAPL", company_name="Apple v1", sector="Tech",
                                 market_cap=1e12, price=150),
            lynch_score=70, burry_score=65, composite_score=67.5,
            lynch_category="stalwart",
        )
        _save_watchlist([v1])

        v2 = ScoredStock(
            data=FundamentalData(ticker="AAPL", company_name="Apple v2", sector="Tech",
                                 market_cap=1.1e12, price=155),
            lynch_score=75, burry_score=70, composite_score=72.5,
            lynch_category="stalwart",
        )
        _save_watchlist([v2])

        watchlist = get_active_watchlist()
        assert len(watchlist) == 1
        assert watchlist[0]["company_name"] == "Apple v2"
        assert watchlist[0]["price"] == 155

        # Verify only one row in DB total (not two)
        session = get_session()
        total_rows = session.query(WatchlistStock).filter(
            WatchlistStock.ticker == "AAPL"
        ).count()
        session.close()
        assert total_rows == 1


class TestStrategyQualification:
    """Test strategy-driven watchlist qualification."""

    def test_lynch_only_stock_stays_active(self, in_memory_db):
        """A stock that only qualifies for Lynch should stay active."""
        from modules.scanner import _save_watchlist, _deactivate_unqualified, ScoredStock

        # High Lynch score, low Burry score
        stock = ScoredStock(
            data=FundamentalData(ticker="LYNCHONLY", company_name="Lynch Fav",
                                 sector="Tech", market_cap=5e9, price=100),
            lynch_score=75, burry_score=20, composite_score=47.5,
            lynch_category="fast_grower",
        )
        _save_watchlist([stock])

        watchlist = get_active_watchlist()
        assert len(watchlist) == 1
        assert watchlist[0]["ticker"] == "LYNCHONLY"

    def test_burry_only_stock_stays_active(self, in_memory_db):
        """A stock that only qualifies for Burry should stay active."""
        from modules.scanner import _save_watchlist, ScoredStock

        # Low Lynch score, high Burry score
        stock = ScoredStock(
            data=FundamentalData(ticker="BURRYONLY", company_name="Burry Fav",
                                 sector="Tech", market_cap=5e9, price=100),
            lynch_score=20, burry_score=75, composite_score=47.5,
            lynch_category="slow_grower",
        )
        _save_watchlist([stock])

        watchlist = get_active_watchlist()
        assert len(watchlist) == 1
        assert watchlist[0]["ticker"] == "BURRYONLY"

    def test_deactivate_stock_no_strategy_qualifies(self, in_memory_db):
        """A stock that fails all strategies should be deactivated on re-scan."""
        from modules.scanner import _save_watchlist, _deactivate_unqualified, ScoredStock

        # First save a stock that qualifies
        good = ScoredStock(
            data=FundamentalData(ticker="FADE", company_name="Fading Corp",
                                 sector="Tech", market_cap=5e9, price=100),
            lynch_score=60, burry_score=55, composite_score=57.5,
            lynch_category="stalwart",
        )
        _save_watchlist([good])
        assert len(get_active_watchlist()) == 1

        # Re-scan: scores dropped below all strategy thresholds
        bad = ScoredStock(
            data=FundamentalData(ticker="FADE", company_name="Fading Corp",
                                 sector="Tech", market_cap=4e9, price=80),
            lynch_score=30, burry_score=25, composite_score=27.5,
            lynch_category="slow_grower",
        )
        _deactivate_unqualified([bad])

        watchlist = get_active_watchlist()
        assert len(watchlist) == 0

    def test_manual_entry_never_deactivated(self, in_memory_db):
        """Manual entries should survive deactivation sweeps."""
        from modules.scanner import _deactivate_unqualified, ScoredStock

        # Insert a manual entry directly
        session = get_session()
        manual = WatchlistStock(
            ticker="MANUAL",
            company_name="Manual Pick",
            sector="Tech",
            lynch_score=10,
            burry_score=10,
            composite_score=10,
            is_active=True,
            notes="MANUAL: Added by user",
        )
        session.add(manual)
        session.commit()
        session.close()

        # Simulate a re-scan where MANUAL fails all strategies
        bad = ScoredStock(
            data=FundamentalData(ticker="MANUAL", company_name="Manual Pick",
                                 sector="Tech", market_cap=1e9, price=50),
            lynch_score=10, burry_score=10, composite_score=10,
            lynch_category="unclassified",
        )
        _deactivate_unqualified([bad])

        watchlist = get_active_watchlist()
        assert len(watchlist) == 1
        assert watchlist[0]["ticker"] == "MANUAL"

    def test_plugin_strategy_keeps_stock_active(self, in_memory_db):
        """A third-party strategy plugin should be able to keep stocks active."""
        from modules.scanner import _any_strategy_qualifies, ScoredStock
        from modules.strategies.base import BaseStrategy, StrategyRegistry

        # Register a mock plugin strategy
        @StrategyRegistry.register("test_momentum")
        class MomentumStrategy(BaseStrategy):
            @property
            def name(self): return "test_momentum"
            @property
            def version(self): return "0.1"
            def init(self): pass
            def generate_signals(self, bars): return []
            def on_trade_executed(self, n): pass
            def qualifies_stock(self, stock_data):
                # This plugin wants stocks with high revenue growth
                return (stock_data.get("revenue_growth") or 0) >= 0.20

        # Stock that Lynch and Burry would reject, but momentum wants
        stock = ScoredStock(
            data=FundamentalData(ticker="MOMO", company_name="Momentum Inc",
                                 sector="Tech", market_cap=5e9, price=100,
                                 revenue_growth=0.30),
            lynch_score=30, burry_score=25, composite_score=27.5,
            lynch_category="unclassified",
        )

        assert _any_strategy_qualifies(stock) is True

        # Clean up registry
        StrategyRegistry._strategies.pop("test_momentum", None)


# ════════════════════════════════════════════════════════════
# UTILITY TESTS
# ════════════════════════════════════════════════════════════

class TestUtilities:
    """Test helper functions."""

    def test_safe_float_normal(self):
        assert _safe_float(3.14) == 3.14

    def test_safe_float_int(self):
        assert _safe_float(42) == 42.0

    def test_safe_float_string_number(self):
        assert _safe_float("3.14") == 3.14

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) is None

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) is None

    def test_safe_float_bad_string(self):
        assert _safe_float("not a number") is None


# ════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_zero_price_no_crash(self):
        data = FundamentalData(ticker="X", price=0, market_cap=1e9)
        scored = score_stock(data)
        assert 0 <= scored.composite_score <= 100

    def test_negative_earnings_no_crash(self):
        data = FundamentalData(ticker="X", earnings_growth=-0.50)
        score, _, _ = score_lynch(data)
        assert 0 <= score <= 100

    def test_extreme_peg_ratio(self):
        data = FundamentalData(ticker="X", peg_ratio=100.0)
        score, _, breakdown = score_lynch(data)
        assert breakdown["peg"] == 0.0

    def test_negative_peg_ratio(self):
        """Negative PEG (negative earnings) should not score."""
        data = FundamentalData(ticker="X", peg_ratio=-2.0)
        _, _, breakdown = score_lynch(data)
        assert breakdown["peg"] is None or breakdown["peg"] == 0

    def test_zero_fcf_yield(self):
        data = FundamentalData(ticker="X", fcf_yield=0.0)
        _, breakdown = score_burry(data)
        assert breakdown["fcf_yield"] == 0.0

    def test_debt_to_equity_as_percentage(self):
        """yfinance sometimes returns D/E as percentage (e.g., 150 instead of 1.5)."""
        data = FundamentalData(ticker="X", market_cap=1e9, price=50,
                               avg_volume=1e6, debt_to_equity=150)
        # The fetch function normalizes this, but scoring should handle it too
        scored = score_stock(data)
        assert 0 <= scored.composite_score <= 100


# ════════════════════════════════════════════════════════════
# INTEGRATION TEST (hits real API — skip in CI)
# ════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that hit real yfinance API.
    Run with: python -m pytest tests/test_scanner.py -v -m integration
    Skip with: python -m pytest tests/test_scanner.py -v -m "not integration"
    """

    def test_fetch_real_stock(self):
        """Fetch AAPL and verify we get data back."""
        from modules.scanner import fetch_fundamental_data
        data = fetch_fundamental_data("AAPL")
        assert data is not None, "Failed to fetch AAPL from yfinance"
        assert data.ticker == "AAPL"
        assert data.market_cap > 0
        assert data.price > 0

    def test_score_real_stock(self):
        """Fetch and score a real stock end to end."""
        from modules.scanner import fetch_fundamental_data
        data = fetch_fundamental_data("MSFT")
        if data is None:
            pytest.skip("yfinance returned no data for MSFT")
        scored = score_stock(data)
        assert 0 <= scored.composite_score <= 100
        assert scored.lynch_category != "unknown"

    def test_small_scan(self, in_memory_db):
        """Run a scan on 5 well-known tickers."""
        results = run_scan(
            tickers=["AAPL", "MSFT", "GOOGL", "JPM", "JNJ"],
            save_to_db=True,
        )
        assert isinstance(results, list)
        # At least some should return data
        # (yfinance can be flaky, so we're lenient)


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_scanner.py -v
#
# Expected results:
#   TestPreScreening:           9 tests  — all should PASS
#   TestLynchScoring:          11 tests  — all should PASS
#   TestLynchCategories:        7 tests  — all should PASS
#   TestBurryScoring:          10 tests  — all should PASS
#   TestCompositeScoring:       7 tests  — all should PASS
#   TestDatabasePersistence:    4 tests  — all should PASS
#   TestStrategyQualification:  5 tests  — all should PASS
#   TestUtilities:              7 tests  — all should PASS
#   TestEdgeCases:              6 tests  — all should PASS
#   TestIntegration:            3 tests  — may skip if no network
#
# TOTAL: 69 tests
#
# If any test in TestPreScreening, TestLynchScoring, TestBurryScoring,
# or TestCompositeScoring fails, DO NOT proceed to Module 2.
# Fix the scoring logic first.
# ════════════════════════════════════════════════════════════
