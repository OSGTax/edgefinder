"""Tests for edgefinder/scanner/scoring.py — multi-factor scoring engine."""

import pytest

from edgefinder.core.models import TickerFundamentals
from edgefinder.scanner.scoring import (
    ScoringFactor,
    ScoringProfile,
    compute_score,
    compute_universe_stats,
    rank_and_filter,
)


def _make_fund(symbol: str = "AAPL", **overrides) -> TickerFundamentals:
    defaults = dict(
        symbol=symbol,
        earnings_growth=0.20,
        revenue_growth=0.15,
        debt_to_equity=1.5,
        current_ratio=1.2,
        fcf_yield=0.05,
        return_on_equity=0.25,
        price_to_earnings=20.0,
    )
    defaults.update(overrides)
    return TickerFundamentals(**defaults)


class TestComputeUniverseStats:
    def test_computes_min_max(self):
        stocks = [
            _make_fund("AAPL", earnings_growth=0.10),
            _make_fund("MSFT", earnings_growth=0.30),
            _make_fund("GOOG", earnings_growth=0.20),
        ]
        factors = [ScoringFactor("earnings_growth", 1.0, "high")]
        stats = compute_universe_stats(stocks, factors)
        assert stats["earnings_growth"] == (0.10, 0.30)

    def test_skips_none_values(self):
        stocks = [
            _make_fund("AAPL", earnings_growth=0.10),
            _make_fund("MSFT", earnings_growth=None),
            _make_fund("GOOG", earnings_growth=0.30),
        ]
        factors = [ScoringFactor("earnings_growth", 1.0, "high")]
        stats = compute_universe_stats(stocks, factors)
        assert stats["earnings_growth"] == (0.10, 0.30)

    def test_single_value_returns_same_min_max(self):
        stocks = [_make_fund("AAPL", earnings_growth=0.20)]
        factors = [ScoringFactor("earnings_growth", 1.0, "high")]
        stats = compute_universe_stats(stocks, factors)
        assert stats["earnings_growth"] == (0.20, 0.20)


class TestComputeScore:
    def test_highest_growth_scores_100(self):
        """Stock with the best values in the universe scores 100."""
        profile = ScoringProfile(factors=[
            ScoringFactor("earnings_growth", 1.0, "high"),
        ])
        stocks = [
            _make_fund("LOW", earnings_growth=0.05),
            _make_fund("HIGH", earnings_growth=0.50),
        ]
        stats = compute_universe_stats(stocks, profile.factors)
        score = compute_score(stocks[1], profile, stats)
        assert score == 100.0

    def test_lowest_growth_scores_0(self):
        """Stock with the worst values in the universe scores 0."""
        profile = ScoringProfile(factors=[
            ScoringFactor("earnings_growth", 1.0, "high"),
        ])
        stocks = [
            _make_fund("LOW", earnings_growth=0.05),
            _make_fund("HIGH", earnings_growth=0.50),
        ]
        stats = compute_universe_stats(stocks, profile.factors)
        score = compute_score(stocks[0], profile, stats)
        assert score == 0.0

    def test_low_ideal_inverts_scoring(self):
        """For 'low' ideal, lower values score higher."""
        profile = ScoringProfile(factors=[
            ScoringFactor("debt_to_equity", 1.0, "low"),
        ])
        low_debt = _make_fund("GOOD", debt_to_equity=0.5)
        high_debt = _make_fund("BAD", debt_to_equity=3.0)
        stats = compute_universe_stats([low_debt, high_debt], profile.factors)
        assert compute_score(low_debt, profile, stats) == 100.0
        assert compute_score(high_debt, profile, stats) == 0.0

    def test_multi_factor_weighted(self):
        """Score is weighted across multiple factors."""
        profile = ScoringProfile(factors=[
            ScoringFactor("earnings_growth", 0.6, "high"),
            ScoringFactor("debt_to_equity", 0.4, "low"),
        ])
        # Stock with great growth but bad debt
        mixed = _make_fund("MIXED", earnings_growth=0.50, debt_to_equity=3.0)
        best = _make_fund("BEST", earnings_growth=0.50, debt_to_equity=0.5)
        worst = _make_fund("WORST", earnings_growth=0.05, debt_to_equity=3.0)

        stats = compute_universe_stats([mixed, best, worst], profile.factors)
        score_best = compute_score(best, profile, stats)
        score_mixed = compute_score(mixed, profile, stats)
        score_worst = compute_score(worst, profile, stats)

        assert score_best == 100.0
        assert score_worst == 0.0
        assert 0 < score_mixed < 100  # partial score

    def test_none_metrics_skipped(self):
        """Metrics with None values are excluded from scoring."""
        profile = ScoringProfile(factors=[
            ScoringFactor("earnings_growth", 0.5, "high"),
            ScoringFactor("return_on_equity", 0.5, "high"),
        ])
        # Only has earnings_growth, not return_on_equity
        stock = _make_fund("AAPL", earnings_growth=0.30, return_on_equity=None)
        other = _make_fund("MSFT", earnings_growth=0.10, return_on_equity=0.5)
        stats = compute_universe_stats([stock, other], profile.factors)
        score = compute_score(stock, profile, stats)
        # Should still produce a score based on available metrics
        assert score > 0

    def test_all_none_returns_zero(self):
        """Stock with all None metrics gets score 0."""
        profile = ScoringProfile(factors=[
            ScoringFactor("earnings_growth", 1.0, "high"),
        ])
        stock = TickerFundamentals(symbol="JUNK")  # all None
        stats = {"earnings_growth": (0.0, 1.0)}
        assert compute_score(stock, profile, stats) == 0.0

    def test_range_ideal(self):
        """Range scoring: 1.0 if within range, decaying outside."""
        profile = ScoringProfile(factors=[
            ScoringFactor("price_to_earnings", 1.0, "range", range_min=10.0, range_max=25.0),
        ])
        in_range = _make_fund("IN", price_to_earnings=18.0)
        out_range = _make_fund("OUT", price_to_earnings=50.0)
        stats = compute_universe_stats([in_range, out_range], profile.factors)
        score_in = compute_score(in_range, profile, stats)
        score_out = compute_score(out_range, profile, stats)
        assert score_in > score_out


class TestRankAndFilter:
    def test_returns_top_n(self):
        stocks = [
            (_make_fund("A"), 90.0),
            (_make_fund("B"), 50.0),
            (_make_fund("C"), 75.0),
            (_make_fund("D"), 95.0),
        ]
        result = rank_and_filter(stocks, top_n=2)
        assert len(result) == 2
        assert result[0][1] == 95.0  # D is first
        assert result[1][1] == 90.0  # A is second

    def test_top_n_larger_than_list(self):
        stocks = [(_make_fund("A"), 80.0)]
        result = rank_and_filter(stocks, top_n=10)
        assert len(result) == 1
