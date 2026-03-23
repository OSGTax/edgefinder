"""
EdgeFinder Module 2.5 Tests: News Sentiment Gate
==================================================
Tests cover: VADER scoring, action mapping, confidence adjustment,
RSS fetching (mocked), HTML cleaning, pipeline integration, and edge cases.

Run: python -m pytest tests/test_sentiment.py -v
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from modules.sentiment import (
    NewsArticle,
    SentimentResult,
    score_text,
    score_articles,
    determine_action,
    apply_sentiment_to_confidence,
    assess_sentiment,
    gate_trade,
    fetch_news_rss,
    _clean_html,
)
from config import settings


# ════════════════════════════════════════════════════════════
# VADER SCORING
# ════════════════════════════════════════════════════════════

class TestVADERScoring:
    """Test VADER sentiment analysis on text."""

    def test_positive_text(self):
        score = score_text("Company reports record earnings and amazing growth")
        assert score > 0.0

    def test_negative_text(self):
        score = score_text("Stock crashes amid fraud investigation and massive losses")
        assert score < 0.0

    def test_neutral_text(self):
        score = score_text("The meeting is scheduled for Tuesday")
        assert -0.3 <= score <= 0.3

    def test_strongly_positive(self):
        score = score_text("Incredible breakthrough! Outstanding results exceed all expectations!")
        assert score >= 0.5

    def test_strongly_negative(self):
        score = score_text("Terrible disaster! Company facing bankruptcy and total collapse!")
        assert score <= -0.5

    def test_empty_string(self):
        assert score_text("") == 0.0

    def test_whitespace_only(self):
        assert score_text("   ") == 0.0

    def test_none_input(self):
        assert score_text(None) == 0.0

    def test_score_range(self):
        """VADER compound scores must be in [-1, 1]."""
        texts = [
            "Best day ever!",
            "Worst day ever!",
            "Meeting at 3pm",
            "",
        ]
        for text in texts:
            score = score_text(text)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {text}"


class TestScoreArticles:
    """Test batch scoring of article lists."""

    def test_scores_all_articles(self):
        articles = [
            NewsArticle(title="Great earnings report"),
            NewsArticle(title="Terrible fraud scandal"),
            NewsArticle(title="Meeting scheduled"),
        ]
        scored = score_articles(articles)
        assert len(scored) == 3
        assert scored[0].compound_score > 0
        assert scored[1].compound_score < 0
        # All should have scores set
        for a in scored:
            assert a.compound_score != 0.0 or "scheduled" in a.title.lower()

    def test_empty_list(self):
        assert score_articles([]) == []

    def test_modifies_in_place(self):
        articles = [NewsArticle(title="Great news")]
        result = score_articles(articles)
        assert result is articles
        assert articles[0].compound_score > 0


# ════════════════════════════════════════════════════════════
# ACTION MAPPING
# ════════════════════════════════════════════════════════════

class TestActionMapping:
    """Test sentiment score → trading action mapping."""

    def test_strong_negative_blocks(self):
        action, reason = determine_action(-0.7)
        assert action == "BLOCK"

    def test_mild_negative_reduces(self):
        action, reason = determine_action(-0.3)
        assert action == "REDUCE_50"

    def test_neutral_proceeds(self):
        action, reason = determine_action(0.0)
        assert action == "PROCEED"

    def test_mild_positive_boosts_10(self):
        action, reason = determine_action(0.3)
        assert action == "CONFIDENCE_PLUS_10"

    def test_strong_positive_boosts_20(self):
        action, reason = determine_action(0.7)
        assert action == "CONFIDENCE_PLUS_20"

    def test_boundary_strong_negative(self):
        """Exactly at strong negative threshold should be BLOCK."""
        action, _ = determine_action(settings.SENTIMENT_STRONG_NEGATIVE)
        assert action == "BLOCK"

    def test_boundary_mild_negative(self):
        """Exactly at mild negative threshold should be REDUCE_50."""
        action, _ = determine_action(settings.SENTIMENT_MILD_NEGATIVE)
        assert action == "REDUCE_50"

    def test_boundary_mild_positive(self):
        """Just below mild positive threshold should be PROCEED."""
        action, _ = determine_action(settings.SENTIMENT_MILD_POSITIVE - 0.001)
        assert action == "PROCEED"

    def test_boundary_strong_positive(self):
        """Exactly at strong positive threshold should be CONFIDENCE_PLUS_20."""
        action, _ = determine_action(settings.SENTIMENT_STRONG_POSITIVE)
        assert action == "CONFIDENCE_PLUS_20"

    def test_extreme_negative(self):
        action, _ = determine_action(-1.0)
        assert action == "BLOCK"

    def test_extreme_positive(self):
        action, _ = determine_action(1.0)
        assert action == "CONFIDENCE_PLUS_20"

    def test_reason_includes_score(self):
        _, reason = determine_action(0.6)
        assert "0.600" in reason


# ════════════════════════════════════════════════════════════
# CONFIDENCE ADJUSTMENT
# ════════════════════════════════════════════════════════════

class TestConfidenceAdjustment:
    """Test how sentiment actions modify trade confidence."""

    def test_block_zeroes_confidence(self):
        assert apply_sentiment_to_confidence(80.0, "BLOCK") == 0.0

    def test_reduce_50_keeps_confidence(self):
        """REDUCE_50 doesn't change confidence — caller halves position size."""
        assert apply_sentiment_to_confidence(80.0, "REDUCE_50") == 80.0

    def test_proceed_no_change(self):
        assert apply_sentiment_to_confidence(65.0, "PROCEED") == 65.0

    def test_plus_10(self):
        assert apply_sentiment_to_confidence(70.0, "CONFIDENCE_PLUS_10") == 80.0

    def test_plus_20(self):
        assert apply_sentiment_to_confidence(70.0, "CONFIDENCE_PLUS_20") == 90.0

    def test_plus_10_capped_at_100(self):
        assert apply_sentiment_to_confidence(95.0, "CONFIDENCE_PLUS_10") == 100.0

    def test_plus_20_capped_at_100(self):
        assert apply_sentiment_to_confidence(90.0, "CONFIDENCE_PLUS_20") == 100.0

    def test_block_zeroes_even_high_confidence(self):
        assert apply_sentiment_to_confidence(100.0, "BLOCK") == 0.0

    def test_unknown_action_no_change(self):
        assert apply_sentiment_to_confidence(60.0, "UNKNOWN") == 60.0

    def test_zero_confidence_stays_zero(self):
        assert apply_sentiment_to_confidence(0.0, "PROCEED") == 0.0


# ════════════════════════════════════════════════════════════
# ASSESS SENTIMENT (PIPELINE)
# ════════════════════════════════════════════════════════════

class TestAssessSentiment:
    """Test the full sentiment assessment pipeline."""

    def test_positive_headlines_proceed_or_boost(self):
        headlines = [
            "Company beats earnings expectations",
            "Stock upgraded by major analysts",
            "Revenue growth exceeds forecasts",
        ]
        result = assess_sentiment("TEST", headlines=headlines)
        assert result.action in ("PROCEED", "CONFIDENCE_PLUS_10", "CONFIDENCE_PLUS_20")
        assert result.avg_compound > 0
        assert result.num_articles == 3

    def test_negative_headlines_block_or_reduce(self):
        headlines = [
            "Company under investigation for fraud",
            "Massive layoffs announced amid losses",
            "Stock crashes on terrible earnings miss",
        ]
        result = assess_sentiment("TEST", headlines=headlines)
        assert result.action in ("BLOCK", "REDUCE_50")
        assert result.avg_compound < 0

    def test_mixed_headlines_moderate(self):
        headlines = [
            "Company reports strong growth",
            "But faces regulatory headwinds",
        ]
        result = assess_sentiment("TEST", headlines=headlines)
        assert result.ticker == "TEST"
        assert result.num_articles == 2
        assert len(result.articles) == 2

    def test_no_headlines_neutral(self):
        result = assess_sentiment("TEST", headlines=[])
        assert result.action == "PROCEED"
        assert result.avg_compound == 0.0
        assert result.num_articles == 0
        assert "No recent news" in result.reason

    def test_none_headlines_fetches_rss(self):
        """When headlines=None, should try RSS fetch."""
        with patch("modules.sentiment.fetch_news_rss") as mock_fetch:
            mock_fetch.return_value = [
                NewsArticle(title="Good earnings report"),
            ]
            result = assess_sentiment("AAPL", headlines=None, feeds=["fake"])
            mock_fetch.assert_called_once()
            assert result.num_articles == 1

    def test_result_has_timestamp(self):
        result = assess_sentiment("TEST", headlines=["Neutral news"])
        assert result.timestamp is not None

    def test_empty_string_headlines_filtered(self):
        result = assess_sentiment("TEST", headlines=["Good news", "", "  "])
        assert result.num_articles == 1

    def test_articles_have_scores(self):
        result = assess_sentiment("TEST", headlines=["Great results!", "Terrible loss"])
        for article in result.articles:
            assert article.compound_score != 0.0


# ════════════════════════════════════════════════════════════
# GATE TRADE (CONVENIENCE FUNCTION)
# ════════════════════════════════════════════════════════════

class TestGateTrade:
    """Test the trade gating convenience function."""

    def test_block_trade(self):
        headlines = [
            "Fraud investigation launched",
            "SEC charges company with deception",
            "Stock plummets on scandal",
        ]
        action, adjusted, result = gate_trade("BAD", 80.0, headlines=headlines)
        assert action == "BLOCK"
        assert adjusted == 0.0

    def test_boost_trade(self):
        headlines = [
            "Incredible earnings beat!",
            "Analysts upgrade to strong buy!",
            "Record revenue growth announced!",
        ]
        action, adjusted, result = gate_trade("GOOD", 70.0, headlines=headlines)
        assert action in ("CONFIDENCE_PLUS_10", "CONFIDENCE_PLUS_20")
        assert adjusted > 70.0

    def test_proceed_no_change(self):
        headlines = ["Meeting scheduled for Tuesday"]
        action, adjusted, result = gate_trade("MEH", 65.0, headlines=headlines)
        # Neutral headline
        assert adjusted >= 65.0  # Either unchanged or slightly boosted

    def test_returns_full_result(self):
        action, adjusted, result = gate_trade("X", 50.0, headlines=["Neutral"])
        assert isinstance(result, SentimentResult)
        assert result.ticker == "X"


# ════════════════════════════════════════════════════════════
# RSS FETCHING (MOCKED)
# ════════════════════════════════════════════════════════════

class TestRSSFetching:
    """Test RSS news fetching with mocked feedparser."""

    def test_fetch_parses_entries(self):
        mock_entry = MagicMock()
        mock_entry.get = lambda k, d="": {
            "title": "Test headline about stock",
            "link": "https://example.com/article",
        }.get(k, d)
        mock_entry.published_parsed = None

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_feed.feed = {"title": "Test Feed"}

        with patch("modules.sentiment.feedparser.parse", return_value=mock_feed):
            articles = fetch_news_rss("AAPL", feeds=["https://test.com/{ticker}"])
            assert len(articles) == 1
            assert articles[0].title == "Test headline about stock"

    def test_fetch_filters_old_articles(self):
        """Articles older than lookback window should be excluded."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=72)

        mock_entry = MagicMock()
        mock_entry.get = lambda k, d="": {
            "title": "Old article",
            "link": "https://example.com",
        }.get(k, d)
        mock_entry.published_parsed = old_time.timetuple()

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_feed.feed = {"title": "Test Feed"}

        with patch("modules.sentiment.feedparser.parse", return_value=mock_feed):
            articles = fetch_news_rss("AAPL", feeds=["https://test.com/{ticker}"],
                                       lookback_hours=48)
            assert len(articles) == 0

    def test_fetch_keeps_recent_articles(self):
        """Articles within lookback window should be included."""
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_entry = MagicMock()
        mock_entry.get = lambda k, d="": {
            "title": "Recent article",
            "link": "https://example.com",
        }.get(k, d)
        mock_entry.published_parsed = recent_time.timetuple()

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_feed.feed = {"title": "Test Feed"}

        with patch("modules.sentiment.feedparser.parse", return_value=mock_feed):
            articles = fetch_news_rss("AAPL", feeds=["https://test.com/{ticker}"],
                                       lookback_hours=48)
            assert len(articles) == 1

    def test_fetch_handles_bad_feed(self):
        """Failed RSS feed should not crash — returns empty list."""
        with patch("modules.sentiment.feedparser.parse", side_effect=Exception("Network error")):
            articles = fetch_news_rss("AAPL", feeds=["https://broken.com/{ticker}"])
            assert articles == []

    def test_fetch_substitutes_ticker(self):
        """Feed URL template should have {ticker} replaced."""
        with patch("modules.sentiment.feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[], feed={})
            fetch_news_rss("MSFT", feeds=["https://feed.com/rss?s={ticker}"])
            mock_parse.assert_called_once_with("https://feed.com/rss?s=MSFT")

    def test_fetch_empty_title_skipped(self):
        """Entries with empty titles should be skipped."""
        mock_entry = MagicMock()
        mock_entry.get = lambda k, d="": {
            "title": "",
            "link": "https://example.com",
        }.get(k, d)
        mock_entry.published_parsed = None

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_feed.feed = {"title": "Test Feed"}

        with patch("modules.sentiment.feedparser.parse", return_value=mock_feed):
            articles = fetch_news_rss("AAPL", feeds=["https://test.com/{ticker}"])
            assert len(articles) == 0


# ════════════════════════════════════════════════════════════
# HTML CLEANING
# ════════════════════════════════════════════════════════════

class TestHTMLCleaning:
    """Test HTML tag stripping from RSS content."""

    def test_plain_text_unchanged(self):
        assert _clean_html("Hello world") == "Hello world"

    def test_html_stripped(self):
        assert _clean_html("<b>Bold</b> text") == "Bold text"

    def test_complex_html(self):
        result = _clean_html('<a href="link">Click <b>here</b></a>')
        assert "Click" in result
        assert "here" in result
        assert "<" not in result

    def test_empty_string(self):
        assert _clean_html("") == ""

    def test_none_input(self):
        assert _clean_html(None) == ""

    def test_whitespace_trimmed(self):
        assert _clean_html("  spaced  ") == "spaced"


# ════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_single_article_decides_action(self):
        result = assess_sentiment("X", headlines=["Total catastrophe and collapse!"])
        assert result.num_articles == 1
        assert result.avg_compound < 0

    def test_many_articles_averaged(self):
        """Average of many positive + one negative should still be positive."""
        headlines = [
            "Great earnings!",
            "Revenue beats estimates!",
            "Strong guidance!",
            "Amazing growth!",
            "Minor concerns about costs",  # One mild negative
        ]
        result = assess_sentiment("X", headlines=headlines)
        assert result.avg_compound > 0

    def test_all_neutral_articles(self):
        headlines = [
            "Meeting at 3pm",
            "Report due Friday",
            "Conference scheduled",
        ]
        result = assess_sentiment("X", headlines=headlines)
        assert result.action == "PROCEED"

    def test_exact_zero_compound(self):
        """Zero average should map to PROCEED."""
        action, _ = determine_action(0.0)
        assert action == "PROCEED"

    def test_assess_returns_correct_ticker(self):
        result = assess_sentiment("TSLA", headlines=["Neutral"])
        assert result.ticker == "TSLA"

    def test_gate_with_zero_confidence(self):
        action, adjusted, result = gate_trade("X", 0.0, headlines=["Good news!"])
        assert adjusted >= 0.0

    def test_gate_with_max_confidence(self):
        action, adjusted, result = gate_trade("X", 100.0, headlines=["Good news!"])
        assert adjusted <= 100.0

    def test_avg_compound_rounded(self):
        result = assess_sentiment("X", headlines=["Somewhat positive outlook"])
        # avg_compound should be rounded to 4 decimal places
        s = str(result.avg_compound)
        if "." in s:
            decimals = len(s.split(".")[-1])
            assert decimals <= 4


# ════════════════════════════════════════════════════════════
# INTEGRATION TEST (hits real API — skip in CI)
# ════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that hit real RSS feeds.
    Run with: python -m pytest tests/test_sentiment.py -v -m integration
    Skip with: python -m pytest tests/test_sentiment.py -v -m "not integration"
    """

    def test_fetch_real_rss(self):
        """Fetch real Yahoo Finance RSS for AAPL."""
        articles = fetch_news_rss("AAPL")
        # Yahoo may return 0 articles, but should not crash
        assert isinstance(articles, list)
        for article in articles:
            assert isinstance(article, NewsArticle)
            assert article.title

    def test_full_pipeline_real(self):
        """Full sentiment pipeline on a real ticker."""
        result = assess_sentiment("MSFT")
        assert isinstance(result, SentimentResult)
        assert result.ticker == "MSFT"
        assert result.action in (
            "BLOCK", "REDUCE_50", "PROCEED",
            "CONFIDENCE_PLUS_10", "CONFIDENCE_PLUS_20",
        )
        assert -1.0 <= result.avg_compound <= 1.0

    def test_gate_real_trade(self):
        """Gate a trade using real news data."""
        action, adjusted, result = gate_trade("GOOGL", 75.0)
        assert action in (
            "BLOCK", "REDUCE_50", "PROCEED",
            "CONFIDENCE_PLUS_10", "CONFIDENCE_PLUS_20",
        )
        assert 0.0 <= adjusted <= 100.0


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_sentiment.py -v
#
# Expected results:
#   TestVADERScoring:          9 tests  — all should PASS
#   TestScoreArticles:         3 tests  — all should PASS
#   TestActionMapping:        12 tests  — all should PASS
#   TestConfidenceAdjustment: 10 tests  — all should PASS
#   TestAssessSentiment:       8 tests  — all should PASS
#   TestGateTrade:             4 tests  — all should PASS
#   TestRSSFetching:           6 tests  — all should PASS
#   TestHTMLCleaning:          6 tests  — all should PASS
#   TestEdgeCases:             8 tests  — all should PASS
#   TestIntegration:           3 tests  — may skip if no network
#
# TOTAL: 69 tests
#
# If any test in TestActionMapping, TestConfidenceAdjustment,
# or TestAssessSentiment fails, DO NOT proceed to Module 3-4.
# Fix the sentiment logic first.
# ════════════════════════════════════════════════════════════
