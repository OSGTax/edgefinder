"""Tests for edgefinder/sentiment/."""

from unittest.mock import MagicMock, patch

import pytest

from edgefinder.core.models import SentimentAction, SentimentSource, TickerSentiment
from edgefinder.db.models import SentimentReading
from edgefinder.sentiment.aggregator import SentimentAggregator
from edgefinder.sentiment.news_rss import NewsSentimentProvider
from edgefinder.sentiment.provider import map_score_to_action
from edgefinder.sentiment.reddit import RedditSentimentProvider
from edgefinder.sentiment.twitter import TwitterSentimentProvider


# ── Action Mapping Tests ─────────────────────────────────


class TestMapScoreToAction:
    def test_strong_negative_blocks(self):
        assert map_score_to_action(-0.6) == SentimentAction.BLOCK

    def test_mild_negative_reduces(self):
        assert map_score_to_action(-0.3) == SentimentAction.REDUCE_50

    def test_neutral_proceeds(self):
        assert map_score_to_action(0.0) == SentimentAction.PROCEED

    def test_mild_positive_boosts_10(self):
        assert map_score_to_action(0.3) == SentimentAction.CONFIDENCE_PLUS_10

    def test_strong_positive_boosts_20(self):
        assert map_score_to_action(0.6) == SentimentAction.CONFIDENCE_PLUS_20


# ── News RSS Tests ───────────────────────────────────────


class TestNewsSentiment:
    def test_score_text_positive(self):
        provider = NewsSentimentProvider()
        score = provider._score_text("Stock surge rally beat expectations strong growth")
        assert score > 0

    def test_score_text_negative(self):
        provider = NewsSentimentProvider()
        score = provider._score_text("Stock crash plunge loss decline weak")
        assert score < 0

    def test_score_text_neutral(self):
        provider = NewsSentimentProvider()
        score = provider._score_text("Company announces quarterly results")
        assert score == 0.0

    def test_parse_rss(self):
        xml = """
        <rss><channel>
            <item><title>AAPL hits record high</title></item>
            <item><title>Market update for today</title></item>
        </channel></rss>
        """
        from datetime import datetime
        items = NewsSentimentProvider._parse_rss(xml, datetime(2020, 1, 1))
        assert len(items) == 2
        assert items[0]["title"] == "AAPL hits record high"

    def test_get_sentiment_no_articles(self):
        provider = NewsSentimentProvider()
        with patch.object(provider, "_fetch_articles", return_value=[]):
            result = provider.get_sentiment("AAPL")
            assert result.score == 0.0
            assert result.mention_count == 0


# ── Reddit Tests ─────────────────────────────────────────


class TestRedditSentiment:
    def test_extract_tickers_dollar_sign(self):
        tickers = RedditSentimentProvider._extract_tickers("I love $AAPL and $TSLA!")
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_extract_tickers_caps(self):
        tickers = RedditSentimentProvider._extract_tickers("NVDA is going to the moon")
        assert "NVDA" in tickers

    def test_extract_tickers_filters_noise(self):
        tickers = RedditSentimentProvider._extract_tickers("THE CEO said YOLO on this DD")
        assert "THE" not in tickers
        assert "CEO" not in tickers
        assert "YOLO" not in tickers

    def test_score_post_high_upvote(self):
        post = {"upvote_ratio": 0.9, "score": 1000, "num_comments": 50}
        score = RedditSentimentProvider._score_post(post)
        assert score > 0

    def test_score_post_low_upvote(self):
        post = {"upvote_ratio": 0.2, "score": 100, "num_comments": 10}
        score = RedditSentimentProvider._score_post(post)
        assert score < 0

    def test_score_post_neutral(self):
        post = {"upvote_ratio": 0.5, "score": 50, "num_comments": 5}
        score = RedditSentimentProvider._score_post(post)
        assert score == 0.0

    def test_get_sentiment_no_results(self):
        provider = RedditSentimentProvider()
        with patch.object(provider, "_search_subreddit", return_value=[]):
            result = provider.get_sentiment("AAPL")
            assert result.score == 0.0
            assert result.mention_count == 0


# ── Twitter Tests ────────────────────────────────────────


class TestTwitterSentiment:
    def test_returns_neutral(self):
        provider = TwitterSentimentProvider()
        result = provider.get_sentiment("AAPL")
        assert result.score == 0.0
        assert result.source == SentimentSource.TWITTER

    def test_trending_empty(self):
        provider = TwitterSentimentProvider()
        assert provider.get_trending() == []


# ── Aggregator Tests ─────────────────────────────────────


class TestSentimentAggregator:
    def test_aggregates_sources(self):
        agg = SentimentAggregator()
        # Mock all providers to return known scores
        with patch.object(agg._providers[0], "get_sentiment", return_value=TickerSentiment(
            symbol="AAPL", source=SentimentSource.NEWS, score=0.5, mention_count=10,
        )), patch.object(agg._providers[1], "get_sentiment", return_value=TickerSentiment(
            symbol="AAPL", source=SentimentSource.REDDIT, score=0.3, mention_count=20,
        )), patch.object(agg._providers[2], "get_sentiment", return_value=TickerSentiment(
            symbol="AAPL", source=SentimentSource.TWITTER, score=0.0, mention_count=0,
        )):
            result = agg.get_sentiment("AAPL")
            assert result.composite_score > 0
            assert result.total_mentions == 30
            assert "news" in result.source_scores
            assert "reddit" in result.source_scores

    def test_gate_trade_block(self):
        agg = SentimentAggregator()
        with patch.object(agg, "get_sentiment", return_value=MagicMock(
            action=SentimentAction.BLOCK, composite_score=-0.7,
        )):
            action, conf, _ = agg.gate_trade("AAPL", 70.0)
            assert action == SentimentAction.BLOCK
            assert conf == 0.0

    def test_gate_trade_proceed(self):
        agg = SentimentAggregator()
        with patch.object(agg, "get_sentiment", return_value=MagicMock(
            action=SentimentAction.PROCEED, composite_score=0.0,
        )):
            action, conf, _ = agg.gate_trade("AAPL", 70.0)
            assert action == SentimentAction.PROCEED
            assert conf == 70.0

    def test_gate_trade_boost(self):
        agg = SentimentAggregator()
        with patch.object(agg, "get_sentiment", return_value=MagicMock(
            action=SentimentAction.CONFIDENCE_PLUS_20, composite_score=0.6,
        )):
            action, conf, _ = agg.gate_trade("AAPL", 70.0)
            assert action == SentimentAction.CONFIDENCE_PLUS_20
            assert conf == 90.0

    def test_confidence_caps_at_100(self):
        adjusted = SentimentAggregator._adjust_confidence(95, SentimentAction.CONFIDENCE_PLUS_20)
        assert adjusted == 100

    def test_persist_readings(self, db_session):
        agg = SentimentAggregator(session=db_session)
        with patch.object(agg._providers[0], "get_sentiment", return_value=TickerSentiment(
            symbol="AAPL", source=SentimentSource.NEWS, score=0.5, mention_count=5,
        )), patch.object(agg._providers[1], "get_sentiment", return_value=TickerSentiment(
            symbol="AAPL", source=SentimentSource.REDDIT, score=0.3, mention_count=10,
        )), patch.object(agg._providers[2], "get_sentiment", return_value=TickerSentiment(
            symbol="AAPL", source=SentimentSource.TWITTER, score=0.0, mention_count=0,
        )):
            agg.get_sentiment("AAPL")
            count = db_session.query(SentimentReading).count()
            assert count == 3

    def test_provider_failure_handled(self):
        agg = SentimentAggregator()
        with patch.object(agg._providers[0], "get_sentiment", side_effect=Exception("fail")), \
             patch.object(agg._providers[1], "get_sentiment", return_value=TickerSentiment(
                 symbol="AAPL", source=SentimentSource.REDDIT, score=0.5, mention_count=5,
             )), \
             patch.object(agg._providers[2], "get_sentiment", return_value=TickerSentiment(
                 symbol="AAPL", source=SentimentSource.TWITTER, score=0.0, mention_count=0,
             )):
            result = agg.get_sentiment("AAPL")
            assert result is not None
            assert "reddit" in result.source_scores
