"""EdgeFinder v2 — Sentiment aggregator.

Combines scores from multiple sources (Reddit, Twitter, News) into
a weighted composite score. Stores readings to DB. Gates trades.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.models import (
    AggregatedSentiment,
    SentimentAction,
    TickerSentiment,
)
from edgefinder.db.models import SentimentReading
from edgefinder.sentiment.news_rss import NewsSentimentProvider
from edgefinder.sentiment.provider import map_score_to_action
from edgefinder.sentiment.reddit import RedditSentimentProvider
from edgefinder.sentiment.twitter import TwitterSentimentProvider

logger = logging.getLogger(__name__)

# Source weights for composite score
SOURCE_WEIGHTS = {
    "reddit": 0.40,
    "twitter": 0.20,
    "news": 0.40,
}


class SentimentAggregator:
    """Combines sentiment from all sources into a single assessment."""

    def __init__(self, session: Session | None = None) -> None:
        self._providers = [
            NewsSentimentProvider(),
            RedditSentimentProvider(),
            TwitterSentimentProvider(),
        ]
        self._session = session

    def get_sentiment(self, ticker: str) -> AggregatedSentiment:
        """Get aggregated sentiment for a ticker from all sources."""
        source_scores: dict[str, float] = {}
        total_mentions = 0
        is_trending = False
        readings: list[TickerSentiment] = []

        for provider in self._providers:
            try:
                reading = provider.get_sentiment(ticker)
                if reading:
                    readings.append(reading)
                    source_scores[reading.source.value] = reading.score
                    total_mentions += reading.mention_count
                    if reading.is_trending:
                        is_trending = True
            except Exception as e:
                logger.warning(
                    "Sentiment provider '%s' failed for %s: %s",
                    provider.source_name, ticker, e,
                )

        # Compute weighted composite
        composite = 0.0
        total_weight = 0.0
        for source, score in source_scores.items():
            weight = SOURCE_WEIGHTS.get(source, 0.0)
            composite += score * weight
            total_weight += weight

        if total_weight > 0:
            composite /= total_weight

        composite = max(-1.0, min(1.0, composite))
        action = map_score_to_action(composite)

        # Persist readings to DB
        if self._session:
            self._persist_readings(readings)

        return AggregatedSentiment(
            symbol=ticker,
            composite_score=round(composite, 4),
            source_scores=source_scores,
            total_mentions=total_mentions,
            is_trending=is_trending,
            action=action,
        )

    def gate_trade(
        self, ticker: str, confidence: float
    ) -> tuple[SentimentAction, float, AggregatedSentiment]:
        """Gate a trade through sentiment analysis.

        Returns (action, adjusted_confidence, full_sentiment).
        """
        sentiment = self.get_sentiment(ticker)
        adjusted = self._adjust_confidence(confidence, sentiment.action)
        return sentiment.action, adjusted, sentiment

    def get_trending(self) -> list[TickerSentiment]:
        """Get trending tickers across all sources."""
        all_trending: list[TickerSentiment] = []
        for provider in self._providers:
            try:
                trending = provider.get_trending()
                all_trending.extend(trending)
            except Exception as e:
                logger.warning(
                    "Trending fetch failed for '%s': %s",
                    provider.source_name, e,
                )
        return all_trending

    @staticmethod
    def _adjust_confidence(confidence: float, action: SentimentAction) -> float:
        """Adjust signal confidence based on sentiment action."""
        if action == SentimentAction.BLOCK:
            return 0.0
        if action == SentimentAction.REDUCE_50:
            return confidence  # Don't reduce confidence, reduce position size instead
        if action == SentimentAction.CONFIDENCE_PLUS_10:
            return min(100, confidence + 10)
        if action == SentimentAction.CONFIDENCE_PLUS_20:
            return min(100, confidence + 20)
        return confidence  # PROCEED

    def _persist_readings(self, readings: list[TickerSentiment]) -> None:
        """Store sentiment readings to database."""
        for reading in readings:
            record = SentimentReading(
                symbol=reading.symbol,
                source=reading.source.value,
                score=reading.score,
                mention_count=reading.mention_count,
                is_trending=reading.is_trending,
                timestamp=datetime.now(timezone.utc),
            )
            self._session.add(record)
        try:
            self._session.commit()
        except Exception:
            logger.exception("Failed to persist sentiment readings")
            self._session.rollback()
