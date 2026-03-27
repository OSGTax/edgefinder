"""EdgeFinder v2 — Twitter/X sentiment provider.

Monitors financial Twitter for ticker mentions and sentiment.
Uses public scraping approach since Twitter API is expensive.

Note: Twitter's API pricing makes it impractical for most use cases.
This provider uses a simplified approach that can be upgraded later
when/if API access is obtained.
"""

from __future__ import annotations

import logging

from edgefinder.core.models import SentimentSource, TickerSentiment

logger = logging.getLogger(__name__)


class TwitterSentimentProvider:
    """Twitter/X sentiment provider.

    Currently a stub that returns neutral sentiment.
    Designed to be upgraded with real Twitter API access later.
    The interface matches SentimentProvider protocol so it plugs
    into the aggregator seamlessly.
    """

    @property
    def source_name(self) -> str:
        return "twitter"

    def get_sentiment(self, ticker: str) -> TickerSentiment | None:
        """Get sentiment for a ticker from Twitter/X.

        Currently returns neutral — upgrade when Twitter API access is obtained.
        """
        logger.debug("Twitter sentiment not yet implemented for %s", ticker)
        return TickerSentiment(
            symbol=ticker,
            source=SentimentSource.TWITTER,
            score=0.0,
            mention_count=0,
            is_trending=False,
            summary="Twitter sentiment not yet connected",
        )

    def get_trending(self) -> list[TickerSentiment]:
        """Get trending tickers from Twitter/X.

        Currently returns empty — upgrade when Twitter API access is obtained.
        """
        return []
