"""EdgeFinder v2 — News RSS sentiment provider.

Fetches news headlines from RSS feeds and scores them using
a keyword-based sentiment approach (no external NLP dependency).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import httpx

from config.settings import settings
from edgefinder.core.models import SentimentSource, TickerSentiment

logger = logging.getLogger(__name__)

# Keyword-based scoring (no VADER dependency)
POSITIVE_WORDS = {
    "surge", "soar", "rally", "beat", "upgrade", "bullish", "profit",
    "growth", "record", "strong", "boost", "outperform", "breakout",
    "buy", "positive", "gain", "rise", "jump", "climb", "high",
    "expand", "recover", "momentum", "optimistic", "innovation",
}
NEGATIVE_WORDS = {
    "crash", "plunge", "drop", "miss", "downgrade", "bearish", "loss",
    "decline", "weak", "cut", "underperform", "sell", "negative",
    "fall", "sink", "low", "concern", "warning", "risk", "fear",
    "lawsuit", "fraud", "investigation", "recall", "bankruptcy",
}

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]


class NewsSentimentProvider:
    """News RSS sentiment provider."""

    @property
    def source_name(self) -> str:
        return "news"

    def get_sentiment(self, ticker: str) -> TickerSentiment | None:
        """Fetch and score recent news for a ticker."""
        articles = self._fetch_articles(ticker)
        if not articles:
            return TickerSentiment(
                symbol=ticker,
                source=SentimentSource.NEWS,
                score=0.0,
                mention_count=0,
            )

        scores = [self._score_text(a["title"]) for a in articles]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return TickerSentiment(
            symbol=ticker,
            source=SentimentSource.NEWS,
            score=max(-1.0, min(1.0, avg_score)),
            mention_count=len(articles),
            is_trending=len(articles) > 10,
            summary=articles[0]["title"] if articles else None,
        )

    def get_trending(self) -> list[TickerSentiment]:
        """Not implemented for RSS — would require scanning many feeds."""
        return []

    def _fetch_articles(self, ticker: str) -> list[dict]:
        """Fetch articles from RSS feeds within lookback window."""
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=settings.sentiment_lookback_hours)

        for feed_template in RSS_FEEDS:
            url = feed_template.format(ticker=ticker)
            try:
                resp = httpx.get(url, timeout=10, follow_redirects=True)
                if resp.status_code != 200:
                    continue
                articles.extend(self._parse_rss(resp.text, cutoff))
            except Exception as e:
                logger.warning("Failed to fetch RSS for %s: %s", ticker, e)

        return articles

    @staticmethod
    def _parse_rss(xml_text: str, cutoff: datetime) -> list[dict]:
        """Simple XML parsing for RSS <item> elements."""
        items = []
        # Basic regex parsing — no feedparser dependency
        for match in re.finditer(r"<item>(.*?)</item>", xml_text, re.DOTALL):
            item_text = match.group(1)
            title_match = re.search(r"<title>(.*?)</title>", item_text, re.DOTALL)
            if not title_match:
                continue
            title = title_match.group(1).strip()
            # Clean HTML entities
            title = re.sub(r"&amp;", "&", title)
            title = re.sub(r"&lt;", "<", title)
            title = re.sub(r"&gt;", ">", title)
            title = re.sub(r"<[^>]+>", "", title)
            items.append({"title": title})
        return items

    @staticmethod
    def _score_text(text: str) -> float:
        """Keyword-based sentiment scoring (-1 to +1)."""
        words = set(text.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total
