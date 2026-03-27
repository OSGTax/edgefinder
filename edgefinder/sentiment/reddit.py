"""EdgeFinder v2 — Reddit sentiment provider.

Monitors stock-related subreddits for ticker mentions and sentiment.
Uses Reddit's public JSON API (no authentication required for read-only).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import httpx

from config.settings import settings
from edgefinder.core.models import SentimentSource, TickerSentiment

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]
REDDIT_BASE = "https://www.reddit.com"
USER_AGENT = "EdgeFinder/2.0 (Stock Research Tool)"


class RedditSentimentProvider:
    """Reddit sentiment provider using public JSON API."""

    @property
    def source_name(self) -> str:
        return "reddit"

    def get_sentiment(self, ticker: str) -> TickerSentiment | None:
        """Search Reddit for ticker mentions and score sentiment."""
        mentions = []

        for subreddit in SUBREDDITS:
            posts = self._search_subreddit(subreddit, ticker)
            mentions.extend(posts)

        if not mentions:
            return TickerSentiment(
                symbol=ticker,
                source=SentimentSource.REDDIT,
                score=0.0,
                mention_count=0,
            )

        scores = [self._score_post(p) for p in mentions]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return TickerSentiment(
            symbol=ticker,
            source=SentimentSource.REDDIT,
            score=max(-1.0, min(1.0, avg_score)),
            mention_count=len(mentions),
            is_trending=len(mentions) > 20,
            summary=mentions[0].get("title", "") if mentions else None,
        )

    def get_trending(self) -> list[TickerSentiment]:
        """Get trending tickers from r/wallstreetbets hot posts."""
        tickers: dict[str, list[dict]] = {}

        posts = self._get_hot_posts("wallstreetbets", limit=50)
        for post in posts:
            found = self._extract_tickers(post.get("title", ""))
            for t in found:
                if t not in tickers:
                    tickers[t] = []
                tickers[t].append(post)

        results = []
        for symbol, posts in sorted(tickers.items(), key=lambda x: -len(x[1])):
            if len(posts) < 3:
                continue
            scores = [self._score_post(p) for p in posts]
            avg = sum(scores) / len(scores) if scores else 0.0
            results.append(TickerSentiment(
                symbol=symbol,
                source=SentimentSource.REDDIT,
                score=max(-1.0, min(1.0, avg)),
                mention_count=len(posts),
                is_trending=True,
            ))

        return results[:20]

    def _search_subreddit(self, subreddit: str, ticker: str) -> list[dict]:
        """Search a subreddit for ticker mentions."""
        url = f"{REDDIT_BASE}/r/{subreddit}/search.json"
        params = {
            "q": ticker,
            "restrict_sr": "on",
            "sort": "new",
            "t": "week",
            "limit": 25,
        }
        try:
            resp = httpx.get(
                url, params=params, timeout=10,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            posts = []
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                posts.append({
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.5),
                })
            return posts
        except Exception as e:
            logger.warning("Reddit search failed for %s in r/%s: %s", ticker, subreddit, e)
            return []

    def _get_hot_posts(self, subreddit: str, limit: int = 50) -> list[dict]:
        """Get hot posts from a subreddit."""
        url = f"{REDDIT_BASE}/r/{subreddit}/hot.json"
        try:
            resp = httpx.get(
                url, params={"limit": limit}, timeout=10,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [
                child.get("data", {})
                for child in data.get("data", {}).get("children", [])
            ]
        except Exception as e:
            logger.warning("Reddit hot posts failed for r/%s: %s", subreddit, e)
            return []

    @staticmethod
    def _score_post(post: dict) -> float:
        """Score a Reddit post based on engagement and upvote ratio.

        High upvotes + high ratio = bullish. Low ratio = bearish/controversial.
        """
        ratio = post.get("upvote_ratio", 0.5)
        score = post.get("score", 0)

        # Map upvote ratio to sentiment: 0.5 = neutral, >0.7 = bullish, <0.3 = bearish
        sentiment = (ratio - 0.5) * 2  # maps 0-1 to -1 to +1

        # Boost by engagement (more comments = more conviction either way)
        comments = post.get("num_comments", 0)
        if comments > 100:
            sentiment *= 1.2

        return max(-1.0, min(1.0, sentiment))

    @staticmethod
    def _extract_tickers(text: str) -> list[str]:
        """Extract potential stock tickers from text ($AAPL or standalone caps)."""
        # Match $TICKER pattern
        dollar_tickers = re.findall(r"\$([A-Z]{1,5})\b", text)
        # Match standalone 2-5 letter uppercase words that look like tickers
        caps = re.findall(r"\b([A-Z]{2,5})\b", text)
        # Filter common English words
        noise = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                 "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET",
                 "HAS", "HIM", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE",
                 "WAY", "WHO", "DID", "ITS", "LET", "SAY", "SHE", "TOO",
                 "USE", "IMO", "YOLO", "DD", "CEO", "IPO", "ETF", "GDP",
                 "USA", "USD", "SEC", "FDA", "NFT", "AI", "ATH", "EOD"}
        caps = [t for t in caps if t not in noise and len(t) >= 2]
        return list(set(dollar_tickers + caps))
