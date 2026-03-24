"""
EdgeFinder Module 2.5: News Sentiment Gate
===========================================
Go/no-go filter applied before trade execution. Fetches recent news
headlines via RSS feeds, scores them with VADER sentiment analysis,
and returns an action decision:

    BLOCK             → Strong negative news, do not trade
    REDUCE_50         → Mild negative news, cut position size 50%
    PROCEED           → Neutral, no change
    CONFIDENCE_PLUS_10 → Mild positive news, boost confidence 10%
    CONFIDENCE_PLUS_20 → Strong positive news, boost confidence 20%

Runs inline whenever a trade signal is about to execute.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import settings

logger = logging.getLogger(__name__)

# Module-level VADER instance (stateless, safe to reuse)
_analyzer = SentimentIntensityAnalyzer()


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class NewsArticle:
    """A single news article/headline with its sentiment score."""
    title: str
    source: str = ""
    published: Optional[datetime] = None
    url: str = ""
    compound_score: float = 0.0  # VADER compound score (-1 to +1)


@dataclass
class SentimentResult:
    """The aggregated sentiment assessment for a ticker."""
    ticker: str
    action: str = "PROCEED"              # BLOCK, REDUCE_50, PROCEED, etc.
    avg_compound: float = 0.0            # Average VADER compound score
    num_articles: int = 0
    articles: list[NewsArticle] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""                     # Human-readable explanation


# ── VADER SCORING ────────────────────────────────────────────

def score_text(text: str) -> float:
    """
    Score a text string using VADER sentiment analysis.

    Args:
        text: The headline or article text to analyze.

    Returns:
        VADER compound score from -1.0 (most negative) to +1.0 (most positive).
    """
    if not text or not text.strip():
        return 0.0

    scores = _analyzer.polarity_scores(text)
    return scores["compound"]


def score_articles(articles: list[NewsArticle]) -> list[NewsArticle]:
    """
    Score a list of NewsArticle objects in place and return them.

    Each article's compound_score is set to the VADER compound
    score of its title.
    """
    for article in articles:
        article.compound_score = score_text(article.title)
    return articles


# ── ACTION MAPPING ───────────────────────────────────────────

def determine_action(avg_compound: float) -> tuple[str, str]:
    """
    Map an average compound sentiment score to a trading action.

    Uses threshold bands from settings.py:
        <= STRONG_NEGATIVE  → BLOCK
        <= MILD_NEGATIVE    → REDUCE_50
        < MILD_POSITIVE     → PROCEED
        < STRONG_POSITIVE   → CONFIDENCE_PLUS_10
        >= STRONG_POSITIVE  → CONFIDENCE_PLUS_20

    Args:
        avg_compound: Average VADER compound score across articles.

    Returns:
        Tuple of (action_string, human_readable_reason).
    """
    if avg_compound <= settings.SENTIMENT_STRONG_NEGATIVE:
        return (
            settings.SENTIMENT_STRONG_NEG_ACTION,
            f"Strong negative sentiment ({avg_compound:.3f} <= {settings.SENTIMENT_STRONG_NEGATIVE})",
        )

    if avg_compound <= settings.SENTIMENT_MILD_NEGATIVE:
        return (
            settings.SENTIMENT_MILD_NEG_ACTION,
            f"Mild negative sentiment ({avg_compound:.3f} <= {settings.SENTIMENT_MILD_NEGATIVE})",
        )

    if avg_compound < settings.SENTIMENT_MILD_POSITIVE:
        return (
            settings.SENTIMENT_NEUTRAL_ACTION,
            f"Neutral sentiment ({avg_compound:.3f})",
        )

    if avg_compound < settings.SENTIMENT_STRONG_POSITIVE:
        return (
            settings.SENTIMENT_MILD_POS_ACTION,
            f"Mild positive sentiment ({avg_compound:.3f})",
        )

    return (
        settings.SENTIMENT_STRONG_POS_ACTION,
        f"Strong positive sentiment ({avg_compound:.3f} >= {settings.SENTIMENT_STRONG_POSITIVE})",
    )


def apply_sentiment_to_confidence(
    confidence: float,
    action: str,
) -> float:
    """
    Adjust a trade's confidence score based on the sentiment action.

    Args:
        confidence: Original confidence score (0-100).
        action: The sentiment action string.

    Returns:
        Adjusted confidence score, clamped to 0-100.
        Returns 0.0 if action is BLOCK (signal should not trade).
    """
    if action == "BLOCK":
        return 0.0

    if action == "REDUCE_50":
        # Don't change confidence, but caller should halve position size
        return confidence

    if action == "CONFIDENCE_PLUS_10":
        return min(100.0, confidence + 10.0)

    if action == "CONFIDENCE_PLUS_20":
        return min(100.0, confidence + 20.0)

    # PROCEED or unknown → no change
    return confidence


# ── NEWS FETCHING ────────────────────────────────────────────

def fetch_news_rss(
    ticker: str,
    feeds: Optional[list[str]] = None,
    lookback_hours: Optional[int] = None,
) -> list[NewsArticle]:
    """
    Fetch recent news articles for a ticker from RSS feeds.

    Args:
        ticker: Stock ticker symbol.
        feeds: List of RSS feed URL templates with {ticker} placeholder.
               Defaults to settings.SENTIMENT_RSS_FEEDS.
        lookback_hours: Only include articles from the last N hours.
                        Defaults to settings.SENTIMENT_LOOKBACK_HOURS.

    Returns:
        List of NewsArticle objects (unscored).
    """
    if feeds is None:
        feeds = settings.SENTIMENT_RSS_FEEDS
    if lookback_hours is None:
        lookback_hours = settings.SENTIMENT_LOOKBACK_HOURS

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    articles: list[NewsArticle] = []

    for feed_template in feeds:
        url = feed_template.format(ticker=ticker)
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        pass

                # Filter by lookback window if we have a date
                if published and published < cutoff:
                    continue

                title = _clean_html(entry.get("title", ""))
                if not title:
                    continue

                articles.append(NewsArticle(
                    title=title,
                    source=parsed.feed.get("title", url),
                    published=published,
                    url=entry.get("link", ""),
                ))

            logger.debug(
                f"{ticker}: Fetched {len(parsed.entries)} entries from {url}"
            )
        except Exception as e:
            logger.warning(f"{ticker}: Failed to fetch RSS feed {url} — {e}")

    logger.info(f"{ticker}: Found {len(articles)} articles within {lookback_hours}h window")
    return articles


# ── MAIN PIPELINE ────────────────────────────────────────────

def assess_sentiment(
    ticker: str,
    headlines: Optional[list[str]] = None,
    feeds: Optional[list[str]] = None,
) -> SentimentResult:
    """
    Full sentiment assessment pipeline for a ticker.

    1. Fetch news articles (or use provided headlines)
    2. Score each with VADER
    3. Compute average compound score
    4. Map to trading action

    Args:
        ticker: Stock ticker symbol.
        headlines: If provided, skip RSS fetch and use these headlines directly.
                   Useful for testing and for injecting pre-fetched headlines.
        feeds: Override RSS feed list (passed to fetch_news_rss).

    Returns:
        SentimentResult with action, scores, and article details.
    """
    # Step 1: Get articles
    if headlines is not None:
        articles = [NewsArticle(title=h) for h in headlines if h and h.strip()]
    else:
        articles = fetch_news_rss(ticker, feeds=feeds)

    # Step 2: Score
    articles = score_articles(articles)

    # Step 3: Aggregate
    if not articles:
        action = settings.SENTIMENT_NEUTRAL_ACTION
        reason = "No recent news found — proceeding with neutral sentiment"
        avg_compound = 0.0
    else:
        avg_compound = sum(a.compound_score for a in articles) / len(articles)
        action, reason = determine_action(avg_compound)

    result = SentimentResult(
        ticker=ticker,
        action=action,
        avg_compound=round(avg_compound, 4),
        num_articles=len(articles),
        articles=articles,
        reason=reason,
    )

    logger.info(
        f"{ticker} | Sentiment: {action} | "
        f"Avg: {avg_compound:.3f} | "
        f"Articles: {len(articles)} | {reason}"
    )

    return result


def gate_trade(
    ticker: str,
    confidence: float,
    headlines: Optional[list[str]] = None,
) -> tuple[str, float, SentimentResult]:
    """
    Convenience function: assess sentiment and return the gate decision.

    This is the main entry point for the paper trader to call before
    executing a trade.

    Args:
        ticker: Stock ticker symbol.
        confidence: Current trade confidence score (0-100).
        headlines: Optional pre-fetched headlines for testing.

    Returns:
        Tuple of:
        - action: "BLOCK", "REDUCE_50", "PROCEED", "CONFIDENCE_PLUS_10", "CONFIDENCE_PLUS_20"
        - adjusted_confidence: Modified confidence score (0 if BLOCK).
        - result: Full SentimentResult for logging/journaling.
    """
    result = assess_sentiment(ticker, headlines=headlines)
    adjusted = apply_sentiment_to_confidence(confidence, result.action)

    logger.info(
        f"{ticker} | Gate: {result.action} | "
        f"Confidence: {confidence:.1f} → {adjusted:.1f}"
    )

    return result.action, adjusted, result


# ── UTILITIES ────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    """Strip HTML tags from a string (RSS titles sometimes have markup)."""
    if not text:
        return ""
    if "<" in text:
        cleaned = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        # Collapse multiple spaces from tag boundaries
        return " ".join(cleaned.split())
    return text.strip()
