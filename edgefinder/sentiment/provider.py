"""EdgeFinder v2 — Sentiment provider base.

Re-exports the SentimentProvider protocol from core/interfaces.py
and defines shared sentiment action mapping.
"""

from edgefinder.core.interfaces import SentimentProvider
from edgefinder.core.models import SentimentAction

from config.settings import settings


def map_score_to_action(score: float) -> SentimentAction:
    """Map a sentiment score (-1 to +1) to a trading action."""
    if score <= settings.sentiment_strong_negative:
        return SentimentAction.BLOCK
    if score <= settings.sentiment_mild_negative:
        return SentimentAction.REDUCE_50
    if score >= settings.sentiment_strong_positive:
        return SentimentAction.CONFIDENCE_PLUS_20
    if score >= settings.sentiment_mild_positive:
        return SentimentAction.CONFIDENCE_PLUS_10
    return SentimentAction.PROCEED


__all__ = ["SentimentProvider", "map_score_to_action"]
