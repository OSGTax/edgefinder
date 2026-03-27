"""Sentiment API — per-ticker sentiment, trending."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.db.models import SentimentReading
from edgefinder.sentiment.aggregator import SentimentAggregator

router = APIRouter()


@router.get("/ticker/{symbol}")
def ticker_sentiment(symbol: str, db: Session = Depends(get_db)):
    """Get aggregated sentiment for a ticker."""
    agg = SentimentAggregator(session=db)
    result = agg.get_sentiment(symbol.upper())
    return {
        "symbol": result.symbol,
        "composite_score": result.composite_score,
        "source_scores": result.source_scores,
        "total_mentions": result.total_mentions,
        "is_trending": result.is_trending,
        "action": result.action.value,
    }


@router.get("/trending")
def trending(db: Session = Depends(get_db)):
    """Get trending tickers across all sentiment sources."""
    agg = SentimentAggregator(session=db)
    trending = agg.get_trending()
    return [
        {
            "symbol": t.symbol,
            "source": t.source.value,
            "score": t.score,
            "mention_count": t.mention_count,
            "is_trending": t.is_trending,
        }
        for t in trending
    ]


@router.get("/history/{symbol}")
def sentiment_history(
    symbol: str,
    days: int = Query(7, le=90),
    db: Session = Depends(get_db),
):
    """Get sentiment reading history for a ticker."""
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    readings = (
        db.query(SentimentReading)
        .filter(SentimentReading.symbol == symbol.upper(), SentimentReading.timestamp >= cutoff)
        .order_by(SentimentReading.timestamp.desc())
        .limit(200)
        .all()
    )
    return [
        {
            "source": r.source,
            "score": r.score,
            "mention_count": r.mention_count,
            "is_trending": r.is_trending,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in readings
    ]
