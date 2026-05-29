"""Market API — surfaces the market-wide snapshots captured at trade time.

EdgeFinder records a full market snapshot (SPY/QQQ/IWM/DIA, VIX, regime,
sector performance, advance/decline) on a cadence and at every trade. That
data sat unused; these endpoints expose it as a live regime panel and as the
"market regime this trade was taken in" context for any trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.db.models import MarketSnapshotRecord, TradeContext, TradeRecord

router = APIRouter()

_ET = ZoneInfo("America/New_York")


def _to_et(dt: datetime | None) -> str | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_ET).isoformat()


def _serialize(s: MarketSnapshotRecord) -> dict:
    return {
        "timestamp": _to_et(s.timestamp),
        "regime": s.market_regime,
        "vix": s.vix_level,
        "advance_decline_ratio": s.advance_decline_ratio,
        "indices": {
            "SPY": {"price": s.spy_price, "change_pct": s.spy_change_pct},
            "QQQ": {"price": s.qqq_price, "change_pct": s.qqq_change_pct},
            "IWM": {"price": s.iwm_price, "change_pct": s.iwm_change_pct},
            "DIA": {"price": s.dia_price, "change_pct": s.dia_change_pct},
        },
        "sector_performance": s.sector_performance,
    }


@router.get("/regime")
def market_regime(limit: int = Query(60, ge=1, le=500), db: Session = Depends(get_db)):
    """Latest market regime plus recent history (oldest→newest for charting)."""
    rows = (
        db.query(MarketSnapshotRecord)
        .order_by(MarketSnapshotRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    snaps = [_serialize(s) for s in rows]  # newest first
    return {
        "latest": snaps[0] if snaps else None,
        "history": list(reversed(snaps)),
    }


@router.get("/regime/trade/{trade_id}")
def regime_at_trade(trade_id: str, db: Session = Depends(get_db)):
    """The market regime a given trade was opened in.

    Snapshots are captured at trade time, so we take the most recent snapshot
    at-or-before the trade's entry (falling back to the earliest available).
    Also returns the rich per-trade context (sectors, short interest, news,
    related tickers, dividends, indicators) when present.
    """
    trade = db.query(TradeRecord).filter_by(trade_id=trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="trade not found")

    snap = None
    if trade.entry_time is not None:
        snap = (
            db.query(MarketSnapshotRecord)
            .filter(MarketSnapshotRecord.timestamp <= trade.entry_time)
            .order_by(MarketSnapshotRecord.timestamp.desc())
            .first()
            or db.query(MarketSnapshotRecord)
            .order_by(MarketSnapshotRecord.timestamp.asc())
            .first()
        )

    ctx = db.query(TradeContext).filter_by(trade_id=trade_id).first()
    return {
        "trade_id": trade_id,
        "symbol": trade.symbol,
        "strategy_name": trade.strategy_name,
        "entry_time": _to_et(trade.entry_time),
        "regime": _serialize(snap) if snap else None,
        "context": {
            "sector_prices": ctx.sector_prices,
            "short_interest": ctx.short_interest,
            "recent_news": ctx.recent_news,
            "related_tickers": ctx.related_tickers,
            "dividends": ctx.dividends,
            "indicators": ctx.indicators,
        } if ctx else None,
    }
