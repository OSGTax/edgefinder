"""
EdgeFinder Dashboard — FastAPI Backend
=======================================
REST API for the EdgeFinder paper trading dashboard.
Includes automated scheduler for continuous market-hours operation.

Endpoints:
    GET  /                      → Dashboard UI (serves index.html)
    GET  /api/watchlist          → Current active watchlist
    GET  /api/signals            → Recent trade signals
    GET  /api/trades             → Trade log
    GET  /api/trades/stats       → Aggregated trade statistics
    GET  /api/equity-curve       → Account snapshots for charting
    GET  /api/account            → Current account state
    GET  /api/skipped-signals    → Signals that were not traded
    GET  /api/scheduler          → Scheduler status, jobs, open positions
    GET  /api/health             → Health check

Run: uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from config import settings
from modules.database import init_db, get_session
from modules.database import (
    WatchlistStock,
    Trade as TradeRecord,
    Signal as SignalRecord,
    AccountSnapshot,
    Suggestion,
)
from modules.journal import TradeJournal

logger = logging.getLogger(__name__)

# Static files / templates
DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"

# Journal instance for stats
_journal = TradeJournal()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB and start scheduler."""
    init_db()

    from modules.scheduler import start_scheduler, stop_scheduler
    start_scheduler()

    yield

    stop_scheduler()


# Initialize app
app = FastAPI(
    title="EdgeFinder Dashboard",
    description="Paper trading system dashboard",
    version="1.0.0",
    lifespan=lifespan,
)


# ── ROUTES ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the main dashboard page."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>EdgeFinder Dashboard</h1><p>Frontend not built yet.</p>",
        status_code=200,
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    from modules.scheduler import get_scheduler_status
    status = get_scheduler_status()
    return {
        "status": "ok",
        "service": "edgefinder-dashboard",
        "scheduler_running": status["running"],
    }


@app.get("/api/scheduler")
async def get_scheduler_info():
    """Return scheduler status, next job run times, and open positions."""
    from modules.scheduler import get_scheduler_status
    return get_scheduler_status()


@app.get("/api/watchlist")
async def get_watchlist(
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get the current active watchlist from the latest scan."""
    session = get_session()
    try:
        stocks = session.query(WatchlistStock).filter(
            WatchlistStock.is_active == True  # noqa: E712
        ).order_by(
            WatchlistStock.composite_score.desc()
        ).limit(limit).all()

        return {
            "count": len(stocks),
            "watchlist": [
                {
                    "ticker": s.ticker,
                    "company_name": s.company_name,
                    "sector": s.sector,
                    "industry": s.industry,
                    "price": s.price,
                    "market_cap": s.market_cap,
                    "composite_score": s.composite_score,
                    "lynch_score": s.lynch_score,
                    "burry_score": s.burry_score,
                    "lynch_category": s.lynch_category,
                    "peg_ratio": s.peg_ratio,
                    "earnings_growth": s.earnings_growth,
                    "debt_to_equity": s.debt_to_equity,
                    "fcf_yield": s.fcf_yield,
                    "ev_to_ebitda": s.ev_to_ebitda,
                    "current_ratio": s.current_ratio,
                    "scan_date": s.scan_date.isoformat() if s.scan_date else None,
                }
                for s in stocks
            ],
        }
    finally:
        session.close()


@app.get("/api/signals")
async def get_signals(
    ticker: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get recent trade signals."""
    session = get_session()
    try:
        query = session.query(SignalRecord).order_by(
            SignalRecord.timestamp.desc()
        )
        if ticker:
            query = query.filter(SignalRecord.ticker == ticker)

        records = query.limit(limit).all()

        return {
            "count": len(records),
            "signals": [
                {
                    "ticker": r.ticker,
                    "signal_type": r.signal_type,
                    "trade_type": r.trade_type,
                    "confidence": r.confidence,
                    "indicators": r.indicators,
                    "was_traded": r.was_traded,
                    "trade_id": r.trade_id,
                    "reason_skipped": r.reason_skipped,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in records
            ],
        }
    finally:
        session.close()


@app.get("/api/trades")
async def get_trades(
    ticker: Optional[str] = Query(default=None),
    trade_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get trade log."""
    entries = _journal.get_trades(ticker=ticker, trade_type=trade_type, limit=limit)
    return {
        "count": len(entries),
        "trades": [
            {
                "trade_id": e.trade_id,
                "ticker": e.ticker,
                "direction": e.direction,
                "trade_type": e.trade_type,
                "entry_price": e.entry_price,
                "exit_price": e.exit_price,
                "shares": e.shares,
                "pnl_dollars": e.pnl_dollars,
                "pnl_percent": e.pnl_percent,
                "r_multiple": e.r_multiple,
                "exit_reason": e.exit_reason,
                "entry_time": e.entry_time.isoformat() if e.entry_time else None,
                "exit_time": e.exit_time.isoformat() if e.exit_time else None,
                "fundamental_score": e.fundamental_score,
                "confidence_score": e.confidence_score,
                "news_sentiment": e.news_sentiment,
            }
            for e in entries
        ],
    }


@app.get("/api/trades/stats")
async def get_trade_stats(
    days: Optional[int] = Query(default=None, ge=1, le=365),
):
    """Get aggregated trade statistics."""
    stats = _journal.compute_stats(days=days)
    return {
        "total_trades": stats.total_trades,
        "winning_trades": stats.winning_trades,
        "losing_trades": stats.losing_trades,
        "breakeven_trades": stats.breakeven_trades,
        "win_rate": round(stats.win_rate, 4),
        "total_pnl": stats.total_pnl,
        "avg_pnl": stats.avg_pnl,
        "avg_winner": stats.avg_winner,
        "avg_loser": stats.avg_loser,
        "largest_winner": stats.largest_winner,
        "largest_loser": stats.largest_loser,
        "avg_r_multiple": stats.avg_r_multiple,
        "profit_factor": stats.profit_factor,
        "day_trades": stats.day_trades,
        "swing_trades": stats.swing_trades,
        "total_signals": stats.total_signals,
        "traded_signals": stats.traded_signals,
        "skipped_signals": stats.skipped_signals,
    }


@app.get("/api/equity-curve")
async def get_equity_curve(
    limit: int = Query(default=365, ge=1, le=1000),
):
    """Get account snapshots for equity curve chart."""
    curve = _journal.get_equity_curve(limit=limit)
    return {
        "count": len(curve),
        "snapshots": curve,
    }


@app.get("/api/account")
async def get_account():
    """Get latest account snapshot."""
    session = get_session()
    try:
        snapshot = session.query(AccountSnapshot).order_by(
            AccountSnapshot.date.desc()
        ).first()

        if snapshot:
            return {
                "date": snapshot.date.isoformat() if snapshot.date else None,
                "cash": snapshot.cash,
                "positions_value": snapshot.positions_value,
                "total_value": snapshot.total_value,
                "open_positions": snapshot.open_positions,
                "peak_value": snapshot.peak_value,
                "drawdown_pct": snapshot.drawdown_pct,
            }
        return {
            "date": None,
            "cash": settings.STARTING_CAPITAL,
            "positions_value": 0.0,
            "total_value": settings.STARTING_CAPITAL,
            "open_positions": 0,
            "peak_value": settings.STARTING_CAPITAL,
            "drawdown_pct": 0.0,
        }
    finally:
        session.close()


@app.get("/api/skipped-signals")
async def get_skipped_signals(
    ticker: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get signals that were detected but not traded."""
    signals = _journal.get_skipped_signals(ticker=ticker, limit=limit)
    return {
        "count": len(signals),
        "signals": signals,
    }


@app.get("/api/suggestions")
async def list_suggestions(
    limit: int = Query(default=100, ge=1, le=500),
):
    """Get all submitted suggestions."""
    session = get_session()
    try:
        rows = session.query(Suggestion).order_by(
            Suggestion.created_at.desc()
        ).limit(limit).all()

        return {
            "count": len(rows),
            "suggestions": [
                {
                    "id": s.id,
                    "category": s.category,
                    "title": s.title,
                    "description": s.description,
                    "status": s.status,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in rows
            ],
        }
    finally:
        session.close()


@app.post("/api/suggestions")
async def create_suggestion(payload: dict):
    """Submit a new suggestion."""
    title = (payload.get("title") or "").strip()
    if not title:
        return {"error": "Title is required"}, 400

    category = payload.get("category", "other").strip().lower()
    if category not in ("feature", "strategy", "bug", "other"):
        category = "other"

    session = get_session()
    try:
        suggestion = Suggestion(
            category=category,
            title=title,
            description=(payload.get("description") or "").strip(),
        )
        session.add(suggestion)
        session.commit()
        return {
            "status": "ok",
            "id": suggestion.id,
            "message": "Suggestion submitted",
        }
    finally:
        session.close()


@app.delete("/api/suggestions/{suggestion_id}")
async def delete_suggestion(suggestion_id: int):
    """Delete a suggestion."""
    session = get_session()
    try:
        row = session.query(Suggestion).filter(Suggestion.id == suggestion_id).first()
        if not row:
            return {"error": "Not found"}, 404
        session.delete(row)
        session.commit()
        return {"status": "ok", "message": "Suggestion deleted"}
    finally:
        session.close()
