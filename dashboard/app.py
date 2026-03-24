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

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
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
PROJECT_ROOT = DASHBOARD_DIR.parent
SUGGESTIONS_JSON = PROJECT_ROOT / "data" / "suggestions.json"

# Journal instance for stats
_journal = TradeJournal()


# ── SUGGESTIONS PERSISTENCE ─────────────────────────────────

def _seed_suggestions_from_json():
    """Load suggestions from seed JSON into SQLite (survives deploys)."""
    if not SUGGESTIONS_JSON.exists():
        return
    try:
        with open(SUGGESTIONS_JSON) as f:
            seeds = json.load(f)
        if not seeds:
            return

        session = get_session()
        existing_titles = {
            s.title for s in session.query(Suggestion.title).all()
        }
        added = 0
        for s in seeds:
            if s["title"] not in existing_titles:
                session.add(Suggestion(
                    category=s.get("category", "other"),
                    title=s["title"],
                    description=s.get("description", ""),
                    status=s.get("status", "new"),
                    created_at=datetime.fromisoformat(s["created_at"]) if s.get("created_at") else datetime.utcnow(),
                ))
                added += 1
        if added:
            session.commit()
            logger.info(f"Seeded {added} suggestions from JSON")
        session.close()
    except Exception as e:
        logger.error(f"Failed to seed suggestions: {e}")


def _export_suggestions_json():
    """Write all suggestions from DB back to the JSON seed file."""
    try:
        session = get_session()
        rows = session.query(Suggestion).order_by(Suggestion.id).all()
        data = [
            {
                "id": s.id,
                "category": s.category,
                "title": s.title,
                "description": s.description,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in rows
        ]
        session.close()

        # Only write if we're not on Render (repo is read-only there)
        try:
            with open(SUGGESTIONS_JSON, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        except OSError:
            pass  # Read-only filesystem (Render), that's fine

        return data
    except Exception as e:
        logger.error(f"Failed to export suggestions: {e}")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, seed suggestions, start scheduler."""
    init_db()
    _seed_suggestions_from_json()

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


# ── MANUAL SCANNER ──────────────────────────────────────────

_scan_status = {"running": False, "last_result": None, "last_error": None}


def _run_scan_background():
    """Run the scanner in a background thread."""
    from modules.scanner import run_scan, fetch_batch, passes_prescreen, score_stock
    try:
        tickers = sorted(set(settings.SCANNER_DEFAULT_TICKERS))
        _scan_status["last_result"] = f"Scanning {len(tickers)} tickers..."

        # Run with detailed tracking
        all_data = fetch_batch(tickers)
        fetched = len(all_data)
        screened = [d for d in all_data if passes_prescreen(d)]
        passed = len(screened)
        scored = [score_stock(d) for d in screened]
        watchlist = [
            s for s in scored
            if s.composite_score >= settings.WATCHLIST_MIN_COMPOSITE_SCORE
        ]

        # Save to DB
        if watchlist:
            from modules.scanner import _save_watchlist
            _save_watchlist(watchlist)

        detail = (
            f"{len(watchlist)} on watchlist "
            f"(fetched {fetched}/{len(tickers)}, "
            f"passed prescreen {passed}, "
            f"scored above {settings.WATCHLIST_MIN_COMPOSITE_SCORE}: {len(watchlist)})"
        )
        _scan_status["last_result"] = detail
        _scan_status["last_error"] = None
        logger.info(f"Manual scan complete: {detail}")
    except Exception as e:
        _scan_status["last_result"] = None
        _scan_status["last_error"] = str(e)
        logger.error(f"Manual scan failed: {e}")
    finally:
        _scan_status["running"] = False


@app.post("/api/scanner/run")
async def run_scanner_manually():
    """Trigger a manual scanner run."""
    if _scan_status["running"]:
        return {"status": "already_running", "message": "Scanner is already running"}

    _scan_status["running"] = True
    _scan_status["last_result"] = None
    _scan_status["last_error"] = None
    thread = threading.Thread(target=_run_scan_background, daemon=True)
    thread.start()
    return {"status": "started", "message": "Scanner started in background"}


@app.get("/api/scanner/status")
async def get_scanner_status():
    """Get the status of a manual scanner run."""
    return {
        "running": _scan_status["running"],
        "last_result": _scan_status["last_result"],
        "last_error": _scan_status["last_error"],
    }


@app.post("/api/watchlist/add")
async def add_ticker_to_watchlist(payload: dict):
    """
    Manually add a ticker to the watchlist.
    Fetches fundamental data, scores it, and adds regardless of score.
    Includes detailed reasoning for why it scored the way it did.
    """
    from modules.scanner import fetch_fundamental_data, score_stock

    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return {"error": "Ticker is required"}

    # Fetch fundamental data (retries with backoff are built into fetch_fundamental_data)
    data = fetch_fundamental_data(ticker)
    if not data:
        return {"error": f"Could not fetch data for {ticker}. Yahoo may be rate limiting — try again in a minute."}

    # Score it
    scored = score_stock(data)
    breakdown = scored.score_breakdown

    # Build plain English reasoning using actual data values
    from modules.scanner import _build_notes
    notes = "MANUAL | " + _build_notes(scored)

    # Save to DB (regardless of composite score)
    session = get_session()
    try:
        # Deactivate any existing entry for this ticker
        session.query(WatchlistStock).filter(
            WatchlistStock.ticker == ticker,
            WatchlistStock.is_active == True,  # noqa: E712
        ).update({"is_active": False})

        entry = WatchlistStock(
            ticker=ticker,
            company_name=data.company_name,
            sector=data.sector,
            industry=data.industry,
            market_cap=data.market_cap,
            price=data.price,
            peg_ratio=data.peg_ratio,
            earnings_growth=data.earnings_growth,
            debt_to_equity=data.debt_to_equity,
            revenue_growth=data.revenue_growth,
            institutional_pct=data.institutional_pct,
            lynch_category=scored.lynch_category,
            lynch_score=scored.lynch_score,
            fcf_yield=data.fcf_yield,
            price_to_tangible_book=data.price_to_tangible_book,
            short_interest=data.short_interest,
            ev_to_ebitda=data.ev_to_ebitda,
            current_ratio=data.current_ratio,
            burry_score=scored.burry_score,
            composite_score=scored.composite_score,
            scan_date=datetime.utcnow(),
            is_active=True,
            notes=notes,
        )
        session.add(entry)
        session.commit()

        return {
            "status": "ok",
            "ticker": ticker,
            "company_name": data.company_name,
            "composite_score": scored.composite_score,
            "lynch_score": scored.lynch_score,
            "burry_score": scored.burry_score,
            "lynch_category": scored.lynch_category,
            "notes": notes,
            "breakdown": breakdown,
        }
    finally:
        session.close()


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
                    "notes": s.notes,
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
        _export_suggestions_json()
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
        _export_suggestions_json()
        return {"status": "ok", "message": "Suggestion deleted"}
    finally:
        session.close()


@app.get("/api/suggestions/export")
async def export_suggestions():
    """Export all suggestions as JSON (for committing to repo)."""
    data = _export_suggestions_json()
    return {"count": len(data), "suggestions": data}
