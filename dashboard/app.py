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
from datetime import datetime, timezone
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
    ArenaTradeLog,
    Suggestion,
)
from modules.utils import to_eastern, verify_hash_chain
logger = logging.getLogger(__name__)

# Static files / templates
DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
PROJECT_ROOT = DASHBOARD_DIR.parent
SUGGESTIONS_JSON = PROJECT_ROOT / "data" / "suggestions.json"


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
                "created_at": to_eastern(s.created_at),
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
    try:
        start_scheduler()
    except Exception as e:
        logger.error(f"Scheduler failed to start: {e}")

    yield

    try:
        stop_scheduler()
    except Exception:
        pass


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
        "scheduler_running": status.get("scheduler_running", status.get("running", False)),
    }


@app.get("/api/scheduler")
async def get_scheduler_info():
    """Return scheduler status, next job run times, and open positions."""
    from modules.scheduler import get_scheduler_status
    return get_scheduler_status()


@app.get("/api/diagnostics")
async def get_diagnostics():
    """Return data pipeline health: data sources, DB type, counts."""
    from modules.scanner import get_active_watchlist, _init_data_service
    from modules.arena.live import get_arena_status
    from modules.database import _get_database_url

    ds = _init_data_service()
    ds_diag = ds.get_diagnostics() if ds else {}
    arena = get_arena_status()
    db_url = _get_database_url()

    return {
        "database": "postgresql" if "postgresql" in db_url else "sqlite",
        "data_sources": ds_diag.get("sources", {}),
        "fundamentals_source": ds_diag.get("fundamentals_source", "yfinance"),
        "bars_source": ds_diag.get("bars_source", "unknown"),
        "watchlist_count": len(get_active_watchlist()),
        "arena_running": arena.get("running", False),
        "strategies": arena.get("strategies", 0),
        "last_signal_check": arena.get("last_signal_check"),
        "last_signal_result": arena.get("last_signal_result"),
        "last_scan": arena.get("last_scan"),
        "errors": arena.get("errors", [])[-5:],
    }


# ── MANUAL SCANNER ──────────────────────────────────────────

_scan_status = {"running": False, "last_result": None, "last_error": None}


def _run_scan_background():
    """Run the scanner in a background thread using today's sector rotation."""
    from modules.scanner import run_scan, get_todays_sectors
    try:
        sectors = get_todays_sectors()
        if sectors:
            _scan_status["last_result"] = f"Scanning sectors: {', '.join(sectors)}..."
        else:
            _scan_status["last_result"] = "Scanning full universe..."

        watchlist = run_scan(sectors=sectors if sectors else None, save_to_db=True)

        sector_info = f"sectors: {', '.join(sectors)}" if sectors else "full universe"
        detail = (
            f"{len(watchlist)} on watchlist "
            f"({sector_info}, "
            f"scored above {settings.WATCHLIST_MIN_COMPOSITE_SCORE}: {len(watchlist)})"
        )
        _scan_status["last_result"] = detail
        _scan_status["last_error"] = None
        logger.info(f"Manual scan complete: {detail}")

        # Refresh arena strategy watchlists with new data
        try:
            from modules.arena.live import _refresh_watchlists
            _refresh_watchlists()
            logger.info("Arena watchlists refreshed after manual scan")
        except Exception as e:
            logger.warning(f"Arena watchlist refresh failed: {e}")

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
def add_ticker_to_watchlist(payload: dict):
    """
    Manually add a ticker to the watchlist.
    Fetches fundamental data, scores it, and adds regardless of score.
    Includes detailed reasoning for why it scored the way it did.

    Uses plain `def` (not async) so FastAPI runs it in a threadpool —
    required because fetch_fundamental_data makes blocking HTTP calls.
    """
    from modules.scanner import fetch_fundamental_data_verbose, score_stock

    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return {"error": "Ticker is required"}

    # Fetch fundamental data with specific error reasons
    data, error_reason = fetch_fundamental_data_verbose(ticker)
    if not data:
        return {"error": error_reason}

    # Score it
    scored = score_stock(data)
    breakdown = scored.score_breakdown

    # Build plain English reasoning using actual data values
    from modules.scanner import _build_notes
    notes = "MANUAL | " + _build_notes(scored)

    # Save to DB (regardless of composite score) — upsert to respect unique constraint
    session = get_session()
    try:
        existing = session.query(WatchlistStock).filter(
            WatchlistStock.ticker == ticker,
        ).first()

        if existing:
            # Update existing row in-place
            existing.company_name = data.company_name
            existing.sector = data.sector
            existing.industry = data.industry
            existing.market_cap = data.market_cap
            existing.price = data.price
            existing.peg_ratio = data.peg_ratio
            existing.earnings_growth = data.earnings_growth
            existing.debt_to_equity = data.debt_to_equity
            existing.revenue_growth = data.revenue_growth
            existing.institutional_pct = data.institutional_pct
            existing.lynch_category = scored.lynch_category
            existing.lynch_score = scored.lynch_score
            existing.fcf_yield = data.fcf_yield
            existing.price_to_tangible_book = data.price_to_tangible_book
            existing.short_interest = data.short_interest
            existing.ev_to_ebitda = data.ev_to_ebitda
            existing.current_ratio = data.current_ratio
            existing.burry_score = scored.burry_score
            existing.composite_score = scored.composite_score
            existing.scan_date = datetime.now(timezone.utc)
            existing.is_active = True
            existing.notes = notes
        else:
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
                scan_date=datetime.now(timezone.utc),
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
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save {ticker} to watchlist: {e}")
        return {"error": f"Database error saving {ticker}: {e}"}
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
                    "scan_date": to_eastern(s.scan_date),
                    "notes": s.notes,
                }
                for s in stocks
            ],
        }
    finally:
        session.close()


@app.get("/api/fundamentals")
async def get_fundamentals(
    sector: Optional[str] = Query(default=None),
    sort_by: Optional[str] = Query(default="composite_score"),
    limit: int = Query(default=500, ge=1, le=1000),
):
    """Get all fundamental data for browsing. Returns every field in the watchlist table."""
    session = get_session()
    try:
        query = session.query(WatchlistStock).filter(
            WatchlistStock.is_active == True  # noqa: E712
        )
        if sector:
            query = query.filter(WatchlistStock.sector == sector)

        # Dynamic sort — fall back to composite_score if column doesn't exist
        sort_col = getattr(WatchlistStock, sort_by, WatchlistStock.composite_score)
        query = query.order_by(sort_col.desc())

        stocks = query.limit(limit).all()

        # Get distinct sectors for filter dropdown
        sectors_query = session.query(WatchlistStock.sector).filter(
            WatchlistStock.is_active == True  # noqa: E712
        ).distinct().all()
        sectors = sorted([s[0] for s in sectors_query if s[0]])

        return {
            "count": len(stocks),
            "sectors": sectors,
            "stocks": [
                {
                    "ticker": s.ticker,
                    "company_name": s.company_name,
                    "sector": s.sector,
                    "industry": s.industry,
                    "market_cap": s.market_cap,
                    "price": s.price,
                    "composite_score": s.composite_score,
                    "lynch_score": s.lynch_score,
                    "burry_score": s.burry_score,
                    "lynch_category": s.lynch_category,
                    "peg_ratio": s.peg_ratio,
                    "earnings_growth": s.earnings_growth,
                    "revenue_growth": s.revenue_growth,
                    "debt_to_equity": s.debt_to_equity,
                    "institutional_pct": s.institutional_pct,
                    "fcf_yield": s.fcf_yield,
                    "price_to_tangible_book": s.price_to_tangible_book,
                    "short_interest": s.short_interest,
                    "ev_to_ebitda": s.ev_to_ebitda,
                    "current_ratio": s.current_ratio,
                    "scan_date": to_eastern(s.scan_date),
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
                    "timestamp": to_eastern(r.timestamp),
                }
                for r in records
            ],
        }
    finally:
        session.close()


@app.get("/api/trades")
async def get_trades(
    ticker: Optional[str] = Query(default=None),
    strategy: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get arena trade log with full metadata."""
    from modules.database import ArenaTradeLog
    session = get_session()
    try:
        query = session.query(ArenaTradeLog).order_by(ArenaTradeLog.created_at.desc())
        if ticker:
            query = query.filter(ArenaTradeLog.ticker == ticker.upper())
        if strategy:
            query = query.filter(ArenaTradeLog.strategy_name == strategy)
        trades = query.limit(limit).all()
        return {
            "count": len(trades),
            "trades": [_format_arena_trade(t) for t in trades],
        }
    finally:
        session.close()


@app.get("/api/trades/stats")
async def get_trade_stats(
    strategy: Optional[str] = Query(default=None),
):
    """Get aggregated trade statistics from arena trades."""
    from modules.database import ArenaTradeLog
    session = get_session()
    try:
        query = session.query(ArenaTradeLog).filter(ArenaTradeLog.status == "CLOSED")
        if strategy:
            query = query.filter(ArenaTradeLog.strategy_name == strategy)
        trades = query.all()

        if not trades:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "profit_factor": 0,
            }

        winners = [t for t in trades if (t.pnl_dollars or 0) > 0]
        losers = [t for t in trades if (t.pnl_dollars or 0) < 0]
        total_pnl = sum(t.pnl_dollars or 0 for t in trades)
        gross_profit = sum(t.pnl_dollars for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_dollars for t in losers)) if losers else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": round(len(winners) / len(trades) * 100, 2) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
            "avg_winner": round(gross_profit / len(winners), 2) if winners else 0,
            "avg_loser": round(-gross_loss / len(losers), 2) if losers else 0,
            "largest_winner": round(max((t.pnl_dollars or 0) for t in trades), 2),
            "largest_loser": round(min((t.pnl_dollars or 0) for t in trades), 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "avg_r_multiple": round(
                sum(t.r_multiple or 0 for t in trades) / len(trades), 2
            ) if trades else 0,
        }
    finally:
        session.close()


@app.get("/api/trades/verify")
async def verify_trade_integrity():
    """Verify the integrity of the trade audit chain.

    Walks the hash chain from first to last trade and checks that
    each trade's integrity_hash matches a recomputed hash. If any
    record has been inserted, edited, deleted, or reordered, the
    chain will break and this endpoint reports where.
    """
    session = get_session()
    try:
        records = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.integrity_hash != None  # noqa: E711
        ).order_by(ArenaTradeLog.sequence_num.asc()).all()

        if not records:
            return {
                "valid": True,
                "total_trades": 0,
                "first_break_at": None,
                "message": "No trades with integrity hashes found.",
            }

        trades = [
            {
                "trade_id": r.trade_id,
                "ticker": r.ticker,
                "action": r.action,
                "execution_price": r.execution_price,
                "shares": r.shares,
                "execution_timestamp": r.execution_timestamp,
                "strategy_name": r.strategy_name,
                "integrity_hash": r.integrity_hash,
                "sequence_num": r.sequence_num,
            }
            for r in records
        ]

        return verify_hash_chain(trades)
    finally:
        session.close()


@app.get("/api/equity-curve")
async def get_equity_curve(
    strategy: Optional[str] = Query(default=None),
    limit: int = Query(default=365, ge=1, le=1000),
):
    """Get arena equity snapshots for charting."""
    from modules.database import ArenaSnapshot
    session = get_session()
    try:
        query = session.query(ArenaSnapshot).order_by(ArenaSnapshot.timestamp.asc())
        if strategy:
            query = query.filter(ArenaSnapshot.strategy_name == strategy)
        snaps = query.limit(limit).all()
        from collections import OrderedDict
        grouped: dict[str, list] = OrderedDict()
        for s in snaps:
            grouped.setdefault(s.strategy_name, []).append({
                "date": to_eastern(s.timestamp),
                "total_value": s.total_equity,
                "cash": s.cash,
                "drawdown_pct": s.drawdown_pct,
                "total_return_pct": s.total_return_pct,
            })
        return {
            "count": len(snaps),
            "strategies": dict(grouped),
            "strategy_list": list(grouped.keys()),
        }
    finally:
        session.close()


@app.get("/api/account")
async def get_account():
    """Get combined arena account state across all strategies."""
    try:
        from modules.arena.live import get_arena_engine
        engine = get_arena_engine()
        if engine:
            total_equity = sum(a.total_equity for a in engine.accounts.values())
            total_cash = sum(a.cash for a in engine.accounts.values())
            total_positions = sum(a.open_position_count for a in engine.accounts.values())
            total_realized = sum(a.realized_pnl for a in engine.accounts.values())
            total_unrealized = sum(a.unrealized_pnl for a in engine.accounts.values())
            num_strategies = len(engine.accounts)
            total_deposited = num_strategies * settings.ARENA_STARTING_CAPITAL_PER_STRATEGY
            return {
                "date": to_eastern(datetime.now(timezone.utc)),
                "cash": round(total_cash, 2),
                "positions_value": round(total_equity - total_cash, 2),
                "total_value": round(total_equity, 2),
                "open_positions": total_positions,
                "strategies": num_strategies,
                "total_deposited": round(total_deposited, 2),
                "total_pnl": round(total_equity - total_deposited, 2),
                "realized_pnl": round(total_realized, 2),
                "unrealized_pnl": round(total_unrealized, 2),
            }
    except Exception as e:
        logger.error(f"Account endpoint failed: {e}")
    return {
        "date": None,
        "cash": 0.0,
        "positions_value": 0.0,
        "total_value": 0.0,
        "open_positions": 0,
        "strategies": 0,
        "total_deposited": 0.0,
        "total_pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
    }


@app.get("/api/skipped-signals")
async def get_skipped_signals(
    ticker: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Get signals that were detected but not traded."""
    session = get_session()
    try:
        query = session.query(SignalRecord).filter(
            SignalRecord.was_traded == False  # noqa: E712
        ).order_by(SignalRecord.timestamp.desc())
        if ticker:
            query = query.filter(SignalRecord.ticker == ticker.upper())
        signals = query.limit(limit).all()
        return {
            "count": len(signals),
            "signals": [
                {
                    "ticker": s.ticker,
                    "signal_type": s.signal_type,
                    "trade_type": s.trade_type,
                    "confidence": s.confidence,
                    "indicators": s.indicators,
                    "reason_skipped": s.reason_skipped,
                    "timestamp": to_eastern(s.timestamp),
                }
                for s in signals
            ],
        }
    finally:
        session.close()


def _format_arena_trade(t) -> dict:
    """Format an ArenaTradeLog record with full metadata for API response."""
    meta = t.extra_data or {}
    indicators = meta.get("indicators", {})
    trade_reason = meta.get("trade_reason", "")

    # Build plain English explanation
    explanation = _build_trade_explanation(t, meta)

    return {
        "trade_id": t.trade_id,
        "strategy_name": t.strategy_name,
        "strategy_version": t.strategy_version,
        "ticker": t.ticker,
        "action": t.action,
        "direction": t.direction,
        "trade_type": t.trade_type,
        "signal_price": t.signal_price,
        "execution_price": t.execution_price,
        "exit_price": t.exit_price,
        "slippage": t.slippage,
        "shares": t.shares,
        "stop_loss": t.stop_loss,
        "target": t.target,
        "confidence": t.confidence,
        "pnl_dollars": t.pnl_dollars,
        "pnl_percent": t.pnl_percent,
        "r_multiple": t.r_multiple,
        "exit_reason": t.exit_reason,
        "price_source": t.price_source,
        "market_regime": t.market_regime,
        "signal_overlap": t.signal_overlap,
        "position_overlap": t.position_overlap,
        "bar_data_at_decision": t.bar_data_at_decision,
        "indicators": indicators,
        "trade_reason": trade_reason,
        "explanation": explanation,
        "strategy_metadata": meta,
        "status": t.status,
        "signal_timestamp": to_eastern(t.signal_timestamp),
        "execution_timestamp": to_eastern(t.execution_timestamp),
        "exit_timestamp": to_eastern(t.exit_timestamp),
        "created_at": to_eastern(t.created_at),
    }


def _build_trade_explanation(trade, meta: dict) -> str:
    """Build plain English explanation for a trade."""
    parts = []
    strategy = trade.strategy_name or "unknown"
    ticker = trade.ticker or "?"
    price = trade.execution_price or trade.signal_price or 0

    parts.append(f"{trade.action or 'BUY'} {ticker} @ ${price:.2f} [{strategy}]")

    # Indicators
    reason = meta.get("trade_reason", "")
    if reason:
        parts.append(f"Signals: {reason}")

    # Strategy-specific scores
    if strategy == "lynch":
        score = meta.get("lynch_score")
        cat = meta.get("lynch_category")
        if score:
            parts.append(f"Lynch score: {score}/100 ({cat or 'unknown'})")
    elif strategy == "burry":
        score = meta.get("burry_score")
        fcf = meta.get("fcf_yield")
        ptb = meta.get("price_to_tangible_book")
        score_parts = []
        if score:
            score_parts.append(f"Burry score: {score}/100")
        if fcf:
            score_parts.append(f"FCF yield: {fcf:.1%}")
        if ptb:
            score_parts.append(f"P/TB: {ptb:.2f}")
        if meta.get("rsi_boost_applied"):
            score_parts.append("RSI oversold boost applied")
        if score_parts:
            parts.append(", ".join(score_parts))

    # Risk
    sl = trade.stop_loss
    tgt = trade.target
    if sl and tgt and price:
        sl_pct = (sl - price) / price * 100
        parts.append(f"Risk: Stop ${sl:.2f} ({sl_pct:.1f}%), Target ${tgt:.2f}")

    # Confidence and source
    extras = []
    if trade.confidence:
        extras.append(f"Confidence: {trade.confidence:.0f}%")
    if trade.slippage:
        extras.append(f"Slippage: ${trade.slippage:.4f}")
    if trade.price_source:
        extras.append(f"Source: {trade.price_source}")
    if extras:
        parts.append(" | ".join(extras))

    return "\n".join(parts)


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
                    "created_at": to_eastern(s.created_at),
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


# ── ARENA API (Multi-Strategy) ──────────────────────────────

@app.get("/api/arena")
async def get_arena_status():
    """Get arena status: strategies, leaderboard, overlap, positions."""
    try:
        from modules.arena.live import get_arena_status as _get_status
        return _get_status()
    except Exception as e:
        logger.error(f"Arena status error: {e}")
        return {"running": False, "error": str(e)}


@app.get("/api/arena/leaderboard")
async def get_arena_leaderboard():
    """Get strategy performance comparison sorted by return."""
    try:
        from modules.arena.live import get_arena_engine
        engine = get_arena_engine()
        if engine is None:
            return {"leaderboard": []}
        return {"leaderboard": engine.get_leaderboard()}
    except Exception as e:
        logger.error(f"Arena leaderboard error: {e}")
        return {"leaderboard": [], "error": str(e)}


@app.get("/api/arena/strategy/{strategy_name}")
async def get_arena_strategy(strategy_name: str):
    """Get detailed info for a specific strategy including open positions."""
    try:
        from modules.arena.live import get_arena_engine
        engine = get_arena_engine()
        if engine is None or strategy_name not in engine.accounts:
            return {"error": f"Strategy '{strategy_name}' not found"}
        account = engine.accounts[strategy_name]
        data = account.to_dict()

        # Enrich positions with metadata and explanations
        enriched_positions = {}
        for tid, pos in account.positions.items():
            entry = {
                "trade_id": tid,
                "ticker": pos.ticker,
                "direction": pos.direction,
                "trade_type": pos.trade_type,
                "entry_price": pos.entry_price,
                "shares": pos.shares,
                "stop_loss": pos.stop_loss,
                "target": pos.target,
                "trailing_stop": pos.trailing_stop,
                "last_price": pos.last_known_price,
                "high_water_mark": pos.high_water_mark,
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(pos.unrealized_pnl_pct, 2),
                "r_multiple": round(pos.r_multiple, 2),
                "market_value": round(pos.market_value, 2),
                "confidence": pos.confidence,
                "entry_time": to_eastern(pos.entry_time),
                "slippage_applied": pos.slippage_applied,
                "metadata": pos.metadata,
            }

            # Build plain English explanation from metadata
            meta = pos.metadata or {}
            explanation_parts = [
                f"BUY {pos.ticker} @ ${pos.entry_price:.2f} [{strategy_name}]"
            ]
            if meta.get("trade_reason"):
                explanation_parts.append(f"Signals: {meta['trade_reason']}")
            if meta.get("lynch_score"):
                explanation_parts.append(
                    f"Lynch score: {meta['lynch_score']}/100 ({meta.get('lynch_category', '')})"
                )
            if meta.get("burry_score"):
                explanation_parts.append(f"Burry score: {meta['burry_score']}/100")
            sl_pct = (pos.stop_loss - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0
            explanation_parts.append(
                f"Risk: Stop ${pos.stop_loss:.2f} ({sl_pct:.1f}%), Target ${pos.target:.2f}"
            )
            entry["explanation"] = "\n".join(explanation_parts)
            enriched_positions[tid] = entry

        data["positions"] = enriched_positions
        data["audit_log"] = engine.executor.get_audit_log(strategy_name)[-20:]
        data["version"] = engine.strategies[strategy_name].version if strategy_name in engine.strategies else ""
        return data
    except Exception as e:
        logger.error(f"Arena strategy error: {e}")
        return {"error": str(e)}


@app.get("/api/arena/trades")
async def get_arena_trades(
    strategy: Optional[str] = None,
    limit: int = Query(default=50, le=500),
):
    """Get arena trade history with full metadata and explanations."""
    try:
        from modules.database import ArenaTradeLog
        session = get_session()
        query = session.query(ArenaTradeLog).order_by(
            ArenaTradeLog.created_at.desc()
        )
        if strategy:
            query = query.filter(ArenaTradeLog.strategy_name == strategy)
        trades = query.limit(limit).all()
        result = [_format_arena_trade(t) for t in trades]
        session.close()
        return {"trades": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Arena trades error: {e}")
        return {"trades": [], "error": str(e)}


@app.get("/api/arena/snapshots")
async def get_arena_snapshots(
    strategy: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    """Get arena equity snapshots for charting."""
    try:
        from modules.database import ArenaSnapshot
        session = get_session()
        query = session.query(ArenaSnapshot).order_by(ArenaSnapshot.timestamp.asc())
        if strategy:
            query = query.filter(ArenaSnapshot.strategy_name == strategy)
        snaps = query.limit(limit).all()
        result = []
        for s in snaps:
            result.append({
                "strategy_name": s.strategy_name,
                "timestamp": to_eastern(s.timestamp),
                "total_equity": s.total_equity,
                "cash": s.cash,
                "drawdown_pct": s.drawdown_pct,
                "total_return_pct": s.total_return_pct,
                "open_positions": s.open_positions,
            })
        session.close()
        return {"snapshots": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Arena snapshots error: {e}")
        return {"snapshots": [], "error": str(e)}


@app.post("/api/arena/strategy/{strategy_name}/pause")
async def pause_arena_strategy(strategy_name: str):
    """Pause a strategy (disable trading)."""
    try:
        from modules.arena.live import get_arena_engine
        engine = get_arena_engine()
        if engine is None:
            return {"error": "Arena not running"}
        engine.disable_strategy(strategy_name)
        return {"status": "paused", "strategy": strategy_name}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/arena/strategy/{strategy_name}/resume")
async def resume_arena_strategy(strategy_name: str):
    """Resume a paused strategy."""
    try:
        from modules.arena.live import get_arena_engine
        engine = get_arena_engine()
        if engine is None:
            return {"error": "Arena not running"}
        engine.enable_strategy(strategy_name)
        return {"status": "enabled", "strategy": strategy_name}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/arena/reset")
async def reset_arena_endpoint():
    """Cancel all open trades and reset all strategy accounts to starting capital."""
    try:
        from modules.arena.live import reset_arena
        result = reset_arena()
        return result
    except Exception as e:
        return {"error": str(e), "reset": False}
