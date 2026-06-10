"""Research API — per-ticker deep dive, search, active tickers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.research.research import ResearchService

router = APIRouter()


def _get_research_service(db: Session = Depends(get_db)) -> ResearchService:
    # Reuse the singleton provider from services.py to avoid creating
    # a new PolygonDataProvider (with 20 connection pools) per request
    from dashboard.services import _provider
    return ResearchService(provider=_provider, session=db)


@router.get("/ticker/{symbol}")
def ticker_report(symbol: str, service: ResearchService = Depends(_get_research_service)):
    """Full research report for a single ticker."""
    report = service.get_ticker_report(symbol.upper())
    return asdict(report)


@router.get("/ticker/{symbol}/bars")
def ticker_bars(
    symbol: str,
    days: int = Query(365, ge=1, le=1825),
    db: Session = Depends(get_db),
):
    """Daily OHLC history for a ticker from the flat-file-backfilled daily_bars.

    Powers the research price chart (with dividend/split/news event markers).
    Returns business-day `time` strings for lightweight-charts. Empty until the
    S3 daily-bar backfill has populated this symbol.
    """
    from datetime import date, timedelta

    from edgefinder.db.models import DailyBar

    cutoff = date.today() - timedelta(days=days)
    rows = (
        db.query(DailyBar)
        .filter(DailyBar.symbol == symbol.upper(), DailyBar.date >= cutoff)
        .order_by(DailyBar.date)
        .all()
    )
    return [
        {
            "time": r.date.isoformat(),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ]


@router.get("/search")
def search_tickers(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, le=50),
    service: ResearchService = Depends(_get_research_service),
):
    """Search tickers by symbol or company name."""
    return service.search_tickers(q, limit)


@router.get("/active")
def active_tickers(service: ResearchService = Depends(_get_research_service)):
    """Get all active (watchlisted) tickers."""
    return service.get_active_tickers()


@router.post("/scan")
def trigger_scan():
    """Trigger a unified data scan in the background. Returns immediately."""
    import threading
    from dashboard import services
    from edgefinder.scanner.unified_scanner import UnifiedScanner

    provider = services.get_provider()
    session_factory = services._session_factory
    if not provider:
        return {"error": "No Polygon API key configured"}
    if not session_factory:
        return {"error": "Database not initialized"}

    def _run_scan():
        import logging
        logger = logging.getLogger(__name__)

        tickers = services._resolve_scan_universe()
        logger.info("Manual scan triggered: %d tickers", len(tickers))

        scanner = UnifiedScanner(provider, session_factory)
        try:
            summary = scanner.run(tickers)
            logger.info("Manual scan results: %s", summary)
        except Exception:
            logger.exception("Manual scan failed")

    threading.Thread(target=_run_scan, daemon=True, name="manual-scan").start()
    return {"status": "scanning", "message": "Unified scan started in background."}
