"""Research API — per-ticker deep dive, search, active tickers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.data.polygon import PolygonDataProvider
from edgefinder.research.research import ResearchService

router = APIRouter()


def _get_research_service(db: Session = Depends(get_db)) -> ResearchService:
    try:
        provider = PolygonDataProvider()
    except ValueError:
        provider = None
    return ResearchService(provider=provider, session=db)


@router.get("/ticker/{symbol}")
def ticker_report(symbol: str, service: ResearchService = Depends(_get_research_service)):
    """Full research report for a single ticker."""
    report = service.get_ticker_report(symbol.upper())
    return asdict(report)


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
def trigger_scan(db: Session = Depends(get_db)):
    """Trigger a per-strategy scan in the background. Returns immediately."""
    import threading
    from dashboard.services import _provider, _session_factory, _load_watchlists, _populate_fundamentals_cache, get_arena
    from edgefinder.scanner.strategy_scanner import StrategyScanner
    from edgefinder.strategies.base import StrategyRegistry

    if not _provider:
        return {"error": "No Polygon API key configured"}
    if not _session_factory:
        return {"error": "Database not initialized"}

    def _run_scan():
        import logging
        logger = logging.getLogger(__name__)
        universe = _provider.get_ticker_universe()
        logger.info("Manual scan triggered: %d tickers", len(universe))
        for strategy in StrategyRegistry.get_instances():
            session = _session_factory()
            try:
                scanner = StrategyScanner(strategy, _provider, session)
                results = scanner.run(tickers=sorted(universe))
                qualified = sum(1 for r in results if r.qualified)
                logger.info("Manual scan [%s]: %d qualified", strategy.name, qualified)
            except Exception:
                logger.exception("Manual scan failed for %s", strategy.name)
            finally:
                session.close()
        arena = get_arena()
        if arena:
            watchlists = _load_watchlists()
            if watchlists:
                arena.set_watchlists(watchlists)
                _populate_fundamentals_cache()
        logger.info("Manual scan complete")

    threading.Thread(target=_run_scan, daemon=True, name="manual-scan").start()
    return {"status": "scanning", "message": "Scan started in background. Data will appear as tickers are processed."}
