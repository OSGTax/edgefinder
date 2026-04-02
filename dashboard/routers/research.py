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
    """Trigger a batched scan for today's batch. Returns scan results summary."""
    from datetime import datetime
    from config.settings import settings
    from dashboard.services import get_arena, _provider, _load_watchlist
    from edgefinder.scanner.scanner import FundamentalScanner

    if not _provider:
        return {"error": "No Polygon API key configured"}

    batch_count = settings.scanner_batch_count
    batch_index = datetime.now().weekday()
    if batch_index >= batch_count:
        batch_index = 0

    universe = _provider.get_ticker_universe()
    sorted_universe = sorted(universe)
    batch = sorted_universe[batch_index::batch_count]

    scanner = FundamentalScanner(_provider, db)
    results = scanner.run(tickers=batch, batch_index=batch_index)
    qualified = sum(1 for s in results if s.qualifying_strategies)

    # Update arena watchlist
    arena = get_arena()
    if arena:
        watchlist = _load_watchlist()
        if watchlist:
            arena.set_global_watchlist(watchlist)

    return {
        "batch": batch_index + 1,
        "batch_count": batch_count,
        "scanned": len(results),
        "qualified": qualified,
        "universe_size": len(universe),
    }
