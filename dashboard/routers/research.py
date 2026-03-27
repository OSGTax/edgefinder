"""Research API — per-ticker deep dive, search, active tickers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_data_provider, get_db
from edgefinder.research.research import ResearchService

router = APIRouter()


def _get_research_service(db: Session = Depends(get_db)) -> ResearchService:
    return ResearchService(provider=get_data_provider(), session=db)


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
