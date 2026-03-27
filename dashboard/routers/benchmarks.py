"""Benchmarks API — strategy vs index comparison charts."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.data.polygon import PolygonDataProvider
from edgefinder.market.benchmarks import BenchmarkService

router = APIRouter()


def _get_benchmark_service(db: Session = Depends(get_db)) -> BenchmarkService:
    try:
        provider = PolygonDataProvider()
    except ValueError:
        provider = None
    return BenchmarkService(provider=provider, session=db)


@router.get("/comparison")
def comparison(
    days: int = Query(90, le=365),
    service: BenchmarkService = Depends(_get_benchmark_service),
):
    """Get benchmark comparison data for charting.

    Returns cumulative % change for each index from start date.
    """
    return service.get_comparison_data(days=days)


@router.post("/collect")
def collect_daily(service: BenchmarkService = Depends(_get_benchmark_service)):
    """Trigger daily benchmark data collection."""
    stored = service.collect_daily()
    return {"stored": stored}


@router.post("/backfill")
def backfill(
    days: int = Query(365, le=730),
    service: BenchmarkService = Depends(_get_benchmark_service),
):
    """Backfill historical benchmark data."""
    stored = service.backfill(days=days)
    return {"stored": stored}
