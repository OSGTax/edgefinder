"""EdgeFinder v2 — FastAPI dashboard application.

Central web app that serves the dashboard UI and all API endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.routers import (
    admin, benchmarks, lab, market, ops, pages, research,
    strategies, symbols, trades,
)
from edgefinder.core.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

__version__ = "5.52.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dashboard.services import init_services, shutdown_services

    logger.info("EdgeFinder v2 dashboard starting")
    init_services()
    yield
    shutdown_services()
    logger.info("EdgeFinder v2 dashboard shutting down")


app = FastAPI(
    title="EdgeFinder v2",
    description="Trading workbench for strategy research and paper trading",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Page routes (before API routers so / is handled by pages)
app.include_router(pages.router, tags=["pages"])

# Register API routers
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(research.router, prefix="/api/research", tags=["research"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(ops.router, prefix="/api/ops", tags=["ops"])
app.include_router(symbols.router, prefix="/api/symbols", tags=["symbols"])
app.include_router(lab.router, prefix="/api/lab", tags=["lab"])


@app.get("/api/health")
def health_check():
    from dashboard.services import get_plan_access
    plan = get_plan_access()
    available = sum(1 for v in plan.values() if v)
    return {
        "status": "ok",
        "version": __version__,
        "data_sources": {
            "available": available,
            "total": len(plan),
            "details": plan,
        },
    }
