"""EdgeFinder v2 — FastAPI dashboard application.

Central web app that serves the dashboard UI and all API endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from dashboard.routers import benchmarks, inject, research, strategies, trades
from edgefinder.core.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

__version__ = "4.8.1"

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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

# Register routers
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(research.router, prefix="/api/research", tags=["research"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(inject.router, prefix="/api/inject", tags=["inject"])


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


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
