"""EdgeFinder — FastAPI app for the autonomous trading agent's desk.

Greenfield rebuild: the app is now a thin read surface over the agent's
``desk_*`` tables (the trading-desk page) plus the kept market-data chart
endpoints. The old trading/research/strategy pages, routers, scheduler, and
jobs were removed in the cutover (see REBUILD-PLAN.md / scripts/cutover.py).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.routers import desk, pages, symbols
from edgefinder.core.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

__version__ = "8.4.2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dashboard.services import init_services, shutdown_services
    from agent import streamer

    logger.info("EdgeFinder trading-desk starting")
    init_services()
    stream_task = streamer.start_in(app)  # None when no Alpaca keys (dev/CI)
    yield
    if stream_task is not None:
        stream_task.cancel()
        try:
            await stream_task
        except (Exception, asyncio.CancelledError):  # noqa: BLE001
            pass
    shutdown_services()
    logger.info("EdgeFinder trading-desk shutting down")


app = FastAPI(
    title="EdgeFinder — Trading Desk",
    description="Autonomous AI paper-trading agent + trading-desk page",
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

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Pages first so "/" is handled by the pages router (redirects to /desk).
app.include_router(pages.router, tags=["pages"])
app.include_router(desk.router, prefix="/api/desk", tags=["desk"])
app.include_router(symbols.router, prefix="/api/symbols", tags=["symbols"])


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": __version__, "app": "trading-desk"}
