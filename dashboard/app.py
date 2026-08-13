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

__version__ = "10.4.2"


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

# Public READ-ONLY API: any origin may read (wildcard), but credentials are
# never accepted — nothing on the desk uses cookies or auth headers, and
# "*" + credentials is a security misconfiguration browsers only sometimes
# save you from. GET-only matches the surface: there are no write routes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
)

class RevalidatedStatic(StaticFiles):
    """StaticFiles that forbids serving stale assets.

    Without a Cache-Control header, browsers apply heuristic caching to
    /static and can keep old JS/CSS for hours after a deploy — v10.3.0
    shipped a page whose new HTML ran against the previous release's cached
    desk.js, so the tab bar rendered as dead unstyled buttons. `no-cache`
    means "revalidate every time": with ETags that is a cheap 304 when the
    file is unchanged, and the new file the moment it isn't. This also
    covers ES-module imports, which resolve WITHOUT the ?v= stamp the
    templates put on entry points.
    """

    async def get_response(self, path, scope):  # noqa: D102 — see class doc
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatedStatic(directory=str(Path(__file__).parent / "static")), name="static")

# Every template stamps ?v=<app_version> on its entry-point assets, forcing
# a fresh fetch on each release even for clients that cached under the old
# heuristic (set before any router import can render a template).
pages.templates.env.globals["app_version"] = __version__

# Pages first so "/" is handled by the pages router (redirects to /desk).
app.include_router(pages.router, tags=["pages"])
app.include_router(desk.router, prefix="/api/desk", tags=["desk"])
app.include_router(symbols.router, prefix="/api/symbols", tags=["symbols"])


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": __version__, "app": "trading-desk"}
