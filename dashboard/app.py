"""EdgeFinder v2 — FastAPI dashboard application.

Central web app that serves the dashboard UI and all API endpoints.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from dashboard.routers import benchmarks, inject, research, strategies, trades
from edgefinder.core.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

__version__ = "4.9.0"

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _cors_origins() -> list[str]:
    """Read `EDGEFINDER_CORS_ORIGINS` (comma-split). Default is dev-only.

    Production deployments should set this to an explicit comma-separated
    list of origins. Setting the value to "*" restores the old wide-open
    behavior but disables credentialed requests (browsers reject `*` with
    allow_credentials=True).
    """
    raw = os.getenv("EDGEFINDER_CORS_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:8000", "http://127.0.0.1:8000"]
    return [o.strip() for o in raw.split(",") if o.strip()]


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

_cors = _cors_origins()
_cors_wildcard = _cors == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    # Browsers disallow credentialed requests when allow_origins=="*", so
    # we have to flip allow_credentials off in that mode.
    allow_credentials=not _cors_wildcard,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Bearer-token auth ────────────────────────────────────
#
# When EDGEFINDER_DASHBOARD_TOKEN is set, every request outside the
# unauthenticated exempt list below must present `Authorization: Bearer
# <token>`. When the env is unset (dev), the middleware is a no-op.

_AUTH_EXEMPT_PREFIXES = ("/api/health", "/docs", "/redoc", "/openapi.json")
_AUTH_EXEMPT_PATHS = ("/",)


@app.middleware("http")
async def _bearer_auth(request: Request, call_next):
    token = os.getenv("EDGEFINDER_DASHBOARD_TOKEN", "").strip()
    if not token:
        return await call_next(request)

    path = request.url.path
    if request.method == "OPTIONS":
        return await call_next(request)
    if path in _AUTH_EXEMPT_PATHS or any(
        path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES
    ):
        return await call_next(request)

    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Missing bearer token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    supplied = header.split(" ", 1)[1].strip()
    if supplied != token:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid bearer token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)

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
