"""EdgeFinder v2 — FastAPI dashboard application.

Central web app that serves the dashboard UI and all API endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.routers import benchmarks, inject, research, sentiment, strategies, trades

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EdgeFinder v2 dashboard starting")
    yield
    logger.info("EdgeFinder v2 dashboard shutting down")


app = FastAPI(
    title="EdgeFinder v2",
    description="Trading workbench for strategy research and paper trading",
    version="2.0.0",
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
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(inject.router, prefix="/api/inject", tags=["inject"])


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}
