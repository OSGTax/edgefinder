"""Tests for the ops health endpoint (dashboard System Health panel)."""

from __future__ import annotations

from datetime import datetime, timezone

from edgefinder.db.models import AgentObservation, SystemHeartbeat


def test_ops_health_endpoint(db_session):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from dashboard.dependencies import get_db
    from dashboard.routers import ops as ops_router

    db_session.add(SystemHeartbeat(
        component="intraday_cycle",
        last_run_at=datetime(2026, 6, 5, 15, 25),  # naive UTC, as stored
        ok=True, detail={"opened": 0, "closed": 0},
    ))
    db_session.add(AgentObservation(
        agent_name="watchdog", severity="CRITICAL", category="cycle_liveness",
        message="stalled", obs_metadata={"key": "intraday_cycle"},
    ))
    db_session.commit()

    app = FastAPI()
    app.include_router(ops_router.router, prefix="/api/ops")
    app.dependency_overrides[get_db] = lambda: db_session
    body = TestClient(app).get("/api/ops/health").json()

    assert body["heartbeats"][0]["component"] == "intraday_cycle"
    assert body["heartbeats"][0]["ok"] is True
    assert body["heartbeats"][0]["age_minutes"] is not None
    assert body["observation_counts"]["critical"] == 1
    assert body["observations"][0]["category"] == "cycle_liveness"
    assert "scheduler" in body
