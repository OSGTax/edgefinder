"""Auth gate + single-flight for the externally-triggered intraday endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from dashboard.routers import admin
import dashboard.services as services


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(admin.router, prefix="/api/admin")
    return TestClient(app)


def test_disabled_when_no_token(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "")
    assert client.post("/api/admin/run-intraday").status_code == 503


def test_rejects_missing_header(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "secret")
    assert client.post("/api/admin/run-intraday").status_code == 401


def test_rejects_bad_token(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "secret")
    resp = client.post(
        "/api/admin/run-intraday", headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 401


def test_accepts_valid_token(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "secret")
    # Stub the worker so the request doesn't run a real cycle.
    monkeypatch.setattr(services, "run_intraday_jobs", lambda *a, **k: {})

    resp = client.post(
        "/api/admin/run-intraday", headers={"Authorization": "Bearer secret"}
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"


def test_single_flight_skips_when_locked(monkeypatch):
    # Record whether the cycle jobs run.
    calls: list[str] = []
    monkeypatch.setattr(services, "_signal_check_job", lambda: calls.append("signal"))
    monkeypatch.setattr(services, "_position_monitor_job", lambda: calls.append("monitor"))

    # Hold the lock → a concurrent fire must drop, running neither job.
    assert services._intraday_lock.acquire(blocking=False)
    try:
        assert services.run_intraday_jobs() == {"status": "busy"}
        assert calls == []
    finally:
        services._intraday_lock.release()

    # Lock free → both jobs run, in order.
    report = services.run_intraday_jobs()
    assert report == {"signal_check": "ok", "position_monitor": "ok"}
    assert calls == ["signal", "monitor"]
