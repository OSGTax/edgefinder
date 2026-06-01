"""Auth gate for the externally-triggered EOD endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from dashboard.routers import admin


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(admin.router, prefix="/api/admin")
    return TestClient(app)


def test_disabled_when_no_token(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "")
    assert client.post("/api/admin/run-eod").status_code == 503


def test_rejects_missing_header(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "secret")
    assert client.post("/api/admin/run-eod").status_code == 401


def test_rejects_bad_token(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "secret")
    resp = client.post(
        "/api/admin/run-eod", headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 401


def test_accepts_valid_token(client, monkeypatch):
    monkeypatch.setattr(settings, "eod_trigger_token", "secret")
    # Stub the worker so the request doesn't kick off real jobs.
    monkeypatch.setattr("dashboard.services.run_eod_jobs", lambda *a, **k: {})

    resp = client.post(
        "/api/admin/run-eod", headers={"Authorization": "Bearer secret"}
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"
