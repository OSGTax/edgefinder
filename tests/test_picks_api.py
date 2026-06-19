"""Picks API — the AI analyst account's decision dossier endpoints."""

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from edgefinder.db.models import AgentDecision


@pytest.fixture
def client(db_engine, db_session):
    from dashboard.app import app
    from dashboard.dependencies import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with patch("dashboard.services.init_services"), \
         patch("dashboard.services.shutdown_services"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


def _decision(session, d, target, picks, *, name="ai_analyst", summary="s"):
    session.add(AgentDecision(strategy_name=name, decision_date=d,
                              target_weights=target, picks=picks, summary=summary))
    session.commit()


def test_latest_empty_is_graceful(client):
    r = client.get("/api/picks/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["decision_date"] is None and body["picks"] == []


def test_latest_returns_most_recent_with_counts(client, db_session):
    _decision(db_session, date(2026, 6, 11), {"OLD": 1.0},
              [{"symbol": "OLD", "action": "buy"}])
    _decision(db_session, date(2026, 6, 12),
              {"AAA": 0.5, "BBB": 0.5},
              [{"symbol": "AAA", "action": "hold"},
               {"symbol": "BBB", "action": "buy"},
               {"symbol": "OLD", "action": "sell"}])
    r = client.get("/api/picks/latest")
    body = r.json()
    assert body["decision_date"] == "2026-06-12"
    assert body["counts"] == {"holdings": 1, "new": 1, "sells": 1}
    assert set(body["target_weights"]) == {"AAA", "BBB"}


def test_by_date_and_404(client, db_session):
    _decision(db_session, date(2026, 6, 12), {"AAA": 1.0},
              [{"symbol": "AAA", "action": "buy"}])
    assert client.get("/api/picks/2026-06-12").json()["decision_date"] == "2026-06-12"
    assert client.get("/api/picks/2026-06-13").status_code == 404
    assert client.get("/api/picks/not-a-date").status_code == 400


def test_dates_list(client, db_session):
    _decision(db_session, date(2026, 6, 11), {}, [])
    _decision(db_session, date(2026, 6, 12), {}, [])
    body = client.get("/api/picks/dates").json()
    assert body["dates"] == ["2026-06-12", "2026-06-11"]


def test_run_returns_202(client):
    # the background thread no-ops without a session factory; endpoint accepts
    with patch("dashboard.services.run_analyst_job", return_value=1):
        r = client.post("/api/picks/run")
    assert r.status_code == 202 and r.json()["status"] == "started"
