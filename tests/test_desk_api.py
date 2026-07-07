"""Smoke the trading-desk page + /api/desk/* endpoints on a seeded SQLite DB."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'desk.db'}")
    monkeypatch.setenv("EDGEFINDER_SCHEDULER_ENABLED", "false")
    # No data provider in tests: blank the key so init_services returns early
    # (no scheduler, no network plan-probe). The desk endpoints don't use it.
    from config.settings import settings as _settings
    monkeypatch.setattr(_settings, "polygon_api_key", "", raising=False)

    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401
    import agent.models  # noqa: F401
    from agent.models import (
        ACCOUNT, DeskDecision, DeskEquity, DeskPosition, DeskThinking, DeskTrade,
    )
    import agent.data as agent_data
    import dashboard.dependencies as deps

    engine = get_engine()
    Base.metadata.create_all(engine)
    agent_data._session_factory = None
    deps._engine = deps._session_factory = None

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        sess.add(DeskTrade(account=ACCOUNT, run_id="R1", symbol="NVDA", side="BUY",
                           shares=100, price=120.0, dollars=12000.0, ts=now))
        sess.add(DeskPosition(account=ACCOUNT, symbol="NVDA", shares=100,
                              avg_price=120.0, last_price=130.0, opened_at=now, marked_at=now))
        sess.add(DeskEquity(account=ACCOUNT, ts=now, cash=88000.0, positions_value=13000.0,
                            equity=101000.0, return_pct=1.0))
        sess.add(DeskThinking(account=ACCOUNT, run_id="R1", phase="research",
                              text="NVDA momentum strong", ts=now))
        sess.add(DeskDecision(account=ACCOUNT, run_id="R1", ts=now, regime="risk_on",
                              summary="added NVDA", target_weights={"NVDA": 0.13},
                              picks=[{"symbol": "NVDA", "action": "buy", "why_now": "breakout"}],
                              watchlist=[{"symbol": "AAPL", "note": "near trigger"}],
                              strategy_version=1))
        sess.commit()
    finally:
        sess.close()

    from dashboard.app import app
    with TestClient(app) as c:
        yield c


def test_desk_page_renders(client):
    r = client.get("/desk")
    assert r.status_code == 200
    assert "Trading Desk" in r.text
    assert "/static/js/pages/desk.js" in r.text


def test_portfolio_endpoint(client):
    r = client.get("/api/desk/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["positions"][0]["symbol"] == "NVDA"
    # cash = 100k start - 12k buy; equity = cash + marked positions value
    assert abs(body["cash"] - 88000.0) < 0.01
    assert body["equity"] == pytest.approx(88000.0 + 100 * 130.0, abs=0.01)


def test_decision_and_thinking(client):
    d = client.get("/api/desk/decision/latest").json()
    assert d["exists"] and d["picks"][0]["symbol"] == "NVDA"
    t = client.get("/api/desk/thinking").json()
    assert t["run_id"] == "R1" and t["lines"]
    e = client.get("/api/desk/equity").json()
    assert e and e[-1]["equity"] == 101000.0


def test_whatsnew_empty_then_announced(client):
    # nothing shipped yet → empty feed, no spotlight badge
    empty = client.get("/api/desk/whatsnew").json()
    assert empty["entries"] == [] and empty["new_count"] == 0 and empty["latest"] is None

    # the routine announces a shipped improvement via the agent tool
    from agent.announce import announce, recent
    new_id = announce("Drawdown band on the equity curve",
                      "The equity chart now shades peak-to-trough drawdowns so "
                      "you can see how deep the book's dips ran.",
                      kind="feature", version="6.1.0")
    assert isinstance(new_id, int)
    assert recent()[0]["title"].startswith("Drawdown band")

    body = client.get("/api/desk/whatsnew").json()
    assert body["new_count"] == 1
    assert body["latest"]["title"].startswith("Drawdown band")
    assert body["latest"]["kind"] == "feature" and body["latest"]["version"] == "6.1.0"
    assert "shades peak-to-trough" in body["entries"][0]["detail"]


def test_announce_validates_kind(client):
    from agent.announce import announce
    with pytest.raises(ValueError):
        announce("bad", kind="totally-not-valid")
    with pytest.raises(ValueError):
        announce("   ")  # blank title rejected


def test_broker_health_no_keys(client, monkeypatch):
    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    from agent import broker as _b
    monkeypatch.setattr(_b.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(_b.settings, "alpaca_api_secret", "", raising=False)
    body = client.get("/api/desk/broker-health").json()
    assert body["keys_present"] is False and "keys" in body["error"]
