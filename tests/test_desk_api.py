"""Smoke the trading-desk page + /api/desk/* endpoints on a seeded SQLite DB."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'desk.db'}")
    monkeypatch.setenv("EDGEFINDER_SCHEDULER_ENABLED", "false")

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


def test_desk_movers(client):
    """Movers endpoint ranks gainers/losers/most-active from daily_bars."""
    from datetime import date, datetime, timezone

    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    now = datetime.now(timezone.utc)
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    seed = [
        ("AAA", d0, 100.0, 1_000), ("AAA", d1, 120.0, 2_000),   # +20% gainer
        ("BBB", d0, 100.0, 5_000), ("BBB", d1, 80.0, 9_000),    # -20% loser, biggest $vol
        ("CCC", d0, 50.0, 100), ("CCC", d1, 50.0, 100),         # flat
        ("PENNY", d0, 0.9, 1), ("PENNY", d1, 0.5, 1),           # sub-$1 → filtered
    ]
    sess = agent_data.session_factory()()
    try:
        for sym, dd, close, vol in seed:
            sess.add(DailyBar(symbol=sym, date=dd, open=close, high=close, low=close,
                              close=close, volume=float(vol), source="test", created_at=now))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/movers?top=3").json()
    assert d["as_of"] == "2026-07-07" and d["prior"] == "2026-07-06"
    gain = [x["symbol"] for x in d["gainers"]]
    lose = [x["symbol"] for x in d["losers"]]
    assert gain[0] == "AAA"                      # +20% is the top gainer
    assert lose[0] == "BBB"                       # -20% is the top loser
    assert "PENNY" not in gain and "PENNY" not in lose   # sub-$1 filtered out
    assert d["most_active"][0]["symbol"] == "BBB"        # 80 * 9000 = biggest $ volume
    assert next(x for x in d["gainers"] if x["symbol"] == "AAA")["change_pct"] == 20.0


def test_desk_holding_stats(client):
    """Holding-stats returns day change, 52-week range, and a spark series."""
    from datetime import date, datetime, timedelta, timezone

    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        # NVDA is the seeded holding; give it 10 rising sessions
        base = date(2026, 6, 24)
        for i in range(10):
            px = 100.0 + i  # 100 → 109, last-session change 108→109
            sess.add(DailyBar(symbol="NVDA", date=base + timedelta(days=i),
                              open=px, high=px, low=px, close=px, volume=1000.0,
                              source="test", created_at=now))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/holding-stats?spark_days=30").json()
    assert "NVDA" in d["symbols"]
    s = d["symbols"]["NVDA"]
    assert s["wk52_high"] == 109.0 and s["wk52_low"] == 100.0
    assert s["day_change_pct"] == round((109 - 108) / 108 * 100, 2)
    assert s["spark"][0] == 100.0 and s["spark"][-1] == 109.0


def test_desk_dividends(client):
    """Dividend calendar returns last/next ex-dates for dividend-paying holdings."""
    from datetime import date

    import agent.data as agent_data
    from edgefinder.db.models import DividendRecord

    sess = agent_data.session_factory()()
    try:  # NVDA is the seeded holding; 2099 is unambiguously "upcoming"
        for ex, amt in [(date(2026, 3, 5), 0.10), (date(2026, 6, 5), 0.10),
                        (date(2099, 9, 5), 0.12)]:
            sess.add(DividendRecord(symbol="NVDA", ex_date=ex, cash_amount=amt))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/dividends").json()
    nvda = next(x for x in d["holdings"] if x["symbol"] == "NVDA")
    assert nvda["has_dividend"] is True
    assert nvda["next_ex_date"] == "2099-09-05"          # the only future ex-date
    assert nvda["last_ex_date"] == "2026-06-05"          # most recent past
    assert nvda["ttm_amount"] == round(0.10 + 0.10 + 0.12, 4)


def test_desk_trades_include_fill_quote(client):
    """The trades endpoint surfaces the stamped live bid/ask receipt."""
    from datetime import datetime, timezone

    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskTrade

    sess = agent_data.session_factory()()
    try:
        sess.add(DeskTrade(account=ACCOUNT, run_id="R2", symbol="LLY", side="BUY",
                           shares=2.0, price=1226.46, dollars=2452.92,
                           ts=datetime.now(timezone.utc),
                           fill_quote={"bid": 1224.96, "ask": 1226.34, "mid": 1225.65}))
        sess.commit()
    finally:
        sess.close()

    rows = client.get("/api/desk/trades?limit=20").json()
    lly = next(r for r in rows if r["symbol"] == "LLY")
    assert lly["fill_quote"]["bid"] == 1224.96 and lly["fill_quote"]["ask"] == 1226.34


def test_wiki_endpoint_empty_and_seeded(client):
    body = client.get("/api/desk/wiki").json()
    assert body["pages"] == []          # empty case, no 500

    from agent.brain import set_wiki
    set_wiki(slug="lessons", body="Momentum works in risk-on.\n\n- cite numbers",
             title="Lessons", reason="seed")
    set_wiki(slug="playbook", body="Trend first.", reason="seed")
    body = client.get("/api/desk/wiki").json()
    assert [p["slug"] for p in body["pages"]] == ["playbook", "lessons"]  # canonical order
    assert body["pages"][1]["revision"] == 1
    assert "Momentum works" in body["pages"][1]["body"]
    assert body["pages"][0]["updated_at"]  # ISO timestamp present
