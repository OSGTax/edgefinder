"""The symbol page's live top-up (v10.0.3): stored bars/news that have gone
stale (off-universe names, the intraday gap) are topped up straight from
Alpaca at view time — read-only, merged in the response, degrading to
stored-only when creds are absent or the call fails."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

TODAY = date.today()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'sym.db'}")
    monkeypatch.setenv("EDGEFINDER_SCHEDULER_ENABLED", "false")
    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401
    import agent.models  # noqa: F401
    import agent.data as agent_data
    import dashboard.dependencies as deps
    Base.metadata.create_all(get_engine())
    agent_data._session_factory = None
    deps._engine = deps._session_factory = None
    import dashboard.symbol_service as svc
    svc.clear_cache()
    from fastapi.testclient import TestClient
    from dashboard.app import app
    with TestClient(app) as c:
        yield c
    svc.clear_cache()


def _seed_bars(last_day: date, n: int = 5, symbol: str = "CPB"):
    import agent.data as agent_data
    from edgefinder.db.models import DailyBar
    sess = agent_data.session_factory()()
    try:
        for i in range(n):
            d = last_day - timedelta(days=n - 1 - i)
            sess.add(DailyBar(symbol=symbol, date=d, open=30.0, high=31.0,
                              low=29.5, close=30.5, volume=1e6, source="test"))
        sess.commit()
    finally:
        sess.close()


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def test_stale_bars_topped_up_live(client, monkeypatch):
    import dashboard.symbol_service as svc
    stale_end = TODAY - timedelta(days=8)
    _seed_bars(stale_end)

    fetched = {}

    def fake_tail(symbol, after):
        fetched["after"] = after
        days = [after + timedelta(days=i) for i in range(1, 9)
                if (after + timedelta(days=i)) <= TODAY]
        days = [d for d in days if d.weekday() < 5]
        return pd.DataFrame([{"date": d, "open": 31.0, "high": 32.0,
                              "low": 30.8, "close": 31.5, "volume": 2e6}
                             for d in days])

    monkeypatch.setattr(svc, "_live_tail", fake_tail)
    svc.clear_cache()
    out = client.get("/api/symbols/CPB/bars?range=1m").json()
    assert out["source"].endswith("+alpaca")
    assert out["live_through"] is not None
    assert fetched["after"] == stale_end
    last_time = out["bars"][-1]["time"]
    # the newest bar is now the last weekday ≤ today, not 8 days ago
    expect = TODAY
    while expect.weekday() >= 5:
        expect -= timedelta(days=1)
    assert last_time == _epoch(expect)


def test_topup_degrades_to_stored(client, monkeypatch):
    import dashboard.symbol_service as svc
    stale_end = TODAY - timedelta(days=8)
    _seed_bars(stale_end, symbol="ESI")
    monkeypatch.setattr(svc, "_live_tail", lambda symbol, after: None)
    svc.clear_cache()
    out = client.get("/api/symbols/ESI/bars?range=1m").json()
    assert "+alpaca" not in out["source"]
    assert out["live_through"] is None
    assert out["bars"][-1]["time"] == _epoch(stale_end)


def test_fresh_bars_skip_the_fetch(client, monkeypatch):
    import dashboard.symbol_service as svc
    expect = TODAY
    while expect.weekday() >= 5:
        expect -= timedelta(days=1)
    _seed_bars(expect, symbol="UPTD")

    def boom(symbol, after):  # pragma: no cover - the assertion IS the test
        raise AssertionError("must not fetch when stored bars are current")

    monkeypatch.setattr(svc, "_live_tail", boom)
    svc.clear_cache()
    out = client.get("/api/symbols/UPTD/bars?range=1m").json()
    assert out["bars"][-1]["time"] == _epoch(expect)


def test_events_news_topped_up_and_deduped(client, monkeypatch):
    import agent.data as agent_data
    import dashboard.routers.symbols as sym_router
    from edgefinder.db.models import TickerNews

    sess = agent_data.session_factory()()
    try:
        sess.add(TickerNews(symbol="CPB", title="old stored story",
                            article_url="https://x/old",
                            publisher_name="Benzinga",
                            published_utc=str(TODAY - timedelta(days=10))))
        sess.commit()
    finally:
        sess.close()

    live = [
        {"time": _epoch(TODAY), "title": "fresh live story",
         "url": "https://x/new", "publisher": "Benzinga"},
        {"time": _epoch(TODAY - timedelta(days=10)), "title": "dupe",
         "url": "https://x/old", "publisher": "Benzinga"},
    ]
    monkeypatch.setattr(sym_router, "_live_news", lambda s: list(live))
    out = client.get("/api/symbols/CPB/events").json()
    urls = [n["url"] for n in out["news"]]
    assert urls == ["https://x/old", "https://x/new"]  # deduped, sorted
    assert out["news"][-1]["title"] == "fresh live story"


def test_events_degrade_without_live(client, monkeypatch):
    import dashboard.routers.symbols as sym_router
    monkeypatch.setattr(sym_router, "_live_news", lambda s: None)
    out = client.get("/api/symbols/ZZZQ/events").json()
    assert out["news"] == [] and out["dividends"] == []


def test_symbol_quote_live_and_ttl_cached(client, monkeypatch):
    import dashboard.routers.symbols as sym_router
    calls = {"n": 0}

    def fake_snap(sym):
        calls["n"] += 1
        return {"last": 22.64, "last_ts": "t", "bid": 22.63, "ask": 22.65,
                "prev_close": 22.0, "day_change_pct": 2.91,
                "day_bar": {"time": 1754784000, "open": 22.1, "high": 22.7,
                            "low": 22.0, "close": 22.64, "volume": 1e5}}

    monkeypatch.setattr(sym_router, "_fetch_snapshot", fake_snap)
    sym_router._quote_cache = None
    out = client.get("/api/symbols/CPB/quote").json()
    assert out["available"] is True
    assert out["last"] == 22.64 and out["day_change_pct"] == 2.91
    assert out["day_bar"]["close"] == 22.64
    client.get("/api/symbols/CPB/quote")
    assert calls["n"] == 1     # second read served from the TTL cache


def test_symbol_quote_degrades_never_500s(client, monkeypatch):
    import dashboard.routers.symbols as sym_router
    monkeypatch.setattr(sym_router, "_fetch_snapshot", lambda s: None)
    sym_router._quote_cache = None
    r = client.get("/api/symbols/ZZZQ/quote")
    assert r.status_code == 200
    assert r.json() == {"symbol": "ZZZQ", "available": False}
