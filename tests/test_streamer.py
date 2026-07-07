"""Unit-test the live-streamer's pure logic: cache, staleness, symbol config.

The WebSocket loop itself needs live keys (proven on Render); everything it
writes through is exercised here without network.
"""

from __future__ import annotations

import time

from agent.streamer import QuoteCache, stream_symbols


def test_stream_symbols_parses_and_dedupes(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "stream_symbols", " spy, QQQ ,spy,, nvda ", raising=False)
    assert stream_symbols() == ["SPY", "QQQ", "NVDA"]


def test_cache_quote_trade_and_snapshot(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "stream_stale_secs", 15, raising=False)
    c = QuoteCache()
    c.update_quote("SPY", 745.49, 745.51, 3, 5, "2026-07-07T15:40:12Z")
    c.update_trade("SPY", 745.50, "2026-07-07T15:40:12.5Z")
    snap = c.snapshot()
    q = snap["quotes"]["SPY"]
    assert q["bid"] == 745.49 and q["ask"] == 745.51 and q["mid"] == 745.5
    assert q["last"] == 745.50
    assert q["stale"] is False and q["age_secs"] < 5
    assert snap["symbols"] == 1 and "server_ts" in snap


def test_cache_staleness(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "stream_stale_secs", 15, raising=False)
    c = QuoteCache()
    c.update_quote("QQQ", 740.0, 740.1, 1, 1, None)
    c._q["QQQ"]["recv"] = time.time() - 60  # simulate an old tick
    assert c.snapshot()["quotes"]["QQQ"]["stale"] is True


def test_warm_does_not_clobber_ws_data():
    c = QuoteCache()
    c.update_quote("SPY", 745.0, 745.1, 1, 1, None)          # fresh WS tick
    c.warm({"SPY": {"bid": 1.0, "ask": 1.1, "mid": 1.05},    # REST warm arrives late
            "IWM": {"bid": 210.0, "ask": 210.1, "mid": 210.05}})
    snap = c.snapshot()["quotes"]
    assert snap["SPY"]["bid"] == 745.0        # WS data preserved
    assert snap["IWM"]["bid"] == 210.0        # gap filled from warm
    assert snap["IWM"].get("warmed") is True


def test_quotes_endpoint_serves_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'q.db'}")
    from config.settings import settings
    monkeypatch.setattr(settings, "polygon_api_key", "", raising=False)
    from fastapi.testclient import TestClient
    import agent.streamer as streamer
    from edgefinder.db.engine import Base, get_engine
    import dashboard.dependencies as deps
    Base.metadata.create_all(get_engine())
    deps._engine = deps._session_factory = None

    streamer.cache.update_quote("NVDA", 130.0, 130.2, 2, 2, None)
    from dashboard.app import app
    with TestClient(app) as client:
        body = client.get("/api/desk/quotes").json()
    assert body["quotes"]["NVDA"]["mid"] == 130.1
    assert body["connected"] in (True, False)
