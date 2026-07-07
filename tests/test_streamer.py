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


def test_pump_authenticated_frame_no_nameerror():
    """Regression (P2 verifier): the authenticated frame must not kill the
    stream — v7.2.0 had a NameError here that froze the prod tape."""
    import asyncio
    import json as _json

    from agent import streamer

    class WS:
        def __init__(self, msgs):
            self.msgs = msgs

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.msgs:
                raise StopAsyncIteration
            return self.msgs.pop(0)

    frames = [_json.dumps([{"T": "success", "msg": "authenticated"}]),
              _json.dumps([{"T": "q", "S": "SPY", "bp": 745.0, "ap": 745.1,
                            "bs": 1, "as": 2, "t": "2026-07-07T16:00:00Z"}])]
    asyncio.run(streamer._pump(WS(frames)))  # must not raise
    assert streamer.cache.connected is True
    assert streamer.cache.get("SPY")["bid"] == 745.0
    streamer.cache.connected = False  # reset module state for other tests


def test_warm_refreshes_stale_and_warmed_entries(monkeypatch):
    """Regression (P2 verifier): warm() must refresh warmed/stale entries —
    the old skip-if-recv logic made every re-warm a permanent no-op."""
    import time as _time

    from config.settings import settings
    monkeypatch.setattr(settings, "stream_stale_secs", 15, raising=False)
    c = QuoteCache()
    # warmed entry → re-warm must replace it
    c.warm({"SPY": {"bid": 1.0, "ask": 1.1, "mid": 1.05}})
    c.warm({"SPY": {"bid": 2.0, "ask": 2.1, "mid": 2.05}})
    assert c.get("SPY")["bid"] == 2.0
    # stale WS entry → re-warm must replace it
    c.update_quote("QQQ", 700.0, 700.1, 1, 1, None)
    c._q["QQQ"]["recv"] = _time.time() - 60
    c.warm({"QQQ": {"bid": 701.0, "ask": 701.1, "mid": 701.05}})
    assert c.get("QQQ")["bid"] == 701.0
    # fresh WS entry → protected
    c.update_quote("IWM", 296.0, 296.1, 1, 1, None)
    c.warm({"IWM": {"bid": 1.0, "ask": 1.1, "mid": 1.05}})
    assert c.get("IWM")["bid"] == 296.0
