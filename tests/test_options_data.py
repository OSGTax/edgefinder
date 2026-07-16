"""O3 tests: the chain-summary reduction (pure), the IV data bank persistence,
and the /api/desk/options endpoints (graceful without keys)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

TODAY = date.today()
E1 = (TODAY + timedelta(days=1)).isoformat()    # too near — skipped for focus
E2 = (TODAY + timedelta(days=10)).isoformat()   # the focus expiry
E3 = (TODAY + timedelta(days=38)).isoformat()


def row(type_, strike, expiry=E2, bid=None, ask=None, iv=None, delta=None):
    return {"symbol": f"X{expiry.replace('-','')[2:]}{type_}{int(strike*1000):08d}",
            "type": type_, "strike": strike, "expiry": expiry,
            "dte": (date.fromisoformat(expiry) - TODAY).days,
            "bid": bid, "ask": ask,
            "mid": round((bid + ask) / 2, 4) if (bid and ask) else None,
            "iv": iv, "delta": delta, "theta": None}


CHAIN = [
    # focus expiry (E2), spot 100
    row("C", 95, bid=6.0, ask=6.2, iv=0.31, delta=0.78),
    row("C", 100, bid=2.4, ask=2.6, iv=0.30, delta=0.52),
    row("C", 105, bid=0.8, ask=1.0, iv=0.29, delta=0.26),
    row("P", 95, bid=0.7, ask=0.9, iv=0.34, delta=-0.24),
    row("P", 100, bid=2.3, ask=2.5, iv=0.32, delta=-0.48),
    row("P", 105, bid=5.9, ask=6.1, iv=0.33, delta=-0.72),
    # a nearer expiry that must be skipped for focus (dte < 3)
    row("C", 100, expiry=E1, bid=1.0, ask=1.2, iv=0.45, delta=0.5),
    # a later expiry present in the expiries list
    row("C", 100, expiry=E3, bid=4.0, ask=4.4, iv=0.28, delta=0.55),
]


def test_summarize_chain_reduction():
    from agent.options_data import summarize_chain

    s = summarize_chain(CHAIN, spot=100.0, today=TODAY)
    assert s["available"] and s["expiry"] == E2 and s["dte"] == 10
    assert s["atm_strike"] == 100
    assert s["atm_iv"] == round((0.30 + 0.32) / 2, 4)
    # straddle: call mid 2.5 + put mid 2.4 = 4.9 → 4.9% of spot
    assert s["expected_move_dollars"] == 4.9
    assert s["expected_move_pct"] == 4.9
    # 25d skew: put iv(0.34 @ -0.24) - call iv(0.29 @ 0.26) = +0.05
    assert s["skew_25d"] == round(0.34 - 0.29, 4)
    assert len(s["expiries"]) == 3
    # strikes table covers ±10% and both sides
    assert [r["strike"] for r in s["calls_table"]] == [95, 100, 105]
    assert [r["strike"] for r in s["puts_table"]] == [95, 100, 105]


def test_summarize_chain_empty():
    from agent.options_data import summarize_chain
    assert summarize_chain([], spot=100.0)["available"] is False
    assert summarize_chain(CHAIN, spot=None)["available"] is False


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'ivbank.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def test_persist_snapshot_once_per_day(store):
    from agent.options_data import history, persist_snapshot

    s = {"available": True, "symbol": "SPY", "spot": 748.5, "atm_iv": 0.14,
         "expected_move_pct": 1.2, "skew_25d": 0.03, "dte": 10, "expiry": E2}
    assert persist_snapshot(store, s) is True
    row = store.select("desk_options_snap",
                       filters={"symbol": "SPY", "snap_date": TODAY}, limit=1)[0]
    assert row["captured_at"] is not None               # capture-time receipt
    assert persist_snapshot(store, s) is False          # same day → no-op
    assert persist_snapshot(store, {"available": False, "symbol": "SPY"}) is False
    # a second day appends; history comes back oldest→newest
    assert persist_snapshot(store, {**s, "atm_iv": 0.15},
                            snap_date=TODAY + timedelta(days=1)) is True
    series = history(store, "spy")
    assert len(series) == 2
    assert series[0]["atm_iv"] == 0.14 and series[1]["atm_iv"] == 0.15


def test_options_endpoints_graceful_without_keys(store, monkeypatch):
    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    from config.settings import settings
    monkeypatch.setattr(settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(settings, "alpaca_api_secret", "", raising=False)
    import agent.options_data as od
    od._cache.clear()
    import dashboard.dependencies as deps
    deps._engine = deps._session_factory = None
    from fastapi.testclient import TestClient
    from dashboard.app import app
    with TestClient(app) as client:
        s = client.get("/api/desk/options/SPY").json()
        assert s["available"] is False and "keys" in s["error"]
        hist = client.get("/api/desk/options/SPY/history").json()
        assert hist["symbol"] == "SPY" and isinstance(hist["series"], list)
