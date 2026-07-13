"""The attention system: tripwires (desk_watch), planned wakes (desk_wakes),
the streamer's pure evaluator, and the desk surface.

Design under test: cheap code (the always-on streamer) watches the tape
continuously; the expensive brain grants itself extra runs only through the
wake-plan budget gate, with a stated reason, capped per ET day.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'attn.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


# ── watch-set / list / clear ──


def test_watch_set_validation(store):
    from agent.brain import watch_set

    assert not watch_set(store, symbol="AMD", reason="x")["ok"]          # neither
    assert not watch_set(store, symbol="AMD", above=1, below=2, reason="x")["ok"]
    assert not watch_set(store, symbol="AMD", below=540, reason="  ")["ok"]  # no reason
    assert not watch_set(store, symbol="AMD", below=-5, reason="x")["ok"]

    r = watch_set(store, symbol="amd", below=540.0, reason="kill level",
                  run_id="R1")
    assert r["ok"] and r["symbol"] == "AMD" and r["kind"] == "below"


def test_watch_list_buckets_and_lazy_expiry(store):
    from agent.brain import watch_list, watch_set

    watch_set(store, symbol="AMD", below=540, reason="kill")
    watch_set(store, symbol="SPY", above=760, reason="breakout",
              until=str(datetime.utcnow() - timedelta(hours=1)))  # already past
    store.update("desk_watch", {"symbol": "AMD"},
                 {"status": "tripped", "tripped_price": 539.2}, returning=False)

    out = watch_list(store)
    assert [w["symbol"] for w in out["tripped"]] == ["AMD"]
    assert out["armed"] == []  # SPY expired lazily, not reported as armed
    assert "done" not in out
    assert watch_list(store, include_done=True)["done"]


def test_watch_clear(store):
    from agent.brain import watch_clear, watch_set

    wid = watch_set(store, symbol="AMD", below=540, reason="kill")["id"]
    assert watch_clear(store, watch_id=wid)["ok"]
    assert store.select("desk_watch", filters={"id": wid})[0]["status"] == "disarmed"
    assert not watch_clear(store, watch_id=99999)["ok"]


# ── the streamer's pure evaluator ──


def _q(bid, ask, age_secs=0.0):
    return {"bid": bid, "ask": ask, "recv": time.time() - age_secs}


def test_evaluate_watches_trips_on_fresh_mid():
    from agent.streamer import evaluate_watches

    watches = [
        {"id": 1, "symbol": "AMD", "kind": "below", "level": 540.0},
        {"id": 2, "symbol": "SPY", "kind": "above", "level": 750.0},
        {"id": 3, "symbol": "NVDA", "kind": "below", "level": 100.0},
    ]
    quotes = {"AMD": _q(539.0, 539.4),      # mid 539.2 <= 540 → trip
              "SPY": _q(750.1, 750.3),      # mid 750.2 >= 750 → trip
              "NVDA": _q(180.0, 180.2)}     # far away → no trip
    tripped, expired = evaluate_watches(watches, quotes)
    assert sorted(w["id"] for w in tripped) == [1, 2]
    assert not expired
    amd = next(w for w in tripped if w["id"] == 1)
    assert amd["tripped_price"] == pytest.approx(539.2)


def test_evaluate_watches_never_trips_on_stale_or_bad_quotes():
    from agent.streamer import evaluate_watches

    watches = [{"id": 1, "symbol": "AMD", "kind": "below", "level": 540.0}]
    stale = {"AMD": _q(539.0, 539.4, age_secs=3600)}
    assert evaluate_watches(watches, stale) == ([], [])
    crossed = {"AMD": {"bid": 539.0, "ask": 500.0, "recv": time.time()}}
    assert evaluate_watches(watches, crossed) == ([], [])
    assert evaluate_watches(watches, {}) == ([], [])


def test_evaluate_watches_expires_past_until():
    from agent.streamer import evaluate_watches

    watches = [{"id": 1, "symbol": "AMD", "kind": "below", "level": 540.0,
                "until": str(datetime.utcnow() - timedelta(minutes=1))}]
    tripped, expired = evaluate_watches(
        watches, {"AMD": _q(500.0, 500.2)})  # would trip, but it's expired
    assert not tripped and [w["id"] for w in expired] == [1]


# ── wake-plan: the budget gate ──


def test_wake_plan_records_and_reports_budget(store):
    from agent.brain import wake_plan

    at = datetime.utcnow() + timedelta(minutes=30)
    r = wake_plan(store, at=at.isoformat(), reason="NVDA near kill", run_id="R1")
    assert r["ok"] and r["budget_left_today"] == 19
    rows = store.select("desk_wakes")
    assert len(rows) == 1 and rows[0]["reason"] == "NVDA near kill"


def test_wake_plan_rejects_past_soon_and_unparseable(store):
    from agent.brain import wake_plan

    past = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    assert not wake_plan(store, at=past, reason="x")["ok"]
    soon = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
    assert "too soon" in wake_plan(store, at=soon, reason="x")["error"]
    assert not wake_plan(store, at="not-a-time", reason="x")["ok"]
    ok_at = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
    assert not wake_plan(store, at=ok_at, reason="   ")["ok"]


def test_wake_plan_enforces_min_gap(store):
    from agent.brain import wake_plan

    base = datetime.utcnow() + timedelta(minutes=60)
    assert wake_plan(store, at=base.isoformat(), reason="a")["ok"]
    close = base + timedelta(minutes=10)  # < 15-min gap to the planned wake
    r = wake_plan(store, at=close.isoformat(), reason="b")
    assert not r["ok"] and "already planned" in r["error"]
    far = base + timedelta(minutes=20)
    assert wake_plan(store, at=far.isoformat(), reason="c")["ok"]


def test_wake_plan_enforces_daily_cap(store):
    from agent.brain import WAKE_MAX_PER_DAY, wake_plan

    base = datetime.utcnow() + timedelta(minutes=30)
    # Seed the day's budget as spent (direct inserts — cheap and exact).
    store.insert("desk_wakes", [
        {"account": "agent", "at": base + timedelta(minutes=20 * i),
         "reason": f"seed {i}", "created_at": datetime.utcnow()}
        for i in range(WAKE_MAX_PER_DAY)
    ], returning=False)
    r = wake_plan(store, at=(base + timedelta(minutes=20 * WAKE_MAX_PER_DAY))
                  .isoformat(), reason="one too many")
    assert not r["ok"] and "budget spent" in r["error"]


# ── the desk surface ──


def test_watch_endpoint(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps
    from agent.brain import wake_plan, watch_set

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    watch_set(store, symbol="AMD", below=540, reason="kill level", run_id="R1")
    wake_plan(store, at=(datetime.utcnow() + timedelta(minutes=45)).isoformat(),
              reason="decide before close", run_id="R1")

    from dashboard.app import app

    with TestClient(app) as c:
        r = c.get("/api/desk/watch").json()
        assert r["watches"][0]["symbol"] == "AMD"
        assert r["watches"][0]["status"] == "armed"
        assert r["wakes"][0]["reason"] == "decide before close"
