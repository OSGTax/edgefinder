"""The V4.1 chain-wake dispatcher (agent/streamer.py): fired Routine
sessions cannot arm their own triggers, so the always-on Render process
polls desk_wakes and fires the chain-wakes Routine's API trigger. These
pin the pure decision (dispatch_reason), the CAS slot claim, the /fire
POST shape, and the end-to-end pass including failure paths."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from agent.streamer import (
    DISPATCH_MAX_PER_DAY,
    DISPATCH_MAX_PER_WAKE,
    DISPATCH_MIN_GAP_SECS,
    ROUTINE_FIRE_BETA,
    _run_dispatch_once,
    claim_dispatch_slot,
    dispatch_reason,
    fire_routine,
)

NOW = datetime(2026, 8, 10, 15, 0, 0)


def _wake(wid, minutes_ago, honored=None, dispatch_count=0):
    return {"id": wid, "at": NOW - timedelta(minutes=minutes_ago),
            "honored_run_id": honored, "dispatch_count": dispatch_count}


# ── dispatch_reason: the pure decision ───────────────────────────────


def test_due_wake_fires():
    d = dispatch_reason([_wake(1, 5)], [], now=NOW)
    assert d and d["wake_ids"] == [1] and "due" in d["reason"]


def test_honored_future_and_stale_wakes_do_not_fire():
    wakes = [
        _wake(1, 5, honored="2026-08-10T14:55-abcd"),   # already honored
        _wake(2, -30),                                   # not due yet
        _wake(3, 9 * 60),                                # beyond the 8h lookback
        _wake(4, 5, dispatch_count=DISPATCH_MAX_PER_WAKE),  # attempts spent
    ]
    assert dispatch_reason(wakes, [], now=NOW) is None


def test_min_gap_debounce_blocks():
    recent = [{"ts": NOW - timedelta(seconds=DISPATCH_MIN_GAP_SECS - 10)}]
    assert dispatch_reason([_wake(1, 5)], recent, now=NOW) is None
    aged = [{"ts": NOW - timedelta(seconds=DISPATCH_MIN_GAP_SECS + 10)}]
    assert dispatch_reason([_wake(1, 5)], aged, now=NOW) is not None


def test_daily_cap_blocks():
    disp = [{"ts": NOW - timedelta(minutes=10 + i * 6)}
            for i in range(DISPATCH_MAX_PER_DAY)]
    assert dispatch_reason([_wake(1, 5)], disp, now=NOW) is None


def test_iso_string_timestamps_accepted():
    wakes = [{"id": 7, "at": (NOW - timedelta(minutes=3)).isoformat(),
              "honored_run_id": None, "dispatch_count": 0}]
    disp = [{"ts": (NOW - timedelta(hours=2)).isoformat() + "Z"}]
    d = dispatch_reason(wakes, disp, now=NOW)
    assert d and d["wake_ids"] == [7]


# ── the chain-restart branch (V4.1.1 — the floor Routine's job) ──────


def test_quiet_chain_restarts_when_no_recent_fire():
    d = dispatch_reason([], [], now=NOW, chain_quiet=True)
    assert d and d["wake_ids"] == [] and "restart" in d["reason"]


def test_restart_paced_by_recent_sent_fire():
    from agent.streamer import RESTART_MIN_GAP_SECS
    recent = [{"ts": NOW - timedelta(seconds=RESTART_MIN_GAP_SECS - 60),
               "status": "sent"}]
    assert dispatch_reason([], recent, now=NOW, chain_quiet=True) is None
    aged = [{"ts": NOW - timedelta(seconds=RESTART_MIN_GAP_SECS + 60),
             "status": "sent"}]
    assert dispatch_reason([], aged, now=NOW, chain_quiet=True) is not None


def test_active_chain_never_restart_fires():
    assert dispatch_reason([], [], now=NOW, chain_quiet=False) is None


def test_due_wake_takes_precedence_over_restart():
    d = dispatch_reason([_wake(1, 5)], [], now=NOW, chain_quiet=True)
    assert d and d["wake_ids"] == [1] and "due" in d["reason"]


# ── store-backed pieces (SQLite) ─────────────────────────────────────


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'disp.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import agent.store as agent_store
    Base.metadata.create_all(get_engine())
    agent_store._store = None
    return agent_store.get_store()


def test_claim_slot_is_at_most_once(store):
    decision = {"reason": "1 wake-plan(s) due", "wake_ids": [1]}
    first = claim_dispatch_slot(store, decision, now=NOW)
    assert first is not None
    # same debounce bucket → the sibling loses the CAS race
    assert claim_dispatch_slot(store, decision,
                               now=NOW + timedelta(seconds=5)) is None
    # next bucket → a fresh slot
    later = NOW + timedelta(seconds=DISPATCH_MIN_GAP_SECS + 1)
    assert claim_dispatch_slot(store, decision, now=later) is not None


def test_run_dispatch_once_fires_and_stamps(store):
    store.insert("desk_wakes", {"account": "agent",
                                "at": NOW - timedelta(minutes=4),
                                "reason": "chain: test", "dispatch_count": 0},
                 returning=False)
    fired = []
    out = _run_dispatch_once(store, now=NOW, fire=lambda r: fired.append(r) or 200,
                             quiet_fn=lambda s, n: False)
    assert out is not None and fired == ["1 wake-plan(s) due"]
    d = store.select("desk_dispatches", limit=5)
    assert len(d) == 1 and d[0]["status"] == "sent" and d[0]["http_status"] == 200
    w = store.select("desk_wakes", limit=5)[0]
    assert int(w["dispatch_count"]) == 1 and not w.get("honored_run_id")


def test_run_dispatch_once_terminal_resolves_spent_wakes(store):
    store.insert("desk_wakes", {"account": "agent",
                                "at": NOW - timedelta(minutes=30),
                                "reason": "chain: spent",
                                "dispatch_count": DISPATCH_MAX_PER_WAKE},
                 returning=False)
    out = _run_dispatch_once(store, now=NOW, fire=lambda r: 200,
                             quiet_fn=lambda s, n: False)
    assert out is None  # nothing eligible to fire
    w = store.select("desk_wakes", limit=5)[0]
    assert w["honored_run_id"] == "missed:auto"


def test_run_dispatch_once_marks_failed_and_journals_on_401(store):
    store.insert("desk_wakes", {"account": "agent",
                                "at": NOW - timedelta(minutes=4),
                                "reason": "chain: test", "dispatch_count": 0},
                 returning=False)

    class Rejected(Exception):
        code = 401

    def bad_fire(reason):
        raise Rejected("token rejected")

    out = _run_dispatch_once(store, now=NOW, fire=bad_fire,
                             quiet_fn=lambda s, n: False)
    assert out is None
    d = store.select("desk_dispatches", limit=5)
    assert d[0]["status"] == "failed" and d[0]["http_status"] == 401
    notes = store.select("desk_journal", limit=5)
    assert any("token rejected" in (n.get("title") or "") for n in notes)
    # the wake keeps its attempt count untouched on a failed POST — the
    # next bucket retries it rather than burning attempts on our own 401s
    w = store.select("desk_wakes", limit=5)[0]
    assert int(w["dispatch_count"]) == 0


# ── the /fire POST shape ─────────────────────────────────────────────


def test_fire_routine_post_shape(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "routine_fire_url",
                        "https://api.anthropic.com/v1/claude_code/routines/trig_x/fire")
    monkeypatch.setattr(settings, "routine_fire_token", "sk-ant-oat01-test")
    seen = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["headers"] = dict(req.header_items())
        seen["body"] = json.loads(req.data.decode())
        return FakeResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert fire_routine("2 wake-plan(s) due") == 200
    assert seen["url"].endswith("/trig_x/fire") and seen["method"] == "POST"
    hdrs = {k.lower(): v for k, v in seen["headers"].items()}
    assert hdrs["authorization"] == "Bearer sk-ant-oat01-test"
    assert hdrs["anthropic-beta"] == ROUTINE_FIRE_BETA
    assert seen["body"] == {"text": "2 wake-plan(s) due"}


def test_run_dispatch_once_restart_path(store):
    """No wakes at all + a quiet chain -> the restart branch fires and the
    ledger records it (the floor Routine's job, absorbed — V4.1.1)."""
    fired = []
    out = _run_dispatch_once(store, now=NOW, fire=lambda r: fired.append(r) or 200,
                             quiet_fn=lambda s, n: True)
    assert out is not None and "restart" in out["reason"]
    assert fired and "restart" in fired[0]
    d = store.select("desk_dispatches", limit=5)
    assert d[0]["status"] == "sent" and d[0]["http_status"] == 200
