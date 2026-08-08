"""The attention system (REBUILD-V4): planned wakes (desk_wakes) as the
self-scheduling BUDGET LEDGER, and chain-health as the hourly floor's gate.

Design under test: the brain grants itself extra runs only through the
wake-plan budget gate (with a stated reason, capped per ET day) — each
honored plan becomes a one-shot Routine trigger the cycle arms itself —
and the chain-restarter floor exits cheaply while the chain is healthy.
The V3 tripwire/hard-stop/dispatcher machinery this file used to test is
gone: protective stops rest on Alpaca's own book.
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

# ── wake-plan: the budget gate ──


def test_wake_plan_records_and_reports_budget(store):
    from agent.brain import wake_plan

    at = datetime.utcnow() + timedelta(minutes=30)
    r = wake_plan(store, at=at.isoformat(), reason="NVDA near kill", run_id="R1")
    from agent.brain import WAKE_MAX_PER_DAY
    assert r["ok"] and r["budget_left_today"] == WAKE_MAX_PER_DAY - 1
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

    # Anchor at tomorrow 14:00 UTC (10:00 ET): the 20 seeds span ~6.7 hours
    # and must all land on ONE ET day for the cap to bind — a "now + 30min"
    # base run in the evening spills seeds past ET midnight and the cap
    # legitimately doesn't trip (time-of-day flake, caught 2026-07-14).
    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    base = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 14, 0)
    # Seed the day's budget as spent (direct inserts — cheap and exact).
    store.insert("desk_wakes", [
        {"account": "agent", "at": base + timedelta(minutes=20 * i),
         "reason": f"seed {i}", "created_at": datetime.utcnow()}
        for i in range(WAKE_MAX_PER_DAY)
    ], returning=False)
    r = wake_plan(store, at=(base + timedelta(minutes=20 * WAKE_MAX_PER_DAY))
                  .isoformat(), reason="one too many")
    assert not r["ok"] and "budget spent" in r["error"]


def test_wake_due_and_honor_loop(store):
    from agent.brain import wake_due, wake_honor

    now = datetime.utcnow()
    store.insert("desk_wakes", [
        {"account": "agent", "at": now - timedelta(minutes=20),
         "reason": "due now", "created_at": now - timedelta(hours=1)},
        {"account": "agent", "at": now + timedelta(hours=2),
         "reason": "future", "created_at": now},
        {"account": "agent", "at": now - timedelta(hours=12),
         "reason": "ancient", "created_at": now - timedelta(hours=13)},
    ], returning=False)

    d = wake_due(store)
    assert [w["reason"] for w in d["due"]] == ["due now"]
    assert [w["reason"] for w in d["missed"]] == ["ancient"]  # reported, not fresh

    wid = d["due"][0]["id"]
    assert wake_honor(store, wake_id=wid, run_id="RID-1")["ok"]
    # Honored exactly once; second honor and second due both refuse.
    assert not wake_honor(store, wake_id=wid, run_id="RID-2")["ok"]
    assert wake_due(store)["due"] == []


def test_wake_honor_is_compare_and_swap(store):
    from agent.brain import wake_honor, wake_plan

    base = datetime.utcnow() + timedelta(minutes=60)
    wid = wake_plan(store, at=base.isoformat(), reason="cas test")["id"]
    a = wake_honor(store, wake_id=wid, run_id="cycle-A")
    b = wake_honor(store, wake_id=wid, run_id="cycle-B")
    assert a["ok"] is True
    assert b["ok"] is False and "cycle-A" in b["error"] or "claimed" in b.get("error", "")
    rows = store.select("desk_wakes", filters={"id": wid})
    assert rows[0]["honored_run_id"] == "cycle-A"


def test_wake_honor_rejects_blank_run_id(store):
    """A blank run id must never reach the row.

    argparse's required=True only checks the flag is PRESENT, so `--run-id ""`
    (a shell variable that didn't survive between tool calls) used to write an
    empty string into honored_run_id — falsy, so wake_due reported the wake
    due forever, while the IS NULL claim filter could never match it."""
    from agent.brain import wake_honor, wake_plan

    base = datetime.utcnow() + timedelta(minutes=60)
    wid = wake_plan(store, at=base.isoformat(), reason="blank run id")["id"]
    for bad in ("", "   ", None):
        res = wake_honor(store, wake_id=wid, run_id=bad)
        assert res["ok"] is False
        assert "non-empty" in res["error"]
    # the row is untouched, so a real cycle can still claim it
    assert not store.select("desk_wakes", filters={"id": wid})[0][
        "honored_run_id"]
    assert wake_honor(store, wake_id=wid, run_id="cycle-A")["ok"] is True


def test_wake_honor_recovers_a_row_poisoned_with_empty_string(store):
    """Rows already poisoned before the fix must still be claimable.

    Regression for wakes #99/#108 (2026-07-24), which sat permanently due and
    permanently unclaimable across 10+ cycles."""
    from agent.brain import wake_due, wake_honor, wake_plan

    past = datetime.utcnow() - timedelta(minutes=30)
    wid = wake_plan(store, at=(datetime.utcnow() + timedelta(
        minutes=60)).isoformat(), reason="poisoned")["id"]
    # simulate the pre-fix damage: due in the past, honored_run_id == ""
    store.update("desk_wakes", {"id": wid},
                 {"honored_run_id": "", "at": past}, returning=False)

    # the symptom: falsy honored_run_id keeps it reported as due
    assert any(e["id"] == wid for e in wake_due(store)["due"])
    # the fix: it is claimable again, exactly once
    ok = wake_honor(store, wake_id=wid, run_id="cycle-A")
    assert ok["ok"] is True
    assert store.select("desk_wakes", filters={"id": wid})[0][
        "honored_run_id"] == "cycle-A"
    assert wake_honor(store, wake_id=wid, run_id="cycle-B")["ok"] is False
    assert not any(e["id"] == wid for e in wake_due(store)["due"])


# ── chain-health: the hourly floor's gate ──


def test_chain_health_quiet_chain_during_desk_hours_runs(store):
    from agent.brain import chain_health

    # Tuesday 2026-08-04 14:00 UTC = 10:00 ET, desk hours, no decisions yet.
    now = datetime(2026, 8, 4, 14, 0)
    h = chain_health(store, now=now)
    assert h["desk_hours"] is True
    assert h["should_run"] is True  # chain never started — restart it


def test_chain_health_recent_cycle_is_a_cheap_no_op(store):
    from agent.brain import chain_health

    now = datetime(2026, 8, 4, 14, 0)
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R-recent",
        "ts": now - timedelta(minutes=10), "picks": []}, returning=False)
    h = chain_health(store, now=now)
    assert h["should_run"] is False
    assert h["last_cycle_minutes_ago"] == pytest.approx(10, abs=0.2)


def test_chain_health_due_wake_always_runs_even_when_chain_is_fresh(store):
    from agent.brain import chain_health, wake_plan

    # wake_plan validates against the real clock, so plan a real-future wake
    # then backdate it into the due window — with a fresh cycle on the
    # books, so ONLY the due wake can be the reason to run.
    real_now = datetime.utcnow()
    at = (real_now + timedelta(minutes=20)).isoformat() + "Z"
    r = wake_plan(store, at=at, reason="chain: next look")
    assert r["ok"]
    store.update("desk_wakes", {"id": r["id"]},
                 {"at": real_now - timedelta(minutes=5)}, returning=False)
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R-fresh",
        "ts": real_now - timedelta(minutes=5), "picks": []}, returning=False)
    h = chain_health(store, now=real_now)
    assert h["wakes_due"] is True and h["should_run"] is True


def test_chain_health_weekend_without_due_wakes_sleeps(store):
    from agent.brain import chain_health

    saturday = datetime(2026, 8, 8, 14, 0)  # Saturday 10:00 ET
    h = chain_health(store, now=saturday)
    assert h["desk_hours"] is False and h["should_run"] is False
