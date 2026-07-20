"""Commitments gate (SCHEMA.md §4, PLAN.md step 4): the structured
falsification clause that closes the free-text hole — validation at the
write, materialization + linked tripwire on save, the grade sweep, and the
fired-unhonored obligation surfacing in context."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'c.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


COMMIT = {"kind": "reentry", "direction": "above", "level": 325.0,
          "until_sessions": 5,
          "text": "AAPL reclaims $325 with no pullback in 5 sessions — re-add."}


# ── the write-gate ───────────────────────────────────────────────────────


def test_trim_with_free_text_clause_is_rejected(store):
    from agent.brain import save_decision

    r = save_decision(store, run_id="R1", picks=[{
        "symbol": "AAPL", "action": "trim",
        "rationale": "Trim to 15%; re-add if it reclaims $325 with no pullback."}])
    assert not r["ok"]
    assert "conditional clause" in r["error"] and "commitment" in r["error"]
    # nothing booked
    assert store.select("desk_decisions", filters={"run_id": "R1"}) == []


def test_trim_with_structured_commitment_accepted_and_materialized(store):
    from agent.brain import save_decision

    r = save_decision(store, run_id="R1", picks=[{
        "symbol": "AAPL", "action": "trim",
        "rationale": "Trim to 15%; re-add if it reclaims $325.",
        "commitment": COMMIT}])
    assert r["ok"], r
    assert r["commitments"] and r["commitments"][0]["symbol"] == "AAPL"

    rows = store.select("desk_commitments", filters={"run_id": "R1"})
    assert len(rows) == 1
    c = rows[0]
    assert c["kind"] == "reentry" and c["direction"] == "above"
    assert c["level"] == 325.0 and c["status"] == "open"
    # a linked advisory tripwire was armed
    assert c["watch_id"] is not None
    w = store.select("desk_watch", filters={"id": c["watch_id"]})[0]
    assert w["kind"] == "above" and w["level"] == 325.0
    assert "commitment" in w["reason"]


def test_clause_reworded_without_promise_passes(store):
    from agent.brain import save_decision

    # a trim that states no conditional promise needs no commitment
    r = save_decision(store, run_id="R1", picks=[{
        "symbol": "AAPL", "action": "trim",
        "rationale": "Trim to 15% on a Stoch-RSI overheat; thesis intact."}])
    assert r["ok"], r
    assert store.select("desk_commitments", filters={"run_id": "R1"}) == []


def test_malformed_commitment_is_rejected(store):
    from agent.brain import save_decision

    r = save_decision(store, run_id="R1", picks=[{
        "symbol": "AAPL", "action": "trim", "rationale": "re-add later",
        "commitment": {"kind": "reentry", "text": "x", "until_sessions": 5}}])
    assert not r["ok"]
    assert "direction" in r["error"] or "level" in r["error"]


def test_materialization_is_idempotent_on_resave(store):
    from agent.brain import save_decision

    save_decision(store, run_id="R1", picks=[{
        "symbol": "AAPL", "action": "trim", "rationale": "re-add if $325",
        "commitment": COMMIT}])
    # amend the same run — must not duplicate the commitment or its wire
    save_decision(store, run_id="R1", picks=[{
        "symbol": "AAPL", "action": "trim", "rationale": "re-add if $325",
        "commitment": COMMIT}])
    assert len(store.select("desk_commitments", filters={"run_id": "R1"})) == 1


def test_opening_pick_still_needs_the_prediction_registry(store):
    """The commitments gate does not relax the buy/add registry."""
    from agent.brain import save_decision

    r = save_decision(store, run_id="R1", picks=[{
        "symbol": "NVDA", "action": "buy"}])
    assert not r["ok"] and "prediction registry" in r["error"]


# ── the grade sweep ──────────────────────────────────────────────────────


def _bars(store, symbol, rows):
    for d, close in rows:
        store.insert("daily_bars", {
            "symbol": symbol, "date": d, "open": close, "high": close,
            "low": close, "close": close, "volume": 1_000_000}, returning=False)


def test_sweep_fires_when_a_close_touches_the_level(store):
    from agent.ledger import sweep_commitments

    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "ts": datetime(2026, 7, 10, 15, 0),
        "picks": [{"symbol": "AAPL", "action": "trim"}]}, returning=False)
    store.insert("desk_commitments", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "kind": "reentry", "direction": "above", "level": 325.0,
        "until": date(2026, 7, 20), "text": "reclaims 325",
        "status": "open", "created_at": datetime(2026, 7, 10, 15, 0)},
        returning=False)
    _bars(store, "AAPL", [(date(2026, 7, 13), 318.0),
                          (date(2026, 7, 15), 327.5),   # fires here
                          (date(2026, 7, 16), 333.0)])
    out = sweep_commitments(store, today="2026-07-17")
    assert out["fired"] == 1
    c = store.select("desk_commitments", filters={"run_id": "R1"})[0]
    assert c["status"] == "fired"
    assert str(c["fired_date"]) == "2026-07-15" and c["fired_close"] == 327.5


def test_sweep_expires_a_deadline_that_passed_without_a_breach(store):
    from agent.ledger import sweep_commitments

    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "ts": datetime(2026, 7, 10, 15, 0),
        "picks": [{"symbol": "AAPL", "action": "trim"}]}, returning=False)
    store.insert("desk_commitments", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "kind": "reentry", "direction": "above", "level": 325.0,
        "until": date(2026, 7, 16), "text": "reclaims 325",
        "status": "open", "created_at": datetime(2026, 7, 10, 15, 0)},
        returning=False)
    _bars(store, "AAPL", [(date(2026, 7, 13), 318.0),
                          (date(2026, 7, 15), 320.0)])   # never reclaims
    out = sweep_commitments(store, today="2026-07-17")
    assert out["fired"] == 0 and out["expired"] == 1
    assert store.select("desk_commitments",
                        filters={"run_id": "R1"})[0]["status"] == "expired"


def test_grade_runs_the_sweep(store):
    from agent.ledger import grade

    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "ts": datetime(2026, 7, 10, 15, 0),
        "picks": [{"symbol": "AAPL", "action": "trim"}]}, returning=False)
    store.insert("desk_commitments", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "kind": "reentry", "direction": "above", "level": 325.0,
        "until": date(2026, 7, 20), "text": "reclaims 325",
        "status": "open", "created_at": datetime(2026, 7, 10, 15, 0)},
        returning=False)
    _bars(store, "AAPL", [(date(2026, 7, 15), 327.5)])
    out = grade(store)
    assert out["commitments"]["fired"] == 1


# ── context surfacing + honor ────────────────────────────────────────────


def test_fired_unhonored_surfaces_in_context_then_clears_on_honor(store):
    from agent.brain import context
    from agent.knowledge import commitment_honor

    store.insert("desk_commitments", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "kind": "reentry", "direction": "above", "level": 325.0,
        "until": date(2026, 7, 20), "text": "reclaims 325",
        "status": "fired", "fired_date": date(2026, 7, 15),
        "fired_close": 327.5, "created_at": datetime(2026, 7, 10, 15, 0)},
        returning=False)
    ctx = context(store)
    assert "commitments" not in ctx["errors"]
    fired = ctx["commitments"]["fired_unhonored"]
    assert len(fired) == 1 and fired[0]["symbol"] == "AAPL"

    # honoring it — even "standing down" — clears the obligation
    cid = fired[0]["id"]
    commitment_honor(store, commitment_id=cid, run_id="R2",
                     note="standing down: chip-sector risk-off, not re-adding")
    row = store.select("desk_commitments", filters={"id": cid})[0]
    assert row["status"] == "honored" and row["honored_run_id"] == "R2"
    ctx2 = context(store)
    assert ctx2["commitments"]["fired_unhonored"] == []
