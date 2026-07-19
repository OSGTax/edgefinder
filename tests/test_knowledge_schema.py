"""Knowledge-layer schema tests (SCHEMA.md step 1): the four tables exist,
round-trip through the store seam on SQLite, and reach prod via DESK_TABLE_DDL.

Behavioral gates (promotion, commitments, proposals) arrive with
``agent.knowledge`` in a later step — these tests pin the schema only.
"""

from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'knowledge.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def test_claim_roundtrip_with_json_fields(store):
    row = store.insert("desk_claims", {
        "account": "agent",
        "kclass": "market_strategy",
        "tier": "candidate",
        "statement": "Selection momentum diverges from since-entry mark on fresh picks.",
        "scope": {"account": "paper", "regimes": ["risk_on"]},
        "evidence": [{"kind": "outcome", "run_id": "R1", "symbol": "IWM"},
                     {"kind": "outcome", "run_id": "R2", "symbol": "LLY"}],
        "stats": {"n": 2, "symbols": ["IWM", "LLY"],
                  "span": ["2026-07-07", "2026-07-10"]},
        "promotion_criteria": {"min_n": 5, "min_symbols": 3, "min_regimes": 2},
        "decay_class": "regime_conditional",
    })[0]
    assert row["id"] and row["status"] == "active"
    assert row["experimental"] in (False, 0)  # SQLite may hand back 0

    got = store.select("desk_claims", filters={"id": row["id"]})[0]
    assert got["scope"]["account"] == "paper"
    assert got["stats"]["n"] == 2
    assert {e["kind"] for e in got["evidence"]} == {"outcome"}
    assert got["promotion_criteria"]["min_n"] == 5

    # supersession is a status flip + link, never a delete
    store.insert("desk_claims", {
        "kclass": "market_strategy", "tier": "candidate",
        "statement": "Sharper restatement.", "decay_class": "regime_conditional",
        "supersedes": row["id"],
    })
    store.update("desk_claims", {"id": row["id"]},
                 {"status": "superseded", "superseded_by": row["id"] + 1},
                 returning=False)
    old = store.select("desk_claims", filters={"id": row["id"]})[0]
    assert old["status"] == "superseded" and old["superseded_by"] == row["id"] + 1


def test_claim_events_append(store):
    ev = store.insert("desk_claim_events", {
        "claim_id": 1, "run_id": "R1", "event": "created",
        "detail": {"tier": "candidate"},
    })[0]
    assert ev["id"] and ev["event"] == "created"
    got = store.select("desk_claim_events", filters={"claim_id": 1})
    assert len(got) == 1 and got[0]["detail"] == {"tier": "candidate"}


def test_commitment_lifecycle_columns(store):
    c = store.insert("desk_commitments", {
        "run_id": "2026-07-10T15:00", "symbol": "AAPL", "kind": "reentry",
        "direction": "above", "level": 325.0, "until": date(2026, 7, 17),
        "text": "AAPL rallies through $325 with no pullback in 5 sessions - re-add.",
    })[0]
    assert c["status"] == "open" and c["watch_id"] is None
    store.update("desk_commitments", {"id": c["id"]},
                 {"status": "fired", "fired_date": date(2026, 7, 15),
                  "fired_close": 327.5}, returning=False)
    got = store.select("desk_commitments", filters={"status": "fired"})
    assert len(got) == 1 and got[0]["fired_close"] == 327.5
    assert got[0]["honored_run_id"] is None  # unhonored until a decision stamps it


def test_proposal_defaults_and_decision_fields(store):
    p = store.insert("desk_proposals", {
        "title": "Raise single-name cap to 25%", "body": "why...",
        "claim_ids": [3, 7], "change_kind": "caps",
        "payload": {"params": {"max_single_name": 0.25}},
        "expires_at": date(2026, 8, 2),
    })[0]
    assert p["status"] == "pending" and p["decided_by"] is None
    store.update("desk_proposals", {"id": p["id"]},
                 {"status": "approved", "decided_by": "owner",
                  "decided_via": "github"}, returning=False)
    got = store.select("desk_proposals", filters={"id": p["id"]})[0]
    assert got["status"] == "approved" and got["decided_via"] == "github"
    assert got["claim_ids"] == [3, 7]


def test_ddl_covers_all_four_tables():
    """The idempotent DDL list is the ONLY path to prod - every new table and
    its RLS lockdown must be present."""
    from agent.models import DESK_TABLE_DDL
    ddl = "\n".join(DESK_TABLE_DDL)
    for table in ("desk_claims", "desk_claim_events",
                  "desk_commitments", "desk_proposals"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in ddl
