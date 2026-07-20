"""Owner-approval gate (SCHEMA.md §6, PLAN.md step 6): the set_state gate on
pivots + cap-raises, the proposal lifecycle, and the GitHub approval sync
(owner-authorship only)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'p.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


CAPS = {"concentration_gate_pct": 30, "starter_position_pct": 8}


def _seed_state(store):
    from agent.brain import set_state
    # first state creation is ungated (no prior to change)
    r = set_state(store, name="v1", thesis="t", params=dict(CAPS))
    assert r["ok"]


# ── the set_state gate ───────────────────────────────────────────────────


def test_first_state_creation_is_ungated(store):
    from agent.brain import set_state
    assert set_state(store, name="v1", params=dict(CAPS))["ok"]


def test_pivot_without_authorization_is_rejected(store):
    from agent.brain import set_state
    _seed_state(store)
    r = set_state(store, name="v2", params=dict(CAPS), bump=True)
    assert not r["ok"] and "owner approval" in r["error"]
    # not written
    assert store.select("desk_strategy_state",
                        filters={"version": 2}) == []


def test_pivot_with_no_learned_basis_is_allowed_and_journaled(store):
    from agent.brain import set_state
    _seed_state(store)
    r = set_state(store, name="v2", params=dict(CAPS), bump=True,
                  run_id="R1", no_learned_basis="owner-directed 2026-07-20")
    assert r["ok"] and r["authorized_by"] == "no-learned-basis"
    j = store.select("desk_journal", filters={"kind": "ungated-change"})
    assert len(j) == 1 and "owner-directed" in j[0]["body"]


def test_raising_a_cap_is_gated(store):
    from agent.brain import set_state
    _seed_state(store)
    # in-place tweak that RAISES concentration 30 -> 35
    r = set_state(store, name="v1", params={**CAPS,
                  "concentration_gate_pct": 35})
    assert not r["ok"] and "concentration_gate_pct" in r["error"]


def test_tightening_a_cap_is_free(store):
    from agent.brain import set_state
    _seed_state(store)
    r = set_state(store, name="v1", params={**CAPS,
                  "concentration_gate_pct": 25})
    assert r["ok"] and "gated_change" not in r


def test_study_log_only_tweak_is_free(store):
    from agent.brain import set_state
    _seed_state(store)
    r = set_state(store, name="v1", params={**CAPS,
                  "study_log": [{"date": "2026-07-20", "slice": "x"}]})
    assert r["ok"] and "gated_change" not in r


def test_approved_proposal_authorizes_and_is_marked_applied(store):
    from agent.brain import set_state
    from agent.knowledge import proposal_add, proposal_approve
    _seed_state(store)
    p = proposal_add(store, title="Raise concentration to 35",
                     body="AVGO/NVDA both backtested >20%",
                     change_kind="caps", claim_ids=[])
    proposal_approve(store, proposal_id=p["id"])  # owner action
    r = set_state(store, name="v1", params={**CAPS,
                  "concentration_gate_pct": 35}, run_id="R9",
                  proposal_id=p["id"])
    assert r["ok"] and r["authorized_by"] == f"proposal {p['id']}"
    # one approval, one application
    from agent.knowledge import proposal_get
    pr = proposal_get(store, proposal_id=p["id"])
    assert pr["status"] == "applied" and pr["applied_run_id"] == "R9"


def test_unapproved_proposal_does_not_authorize(store):
    from agent.brain import set_state
    from agent.knowledge import proposal_add
    _seed_state(store)
    p = proposal_add(store, title="raise", body="b", change_kind="caps")
    r = set_state(store, name="v2", bump=True, proposal_id=p["id"])
    assert not r["ok"] and "not approved" in r["error"]


# ── proposal lifecycle ───────────────────────────────────────────────────


def test_proposal_add_requires_plain_text_and_valid_kind(store):
    from agent.knowledge import proposal_add
    assert not proposal_add(store, title="", body="b", change_kind="caps")["ok"]
    assert not proposal_add(store, title="t", body="b",
                            change_kind="whatever")["ok"]


def test_applied_proposal_cannot_be_reapplied(store):
    from agent.knowledge import (proposal_add, proposal_approve,
                                 proposal_mark_applied)
    p = proposal_add(store, title="t", body="b", change_kind="rules")
    proposal_approve(store, proposal_id=p["id"])
    assert proposal_mark_applied(store, proposal_id=p["id"], run_id="R1")["ok"]
    # second application refused — one approval, one use
    r = proposal_mark_applied(store, proposal_id=p["id"], run_id="R2")
    assert not r["ok"] and "not approved" in r["error"]


# ── GitHub approval sync (owner authorship only) ─────────────────────────


def _fetcher(comments=None, labels=None):
    def f(title):
        return {"comments": comments or [], "labels": labels or []}
    return f


def test_sync_approves_only_on_owner_comment(store, monkeypatch):
    from agent.knowledge import proposal_add, proposal_sync
    monkeypatch.setattr("config.settings.settings.github_owner_login", "OSGTax")
    p = proposal_add(store, title="t", body="b", change_kind="caps")

    # a non-owner "approve" does NOT approve
    r = proposal_sync(store, proposal_id=p["id"],
                      fetch=_fetcher(comments=[{"user": "randobot",
                                               "body": "approve"}]))
    assert r["status"] == "pending"

    # the agent's own token identity cannot self-approve
    r = proposal_sync(store, proposal_id=p["id"],
                      fetch=_fetcher(comments=[{"user": "github-actions[bot]",
                                               "body": "approved"}]))
    assert r["status"] == "pending"

    # the owner's approve DOES
    r = proposal_sync(store, proposal_id=p["id"],
                      fetch=_fetcher(comments=[{"user": "osgtax",
                                               "body": "approve — ship it"}]))
    assert r["ok"] and r["status"] == "approved" and r["decided_via"] == "github"


def test_sync_honors_owner_label_and_reject(store, monkeypatch):
    from agent.knowledge import proposal_add, proposal_sync
    monkeypatch.setattr("config.settings.settings.github_owner_login", "OSGTax")

    p = proposal_add(store, title="t", body="b", change_kind="caps")
    r = proposal_sync(store, proposal_id=p["id"],
                      fetch=_fetcher(labels=[{"name": "approved",
                                             "user": "OSGTax"}]))
    assert r["status"] == "approved"

    p2 = proposal_add(store, title="t2", body="b", change_kind="caps")
    r = proposal_sync(store, proposal_id=p2["id"],
                      fetch=_fetcher(comments=[{"user": "OSGTax",
                                               "body": "reject — too risky"}]))
    assert r["status"] == "rejected"


def test_sync_no_issue_stays_pending(store, monkeypatch):
    from agent.knowledge import proposal_add, proposal_sync
    monkeypatch.setattr("config.settings.settings.github_owner_login", "OSGTax")
    p = proposal_add(store, title="t", body="b", change_kind="caps")
    r = proposal_sync(store, proposal_id=p["id"], fetch=lambda title: None)
    assert r["status"] == "pending"
