"""Step 7: the extended lint checks and the loop-honesty report."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'l.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def test_lint_clean_on_a_well_formed_base(store):
    from agent.knowledge import claim_add, lint
    claim_add(store, kclass="risk_rule", tier="established",
              statement="honor fired kills same-cycle", scope={"account": "paper"},
              evidence=[{"kind": "probe", "note": "AAPL"}])
    out = lint(store)
    assert out["ok"] and out["errors"] == []


def test_lint_flags_prose_number_drift(store):
    from agent.brain import set_wiki
    from agent.knowledge import claim_add, lint
    cid = claim_add(store, kclass="market_strategy", tier="candidate",
                    statement="s", scope={"account": "paper",
                                          "regimes": ["risk_on"]},
                    evidence=[{"kind": "probe", "note": "n"}],
                    promotion_criteria={"min_n": 5},
                    stats={"n": 2})["id"]
    set_wiki(store, slug="lessons",
             body=f"Momentum divergence [C-{cid}] (n=4 across names).",
             reason="t", run_id="R")
    out = lint(store)
    assert any("n=4 but the claim's stored n=2" in w for w in out["warnings"])


def test_lint_flags_tier_violation_in_a_saved_decision(store):
    """A citation that slipped in before a demotion is caught belt-and-braces."""
    from agent.knowledge import claim_add, lint
    cid = claim_add(store, kclass="market_strategy", tier="candidate",
                    statement="s", scope={"account": "paper",
                                          "regimes": ["risk_on"]},
                    evidence=[{"kind": "probe", "note": "n"}],
                    promotion_criteria={"min_n": 5})["id"]
    # write the decision directly (bypassing the save gate) to simulate drift
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R1", "ts": datetime(2026, 7, 20, 15, 0),
        "picks": [{"symbol": "NVDA", "action": "buy", "claims": [cid]}]},
        returning=False)
    out = lint(store)
    assert any("only established/experimental" in e for e in out["errors"])


def test_lint_flags_unstructured_clause_and_unhonored_commitment(store):
    from agent.knowledge import lint
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "R1", "ts": datetime(2026, 7, 20, 15, 0),
        "picks": [{"symbol": "AAPL", "action": "trim",
                   "rationale": "trim; re-add if it reclaims $325"}]},
        returning=False)
    store.insert("desk_commitments", {
        "account": "agent", "run_id": "R0", "symbol": "MSFT", "kind": "reentry",
        "direction": "above", "level": 500.0, "text": "x", "status": "fired",
        "fired_date": date(2026, 7, 18), "created_at": datetime(2026, 7, 10)},
        returning=False)
    out = lint(store)
    txt = "\n".join(out["warnings"])
    assert "conditional clause" in txt and "unhonored" in txt


def test_lint_flags_hindsight_and_hygiene(store):
    from agent.knowledge import claim_add, lint
    # a claim created BEFORE the decision it cites → hindsight risk
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "RF", "ts": datetime(2026, 8, 1, 15, 0),
        "regime": "risk_on",
        "picks": [{"symbol": "X", "action": "buy"}]}, returning=False)
    store.insert("desk_outcomes", {
        "account": "agent", "run_id": "RF", "symbol": "X",
        "grade_date": date(2026, 8, 1), "alpha_pct": 1.0, "status": "open"},
        returning=False)
    claim_add(store, kclass="operational", tier="observation",
              statement="future-cited", scope={"account": "paper"},
              evidence=[{"kind": "outcome", "run_id": "RF", "symbol": "X"}])
    out = lint(store)
    assert any("hindsight" in w for w in out["warnings"])


# ── loop report ──────────────────────────────────────────────────────────


def test_loop_report_counts_the_loop_activity(store):
    from agent.brain import save_decision
    from agent.knowledge import claim_add, loop_report

    est = claim_add(store, kclass="risk_rule", tier="established",
                    statement="est", scope={"account": "paper"},
                    evidence=[{"kind": "probe", "note": "n"}])["id"]
    claim_add(store, kclass="market_strategy", tier="candidate",
              statement="cand", scope={"account": "paper",
                                       "regimes": ["risk_on"]},
              evidence=[{"kind": "probe", "note": "n"}],
              promotion_criteria={"min_n": 5})
    # a pick that CITES the established claim → the "read" signal
    save_decision(store, run_id="R1", target_weights={"NVDA": 0.2},
                  picks=[{"symbol": "NVDA", "action": "buy", "prediction": "up",
                          "horizon_days": 10, "kill": "x", "claims": [est]}])

    rep = loop_report(store, days=7)
    assert rep["tier_census"]["established"] == 1
    assert rep["tier_census"]["candidate"] == 1
    assert rep["read"]["picks_citing_claims"] == 1
    assert rep["written"]["by_event"]["created"] == 2
    assert "nominal" in " ".join(rep["health"]) or rep["read"][
        "picks_citing_claims"] == 1


def test_loop_report_health_flags_unread_knowledge(store):
    from agent.knowledge import claim_add, loop_report
    claim_add(store, kclass="risk_rule", tier="established", statement="est",
              scope={"account": "paper"},
              evidence=[{"kind": "probe", "note": "n"}])
    rep = loop_report(store, days=7)
    assert any("not read" in f for f in rep["health"])


def test_loop_report_degrades_on_empty_base(store):
    from agent.knowledge import loop_report
    rep = loop_report(store, days=7)
    assert rep["tier_census"] == {}
    assert any("empty" in f for f in rep["health"])
