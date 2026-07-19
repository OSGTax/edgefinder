"""agent.knowledge behavior tests: claim validation, the pre-registered
promotion gate (recomputed stats, tighten-only criteria, regime scoping),
supersession/quarantine, the tier-gated context read, and basic lint."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'k.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


SCOPE = {"account": "paper", "regimes": ["risk_on", "neutral"]}


def _seed_graded_pick(store, run_id, symbol, *, regime="risk_on",
                      ts=None, alpha=1.0, status="open", verdict=None):
    """One decision + its graded outcome — the evidence substrate."""
    store.insert("desk_decisions", {
        "account": "agent", "run_id": run_id, "regime": regime,
        "ts": ts or datetime(2026, 7, 7, 15, 0),
        "picks": [{"symbol": symbol, "action": "buy",
                   "prediction": "x", "horizon_days": 10, "kill": "y"}],
    }, returning=False)
    store.insert("desk_outcomes", {
        "account": "agent", "run_id": run_id, "symbol": symbol,
        "grade_date": date(2026, 7, 17), "alpha_pct": alpha,
        "status": status, "verdict": verdict,
    }, returning=False)
    return {"kind": "outcome", "run_id": run_id, "symbol": symbol}


# ── claim_add validation ─────────────────────────────────────────────────


def test_add_rejects_bad_shapes(store):
    from agent.knowledge import STATEMENT_MAX_CHARS, claim_add

    assert not claim_add(store, kclass="vibes", tier="candidate",
                         statement="s", scope=SCOPE)["ok"]
    assert not claim_add(store, kclass="market_strategy", tier="candidate",
                         statement="x" * (STATEMENT_MAX_CHARS + 1),
                         scope=SCOPE)["ok"]
    # scope.account mandatory
    r = claim_add(store, kclass="operational", tier="observation",
                  statement="s", scope={})
    assert not r["ok"] and "scope.account" in r["error"]
    # market/risk claims need evidence
    r = claim_add(store, kclass="market_strategy", tier="candidate",
                  statement="s", scope=SCOPE, promotion_criteria={"min_n": 5})
    assert not r["ok"] and "hunch" in r["error"]
    # a candidate must pre-register criteria
    r = claim_add(store, kclass="market_strategy", tier="candidate",
                  statement="s", scope=SCOPE,
                  evidence=[{"kind": "probe", "note": "n"}])
    assert not r["ok"] and "promotion_criteria" in r["error"]
    # market_strategy cannot be born established
    r = claim_add(store, kclass="market_strategy", tier="established",
                  statement="s", scope=SCOPE,
                  evidence=[{"kind": "probe", "note": "n"}])
    assert not r["ok"] and "born established" in r["error"]


def test_risk_rule_decay_is_forced_never(store):
    from agent.knowledge import claim_add

    r = claim_add(store, kclass="risk_rule", tier="established",
                  statement="A fired kill with a stated remediation needs a "
                            "same-cycle decision.",
                  scope={"account": "paper"},
                  evidence=[{"kind": "probe", "note": "AAPL ~$500", "date":
                             "2026-07-17"}],
                  decay_class="regime_conditional")  # ignored on purpose
    assert r["ok"] and r["decay_class"] == "never" and r["expires_at"] is None


def test_regime_conditional_gets_default_expiry(store):
    from agent.knowledge import claim_add

    r = claim_add(store, kclass="market_strategy", tier="candidate",
                  statement="s", scope=SCOPE,
                  evidence=[{"kind": "probe", "note": "n"}],
                  promotion_criteria={"min_n": 5})
    assert r["ok"] and r["decay_class"] == "regime_conditional"
    assert r["expires_at"] is not None


def test_supersedes_flips_old_claim(store):
    from agent.knowledge import claim_add, get_claim

    a = claim_add(store, kclass="operational", tier="observation",
                  statement="v1", scope={"account": "paper"})
    b = claim_add(store, kclass="operational", tier="observation",
                  statement="v2 sharper", scope={"account": "paper"},
                  supersedes=a["id"])
    old = get_claim(store, claim_id=a["id"])["claim"]
    assert old["status"] == "superseded" and old["superseded_by"] == b["id"]
    events = [e["event"] for e in get_claim(store, claim_id=a["id"])["events"]]
    assert "superseded" in events


# ── the promotion gate ───────────────────────────────────────────────────


def _candidate(store, evidence, criteria=None, scope=None):
    from agent.knowledge import claim_add

    r = claim_add(store, kclass="market_strategy", tier="candidate",
                  statement="Momentum entries underperform their trailing "
                            "signal in week one.",
                  scope=scope or SCOPE, evidence=evidence,
                  promotion_criteria=criteria or {"min_n": 5})
    assert r["ok"]
    return r["id"]


def test_promote_refused_small_n_and_missing_criteria(store):
    from agent.knowledge import claim_promote

    refs = [_seed_graded_pick(store, "R1", "IWM"),
            _seed_graded_pick(store, "R2", "LLY")]
    cid = _candidate(store, refs)
    out = claim_promote(store, claim_id=cid)
    assert not out["ok"]
    assert any("min_n" in f for f in out["failures"])
    # refusal is auditable
    ev = store.select("desk_claim_events", filters={"claim_id": cid})
    assert any(e["event"] == "demoted" for e in ev)


def test_promote_refused_on_unjudged_closed_outcome(store):
    from agent.knowledge import claim_promote

    refs = [_seed_graded_pick(store, f"R{i}", s, alpha=2.0,
                              regime=("risk_on" if i % 2 else "neutral"),
                              ts=datetime(2026, 6, 1, 15, 0) + timedelta(days=i * 7))
            for i, s in enumerate(["IWM", "LLY", "NVDA", "AVGO"])]
    refs.append(_seed_graded_pick(store, "R9", "DDOG", alpha=2.0,
                                  status="closed", verdict=None,
                                  ts=datetime(2026, 7, 10, 15, 0)))
    cid = _candidate(store, refs)
    out = claim_promote(store, claim_id=cid)
    assert not out["ok"]
    assert any("no verdict" in f for f in out["failures"])


def test_promote_passes_with_pre_registered_gate_met(store):
    from agent.knowledge import claim_promote, get_claim

    syms = ["IWM", "LLY", "NVDA", "AVGO", "DDOG"]
    refs = [_seed_graded_pick(
        store, f"R{i}", s,
        alpha=(2.0 if i else -1.0),  # 4 wins, 1 loss = 0.8 win rate
        regime=("risk_on" if i % 2 else "neutral"),
        ts=datetime(2026, 6, 1, 15, 0) + timedelta(days=i * 10))
        for i, s in enumerate(syms)]
    cid = _candidate(store, refs)
    out = claim_promote(store, claim_id=cid, run_id="RP")
    assert out["ok"], out
    got = get_claim(store, claim_id=cid)["claim"]
    assert got["tier"] == "established"
    assert got["stats"]["n"] == 5 and got["stats"]["wins"] == 4
    ev = [e for e in get_claim(store, claim_id=cid)["events"]
          if e["event"] == "promoted"]
    assert ev and ev[0]["detail"]["effective_criteria"]["min_n"] == 5


def test_registered_criteria_tighten_but_never_loosen(store):
    from agent.knowledge import claim_promote

    refs = [_seed_graded_pick(
        store, f"R{i}", s, alpha=2.0,
        regime=("risk_on" if i % 2 else "neutral"),
        ts=datetime(2026, 6, 1, 15, 0) + timedelta(days=i * 10))
        for i, s in enumerate(["IWM", "LLY", "NVDA", "AVGO", "DDOG"])]
    # author tries to weaken the gate to n=2 — defaults win
    cid = _candidate(store, refs, criteria={"min_n": 2})
    out = claim_promote(store, claim_id=cid)
    assert out["ok"]
    assert out["effective_criteria"]["min_n"] == 5


def test_cautionary_claims_win_rate_scores_negative_alpha(store):
    from agent.knowledge import claim_promote

    # 4 of 5 instances UNDERPERFORMED — which is what this pattern predicts
    refs = [_seed_graded_pick(
        store, f"R{i}", s, alpha=(-2.0 if i else 1.0),
        regime=("risk_on" if i % 2 else "neutral"),
        ts=datetime(2026, 6, 1, 15, 0) + timedelta(days=i * 10))
        for i, s in enumerate(["IWM", "LLY", "NVDA", "AVGO", "DDOG"])]
    # without win_is, 1/5 positive alpha would refuse promotion
    cid = _candidate(store, refs, criteria={"min_n": 5,
                                            "win_is": "negative_alpha"})
    out = claim_promote(store, claim_id=cid)
    assert out["ok"], out
    assert out["effective_criteria"]["win_is"] == "negative_alpha"


def test_single_regime_scope_establishes_but_forces_expiry(store):
    from agent.knowledge import claim_promote, get_claim

    refs = [_seed_graded_pick(
        store, f"R{i}", s, alpha=2.0, regime="risk_on",
        ts=datetime(2026, 6, 1, 15, 0) + timedelta(days=i * 10))
        for i, s in enumerate(["IWM", "LLY", "NVDA", "AVGO", "DDOG"])]
    cid = _candidate(store, refs,
                     scope={"account": "paper", "regimes": ["risk_on"]})
    out = claim_promote(store, claim_id=cid)
    assert out["ok"], out
    got = get_claim(store, claim_id=cid)["claim"]
    assert got["decay_class"] == "regime_conditional"
    assert got["expires_at"]  # dies unless renewed


def test_promote_refused_when_evidence_spans_outside_scoped_regime(store):
    from agent.knowledge import claim_promote

    refs = [_seed_graded_pick(
        store, f"R{i}", s, alpha=2.0,
        regime=("neutral" if i == 0 else "risk_on"),
        ts=datetime(2026, 6, 1, 15, 0) + timedelta(days=i * 10))
        for i, s in enumerate(["IWM", "LLY", "NVDA", "AVGO", "DDOG"])]
    cid = _candidate(store, refs,
                     scope={"account": "paper", "regimes": ["risk_on"]})
    out = claim_promote(store, claim_id=cid)
    assert not out["ok"]
    assert any("widen the scope" in f for f in out["failures"])


# ── transitions + tier-gated context ─────────────────────────────────────


def test_risk_rule_retire_is_proposal_gated(store):
    from agent.knowledge import claim_add, claim_retire, proposal_add

    r = claim_add(store, kclass="risk_rule", tier="established",
                  statement="s", scope={"account": "paper"},
                  evidence=[{"kind": "probe", "note": "loss"}])
    out = claim_retire(store, claim_id=r["id"], reason="loosen it")
    assert not out["ok"] and "tighten-only" in out["error"]
    # an unapproved proposal doesn't unlock it either
    p = proposal_add(store, title="Retire rule", body="why",
                     change_kind="rules", claim_ids=[r["id"]])
    out = claim_retire(store, claim_id=r["id"], reason="loosen",
                       proposal_id=p["id"])
    assert not out["ok"] and "not approved" in out["error"]


def test_context_claims_is_tier_gated_and_bounded(store):
    from agent.knowledge import claim_add, context_claims

    claim_add(store, kclass="risk_rule", tier="established", statement="est",
              scope={"account": "paper"},
              evidence=[{"kind": "probe", "note": "n"}])
    claim_add(store, kclass="market_strategy", tier="candidate",
              statement="watching", scope=SCOPE,
              evidence=[{"kind": "probe", "note": "n"}],
              promotion_criteria={"min_n": 5})
    claim_add(store, kclass="market_strategy", tier="candidate",
              statement="experimental bet", scope=SCOPE,
              evidence=[{"kind": "probe", "note": "n"}],
              promotion_criteria={"min_n": 5}, experimental=True)
    out = context_claims(store)
    statements = {c["statement"] for c in out["claims"]}
    assert statements == {"est", "experimental bet"}  # candidate stays out
    assert out["candidates_watching"] == 1
    assert out["caps"]["experimental_total_weight"] == 0.10


def test_brain_context_carries_claims_section(store):
    from agent.brain import context
    from agent.knowledge import claim_add

    claim_add(store, kclass="risk_rule", tier="established", statement="est",
              scope={"account": "paper"},
              evidence=[{"kind": "probe", "note": "n"}])
    ctx = context(store)
    assert "claims" not in ctx["errors"]
    assert ctx["claims"]["count"] == 1
    assert ctx["claims"]["claims"][0]["cite"] == "[C-1]"


# ── lint ─────────────────────────────────────────────────────────────────


def test_lint_catches_the_step2_failure_modes(store):
    from agent.brain import set_wiki
    from agent.knowledge import claim_add, lint

    # orphaned evidence ref
    claim_add(store, kclass="market_strategy", tier="candidate",
              statement="s", scope=SCOPE,
              evidence=[{"kind": "outcome", "run_id": "NOPE", "symbol": "X"}],
              promotion_criteria={"min_n": 5})
    # expired-but-active regime claim
    claim_add(store, kclass="market_strategy", tier="candidate",
              statement="old", scope=SCOPE,
              evidence=[{"kind": "probe", "note": "n"}],
              promotion_criteria={"min_n": 5}, expires_at=date(2026, 1, 1))
    # wiki citing a claim that doesn't exist
    set_wiki(store, slug="lessons", body="Momentum lesson [C-99].",
             reason="test", run_id="R")
    # closed outcome with no verdict
    store.insert("desk_outcomes", {
        "account": "agent", "run_id": "RX", "symbol": "GEV",
        "grade_date": date(2026, 7, 17), "status": "closed",
    }, returning=False)

    out = lint(store)
    assert not out["ok"]
    text = "\n".join(out["errors"] + out["warnings"])
    assert "unresolvable" in text
    assert "expired" in text
    assert "[C-99]" in text
    assert "unjudged" in text


def test_search_supports_false_absence_checks(store):
    from agent.knowledge import claim_add, search_claims

    claim_add(store, kclass="operational", tier="observation",
              statement="Streamer sweep lagged a fast pre-market gap.",
              scope={"account": "paper"})
    assert search_claims(store, terms="websocket")["count"] == 0
    assert search_claims(store, terms="sweep gap")["count"] == 1  # 2nd phrasing
