"""Tier authority + experimental caps (SCHEMA.md §5, PLAN.md step 5): a pick
may only cite claims allowed to justify a decision (established, or
experimental under exposure caps); candidates stay watch-only."""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def _established(store, **over):
    from agent.knowledge import claim_add
    kw = dict(kclass="risk_rule", tier="established", statement="est",
              scope={"account": "paper"},
              evidence=[{"kind": "probe", "note": "n"}])
    kw.update(over)
    r = claim_add(store, **kw)
    assert r["ok"], r
    return r["id"]


def _candidate(store, *, experimental=False):
    from agent.knowledge import claim_add
    r = claim_add(store, kclass="market_strategy", tier="candidate",
                  statement="watching", scope={"account": "paper",
                                               "regimes": ["risk_on"]},
                  evidence=[{"kind": "probe", "note": "n"}],
                  promotion_criteria={"min_n": 5}, experimental=experimental)
    assert r["ok"], r
    return r["id"]


def _buy(sym, claims, **extra):
    p = {"symbol": sym, "action": "buy", "prediction": "up", "horizon_days": 10,
         "kill": "closes below x", "claims": claims}
    p.update(extra)
    return p


def test_citing_established_claim_is_allowed(store):
    from agent.brain import save_decision
    cid = _established(store)
    r = save_decision(store, run_id="R1",
                      target_weights={"NVDA": 0.2},
                      picks=[_buy("NVDA", [cid])])
    assert r["ok"], r


def test_citing_a_candidate_is_rejected(store):
    from agent.brain import save_decision
    cid = _candidate(store)  # not experimental
    r = save_decision(store, run_id="R1", target_weights={"NVDA": 0.2},
                      picks=[_buy("NVDA", [cid])])
    assert not r["ok"]
    assert "watch-only" in r["error"] and f"claim {cid}" in r["error"]
    assert store.select("desk_decisions", filters={"run_id": "R1"}) == []


def test_citing_a_nonexistent_or_retired_claim_is_rejected(store):
    from agent.brain import save_decision
    from agent.knowledge import claim_retire
    cid = _established(store, kclass="operational")
    claim_retire(store, claim_id=cid, reason="stale")
    r = save_decision(store, run_id="R1", target_weights={"NVDA": 0.2},
                      picks=[_buy("NVDA", [cid])])
    assert not r["ok"] and "retired" in r["error"]
    r2 = save_decision(store, run_id="R2", target_weights={"NVDA": 0.2},
                       picks=[_buy("NVDA", [9999])])
    assert not r2["ok"] and "no such claim" in r2["error"]


def test_experimental_candidate_may_justify_within_caps(store):
    from agent.brain import save_decision
    cid = _candidate(store, experimental=True)
    r = save_decision(store, run_id="R1", target_weights={"DDOG": 0.05},
                      picks=[_buy("DDOG", [cid])])
    assert r["ok"], r


def test_experimental_per_claim_cap_breach_is_rejected(store):
    from agent.brain import save_decision
    cid = _candidate(store, experimental=True)
    # one name at 8% exceeds the 5% per-claim experimental cap
    r = save_decision(store, run_id="R1", target_weights={"DDOG": 0.08},
                      picks=[_buy("DDOG", [cid])])
    assert not r["ok"] and "per-claim cap" in r["error"]


def test_experimental_total_cap_breach_is_rejected(store):
    from agent.brain import save_decision
    a, b, c = (_candidate(store, experimental=True) for _ in range(3))
    # three names each within the 5% per-claim cap but summing over 10% total
    r = save_decision(store, run_id="R1",
                      target_weights={"AA": 0.04, "BB": 0.04, "CC": 0.04},
                      picks=[_buy("AA", [a]), _buy("BB", [b]), _buy("CC", [c])])
    assert not r["ok"] and "total" in r["error"]


def test_claims_must_be_a_list_of_ints(store):
    from agent.brain import save_decision
    r = save_decision(store, run_id="R1", target_weights={"NVDA": 0.2},
                      picks=[_buy("NVDA", ["C-3"])])
    assert not r["ok"] and "list of integer claim ids" in r["error"]


def test_uncited_picks_are_unaffected(store):
    """The gate only bites when a pick actually cites a claim."""
    from agent.brain import save_decision
    r = save_decision(store, run_id="R1", target_weights={"NVDA": 0.2},
                      picks=[{"symbol": "NVDA", "action": "buy",
                              "prediction": "up", "horizon_days": 10,
                              "kill": "x"}])
    assert r["ok"], r


# ── context truncation must not starve new or capped claims ──────────────

def test_context_truncation_is_prioritised_not_oldest_first(store):
    """The cap bounds prompt growth, but slicing an id-ASCENDING list meant
    the oldest 20 claims filled every slot and everything registered since
    was invisible — a newly written claim could never influence a cycle,
    which defeats the write-then-read loop the registry exists for."""
    from agent import knowledge

    # three risk rules, then enough established claims to overflow the cap
    risk = [_established(store, statement=f"risk {i}") for i in range(3)]
    for i in range(knowledge.CONTEXT_MAX_CLAIMS + 5):
        _established(store, kclass="operational", statement=f"op {i}")
    # the claim under test, registered LAST and expiring SOONEST
    newest = _candidate(store, experimental=True)

    cc = knowledge.context_claims(store)
    ids = [c["id"] for c in cc["claims"]]

    assert cc["count"] == knowledge.CONTEXT_MAX_CLAIMS
    assert cc["truncated"] == cc["injectable_total"] - cc["count"] > 0
    assert newest in ids, "the newest experimental claim must survive the cap"
    for r in risk:
        assert r in ids, "risk rules must never be starved out"


def test_soonest_expiring_experimental_wins_its_band(store):
    """Within the experimental band the one about to lapse is the urgent
    one — ordering that band by id re-created the starvation one level down."""
    from agent.knowledge import claim_add, context_claims, CONTEXT_MAX_CLAIMS

    def _exp(stmt, expires):
        r = claim_add(store, kclass="market_strategy", tier="candidate",
                      statement=stmt, scope={"account": "paper"},
                      evidence=[{"kind": "probe", "note": "n"}],
                      promotion_criteria={"min_n": 5}, experimental=True,
                      expires_at=expires)
        assert r["ok"], r
        return r["id"]

    # fill the cap with far-dated experimental claims first
    for i in range(CONTEXT_MAX_CLAIMS + 3):
        _exp(f"far {i}", "2099-01-01")
    urgent = _exp("expires tomorrow", "2026-07-30")

    ids = [c["id"] for c in context_claims(store)["claims"]]
    assert urgent in ids and ids[0] == urgent
