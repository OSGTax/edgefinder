"""The knowledge layer: the structured claims registry behind the wiki.

SCHEMA.md's source of truth for every behavior-influencing fact. The wiki
stays the trader-readable narrative; its prose must cite claims by token
(``[C-<id>]``). Prose can inform — only claims can justify.

Tiers: observation → digest → candidate → established. A candidate carries
``promotion_criteria`` registered at creation, BEFORE results; promotion is
refused unless stats RECOMPUTED here from ``desk_outcomes`` (never the
claim's self-reported numbers) meet them. No confidence floats anywhere —
recorded sample sizes only. Supersession, never deletion: status flips,
``superseded_by`` links, and every transition lands in ``desk_claim_events``.

Decay classes are forced by knowledge class: risk rules never decay (and are
tighten-only — retiring one is proposal-gated), system mechanics are stable,
market/strategy claims are regime-conditional and expire unless renewed.

CLI (JSON out), mirroring agent.brain:
  python -m agent.knowledge claim-add --kclass market_strategy --tier candidate \
      --statement "..." --scope '{"account":"paper","regimes":["risk_on"]}' \
      --evidence '[{"kind":"outcome","run_id":"R1","symbol":"IWM"}]' \
      --criteria '{"min_n":5}' --run-id R
  python -m agent.knowledge claim-list --status active
  python -m agent.knowledge claim-search --terms "momentum since-entry"
  python -m agent.knowledge claim-promote --claim-id 3 --run-id R
  python -m agent.knowledge lint
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

# ── policy constants (SCHEMA.md §5 — owner adjusts here) ─────────────────

KCLASSES = ("market_strategy", "system_mechanics", "operational", "risk_rule")
TIERS = ("observation", "digest", "candidate", "established")
STATUSES = ("active", "superseded", "retired", "quarantined")
DECAY_CLASSES = ("regime_conditional", "stable", "never")

# Defaults forced by knowledge class; risk_rule may NOT be overridden.
DECAY_BY_KCLASS = {
    "market_strategy": "regime_conditional",
    "system_mechanics": "stable",
    "operational": "stable",
    "risk_rule": "never",
}

STATEMENT_MAX_CHARS = 400
CANDIDATE_MIN_N = 2  # one instance is an observation ("watching")

ESTABLISH_DEFAULTS: dict[str, dict] = {
    "market_strategy": {"min_n": 5, "min_symbols": 3, "min_regimes": 2,
                        "min_span_sessions": 20, "min_win_rate": 0.6},
    "system_mechanics": {"min_n": 1, "requires_probe_evidence": True},
    "operational": {"min_n": 1},
    "risk_rule": {"min_n": 1, "tighten_only": True},
}

EXPERIMENTAL_TOTAL_WEIGHT_CAP = 0.10   # of equity, across experimental-cited picks
EXPERIMENTAL_PER_CLAIM_WEIGHT_CAP = 0.05
REGIME_CLAIM_TTL_SESSIONS = 60         # expiry ≈ last evidence + 60 sessions
PROPOSAL_TTL_DAYS = 14

CONTEXT_MAX_CLAIMS = 20
CONTEXT_STATEMENT_CLIP = 300

EVIDENCE_KINDS = ("outcome", "decision", "trade", "backtest",
                  "wiki_history", "probe")

ACCOUNT = "agent"


def _store():
    from agent.store import get_store

    return get_store()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _today() -> date:
    return _utcnow().date()


def _as_date(v) -> date | None:
    if v is None or isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _event(store, *, claim_id: int, event: str, run_id: str | None,
           detail: dict | None, account: str = ACCOUNT) -> None:
    store.insert("desk_claim_events",
                 {"account": account, "claim_id": claim_id, "event": event,
                  "run_id": run_id, "detail": detail}, returning=False)


# ── claims: create / read / search ───────────────────────────────────────


def claim_add(store=None, *, kclass: str, tier: str, statement: str,
              scope: dict | None = None, evidence: list | None = None,
              stats: dict | None = None, promotion_criteria: dict | None = None,
              decay_class: str | None = None, expires_at=None,
              review_after=None, experimental: bool = False,
              supersedes: int | None = None, run_id: str | None = None,
              account: str = ACCOUNT) -> dict:
    """Register a claim. Validation gates (shape only — truth is on the
    author): class/tier enums, statement cap, mandatory ``scope.account``,
    evidence for market/risk claims, criteria registered WITH a candidate,
    and no market_strategy claim born established (it must earn promotion).
    ``supersedes`` links and auto-flips the replaced claim in one call."""
    store = store or _store()

    if kclass not in KCLASSES:
        return {"ok": False, "error": f"kclass must be one of {'/'.join(KCLASSES)}"}
    if tier not in TIERS:
        return {"ok": False, "error": f"tier must be one of {'/'.join(TIERS)}"}
    statement = (statement or "").strip()
    if not statement:
        return {"ok": False, "error": "statement required (one falsifiable sentence)"}
    if len(statement) > STATEMENT_MAX_CHARS:
        return {"ok": False,
                "error": f"statement over the {STATEMENT_MAX_CHARS}-char cap "
                         f"({len(statement)}) — one falsifiable sentence, not prose"}
    if not isinstance(scope, dict) or not scope.get("account"):
        return {"ok": False,
                "error": "scope.account is REQUIRED (\"paper\" today) — "
                         "paper-derived claims are their own class by construction"}
    evidence = evidence or []
    for ref in evidence:
        if not isinstance(ref, dict) or ref.get("kind") not in EVIDENCE_KINDS:
            return {"ok": False,
                    "error": f"every evidence ref needs kind ∈ {EVIDENCE_KINDS}"}
    if kclass in ("market_strategy", "risk_rule") and not evidence:
        return {"ok": False,
                "error": f"a {kclass} claim with no evidence is a hunch — "
                         "attach at least one typed ref"}
    if tier == "candidate" and not promotion_criteria:
        return {"ok": False,
                "error": "a candidate must register promotion_criteria at "
                         "creation, BEFORE results (SCHEMA.md outcome 2)"}
    if tier == "established" and kclass == "market_strategy":
        return {"ok": False,
                "error": "market_strategy claims cannot be born established — "
                         "create a candidate and earn promotion through the gate"}

    dc = DECAY_BY_KCLASS[kclass] if kclass == "risk_rule" else \
        (decay_class or DECAY_BY_KCLASS[kclass])
    if dc not in DECAY_CLASSES:
        return {"ok": False, "error": f"decay_class must be one of {DECAY_CLASSES}"}
    if kclass == "risk_rule" and dc != "never":
        dc = "never"  # code constant, not a preference
    exp = _as_date(expires_at)
    if dc == "regime_conditional" and tier in ("candidate", "established") \
            and exp is None:
        exp = _today() + timedelta(days=round(REGIME_CLAIM_TTL_SESSIONS * 7 / 5))

    row = store.insert("desk_claims", {
        "account": account, "kclass": kclass, "tier": tier,
        "experimental": bool(experimental), "status": "active",
        "statement": statement, "scope": scope, "evidence": evidence,
        "stats": stats, "promotion_criteria": promotion_criteria,
        "decay_class": dc, "expires_at": exp,
        "review_after": _as_date(review_after), "supersedes": supersedes,
        "created_run_id": run_id, "updated_run_id": run_id,
    })[0]
    _event(store, claim_id=row["id"], event="created", run_id=run_id,
           detail={"tier": tier, "kclass": kclass}, account=account)

    if supersedes:
        old = store.select("desk_claims", filters={"id": supersedes,
                                                   "account": account})
        if old:
            store.update("desk_claims", {"id": supersedes},
                         {"status": "superseded", "superseded_by": row["id"],
                          "updated_run_id": run_id}, returning=False)
            _event(store, claim_id=supersedes, event="superseded",
                   run_id=run_id, detail={"superseded_by": row["id"]},
                   account=account)
        else:
            _event(store, claim_id=row["id"], event="created", run_id=run_id,
                   detail={"warning": f"supersedes {supersedes} not found"},
                   account=account)

    return {"ok": True, "id": row["id"], "tier": tier, "kclass": kclass,
            "decay_class": dc, "expires_at": str(exp) if exp else None}


def get_claim(store=None, *, claim_id: int, account: str = ACCOUNT) -> dict:
    store = store or _store()
    rows = store.select("desk_claims", filters={"id": claim_id,
                                                "account": account})
    if not rows:
        return {"ok": False, "error": f"claim {claim_id} not found"}
    events = store.select("desk_claim_events",
                          filters={"claim_id": claim_id, "account": account},
                          order=[("id", "asc")], limit=50)
    return {"ok": True, "claim": _row_out(rows[0]),
            "events": [{k: (str(v) if hasattr(v, "isoformat") else v)
                        for k, v in e.items() if k != "account"}
                       for e in events]}


def list_claims(store=None, *, status: str | None = "active",
                tier: str | None = None, kclass: str | None = None,
                limit: int = 100, account: str = ACCOUNT) -> dict:
    store = store or _store()
    filters: dict = {"account": account}
    if status:
        filters["status"] = status
    if tier:
        filters["tier"] = tier
    if kclass:
        filters["kclass"] = kclass
    rows = store.select("desk_claims", filters=filters,
                        order=[("id", "asc")], limit=limit)
    return {"ok": True, "count": len(rows), "claims": [_row_out(r) for r in rows]}


def search_claims(store=None, *, terms: str, include_inactive: bool = True,
                  account: str = ACCOUNT) -> dict:
    """Substring search over statement + scope + evidence text — the tool the
    false-absence rule leans on (search TWICE with alternate phrasings before
    claiming something isn't recorded). Corpus is tiny; Python-side scan keeps
    both DB transports untouched."""
    store = store or _store()
    rows = store.select("desk_claims", filters={"account": account},
                        order=[("id", "asc")], limit=500)
    needles = [t.strip().lower() for t in (terms or "").split() if t.strip()]
    if not needles:
        return {"ok": False, "error": "terms required"}
    hits = []
    for r in rows:
        if not include_inactive and r.get("status") != "active":
            continue
        hay = " ".join([str(r.get("statement") or ""),
                        json.dumps(r.get("scope") or {}),
                        json.dumps(r.get("evidence") or [])]).lower()
        if any(n in hay for n in needles):
            hits.append(_row_out(r))
    return {"ok": True, "terms": terms, "count": len(hits), "claims": hits}


def _row_out(r: dict) -> dict:
    return {k: (str(v) if hasattr(v, "isoformat") else v)
            for k, v in r.items() if k != "account"}


# ── evidence + recomputed stats (the promotion gate's ground truth) ──────


def add_evidence(store=None, *, claim_id: int, evidence: list,
                 run_id: str | None = None, account: str = ACCOUNT) -> dict:
    store = store or _store()
    rows = store.select("desk_claims", filters={"id": claim_id,
                                                "account": account})
    if not rows:
        return {"ok": False, "error": f"claim {claim_id} not found"}
    for ref in evidence or []:
        if not isinstance(ref, dict) or ref.get("kind") not in EVIDENCE_KINDS:
            return {"ok": False,
                    "error": f"every evidence ref needs kind ∈ {EVIDENCE_KINDS}"}
    merged = list(rows[0].get("evidence") or []) + list(evidence or [])
    store.update("desk_claims", {"id": claim_id},
                 {"evidence": merged, "updated_run_id": run_id},
                 returning=False)
    _event(store, claim_id=claim_id, event="evidence_added", run_id=run_id,
           detail={"added": len(evidence or []), "total": len(merged)},
           account=account)
    return {"ok": True, "id": claim_id, "evidence_count": len(merged)}


def _resolve_evidence(store, refs: list, account: str) -> dict:
    """Resolve typed refs against their tables. Returns outcome/decision rows
    plus a list of unresolvable refs (lint errors, promotion blockers)."""
    outcomes, decisions, orphans = [], {}, []
    for ref in refs or []:
        kind = (ref or {}).get("kind")
        try:
            if kind == "outcome":
                rows = store.select("desk_outcomes",
                                    filters={"account": account,
                                             "run_id": ref.get("run_id"),
                                             "symbol": ref.get("symbol")})
                if rows:
                    outcomes.append(rows[0])
                else:
                    orphans.append(ref)
            elif kind == "decision":
                rows = store.select("desk_decisions",
                                    filters={"account": account,
                                             "run_id": ref.get("run_id")})
                if rows:
                    decisions[ref.get("run_id")] = rows[0]
                else:
                    orphans.append(ref)
            elif kind == "trade":
                if not store.select("desk_trades", filters={"id": ref.get("id")}):
                    orphans.append(ref)
            elif kind == "backtest":
                if not store.select("desk_backtests",
                                    filters={"id": ref.get("id")}):
                    orphans.append(ref)
            elif kind == "wiki_history":
                if not store.select("desk_wiki_history",
                                    filters={"account": account,
                                             "slug": ref.get("slug"),
                                             "revision": ref.get("revision")}):
                    orphans.append(ref)
            elif kind == "probe":
                pass  # self-contained observation note
            else:
                orphans.append(ref)
        except Exception:  # noqa: BLE001 — a dead table read = unresolvable ref
            orphans.append(ref)
    # decisions behind outcome refs supply the regime + timing context
    for oc in outcomes:
        rid = oc.get("run_id")
        if rid and rid not in decisions:
            rows = store.select("desk_decisions",
                                filters={"account": account, "run_id": rid})
            if rows:
                decisions[rid] = rows[0]
    return {"outcomes": outcomes, "decisions": decisions, "orphans": orphans}


def recompute_stats(store=None, *, claim_id: int,
                    account: str = ACCOUNT) -> dict:
    """Recorded statistics from the evidence refs — never self-reported.
    n counts resolvable graded outcome refs; wins/losses judge alpha sign
    (null alpha = too immature to score, counted but not judged); regimes
    come from the citing decisions; span sessions are approximated from the
    calendar span (×5/7) — stated, not hidden."""
    store = store or _store()
    rows = store.select("desk_claims", filters={"id": claim_id,
                                                "account": account})
    if not rows:
        return {"ok": False, "error": f"claim {claim_id} not found"}
    res = _resolve_evidence(store, rows[0].get("evidence") or [], account)
    outcomes, decisions = res["outcomes"], res["decisions"]

    wins = sum(1 for o in outcomes
               if isinstance(o.get("alpha_pct"), (int, float)) and o["alpha_pct"] > 0)
    losses = sum(1 for o in outcomes
                 if isinstance(o.get("alpha_pct"), (int, float)) and o["alpha_pct"] < 0)
    alphas = [o["alpha_pct"] for o in outcomes
              if isinstance(o.get("alpha_pct"), (int, float))]
    symbols = sorted({o.get("symbol") for o in outcomes if o.get("symbol")})
    regimes: dict[str, int] = {}
    dates: list[date] = []
    for rid, d in decisions.items():
        reg = str(d.get("regime") or "").strip().lower() or "unknown"
        regimes[reg] = regimes.get(reg, 0) + 1
        dt = _as_date(d.get("ts"))
        if dt:
            dates.append(dt)
    span = [str(min(dates)), str(max(dates))] if dates else None
    span_sessions = round((max(dates) - min(dates)).days * 5 / 7) if dates else 0
    closed_unjudged = [
        {"run_id": o.get("run_id"), "symbol": o.get("symbol")}
        for o in outcomes
        if o.get("status") == "closed" and not o.get("verdict")]

    stats = {"n": len(outcomes), "wins": wins, "losses": losses,
             "immature": len(outcomes) - wins - losses,
             "avg_alpha_pct": round(sum(alphas) / len(alphas), 2) if alphas else None,
             "symbols": symbols, "regimes": regimes, "span": span,
             "span_sessions_approx": span_sessions,
             "recomputed_at": str(_utcnow())}
    return {"ok": True, "id": claim_id, "stats": stats,
            "orphans": res["orphans"], "closed_unjudged": closed_unjudged}


# ── promotion: the pre-registered gate, evaluated in code ────────────────


def claim_promote(store=None, *, claim_id: int, run_id: str | None = None,
                  account: str = ACCOUNT) -> dict:
    """candidate → established, refused unless the criteria registered at
    candidate creation are met by RECOMPUTED stats. Effective thresholds are
    the class defaults, tightened (never loosened) by the claim's own
    registered criteria — with one honest escape: a claim explicitly scoped
    to a single regime may establish inside it, but is then FORCED
    regime-conditional with an expiry, so it dies unless renewed."""
    store = store or _store()
    rows = store.select("desk_claims", filters={"id": claim_id,
                                                "account": account})
    if not rows:
        return {"ok": False, "error": f"claim {claim_id} not found"}
    c = rows[0]
    if c.get("status") != "active":
        return {"ok": False, "error": f"claim {claim_id} is {c.get('status')}, "
                                      "not active"}
    if c.get("tier") != "candidate":
        return {"ok": False,
                "error": f"only candidates promote (claim is {c.get('tier')})"}
    registered = c.get("promotion_criteria")
    if not registered:
        return {"ok": False,
                "error": "no promotion_criteria registered at creation — "
                         "criteria written after seeing results don't count"}

    kclass = c.get("kclass")
    defaults = dict(ESTABLISH_DEFAULTS.get(kclass) or {})
    effective = dict(defaults)
    for k, v in registered.items():
        if isinstance(v, (int, float)) and isinstance(effective.get(k),
                                                      (int, float)):
            effective[k] = max(effective[k], v)  # tighten only
        elif k not in effective:
            effective[k] = v

    scope_regimes = [str(r).lower() for r in
                     ((c.get("scope") or {}).get("regimes") or [])]
    single_regime_scoped = len(scope_regimes) == 1
    if single_regime_scoped:
        effective["min_regimes"] = 1

    rc = recompute_stats(store, claim_id=claim_id, account=account)
    if not rc.get("ok"):
        return rc
    stats = rc["stats"]
    failures: list[str] = []
    if rc["orphans"]:
        failures.append(f"{len(rc['orphans'])} evidence ref(s) unresolvable")
    if rc["closed_unjudged"]:
        failures.append("cites closed outcomes with no verdict — grade first, "
                        f"judge second: {rc['closed_unjudged']}")
    if stats["n"] < effective.get("min_n", 1):
        failures.append(f"n={stats['n']} < min_n={effective.get('min_n')}")
    if len(stats["symbols"]) < effective.get("min_symbols", 0):
        failures.append(f"symbols={len(stats['symbols'])} < "
                        f"min_symbols={effective.get('min_symbols')}")
    n_regimes = len([r for r in stats["regimes"] if r != "unknown"])
    if n_regimes < effective.get("min_regimes", 0):
        failures.append(f"regimes={n_regimes} < "
                        f"min_regimes={effective.get('min_regimes')}")
    if single_regime_scoped and stats["regimes"]:
        outside = [r for r in stats["regimes"]
                   if r not in scope_regimes and r != "unknown"]
        if outside:
            failures.append(f"scoped to {scope_regimes} but evidence spans "
                            f"{outside} — widen the scope or drop the refs")
    if stats["span_sessions_approx"] < effective.get("min_span_sessions", 0):
        failures.append(f"span≈{stats['span_sessions_approx']} sessions < "
                        f"min_span_sessions={effective.get('min_span_sessions')}")
    judged = stats["wins"] + stats["losses"]
    if "min_win_rate" in effective:
        # A cautionary pattern ("X underperforms") is SUPPORTED by
        # negative-alpha instances — its criteria declare win_is at
        # registration so the gate scores the claim's own direction.
        supporting = (stats["losses"]
                      if effective.get("win_is") == "negative_alpha"
                      else stats["wins"])
        if not judged:
            failures.append("no judged instances (all alpha null) — nothing "
                            "to win-rate")
        elif supporting / judged < effective["min_win_rate"]:
            failures.append(f"win_rate={supporting}/{judged} < "
                            f"{effective['min_win_rate']} "
                            f"(win_is={effective.get('win_is', 'positive_alpha')})")
    if effective.get("requires_probe_evidence") and not any(
            (r or {}).get("kind") == "probe" for r in c.get("evidence") or []):
        failures.append("system_mechanics needs at least one probe evidence ref")

    evaluation = {"effective_criteria": effective, "stats": stats,
                  "single_regime_scoped": single_regime_scoped,
                  "failures": failures}
    if failures:
        _event(store, claim_id=claim_id, event="demoted", run_id=run_id,
               detail={"promotion_refused": evaluation}, account=account)
        return {"ok": False, "error": "promotion refused", **evaluation}

    values: dict = {"tier": "established", "stats": stats,
                    "updated_run_id": run_id}
    if single_regime_scoped:
        end = _as_date((stats.get("span") or [None, None])[1]) or _today()
        values["decay_class"] = "regime_conditional"
        values["expires_at"] = end + timedelta(
            days=round(REGIME_CLAIM_TTL_SESSIONS * 7 / 5))
    store.update("desk_claims", {"id": claim_id}, values, returning=False)
    _event(store, claim_id=claim_id, event="promoted", run_id=run_id,
           detail=evaluation, account=account)
    return {"ok": True, "id": claim_id, "tier": "established", **evaluation}


# ── status transitions (never delete) ────────────────────────────────────


def claim_retire(store=None, *, claim_id: int, reason: str,
                 run_id: str | None = None, proposal_id: int | None = None,
                 account: str = ACCOUNT) -> dict:
    store = store or _store()
    rows = store.select("desk_claims", filters={"id": claim_id,
                                                "account": account})
    if not rows:
        return {"ok": False, "error": f"claim {claim_id} not found"}
    if rows[0].get("kclass") == "risk_rule":
        # tighten-only: loosening or retiring a risk rule is itself a
        # learned-behavior change and needs an APPROVED owner proposal.
        if not proposal_id:
            return {"ok": False,
                    "error": "risk_rule claims are tighten-only — retiring one "
                             "requires an approved proposal (--proposal-id)"}
        prop = store.select("desk_proposals", filters={"id": proposal_id,
                                                       "account": account})
        if not prop or prop[0].get("status") != "approved":
            return {"ok": False,
                    "error": f"proposal {proposal_id} not found or not approved"}
    store.update("desk_claims", {"id": claim_id},
                 {"status": "retired", "updated_run_id": run_id},
                 returning=False)
    _event(store, claim_id=claim_id, event="retired", run_id=run_id,
           detail={"reason": reason, "proposal_id": proposal_id},
           account=account)
    return {"ok": True, "id": claim_id, "status": "retired"}


def claim_quarantine(store=None, *, claim_id: int, reason: str,
                     run_id: str | None = None,
                     account: str = ACCOUNT) -> dict:
    """Readable, never injected, resurrectable by attaching evidence."""
    store = store or _store()
    rows = store.select("desk_claims", filters={"id": claim_id,
                                                "account": account})
    if not rows:
        return {"ok": False, "error": f"claim {claim_id} not found"}
    store.update("desk_claims", {"id": claim_id},
                 {"status": "quarantined", "updated_run_id": run_id},
                 returning=False)
    _event(store, claim_id=claim_id, event="quarantined", run_id=run_id,
           detail={"reason": reason}, account=account)
    return {"ok": True, "id": claim_id, "status": "quarantined"}


# ── the decision-context read (what brain.context injects) ───────────────


def context_claims(store=None, *, account: str = ACCOUNT) -> dict:
    """The bounded advisory read: ONLY established claims and
    experimental-flagged candidates reach decision context (outcome 2 —
    visibility is the wiki's job; authority is tier-gated here)."""
    store = store or _store()
    rows = store.select("desk_claims",
                        filters={"account": account, "status": "active"},
                        order=[("id", "asc")], limit=500)
    injectable, candidates = [], 0
    for r in rows:
        if r.get("tier") == "established" or r.get("experimental"):
            injectable.append(r)
        elif r.get("tier") == "candidate":
            candidates += 1
    out = []
    for r in injectable[:CONTEXT_MAX_CLAIMS]:
        stats = r.get("stats") or {}
        out.append({
            "id": r["id"], "cite": f"[C-{r['id']}]",
            "kclass": r.get("kclass"), "tier": r.get("tier"),
            "experimental": bool(r.get("experimental")),
            "statement": (r.get("statement") or "")[:CONTEXT_STATEMENT_CLIP],
            "regimes": (r.get("scope") or {}).get("regimes"),
            "n": stats.get("n"), "wins": stats.get("wins"),
            "losses": stats.get("losses"),
            "expires_at": str(r["expires_at"]) if r.get("expires_at") else None,
        })
    return {"note": "established + experimental claims only — the tiers that "
                    "may JUSTIFY a pick (cite ids in picks[].claims). "
                    "Candidates stay watch-only in the wiki prose.",
            "caps": {"experimental_per_claim_weight":
                     EXPERIMENTAL_PER_CLAIM_WEIGHT_CAP,
                     "experimental_total_weight":
                     EXPERIMENTAL_TOTAL_WEIGHT_CAP},
            "count": len(out), "candidates_watching": candidates,
            "claims": out}


# ── commitments (materialization gate arrives with brain step 4) ─────────


def commitment_list(store=None, *, status: str | None = None,
                    account: str = ACCOUNT) -> dict:
    store = store or _store()
    filters: dict = {"account": account}
    if status:
        filters["status"] = status
    rows = store.select("desk_commitments", filters=filters,
                        order=[("id", "asc")], limit=200)
    return {"ok": True, "count": len(rows),
            "commitments": [_row_out(r) for r in rows]}


def commitment_honor(store=None, *, commitment_id: int, run_id: str,
                     note: str | None = None, account: str = ACCOUNT) -> dict:
    """Stamp the deciding run on a fired commitment — even when the decision
    is "standing down, because Y". Silence is the failure mode this exists
    to kill; the stamp is the receipt that the obligation was faced."""
    store = store or _store()
    rows = store.select("desk_commitments", filters={"id": commitment_id,
                                                     "account": account})
    if not rows:
        return {"ok": False, "error": f"commitment {commitment_id} not found"}
    values = {"honored_run_id": run_id}
    if rows[0].get("status") == "fired":
        values["status"] = "honored"
    store.update("desk_commitments", {"id": commitment_id}, values,
                 returning=False)
    return {"ok": True, "id": commitment_id, "status":
            values.get("status", rows[0].get("status")), "note": note}


# ── proposals (approval sync + state-set gate arrive in step 6) ──────────


def proposal_add(store=None, *, title: str, body: str, change_kind: str,
                 claim_ids: list | None = None, payload: dict | None = None,
                 run_id: str | None = None, account: str = ACCOUNT) -> dict:
    store = store or _store()
    if change_kind not in ("params", "rules", "caps", "setup_adoption"):
        return {"ok": False,
                "error": "change_kind must be params/rules/caps/setup_adoption"}
    if not (title or "").strip() or not (body or "").strip():
        return {"ok": False, "error": "title and body required — the owner "
                                      "reads these, write them plainly"}
    row = store.insert("desk_proposals", {
        "account": account, "title": title.strip(), "body": body,
        "claim_ids": claim_ids or [], "change_kind": change_kind,
        "payload": payload, "run_id": run_id,
        "expires_at": _today() + timedelta(days=PROPOSAL_TTL_DAYS),
    })[0]
    for cid in claim_ids or []:
        _event(store, claim_id=cid, event="proposal_linked", run_id=run_id,
               detail={"proposal_id": row["id"]}, account=account)
    return {"ok": True, "id": row["id"], "status": "pending",
            "expires_at": str(row.get("expires_at") or "")}


def proposal_list(store=None, *, status: str | None = None,
                  account: str = ACCOUNT) -> dict:
    store = store or _store()
    filters: dict = {"account": account}
    if status:
        filters["status"] = status
    rows = store.select("desk_proposals", filters=filters,
                        order=[("id", "asc")], limit=100)
    return {"ok": True, "count": len(rows),
            "proposals": [_row_out(r) for r in rows]}


def proposal_get(store=None, *, proposal_id: int,
                 account: str = ACCOUNT) -> dict | None:
    store = store or _store()
    rows = store.select("desk_proposals", filters={"id": proposal_id,
                                                   "account": account})
    return rows[0] if rows else None


def _decide_proposal(store, *, proposal_id: int, status: str, by: str,
                     via: str, account: str) -> dict:
    p = proposal_get(store, proposal_id=proposal_id, account=account)
    if not p:
        return {"ok": False, "error": f"proposal {proposal_id} not found"}
    if p.get("status") not in ("pending",):
        return {"ok": False,
                "error": f"proposal {proposal_id} is already "
                         f"{p.get('status')} — only a pending proposal decides"}
    store.update("desk_proposals", {"id": proposal_id},
                 {"status": status, "decided_by": by, "decided_via": via,
                  "decided_at": _utcnow()}, returning=False)
    return {"ok": True, "id": proposal_id, "status": status,
            "decided_by": by, "decided_via": via}


def proposal_approve(store=None, *, proposal_id: int, by: str = "owner",
                     account: str = ACCOUNT) -> dict:
    """CLI approval fallback (the owner, working in a session/Codespace).
    Weaker than the GitHub channel — no cryptographic authorship — so it
    records ``decided_via='cli'`` for the audit. The agent must NEVER call
    this; it is an owner action."""
    return _decide_proposal(store, proposal_id=proposal_id, status="approved",
                            by=by, via="cli", account=account)


def proposal_reject(store=None, *, proposal_id: int, by: str = "owner",
                    account: str = ACCOUNT) -> dict:
    return _decide_proposal(store, proposal_id=proposal_id, status="rejected",
                            by=by, via="cli", account=account)


# An approving signal on the GitHub issue: a comment body starting with
# "approve"/"approved"/"lgtm", or an "approved" label. Authorship is what
# matters — only the repo owner's login counts (see proposal_sync).
_APPROVE_RE = __import__("re").compile(
    r"^\s*(approve[d]?|lgtm|ship it)\b", __import__("re").IGNORECASE)


def _fetch_github_issue(repo: str, title: str, token: str) -> dict | None:
    """Find the open issue titled ``title`` and return its author-attributed
    comments + labels. stdlib urllib, matching agent.streamer's dispatch
    call. Returns None when the issue isn't found or the read fails."""
    import urllib.parse
    import urllib.request

    def _get(path: str):
        url = f"https://api.github.com/repos/{repo}/{path}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    try:
        q = urllib.parse.quote(f'repo:{repo} in:title "{title}"')
        found = _get(f"../../search/issues?q={q}")
        items = [i for i in found.get("items", [])
                 if (i.get("title") or "").strip() == title]
        if not items:
            return None
        issue = items[0]
        num = issue["number"]
        comments = _get(f"issues/{num}/comments")
        return {
            "labels": [{"name": lb.get("name"),
                        "user": (issue.get("user") or {}).get("login")}
                       for lb in issue.get("labels", [])],
            "comments": [{"user": (c.get("user") or {}).get("login"),
                          "body": c.get("body") or ""} for c in comments],
        }
    except Exception:  # noqa: BLE001 — network/parse failure → unavailable
        return None


def proposal_sync(store=None, *, proposal_id: int, fetch=None,
                  owner: str | None = None, account: str = ACCOUNT) -> dict:
    """Reconcile a pending proposal with its GitHub issue ``PROPOSAL-<id>``.
    Approves ONLY when an approving comment (or an ``approved`` label) was
    authored by the repo owner's login — not the agent's token identity, not
    a bot. ``fetch`` is injectable for testing; the default reads the live
    issue with the issues:read token. Idempotent: a non-pending proposal is
    returned unchanged."""
    from config.settings import settings

    store = store or _store()
    p = proposal_get(store, proposal_id=proposal_id, account=account)
    if not p:
        return {"ok": False, "error": f"proposal {proposal_id} not found"}
    if p.get("status") != "pending":
        return {"ok": True, "id": proposal_id, "status": p.get("status"),
                "note": "already decided — nothing to sync"}
    owner = (owner or settings.github_owner_login or "").strip().lower()
    if not owner:
        return {"ok": False, "error": "no github_owner_login configured"}
    fetcher = fetch
    if fetcher is None:
        token = (settings.github_read_token
                 or __import__("os").environ.get("GITHUB_TOKEN") or "").strip()
        if not token:
            return {"ok": False, "error":
                    "no issues:read token (github_read_token / GITHUB_TOKEN) — "
                    "use the CLI fallback: knowledge proposal-approve"}
        repo = settings.github_dispatch_repo
        data = _fetch_github_issue(repo, f"PROPOSAL-{proposal_id}", token)
    else:
        data = fetcher(f"PROPOSAL-{proposal_id}")
    if not data:
        return {"ok": True, "id": proposal_id, "status": "pending",
                "note": "no PROPOSAL issue found or read failed — still pending"}

    def _by_owner(login):
        return (login or "").strip().lower() == owner

    approved = any(
        _by_owner(lb.get("user")) and (lb.get("name") or "").lower() == "approved"
        for lb in data.get("labels", [])) or any(
        _by_owner(c.get("user")) and _APPROVE_RE.match(c.get("body") or "")
        for c in data.get("comments", []))
    rejected = any(
        _by_owner(c.get("user"))
        and __import__("re").match(r"^\s*reject", c.get("body") or "",
                                   __import__("re").IGNORECASE)
        for c in data.get("comments", []))
    if approved:
        return _decide_proposal(store, proposal_id=proposal_id,
                                status="approved", by=owner, via="github",
                                account=account)
    if rejected:
        return _decide_proposal(store, proposal_id=proposal_id,
                                status="rejected", by=owner, via="github",
                                account=account)
    return {"ok": True, "id": proposal_id, "status": "pending",
            "note": f"no approving comment/label by {owner} yet"}


def proposal_mark_applied(store=None, *, proposal_id: int, run_id: str,
                          account: str = ACCOUNT) -> dict:
    """Stamp the run that consumed an approved proposal — one approval, one
    application (the set_state gate calls this)."""
    store = store or _store()
    p = proposal_get(store, proposal_id=proposal_id, account=account)
    if not p:
        return {"ok": False, "error": f"proposal {proposal_id} not found"}
    if p.get("status") != "approved":
        return {"ok": False,
                "error": f"proposal {proposal_id} is {p.get('status')}, "
                         "not approved — cannot apply"}
    store.update("desk_proposals", {"id": proposal_id},
                 {"status": "applied", "applied_run_id": run_id},
                 returning=False)
    return {"ok": True, "id": proposal_id, "status": "applied"}


# ── lint: the code-checkable half of the honesty checklist ───────────────


def lint(store=None, *, account: str = ACCOUNT) -> dict:
    """Step-2 scope: citation integrity, orphaned evidence, missing verdicts,
    expired-but-active claims. Stats drift, tier violations, commitment and
    hindsight checks extend this in step 7."""
    import re

    store = store or _store()
    errors: list[str] = []
    warnings: list[str] = []

    claims = store.select("desk_claims", filters={"account": account},
                          order=[("id", "asc")], limit=500)
    by_id = {c["id"]: c for c in claims}

    # 1. every [C-n] token in wiki prose resolves to an ACTIVE claim
    try:
        pages = store.select("desk_wiki", filters={"account": account})
    except Exception:  # noqa: BLE001 — pre-deploy grace
        pages = []
    for page in pages:
        for tok in re.findall(r"\[C-(\d+)\]", page.get("body") or ""):
            cid = int(tok)
            c = by_id.get(cid)
            if not c:
                errors.append(f"wiki/{page.get('slug')} cites [C-{cid}] — "
                              "no such claim")
            elif c.get("status") != "active":
                errors.append(f"wiki/{page.get('slug')} cites [C-{cid}] which "
                              f"is {c.get('status')} — update the prose")

    # 2. picks[].claims ids in recent decisions resolve (tier gate = step 5)
    decisions = store.select("desk_decisions", filters={"account": account},
                             order=[("ts", "desc")], limit=40)
    for d in decisions:
        for p in d.get("picks") or []:
            for cid in p.get("claims") or []:
                c = by_id.get(cid)
                if not c:
                    errors.append(f"decision {d.get('run_id')} pick "
                                  f"{p.get('symbol')} cites claim {cid} — "
                                  "no such claim")
                elif c.get("status") != "active":
                    warnings.append(f"decision {d.get('run_id')} pick "
                                    f"{p.get('symbol')} cites claim {cid} "
                                    f"({c.get('status')})")

    # 3. orphaned evidence + closed-unjudged refs on active claims
    for c in claims:
        if c.get("status") != "active":
            continue
        res = _resolve_evidence(store, c.get("evidence") or [], account)
        for ref in res["orphans"]:
            errors.append(f"claim {c['id']} evidence unresolvable: {ref}")

    # 4. missing verdicts on closed outcomes — the standing reflection queue
    try:
        closed = store.select("desk_outcomes",
                              filters={"account": account, "status": "closed"})
    except Exception:  # noqa: BLE001
        closed = []
    for o in closed:
        if not o.get("verdict"):
            warnings.append(f"closed pick unjudged: {o.get('run_id')} "
                            f"{o.get('symbol')} — grade first, judge second")

    # 5. expired-but-active regime claims
    today = _today()
    for c in claims:
        if (c.get("status") == "active"
                and c.get("decay_class") == "regime_conditional"):
            exp = _as_date(c.get("expires_at"))
            if exp and exp < today:
                errors.append(f"claim {c['id']} expired {exp} but still "
                              "active — renew with fresh evidence or supersede")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "counts": {"claims": len(claims),
                       "active": sum(1 for c in claims
                                     if c.get("status") == "active"),
                       "errors": len(errors), "warnings": len(warnings)}}


# ── CLI ──────────────────────────────────────────────────────────────────


def _json_arg(s: str | None):
    return json.loads(s) if s else None


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ca = sub.add_parser("claim-add")
    ca.add_argument("--kclass", required=True, choices=KCLASSES)
    ca.add_argument("--tier", required=True, choices=TIERS)
    ca.add_argument("--statement", required=True)
    ca.add_argument("--scope", help="JSON, must include account")
    ca.add_argument("--evidence", help="JSON list of typed refs")
    ca.add_argument("--stats", help="JSON")
    ca.add_argument("--criteria", help="JSON promotion criteria (candidates)")
    ca.add_argument("--decay-class", choices=DECAY_CLASSES)
    ca.add_argument("--expires-at")
    ca.add_argument("--experimental", action="store_true")
    ca.add_argument("--supersedes", type=int)
    ca.add_argument("--run-id")

    cg = sub.add_parser("claim-get")
    cg.add_argument("--claim-id", type=int, required=True)

    cl = sub.add_parser("claim-list")
    cl.add_argument("--status", default="active")
    cl.add_argument("--tier")
    cl.add_argument("--kclass")

    cs = sub.add_parser("claim-search")
    cs.add_argument("--terms", required=True)

    cp = sub.add_parser("claim-promote")
    cp.add_argument("--claim-id", type=int, required=True)
    cp.add_argument("--run-id")

    cr = sub.add_parser("claim-retire")
    cr.add_argument("--claim-id", type=int, required=True)
    cr.add_argument("--reason", required=True)
    cr.add_argument("--proposal-id", type=int)
    cr.add_argument("--run-id")

    cq = sub.add_parser("claim-quarantine")
    cq.add_argument("--claim-id", type=int, required=True)
    cq.add_argument("--reason", required=True)
    cq.add_argument("--run-id")

    ea = sub.add_parser("evidence-add")
    ea.add_argument("--claim-id", type=int, required=True)
    ea.add_argument("--evidence", required=True, help="JSON list")
    ea.add_argument("--run-id")

    rs = sub.add_parser("recompute-stats")
    rs.add_argument("--claim-id", type=int, required=True)

    cm = sub.add_parser("commitment-list")
    cm.add_argument("--status")

    ch = sub.add_parser("commitment-honor")
    ch.add_argument("--commitment-id", type=int, required=True)
    ch.add_argument("--run-id", required=True)
    ch.add_argument("--note")

    pa = sub.add_parser("proposal-add")
    pa.add_argument("--title", required=True)
    pa.add_argument("--body", required=True)
    pa.add_argument("--change-kind", required=True,
                    choices=("params", "rules", "caps", "setup_adoption"))
    pa.add_argument("--claim-ids", help="JSON list of ints")
    pa.add_argument("--payload", help="JSON — exact intended diff")
    pa.add_argument("--run-id")

    pl = sub.add_parser("proposal-list")
    pl.add_argument("--status")

    ap = sub.add_parser("proposal-approve",
                        help="OWNER action — CLI approval fallback")
    ap.add_argument("--id", type=int, required=True, dest="proposal_id")
    ap.add_argument("--by", default="owner")

    rp = sub.add_parser("proposal-reject", help="OWNER action")
    rp.add_argument("--id", type=int, required=True, dest="proposal_id")
    rp.add_argument("--by", default="owner")

    sy = sub.add_parser("proposal-sync",
                        help="reconcile a pending proposal with its GitHub issue")
    sy.add_argument("--id", type=int, required=True, dest="proposal_id")

    sub.add_parser("lint")
    sub.add_parser("context-claims")

    args = p.parse_args(argv)
    store = _store()

    if args.cmd == "claim-add":
        out = claim_add(store, kclass=args.kclass, tier=args.tier,
                        statement=args.statement,
                        scope=_json_arg(args.scope),
                        evidence=_json_arg(args.evidence),
                        stats=_json_arg(args.stats),
                        promotion_criteria=_json_arg(args.criteria),
                        decay_class=args.decay_class,
                        expires_at=args.expires_at,
                        experimental=args.experimental,
                        supersedes=args.supersedes, run_id=args.run_id)
    elif args.cmd == "claim-get":
        out = get_claim(store, claim_id=args.claim_id)
    elif args.cmd == "claim-list":
        out = list_claims(store, status=args.status or None, tier=args.tier,
                          kclass=args.kclass)
    elif args.cmd == "claim-search":
        out = search_claims(store, terms=args.terms)
    elif args.cmd == "claim-promote":
        out = claim_promote(store, claim_id=args.claim_id, run_id=args.run_id)
    elif args.cmd == "claim-retire":
        out = claim_retire(store, claim_id=args.claim_id, reason=args.reason,
                           proposal_id=args.proposal_id, run_id=args.run_id)
    elif args.cmd == "claim-quarantine":
        out = claim_quarantine(store, claim_id=args.claim_id,
                               reason=args.reason, run_id=args.run_id)
    elif args.cmd == "evidence-add":
        out = add_evidence(store, claim_id=args.claim_id,
                           evidence=_json_arg(args.evidence),
                           run_id=args.run_id)
    elif args.cmd == "recompute-stats":
        out = recompute_stats(store, claim_id=args.claim_id)
    elif args.cmd == "commitment-list":
        out = commitment_list(store, status=args.status)
    elif args.cmd == "commitment-honor":
        out = commitment_honor(store, commitment_id=args.commitment_id,
                               run_id=args.run_id, note=args.note)
    elif args.cmd == "proposal-add":
        out = proposal_add(store, title=args.title, body=args.body,
                           change_kind=args.change_kind,
                           claim_ids=_json_arg(args.claim_ids),
                           payload=_json_arg(args.payload), run_id=args.run_id)
    elif args.cmd == "proposal-list":
        out = proposal_list(store, status=args.status)
    elif args.cmd == "proposal-approve":
        out = proposal_approve(store, proposal_id=args.proposal_id, by=args.by)
    elif args.cmd == "proposal-reject":
        out = proposal_reject(store, proposal_id=args.proposal_id, by=args.by)
    elif args.cmd == "proposal-sync":
        out = proposal_sync(store, proposal_id=args.proposal_id)
    elif args.cmd == "lint":
        out = lint(store)
    elif args.cmd == "context-claims":
        out = context_claims(store)
    else:  # pragma: no cover
        out = {"ok": False, "error": f"unknown cmd {args.cmd}"}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
