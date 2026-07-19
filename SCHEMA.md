# SCHEMA.md — Upgraded knowledge architecture

> Phase 2 of the knowledge-layer upgrade, mapped onto what exists (see
> `STATE.md` for the audit). Approved by the owner 2026-07-19, thresholds
> accepted as proposed. Rollout sequencing lives in `PLAN.md`.

## 0. Two audit corrections that shaped this design

- The trim/exit registry hole is **unvalidated, not prohibited**: picks of any
  action may already carry `prediction`/`horizon_days`/`kill`; `_validate_picks`
  (brain.py:173-174) simply skips non-opening actions. The fix is "validate",
  not "allow".
- **The trading skill's owner-mandate block (SKILL.md:84-91) contradicts
  outcome 6 (owner approval for learned-behavior changes)**: it currently
  tells the agent its caps "are yours to raise… worth a pivot." That prose
  must be reconciled in the same commit as any approval gate, or the agent
  follows two masters.

## 1. Core decision: structured claims registry + wiki as narrative view

**`desk_claims` becomes the source of truth for every behavior-influencing
fact. The wiki stays exactly as built** (6 slugs, 8k/40k caps, history
archival) **as the trader-readable narrative and public `/desk` surface — and
must cite claims by token (`[C-12]`).** One arrow: claims → cited by wiki
prose → cited by decisions (`picks[].claims`).

Rejected alternatives: wiki-with-frontmatter (whole-page rewrites destroy
per-claim provenance; frontmatter eats the caps; code gates need columns, not
a markdown parser working over two DB transports); claims-only (the prose
narrative demonstrably works and is the owner's/public reading surface).

**Visibility vs authority (resolves the "candidates in the lessons page"
tension):** the wiki remains fully visible to the trader every cycle —
including candidate/observation prose — but *authority* is citation-gated:
only `established` or `experimental`-flagged claim ids may justify a pick, and
that tier check is enforced in code at decision-save. Candidates are
watch-only context. Prose can inform; only claims can justify.

## 2. New tables (4)

Additive, DDL via `DESK_TABLE_DDL` + ORM (the desk tables' only migration
path; alembic is NOT used for `desk_*`), no FKs (matching existing
convention; integrity checked by lint), both transports untouched (new access
is eq/range filters + Python-side JSON scanning).

**`desk_claims`** — one row per behavior-influencing fact:
`id, account, kclass (market_strategy|system_mechanics|operational|risk_rule),
tier (observation|digest|candidate|established), experimental bool,
status (active|superseded|retired|quarantined), statement (≤400 chars,
falsifiable), scope JSON ({account:"paper" REQUIRED, universe, regimes[],
strategy_versions[]}), evidence JSON (typed machine-resolvable refs:
outcome/decision/trade/backtest/wiki_history/probe), stats JSON (n, wins,
losses, avg_alpha_pct, span, regimes{}, symbols[] — recorded statistics with
sample sizes; NO confidence floats anywhere in the schema),
promotion_criteria JSON (written at candidate creation, BEFORE results; code
refuses promotion without it), decay_class (regime_conditional|stable|never —
code-forced default by kclass: risk_rule→never, system_mechanics→stable,
market_strategy→regime_conditional), expires_at (required for
regime_conditional), review_after, supersedes/superseded_by,
created_at/created_run_id/updated_at/updated_run_id.`

**`desk_claim_events`** — append-only lifecycle audit:
`claim_id, ts, run_id, event (created|evidence_added|promoted|demoted|
superseded|retired|quarantined|expired|proposal_linked), detail JSON` (tier
before/after, stats snapshot, criteria evaluation). Makes outcome 1's
traceable path queryable without prose archaeology.

**`desk_commitments`** — the falsification-clause fix:
`run_id, symbol, kind (reentry|stop|review), direction (above|below), level,
until, text (verbatim clause), status (open|fired|honored|expired|withdrawn),
fired_date, fired_close, honored_run_id, watch_id (linked tripwire)`.

**`desk_proposals`** — owner-approval queue:
`title, body (plain-English what/why), claim_ids JSON (the justifying facts),
change_kind (params|rules|caps|setup_adoption), payload JSON (exact intended
state-set diff), status (pending|approved|rejected|expired|applied),
decided_at/decided_by/decided_via, applied_run_id, expires_at (TTL 14d)`.

**No column changes to existing tables.** Decision→claim links ride in the
existing `picks` JSON (`"claims": [12, 17]`); commitments ride in a pick's
`"commitment": {...}` object, materialized on save.

New code in **`agent/knowledge.py`** (claims/commitments/proposals CRUD,
promotion gate, lint; store-injected functions, SQLite-testable), CLI
`python -m agent.knowledge`. `brain context` imports its read side.

## 3. Outcomes → enforcement map (code vs discipline)

Code can enforce shape, thresholds, arithmetic, existence, expiry, exposure
caps. It cannot enforce truth, hindsight-free semantics, or whether the agent
heeds what it read — those remain LLM discipline made cheap by tooling and
visible by lint + reflection audit.

| # | Code-enforced | LLM discipline + lint |
|---|---|---|
| 1 Loop/trace | `picks[].claims` validated (existence + tier) in `_validate_picks`; claim_events on every transition; `context()` injects established+experimental claims with ids | Citing the claim that actually motivated the pick; reflection audits paraphrase-without-citation |
| 2 Tiers/promotion | `claim-promote` refuses candidate→established unless pre-stated `promotion_criteria` met by stats **recomputed from `desk_outcomes`** (not self-reported); context injects only established+experimental | Choosing honest criteria; lint cross-checks n against ALL matching outcome rows, not just cited ones |
| 3 Provenance | `claim-add` rejects market/risk claims with empty evidence; `scope.account` required; schema has no confidence column | Scope honesty; paper→live revalidation is schema-ready, future code |
| 4 Hindsight | Evidence run timestamps < claim `created_at`; `stats.span` end ≤ creation; numeric cites checkable against decision/outcome rows | "Should have known" semantics; lessons must quote decision-time record (reflection checklist) |
| 5 Supersession/decay | No delete path; supersede requires `superseded_by`; expiry mandatory for regime_conditional; risk_rule→never is a code constant; lint flags expired-active | Writing the superseding claim honestly |
| 6 Write-gate/approval | observation/digest ungated; `set_state(bump=True)` + registered cap-key changes require `--proposal-id` (approved) or `--no-learned-basis "<reason>"` (audited escape hatch); experimental weight caps checked at decision-save | Every `--no-learned-basis` is a mandatory reflection-audit item (same pattern as the working bear-case audit) |
| 7 Classes | `kclass` drives decay defaults, promotion defaults, lint rules | Correct classification at write time |

**Honesty note on experimental caps:** fills execute before the decision saves
within a cycle, so the save-time weight-cap check catches a breach same-cycle
(forcing remediation) rather than preventing the fill. The preventive layer is
skill prose; the detective layers are the save gate + lint. Stated, not
papered over.

## 4. Commitments — closing the free-text falsification hole

1. **Write-gate** in `_validate_picks`: conditional-clause regex
   (`re-add|re-enter|reload|back in|if it (reclaims|closes|breaks|holds)|unless`)
   over trim/exit/hold picks' `why_now`/`rationale`/`kill`; a match REJECTS the
   save unless the pick carries a structured
   `commitment {kind, direction, level, until_sessions, text-verbatim}`. False
   positives cost one small JSON object. (Residual risk: phrasing that dodges
   the regex — lint keeps sweeping, reflection audits.)
2. **Materialization**: `save_decision` writes each commitment to
   `desk_commitments` and arms a linked advisory tripwire via existing
   `watch_set` — so a fired re-add clause *wakes the agent* through the
   already-running streamer sweep. Zero new infrastructure.
3. **Machine-check**: `ledger grade` gains a commitments sweep reusing
   `_kill_breached` (split-aware touch semantics) → `status=fired` +
   `fired_date/fired_close`. Fired-and-unhonored commitments surface in
   `context()` as unhandled obligations and in lint. Any later decision on the
   symbol stamps `honored_run_id` — even if the answer is "standing down,
   because Y."

The full prediction registry is NOT extended to trims (nothing falsifiable to
grade in a risk-closing trim per se); the dangerous residue is exactly the
conditional clause, and commitments capture that machine-checkably.

## 5. Promotion thresholds (constants in `agent/knowledge.py`; owner adjusts)

```python
CANDIDATE_MIN_N = 2                    # 1 instance = observation ("watching")
ESTABLISH_DEFAULTS = {
  "market_strategy": {"min_n": 5, "min_symbols": 3, "min_regimes": 2,
                      "min_span_sessions": 20, "min_win_rate": 0.6},
  "system_mechanics": {"min_n": 1, "requires_probe_evidence": True},
  "operational":      {"min_n": 1},
  "risk_rule":        {"min_n": 1, "tighten_only": True},
}
EXPERIMENTAL_TOTAL_WEIGHT_CAP = 0.10   # of equity, across experimental-cited picks
EXPERIMENTAL_PER_CLAIM_WEIGHT_CAP = 0.05
REGIME_CLAIM_TTL_SESSIONS = 60         # regime_conditional expiry after last evidence
PROPOSAL_TTL_DAYS = 14
```

Reasoning: at ~2–5 graded picks/week, n=5 across ≥3 symbols lands in 3–6
weeks — fast enough to matter, big enough to kill single-story generalization.
`min_regimes: 2` means most market claims cannot establish inside one regime —
correct; the escape is explicit regime scoping (`regimes:["risk_on"]` may
establish with min_regimes 1 but is FORCED regime_conditional with 60-session
expiry — it dies unless renewed). Win-rate 0.6 at n=5 is a generalization
brake, not a significance test — the skill says so plainly. **Risk lessons
establish at n=1 by design** (demanding repeated $500 losses is perverse); the
compensating constraint is `tighten_only`, and loosening/retiring a risk_rule
is itself proposal-gated.

## 6. Owner approval — realistic mechanism

Honest constraint: in a single-agent system with full env access, approval
cannot be forge-proof against a deliberately lying agent; the threat model is
a *rationalizing* agent. Design: durable, externally-verifiable record; loud
bypass.

1. Agent files `knowledge proposal-add` (claims + plain-English body + exact
   payload); pending, 14-day TTL.
2. Surfaced in the cycle-report email (`scripts/cycle_report.py` gains one
   REST read) and on `/desk` (app-evolver delivery).
3. **Approval channel = GitHub issue `PROPOSAL-<id>`**: owner approves by
   comment/label; `knowledge proposal-sync` flips status only when the
   approving author is the repo owner's login (verifiable authorship; not the
   agent's token). CLI fallback for Codespaces recorded as weaker
   (`decided_via='cli'`).
4. Gate as in §3 row 6; skill prose reconciled: caps become the agent's to
   *propose* raising (tightening stays free) — rewrite of SKILL.md:84-91 ships
   in the same commit.

## 7. Lint — `python -m agent.knowledge lint` (code) + reflection checklist (LLM)

**Code lint** (runs at reflection start; cheap enough for nightly
data-refresh): citation integrity ([C-n] tokens + picks[].claims resolve to
active claims); orphaned evidence refs; missing verdicts on closed outcomes
(promotion additionally hard-refuses claims citing unjudged closed outcomes);
expired-but-active claims; stats drift (recompute from desk_outcomes vs stored
stats; plus prose-number drift for figures adjacent to [C-n] citations);
tier violations (sub-established cited without experimental flag; cap
breaches); fired-unhonored commitments + legacy free-text clause regex sweep;
hindsight window check; hygiene (missing regime scope, empty evidence,
stale proposals).

**Reflection checklist (prose)**: contradiction review per active claim vs the
week's graded outcomes (supersede, never delete); **false-absence rule** —
before claiming "nothing recorded about X", run `knowledge search` twice with
two alternate phrasings and paste both queries into the thinking note; hindsight
semantics (quote the decision-time record); promotion review (candidates whose
recomputed stats meet their pre-stated criteria: promote or explain why not).

## 8. Remediation of existing content (one supervised session; nothing deleted)

| Item | Action |
|---|---|
| GOOGL + GEV closed picks, verdict NULL | Backfill via `brain verdict`, citing the original reflection prose in `--note` |
| lessons: n=2 "momentum ≠ since-entry" rule | **Demote to candidate** — register claim with n=2 evidence + promotion_criteria written now (post-hoc noted in claim_event); wiki line gets `[C-x]` + "candidate — needs n=5 / 2 regimes". It violates the skill's own 3-instance gate |
| lessons: 2 single-instance + META tension | Register as observation (watch-only) / candidate per actual evidence state; evidence refs where runs identifiable |
| mistakes: AAPL trim-clause $500 loss | → `risk_rule`, established n=1, decay never — becomes the charter citation for the commitments gate itself |
| mistakes: other 3 | `operational` / `risk_rule` per content; established n=1 where a concrete loss/incident backs them |
| postmortems (3) | Keep as-is — episodic evidence referenced BY claims |
| playbook | Split: code-enforced mechanics → `system_mechanics` claims; performance assertions → candidates with expiry; guardrails stay prose + risk_rule claims |
| market-notes ops incident (streamer sweep latency) | → `operational` claim; pointer stays in prose |
| Anything with no traceable evidence | `quarantined` — readable, never injected, resurrectable with evidence (expect ~0–2 items) |

## 9. Rollout ordering (system runnable at every step; each gate ships with its skill prose)

0. Backfill 2 verdicts + `ledger grade` (no schema; can run today).
1. Schema only: 4 tables (ORM + DDL + RLS), CRUD unit tests. Behaviorally inert.
2. Claims read/write advisory: `agent/knowledge.py`, `context()` gains bounded
   `claims` section (established+experimental only, `_safe`-wrapped);
   reflection skill registers claims. Trading skill untouched.
3. Remediation session (§8) using Phase-2 tooling.
4. Commitments gate (+ trading SKILL.md step-6 update, same commit) — first
   rejection-capable gate.
5. Tier enforcement + experimental caps in `save_decision` (+ skill prose).
6. Proposals + `set_state` gate + GitHub sync + SKILL.md:84-91 reconciliation
   + cycle-report section, one commit.
7. Lint + reflection checklist (same commit); `/desk` claims+proposals panel
   via app-evolver (API contract handed over — skill boundaries respected).

Rollback at any phase = revert code; tables are inert data.

### Critical files
- `agent/brain.py` — `_validate_picks` (:144), `save_decision` (:195),
  `set_state` (:75), `context()` (:776): all four gates + context injection
- `agent/models.py` — ORM + `DESK_TABLE_DDL` (:532): the only desk migration path
- `agent/ledger.py` — `grade()` (:1973) commitments sweep reusing `_kill_breached` (:1851)
- `.claude/skills/trading-agent/SKILL.md` — step 6 (:361-398) + owner-mandate (:84-91)
- `.claude/skills/reflection-agent/SKILL.md` — claim registration, promotion review, lint, false-absence
- NEW `agent/knowledge.py` — claims/commitments/proposals/lint, store-injected, SQLite-testable
