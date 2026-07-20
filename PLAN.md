# PLAN.md — Knowledge-layer upgrade rollout

> Phase 3 of the knowledge-layer upgrade. Executes the design in `SCHEMA.md`
> against the findings in `STATE.md`. All three documents were approved by the
> owner 2026-07-19.
>
> **STATUS: COMPLETE (2026-07-20).** All steps shipped and deployed:
> step 0 (verdict backfill, prod data), step 1 = v9.13.0 (schema), step 2 =
> v9.14.0 (agent/knowledge.py, advisory claims in context), step 3
> (remediation run `remediate-2026-07-19` — 9 claims registered, wiki cites
> `[C-n]`, lint 0/0, owner-reviewed), step 4 = v9.15.0 (commitments gate),
> step 5 = v9.16.0 (tier authority + experimental caps), step 6 = v9.17.0
> (owner-approval gate + GitHub PROPOSAL sync + skill-mandate
> reconciliation), step 7 = v9.18.0 (full lint + loop-report), step 8 =
> v9.19.0 (`/api/desk/claims` + `/api/desk/proposals` + the desk Knowledge
> card; `proposal-publish` completing the approval channel; nightly
> report-only lint in data-refresh). Deviation of record: GEV's backfilled
> verdict is NOT_YET (final/unjudgeable — the reflection's prose called it
> a process mistake, not a refuted thesis), not FALSE as sketched.
> This file is kept as the record of the rollout, not live guidance.

## Context

EdgeFinder's self-learning loop was audited (STATE.md) and found genuinely
closed but structurally fragile: epistemic discipline lives in prose
conventions, falsifiable clauses outside the buy/add registry escape machine
checking (one $500 loss already), verdict coverage has drifted from prose, and
promotion/decay/approval are unenforced. SCHEMA.md (approved) rebuilds the
layer around a structured claims registry with code-enforced tiers,
commitments, proposals, and lint — mapped onto the existing wiki, tools, and
skills. This plan sequences the build so the desk keeps trading at every step.

## Ground rules for every step

- Test gate before every merge: `DATABASE_URL= python -m pytest tests/ -q -m "not integration"`.
- Version-bump `dashboard/app.py` per functional merge; commit format `[vX.Y.Z] …`.
- Every rejection-capable code gate ships in the SAME commit as its skill-prose
  update (the skills are the agent's operating manual — code and manual must
  never disagree across a deploy).
- Skill boundaries: trading-side code never touches UI files; the `/desk`
  panel is app-evolver's (step 8). No writes to sacred tables anywhere.
- Rollback story per step: revert the commit — new tables are inert data.
- After each gate deploys, soak: watch the next ~10 live cycles'
  `desk_thinking` + context `errors` for unexpected rejections before starting
  the next step.

## Ordered steps

**Step 0 — verdict backfill (ops only, no code, can run today).**
Run `python -m agent.ledger grade`, then `agent.brain verdict` for the two
prose-judged closed picks: GOOGL `2026-07-07T17:26` (FALSE-leaning "right
process" — wording from the 07-17 postmortem) and GEV `2026-07-09T14:06`
(process mistake), citing the original reflection prose in `--note`.
Verify: `/api/desk/outcomes` shows 3/3 closed picks with verdicts; hit-rate no
longer rests on one row.

**Step 1 — schema (v9.13.0).**
`agent/models.py`: 4 ORM classes (`DeskClaim`, `DeskClaimEvent`,
`DeskCommitment`, `DeskProposal`) + `DESK_TABLE_DDL` entries (+ RLS lines,
matching existing convention). New tests: CRUD via the store seam on SQLite.
Deploy is behaviorally inert (nothing reads the tables yet).
Verify: test suite green; `render_start.py` applies DDL idempotently on deploy.

**Step 2 — knowledge module, advisory (v9.14.0).**
NEW `agent/knowledge.py` (store-injected functions, pattern of brain.py):
`claim-add/get/list/search/supersede/retire/quarantine`, `claim-promote`
(evaluates pre-stated `promotion_criteria` against stats RECOMPUTED from
`desk_outcomes`), claim events, proposal + commitment CRUD (no gates yet),
basic lint (citation integrity, orphaned evidence, missing verdicts, expired
claims). `brain context` gains a `_safe`-wrapped, bounded `claims` section
(established + experimental only, ≤20, statements clipped).
`reflection-agent/SKILL.md` updated: register claims, promotion review,
false-absence rule (two alternate `knowledge search` queries pasted into the
thinking note before any "nothing recorded" statement).
Verify: unit tests incl. promotion-refusal cases; `brain context` on a seeded
SQLite DB shows the claims section; `agent.preflight` clean.

**Step 3 — remediation batch 1 (data, OWNER-REVIEWED before it counts).**
One supervised session executes SCHEMA.md §8: register claims for the existing
corpus (demote the n=2 rule to candidate with criteria written now; AAPL
trim-clause loss → founding `risk_rule` n=1 never-decay; ops incident →
`operational`; playbook split; observations registered watch-only), then edit
wiki pages to carry `[C-n]` citations (history archives as always).
**Deliverable to owner before proceeding: a remediation report** — every claim
created (id, tier, class, evidence refs) + wiki diffs — via the journal and the
cycle-report email (or pasted in chat). Step 4 starts only on owner OK.

**Step 4 — commitments gate (v9.15.0).** First rejection-capable gate.
`_validate_picks`: conditional-clause detector over trim/exit/hold picks →
reject unless structured `commitment` present. `save_decision`: materialize to
`desk_commitments` + arm linked advisory tripwire via existing `watch_set`.
`ledger grade`: commitments sweep reusing `_kill_breached`; fired-unhonored
commitments surface in `context()`. Trading `SKILL.md` step-6 (+ step-0/8
attention prose) updated in the same commit.
Verify: validator tests (clause → reject; +commitment → accept; false-positive
rephrase → accept); sweep test against seeded bars; live soak.

**Step 5 — tier authority + experimental caps (v9.16.0).**
`_validate_picks`: every `picks[].claims` id must resolve to an active claim
of tier established (or experimental-flagged); `save_decision`: experimental
weight caps (5%/claim, 10% total) vs `target_weights` — a same-cycle detective
check (fills precede the save; stated honestly in the skill). Trading skill:
"prose can inform; only claims can justify" — cite claim ids on picks they
motivated. Verify: tests for tier rejection, cap breach; live soak.

**Step 6 — proposals + strategy gate + GitHub sync (v9.17.0).**
`set_state(bump=True)` and registered cap-key params changes require
`--proposal-id <approved>` or `--no-learned-basis "<reason>"` (every use →
claim_event + mandatory reflection-audit item). `knowledge proposal-sync`
reads issue `PROPOSAL-<id>` via the GitHub API (workflow `GITHUB_TOKEN`) and
approves only on the repo owner's login. `scripts/cycle_report.py` gains a
pending-proposals section. **Same commit: rewrite trading SKILL.md:84-91** —
caps become the agent's to propose raising; tightening stays free.
Verify: gate tests with fake store; sync parsing with mocked API; end-to-end
dry run: file proposal → approve on GitHub → sync → gated `state-set` passes.

**Step 7 — full lint + instrumentation (v9.18.0).**
Complete `knowledge lint` (stats/prose-number drift, tier violations,
hindsight window, hygiene). NEW `knowledge loop-report [--days 7]` — the
loop-honesty instrument: counts of observations banked, claims
created/promoted/superseded/expired, **claims actually cited by decisions**
(the honest "was it read" metric), commitments armed/fired/honored, verdicts
written vs closed picks, `--no-learned-basis` uses, proposal queue age.
Reflection runs lint at start and embeds loop-report in the Friday journal;
cycle-report email carries it weekly; data-refresh may run lint nightly
report-only. Verify: lint clean on remediated prod (read-only); loop-report
numbers reconcile with raw tables.

**Step 8 — /desk surface (app-evolver, decoupled).**
`/api/desk/claims` (+ proposals pane) delivered through the app-evolver skill
per boundary rules; hand over the API contract in a short docs note. Not on
the critical path.

## End-to-end verification (after step 5, on SQLite; again on prod read-only)

Full simulated cycle: seed bars → create candidate claim (criteria recorded) →
`claim-promote` refused (n too small) → add graded evidence → promote passes →
decision citing it saves → decision citing an observation-tier claim REJECTED →
trim with free-text re-add clause REJECTED → same trim with structured
commitment ACCEPTED → `grade` marks the commitment fired when bars breach →
next `context` shows the unhonored obligation → lint returns clean.
Plus `scripts/smoke_dashboard.py`, `agent.preflight`, and one observed live
GitHub-Actions cycle after each gate deploy.

## Deferred items — reactivation triggers

- **Embeddings/vector search**: active claims > ~200, or the false-absence
  audit catches two-term search failing to find items that exist.
- **Graph tooling**: contradiction/supersession analysis needs multi-hop
  traversal beyond what lint's SQL+Python handles.
- **ML on the knowledge layer**: not before live-money questions exist; only
  if claim volume defeats manual curation.
- **Multi-agent separation** (independent verifier identity for the approval
  gate): evidence in reflection audits of rationalizing bypass —
  `--no-learned-basis` abuse or paraphrase-without-citation recurring.
- **Paper→live revalidation class activation**: any move toward live trading
  (schema is ready; `scope.account` already mandatory).
