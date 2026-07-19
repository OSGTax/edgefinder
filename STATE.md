# STATE.md — The self-learning layer as built

> Phase 1 (read-only audit) of the knowledge-layer upgrade. Confirmed by the
> owner 2026-07-19. Companion docs: `SCHEMA.md` (approved design), `PLAN.md`
> (rollout).
>
> Audit date: 2026-07-19 (Sat). Evidence: repo at commit 9deea5d (v9.12.2) +
> read-only SQL against the production Supabase (`desk_*` tables). Book age at
> audit: 10 calendar days (inception 2026-07-07), 22 fills, 3 closed round trips.

## 1. The knowledge store as built

**Storage form:** Postgres (Supabase), plain relational tables + JSON columns.
No embeddings, no vector store, no graph. All agent access via Python CLIs
(`agent/brain.py`, `agent/ledger.py`) — the skills forbid raw SQL.

**The store is layered, not monolithic** — six surfaces hold learned/learnable
content:

| Surface | Table | Size today | Role |
|---|---|---|---|
| Lessons wiki | `desk_wiki` (+ `desk_wiki_history`) | 5 pages (+3 archived revisions) | Curated durable knowledge — THE "learning" store |
| Prediction registry | `desk_decisions.picks[]` (`prediction`/`horizon_days`/`kill`) + `rejected[]` | 78 decision rows | Falsifiable claims registered at decision time |
| Machine grades | `desk_outcomes` | 13 rows | Per-pick facts (alpha, kill-breach, exit) + reflection's `verdict` |
| Journal | `desk_journal` | 20 rows | Pivots, wiki-edit receipts, weekly reflection notes |
| Thinking feed | `desk_thinking` | 375 rows | Append-only narration incl. per-cycle "falsifiable observation" study notes |
| Backtest evidence | `desk_backtests` | 513 rows | Lab sweep results (spec + result JSON, split-sample) |

Adjacent-but-relevant: `desk_strategy_state` (versioned strategy, v2→v5;
`study_log` scratch memory), `desk_briefs` (nightly research pack — market data
+ lab leaderboard only, does NOT contain wiki content).

**Wiki schema** (`agent/models.py:299-350`): `(account, slug)` unique; columns
`slug, title, body (markdown), revision, updated_at, updated_run_id`. Six fixed
slugs hard-coded in `agent/brain.py:518` — `playbook, setups, lessons,
mistakes, postmortems, market-notes`. Caps: 8,000 chars/page, 40,000 chars
total (`WIKI_PAGE_MAX_CHARS`/`WIKI_TOTAL_MAX_CHARS`, brain.py:521-529) —
bounded prompt growth by construction. **No columns for**: tier/status,
sample size, confidence, provenance beyond `updated_run_id`, decay/expiry,
or supersession links. All epistemic structure lives in prose conventions
inside `body`.

**Current wiki inventory** (all 5 pages read in full):
- `playbook` r3 — posture (owner-directed "lean aggressive" v5), aggression
  toolkit, structural guardrails ("Wiki is advisory; it can NEVER loosen a
  guardrail", "Every lesson must cite its evidence: name, run, dollars").
- `lessons` r2 — 4 entries, each tagged `[risk_on]`/`[neutral]` + an explicit
  evidence state: "confirmed, repeated" (1), "one instance, watching" (2),
  "repeated 2 weeks, not yet a rule change" (1).
- `mistakes` r2 — 4 entries with dated dollar costs.
- `postmortems` r1 — 3 closed round trips, "sourced from `agent.ledger grade`'s
  closed rows, not memory."
- `market-notes` r3 — weekly episodic digest (scoreboard, road-not-taken score,
  an unresolved-tripwire flag, an ops note).
- `setups` — **does not exist yet** (its 3-graded-instances promotion gate has
  never been met; honest restraint).

## 2. Write path

**Single choke point:** `agent.brain wiki-set` → `set_wiki()`
(`agent/brain.py:551-638`) is the ONLY wiki writer in production code
(verified by repo-wide grep). Gates, in order: slug ∈ 6 fixed slugs → page cap
→ total cap → **outgoing body banked to `desk_wiki_history` BEFORE overwrite**
(edit aborts if archival fails, brain.py:586-621) → auto-journal receipt
(`kind="wiki"`). There is **no delete path**; "pruning" = rewriting smaller.

**Who writes, when (trade lifecycle):**
- **Every cycle** (trading-agent skill): `think` narration; one "falsifiable
  observation" per study block (thinking + `study_log`); `decision` saves the
  dossier — **code-enforced registry**: `_validate_picks`
  (brain.py:144-192) REJECTS any buy/add pick missing `prediction`,
  `horizon_days ≥ 1`, or `kill`; `rejected[]` records road-not-taken.
- **Post-trade**: `ledger grade` (ledger.py:1973+) materializes machine facts
  into `desk_outcomes` — the only writer of the fact columns; explicitly never
  touches `verdict`/`verdict_note` ("grade first, judge second").
- **Wiki edits from trading cycles are deliberately rare**: skill says "most
  cycles write NOTHING… AT MOST ONE page… cite the numbers." Observed: true —
  all lessons/mistakes/postmortems revisions came from the two Friday
  reflections (`reflect-2026-07-10`, `reflect-2026-07-17`); playbook edits came
  from owner-directed pivots.
- **Weekly** (reflection-agent skill, Fridays): `brain verdict` writes
  TRUE/FALSE/NOT_YET per graded pick (sole writer of `desk_outcomes.verdict`,
  requires the graded row to pre-exist); wiki curation with skill-defined
  rules — generalize only on REPEAT, setups need ≥3 graded instances, regime
  tag every lesson, postmortems from grade rows never memory, alpha not
  dollars, maturity deferral (`spy_window_sessions < 2` = benchmark noise,
  draw no lesson).

**Notable:** the promotion-ish criteria that DO exist (repeat-gated
generalization, 3-instance setups gate, maturity rules) live in **skill prose,
enforced by model discipline** — nothing in code checks them. The only
code-enforced learning gates are the prediction registry and wiki slug/size
caps.

## 3. Read path

**One funnel:** `agent.brain context` (brain.py:776-927) — the skill-mandated
first read of every trading cycle — assembles: account state, the brief
(clipped), **the ENTIRE wiki (all pages, no filtering — there is no status
column to filter on)**, strategy state, **open predictions joined to their
machine grades and verdicts** (capped 20; excludes only picks both closed AND
verdict-judged), a condensed outcomes summary (capped 15 runs), tripped wires,
due wakes.

- The brief does NOT include the wiki; `agent/market.py` never reads
  `desk_wiki`/`desk_outcomes` (grep-verified). Lab and data-refresh skills
  never touch the wiki. Dashboard `/api/desk/wiki` and `/api/desk/outcomes`
  are read-only public projections.
- **Consequence:** knowledge reaches decisions solely via `context`, every
  cycle, as one undifferentiated markdown corpus. Read-side traceability
  (which fact touched which decision) exists only when the agent happens to
  cite a lesson in prose (it sometimes does — see §4, trace E).

## 4. Loop-closure trace — 5 trades end to end

**A. NVDA buy 2026-07-10 (registry-era happy path — loop VERIFIED).**
Decision `2026-07-10T15:00` carries prediction ("closes above $215 within 10
sessions…"), horizon 14, kill ("two consecutive closes below $200…"). Fill
booked same run. Graded: `desk_outcomes` row, since -4.04% / alpha -2.92%,
open, horizon not elapsed. Replayed to every subsequent cycle via
`context.open_predictions` with its grade attached. Kill mirrored as a
streamer tripwire (id 8, $190). Learned: nothing yet (inside horizon —
correctly so). **Write ✓ read ✓ behavior-arm ✓.**

**B. AMD buy 07-09 → hard stop 07-17 (fullest loop, pre-registry entry).**
Entry predates registry enforcement — no structured prediction. Kill existed
only as a $475 tripwire in the attention system. 07-17 pre-market gap: streamer
sweep missed it; the cycle sold manually at the live bid as backstop (fill
rationale records this). Graded closed: -$1,042.66, **alpha -11.31%**, verdict
**FALSE** written by reflection ("Soft grade (pre-registry, no formal
prediction attached at entry)…"). Lesson written (single-day-rip entry timing,
"one instance, watching"), postmortem written, ops note on sweep latency
written to market-notes. **Loop closed including honest behavior (stop
honored) — and the reflection explicitly acknowledged grading a pre-registry
pick was "soft."**

**C. GOOGL buy 07-07 → rule-based exit 07-09 (loop closed, verdict missing).**
Mechanical exit rule fired on a live snapshot (-2.46% intraday); day closed
-0.84%. Graded closed: -$887, alpha -4.5%. Lesson written
("snapshot can reverse — one instance, NOT proof the rule is wrong… if this
repeats, reconfirm one cycle later"). Postmortem written. **But
`desk_outcomes.verdict` is NULL** — judged in prose only. On 07-14 the same
exit rule fired on LLY and was executed without the "reconfirm next cycle"
refinement — defensible (the lesson was explicitly not-yet-a-rule) but shows
sub-established observations have no defined influence semantics.

**D. AAPL trim 07-10 (LOOP FAILURE — the audit's clearest defect).**
The trim's falsification clause lived in free-text rationale: "AAPL rallies
through $325 with no pullback in 5 sessions → the trim was too early; re-add."
It fired 07-15 ($327.50 close). **No structured field carried it, so nothing
machine-checked it**: `grade` tracks only buy/add picks' kills; the trim's
clause was invisible. Silent until the 07-17 reflection caught it by rereading
prose — ~$500 estimated opportunity cost, now a mistakes-page entry. **A
falsifiable claim that escaped the registry escaped the whole loop.**

**E. Read→behavior evidence (is the wiki actually consulted?). Yes — traced:**
thinking 07-13 13:10 cites the momentum-vs-since-entry lesson while assessing
LLY; 07-15 19:10 says a drawdown pattern is "exactly the pattern the wiki
already has a rule for"; the 07-14 LLY exit note states peer-median momentum
AND the since-entry mark side by side — the exact discipline the 07-10 lesson
promoted to a rule. The rejected-registry → road-not-taken grading →
META-filter-tension lesson chain also closed (a lesson learned from trades NOT
taken). The reflection's 07-17 "AMD spread tripwire unresolved" flag was
carried by weekend cycles (kills re-armed, Monday prep queued) — deferred, not
dropped, though its resolution rests on prose + one tripwire, not a tracked
obligation.

**Verdict on outcome 1 (closed loop):** the loop exists and demonstrably runs
in both directions. Its weak edges: claims outside the buy/add registry (trace
D), verdict coverage (trace C), and no structured fact→decision trace.

## 5. Content quality vs the target tiers

Mapping the existing corpus onto observation → episodic digest → candidate →
established:

| Tier (target) | What exists today | Fit |
|---|---|---|
| Observation | thinking feed (append-only, run-stamped), study observations, `desk_outcomes` facts, postmortems | Good raw material; no structured tier label |
| Episodic digest | market-notes weekly recap; journal wrap notes | Exists, prose |
| Candidate | lessons entries marked "one instance, watching" / "not yet a rule change" | Exists as **prose convention only** |
| Established | lessons entry marked "confirmed, repeated"; playbook rules | **No pre-defined numeric threshold**; see below |

**Flags (ranked by impact on decision quality):**

1. **Small-n promotion:** "Selection momentum ≠ since-entry mark" was promoted
   to a rule at **n=2** (IWM, LLY), single regime (`[risk_on]`), 3-day-old
   book. The promotion criterion ("it has now repeated") was applied honestly
   but was not defined numerically before results, and n=2/one-regime would
   fail the target's established bar. It already influences decisions (trace E).
2. **Verdict coverage 1 of 3 closed picks** (AMD FALSE; GOOGL, GEV NULL despite
   both being judged in postmortem prose). The dashboard's `hit_rate_pct`
   currently rests on a single verdict. Prose and DB have already diverged.
3. **Unstructured falsification clauses** (trace D) — class of defect, will
   recur on every trim/exit/stance carrying a prose promise.
4. **Prose numbers drift:** lessons page says META was rejected "18 separate
   times this week"; the rejected registry holds 14 META rejections since
   07-13. Not fabricated (dashboard-era counts may differ), but the claim
   isn't mechanically derivable — exactly how noise creeps in.
5. **Hindsight contamination: none found.** Lessons cite decision-time
   rationales + measured outcomes; grading uses later data only to score, not
   to rewrite entry reasoning. The GOOGL lesson explicitly defends the rule
   against hindsight ("NOT proof the rule is wrong"). No schema enforcement,
   though — clean by discipline, not by construction.
6. **Contradictions: none found** across the 5 pages.
7. **Provenance: present in prose, absent in structure.** Every lesson cites
   name/date/dollars (the playbook mandates it); `updated_run_id` stamps the
   writing run. But no trade IDs / decision links a query could follow, no
   paper-vs-live class (all fills are paper by charter — any future live use
   would need re-validation tagging that doesn't exist), no strategy-version
   scope on lessons.
8. **No self-assessed confidence floats anywhere** — the target's ban is
   already satisfied; evidence states are verbal + counted instances.
9. **Decay:** none modeled. Regime tags exist (prose). Strategy-performance
   claims never expire; risk lessons and mechanics aren't distinguished.
10. **Class separation:** partial via fixed slugs. Ops incidents have no home —
    the streamer-sweep-latency incident (a system-mechanics fact that should
    never decay like a regime claim) lives inside market-notes (a weekly digest
    page that gets compressed). System-mechanics knowledge otherwise lives in
    git-versioned files (skills, CLAUDE.md) — outside the wiki, which is
    arguably correct, but unwritten as policy.
11. **Early history gap:** 3 of 6 replaced wiki bodies are missing from
    `desk_wiki_history` (playbook r1/r2, market-notes r1 — replaced 07-08/09/10,
    before archival demonstrably worked on 07-17). Silent loss happened
    historically; archival now enforced in code.
12. **Bear-case beat is near-vestigial in practice:** exactly ONE `bear-case`
    thinking entry exists (07-16, the options spread). The 07-08 pivot's
    missing entry is self-logged as a mistake; owner-directed pivots (v4, v5)
    also carry none.

**Overall content grade:** far cleaner than the "assume it taught itself
noise" prior — the reflection skill's evidence discipline (repeat-gating,
regime tags, cite-the-dollars, maturity deferral) is genuinely practiced. The
corpus is also tiny (10 days), which means: nothing yet honestly qualifies as
"established" under the target thresholds, and remediation is nearly free if
done now.

## 6. Credentials & keys

**Clean.** Full sweep of working tree + git history: no literal API keys,
JWTs, AWS keys, or password-embedded connection strings anywhere; all secrets
are env-var references (`${{ secrets.* }}` in workflows, `EDGEFINDER_`-prefixed
settings); `.gitignore` covers `config/secrets.env` / `.env`; committed
examples are placeholders; history pickaxe for JWT/AWS/key patterns: no hits.
Two benign notes: owner contact email in `config/settings.py:33` (required SEC
EDGAR User-Agent, documented) and the Supabase project-ref in `.mcp.json`
(public identifier). Nothing in the wiki/journal/thinking content contains
credentials.

## 7. Summary against the 7 target outcomes

| Outcome | Status today |
|---|---|
| 1 Closed loop, traceable | **Loop verified live** (writes fire, `context` reads every cycle, behavior provably references lessons). Traceability is prose-only; claims outside the buy/add registry escape the loop (AAPL, ~$500). |
| 2 Epistemic tiers + pre-defined promotion | Tiers exist as prose conventions; two real gates in skill prose (repeat-to-generalize, 3-instance setups); no schema, no numeric thresholds fixed ex ante; one n=2 promotion already live |
| 3 Evidence discipline / provenance | Prose provenance consistently practiced; no structured links; no paper-derived-claim class; **no confidence floats (already compliant)** |
| 4 Hindsight firewall | Clean in practice, unenforced in schema |
| 5 Supersession + decay classes | History archival enforced in code (3 early revisions lost pre-feature); no decay classes; regime tags prose-only |
| 6 Tiered write-gate + owner approval | Observation/digest auto-append ✓; wiki gated on slug+size only; owner approval for learned-fact-driven behavior change exists culturally (v4/v5 pivots were owner-requested) but nothing enforces it |
| 7 Class separation | 6 fixed slugs approximate it; ops incidents and system-mechanics facts have no dedicated lifecycle |

**Deferred-item check (embeddings / vector / graph / ML-on-knowledge /
multi-agent):** nothing in this audit shows immediate need. Corpus is 5 pages
/ ~20KB with a hard 40KB cap; the whole store fits in one context read.
