# The knowledge layer on `/desk` — API reference

> **IMPLEMENTED in v9.19.0** (owner-directed dev session, 2026-07-20): the
> endpoints below plus the "Claims registry" card on `/desk`
> (`dashboard/templates/desk.html`, `static/js/pages/desk.js` `loadClaims`).
> This doc stays as the API reference; the app-evolver may still iterate on
> the card's presentation (its normal one-small-improvement lane), reading
> this contract.

## What to build

1. **`GET /api/desk/claims`** — a read-only projection of `desk_claims`
   (`dashboard/routers/desk.py`, same pattern as the existing `/api/desk/wiki`
   and `/api/desk/outcomes` handlers — `db.query(...).filter(account==ACCOUNT)`,
   no writes). Suggested shape:
   ```json
   {
     "claims": [
       {"id": 1, "cite": "[C-1]", "kclass": "risk_rule", "tier": "established",
        "experimental": false, "status": "active",
        "statement": "...", "scope": {"account":"paper","regimes":[...]},
        "stats": {"n": 5, "wins": 4, ...}, "decay_class": "never",
        "expires_at": null, "updated_at": "..."}
     ],
     "summary": {"active": 9, "by_tier": {"established": 5, "candidate": 2,
                 "observation": 2}, "by_class": {...}, "experimental": 0}
   }
   ```
   Order active first, then by tier (established → candidate → observation →
   digest), then id. Include `superseded`/`retired`/`quarantined` behind a
   `?include_inactive=true` flag (default false) so the supersession chain is
   inspectable without cluttering the default view.

2. **`GET /api/desk/proposals`** — read-only projection of `desk_proposals`
   (pending first). Fields: `id, title, change_kind, status, claim_ids,
   created_at, decided_at, decided_by, decided_via, expires_at`. Do **not**
   expose `payload` verbatim if it's noisy — a one-line summary is enough for
   the panel. The point is owner visibility: which learned-behavior changes
   are awaiting sign-off.

3. **A desk panel** rendering both: a "Knowledge" card (claims grouped by
   tier, each showing its `[C-n]` id, statement, regime tags, and `n`/win-rate
   from `stats` — never a confidence float, there are none), and an "Awaiting
   your approval" strip when any proposal is pending (title + `PROPOSAL-<id>`
   + change_kind). Keep it consistent with the existing dark-terminal design
   system; no CDN, no inline styles (repo rule).

## Data model notes (so the panel reads it honestly)

- **Tiers carry authority, not just labels.** Only `established` and
  `experimental`-flagged claims influence decisions; `candidate`/`observation`
  are watch-only. Make that visible (e.g. an "advisory" vs "in force" marker)
  so a reader doesn't mistake a candidate for a rule.
- **`stats` is recorded sample sizes, never self-assessed confidence.** Render
  `n`, `wins`/`losses`, `avg_alpha_pct`, `span`, `regimes` — the honesty
  contract forbids a confidence score, so don't invent a % "strength" bar.
- **Decay is real.** A `regime_conditional` claim has an `expires_at`; show it,
  and flag expired-but-active rows (lint already errors on these).
- **Provenance exists.** `evidence` is a list of typed refs
  (outcome/decision/trade/backtest/wiki_history/probe) — a "3 sources" pill or
  an expandable list closes the loop from a claim back to the trades that
  taught it.
- **Read the current numbers live** with `python -m agent.knowledge lint` and
  `loop-report` before designing the card — they tell you the real shape
  (currently 9 claims: 5 established, 2 candidate, 2 observation).

## Explicitly NOT in scope here

- No write endpoints. Approvals happen on GitHub (`PROPOSAL-<id>` issue) or via
  `agent.knowledge proposal-approve` — never from the web surface.
- No schema changes. Everything above reads existing `desk_*` tables.
- The loop-report can be surfaced later (a "loop health" tile) but is optional;
  start with claims + proposals.

## Reference

- Design: `SCHEMA.md` §1 (claims-as-source-of-truth), §5 (caps), §6 (approval).
- The read the trader already gets: `agent.knowledge context_claims` (tier-gated)
  and `agent.brain context` → `claims` / `commitments` sections.
- Existing router patterns to mirror: `dashboard/routers/desk.py` `wiki()`,
  `outcomes_scoreboard()`, `decisions_archive()`.
