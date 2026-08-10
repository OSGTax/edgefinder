# CUTOVER-V4 — the ordered runbook (V3 ledger → Alpaca paper account)

## EXECUTION RECORD — the cutover RAN on 2026-08-08 (UTC ~16:00–17:00)

> This section is the durable record of what actually happened; the
> numbered runbook below is kept as the reference it was executed from.

**Done (verified):**
- Step 1 — Supabase restored by the owner on a **Pro** account; same
  project, all data intact. (Size check recalibrated to the 8GB Pro
  quota in `agent/backup.py` — DB was 489MB.)
- Step 2 — Full export to R2: **32 tables, 1,390,770 rows, 0 failures**
  → `backups/2026-08-08/manifest-full.json`.
- Steps 3–5 — Old loop retired (the GH workflow file left main with the
  merge; the loop-monitor Routine disabled; the old daily trading
  Routine CONVERTED in place to the V4 hourly floor). Final V3 pass ran
  from an old-code worktree: bars topped up through Fri 2026-08-07,
  `settle` (booked one CCK dividend), `mark` (all 57 positions, live
  tier), `grade` (84 rows), then all **64 open outcome rows closed as
  `exit_kind='cutover'`** and the **ERA 1 FREEZE** journal note written.
- **E = final Era-1 equity = $94,877.79** (cash $9,113.85, positions
  $85,763.94 at Friday marks).
- Step 6 (rename) — executed via a Supabase migration
  (`v4_cutover_freeze_era1`, guarded/idempotent DO block):
  `desk_trades→era1_trades` (172 rows), `desk_positions→era1_positions`,
  `desk_equity→era1_equity` (348 marks). `desk_trades` now errors
  loudly, as designed. (The sandbox blocks port 6543, so the runbook's
  "direct DATABASE_URL" path was replaced by the Supabase MCP migration
  — same three ALTERs.)
- Step 9 (deploy) — main fast-forwarded `b175aeb → d1814de` (+ the
  `83a44e9` Pro-quota ops fix). Render serves **v10.0.0**; `/api/health`
  verified; `/api/desk/equity` serves the Era-1 curve; account panels in
  the honest degraded state pending trade keys.
- Routine roster: floor = `trig_01XG54FqViuXuA2xryjQPmYk`
  ("EdgeFinder trading floor (chain restarter)", `0 13-20 * * 1-5`,
  prompt "Run the trading-agent skill."); loop monitor
  `trig_01H4n1qmGrWxPp5KTc5BhYCn` disabled; data/lab/app-evolver
  triggers unchanged.
- Post-merge catch-up: full-market refresh on new main — 1,017-name
  universe, **274,528 bars** backfilled (the pause's dark week), R2
  synced (969), EDGAR +694 rows. The V4 nightly duties ran live for the
  first time: knowledge backup ok (26 tables), size RPC ok. One
  transient: the corp-actions pass hit an SSL handshake timeout —
  idempotent, re-covered by the next nightly.

**Steps 7–8 DONE (owner, Mon 2026-08-10 ~9:00 ET):** the paper account
was created fresh (new account + new keys — an Alpaca reset always mints
both) and funded at **exactly $94,877.79** — verified live via
`/api/desk/broker-health`: status ACTIVE, equity 94877.79. The env vars
were initially saved under wrong names in BOTH environments (missing the
`EDGEFINDER_` prefix — the settings loader requires it); fixed ~8:55 ET.
A first-light one-shot trigger fired 12:32 UTC while the names were
still wrong and stopped at the creds check, as designed — superseded by
the fix; its trigger (`trig_01NAFSLgRVLDouViosj31rQK`) is spent.

**Remaining (the first armed session runs these — ~3 minutes):**
1. `python -m agent.trade config` — apply `no_shorting=true` +
   `max_margin_multiplier="1"`, verify options Level 3. THE ACCOUNT IS
   UNCONFIGURED UNTIL THIS RUNS (a fresh paper account defaults to
   margin + shorting enabled; the charter's server-side enforcement
   starts here).
2. `python -m agent.trade probe --suite cutover` — journaled
   automatically as "V4 cutover probe results".
3. `python -m agent.trade snapshot` — Era 2's first
   desk_portfolio_history row.
4. Confirm `EDGEFINDER_STARTING_CAPITAL` (94877.79) matches the
   account's actual equity — it does, per the broker-health read above.

**Still owner, non-blocking:** the weekly-reflection Routine's prompt
swap (web-UI-created, agents cannot edit it; replacement text: "Run the
reflection-agent skill exactly
(.claude/skills/reflection-agent/SKILL.md). You are read-only on the
book — never trade submit/cancel/arm-stop. Run id reflect-YYYY-MM-DD.
Grade with agent.grade run / agent.grade outcomes, alpha not dollars,
then curate the wiki."), and deleting the dead GitHub repo secrets
(dispatch PAT, `SMTP_*`, `CLAUDE_CODE_OAUTH_TOKEN`).

**First light:** the floor Routine (hourly, `0 13-20 * * 1-5` UTC) runs
cycles from Mon 2026-08-10; any cycle before the config/probes ran is
research-only-by-circumstance and says so. The first ARMED cycle
re-enters only what the agent still believes from the 57-position Era-1
book, as fresh picks with prediction/horizon/kill, and arms protective
stops.

---



> One pass, in order. **[OWNER]** steps are manual (dashboard/UI work an
> agent cannot do). Everything else an agent session runs from this repo.
> Main keeps trading on the V3 stack until step 9 — the branch merge IS
> the cutover deploy.

## Pre-cutover (safe any time, reversible)

1. **[OWNER] Restore the paused Supabase project** (supabase.com dashboard
   → the project → Restore). Free tier is fine — V4 adds a nightly R2
   backup + a size check against the 500MB cap.
   Verify from a session: `python -m agent.preflight` on the rest lane
   (SUPABASE_* set) and, where TCP is allowed, the pg lane.
2. **Full archive export to R2** — the whole pre-migration database,
   immutable, before anything moves:
   `python -m agent.backup run --full`
   Verify the printed manifest: every table `ok`, row counts sane. This
   includes `desk_trades`/`desk_positions`/`desk_equity`, all knowledge
   tables, `daily_bars`, `ticker_news`, `fundamentals_pit`,
   `index_daily`, `fundamentals_snapshots`.
3. **[OWNER] Stop the old loop**: on GitHub → Actions → "Trading Agent" →
   Disable workflow. On claude.ai/code/routines: pause the fallback
   trading Routine and the hourly loop-monitor Routine. (Leave data/lab/
   reflection/app-evolver running.) The Render streamer stays up.
4. **[OWNER] Pause the app-evolver Routine** until step 13 — it pushes to
   main, and main is about to move under it.

## The freeze (the V3 book's last day)

5. **Final pass on the OLD ledger** (a session on MAIN, old code):
   `python -m agent.ledger settle` → `python -m agent.ledger mark` →
   `python -m agent.ledger grade`. Then close every still-open outcome
   row at the final mark (one-time, journaled):
   for each `desk_outcomes` row with `status='open'`, set
   `status='closed', exit_kind='cutover', mark_basis='exit'` keeping its
   last graded facts. Record the final marked equity **E** (from
   `agent.ledger state`) in a journal note titled "ERA 1 FREEZE".
6. **Freeze Era 1** — rename, so stale readers break loudly instead of
   reading a dead book (run via the store's SQL lane or the Supabase SQL
   editor; these renames are the ONLY sanctioned raw SQL of the
   migration):
   ```sql
   ALTER TABLE desk_trades    RENAME TO era1_trades;
   ALTER TABLE desk_positions RENAME TO era1_positions;
   ALTER TABLE desk_equity    RENAME TO era1_equity;
   ```

## The new account

7. **[OWNER] Create the Alpaca paper account** (app.alpaca.markets →
   paper → create/reset): **starting cash = E** (custom amounts are
   supported at creation/reset). Generate the paper account's **API key
   pair** — these are the TRADE keys.
8. **[OWNER] Set the secrets** everywhere the trading brain runs:
   `EDGEFINDER_ALPACA_TRADE_KEY` + `EDGEFINDER_ALPACA_TRADE_SECRET` in
   the Render service env AND every Claude Routine environment (alongside
   the existing data keys, Supabase pair, R2_*). Also set
   `EDGEFINDER_STARTING_CAPITAL=<E>` in the same places (the all-time
   P&L line stays continuous).
9. **Deploy**: merge the V4 branch → main. Render redeploys;
   `render_start.py` runs the idempotent DDL (mirror tables + the size
   RPC). Then, from any session with the new secrets:
   - `python -m agent.trade config` — applies `no_shorting=true`,
     `max_margin_multiplier="1"`, verifies options Level 3;
   - `python -m agent.trade probe --suite cutover` — the empirical
     checks the docs left open (client_order_id round-trip via a $1
     far-limit place-and-cancel, SIP entitlement on the trade keys,
     config read-back). Results are journaled to the desk.
   - `python scripts/smoke_dashboard.py` against the deployed app.

## First light

10. **First Era-2 cycle** (fire the trading skill manually once): the
    agent reads context + the Era-1 archive, decides what it still
    believes in, re-enters ONLY those at live prices as fresh picks with
    fresh prediction/horizon/kill, arms protective stops, and plans +
    arms the first chain wake. Anything not re-entered stayed closed at
    the cutover mark — that too is a call, and it is on the record.
11. **[OWNER] Create the Routine roster** (claude.ai/code/routines, all
    fresh-session, completion notifications push+email ON) per
    `docs/ROUTINES.md`: the hourly chain-restarter floor
    (`0 13-20 * * 1-5` UTC, prompt "Run the trading-agent skill."),
    nightly data (`45 0 * * 2-6`), strategy lab (`0 2 * * 2-6`),
    reflection (`30 22 * * 5`), app-evolver (`0 15 * * 6` — re-enable
    from step 4). Delete the old loop-monitor and fallback trading
    Routines.
12. **[OWNER] Decommission the old machinery**: delete the GitHub
    fine-grained dispatch PAT; remove `EDGEFINDER_GITHUB_DISPATCH_*`,
    `SMTP_*`, `CYCLE_REPORT_TO`, and `CLAUDE_CODE_OAUTH_TOKEN` from
    GitHub repo secrets (the workflow is gone); remove the same from
    Render env; delete the Vercel project if one exists.

## Verify (first 48 hours)

13. Watch for:
    - the chain self-sustains (each cycle's summary names the next armed
      trigger; `desk_wakes` plan-vs-honor lines up; the floor Routine's
      firings mostly exit on "chain healthy");
    - a real fill graded end-to-end (order → mirror → `desk_outcomes`
      row with entry/mark/alpha);
    - the nightly refresh reports `v4_nightly` complete (mirror sync,
      snapshot, split guard, backup `ok`, db_size `ok`);
    - the desk stitches both eras on the equity chart and shows the
      honesty strip (dividends disclosure + missed-dividends counter);
    - completion notifications arrive on the phone.
    Anything off: the journal + the Routine's session log is the trail.
