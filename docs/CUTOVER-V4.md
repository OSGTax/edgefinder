# CUTOVER-V4 — the ordered runbook (V3 ledger → Alpaca paper account)

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
