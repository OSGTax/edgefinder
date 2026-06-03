# HANDOFF — trading-edge validation effort

Continuation notes for picking this work up in a Codespace (or any fresh
Claude Code session). The chat history doesn't transfer between environments,
but everything needed to continue is here + in git.

**Branch:** `main`  •  **Version:** 5.13.0
**Read first:** this file, then `CLAUDE.md`.

---

## Update — 2026-06-03 (validation answered + structural fixes #1/#2)

**Validation ran on real data. Verdict: all three strategies FAIL OOS — and
badly, once look-ahead is removed.** Completed the SPY/QQQ/IWM/DIA daily-bars
backfill (full 2023-05-30 → 2026-05-26, 750 bars each — benchmark gap closed)
and ran the walk-forward validator (top-50, 2-fold). The daily backtester used
to fill entries at the *same close that generated the signal* (look-ahead);
v5.13.1 fixes this to **next-day-open fills**. The honest OOS scorecards:

| strategy | total ret | Sharpe | vs SPY | folds>SPY | trades | win% |
|---|---|---|---|---|---|---|
| coward | −16.64% | −0.87 | −15.24% | 0/2 | 43 | 46.2 |
| gambler | −6.68% | −0.58 | −9.94% | 0/2 | 42 | 34.2 |
| degenerate | −15.30% | −1.45 | −14.46% | 0/2 | 13 | 32.5 |

For comparison, the look-ahead (same-close) numbers were +1.59% / +1.13% /
+14.24%. The look-ahead inflated returns by ~18–30 points; degenerate's
apparent "+1.05% vs SPY edge" was **entirely the bug** (now −14.46% vs SPY).
None show validated edge — they are net losers vs SPY. Caveats still stand
(2 folds, thin trade counts, fundamentals gate off, survivorship bias) but the
conclusion is now firmly negative, not marginal.

**Structural fixes built (v5.13.0), tested (460 pass), NOT yet cut over:**
- #1 Liveness watchdog + GitHub-issue alerts — `system_heartbeat` table +
  `check_cycle_liveness` in the watchdog + `edgefinder/agents/alerts.py` +
  `.github/workflows/liveness.yml`. Detects a stalled loop, pages via a
  GitHub issue, auto-closes on recovery. See CLAUDE.md "Cycle liveness" + alerts.
- #2 Cron-driven intraday loop (single driver) — `POST /api/admin/run-intraday`
  + `run_intraday_jobs` (single-flight) + `.github/workflows/intraday-cycle.yml`
  + `intraday_external_driver` flag; keepalive superseded. See CLAUDE.md
  "Live trading loop — cron-driven".
- **Cutover (all config, no code):** deploy (inert until opted in) → Render
  `EDGEFINDER_INTRADAY_EXTERNAL_DRIVER=true` + restart → repo vars
  `INTRADAY_CYCLE_ENABLED=true`, `LIVENESS_ENABLED=true`, `KEEPALIVE_ENABLED=false`.

**DB access from a Codespace:** `DATABASE_URL` must be the **pooler** host
(`postgresql://postgres.<ref>:<pw>@aws-1-us-east-1.pooler.supabase.com:5432/postgres`),
NOT the direct `db.<ref>.supabase.co` host — the direct host is IPv6-only and
unreachable from Codespaces. Tests run fine with `DATABASE_URL=` (SQLite).

**Next structural step (not built):** always-on Render worker to replace the
cron driver for real-money-grade reliability (sub-5-min, no cron drift,
event-driven exits). The cron-driven model is the interim.

---

## The goal (locked with the owner)
One or more strategies that make money, eventually **toward real money**. The
bar for "it works": **positive out-of-sample expectancy that beats SPY on a
risk-adjusted (Sharpe) basis, across multiple windows/regimes.** There are no
"sure-fire winners" — success = validated positive expectancy net of costs.

## Why we're here (the review finding)
The system had **0 closed trades / $0 realized P&L ever**. Priorities were
inverted: lots of operational machinery around three hand-written heuristic
strategies (coward/gambler/degenerate) whose edge was **never measured**, while
the backtester — the one tool that can prove an edge — was a disconnected UI
feature with no validation rigor or feedback loop.

Root causes of the stall (now addressed): in-process scheduler idles on Render
so jobs barely run; risk caps (max positions, concentration) were documented
but not enforced, so one trade locked the account; exits (fixed 20% stop / wide
targets) almost never fired so positions never closed; gambler needed prior-day
history that was never persisted.

## What's been done (commits this session)
- **v5.10.0** — seed indicator history from `daily_bars` so gambler works after
  restarts; `POST /api/admin/run-eod` + `.github/workflows/eod-jobs.yml` to run
  post-close jobs via external cron (engine idles otherwise).
- **v5.11.0** — Phase 1 mechanics: enforce concentration + max-position caps in
  the live path; capital-recycling exits (TIME_EXIT, TRAILING_STOP) with an
  injectable arena clock; `.github/workflows/keepalive.yml`.
- **v5.12.0** — Phase 2 validation lab: `edgefinder/backtest/optimize.py`
  (in-sample param search), `walkforward.py` (rolling IS→OOS, OOS scorecard,
  regime tagging, PASS/FAIL vs SPY), `validate.py` CLI, `validate.yml` workflow.
  Strategies are now parameter-driven (`SwingStrategy.configure()/_p()`);
  `ArenaEngine.configure_strategy()` applies a config + rebuilds risk/caps.
  Exit fills now include slippage (cost fidelity).
- **v5.12.1** — validator uses longest available SPY series (`daily_bars` ∪
  `index_daily`) for the benchmark.

All tests green: `pytest -m "not integration" --ignore=tests/test_market.py` (440).

## Known data gap — RESOLVED (2026-06-03)
~~`daily_bars` SPY had only 50 rows.~~ Backfilled SPY/QQQ/IWM/DIA to the full
2023-05-30 → 2026-05-26 range (750 bars each) via
`scripts/backfill_daily_bars.py`. The benchmark now spans the full history and
the validator produced verdicts (see the 2026-06-03 update above).

---

## Running it from a Codespace (this is the big advantage — direct DB access)
The committed `.devcontainer/devcontainer.json` installs Python + Node + the
package + `@anthropic-ai/claude-code` on create.

1. **Codespaces secrets** (GitHub → Settings → Codespaces → Secrets), scoped to
   this repo: `DATABASE_URL`, `EDGEFINDER_POLYGON_S3_ACCESS_KEY_ID`,
   `EDGEFINDER_POLYGON_S3_SECRET_ACCESS_KEY`, and (for Claude Code)
   `CLAUDE_CODE_OAUTH_TOKEN` (run `claude setup-token` locally to mint one).
2. Open the repo in a Codespace on branch `claude/magical-cannon-utlBM`.
3. Start a fresh agent: run `claude`, then tell it: *"Read HANDOFF.md and CLAUDE.md and continue the validation work."*

### Commands that work directly in a Codespace (DB reachable there)
```bash
# 1) Complete the benchmark backfill (fixes the SPY gap)
python -m alembic upgrade head
python scripts/backfill_daily_bars.py --symbols SPY,QQQ,IWM,DIA \
    --start 2023-05-30 --end 2026-05-26 --execute

# 2) Run walk-forward OOS validation and write reports
python -m edgefinder.backtest.validate --all --mode top --top-n 50 --write
#   -> prints OOS scorecard + verdict; writes reviews/validation-<strategy>-<date>.md

# 3) Tests
python -m pytest tests/ -q -m "not integration" --ignore=tests/test_market.py
```

---

## What remains (full checklist)
**Track 1 — get the validation answer (no Render deploy needed):**
1. Ensure Actions/Codespaces secrets: `DATABASE_URL`, `EDGEFINDER_POLYGON_S3_*`.
2. Run the daily-bars backfill for `SPY,QQQ,IWM,DIA` (2023-05-30 → 2026-05-26),
   `execute=true` (Actions: "Daily-bars backfill", or the command above).
3. Run validation (Actions: set var `VALIDATE_ENABLED=true` → "Strategy
   validation"; or the CLI above). Read the OOS verdicts.

**Track 2 — make live paper trading work day-to-day:**
4. Merge this branch → `main` so Render deploys the engine fixes and the cron
   workflows run on schedule.
5. Set `EDGEFINDER_EOD_TRIGGER_TOKEN` on Render and the same value as Actions
   secret `EOD_TRIGGER_TOKEN`.
6. Actions variables: `EDGEFINDER_URL`, `EOD_JOBS_ENABLED=true`,
   `KEEPALIVE_ENABLED=true`.
7. Verify after a market day: trades open AND close, realized P&L ≠ 0, no
   position >20%, ≤5 open positions.

## Next build (Phase 3 — not started): close the loop
Persist OOS-validated configs to the (currently empty) `strategy_parameters`
table; load the active config into the live arena at startup/scan via
`ArenaEngine.configure_strategy()`; only promoted, validated configs trade live.
A token-guarded `promote` step writes the winning config. This is the
backtest→live feedback loop the system still lacks.

## Honest caveats
- The validation hasn't been run on real data yet — first verdict may be FAIL
  for all three strategies. That's the valuable, honest answer we've been missing.
- Look-ahead via fundamentals: backtests run with the fundamental gate off
  (no point-in-time fundamentals) — validate the price/volume edge, treat
  fundamentals as a separate filter.
- Survivorship bias in the `daily_bars` universe may flatter results.
