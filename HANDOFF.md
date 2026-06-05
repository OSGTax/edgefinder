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

**Wider walk-forward (v5.13.2): top-200, broadened param ranges, 5 rolling
folds + a SEALED 6-month holdout (2025-11-21..2026-05-26), search-iters 20.**
Bar = positive OOS Sharpe AND beats SPY in a majority of folds AND ≥30 trades.

| strategy | OOS ret | OOS Sharpe | vs SPY | folds>SPY | trades | holdout (Sharpe / vs SPY / trades) | meets bar |
|---|---|---|---|---|---|---|---|
| coward | −12.39% | −0.82 | −6.95% | 1/5 | 124 | −0.16 / −14.5% / 7 | NO |
| gambler | −10.35% | −0.47 | −6.32% | 2/5 | 88 | −1.17 / −24.9% / 43 | NO |
| degenerate | +43.67% | **+1.10** | **+4.08%** | **3/5** | **18** | **+1.97 / +2.15% / 8** | NO* |

**Plain answer: NO configuration meets the full bar.** coward/gambler fail
outright (Sharpe-negative, lose to SPY). *degenerate is the near-miss: it hits
positive OOS Sharpe (1.10) AND majority folds beat SPY (3/5) AND even passes the
sealed holdout (Sharpe 1.97, +2.15% vs SPY) — but on only **18 OOS trades** (and
8 in the holdout), below the 30-trade floor. The ≥30 bar exists precisely to
reject thin, possibly-lucky results, and 18/8 trades cannot distinguish edge
from variance. So it is NOT a validated edge — but it is the one signal worth a
targeted follow-up (degenerate's volume-spike entries benefited a lot from the
4× larger universe: top-50 → −15% FAIL, top-200 → this). Next: push its trade
count over 30 (more history, larger/looser universe) and re-test the SAME bar.

**Degenerate trade-count re-test (v5.13.3): extended history (backfilled
top-300 to 2021-06, ~5y incl. the 2022 bear), top-300, looser entry
(volume_spike_mult floor 1.25, rsi_min floor 35), 7 folds + sealed holdout,
holdout config fit on the trailing 252d.** Result — the walk-forward criteria
now ALL pass, but the SEALED HOLDOUT fails:

- OOS: +21.72%, **Sharpe +0.63**, **+0.66% vs SPY**, **4/7 folds beat SPY**,
  **77 trades** → criteria all_met = **TRUE** (trade-count bar finally cleared).
- **Sealed holdout: −10.13%, Sharpe −1.69, −24.0% vs SPY, 11 trades → FAILS.**

**Verdict: degenerate has NO robust edge.** The trade count went over 30 (77),
but the moment you seal the future and stop letting the optimizer peek, it
loses to SPY by 24% (Sharpe −1.69). And the holdout flips wildly by config:
top-200/S20 run → holdout **+16%** (8 trades); this run → **−10%** (11 trades),
SAME 6-month window. The walk-forward "PASS" is the optimistic number (per-fold
re-optimization is selection bias on short adjacent OOS windows); the sealed
holdout is the honest one, and it says no. Loosening the entry to raise trade
count made the holdout WORSE (+16% → −10%) — the extra trades were noise. This
is the "edge evaporates as trades rise → it was variance" outcome. **No
configuration of any of the three strategies is a validated, SPY-beating edge.**
Caveat: the holdout is one 6-month bull_calm window (small); but combined with
the sign-flip across runs and the selection-biased walk-forward, the honest
read is firmly negative.

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

## Update — 2026-06-05 EVE (strategy research round 1)

**Protocol** (committed before results): forensics on the failed three →
6 candidates → adversarial selection of 3 → PRE-REGISTERED defaults committed
to git (fa71fb9) → cheap screen (fixed defaults, dev window = all data minus
the sealed final 126d) → walk-forward folds → at most ONE holdout look/round.
New tooling: edgefinder/backtest/screen.py; --no-holdout-eval (seal without
burning); settings.live_strategies allowlist (candidates are lab-only).

**Round-1 results (top-300, dev region 2021-06..2025-11):**
- pullback_rider (21EMA-reclaim-in-uptrend): screen KILL — PF 0.80, −14.9%.
- turtle_adx (30d-high breakout + ADX): screen KILL — PF 0.66, W/L 1.13
  (the asymmetric-exit thesis didn't materialize).
- gap_drift (held gap-up continuation / PEAD proxy, survivorship control):
  - Screen PASS, strongly: +54.5% (≈SPY) at 53% exposure, PF 1.92, Sharpe
    1.12, max DD 8.4% (SPY −25% in 2022), median trade +$28, top quarter
    only 14.4% of gross P&L, ~flat through the 2022 bear.
  - Fixed-defaults folds (no optimizer): FAIL — Sharpe −0.29, excess −2.32%,
    3/7 folds. One fold (+19.4%, PF 3.77) carries; the effect is
    time-concentrated and relies on cross-quarter carry a fresh 63d account
    can't harvest.
  - Optimized folds (warm harness): criteria ALL MET and legacy PASS —
    +24.04%, Sharpe 0.22, excess +0.84%, 4/7 folds, 119 trades (recorded in
    validation_runs). BUT +0.84% is inside the survivorship/variance noise
    band, and defaults-fail/optimized-pass means the fold edge needs
    per-fold adaptation (possible selection bias — degenerate's signature).
  - **Holdout: NOT evaluated — still sealed.** The pre-agreed rule (holdout
    only if fixed defaults pass folds) held. gap_drift is PARKED: real
    effect in the dev aggregate, not yet harvestable to the bar.

**Lab bug #3 found & fixed (v5.17.1): cold-fold indicators.** Every prior
fold ran its window with NO indicator warmup (ema_200 None for whole 63d
OOS windows; RSI/MACD/BB dead for the first 3-5 weeks). Discovered when
gap_drift's defaults produced 0 fold trades vs 131 in the screen. Fix:
warmup_days=210 of preceding bars feed indicators/history; trading starts
at trade_start (live-faithful — live seeds history from daily_bars). All
pre-fix fold numbers (including the old three's) are understated/distorted;
verdicts stand on their margins.

**`validated` semantics tightened (v5.17.2):** a run with a sealed,
unevaluated holdout is criteria-passing but NOT validated on the dashboard.

**Round 2 directions (pre-registered intent, not yet run):** make
gap_drift's regime-adaptation explicit instead of optimizer-implied (e.g.
a per-ticker trend/vol state machine choosing gap thresholds), or test
holding-period/portfolio variants that capture cross-quarter carry; fresh
defaults, screen first; holdout stays sealed until a fixed config passes
folds.

## Update — 2026-06-05 PM (dashboard verifiability roadmap shipped)

v5.14.0–v5.16.0, all deployed + verified live the same day:
- **P1 Live Proof** (`edgefinder/analytics/live_scorecard.py`, GET
  /api/strategies/scorecard): the offline validation bar (Sharpe>0 AND beats
  SPY AND >=30 trades) applied continuously to LIVE data; PASS/FAIL card on
  the dashboard. All three strategies currently FAIL it — by design.
- **P2 validation_runs** table + GET /api/strategies/validation: every lab run
  persists its scorecard; dashboard shows offline verdict beside live evidence
  ("validated" = criteria.all_met AND sealed holdout passes). Seeded with the
  2026-06-03 results.
- **P4** trade timeline fixed (reasoning/indicators now serialized).
- **P5 hash chain v2**: chain now computed at persist time in TradeJournal,
  anchored to stored rows (v1 was unverifiable: in-memory anchor + discarded
  close hashes). verify_chain() + GET /api/trades/integrity + trades-page
  badge. Existing 8 trades = legacy (2 verify); new trades verify end-to-end.
- **P6 ops panel**: GET /api/ops/health + System Health card (heartbeat age,
  alerts, scheduler).

**Remaining roadmap: P3 promotion pipeline** (strategy_parameters → only
validated configs trade) — deferred until something passes validation.
Strategy research through the honest lab is the open frontier.

## Update — 2026-06-05 (Render incident root-caused + real URL)

- **The real live service is `https://edgefinder-pm8h.onrender.com`** (service
  `srv-d7agnd6a2pns73dg6qeg`, Starter $7/mo, always-on, Oregon, auto-deploy on
  `main`). **`edgefinder.onrender.com` is NOT ours** — onrender subdomains are
  global and the bare name belongs to some other Render customer (a Node stub
  whose `/api/health` masquerades convincingly). Never probe the bare URL.
- **June 3 deploys all failed fast (`update_failed`)**, not stuck: startup
  migrations crashed with `password authentication failed` against the pooler
  (`aws-1-us-east-1.pooler.supabase.com:6543`) — the Render `DATABASE_URL` had
  a bad credential at deploy time. The May-29 instance (v5.9.1, commit 86dd489)
  kept serving because env edits only apply to new deploys.
- **Consequence of serving v5.9.1:** the live engine ran with pre-v5.10 bugs —
  8 OPEN trades (May 28–Jun 2), 0 ever closed, degenerate ~100% concentrated in
  PNFP, no exits firing. The v5.13.x deploy activates enforced caps + working
  exits + the heartbeat.
- **Always-on is confirmed** (paid instance) ⇒ the **in-process scheduler is
  the production driver** (`intraday_external_driver` stays false). The cron
  cutover plan is shelved; `intraday-cycle.yml` stays dormant as break-glass.
  Enable only the detector: repo var `LIVENESS_ENABLED=true`.
- Render access for agents: `render` CLI (`render login` device flow; shim
  `xdg-open` to a no-op first or the CLI exits before polling). Workspace
  `tea-d70q01nkijhs73a20j1g`.

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
