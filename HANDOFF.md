# HANDOFF — trading-edge validation effort

Continuation notes for picking this work up in a Codespace (or any fresh
Claude Code session). The chat history doesn't transfer between environments,
but everything needed to continue is here + in git.

**Branch:** `main`  •  **Version:** 5.22.0
**Read first:** the CURRENT INITIATIVE section immediately below, then `CLAUDE.md`.
Everything under "Update — 2026-06-09 (MICROCAP BUILD…)" and earlier is the prior
research log — valuable history and honest negative results, but the project has
since PIVOTED (see below). Read it for context, not as the current plan.

---

## ⏱️ INTRADAY INITIATIVE — PHASE 1 (2026-06-12): minute-bar data layer

**Status: PHASE 1 COMPLETE (2026-06-12).** Store backfilled and verified:
52/52 menu symbols, 0 failed, 3,049 Parquet objects, **628 MB** (~25M RTH
minute rows, 2021-06 → now; ~6% of the R2 free tier). The resume-or-noop
path is proven in production (re-push of the EXECUTE flag scanned the
manifest and exited green in 2 min, topping up only the in-progress
month). Nightly minute-append is live in bars-nightly.yml. Frozen menu:
`intraday/menu.json` (52 symbols, mechanical criteria, committed before
any intraday backtest exists). No dashboard changes, no version bump
(data-layer only). **NEXT: PHASE 2 — the intraday backtest engine**
(next-bar fills, punitive intraday spread/impact cost model, session
handling, day-based folds over minute bars) — start it in a fresh
session from this note.

**Phases:** 1 data ✅ · 2 intraday backtest semantics ✅ · 3 intraday hunt
(roster pre-registered ✅; wave fired separately) · 4 live intraday loop ·
5 full-market streaming (NOT the pilot loader).

---

## ⏱️ INTRADAY INITIATIVE — PHASE 3 (2026-06-14): the pre-registered hunt

**Status: PHASE 3 ROSTER PRE-REGISTERED (2026-06-14, v5.54.0).** The intraday
hunt roster + matrix workflow are built and tested (667 tests green; +24 new),
committed with the queue IDLE — the roster code and FIXED parameters land
BEFORE any run (pre-registration is structural, not procedural). No
commit/push yet (`git add -A` staged per task scope; offline/stubbed
everywhere). The orchestrator fires the wave SEPARATELY by filling +
committing `intraday/queue.json`.

THE LANE (one config for the round): the frozen `intraday/menu.json`
(top-50 by trailing-126d dollar volume + protected ETFs, as of 2026-06-11);
`--costed` (FIXED intraday cost model); `--flatten-at-close` (no overnight
risk); decision every 5 bars; walk-forward vs SPY scored over the same
sessions; **sealed holdout pinned at `--holdout-start 2026-04-01`** (never
evaluated without `--burn-holdout` + owner sign-off); total-return bar
PRIMARY, risk-adjusted alongside; all-three-adversarial-re-checks before any
"finalist" claim. LONG-ONLY (no shorting in the engine): reversal/gap-fade
go LONG the losers/gapped-down names.

What exists (this phase):
- `edgefinder/engine/intraday_strategy.py` — added ONE accessor:
  `IntradayAssetView.opening_range(m) -> (high, low) | None` over THIS
  session's first m bars, look-ahead-free (slice `[session_start :
  min(i+1, session_start+m)]`; partial range before it forms).
- `edgefinder/engine/intraday_roster.py` — the pre-registration (module
  docstring = the statement). 10 specs in `INTRADAY_R1_SPECS`, all
  LONG-ONLY cross-sectional baskets (equal-weight top-K, K=5; `_topk_ew`
  helper; fewer than K signaling ⇒ partial; none ⇒ all-cash; always {} on
  `is_last_decision_bar`). Families:
  - REVERSAL (buy intraday losers): `ir_reversal` (most-negative ret(30)),
    `ir_vwap_rev` (most below VWAP, band 0.3%, needs ≥10 bars).
  - MOMENTUM (deliberate opposite): `ir_momentum` (most-positive ret(30)),
    `ir_high_break` (new session highs, ranked by ret(30), ≥10 bars).
  - BREAKOUT: `ir_orb` (price > opening_range(30).high, ranked by % above).
  - GAP: `ir_gap_fade` (fade open gap-downs ≥0.5%, first half-hour only),
    `ir_gap_go` (ride gap-ups ≥0.5% holding the gain, first hour only).
  - CLOSE-EFFECT: `ir_late_mom` (last-hour day-return winners).
  - CONTROLS: `ir_random_101`, `ir_random_103` — deterministic
    hash(seed, session_date, symbol) coin flips (the false-positive floor).
  (The `flat` null + `buy_hold_open:SPY` anchor already exist in
  `make_intraday_factory`; the queue includes them — not re-added.)
- `edgefinder/engine/intraday_validate.py` — `make_intraday_factory` now
  resolves any `INTRADAY_R1_SPECS` spec (after the built-ins, before the
  ValueError, which now lists the roster).
- `.github/workflows/intraday-batch.yml` — mirrors hunt-batch.yml EXACTLY
  (plan reads `intraday/queue.json` → matrix; run job max-parallel 4,
  fail-fast false, 3-attempt retry, R2_* + DATABASE_URL secrets, session-
  branch-only trigger on `paths: [intraday/queue.json]`) but runs
  `python -m edgefinder.engine.intraday_validate`.
- `intraday/queue.json` — `{"wave":"idle","runs":[]}`. **Committing a
  filled queue FIRES the wave.**
- Tests: `tests/test_intraday_engine.py` (+4: opening_range equals first-m
  hi/lo, partial before it forms, never reads future bars, respects
  session_start). `tests/test_intraday_roster.py` (+20): every spec builds /
  valid weights / flat-at-close / empty-universe-safe; reversal vs momentum
  pick opposites; ORB only after the range forms; gap_fade only on
  gap-downs + cash after the half-hour; HighBreak waits to settle;
  late_mom silent outside the last hour; RandomBasket deterministic per
  (seed, date) and seed/date-dependent; factory wiring + error message.

**NEXT: PHASE 4 — the live intraday loop** (once the round names finalists).

---

## ⏱️ INTRADAY INITIATIVE — PHASE 2 (2026-06-14): the intraday backtest engine

**Status: PHASE 2 COMPLETE (2026-06-14, v5.53.0).** The honest gate every
intraday strategy idea faces is built + tested (643 tests green; +18 new,
+1 stale PHASE-1 menu test corrected — it asserted the menu was still
unfrozen, but PHASE 1 froze 52 symbols). No commit/push yet (`git add -A`
staged per task scope; offline/stubbed everywhere).

What exists (this phase):
- `edgefinder/engine/intraday_strategy.py` — `IntradayAssetView` (frozen,
  backed by numpy arrays + a current index for O(1) per-bar access — NEVER
  slices the full history, so a 390-bar session stays O(n) not O(n²)),
  `IntradayContext` (ts/session_date/minute_of_day/bars_since_open/
  bars_until_close/is_last_decision_bar), `IntradayStrategy` Protocol
  (`decide(ctx) -> weights`), and reference strategies `IntradayFlat`,
  `BuyHoldFromOpen(sym)` (per-session anchor), `IntradayMeanReversion`.
- `edgefinder/engine/intraday_backtest.py` — `run_intraday_backtest`. Steps
  bar-by-bar over the unified minute calendar but emits a DAILY equity curve
  (one (session_date, equity) point per session close) so Sharpe annualizes
  sqrt(252) and reuses the daily aggregate — directly comparable to SPY.
  Honesty invariants copied EXACTLY from engine/backtest.py: decision through
  bar i, fill at bar i+1's open, cost context read strictly before the fill
  bar, sells before buys, integer shares, weights>1 scaled, trade_start_day
  gates trading AND marks. Reuses CostModel (FIXED N=30 trailing bars for
  ADV/vol, corwin_schultz_spread on the two bars before the fill). Also
  reports `intraday_max_drawdown_pct` (from intra-session lows) and
  `avg_trades_per_day`. Assert-guards raise on any fill_idx<=decision_idx.
- `edgefinder/engine/intraday_walkforward.py` — `run_intraday_walkforward`.
  Plans folds on the trading-DAY calendar via the EXISTING `plan_folds`,
  replays each fold with the intraday engine, tags regimes from daily SPY
  (derived from minute session closes), and feeds the daily `_aggregate` so
  the scorecard is byte-shape-identical to a daily one (criteria/verdict/
  holdout/by_regime free). Adds config engine=intraday, bar=1min,
  flatten_at_close, decision_interval.
- `edgefinder/engine/intraday_validate.py` — CLI mirroring engine/validate.py:
  `--strategy {flat|buy_hold_open:SYM|mean_rev:SYM[:lookback:z]}`,
  `--symbols`|`--menu`, `--start/--end`, `--bars-from r2` (MinuteStore),
  `--costed`, `--flatten-at-close/--hold-overnight`, `--decision-interval`,
  fold/holdout flags, `--record` (reuses engine/record.py with the hardened
  4-attempt/SystemExit(3) retry). SPY benchmark from the minute store.
- Tests: `tests/test_intraday_engine.py` (18) — synthetic RTH minute frames,
  no look-ahead (cheater earns no excess), the BuyHoldFromOpen anchor
  (daily return == open->close; costed == 1 entry + 1 MOC/day), next-bar
  fill on a hand-built frame, toll bleed (churn < trade-once; flat exactly
  flat), session counters + flatten/hold-overnight, daily-curve + intraday-dd
  >= close-to-close dd, walkforward shape == daily + holdout + determinism,
  and a performance guard (5 syms × 10 sessions = 19,500 bars in well under
  10s — proves no O(n²) slicing).

KEY JUDGMENT CALLS (carried for PHASE 3+):
- **flatten-at-close MOC proxy:** on each session's last bar the engine forces
  target {} and sells at THAT bar's CLOSE with tolls (the one place a fill
  uses the same bar's close). Justified: MOC is a real order type, and the
  alternative (gap out at next open) would let an overnight move flatter a
  flat-by-design strategy's daily return. Documented in the module docstring.
- **bars_until_close from the ET CLOCK** (minutes to 16:00 / bar interval),
  NOT by counting future bars — so it's live-replicable and not look-ahead.
- **O(1) history:** every AssetView accessor is a numpy slice/scalar read of
  a pre-built whole-window array at the current index; no per-bar DataFrame
  copies (the performance-guard test enforces this).

**NEXT: PHASE 3 — the intraday hunt.** Pre-register an intraday strategy
roster (a hunt_intraday.py module, committed BEFORE the first run, same
discipline as the daily hunt_r{1..4}: roster fixed, a flat null control per
batch, all-three-adversarial-re-check standard, holdout sealed at
2026-06-11). Run it via a flag-gated push-triggered workflow (mirror
hunt-batch.yml) against the frozen `intraday/menu.json` over R2 minute bars.

What exists (PHASE 1):
- `intraday/menu.json` — the FROZEN pilot menu (pre-registration: the
  criteria string — top-50 by trailing-126-trading-day mean dollar volume
  as of 2026-06-11, resolve_universe semantics, + the 10 protected ETFs —
  is committed with `symbols: []`; the first real MENU-mode run prints the
  list and the orchestrator commits it).
- `scripts/resolve_intraday_menu.py` — computes the menu from the DAILY
  store/DB (`engine/data.resolve_universe` + `trailing_rank_start`, SPY
  calendar).
- `edgefinder/data/minutestore.py` — minute-bar R2 store:
  `minute/{SYMBOL}/{YYYY-MM}.parquet` (ts = UTC epoch seconds int64,
  OHLCV), **RTH bars only** (09:30–15:59 ET bar-starts), sorted/deduped,
  merge-only grow-only sync (same discipline as the daily barstore),
  `minute/_manifest.json` with per-month rows/min_ts/max_ts/`complete`,
  `verify()` proves manifest == objects both directions. Pilot budget:
  ~60 symbols × ~390 bars/day × ~1,260 days ≈ 30M rows ≈ ~2 GB.
- `scripts/backfill_minute_bars.py` — RESUMABLE backfill from Polygon
  REST minute aggs: month-aligned 2-month chunks (worst-case extended
  session 2×23×960 = 44k bars < the 50k REST cap; a maxed response fails
  loudly), only manifest-`complete` months are skipped on rerun (a partial
  nightly top-up can never mask a hole), per-symbol failure logs and
  continues but the run **exits non-zero if any symbol is incomplete**.
- `.github/workflows/minute-backfill.yml` — push-triggered on
  `ops/minute-backfill.flag` (currently `dry`): `MENU` → resolve+print the
  menu; `EXECUTE` → full backfill from the committed menu; anything else →
  `--dry-run` plan. `bars-nightly.yml` gained a `minute-append` job
  (trailing 5-day RTH append for the menu; skips green while the menu is
  empty or secrets are missing — same convention as the daily job).
- Tests: `tests/test_minutestore.py`, `tests/test_minute_backfill.py`,
  `tests/test_intraday_menu.py` (29 tests, stubbed S3/client, no network).

Go-live sequence (owner/orchestrator): flag `MENU` → commit the printed
symbols into `intraday/menu.json` → flag `EXECUTE` (resumable; re-flip on
failure) → nightly append self-enables once the menu has symbols.

---

## 🎯 CURRENT INITIATIVE (2026-06-09) — CLEAN ENGINE REBUILD + GENERAL WORKBENCH

> **If you are a fresh session asked to "continue" or "finish the project
> autonomously," THIS is the plan. Resume at Phase 2 (below).**

### The goal (owner's vision — this is what matters most)

The owner (Mike, a Peter Lynch fan) wants **a permanent trading research lab**, not
a single strategy:

1. **A permanent, growing database of stock history + live data.** It just *exists*
   — it is the asset, the foundation. **Never delete it.** We build on it.
2. **A fully general workbench:** throw in *any* strategy idea — technical,
   fundamental/GARP, or **deliberately dumb** ("buy the top-traded stocks starting
   with B every Tuesday") — have it **honestly backtested**, then **auto-promoted to
   self-running paper trading**, and **monitored on the dashboard**. Drop in an idea;
   the machine does the rest.
3. **A wide, ongoing search** — typical or weird — for *anything that consistently
   beats the market.*

**Peter Lynch ethos — this CORRECTS the prior failure mode.** The market *can* be
beaten by curiosity and breadth. Do NOT get fatalistic. The research log below
concludes "very hard to beat SPY net of cost" — but that hunt ran on an OLD engine
with **at least 4 trading-sequence bugs** (wall-clock cooldowns, phantom stop on a
bad print, daily churn, wall-clock timestamps), each of which silently corrupted
results. So some "strategy failures" were really PLUMBING failures. And the
**fundamentals table (earnings growth, PEG) has never had a strategy built on it** —
the most on-brand Lynch lane is entirely unexplored. Go wide; stay curious.

### The rebuild decision

The owner authorized a **clean-slate engine**. Build a backtest core where the
trading sequence is **correct by construction and fully tested**, so a strategy
failure is the strategy's, not a bug's. Keep the old engine only so the live system
runs untouched until the new one is proven. Old engine is otherwise not important.

### Design principles (established in Phase 1)

- **A backtest is a PURE function:** `(bars, strategy, costs) -> equity curve`. No
  account state machine, no cooldown/PDT/revenge, no wall-clock — those are exactly
  where the old bugs lived, so they are absent here.
- **One general strategy interface:** `rebalance(ctx) -> {symbol: target_weight}`.
  The strategy sees the WHOLE universe point-in-time (every symbol's bars,
  indicators, fundamentals, the date) and returns the portfolio it wants. Weights in
  `[0,1]`; their sum is the invested fraction (rest cash). Expresses technical,
  fundamental, cross-sectional, calendar, or dumb strategies IDENTICALLY — and a
  strategy CANNOT have a trading-sequence bug (order mechanics live in one tested
  place).
- **Correct by construction:** point-in-time decision (data through *yesterday*) →
  next-open fill → realistic costs (bps of traded notional) → mark-to-market. Bad
  OHLC sanitized; delisted holdings closed at last real price.
- **Honesty is the foundation:** survivorship correction, real costs, point-in-time,
  walk-forward folds + sealed holdout. Every result must be trustworthy enough to
  auto-promote to paper trading.

### PHASE 1 — DONE ✅ (committed `48771df`, v5.22.0; additive, 577 tests pass)

- `edgefinder/engine/strategy.py` — general interface: `AssetView`,
  `RebalanceContext`, `Strategy` protocol, reference `BuyAndHold` / `EqualWeight`.
- `edgefinder/engine/backtest.py` — pure `run_backtest(...)` + `BacktestResult` +
  the single tested trade loop.
- `tests/test_engine.py` — 8 correctness tests incl. the buy-and-hold anchor.
- **Validation:** `BuyAndHold(SPY)` = +523.1% vs SPY actual +523.9% over 21 yr
  (matches to a fraction of a %, same Sharpe 0.55, same maxDD 56%) → sequence is
  provably right. First real strategy beats SPY risk-adjusted: equal-weight 7 ETFs,
  monthly → Sharpe 0.64 vs 0.55, maxDD 40% vs 56% (incl. 2008+2020). **Caveats:**
  lags raw return (+413% vs +524%), modest +0.09 Sharpe edge, known diversification
  effect, NOT yet through the fold + sealed-holdout gate.

### PHASE 2 — DONE ✅ (v5.23.0 `88147b8` + v5.23.1 `9053382`; 599 tests pass)

Built: `engine/walkforward.py` (rolling IS→OOS folds + sealed holdout + both criteria
modes, fixed pre-registered params only — no per-fold optimizer), `engine/data.py`
(the one DB→engine seam; Phase 4's R2 store slots in behind it), `engine/record.py`
(validation_runs recording, engine:'v2' tagged), `engine/validate.py` (reproducible
CLI: `python -m edgefinder.engine.validate ...`; burning the holdout requires explicit
`--burn-holdout`), `engine/strategies.py` (dual_momentum_v2 + trend_timer_v2 ported
verbatim from their pre-registrations), and a `trade_start` engine affordance (warmup
bars feed indicators only — no trading, no equity marks).

**A 13-agent adversarial review then de-biased the harness itself** (v5.23.1): the
fold geometry carried a ~−1.8pp/fold structural handicap (fold-start dead-cash gap +
int-share flooring at $10k vs a frictionless day-1-invested benchmark) — nearly the
whole initially-reported deficit. After fixes, the **null control (buy_and_hold:SPY
vs SPY) reads 0.00 excess Sharpe / −0.02pp** — the instrument is honest to ~2bps.
Also fixed: all-cash folds now count in the Sharpe-majority denominator (imputed
excess = −benchmark_sharpe, symmetric), sealed-holdout dates pinned in the record,
deterministic fill order, open-anchored benchmark return.

**Clean-engine verdicts (21yr ETF lane, 38 folds incl 2008+2020, holdout SEALED):**
- `equal_weight` etf7 monthly: −0.04 excess Sharpe, 19/38 folds, +2.66pp dd cut →
  **FAIL by a hair** (the Phase-1 "beats SPY" full-period claim does not survive fold
  granularity; most of the earlier −0.18 deficit was harness artifact). SPY-equivalent
  with a drawdown cushion — closest candidate yet.
- `dual_momentum_v2` monthly: −0.34, 11/38, +3.03pp dd cut, worst fold dd 16.75% →
  **FAIL**; crash protection real, Sharpe edge absent (matches the old-engine verdict).
- `trend_timer_spy_v2` daily: −0.60, 2/38 → **decisive FAIL**.
- **The rebuild's motivating question is answered:** the old engine's 4 bugs
  exaggerated deficits but flipped no verdict. The calibrated instrument is the win.
- Engine note: daily-schedule runs re-true to exact weights every day (dual_momentum
  daily = 13k fills of churn). A no-trade band is a sensible future engine feature;
  monthly cadence is the canonical workaround meanwhile.

### ⚠️ SCOPE (owner directive, 2026-06-09): build and verify the MACHINE only

Phases 3 and 4 plus the PIT fundamentals store are in scope. **The Phase-5 wide
strategy hunt (batch-testing new strategies, the Lynch/GARP screens, dumb-strategy
sweeps) is explicitly OUT OF SCOPE until the owner says go.** A major dashboard
overhaul is planned AFTER the mechanics are solid — defer frontend work and park
consolidation debt for that phase (heartbeat-upsert duplication, lab/live
_is_rebalance + execute-math duplication, the validated-rule living in both
promote.py and the strategies router, SLIPPAGE_BPS(5bps live) vs cost_bps(2bps
lab) disclosure).

### PHASE 3 — DONE ✅ (v5.24.0 `12742bf` + review fixes v5.24.1 `61bde2a`; 608 tests)

The owner's loop, closed and verified live: **promote → self-running paper
trading → dashboard tables.**
- `promoted_strategies` table + `engine/promote.py` CLI (tier `validated` requires
  criteria.all_met AND a passing sealed holdout — the dashboard's exact rule;
  `experimental` is the explicit throw-it-in-and-watch tier; demotion deactivates).
- `engine/live.py` — daily portfolio runner: SAME decision semantics as the
  backtest engine (shared prepare_bars/_build_context path, decide on data through
  yesterday, fill at today's price +5bps); schedule-gated with forced first-cycle
  fill; 1%-of-equity no-trade band; partial sells split lots (remainder reopens at
  ORIGINAL entry); cash recomputed from trades every cycle (CLAUDE.md formula);
  bar-freshness step appends missing daily_bars (the data asset grows itself);
  v2 accounts start at **$100k** (a $5k book can't hold 1 share each of 7 modern
  ETFs — SPY ~$739). Scheduler job 9:45 AM ET + `v2_portfolio_cycle` heartbeat.
- **Verified against prod:** `equal_weight` + `dual_momentum_v2` promoted
  (experimental), seeded — integrity exact to the cent; correct holds on
  non-boundary days; daily_bars backfilled 05-27→06-08 by the freshness step.
- 7-angle review pass applied (v5.24.1): promoted_strategies DDL in render_start
  (create_all is SKIPPED on Render — new tables MUST be added there), cache-bypass
  `get_bars_fresh` for the freshness step (18h TTL + range-ignoring cache served
  stale frames), ET-not-UTC trading day, gate-before-prep, per-cycle price memo.

### ROADMAP — resume here
- **PHASE 4 — DONE ✅ (v5.26.0 `afabd6a`).** `data/barstore.py`: Parquet-per-symbol
  on Cloudflare R2, incremental manifest-driven sync, parallel uploads. **Full
  export done + verified: all 8,651 symbols / 4.51M rows / 179 MB** (the 1.3 GB
  table, ~7x compressed); `verify -n 50` = zero mismatches; the 21-yr null control
  run `--bars-from r2` reproduces the DB calibration exactly (0.00 / −0.02pp).
  Read path: `engine/data.load_bars_from_store` + validate `--bars-from r2`.
  Nightly mirror: `r2_sync` job 7 PM ET, self-enabling on R2_* env presence —
  **add the four R2_* vars to Render env to turn the nightly mirror on** (it
  logs-and-skips without them). The DB is never deleted from. R2 secrets live in
  GitHub → Codespaces → Secrets; values are .strip()ed in code (a pasted secret
  carried a trailing newline once).
- **PHASE 5a — DONE ✅ (v5.25.0 `61c8431` + backfill v5.27.0 `cd274df`).**
  `fundamentals_snapshots` table + `data/pit_fundamentals.py` (snapshot writer
  hooked into the nightly scan; `PITFundamentals.asof(symbol, date)` reader;
  None before coverage). Engine `_build_context` accepts static dict (disclosed
  look-ahead) OR any object with `asof()` (the honest path). **HISTORICAL
  BACKFILL DONE + VERIFIED: 128,854 PIT snapshots / 7,702 symbols /
  2019→now from Polygon SEC quarterly filings, keyed by FILING date
  (conservative SEC-deadline imputation where missing), including 2,060
  graveyard (delisted) symbols** — survivorship-free fundamental backtests
  over 2021-06→now are now POSSIBLE (bounded by stock bars, not fundamentals).
  Stored facts are price-independent (TTM EPS, earnings/revenue growth, D/E,
  ROE, ROA, current ratio); strategies compute P/E and PEG at decision time
  from `AssetView.price`. Daily scan snapshots keep accruing on top.
  Also shipped: `--holdout-start` (fixed-DATE sealed holdout — pins the
  boundary so it can't roll forward as bars accrue; use it for any multi-run
  research program).

### Stock-universe machine — DONE ✅ (v5.28.0 `233e271`; 35-agent review; 647 tests)

All three former gaps closed and verified:
1. **PIT universes**: `--universe top:N[+OFFSET] --start ...` — per-window
   top-N resolved as of the day before each window's first scored day; ONE
   planning calendar (CLI passes it via `calendar=`; resolver misses fail
   loud). **Bias demo: same strategy, PIT menu +0.17pp vs future-selected
   menu +5.79pp — ~5.6pp/fold of selection bias stripped.**
2. **Realistic costs**: `--costed` — Corwin-Schultz spread + sqrt impact +
   participation caps, PIT stats, **liquidity-tiered FIXED spread floors**
   (2bps ≥$1B ADV … 50bps microcap), costed delist liquidations, no negative
   fills, collapsed-liquidity holdings freeze (never force-dumped).
3. **Total return**: `--div-adjust` — dividends table (167k records / 3,443
   symbols incl. graveyard) + CRSP back-adjustment at load time (raw bars
   immutable in DB+R2); declared-future dividends ignored; ADV from RAW
   closes; SPY-dividends-missing hard-fails; coverage recorded; prices basis
   stored at promotion (live trades raw — disclosed divergence for yielders).
   **TR null control: 0.00 / −0.02pp. SPY 21-yr OOS 253% → 566.7% (~2.1%/yr
   yield — checks out).**
   Known follow-ups — ALL CLOSED by the 2026-06-10 loose-ends round (see
   the LOOSE-ENDS POLISH section below): live dividend cash credits built
   (v5.31), phantom index_daily rows deleted + holiday gate added (v5.30),
   resolve_universe trailing-window ranking shipped as the hunt default
   (v5.32).

### 🔬 MARKET-FIDELITY AUDIT — PASSED (2026-06-10, v5.29.0 `98e940e`)

A 14-agent audit checked the machine against PUBLISHED market reality
(every lens re-verified with independent sources):
- **ETF history 12/12 penny-exact** (GFC −56.47% peak-trough, COVID −34.10%,
  famous closes to the cent; the 2021 Yahoo/Polygon splice is seamless).
- **Graveyard 7/7 exact**: SIVB dies 2023-03-09 @ $106.04, FRC, BBBY @
  $0.0751, TWTR, ATVI, SBNY — all on the real last trading day, no OTC
  afterlife leakage. Survivorship-bias protection verified.
- **Dividends/TR 5/5**: SPY dividends match to 5-6 decimals; the TR transform
  reproduces published SPY calendar total returns **within 0.01pp**.
- **PIT fundamentals**: TTM EPS matches filings to pennies; snapshot timing
  errs only LATE (conservative) — zero look-ahead found.
- **Engine**: buy-and-hold identities <0.01pp; all three nulls 0.00; delist
  mechanics exact; 7-ETF monthly cost drag single-digit bps/yr.
- **ONE systemic bug found and FIXED (v5.29.0): the stock lane was not
  split-adjusted** (fake ±60-99% cliffs at ~256 split ex-dates; NVDA lane
  read −68% instead of +1,181%). Now: market-wide ticker_splits (9,884
  rows) + load-time adjustment in load_bars (default ON), dividend amounts
  scaled across splits, META pre-2022-06-09 contamination quarantined.
  Verified: TSLA 2022 −65.0%, AMZN −49.6%, NVDA +1,181% — all exact.
  **All committed ETF-lane verdicts stand (no ETF splits in range); any
  earlier ad-hoc single-stock numbers spanning splits are void.**
- Recorded caveats: BBBY = two companies (gap-separated; PIT universes are
  safe, fixed-symbol runs beware); SPY dividends start 2007 (pre-2007 ETF
  TR = price-only — owner chose to accept this). The other three caveats
  (bulk-edge staleness, missing TLT 2015-08 dividend, phantom 2026-05-25
  index_daily rows) were closed by the 2026-06-10 loose-ends round below.

### 🧹 LOOSE-ENDS POLISH — DONE ✅ (2026-06-10, v5.30–v5.32, branch `claude/handoff-doc-review-176vbl`)

Owner-directed cleanup of every named loose end EXCEPT the hunt (explicitly
deferred) and the consolidation debt (owner chose to keep it parked). Built
from a claude.ai cloud sandbox that cannot reach Postgres directly — DB work
ran via the supabase MCP and temporary push-triggered Actions jobs.

- **Phantom 2026-05-25 index_daily rows DELETED** (owner-approved) and the
  cause fixed: `BenchmarkService.collect_daily` now has a weekend + Polygon
  market-holiday gate (service-level, so the manual `/api/benchmarks/collect`
  route is covered too; fails open on API errors).
- **Dividend cash credits for v2 paper accounts (TR live parity), v5.31.**
  New `dividend_credits` table (one row per strategy/symbol/ex_date; DDL in
  render_start). `engine/live` credits cash when an ex-date passes while lots
  are held (hold days included; missed cycles self-heal; splits can't
  double-credit), tops up the dividends table per cycle, and the integrity
  formula extends to `cash = start + closed P&L + credits − open cost`
  (CLAUDE.md updated; watchdog cash-drift check uses the same formula, so
  credits never read as drift). Also fixed en route: live trades were
  stamped with WALL-CLOCK time even on simulated `--date` cycles (same bug
  family the rebuild killed) — now stamped with the cycle's trading day.
- **Trailing-window universe ranking, v5.32 (the resolve_universe caveat).**
  `resolve_universe(rank_start=...)` + validate `--rank-window N` (default
  126 trading days; 0 = legacy lifetime for reproducing old runs). Disclosed
  in labels (`top:N@pit,rw126+v2`) and scorecard config.
- **Backtest no-trade band, v5.32.** `run_backtest(rebalance_band=F)` /
  validate `--rebalance-band` — the live runner's 1%-of-equity dust guard as
  an opt-in engine feature (opens/closes always exempt; default 0 = exact
  re-true, bit-identical to all committed results). Kills the daily-schedule
  13k-fill churn artifact when enabled.
- **Cost disclosure where live and lab meet:** structured `cost_disclosure`
  on `/api/strategies/scorecard` + `/validation` payloads and the Live Proof
  card footer (live ~5bps/side vs lab 2bps flat / costed model).
- **TLT 2015-08 dividend:** confirmed missing from Polygon itself; official
  value $0.267 (ex 2015-08-03) verified against two independent sources;
  `dividends_backfill add` subcommand built for such patches. Insert pending
  owner approval (sandbox blocks agent-initiated prod writes of web-sourced
  values).
- **Stock-bar bulk edge advanced:** 2026-05-27→2026-06-09 ingested
  (top-1000/day, active+delisted + benchmark ETFs — same survivorship-free
  methodology), rows computed from flat files and loaded via a temporary
  Actions job. Go-forward automation: `.github/workflows/bars-nightly.yml`
  (nightly trailing-5-day self-healing catch-up, kill switch
  `BARS_NIGHTLY_ENABLED`, requires `EDGEFINDER_POLYGON_API_KEY` in Actions
  secrets). The manual backfill workflow gained a `top_per_day` input.
- **engine-validate.yml** dispatch workflow: run any v2 validation lane from
  Actions (the hunt's lanes can run there too; sealed-holdout discipline
  noted in the workflow header).
- **CALIBRATION RE-VERIFIED on v5.32 (validation_runs 27/28):** null control
  **0.00 excess Sharpe / −0.02pp over 38 folds — identical to the committed
  calibration.** PIT lane re-measured with the trailing window: +1.04pp /
  −0.04 excess Sharpe / FAIL (vs +0.17pp lifetime-ranked id 26; both at
  honest-noise scale, ~5pp below the measured future-selection bias). The
  shift is menu composition (currently-liquid names), not look-ahead — rank
  windows only see data through each as-of day.

**Closed with owner approval in-session (2026-06-10):** TLT dividend
inserted (dividends id 167005), `dividend_credits` pre-created in prod,
branch merged to main (Render deploys v5.32).

**Owner actions: NONE remaining.** All secrets are placed everywhere they're
needed (R2_* in Codespaces + Actions + Render; EDGEFINDER_POLYGON_API_KEY in
Actions; nightly bars catch-up and R2 mirror both self-enabled and verified).

### 🗄️ TWO-TIER STORAGE OPERATIONAL (2026-06-10, v5.35 — owner-approved "Option B")

Supabase free tier was at 2.7x its 500MB cap (1.36GB; `daily_bars` alone
1.26GB) and throttling (statement timeouts, dropped connections). Fixed
architecturally, no paid tier:

- **R2 = the permanent asset, GROW-ONLY.** `barstore.sync` now MERGES
  (parquet = R2 ∪ DB, DB wins on conflicting dates, shed rows preserved) —
  a sync after a DB prune can never shrink the mirror. The manifest carries
  a separate `db_rows/db_max` fingerprint for change detection; `verify`
  proves DB ⊆ R2 (subset). Known limitation: a pure in-place VALUE edit
  doesn't move the fingerprint — force with `sync --symbols X`.
- **DB = the operational hot set (184MB total, was 1.36GB).** `daily_bars`
  rebuilt to 318,224 rows / 69MB: 24 protected symbols full-history (deep
  ETF menu SPY/QQQ/IWM/DIA/GLD/TLT/EFA/AGG/LQD/HYG + index symbols +
  promoted universes + open-trade names) + trailing-365d per-day top-1000.
  One-shot tool: `scripts/slim_daily_bars.py` (dry-run default, R2-currency
  gate, executed via `.github/workflows/db-slim.yml` flag file).
- **Self-maintaining:** the 7PM-ET Render job is now sync→prune
  (`prune_db`: age-based, fingerprint-guarded, protected set exempt), so
  the DB stays flat as the nightly ingest appends.
- **Deep/breadth backtests read R2:** `validate --bars-from r2` now works
  WITH `--universe` (full-market store load + in-memory PIT ranking,
  parity-tested vs the SQL path). **Equivalence proven: the xsec_mom_12_1
  top-500 run reproduces from R2 bit-identically** (validation_runs ids
  45=75; re-verified post-slim). The microcap band (top:2000+1000) MUST
  use `--bars-from r2` now — those rows live only in R2.
- Verification trail (2026-06-10): mirror catch-up 1385/1385 + verify 50/50
  clean → equivalence exact → dry-run counts → EXECUTE (owner go) →
  318,224 rows kept exactly as projected → post-slim null control + R2
  equivalence re-run clean.

### 🏹 HUNT ROUND 1 — COMPLETE (2026-06-10, v5.33; full report `reviews/HUNT-ROUND-1.md`)

35 pre-registered candidates (engine/hunt_r1.py, commit f5feb12) through
fixed-param folds, PIT universes (trailing rw126), costed, TR; sealed
holdout 2025-12-05 NEVER evaluated. Controls in-batch: null 0.00/−0.02pp,
menu control −3.3pp (no tilt), dumb sweep 0/6 false positives (noise floor
≈ −7pp).

**Scoreboard: 1 confirmed finalist + 1 borderline.**
- ✅ `xsec_mom_12_1` (12-1 momentum, trend-gated, top-20): +9.17pp/fold,
  4/6 folds, survived ALL THREE adversarial re-checks (fold shifts ±21d:
  +18.6/+7.2pp; late subperiod: +21.1pp 4/4). Total-return bar only —
  drawdowns 13-20pp deeper than SPY (disclosed).
- 🟡 `deep_value_pe10` (P/E<10 profitable, top-20): +8.7pp/fold, fold
  Sharpe 1.98 (round's best), but re-check 2/3 (shift− dropped to 3/6).
  Ruled NOT confirmed (conservative); destined for experimental paper
  trading. **Re-check standard now pinned: all three perturbations must
  pass** (set before the next candidate reaches it).
- Family learnings for round 2: momentum return-premium is real but
  violent; value-with-profitability is the Lynch lane's live vein (3 more
  cousins each ONE fold short); ETF defensive cuts drawdowns, never beats
  SPY Sharpe (inverse-vol missed by literally one fold, 19/38); churn dies
  to costs exactly as modeled.
- **Promotion to paper trading PENDING one design decision:** a top-500
  cross-sectional strategy needs a concrete live universe (resolve-at-
  promotion static list vs nightly re-resolve). Flagged for the owner with
  the dashboard revamp.
- Round 2 runs the same loop via `hunt/queue.json` + the hunt-batch
  workflow (waves of ~10-25; stock/Lynch lanes should now use
  `--bars-from r2`).

### 🎨 DASHBOARD REDESIGN — COMPLETE (2026-06-10, v5.36→v5.45)

Dark trading-terminal, mobile-equal, data-rich, reliable. All 10 phases
shipped; suite 792, smoke 37/37 vs a live server on the seeded demo DB.

**Pages:** `/` Portfolio (lane-segmented Arena/V2 hero from server-computed
/summary — the fake-capital fallback is dead), `/symbol/{sym}` Workstation
(candles+volume, EMA/BB overlays, synced RSI/MACD panes w/ locked
crosshair, trade/dividend/split/news markers + trade drawer, search,
MAX = full R2 history, URL state), `/lab` (Scoreboard N-of-10 · Runs
browser w/ detail drawer + compare≤4 · Backtest tab w/ bounded polling),
`/trades`, `/strategies` (lanes + dividend/param ledgers), `/screener`
(hand-rolled treemap + qualification watchlist), `/ops` (heartbeats,
storage panel, agent timeline). `/research`→/symbol, `/backtest`→/lab.

**Architecture:** zero CDN deps (lightweight-charts 4.1 vendored; Chart.js
retired), zero inline styles (guard test, all templates), ES modules
(core/net w/ AbortController+retry+dedup; epoch-sec time boundary in
fmt.toEpochSec; dom error-cards everywhere; bounded poller; theme tokens
w/ live chart restyle), css/{tokens,base,components,charts}.css.
common.js/theme.css deleted; nav lives in core/nav.js.

**Verification tooling that outlives the redesign:**
`scripts/seed_demo_data.py` (offline demo DB) + `scripts/smoke_dashboard.py`
(every GET, run vs localhost or Render) + `tests/test_dashboard_static.py`
(asset serving, no-CDN, no-inline-style guards).

**Post-deploy checklist (owner, ~5 min on the phone):** open
edgefinder-pm8h.onrender.com on mobile — bottom tabs, /symbol pinch-zoom
+ MAX range (first R2 read warms the server cache), lab drawer, theme
toggle. Report anything off; charts restyle live on theme switch.

### 🪦 ARENA RETIREMENT — CODE DONE (2026-06-10, v5.47 `10bcaf4`)

Owner directive: "clear the code, base, database, and anything else of the
old strategies and arena." Scope decisions (owner, via AskUserQuestion):
DB history = **just delete, no archive**; legacy quick-backtest + old lab =
**remove**; agents = **retire coach, keep data collection** (scanner stays
as nightly fundamentals collector, no strategy qualification).

**Code (v5.47, 106 files, +955/−14,179):** deleted `edgefinder/strategies/`
(17 files), `trading/{arena,executor,account,risk}.py`, the old minute-bar
lab (`backtest/*` except costs.py), coach + weekly_summary + their
workflows, intraday-cycle/keepalive/eod-jobs/validate workflows, routers
backtest+inject, scanner qualification layer. Relocated verbatim:
`_sanitize_ohlcv`+`precompute_snapshots` → engine/backtest.py,
`resolve_universe` → engine/data.py. `services.py` rewritten 1,489→482
lines (9 jobs; NEW `_v2_snapshot_job` every 30m 09:45–16:15 ET appends
StrategySnapshot equity-curve rows; NEW `_market_snapshot_job` 16:05 ET).
Watchdog liveness repointed to `v2_portfolio_cycle` (daily cadence, 26h
staleness, weekdays after 10:00 ET). Gates: 553 tests, demo boot, smoke
35/35 — verified twice (subagent + orchestrator).

**DB — ✅ EXECUTED (2026-06-10 18:34 UTC, run f92a422 after v5.47 was
live + smoked 35/35):** 12,828 rows deleted via `ops/retire-arena.flag`
→ retire-arena.yml — arena trades(14)+context+orphaned
market_snapshots(519), strategy_snapshots(789), 7 accounts
(coward/gambler/degenerate + dead alpha/bravo/charlie/echo shells),
ticker_strategy_qualifications(11,485, whole table), manual_injections.
Verified by SQL after: strategy_accounts = exactly
{dual_momentum_v2, equal_weight}; 10 v2 trades; 0 arena rows anywhere;
prod re-smoked 35/35 on the emptied tables. The flag was reset to `done`
(any future trigger is a harmless dry-run; the script is idempotent).
market_snapshots starts empty — the 16:05 ET job repopulates daily.

### 🏹 HUNT ROUND 2 — COMPLETE (2026-06-10, v5.46 `322afcb` roster; full report `reviews/HUNT-ROUND-2.md`)

12 pre-registered candidates + 2 fresh randoms (`engine/hunt_r2.py`),
top-500 PIT, costed, TR, all from R2; 33 runs (wave ids 78–92, re-checks
93–110), zero lost jobs; null −0.02pp, randoms −8.3/−8.9 — instrument
clean. **THREE NEW CONFIRMED FINALISTS** (all-three-re-checks standard):
**value_mom_barbell** (+16.9pp 5/6 main; re-checks +22.7/+17.3/+28.7 —
strongest candidate in lab history; the value×momentum interaction is the
find of the round), **mom_inverse_vol** (+8.1 main; +15.8/+5.7/+17.9),
**mom_earnings_tilt** (+5.9 main; +4.9/+4.5/+13.8 — thinnest margins).
Re-checks KILLED value_pe12 (shift− 3/6, the pe10 pattern repeats),
mom_6m_k20 (shift+ collapsed to +0.45pp 2/5), mom_12_1_k40 (shift− 3/6).
All confirmed finalists fail the risk-adjusted bar (deeper drawdowns) —
return edges, not comfort edges; and 3 of 4 share the 12-1 momentum
engine (correlation caveat disclosed). earnings_yield_top was
bit-identical to value_pe12 (EY ≡ inverse P/E; redundant registration).
**Scoreboard: 4 of 10 confirmed** (xsec_mom_12_1, value_mom_barbell,
mom_inverse_vol, mom_earnings_tilt). Holdout still sealed.

**Open owner decision (still unresolved):** live paper-trading universe
mechanics for cross-sectional finalists (resolve-at-promotion vs nightly
re-resolve) — blocks promoting xsec_mom_12_1 and any round-2 confirmations.

### 🏹 HUNT ROUND 3 — COMPLETE (2026-06-10, v5.48; report `reviews/HUNT-ROUND-3.md`)

**TWO NEW CONFIRMED FINALISTS → scoreboard 6 of 10:**
**barbell_trend_value** (+18.0pp 5/6 main; re-checks +22.4/+18.3/+30.1 —
beats its r2 sibling on every number; counts as the barbell FAMILY's
second slot) and **fast_growers** (+7.5pp 5/6; re-checks +7.7/+6.9/+15.1
— the stablest re-check set recorded and the first finalist with NO 12-1
momentum and NO P/E). Killed: both soft risk overlays (3/6), rank blend
(+1.2 — the interaction needs separate sleeves), quality standalone/
momentum-filter, 52w-high, seasonality, ew_top100. fcf_yield_top +
dividend_value INVALID (PIT field coverage ~0% — audit in report; only
eg/rg/roe/cr/de/eps are usable). TWO machine bugs caught+fixed in-round:
double-fired wave (trigger now session-branch-only) and silently lost
records on pooler timeouts (validate.py exits 3 after 4 record attempts;
max-parallel 6→4; v5.48.1).

### 🏆 HUNT ROUND 4 — COMPLETE · **GOAL REACHED: 12 of 10 CONFIRMED FINALISTS** (2026-06-11; report `reviews/HUNT-ROUND-4.md`)

Round 4 confirmed SIX of seven re-checked candidates — the goal round:
**growth_mom_barbell** (+19.6 main; re-checks +26.7/+15.6/+36.2 — lab
records), **tri_sleeve**, **growth_value_barbell** (best risk profile:
Sharpe 2.10, dd 21.5%), **mom_3_12**, **peg_growers**,
**fast_growers_rev** (thinnest pass). Killed: fast_growers_mom (2/3 —
integrations die, sleeves survive), mom_sharpe_rank, roe_value,
value_cr_fortress, steady_compounders, min_vol_uptrend. Controls clean.

**FINAL SCOREBOARD (4 rounds, 73 candidates, 12 confirmed, 16% hit
rate):** xsec_mom_12_1 · value_mom_barbell · mom_inverse_vol ·
mom_earnings_tilt · barbell_trend_value · fast_growers ·
growth_mom_barbell · tri_sleeve · growth_value_barbell · mom_3_12 ·
peg_growers · fast_growers_rev. Full rollup + caveats (THREE underlying
engines, not 12 independent edges; total-return bar only — none clear
risk-adjusted) in `reviews/HUNT-ROUND-4.md`. The CONTINUOUS LOOP IS
STOPPED — goal exceeded. Queue idle.

**HOLDOUT BURNED 2026-06-11 — COHORT PASS, 12/12 positive excess
(median +25.9pp; null −0.02 valid).** Pre-registered protocol
`reviews/HOLDOUT-BURN-PROTOCOL.md`; restricted report
`reviews/HOLDOUT-BURN.md` (cohort-only read — do NOT dissect the burned
window). **NEW HOLDOUT WALL: 2026-06-11** — all future validations use
`--holdout-start 2026-06-11`; the burned window is ordinary in-sample
now and may never be cited as out-of-sample.

**DONE — ALL TWELVE PROMOTED 2026-06-11 (v5.50 live in prod):**
live-universe mechanics are BUILT: the live runner re-resolves
`--universe top:N` point-in-time at each rebalance boundary (R2 frames +
DB recency top-up, trailing `--rank-window` dollar-volume ranking, PIT
fundamentals in the context — validator-mirroring helpers shared via
`engine/data.py`), with a <90%-of-top_n shrink guard that falls back to
the last good `resolved_symbols` and raises a CRITICAL `live_universe`
observation; hold days stay cheap (no R2 load). Schema: production ALTER
applied 2026-06-11 (migration f1a9c4d27e55). All 12 finalists promoted
via the flag-gated promote-finalists workflow: tier "validated" through
`--finalist` (the pre-registered total-return + burned-holdout standard),
universe top:500 @ rank-window 126, monthly, each row linked to its
holdout-burn run (ids 184–195). First cycle EXECUTED 2026-06-11 via the
run-v2-cycle workflow (the 9:45 in-process cycle was killed by the
full-market R2 load — fixed in v5.51: SQL ranking over the DB hot set +
targeted R2 frames; dry-run verified, then executed: 12/12 resolved
500/500, 138 lots opened, 14 accounts live). v5.51 also fixed the lab
scoreboard (finalists = tier-validated promotions, was criteria.all_met),
pending-account rows, arena-lane removal, retired-heartbeat filtering. **MONITORING PHASE:** watchdog liveness + alerts cover failures;
the live-vs-lab scorecard (/strategies) accumulates the real proof over
the coming weeks/months. Disclosed deviation: validated on total-return
prices; live trades raw prices + dividend_credits.

### 🚀 HUNT KICKOFF (for the next session, when the owner says go)

The machine is fidelity-verified end to end. Lanes ready:
- **ETF lane** (21 yr, 2005→now, protected symbols — runs from the DB):
  `python -m edgefinder.engine.validate --strategy X --symbols ... --schedule monthly --holdout-start 2026-06-11 --record`
- **Stock lane** (5 yr, 2021-06→now, PIT top-N incl. graveyard, costed, TR
  — **runs from R2 since the slim**):
  `python -m edgefinder.engine.validate --strategy X --universe top:500 --start 2021-06-01 --costed --div-adjust --bars-from r2 --holdout-start 2026-06-11 --record`
- **Fundamental/Lynch lane**: PIT fundamentals 2019→now (128,854 snapshots);
  add `--pit-fundamentals`; strategies read `a.fundamentals.earnings_growth`
  etc. in rebalance(ctx); compute P/E-style ratios from `a.price` at
  decision time.
- New strategy = ~10 lines in `engine/strategies.py` (or a hunt roster
  module) + a spec in `make_strategy_factory`. Pre-register (commit params
  BEFORE first run).
- **Discipline:** pin ONE `--holdout-start` date for the whole research
  round and never evaluate it without explicit owner sign-off
  (`--burn-holdout`); promote candidates as `experimental` tier to
  paper-trade; `validated` tier requires the passing sealed holdout;
  adversarial re-check = all three perturbations must pass.

### Remaining cleanup (parked for the dashboard-overhaul phase)

4. **Consolidation debt** (shrunk by the v5.47 retirement — old lab stack
   now gone): lab/live duplication (`_is_rebalance` semantics + order math
   exist in backtest.py AND live.py — parity by parallel code, not shared
   code), heartbeat upsert x2 (services + engine/live), validated-rule x2
   (promote.py + strategies router). All intentional-for-now.
- **PHASE 5b — Go wide (the actual hunt). ✅ UNLOCKED 2026-06-10** (owner:
  "Try at least 30 strategies… that is our /goal"). Round 1 done (35
  candidates), Round 2 in flight — see the HUNT sections above.

### Working agreements (how the owner wants this run)

- **Keep the live system running and untouched** until the new engine is proven; all
  engine work is additive.
- **Small phases, each with tests + a `/code-review` pass** before moving on. The owner
  explicitly wants major review work to keep the code clean — bake it into every phase.
- **Keep all data. Never delete the database.** (A prior session wrongly proposed
  deleting it to fit a free tier — the opposite of the vision.)
- **Git:** commit + push directly to `main`; format `[vX.Y] short description`; bump
  `__version__` in `dashboard/app.py` for functional changes; run
  `pytest -m "not integration"` before committing code. If `git push` fails with a
  Git LFS hook error (`git-lfs not found`), use `git push --no-verify` — the repo has
  vestigial LFS hooks but no LFS-tracked files. The devcontainer now installs git-lfs,
  so this should be resolved after the next rebuild.
- **Autonomy:** finish the project autonomously, phase by phase, committing as you go;
  surface decisions only when they genuinely change direction.

### 🔑 SECRETS MAP — every secret × every place it must exist

Secrets do NOT follow your account; each runtime has its own vault and nothing
syncs them. When something can't authenticate, check this table first.

| Secret | Codespaces¹ | Actions² | Render³ | claude.ai cloud⁴ |
|---|---|---|---|---|
| `DATABASE_URL` (use the **pooler** form!) | ✅ set | ✅ set | ✅ set | add if needed |
| `EDGEFINDER_POLYGON_API_KEY` | ✅ set | — | ✅ set | add if needed |
| `EDGEFINDER_POLYGON_S3_ACCESS_KEY_ID` / `_SECRET_ACCESS_KEY` | ✅ set | ✅ set | — | — |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_ENDPOINT` / `R2_BUCKET` | ✅ set (rotated 2026-06-10) | ✅ set | ✅ set | add if needed |
| `CLAUDE_CODE_OAUTH_TOKEN` | ⬜ (re-add if watchdog CLI needed here) | ✅ set | — | n/a |

¹ GitHub → Settings → Codespaces → Secrets (scoped to this repo). Env vars in
  every Codespace. **Rebuild/restart the Codespace to pick up changes.**
² GitHub repo → Settings → Secrets and variables → Actions. Workflows only.
³ Render dashboard → service → Environment. **Applies on next deploy.**
⁴ claude.ai/code → environment settings (cloud icon → settings) → env vars in
  .env format. Used by mobile/web cloud sessions. ⚠️ Stored PLAIN TEXT,
  visible to anyone who can edit the environment — fine for this project's
  paper-trading keys, think twice for anything high-value. ⚠️ Cloud sandboxes
  do NOT set `CODESPACES`, so the direct→pooler DATABASE_URL self-heal won't
  fire there — paste the POOLER form
  (`postgresql://postgres.<ref>:<pw>@aws-1-us-east-1.pooler.supabase.com:5432/postgres`).

Gotchas that have already bitten: a pasted secret carried a trailing newline
(code now .strip()s R2 values); the direct `db.<ref>.supabase.co` host is
IPv6-only and unreachable from Codespaces (self-heal rewrites it, Codespaces
only). The zero-secret-movement alternative for mobile: run
`claude remote-control` in this Codespace and drive it from the phone — all
execution and secrets stay here (requires the Codespace to stay awake; bump
the idle timeout in GitHub → Settings → Codespaces, max 240 min).

### ⚠️ Codespace persistence warning (why this section exists)

Conversation transcripts and the agent memory dir live under `~/.claude/`, which is
**wiped on every Codespace rebuild** — only `/workspaces` (this git repo) survives. So
**persist all durable context to the repo, not to chat or memory.** When you make
material progress or change the plan, UPDATE THIS SECTION and commit, so the next
session continues seamlessly after any rebuild. (Cloud sessions can also be recovered
via `claude --teleport <session-id>` in a real terminal — but the committed handoff is
the reliable mechanism.)

---

## Update — 2026-06-09 (MICROCAP BUILD — Layers 1-3 + first net-of-cost survivor)

Built the "go where edges live" microcap workbench end-to-end (v5.20.0):
- **Layer 1 universe:** ingested top-3000/day (3.75M bars, delisted-incl.);
  `resolve_universe(rank_offset=N)` selects the small-cap/illiquid band
  (ranks 1000-3000, ADV ~$10-58M/day — small-cap, NOT true microcap, which
  trades thinner than a top-3000 pull reaches; disclosed).
- **Layer 2 cost engine** (`edgefinder/backtest/costs.py`): replaces the flat
  5bps with Corwin-Schultz half-spread + sqrt market impact + participation
  cap; plus delist force-close (dead names realize their loss, not freeze) and
  gap-through-stop (stop fires on the day's low, fills at the gap). FIXED cost
  assumptions, never optimized. Opt-in: cost_model=None ⇒ liquid path
  byte-identical. 20+ tests.
- **Cost threading:** cost_model flows through optimize→walkforward→holdout, so
  the optimizer searches NET-of-cost. validate/screen `--microcap`
  (=`--rank-offset 1000 --top-n 2000 --costed`); disclosed in scorecard
  (`costed`) + validation_runs label.
- **Layer 3 strategy:** `micro_reversal` (Connors-style: buy a multi-day
  washout above the 200dma, sell the RSI recovery) — pre-registered + committed
  (682d6a6) before any screen.

**First honest microcap result (screen, rank 1000-1500, fixed defaults):**
gross PF 2.17 / +5.95% → **costed PF 1.41 / +2.1%**, 35 trades, win 45.7%,
~$9 net/trade, 9.9% exposure. The cost engine ate ~65% of the gross edge but
**did NOT kill it — the first strategy in the project with a positive edge
after realistic costs** (every liquid strategy died honestly). HEAVILY
caveated: survived a SCREEN not validation; only 35 trades / 4.5yr at ~10%
exposure (thin, mostly cash); small-cap not true-micro; screen used
full-period (not PIT) universe ranking. Fold validation (PIT, net-of-cost,
+overlay since ~90% cash) is the next gate; the low frequency likely fails the
≥30-trades-per-fold criterion at fixed defaults.

## Update — 2026-06-09 PM (micro_reversal fold validation — FAIL, but the closest yet)

Fold validation, rank 1000-2000 band, costed (net-of-cost), PIT
(`--universe-as-of 2022-11-29`), +cash-overlay (≈90% cash), holdout reserved
NOT burned:
- **Fixed defaults:** Sharpe **1.13**, **3/5 folds beat SPY**, 58 trades (clears
  the ≥30 floor), BUT mean excess **−0.74pp** → FAIL on the positive-mean-excess
  criterion. Essentially SPY-EQUIVALENT, a hair under — the closest any strategy
  has come on honest data (liquid strategies failed by −1.3 to −6pp).
- **Optimized (net-of-cost, 12 iters):** Sharpe 0.53, 2/5 folds, excess
  **−4.2pp**, 164 trades. Forcing frequency by loosening the washout DILUTES the
  edge and feeds costs — the optimizer's higher-turnover configs generalize
  WORSE OOS than the rarer fixed-default washouts. Classic overfit/dilution.

**Verdict: micro_reversal FAILS — it MATCHES SPY net-of-cost but does not beat
it.** The screen's per-trade edge (PF 1.41) was real but, once measured against
the SPY it displaces (overlay) across out-of-sample regimes, it nets to ~flat.
Holdout stays sealed (correct — no clear fold pass).

**Microcap thesis status:** partially supported, not closed. The small-cap
reversal edge SURVIVES realistic costs at the per-trade level (unlike anything
liquid) but is too thin to beat SPY in the rank-1000-2000 band. TRUE microcaps
(deeper than a top-3000 ingest, where the documented edge is larger but costs
worse) remain UNTESTED — the door isn't fully shut, but testing them needs a
deeper/illiquid data pull and even more conservative cost modelling. The
cumulative honest finding across ~10 strategies and 2 universes stands: a
net-of-cost SPY-beating edge in tradeable US equities is very hard to find, and
the lab now proves that rather than faking the opposite.

**Build delivered (v5.20.0):** the realistic cost engine (costs.py — spread,
impact, participation cap, delist force-close, gap-through-stop) is the durable
asset — it makes every future backtest in this lab trustworthy, and it is what
turned a screen that "passed" (PF 1.41) into an honest fold FAIL.

## Update — 2026-06-09 EVE (strategic-direction panel — all 5 low-probability; the goal needs disambiguation)

10-agent adversarial panel over 5 forward directions, scored vs the user's real
goal (beat SPY in backtest, prove it, deploy to a small personal account). ALL
FIVE came back LOW or VERY_LOW probability of a backtest-provable, net-of-cost,
SPY-RETURN-beating edge. Specifics:
- **True microcaps** — LOW. Decisive correction to the "edge grows as universe
  shrinks" reframe: short-term reversal profit IS the bid-ask spread earned by
  PROVIDING liquidity; a retail account CROSSES (pays) that spread — the opposite.
  PF 1.41 was a gross/per-trade figure; net-of-SPY already FAILED (-0.74pp) and
  going deeper makes the trader pay MORE of the spread that IS the edge. Daily-
  OHLCV backtests are least trustworthy exactly there. ONE cheap decisive test
  worth doing: re-screen costed + PIT on the existing thin band, measured net-of-
  cost vs SPY AND vs the menu's own equal-weight buy-hold ("beat your own menu");
  if <=0, the microcap door closes cheaply.
- **Ensemble** — LOW, provability fails: components (reversal+momentum) aren't
  error-independent, so no diversification benefit to push -0.74pp across zero.
- **Untried families (low-vol / 52wk-high / value+quality)** — LOW on the RETURN
  bar but the MOST provable (liquid, low-turnover, trustworthy data, ~1.5wk, no
  live proof). Low-vol's verdict is already public: USMV +45% vs SPY +92% over
  2021-2026 — it's a Sharpe/drawdown play, a guaranteed FAIL on positive-mean-
  excess-RETURN.
- **Live-forward-proof pipeline** — premature: it would forward-prove a strategy
  that fails the offline gate; the scorecard is the eventual final gate, activates
  only once something passes offline. Don't build now.
- **Instruments (options/leverage/vol-target)** — VERY_LOW. Vol-targeting SPY
  lever-DOWN is pre-falsified by identity (excess <=0 vs SPY); options
  un-backtestable on our data; leveraged-ETF rotation is a regime-timing bet.

**THE REAL FINDING — the goal contains a tension that decides tractability:**
"beat SPY" splits into (a) beat SPY on TOTAL RETURN — the lab's current bar
(positive mean excess return), which ~10 honest falsifications + the academic
consensus say is near-impossible net of cost in tradeable US equities; vs
(b) beat SPY RISK-ADJUSTED (Sharpe/drawdown) — which IS achievable and provable
(low-vol, vol-targeting, regime gates) but trades raw return for much less risk,
and only beats SPY on RETURN if levered (retail-constrained). The project has
implicitly chased (a). The highest-value next step is the OWNER choosing (a) vs
(b) — it determines whether the search is tractable at all. Pending owner call.

## Update — 2026-06-09 NIGHT (risk-adjusted pivot built; trend_timer FAILs; 3 engine bugs fixed)

Owner chose the tractable goal: SPY-like return with much less RISK (beat SPY
on Sharpe/drawdown), not raw-return alpha. Built + tested:
- **Risk-adjusted scoring (v5.21.0):** benchmark now computes SPY's own Sharpe
  + max-drawdown per window; new `--risk-adjusted` criteria mode (beat SPY's
  Sharpe in a majority of folds AND a smaller drawdown; no >=30-trade floor).
- **trend_timer (pre-registered):** hold SPY above its 200-EMA, cash below
  (Faber). FAILS the risk-adjusted bar 2021-2026: fold mean Sharpe 0.91 vs SPY,
  excess_sharpe **-0.36**, only **1/5 folds** higher Sharpe, drawdown cut
  +2.41pp. Full-period agrees: +40.7% / Sharpe 0.69 / maxDD 22.3% vs SPY +75% /
  0.74 / 25.4% — TIES SPY risk-adjusted, doesn't beat it. 2021-26 is a known-hard
  window for trend-following (sharp V-recovery → whipsaw + late re-entry); its
  drawdown-avoidance benefit never triggered. Holdout untouched.

**THREE backtest-engine bugs found while validating a single-ticker strategy
(all silently distorted EVERY prior backtest, masked by multi-ticker books):**
1. Re-entry/revenge/PDT cooldowns compared vs WALL-CLOCK now() — a backtest runs
   in seconds, so a ticker was locked out for the whole run after its first
   close (trend_timer traded once). Fixed: VirtualAccount._clock injectable,
   driven off the simulated day.
2. A corrupt data bar (SPY 2026-02-02 low=69 vs ~690) tripped the 20% stop →
   fabricated -20%/-$2158 trade, turning trend_timer's honest +41% into -13%.
   Fixed: _sanitize_ohlcv clamps impossible OHLC in the precompute (protects all
   runs; matters most for microcaps). Note: prior microcap/liquid results may
   shift slightly now that bad prints can't trip phantom stops.
3. Closed-trade entry_time defaulted to wall-clock (cosmetic; max-hold used the
   position's correct simulated entry_time). Fixed: carry it onto the trade.

**Next within risk-adjusted (intent):** VOL-TARGETING SPY (scale exposure to a
constant vol target; de-risk in high-vol regimes). It is the complement to
trend-timing and historically more robust in fast-recovery windows — the
trailing-vol + cash_overlay plumbing already exists. Then optionally a
multi-confirmation regime gate (trend+breadth+vol) to cut the 200-EMA whipsaw.

## Update — 2026-06-09 LATE-NIGHT (dual_momentum FAILs; THE WINDOW is the problem — backfilling to 2008)

- **dual_momentum (pre-registered, Antonacci/Faber asset-class rotation over
  SPY/QQQ/IWM/DIA/GLD/TLT/EFA):** found + fixed a churn bug (fill-day buffer gap
  made every position exit after 1 day, ~11% exposure; fix = hold when a name is
  missing from yesterday's buffer). Post-fix full-period top_k=3 TIES SPY Sharpe
  (0.65 vs 0.74) and cuts drawdown 5.8pp — best risk-adjusted result yet — but
  the OOS folds FAIL: excess_sharpe -0.54, 2/5 folds higher Sharpe, DD cut
  +2.04pp. Cuts risk, doesn't beat SPY's Sharpe.

**THE INSIGHT (both risk-adjusted candidates converge on it):** trend_timer and
dual_momentum BOTH cut drawdown but BOTH fail to beat SPY's Sharpe over
2021-2026 — because 2021-2026 is a STRONG-BULL window with only one brief bear
(2022). SPY's own Sharpe is ~0.74 here; defensive/risk-managed strategies are
DESIGNED to prove their worth in SUSTAINED bears (2008, 2000-02), which our data
does not contain. The window, not the strategies, is the binding constraint.

**Polygon flat-files go back to 2004** (verified) — so we can build a window that
includes the 2008 financial crisis AND the 2020 COVID crash, where these
strategies should clearly win if they work at all. Backfilling the ETFs (and the
broad universe) to 2004 is the highest-value move and is squarely strategy-
hunting, not lab-fixing: the strategies are plausibly winners our window can't
reveal. Re-test trend_timer + dual_momentum on the long (incl-2008) window next.

## Update — 2026-06-09 ENDGAME (21-year test incl. 2008: defensive strategies HALVE drawdown but don't beat SPY's Sharpe)

Polygon can't serve pre-2021 (aggs API ~5yr cap; flat-files 403 for old days).
Got the deep ETF history FREE from Yahoo (2005-2021, spliced cleanly with
Polygon 2021+, both split-adj/div-unadj). ETFs now span 2005-2026 (~21yr, incl.
the 2008 crisis + 2020 COVID). Tested trend_timer (SPY) and dual_momentum
(top_k=3, 7 ETFs). Sub-period Sharpe/return/maxDD:

```
period               SPY ret/Sh/DD      trend_timer        dual_momentum
2008 crisis          -48% -0.96  56%    -15% -1.63  16%    -9% -0.46  18%
recovery+bull 09-20  297%  0.92  20%     83%  0.54  20%    68%  0.49  18%
covid crash          -10% -0.45  34%    -10% -2.33  15%    -6% -1.19  15%
2021-26              104%  0.87  25%     60%  0.81  27%    45%  0.69  23%
FULL 05-26           524%  0.55  56%    197%  0.50  27%   179%  0.48  23%
```

**The honest verdict — a genuine result, just not the hoped-one:**
- DRAWDOWN: the strategies WIN decisively. SPY's worst loss was **-56% (2008)**;
  trend_timer -27%, dual_momentum **-23%** — less than HALF. In 2008 itself
  dual_momentum lost only **-9% vs SPY's -48%.** Provable, robust crash
  protection across 2008 AND 2020.
- SHARPE: they LOSE, narrowly. Full-period 0.48-0.50 vs SPY 0.55. Halving the
  drawdown does NOT flip Sharpe because Sharpe penalizes VOLATILITY not tail
  loss, and the bull decades (SPY Sharpe 0.87-0.92) dominate 21 years — the
  defensive bull-lag drags full-period Sharpe just below SPY's.
- RETURN: they lag a lot (+179/197% vs +524%) — the price of sitting defensive
  through long bulls.

**So "SPY return with much less risk" splits again:** if RISK = max drawdown
(don't lose half in a crash), dual_momentum is a real, deployable, provable win
(half the drawdown, -9% vs -48% in 2008) — at the cost of ~1/3 the return and
~equal Sharpe. If RISK = volatility/Sharpe, NO canonical strategy beats SPY even
across 2008. Leverage can't rescue it (Sharpe 0.48 < 0.55, so a levered version
is worse risk-adjusted). This converges with the academic consensus: trend/
momentum's durable benefit is CRASH PROTECTION, not Sharpe improvement.

Costs negligible here (liquid ETFs, infrequent trades). Owner decision: is
"half the drawdown / proven crash protection / lower return / ~equal Sharpe" the
deployable win, or keep hunting a true Sharpe-beater (evidence says it's very
unlikely to exist after costs)?

## Update — 2026-06-06 (free-data-source vetting — adversarially verified)

Owner asked: can we add alternative free datasets to find an edge bar-data
alone can't? Vetted 5 candidates with live probes + a 10-agent adversarial
workflow (5 analysts web-searching current OOS evidence, 5 skeptics
refuting). Verdict on the literal three axes (add / reliable / free):

**No standalone alpha among them for our liquid top-1000 universe.** The
honest scorecard:
- **PEAD** — free (on plan), reliable=NO, edge=NO. Two killers: (1) Polygon
  serves only 10-K/10-Q, NOT 8-K, so the announcement-timestamp premise is
  unbuildable from Polygon; filing_date lags the press release 10-14d, inside
  the drift window (signal decay + look-ahead); filing_date was even None on
  SIVB's actual earnings quarter. (2) Martineau 2022 "Rest in Peace PEAD" +
  2024-25 replications: drift t-stat 2.18→1.43 (insignificant) ex-microcaps;
  survives only in microcaps where ~10% round-trip cost eats it. Top-1000 =
  the dead segment. USE ONLY as (a) a known-dead falsification control, (b)
  motivation to build an EDGAR 8-K item-2.02 event clock as PIT infra.
- **Insider clusters (Form 4)** — free (SEC), reliable=NO out of box, edge=YES
  but decayed + microcap-bound (CMP 82bps/mo; 2018-24 microcap OOS AUC 0.70,
  gross). Needs delisted ticker→CIK backfill (company_tickers.json is
  active-only). Most engineering (~9d). Real, but capacity-constrained to where
  this lab can barely trade.
- **Regime gates (VIX + breadth + credit)** — reliable=YES, edge=risk-mgmt
  only (cuts drawdowns ~hugely at ~1.5pp/yr CAGR cost; Sharpe +0.1..0.3, no new
  alpha). Cheapest (~3d). LIVE CAVEAT the skeptic caught: FRED is truncating the
  ICE BofA OAS credit-spread family to 3yr history starting April 2026 — do NOT
  architect on it; lean on VIX (CBOE free) + our-own-breadth (% of universe
  above 200dma, one SQL query on bars we already have, bulletproof PIT).
- **Short interest** — Polygon endpoint is STALE (probe: all tickers return
  ~2017). FINRA files are the free alternative but the edge is short-side,
  small-cap, net-of-borrow ~zero (Muravyev-Pearson-Pollet 2025) — uncapturable
  by a long-only top-1000 lab. CUT.
- **Macro/FRED** — needs free key; reliable=with care (NFCI/ANFCI ARE revised →
  need ALFRED vintages; only DGS10-DGS2 + credit spreads safe from current
  FRED). Edge=risk-mgmt only. Same family as regime gates.

**Takeaway:** none is an "add data → beat SPY" button. Highest-value builds
if pursued: (1) regime/breadth OVERLAY on existing strategies — cheap,
reliable, improves drawdown/Sharpe of what we have (uses the cash_overlay
plumbing already built); (2) EDGAR event-clock + PEAD as an integrity control
proving the survivorship-corrected lab manufactures no phantom alpha. Insider
clusters only if willing to fund the CIK backfill and accept microcap costs.
No build started — owner deciding.

## Update — 2026-06-06 PM (FMP + historical-sentiment vetting — live-probed)

Extends the data-source vetting. 8-agent adversarial workflow + live probes
(Polygon API direct; FMP MCP authenticated this session).

**FMP — DEAD for this lab, empirically confirmed.** User's FMP tier returns
ONLY real-time quotes: `income-statement` DENIED (needs higher plan), `news`
DENIED (needs Starter+), `quote` works. So zero historical/fundamental/news
value at the current tier. Even on paid: "as-reported" ≠ point-in-time (no
restatement vintage → look-ahead), documented data-quality failures (e.g. AA
reported NEGATIVE revenue per Trustpilot; no internal reconciliation), and
fully redundant with Polygon flat-files + Polygon news already owned. Do NOT
add or upgrade.

**Polygon news sentiment — the surprise, then the disqualifier (live-probed).**
Polygon news text goes back to 2017 and covers delisted names (SIVB/FRC), BUT
the per-ticker `insights.sentiment` label was probed across time:
AAPL Jun-2021 0% / Jun-2022 0% / Mar-2024 0% / Sep-2024 100% / Sep-2025 100%;
SIVB-2023 0%. The label TURNS ON ~mid-2024 and is NOT backfilled. Consequence:
(+) no hindsight look-ahead, but (−) NO sentiment for 2021–mid-2024 and ZERO on
the delisted graveyard names. Unusable as a historical feature for the
2021-2026 survivorship-corrected folds. Only two honest uses: (a) self-score
sentiment from the raw news TEXT (which IS 2017+, delisted-covered, original
timestamps — point-in-time clean) as a build; (b) bank Polygon's forward
sentiment from now and validate live over time. Both real, neither a shortcut;
and Tetlock-family evidence says news-sentiment alpha is ~arbitraged in liquid
large-caps anyway (lives in small/illiquid).

**GDELT — not for single-name equity.** Free DOC API is a rolling 3-month
window only (useless for 2021+ history); full history = multi-TB raw-file ETL
(~12d). Documented edge is macro/FX/index-level, not cross-sectional
single-name; entity→ticker mapping is noisy. Possible future macro/regime
overlay input, not a stock picker.

**Other free news datasets (Alpha Vantage, Tiingo, FNSPID, etc.) — skip.**
All either survivorship-biased (FNSPID = current S&P500 membership, delisted
names absent), too short, or strictly dominated by the already-owned Polygon
news.

**STRATEGIC INFLECTION (the real takeaway):** ~9 data/signal angles vetted
across two sessions (PEAD, insiders, short-interest, regime gates, macro/FRED,
FMP, Polygon-sentiment, GDELT, other-news) ALL converge on the same finding:
in the liquid top-1000 universe, every documented edge is arbitraged to ~0
after costs, lives in microcaps we can't trade cheaply, or is
risk-management-only. The data is NOT the bottleneck — market efficiency in
liquid large-caps is. Adding sources keeps re-confirming the textbook result.
Realistic forward options: (1) accept + publish the honest negative result as
the dashboard's product; (2) go where edges live (microcaps) with brutal
explicit cost modeling — lab not currently built for it; (3) pursue
risk-management/regime overlays (improve drawdown/Sharpe of existing
strategies, won't beat SPY on return); (4) forward live sentiment experiment
(slow, honest, uncertain). No build started — owner deciding direction.

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

## Update — 2026-06-05 EVE (Round 2 complete: all candidates dead, holdout still sealed)

Run as a continuous research loop. Protocol held throughout: every
candidate's defaults pre-registered + committed BEFORE its first screen
(4c37284, 34b8118, 795e457); dev window = all data minus the sealed final
126d; the holdout was never evaluated.

**Round-2 lab improvements (shipped first):**
- v5.18.0: real per-day market context in the lab (spy_price/change/sma_200
  per simulated day — context-gated strategies now testable).
- v5.18.1: `validate.py --fixed` (defaults-only folds, no per-fold
  optimizer — the selection-bias-free test) + `optimized: true/false`
  disclosed in every recorded scorecard. **Methodology change, disclosed:**
  all fold tests now use the uniform 126d OOS window default (round-1 fold
  runs used 63d windows, which clip cross-quarter carry).

**Round-2 candidates (top-300 dev screens, all DEAD):**
1. **trend_dip** (Connors-family: 3-day stretch above 200dma, W%R ≤ −90,
   exit RSI ≥ 60) — PF 1.14, 263 trades, but **avg $3.23/trade**, under
   the ~$5/trade friction floor (slippage both sides). The trend gate DID
   fix coward's negative expectancy (coward bought knives, this doesn't) —
   the effect is real but too thin to harvest at retail cost. DEAD.
2. **gap_drift_v2** (ATR-normalized gap threshold, one variable changed
   from v1) — +18.7%, Sharpe 0.72, PF 1.62, 95 trades vs v1's +54.5% /
   1.12 / 1.92 / 131 on the same window. Dominated by v1 on every metric:
   the "absolute thresholds miscalibrate across vol regimes" hypothesis is
   falsified. DEAD.
3. **gap_carry** (v1's exact entry, exits rebuilt to hold the drift: 21EMA
   trail, 6% fail-stop, 30% target, 45d max hold) — +36.7%, Sharpe 0.69,
   PF 1.58, 127 trades, max DD 13.8% at **73% exposure** vs v1's 53%.
   Dominated by v1 on every metric while using more capital: holding
   individual winners longer gives profit back. **Key learning: v1's
   cross-quarter carry lives in re-entry compounding across gap events,
   not in longer per-trade holds.** DEAD.

**The decisive test — gap_drift v1, FIXED defaults, uniform 126d folds**
(`--fixed --oos-days 126 --step-days 126 --holdout-days 126
--no-holdout-eval`, top-300, recorded in validation_runs with
optimized:false): mean Sharpe **+0.49** (was −0.29 at 63d), OOS compounded
**+26.99%**, 114 trades, worst fold DD 11.4% — but mean excess vs SPY
**−1.90%** and only **2/5 folds beat SPY**. Verdict **FAIL**. The longer
windows confirmed half the carry diagnosis (Sharpe flipped strongly
positive) but the strategy is a lower-vol/lower-return profile that does
not beat SPY. gap_drift stays PARKED; with all three variants dead, the
gap family is closed for this round. **Holdout: still sealed — zero looks
burned across rounds 1 and 2.**

**Cumulative scoreboard (8 candidates tested honestly):** coward, gambler,
degenerate — falsified (round 0). pullback_rider, turtle_adx — screen-
killed. trend_dip, gap_drift_v2, gap_carry — dead (round 2). gap_drift —
parked (profitable, sub-SPY). Nothing has earned a holdout look.

**Round 3 directions (intent, not yet pre-registered):** (a) regime/
portfolio layer over gap_drift — accept its absolute-return profile and
test leverage-free ways to close the SPY gap (e.g. SPY overlay when flat:
strategy is in cash 47% of the time; the idle-cash drag IS the excess-
return gap), (b) candidates from effect families not yet tried
(cross-sectional momentum, post-earnings drift with actual earnings dates
from Polygon, index-fund-flow seasonality), (c) accept-and-publish: the
honest conclusion that nothing tested beats SPY after costs is itself a
defensible research output the dashboard can display.

## Update — 2026-06-05 NIGHT (Round 3 ws1: HOLDOUT BURNED — overlay passes folds, fails holdout)

Owner picked round-3 direction "both, overlay first". Workstream 1 ran
start to finish:

**SPY cash-overlay built + pre-registered (v5.18.2, 32eb6cb, committed
before any run):** idle cash parks in SPY; 5 bps per conversion at the
day's close; ZERO tunable knobs; provably does not change which trades
happen (engine sizes/fills off raw cash — overlay is pure equity
accounting). `validate.py --cash-overlay`; `cash_overlay` disclosed in
every recorded scorecard config.

**Fold test (fixed v1 defaults + overlay, 126d folds, top-300):**
**PASS — first all_met in project history.** Sharpe 0.49→**1.30**, mean
excess −1.90%→**+4.34%** (above the +2–3pp noise band), 3/5 folds beat
SPY, +65.5% compounded OOS, 114 trades. The idle-cash-drag hypothesis
confirmed on dev data — and the invested slice beat the SPY it displaced.
Cost: worst fold DD 11.4%→26.4% (SPY beta in the 2022 fold).

**THE HOLDOUT LOOK (first ever burned, per the pre-agreed rule that a
clearly-passing FIXED config earns one):** window 2025-11-21..2026-05-26
(bull_calm), params {} (pure pre-registered defaults), overlay on.
Result: +12.26%, Sharpe 1.36, 39 trades, win 46.2%, DD 12.9% —
**excess vs SPY −1.63% → holdout FAILS the beat-SPY criterion.**

**Honest verdict: gap_drift + overlay ≈ SPY + noise.** Its excess
oscillates around zero (+4.3pp folds, −1.6pp holdout, both small). Superb
absolute profile (made money in every window ever tested, fold Sharpe
1.30, holdout Sharpe 1.36) but no reliable edge OVER the index. The
sealed holdout did exactly its job: caught a dev-region pass that didn't
generalize, on the first config good enough to reach it. Disclosed
caveat upheld: the overlay idea was motivated by dev-region diagnostics;
the holdout was the unbiased test and it said no.

Dashboard state is automatically correct: the validation_runs row records
criteria all_met=true WITH an evaluated holdout passes=false → NOT
validated (v5.17.2 semantics). Round-3 holdout budget: SPENT.

**Status after rounds 1–3ws1: 9 honest tests, 0 validated strategies.
Remaining round-3 queue: workstream 2 (new effect families:
cross-sectional momentum, PEAD with real Polygon earnings dates,
seasonality) — pre-register before screening, NEW holdout look not
available until next round per one-look-per-round.**

## Update — 2026-06-05 LATE (Round 3 ws2: xsec_mom passes 5/5 folds — ROUND-4 FINALIST, holdout deferred)

**Engine extension (v5.19.0, 6c730f8):** `MarketContext.as_of` — strategies
now know the session date (simulated day in the lab, ET date live), enabling
calendar-aware and cross-sectional designs with identical lab/live behavior.

**xsec_mom pre-registered + committed before any screen (6c730f8):**
cross-sectional momentum (Jegadeesh-Titman family — first try from the most
evidence-backed class). Score = close/ema_200 − 1 (the long-horizon momentum
proxy the 30d history window allows; doubles as trend gate). Per-day rank
buffer filled by the engine's daily watchlist sweep; entries rank on
YESTERDAY'S completed cross-section (no look-ahead; at most top_k qualify
per day so no slot-order bias); exit on decay out of 3×top_k (derived) or
max_hold; target fixed wide 0.50. Searched: top_k [3,5,10], max_hold
[21,42,63]. Kill criteria: <$5/trade or PF<1.10.

**Screen (top-300 dev): SURVIVED** — $13.59/trade, PF 1.58, Sharpe 0.61,
win 53.3%, max DD 5.2%, 105 trades… at **8.5% exposure** (trend gate flat
through the 2022 bear + 200dma warmup; rank churn). Raw it cannot beat SPY
(−41% excess) — disclosed skip of raw folds; the honest portfolio
construction for a 91.5%-cash strategy is the zero-knob SPY overlay.

**Fixed-defaults 126d folds + overlay: PASS — strongest result ever.**
Sharpe **1.98**, mean excess **+7.86%** (far above the noise band),
**5/5 folds beat SPY**, +100.1% compounded OOS, 137 trades, win 59.1%,
worst fold DD 22.3% (SPY beta). Recorded in validation_runs
(optimized:false, cash_overlay:true).

**Disclosed caveats:** (1) all five OOS windows land in the 2023–2025
bull — the 2022 bear only ever appears in IS windows, so the momentum
tilt's bear behavior is OOS-untested (the trend gate held it flat through
2022 in the screen, which is why exposure is 8.5%); (2) 9th candidate
tested — one strong pass among nine could be selection luck; the sealed
holdout adjudicates; (3) entries are one day staler than necessary
(T−1 ranks, T+1 fills) — conservative direction.

**ROUND-4 FINALIST (frozen now, pre-registered):** xsec_mom pure committed
defaults (top_k=5, max_hold=42, exit_rank=15 derived, target 0.50,
risk 0.03) + zero-knob SPY cash overlay. Round 4 opens with its ONE
holdout look (`--fixed --cash-overlay --holdout-days 126`, eval ON) and
nothing else may touch the sealed region first. This round's look was
already burned on gap_drift+overlay (failed).

## Update — 2026-06-06 (benchmark audit: FINALIST DEMOTED — edge was universe bias)

Owner challenged the benchmark ("is beating 55% realistic?"). Full audit:

**The benchmark itself is clean.** SPY +55.24% dev-window verified against
raw closes (420.33 → 652.53, ~10.3%/yr price-only); per-window comparison
is relative; price-only is slightly SOFT vs total-return reality.

**The bias is the universe (two layers, both measured):**
1. *Future-selection*: "top-300" ranked by FULL-PERIOD dollar volume —
   tomorrow's winners selected into yesterday's menus. FIXED in v5.19.1:
   `resolve_universe(as_of=...)` + `--universe-as-of` (validate + screen),
   disclosed in validation_runs labels (`top-300@2022-11-29`).
2. *Survivors-only table (graveyard missing)*: 2679/2692 daily_bars
   symbols alive to table end, zero ending before 2024 — delisted losers
   absent entirely. NOT yet fixed; needs a Polygon `active=false`
   delisted-ticker backfill (now the top research-infra priority).

**Measured do-nothing tilt** (equal-weight buy-hold of the menu vs SPY,
per 126d window): full-period menu +0.4..+7.0pp (mean +2.9pp), holdout
window +4.8pp. Point-in-time menu (layer-1 fixed, layer-2 remains):
+1.3..+8.0pp (mean **+3.5pp**), holdout window **+5.5pp**.

**xsec_mom+overlay re-test on the point-in-time universe (fixed defaults,
126d folds, holdout still sealed):** mean excess **+7.86pp → +2.16pp**,
5/5 → 3/5 folds, Sharpe 1.98 → 1.37, win 59.1% → 49.5%. Criteria
technically all_met, BUT the remaining +2.16pp is BELOW the same menu's
+3.5pp do-nothing tilt — i.e. relative to its own opportunity set the
strategy adds ~−1.3pp of selection value. **The entire original edge is
explained by menu bias. DEMOTED — no holdout look will be spent on it.**
Decomposition: ~5.7pp future-selection + remainder within survivors-only
tilt.

**Blanket consequence:** every positive lab result to date sits inside
menu bias (gap_drift's fold passes face the same discount). The lab
cannot certify ANY SPY-beating strategy until the graveyard is restored.
Round-4 plan REVISED: (1) delisted-ticker backfill infra, (2) interim
control — score excess vs the menu's own EW return alongside SPY ("beat
your own menu" = skill; beats-SPY-but-not-menu = bias harvest), (3) only
then resume candidate testing; holdout budget preserved.

## Update — 2026-06-06 PM (GRAVEYARD RESTORED — xsec_mom edge fully falsified; honest baseline starts here)

The graveyard backfill ran (`--top-per-day 1000`, 1,251 days, 1.25M bars,
0 failures): daily_bars now holds each day's top-1000 common stocks by
dollar volume drawn from Polygon reference tickers ACTIVE AND DELISTED
(5,288 + 6,498). Result: 6,160 symbols, **3,082 with histories that END
mid-range** (396 died in 2021, 432 in '22, 469 in '23, 501 in '24, 918
in '25). The old table had 13.

**Menu tilt re-measured on the honest menu (EW buy-hold vs SPY,
PIT cut 2022-11-29):** −2.4, −1.9, +12.2, +0.7, −1.2pp per fold
(fold 3 = the genuine Dec-23..Jun-24 small/mid rally), holdout window
+1.4pp. The systematic survivors-only free lunch (+2.9..+3.5pp mean) is
GONE. Beating SPY on this menu now means beating SPY.

**xsec_mom+overlay, frozen config, honest menu, fixed-defaults 126d
folds: FAIL.** Mean excess **−1.30pp** (negative), 2/5 folds beat SPY,
win rate 44.1% (was 59.1% on the rigged menu), Sharpe 0.93, 123 trades.
Full decomposition of the lab's best-ever result:
rigged menu +7.86pp 5/5 → fair ranking (survivors-only) +2.16pp 3/5 →
honest market **−1.30pp 2/5**. The entire edge was data bias. The
holdout was never spent on it — protocol held; round-4 look preserved.

**Blanket void:** every pre-graveyard positive result (gap_drift's PF
1.92 screen, both overlay fold passes, every screen survival) was
measured on survivors-only data and is hereby treated as void. The
honest research baseline starts at this commit. The criteria bar is
unchanged; the data finally deserves it.

**Engine note (known, accepted):** a position whose symbol stops
trading mid-run freezes at its last print until run end (no forced
delisting close). Final prints of collapses are in the data and the 20%
stop fires on the way down, so the distortion is small; a "delist ⇒
force-close at last bar" engine rule is queued as a small fidelity item.

## Round 3 CLOSED — 2026-06-05 (final)

**tom_seasonality (pre-registered 760653c, first calendar candidate):
screen DEAD on both kill criteria** — $1.30/trade (< $5 floor), PF 1.04
(< 1.10), Sharpe 0.06, win 48.2%. The TOM effect in 2021-25 large-caps,
if any, is far below single-stock friction. No fold test warranted.

**Round-3 final ledger:** 5 candidates evaluated (gap_carry dead,
gap_drift+overlay fold-pass→holdout-FAIL, xsec_mom fold-PASS 5/5 →
round-4 finalist, tom_seasonality dead; PEAD not run — blocked on a
Polygon earnings-date backfill, the next research-infra investment).
Lab improvements shipped: --fixed defaults-only folds + methodology
disclosure in scorecards (v5.18.1), zero-knob SPY cash overlay
(v5.18.2), MarketContext.as_of date for calendar/cross-sectional
strategies lab+live (v5.19.0). One holdout look burned, honestly, on
gap_drift+overlay: FAIL. Cumulative: 11 honest tests, 0 validated, 1
strong finalist awaiting round 4.

**Round 4 plan (pre-registered intent):**
1. FIRST ACTION, before any other evaluation: the finalist's holdout
   look — xsec_mom frozen defaults + overlay,
   `--fixed --cash-overlay --holdout-days 126` (eval ON). Pass ⇒ first
   VALIDATED strategy (dashboard flips automatically); fail ⇒ the
   strongest fold result the lab has produced was still selection noise
   — strong evidence the +2-3pp band should widen.
2. Strongly preferred before/alongside: live paper evidence. Promoting
   xsec_mom to live needs (a) live overlay mechanics (virtual accounts
   hold idle cash as SPY — engine work), (b) live watchlist = the same
   top-300 cross-section, (c) live_strategies allowlist entry. The live
   scorecard then applies the same bar to real-time data.
3. PEAD infra: backfill real earnings dates from Polygon, then
   pre-register a post-earnings-drift candidate.

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
