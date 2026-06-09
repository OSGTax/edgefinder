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

### ROADMAP — resume here

- **PHASE 2 — Honest validation harness on the new engine ← NEXT.** Port walk-forward
  folds + sealed holdout + risk-adjusted acceptance criteria (beat SPY's Sharpe / cut
  drawdown, net of real costs) onto `engine/`. Then properly validate equal-weight
  (and variants). Reuse logic from `edgefinder/backtest/` + `tests/test_walkforward.py`
  but drive the new pure engine.
- **PHASE 3 — Promotion pipeline (the loop the owner wants).** `backtest clears bar →
  auto-deploy to self-running paper trading → shows on dashboard.` Pieces exist
  (plugins, virtual accounts, intraday loop, dashboard, `live_strategies` allowlist);
  mostly wiring + a clean "promote" action. (This is the long-deferred P3.)
- **PHASE 4 — Storage: two-tier (R2), parallel & non-blocking.** History (`daily_bars`,
  ~1.3 GB, write-once/read-heavy) → **Parquet on Cloudflare R2** (10 GB free, zero
  egress, S3-compatible; ~1.3 GB → ~200-400 MB compressed). `data/flatfiles.py`
  already speaks boto3/S3; `data/cache.py` already does Parquet. Route the data layer
  through a "bar store" abstraction instead of reading `daily_bars` directly (few
  choke points — not a rewrite). Live/operational data (<30 MB) stays in **Supabase**,
  lean. **R2 secrets** the owner is adding as GitHub → Codespaces → Secrets (scoped to
  `OSGTax/edgefinder`, injected as env vars every session):
  `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`
  (`https://<accountid>.r2.cloudflarestorage.com`), `R2_BUCKET=edgefinder-data`.
  **On resume verify first:** `env | grep -E '^R2_'` (never print secret values). If
  present, wire the bar store to R2; if absent, build on current data and note it's
  pending (non-blocking).
- **PHASE 5 — Go wide (the actual hunt).** Stop hand-crafting one strategy at a time;
  use the honest lab as a high-throughput screen. Port the old strategies onto the new
  interface and re-test HONESTLY (some "failures" were the old bugs). **Build the
  untouched fundamental/Lynch lane** (e.g. "hold 20 lowest-PEG names with >15% earnings
  growth, equal-weight, monthly"). Add a few deliberately-dumb ones. Batch through
  validation; surface survivors. With ultracode/workflows on, fan out: many backtests
  in parallel, survivors adversarially re-validated, winners promoted.

### Working agreements (how the owner wants this run)

- **Keep the live system running and untouched** until the new engine is proven; all
  engine work is additive.
- **Small phases, each with tests + a `/code-review` pass** before moving on. The owner
  explicitly wants major review work to keep the code clean — bake it into every phase.
- **Keep all data. Never delete the database.** (A prior session wrongly proposed
  deleting it to fit a free tier — the opposite of the vision.)
- **Git:** commit + push directly to `main`; format `[vX.Y] short description`; bump
  `__version__` in `dashboard/app.py` for functional changes; run
  `pytest -m "not integration"` before committing code.
- **Autonomy:** finish the project autonomously, phase by phase, committing as you go;
  surface decisions only when they genuinely change direction.

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
