# The Skeptic's Audit — EdgeFinder, torn down by the friend who only sees the sticky parts

*2026-07-16. Written as the merciless desk head you asked for: every claim below
was adversarially verified against the current code by independent reviewers
(each instructed to refute it), or taken directly from the production database
with SELECT-only queries. Where a finding turned out smaller than it first
looked, it says so — a skeptic who exaggerates is just a different kind of liar.*

---

> ## Remediation status (added same day, after the fix run)
>
> Every finding below was remediated on this branch across ten commits
> (v9.6.0–v9.10.1), each phase implemented by one agent and adversarially
> re-reviewed by another before landing; the reviews caught and fixed
> nine further bugs the fixes themselves introduced (notably: a PIT-probe
> collision that would have mislabeled an ETF-only universe as honest, a
> hard-stop/split interaction that destroyed value, a double-sell race,
> and sanity gates that would have vetoed protective exits mid-crash).
>
> - **P0 scoreboard** (F1–F4) → v9.6.0/v9.6.1 — TR benchmark + regression
>   test, PIT plumbing + survivorship labels, full-market corp actions,
>   split-guarded screens.
> - **P1 book + reflexes** (F5, F6, F8–F10) → v9.7.0/v9.7.1 — corp actions
>   on the live book, expiry-day settlement, post-market session fix,
>   enforced prediction registry, opt-in hard stops with CAS claims.
> - **P2 measurement** (F11–F14 + loop closure) → v9.8.0/v9.8.1 — fill
>   friction/sanity gates, mark provenance, desk_outcomes + grade/verdict/
>   context commands, charter truth pass.
> - **P3 durability** (F17–F19, F21) → v9.9.0/v9.9.1 — safe nightly
>   writes, finalized-bars R2 sync + fingerprints, RTH-only IV bank,
>   wiki history + setups/postmortems pages.
> - **P4 desk** (F15, F16, F20, F22, F23) → v9.10.0/v9.10.1 — predictions
>   panel, decision archive, SPY overlay, honest LIVE pill, one book,
>   options allowlist + rate limit, CORS fix, preflight --strict.
>
> Deferred (not silently skipped): dropping the dead legacy prod tables
> (destructive — owner call), true 2006-era PIT universe ranking from R2
> depth (data engineering; leaderboard rows are honestly labeled until
> then), Routine-trigger wiring so tripped wires fire a run (lives outside
> the repo; hard stops cover the protective case), and the prod schema
> migration (applies automatically via render_start on the next deploy;
> new code degrades gracefully until then).

---

## 0. The 60-second verdict

You built an honest cash register attached to a rigged scale.

The **ledger is genuinely good** — append-only fills at live quotes with the
quote stamped on every row, cash replayed from scratch on every read (I
recomputed all 16 fills by hand: $655.88, to the cent), options rules that
actually reject naked risk, a backtest engine with no look-ahead and a cost
model more conservative than most professionals use. The narration layer is
honest to a fault — the wiki's *Mistakes* page wrote up its own GEV churn
before I found it.

But the **research layer the trades are grounded on is structurally
flattering**, in three compounding ways: the lab benchmarks total-return
strategies against price-only SPY (a free ~50pp head start over the in-sample
half — *every* strategy gets it), it backtests **today's** winners across 20
years of history (survivorship — the overnight sweep printed an +855pp
"out-of-sample excess" and nobody blinked), and the discovery screens compute
3-month returns on raw closes in the one market segment where reverse splits
are common, so a 1:10 reverse split prints as a +900% "momentum leader" in the
brief the agent is required to shop from.

And the **learning system is a notebook, not a brain**: nothing in code makes
the agent read the wiki, fill in the predictions the skill calls REQUIRED
(production picks carry `prediction: null` today), grade outcomes, or act on a
tripped kill. Every arrow in the learning loop is prose asking a stateless
LLM to please behave. Meanwhile the live book's 8 days of actual trading show
exactly the failure mode the missing enforcement predicts: 99.8% deployed on
day one into one correlated momentum bet, entries chased on +3.9% and +6.3%
days, "grounding" backtests run on the already-chosen winners over the exact
years they won, and a headline alpha number (−2.18%) I could not reproduce
from the desk's own stored data.

None of this is fatal. Almost all of it is fixable, most of it cheaply. But
until the scale is fixed, "the backtest says it works" means very little, and
making this system *more aggressive* before making it honest would just be
paying more to lie to yourself faster.

---

## 1. Findings index

| # | Severity | Finding | Status |
|---|---|---|---|
| F1 | **P0** | Lab/backtest excess vs SPY inflated ~50pp (in-sample) by dividend-basis mismatch | Confirmed, adjusted **upward** |
| F2 | **P0** | Lab universe is today's top-N over 2006–2025 — survivorship/selection look-ahead | Confirmed |
| F3 | **P0** | Splits/dividends never ingested for the wide universe — "adjusted" research data isn't | Confirmed (new) |
| F4 | **P0** | Brief screens rank raw-close 3-month returns — split artifacts fabricate leaders | Confirmed (new) |
| F5 | **P1** | No corporate actions on the live book: splits mismark catastrophically, dividends vanish | Confirmed |
| F6 | **P1** | Prediction registry unenforced — production picks carry nulls; "BOOK" pseudo-picks | Confirmed + prod evidence |
| F7 | **P1** | Learning loop is not code-closed: wiki/outcomes/brief consumption is voluntary prose | Confirmed |
| F8 | **P1** | Tripwires and wakes record but never act; nothing fires a run between owner pokes | Confirmed |
| F9 | **P1** | Post-market session detection is dead code — 16:00–20:00 exits silently impossible | Confirmed (new) |
| F10 | **P1** | Option expiry settles at next-run price, not expiry-day close (free weekend look-back) | Confirmed (new) |
| F11 | **P2** | Live fill sanity band is a tautology; no independent last-close check on live fills | Confirmed |
| F12 | **P2** | Live book models ~zero trading friction beyond spread+1bp; no size/liquidity gate | Confirmed |
| F13 | **P2** | Marks: mid-based, no staleness stamp, cost-basis fallback bakes fake-flat P&L into history | Confirmed, honest sizing |
| F14 | **P2** | Crypto enabled end-to-end but absent from the charter; weakest guards of any asset class | Confirmed (new) |
| F15 | **P2** | The desk hides the scoreboard: predictions, per-pick outcomes, decision archive all invisible | Confirmed |
| F16 | **P2** | Three parallel book implementations (ledger, router, JS) that already disagree in details | Confirmed |
| F17 | **P3** | Nightly ingest = non-atomic 400-day delete-then-insert on a "sacred" table | Confirmed (new) |
| F18 | **P3** | R2 archive can permanently keep partial intraday bars; same-date corrections invisible | Confirmed (new) |
| F19 | **P3** | IV bank: first write of the day wins, no session guard — pre-market garbage locks in | Confirmed (new) |
| F20 | **P3** | LIVE pill lies when stale; equity chart draws synthetic points; "6 sessions" alpha not reproducible | Confirmed + prod evidence |
| F21 | **P3** | Wiki capacity: 4 pages / 12k chars, destructive in-place edits, no revision history | Confirmed |
| F22 | **P4** | Preflight "ok" means almost nothing; single process owns tape+dashboard; REST lane non-transactional | Confirmed, partly mitigated |
| F23 | **P4** | Public options endpoint fans out to paid Alpaca calls on the trading keys; wildcard CORS | Confirmed (new) |
| F24 | — | Downgraded after verification: staleness-`None` bypass, REST-vs-stream quotes, /ask spread math | Real but ≈$0 impact |

Sections 2–8 are the details. Section 9 is the aggression roadmap, section 10
the knowledge roadmap, section 11 the prioritized fix list.

---

## 2. "Are we sure this calculates correctly?" — the scale is rigged in the lab, honest at the register

### F1 — Every lab strategy gets a free ~50pp head start (P0)

The single worst number in the system. Strategy bars load
**dividend-adjusted** (total-return basis) in both the ad-hoc backtest tool
and the nightly lab sweep (`agent/backtest_tool.py:331-336`,
`agent/lab.py:130`). The SPY benchmark on the other side is **price-only** on
both transport lanes (`agent/data.py:355`; `edgefinder/engine/data.py:392-411`).
The engine then reports `excess_return_pct = strategy_total_return −
SPY_price_return` (`edgefinder/engine/backtest.py:438-443`).

The verifier sharpened this beyond the original claim: the inflation equals
**SPY's missing dividend contribution, so every strategy gets it — even one
holding only zero-dividend names**. Compounded over the 2006–2018 in-sample
half (SPY price ~2.1×, ~2%/yr yield) that's roughly **+50pp** of phantom
excess; over the 2018-present half, roughly **+30pp**. Because the bias sits
in *both* halves, the lab's split-sample maximin scoring — its whole honesty
mechanism — is structurally blind to it. A no-edge strategy that merely
tracks the market's total return **qualifies in both halves** on the
benchmark's missing dividends alone. `excess_sharpe` and
`drawdown_reduction_pct` inherit the same bias (`backtest.py:440-448`).

A real trader's read: the lab's `qualifies: true` currently means "beat SPY's
price return," which is 1.5–2pp/yr easier than beating SPY. Every leaderboard
number, every "grounded in backtest" rationale on a fill, every brief
`lab_leaderboard` entry is quietly marked up.

**Fix (small):** benchmark against dividend-adjusted SPY — load the benchmark
through the same `div_adjust=True` path as the strategy bars (one flag at
`agent/data.py:355` plus the pg-lane `spy_series`), or compare price-vs-price.
TR-vs-TR is the correct choice. Add one regression test: a buy-and-hold-SPY
strategy must produce `excess_return_pct ≈ 0` in both halves. That test
would have caught this on day one.

### F2 — The lab backtests today's winners across 20 years (P0)

`agent/lab.py:86-96` builds every sweep universe by calling `data.universe(n)`
with **no `as_of`** — which ranks on a window ending *today*
(`agent/data.py:294-311`, rest lane `:316-349`). The 2006–2018 in-sample half
then trades a basket pre-selected for having survived and won through 2026.
The bitter irony: a point-in-time-safe ranking **already exists** in this
codebase — `resolve_universe('top', as_of=...)`
(`edgefinder/engine/data.py:57-106`), whose own docstring explains this exact
failure ("selects tomorrow's winners into yesterday's universe") — and the lab
bypasses it. The inline comment at `lab.py:53-57` admits the "survivorship
shine" and offers the mid200 universe as a partial deflator, but mid200 is
also ranked as-of-today.

The production receipts make it vivid. Last night's sweep
(`desk_backtests`, 2026-07-16 02:37–02:40 UTC):

- `regime_momentum:3@top60/weekly` → out-of-sample excess **+855.76pp**
  (disqualified only because in-sample was −17pp)
- `meanrev:8@top60/monthly` → **qualifies: true**, +66.6pp / **+292.8pp**
- `breakout:8@top60/monthly` → **qualifies: true**, +22.9pp / +98.5pp with a
  **44% max drawdown** (the weekly variant of the same rule: **−95pp**)

Nobody earns +855pp out-of-sample excess. When numbers like that come off the
line and go straight to a leaderboard, the leaderboard is a survivorship
detector, not a strategy detector. Note also the whiplash between adjacent
cells (breakout monthly +98, breakout weekly −95): that's parameter noise
wearing a lab coat.

**Fix:** thread `as_of` through the lab: in-sample half resolves its universe
as of 2006 rank data, out-sample as of 2018 (requires the pg lane or serving
the ranking from R2 depth, since the hot `daily_bars` set may not reach 2006).
Until that's done, relabel the leaderboard on the desk and in the brief —
"survivorship-inflated, directional only" — so the trading brain stops citing
these numbers as evidence.

### F3 — The "split-adjusted" research data mostly isn't (P0, new — the critic found it)

`load_bars` split-adjusts using the `ticker_splits` table
(`agent/data.py:167`, `edgefinder/engine/data.py:159-160`). But the **only**
writer of `ticker_splits` is the hourly refresh's corporate-actions pass
(`agent/refresh.py:289`), which runs **only for the ~15 streamed/held/watchlist
names, 45 days back** (`refresh.py:99-117,183`). The nightly full-market
ingest (`refresh.py:467-522`) ingests bars for ~1,000 names and **never
ingests their splits or dividends**.

Consequence: for roughly 99% of the tradable set, the "adjusted" bars feeding
the lab sweeps, the brief's trend roster, `latest_indicators`, and the
backtests that ground real fills are **unadjusted**. Every split in the
mid-tier universe becomes a fake ±50–90% one-day move in the research data.
The brief's movers section even has a split guard (`agent/market.py:83`) —
which reads the same nearly-empty table, so it guards against nothing.

### F4 — The discovery screens fabricate momentum leaders (P0, new)

`agent/market.py:151` computes the screens' 3-month return as
`closes[-1]/closes[-64] − 1` on **raw** closes (`market.py:174-180`) — in the
dollar-volume-rank-41–1000 segment where reverse splits are most common. A
1:10 reverse split inside the window prints as ~+900% `ret_3m` and tops the
`beyond_megacaps` screen; a forward split prints −50/−90% and buries a real
leader. The same file guards its 1-day movers against exactly this artifact
(`market.py:81-92` — "daily_bars stores RAW closes") and then doesn't guard
the 64-day version.

This is not hypothetical plumbing: the trading skill **requires** at least one
shortlist name from `screens` on every shopping cycle
(`trading-agent/SKILL.md:143-149`), and DDOG — an actual position — was
"screens-sourced (rank 115, ret_3m +146%)" per its fill rationale. That
number happens to be a real one (DDOG hasn't split), but the pipeline that
produced it cannot currently tell the difference between DDOG and a reverse
split.

**Fix for F3+F4 together:** add a corporate-actions pass to the nightly ingest
(Alpaca serves splits/dividends for the whole market cheaply), or compute
screen returns off the R2 archive (which merge-syncs adjusted history), or at
minimum apply the movers-style split-symbol exclusion to the screens until
the table is populated.

### F5 — The live book doesn't know corporate actions exist (P1)

Nothing rebases an open position on a split: `settle()` handles only option
expiry (`agent/ledger.py:867-942`); no other code touches `desk_positions`
except to rebuild it from fills. Hold a stock through an N:1 split and the
book keeps pre-split shares at pre-split cost while live quotes come in
post-split → a fabricated ≈−(1−1/N) unrealized loss **with no correction
path** (hold NVDA through a 10:1 and the desk reports −90% overnight).
Dividends: never credited, anywhere — a held name drops by its dividend on
ex-date and the book records a pure loss. The verifier's honest sizing: the
dividend leak on this low-yield growth book is small ($1k–1.5k/yr territory
at full deployment, partly offset in *alpha* terms because the SPY comparison
is also ex-dividend — in fact, since SPY yields more than this tech book, the
measured alpha is currently slightly **flattered**). The split hazard is the
real bomb: low frequency, catastrophic mismark, and the agent trades momentum
names — exactly the cohort that splits.

**Fix:** an equity corporate-actions pass in `settle()` (run each cycle like
option expiry): forward-split → book a BUY of (N−1)× shares at price 0
(`_compute_book` at `ledger.py:128-130` folds that in with unchanged cost
basis — the verifier confirmed the existing schema carries it, the same way
expiry settlement already piggybacks on BUY/SELL rows with
`fill_quote.src='expiry_settlement'`, `ledger.py:857-864`); dividend →
a SELL row with `shares=0, dollars=amount, src='dividend'`. Both are
append-only, auditable, and keep the cash-replay invariant.

### F10 — Expiries settle at "whenever the agent next woke up" prices (P1, new)

`settle()` only settles contracts with expiry **strictly before today**
(`ledger.py:890`) and prices the underlying at the **live mid at run time**
(`ledger.py:893`). A Friday expiry settles Monday at the earliest, at
Monday's price. A contract that finished OTM at Friday's close but gapped ITM
over the weekend gets "exercised" at Monday intrinsic — a free look-back the
real market never gives (and vice versa: a Friday winner that gaps back
settles worthless). Since the agent is owner-paced with no guaranteed weekend
run, **every Friday expiry is exposed**. The options book is empty today, so
this has cost nothing yet — fix it before the options toolkit gets used in
anger. **Fix:** settle at the expiry date's stored daily close (it's in
`daily_bars`), not the current mid.

### F13 — Marks: directionally honest, historically unfixable (P2)

Marks price at mid (`_live_mids`, `ledger.py:828-848`) with no staleness gate
(staleness is enforced on fills only) and silent error-swallowing; the
fallback chain is live mid → latest local daily close → **cost basis**
(`ledger.py:961`). Verification sized this honestly: equities almost always
catch the daily-close fallback, so the cost-basis branch effectively fires for
**options and crypto** when the quote call fails — which is exactly where
you'd least want a position silently marked flat. The sting: any
`desk_equity` snapshot written during such an outage embeds the fake-flat
value **forever** — equity history is never restated. Mid-marking also
overstates liquidation value by half the spread (trivial on AAPL; real on
wide OPRA contracts). **Fix:** stamp each equity snapshot with per-position
mark sources (`live|close|cost`), refuse to write a snapshot that marks >X%
of book value at cost, and mark options at bid for longs / ask for shorts if
you want the curve to mean "what could I actually get."

### F11 + F12 — The register is honest about price, silent about everything else (P2)

The live fill's "sanity band" checks the computed price against the same
quote it was computed from (`record_trade` band `ledger.py:593-612` vs price
construction `:790-791`) — it can only ever catch a code bug, never a bad
print. The 25% last-close band exists but applies **only to the legacy
no-quote branch** (`:616-623`). So the live path's real protections are the
spread cap and the crossed-quote check; a halted name reopening 40% away
with a tight spread fills without comment. Meanwhile `bid_size`/`ask_size`
are captured and never used — a $100k clip fills at the touch no matter what
the quote is good for — and the only friction is spread + 1bp: no regulatory
fees (trivial: ~$0.28 per $10k sold), no OPRA per-contract fees, no impact.
For $5–20k clips in top-40 liquidity names this flatters you by only a few
dollars a trade — the honest sizing — but it means **live "alpha" is earned
on easier terms than lab alpha**, the exact opposite of how a research desk
should be calibrated, and it will matter the day the agent takes the screens
seriously and buys a rank-800 name pre-market (see DDOG, section 6).

**Fix:** reject fills >20% from the latest stored close absent an explicit
override flag; cap notional at a multiple of displayed quote size or recent
ADV; add the ~$0.65/contract OPRA fee when options trading starts.

### F24 — Three findings that got honestly downgraded

Adversarial verification killed the drama on three claims worth keeping only
as one-line fixes: (a) a quote with a missing/unparseable timestamp bypasses
the staleness gate (`ledger.py:778-788`) — real fail-open, but Alpaca quotes
always carry timestamps; expected impact ≈ $0; still make `age is None` a
rejection. (b) Fills price off a fresh REST quote rather than the streamer
cache the desk displays — same SIP feed, milliseconds apart, arguably *more*
correct, and the fill stamps its own quote for audit; not a violation.
(c) Spread% divides by ask instead of mid (`:770`) — loosens the equity gate
by ~0.13pp at the cap; only non-trivial for options, where the 50% cap is
effectively 66% of mid.

---

## 3. The backtest citations in your fill rationales are theater

Separate from the engine's math: look at how backtests are actually *used*.
From the production ledger, verbatim patterns:

- NVDA add: "momentum:5 on {NVDA,AMD,AVGO,MU,SMH} 2021-2026 beat SPY by 458%"
- AVGO: "momentum:6 on {AVGO,MU,NVDA,QQQ,SPY} 2022-01: +447% vs SPY +57%"
- DDOG: "momentum:5 monthly on book+DDOG: +376.2% vs SPY +100.3% since 2021"
- GEV: "momentum:3 with GEV in the roster: +128% vs SPY +58% since 2024-01"

Every one backtests **the names already chosen**, over **a window starting
when those names started winning** (2021, 2022, 2024). "Would a momentum rule
have made money holding NVDA/AMD/AVGO since 2021?" is not a question — it's a
tautology wearing a methodology. The 21-year archive and the split-sample
discipline exist precisely so the agent doesn't do this, and the skill's own
grounding step points at the lab leaderboard first — yet the fills cite these
pocket backtests as evidence. Worse, the day-one basket rationales literally
say "**basket grounded vs SPY null**" — the grounding backtest returned
nothing, and $80k was deployed anyway.

**Fix (process, cheap):** the skill should require grounding backtests to (1)
start no later than 2015 or span two regimes, (2) exclude the candidate's own
trailing-12-month period, or (3) cite a *lab* row (once the lab is fixed) —
and `save_decision` should reject a buy pick whose `evidence` cites a
backtest window shorter than N years. Until then, treat every "+400% vs SPY"
in a rationale as decoration.

---

## 4. The honesty contract, audited line by line

What the charter promises vs. what the code does. Credit first: the core
promise — *every fill prices off a live quote at the decision moment, with
the snapshot stamped on the row* — **holds**, and it's the best thing in this
system. All 16 production fills carry `{bid, ask, mid, t, src, slippage_bp}`
and reconcile: SELLs at bid×0.9999, BUYs at ask×1.0001, to four decimals.
Alpaca is data-only as promised (`broker.py` has no write methods). Cash
replays exactly. That's rarer than you'd think.

Now the gaps:

1. **"Refuses to book when the market is closed"** (CLAUDE.md) — literally
   true, but extended-hours equity fills are allowed by design
   (`MAX_SPREAD_EQ_EXT=0.02`), and neither CLAUDE.md nor REBUILD-V3 mentions
   it. The skill discloses it; the two summary contracts don't. Verified
   verdict: a doc-surface gap, not a violation — but the owner reading the
   charter would not expect trade #69 (DDOG) to have been booked at 9:13 AM.
2. **F9 (new, and rich): the after-hours branch is dead code.**
   `broker.session()`'s post-market condition requires the current ET date to
   equal the *next close's* date after 16:00 — but Alpaca's `next_close` is
   always the next trading day once today's bell rings, so the conjunction is
   unsatisfiable (`agent/broker.py:453`). Only the **pre-market** branch
   works (`:447`). The skill tells the agent it can act 04:00–20:00; in
   reality a 16:30 earnings-blowup exit is silently rejected as "market
   closed," and the fake-session tests (`tests/test_live_fill.py:253-281`)
   only cover regular/closed, so nothing catches it. The system's advertised
   after-hours defense **does not exist**.
3. **F14 (new): crypto is live and uncharted.** The skill puts "crypto on the
   menu, 24/7" and the ledger has a full crypto path with the *weakest* gates
   of any asset class (3% spread, no session gate, no size cap) — while
   REBUILD-V3's non-negotiable contract enumerates "equities long-only,
   options defined-risk" and never says the word crypto. And crypto sits
   entirely outside the evidence loop: no bars in `daily_bars`, no R2 series,
   no SPY benchmark, no lab coverage. Nothing stops a Saturday-night DOGE
   position that no backtest, brief, or reflection can even see. Either
   charter it with rules or turn it off.
4. **"Tripwires protect the book while you sleep"** — they don't (F8). The
   streamer sweep flips `desk_watch.status` to `'tripped'` and logs
   (`agent/streamer.py:271-308`). No sell, no notification, no run trigger.
   With owner-fired scheduling, a kill can trip at 10:00 and sit ignored all
   day. The skill even admits it: "a kill you ignore is a lesson you chose
   not to learn."
5. **"Hourly brain"** — REBUILD-V3 still says it; `docs/ROUTINES.md` and the
   skill say the owner is the scheduler. The charter contradicts the runtime.
6. **The LIVE pill lies** (F20): the desk unhides the pulsing LIVE dot on
   every 1-second SSE frame regardless of quote age
   (`dashboard/static/js/pages/desk.js:470-474`); the only watchdog reacts to
   missing frames, and frames never stop. Warmed REST quotes are stamped
   `recv=now` (`streamer.py:98-108`), so a closed market can read as a live
   tape. The equity chart also **draws ~240 synthetic points/hour** past the
   last real mark (`desk.js:18-24, 233-244`) — client-side only, never
   persisted, but on a page whose brand is honesty, fabricated pixels are a
   bad look.
7. **The headline alpha doesn't reconcile** (F20). The 2026-07-15 20:13 UTC
   decision says "book −1.82% vs SPY +0.36% over 6 sessions," alpha −2.18%.
   From the desk's own stored SPY closes, since-inception SPY was +0.07%
   (through 7/14's close) or +0.47% (through 7/15's); +0.36% matches neither
   documented convention (it happens to equal the 7/13→7/14 single session).
   Honest alpha at that moment was ≈−1.9% or ≈−2.29%. A quarter-point
   discrepancy on a six-day-old book is small; a scoreboard number that
   can't be reproduced from the system's own data is not.

---

## 5. The "learning" system is a notebook, not a brain

The design says: decide → predict → score → reflect → wiki → next decision.
Here's each arrow's actual status, verified:

| Arrow | Status |
|---|---|
| decision → prediction | **Data capture only.** `save_decision` (`agent/brain.py:124-152`) accepts null `prediction/horizon_days/kill`. "REQUIRED" lives in skill prose. |
| prediction → outcome | **Real and tested** — `ledger.outcomes` (`agent/ledger.py:258-432`) computes per-pick alpha vs SPY windows. The best artifact in the loop, because it's computed from fills, not narrated. But it only *passes the prediction through*; nothing mechanically grades predicted-vs-happened. |
| outcome → reflection | **Prose.** The Friday skill tells the LLM to run `outcomes` and grade TRUE/FALSE/NOT-YET. If the Routine doesn't fire or writes nothing, nothing notices. |
| reflection → wiki | **Prose** (the write mechanism `set_wiki` is real, size-capped, auto-journaled — good bones). |
| wiki → next decision | **The open end.** The only non-test caller of `get_wiki` is the brain CLI (`brain.py:530`); the dashboard queries the ORM directly for display. **No code injects the wiki, the brief, or past outcomes into a cycle's context.** A stateless model each run *may* read its notebook, if it follows instructions, every time, forever. |

And production shows the slippage already happening, 8 days in:

- Recent picks carry `prediction: null, horizon_days: null, kill: null` — the
  registry the whole grading system hangs on, empty, accepted without
  complaint.
- Hold decisions are logged as picks with pseudo-symbol `"BOOK"` — which
  means the outcomes engine will happily try to score a ticker named BOOK
  against SPY windows, and the registry mixes real picks with bookkeeping.
- The reflection agent ran Thursday *and* Friday in week one (spec says
  Friday post-close). Harmless, but it shows the schedule is vibes.
- The wiki after one week: 4 pages, ~8.4k of its 12k characters. The content
  is genuinely good — candid, specific, plain-English — but the ceiling is
  **12,000 characters for the accumulated knowledge of a system whose stated
  goal is to accumulate as much trading knowledge as possible**, and edits
  destroy the previous revision (no history table). One overzealous
  "curation" pass and a hard-won lesson is gone without trace.

The honest label for today's architecture is: *a well-designed memory with a
voluntary reader*. Section 10 is about making the reader involuntary.

---

## 6. The trading record, graded like a PM review

Eight days, 16 fills, −1.82%, self-reported alpha −2.18%. Too early to grade
the *returns*; exactly the right time to grade the *behavior*. From the
ledger:

**What a review board would flag:**

1. **All-in on day one.** 99.8% of capital deployed in the first session
   (4×$20k + $19.8k next day), no scaling, no dry powder. The book then
   fought a semis rout with 0.7% cash — every subsequent idea (AVGO, AMD,
   NVDA, DDOG) had to be funded by selling something. For a discretionary
   book, cash is optionality; this desk sold all of it on day one at what
   turned out to be a local top.
2. **One trade wearing eight tickers.** QQQ 20.5%, IWM 20.2%, AAPL 16%,
   NVDA 9.7%, AVGO 9.6%, AMD 8.8%, DDOG 4.9% — call it 85%+ exposure to a
   single factor (US large-cap momentum beta) plus LLY. The 20%-concentration
   bear-case rule watches *names*, not *factors*, so the book passed every
   check while being, functionally, one position.
3. **Chasing prints.** NVDA bought on a +3.9% day, added four days later;
   AMD bought +6.3% intraday ("confirming base-and-turn today"); DDOG bought
   on a breakout, pre-market, paying the ask across a **0.60% spread**
   (bid 273.25/ask 274.90, filled 274.93) — the single most expensive
   execution decision on the ledger, spending ~60bp before the trade began,
   in the session where the at-the-touch fill model is most fictional.
4. **Churn without a rule.** GEV: bought 14:11, sold 17:29 the same day
   ("no active theme, −17 unrealized" — i.e., no rule fired; it changed its
   mind), paying the spread twice for −$2. GOOGL: $20k entry, fully exited
   within 2 sessions for −$887. To its credit the wiki's *Mistakes* page
   wrote both up — the self-awareness layer works; the discipline layer
   doesn't.
5. **Grounding theater** (section 3): every backtest citation is the chosen
   winners over their winning years, and day one went ahead on a literal
   `null` grounding result.
6. **Strategy identity churn:** v2→v3→v4→v5 in three days (two owner-directed
   — that's you nudging aggression mid-week, which the journal transparently
   records — but a strategy that re-versions daily has no baseline to be
   graded against).

**What deserves genuine credit:** exit rules exist and fire mechanically
(rule (b) exits are documented with the arithmetic in the rationale); kills
are armed with stated buffers and re-armed on rallies (AAPL kill tightened
$313→$317 to lock a +2.1% day); the mid-rout hold on 7/15 (AMD −6.7%
intraday, no level broken, held, correct) was narrated with the exact logic
"selling mid-rout without a broken level is how momentum books bleed" —
that's real trader discipline, in prose that a human can learn from. The
narration quality throughout (287 thinking rows, phase-tagged; 13 journal
entries) is the best part of the product.

---

## 7. What /desk hides from you

For a system whose owner wants to *learn from reading it*, the desk buries
its most educational data:

- **The prediction registry is invisible.** `prediction`, `horizon_days`,
  `kill` are stored on picks and returned by the API, but `loadDecision`
  renders only action/why_now/rationale/evidence/news
  (`dashboard/static/js/pages/desk.js:624-651`). The one falsifiable thing
  the agent writes never reaches your eyes.
- **No outcomes anywhere.** `ledger outcomes` — per-pick alpha vs SPY, the
  single most trustworthy artifact in the system — has **no endpoint and no
  panel** (grep: no `/outcomes` route in `desk.py`). The desk shows one
  since-inception alpha scalar and nothing per-decision.
- **No history.** Only `/decision/latest` exists — yesterday's dossier is
  unreachable. Trade receipts are capped at 20 rows with 60-char rationales
  (full text only on hover), while the API happily serves 1,000.
- **No benchmark on the chart.** The equity curve plots alone; the SPY
  comparison that defines success is a hero-tile scalar, not an overlay.
- **Three books** (F16): the ledger, the `/portfolio` router
  (`desk.py:60-112` — deliberately copies the ledger's SPY convention per its
  own comment, but already differs in rounding and multiplier handling), and
  the JS live fold (`desk.js:404-490`, recomputing equity/alpha client-side
  against cash frozen at the last poll). Three implementations that agree
  *most* of the time is how dashboards teach owners to distrust them.

---

## 8. Operational fragility (with honest mitigations)

- **Preflight "ok" ≈ "the database answered."** Only `db` is critical
  (`agent/preflight.py:41`); stale universe, dead R2 env, broken siblings all
  return ok:true/exit 0. Partial credit: `research_ok` is surfaced separately
  and the skill honors it as a degrade gate — but nothing checks Alpaca
  reachability, streamer liveness, or the dashboard's pg lane, so "ready to
  trade" is asserted without checking the things trading needs.
- **One process owns the tape and the site** (`dashboard/app.py:37`), and
  Alpaca allows one SIP socket — deploy overlap 406s the stream. Verified
  mitigation: after 8s of backoff the cache re-warms from REST snapshots, so
  the tape degrades to delayed-snapshot mode rather than going dark, and
  fills are unaffected (they quote REST directly). Transient, real,
  survivable.
- **The REST lane is non-transactional** (trade insert and position rebuild
  are separate HTTP calls; writes never retry, `agent/rest.py:107-119`) and
  the skill books sells before buys — a mid-cycle death can leave a
  half-rotated book with no decision row and no mark. Cash always
  re-derives correctly (append-only saves you again), but the desk shows
  changed positions with no explanation until the next run.
- **F17 (new): the nightly ingest's normal write is a 400-day DELETE then
  INSERT per symbol, non-atomic, on `daily_bars`** — the table CLAUDE.md
  calls sacred/never-clear (`agent/refresh.py:448-464,501`). Delete lands,
  all three insert retries fail (the REST lane's documented failure mode) →
  that symbol silently loses its hot-set history; per-date coverage checks
  don't notice per-symbol holes. Upsert, or insert-then-prune.
- **F18 (new): the R2 archive inherits intraday bars.** Hourly merge-sync
  pushes today's in-progress bar with `db_max=today`; the corrected close
  only re-syncs when a later date advances the fingerprint — a symbol that
  drops out of the sync universe that day keeps a 10:30 AM "daily bar"
  forever in the 21-year archive backtests read as truth
  (`refresh.py:419-427,604-607`).
- **F19 (new): the IV bank locks in the first snapshot of the day**
  (`options_data.py:146-164` — "first write of the day wins"), with no
  session guard on the chain fetch; an agent that wakes at 07:00 permanently
  records pre-open crossed-quote IV/expected-move as the day's row, with no
  capture-time column to even tell you later.
- **F23 (new): the public, unauthenticated `/api/desk/options/{symbol}`
  fans out per novel symbol to live paid Alpaca calls on the same keys the
  ledger needs to price fills** (`desk.py:390-397`), 60s cache per symbol,
  no allowlist, no per-IP throttle — a stranger iterating tickers can
  rate-limit your agent out of its own executions at decision time. Plus
  wildcard-with-credentials CORS (`app.py:56-62`), harmless only until auth
  exists. Allowlist to held/watched symbols and throttle.
- Housekeeping: prod still carries dead workbench tables (`validation_runs`,
  `promoted_strategies`, `agent_decisions`, `fundamentals`, `tickers`,
  `market_snapshots`, `system_heartbeat` at 0 rows) — none dangerous, all
  noise for anyone auditing the schema.

---

## 9. The aggression roadmap — how to trade harder without lying to yourself

You said you're open to absolutely anything. Here's the order that doesn't
blow up:

**Step 0 — fix the scale before adding weight (P0s, F1–F4).** Aggression is a
multiplier on signal quality. Right now the signal pipeline overstates edge
(~50pp benchmark gift, survivorship universe, split-fabricated screen
leaders). Size up on that and you're compounding noise with leverage.
Everything below assumes the P0 fixes land first.

**1. Make the desk able to act between your pokes (the single biggest
unlock).** Today the agent literally cannot respond to anything until you
fire a Routine — kills trip into a void (F8), after-hours exits are dead code
(F9). Aggression isn't position size, it's *reaction time*:
   - Wire `desk_wakes` and tripped `desk_watch` rows to actually fire the
     trading Routine (a one-shot trigger per wake; the streamer already
     sweeps the tape every few seconds — let a trip create the trigger).
     The 20/day, 15-min-gap budget already exists as the safety rail.
   - Add an opt-in `kind='hard_stop'` watch that the **streamer executes**
     as a SELL through the existing `ledger.fill` path with every gate
     intact. Code-enforced stops are what let a discretionary book run
     bigger positions with a straight face. Paper account = zero blast
     radius; this is the perfect sandbox for guarded automation.
   - Fix `session()`'s post-market branch so the 16:00–20:00 window the
     skill already promises actually works.

**2. Concentration with teeth.** The current implicit style is ~8 positions,
equal-ish weights, no code caps. If you want conviction sizing (25–35%
positions), pair it with *enforced* preconditions in `save_decision`/
`ledger.fill`: no buy/add pick books without a non-null prediction, horizon,
and kill; concentration above 20% requires a bear-case thinking row to exist
in the same run (queryable — enforce it). You get bigger swings AND the
falsifiable record that tells you whether they were earned.

**3. Stop paying breakout tax.** The ledger shows entries at +3.9% and +6.3%
intraday prints and a pre-market ask-lift across a 60bp spread. Cheap fixes:
extended-hours buys require spread ≤ 20bp (keep 2% for exits); a skill rule
preferring limit-style patience — "if today's gain > +3%, enter on a pullback
wake, not now" (the wake system makes this natural once #1 lands).

**4. Actually use the options toolkit.** Zero option trades to date despite a
full defined-risk engine. Long calls / call verticals are the honest way to
express the momentum conviction this book keeps buying late — defined risk,
convex, and they'd have cost less than the GOOGL round trip. Prereqs: F19
(IV bank session guard) and F10 (expiry-day settle), then let the skill's
options doctrine earn its keep. Covered calls on the index sleeves
(QQQ/IWM, 40% of book) are free aggression funding in a paper account.

**5. Downside expression.** v5 already authorizes inverse ETFs and long puts.
The book has no bear tool in use and rule (b) only *trims*. Give the
playbook a regime trigger (e.g., SPY below 200-day → up to 20% inverse/put
sleeve) and backtest it in the fixed lab, not in prose.

**6. Leveraged ETFs — with a decay rule.** Authorized since v5, but the lab
has no leveraged-decay modeling and 21 years of TQQQ-style paths will
mislead a naive backtest. Rule of thumb to encode: leveraged sleeves only in
confirmed trend + low-vol regimes, hard 15% sleeve cap, kill at −10% sleeve
drawdown, and never grounded on a backtest that doesn't include 2008/2020/
2022.

**7. Cash floor as ammunition, not safety.** Not 20% idle drag — a hard 5–10%
minimum so the next A+ setup never again requires selling something at a
3-hour-old thesis (GEV died to fund AMD).

---

## 10. The knowledge roadmap — from notebook to flywheel

**1. Enforce the registry at the write.** `save_decision` rejects buy/add
picks with null prediction/horizon/kill; `"BOOK"` moves to a separate
`stance` field (or `kind='book_note'`) so the pick registry contains only
gradeable picks. One afternoon of work; converts the wiki's grading raw
material from optional to guaranteed. (Prod already shows nulls — this decay
is happening *now*, in week one.)

**2. Grade predictions in code, not prose.** A nightly `outcomes-grade` pass
that materializes `ledger outcomes` into a `desk_outcomes` table: one row per
pick per grading date with `alpha_pct`, `prediction_verdict`
(TRUE/FALSE/NOT_YET at horizon), `kill_breached`. The Friday reflection then
starts from machine verdicts and spends its intelligence on *why*, not on
arithmetic it might fudge or skip.

**3. Close the loop mechanically.** The cycle's step-0 tooling should *hand*
the agent its memory instead of asking it to fetch: a single
`agent.brain context` command (or an extension of `preflight`) that emits
brief + wiki + open predictions + last outcomes + tripped watches in one
JSON blob, and a skill line making it the mandatory first call. You can't
force a model to *heed* its notebook, but you can make "didn't even look"
impossible — and auditable, because the command logs a thinking row.

**4. Let the wiki grow a memory.** Raise caps (8 pages / 32k is still tiny),
add an append-only `desk_wiki_history` (slug, revision, body, ts) so curation
stops destroying evidence, and add two pages: `setups` (named, statistically
tracked patterns — "50-day reclaim add", "pre-market breakout chase" — with
running win rates fed from `desk_outcomes`) and `postmortems` (one dated
entry per closed round trip, auto-skeleton created at close, filled by
reflection). That's the difference between "lessons" as essays and lessons
as a queryable track record.

**5. Show the scoreboard on /desk (this is where *you* learn).**
   - A **Predictions panel**: every open pick — prediction text, kill level
     vs live price, horizon countdown, since-entry vs SPY. The moment this
     is visible, both you and the agent live under the same falsifiable
     record. (The data already exists; it needs an endpoint and a table.)
   - A **decision archive** (`/api/desk/decisions?before=`) and a full trade
     history page with untruncated rationales.
   - **SPY overlay** on the equity chart, and mark-source flags so a
     cost-marked point renders visibly different.
   - Kill the synthetic live-tip and the always-on LIVE pill: show real
     staleness. An honesty-branded page should never draw invented pixels.

**6. One book, one math.** Delete the router and JS re-implementations of
cash/marks/alpha; serve `ledger.state()`/`outcomes()` through the API and let
the client render only. Every duplicated formula is a future "why do these
two numbers disagree" ticket — you have three today, and they already differ
in rounding.

---

## 11. The prioritized fix list

**P0 — the scale (do these before trusting or citing another backtest):**
1. Benchmark SPY total-return in lab + backtest tool (F1) + the
   SPY-buy-and-hold ≈ 0-excess regression test.
2. Thread `as_of` through lab universes via `resolve_universe` (F2); until
   then, label the leaderboard "survivorship-inflated" on the desk and in
   the brief.
3. Nightly corporate-actions ingest for the full universe (F3).
4. Split-guard or R2-source the screens' return math (F4).

**P1 — the book and the reflexes:**
5. Equity corporate-actions pass in `settle()`: split rebase + dividend cash
   rows (F5).
6. Enforce prediction/horizon/kill at `save_decision`; separate `BOOK`
   stance rows (F6).
7. Fix `session()` post-market detection + real fake-clock tests (F9).
8. Settle expiries at expiry-day close (F10).
9. Fire Routines from tripped watches/planned wakes; opt-in hard-stop
   execution in the streamer (F8).

**P2 — honest measurement of live results:**
10. Last-close sanity band + `age is None` rejection + size-vs-quote-size cap
    on live fills (F11, F12, F24a).
11. Mark-source stamps on equity snapshots; refuse cost-marked snapshots
    above a threshold; bid/ask-side option marks (F13).
12. `desk_outcomes` materialization + `brain context` single-call memory
    (roadmap #2–3).
13. Charter or disable crypto (F14).

**P3 — durability and data hygiene:**
14. Upsert-style nightly bar writes (F17); same-date R2 re-sync on fingerprint
    mismatch (F18); IV bank session guard + capture-time column (F19).
15. Wiki history table + raised caps + setups/postmortems pages (roadmap #4).

**P4 — the desk you can learn from:**
16. Predictions panel, decision archive, full trade history, SPY overlay,
    honest LIVE pill, no synthetic points (F15, F20).
17. Single book implementation served to the client (F16).
18. Options endpoint allowlist + throttle; CORS tightening (F23);
    preflight criticality review; drop the dead workbench tables (F22).

---

## Appendix: what survived scrutiny (the good bones)

So the record is fair — these were attacked and held:

- Append-only ledger with cash-replay: reproduced independently to the cent.
- Fill pricing honesty: all 16 fills reconcile with their stamped quotes.
- Options coverage enforcement (CSP reserve, covered-call locks, vertical
  checks): correctly implemented and well-tested.
- Backtest engine causality: decision on yesterday's data, fill at today's
  open, causal indicators, delisting handling — no look-ahead found.
- The backtest cost model (Corwin-Schultz + square-root impact, fixed
  parameters): genuinely conservative; the lab's *scoring design*
  (split-sample maximin) is the right idea — it's the inputs that betray it.
- The narration: thinking feed, journal, and wiki are candid, specific, and
  written in plain English, including unprompted self-criticism. The style
  rules in the skill are working. This is the foundation the learning
  system deserves to be built on.

*Every file:line reference above was re-verified against the current tree by
an independent adversarial pass; production figures were pulled read-only
from the live database on 2026-07-16. No code was changed in this audit.*
