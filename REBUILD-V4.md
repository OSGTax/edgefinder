# REBUILD V4 — the broker-native desk (source of truth; supersedes V3)

> **Read this first on resume.** REBUILD-V3 (the hand-rolled-ledger live
> desk) is retired and archived at `docs/history/REBUILD-V3.md`. This is
> the whole design, and it is deliberately smaller than what it replaces.

## What this is

An **autonomous AI paper-trading desk**, private to the owner:

- **The book of record is an Alpaca PAPER brokerage account.** Orders are
  real orders on a real broker's paper simulator: they fill against the
  live consolidated market (NBBO), limits rest until marketable, partial
  fills happen, protective stops sit on Alpaca's own book and fire with
  nothing of ours running. Positions, cash, equity, corporate actions,
  and option expiry are the broker's job now.
- **An agent-paced AI brain** (Claude Code Routine sessions running
  `.claude/skills/trading-agent/SKILL.md`) that observes, researches,
  decides, submits orders via `agent.trade`, and **sets its own clock** —
  every cycle plans its next wake 15–60 minutes out (`brain wake-plan`,
  the budgeted `desk_wakes` ledger); the always-on Render process fires
  each plan when due through the "EdgeFinder chain wakes" Routine's API
  trigger (V4.1 — fired Routine sessions have no scheduler tools, probed
  2026-07-13 and re-proven 2026-08-10, so the dispatcher is the chain's
  clock), with an hourly floor Routine restarting a dropped chain.
- **Judged on the calls themselves**: the decision-side registry —
  falsifiable prediction/horizon/kill on every entry, structured
  commitments on conditional exits, tier-gated claims, owner-approved
  proposals — survives V3 unchanged. It was always the point.
- **Self-evolving**: the agent owns its strategy (versioned, every pivot
  journaled); nightly Routines keep the data fresh and the strategy lab
  sweeping; the app-evolver ships one small announced UI improvement per
  run.

## The honesty contract V2 (why this is trustworthy)

1. **A real broker executes every fill.** Nothing here invents a price:
   orders go to `paper-api.alpaca.markets` and fill (or don't) against
   the live NBBO. Our record of a fill is the broker's record, mirrored
   locally (`desk_orders`/`desk_activities`) and re-synced every cycle —
   the mirror is a cache, never the arbiter.
2. **What the simulator does NOT simulate is disclosed, never papered
   over:** no market impact or slippage (fills at the touch regardless of
   size), no commissions, **no dividends** (the book is price-return),
   and option expiry/assignment activity records land T+1. The desk page
   carries these disclosures, including an estimated-missed-dividends
   counter computed from our own dividend data.
3. **The scoreboard cannot flatter.** SPY comparisons are **symmetric
   price return** on both sides — the paper book earns no dividend yield,
   so the benchmark doesn't get to carry its own. (V3 was total-return
   both sides; symmetry is the invariant, the basis moved with the book.)
   Every buy/add pick still registers a falsifiable prediction + horizon
   + kill before the decision saves; grading still runs from machine
   facts (`agent.grade` → `desk_outcomes`), never from memory.
4. **Risk shape is enforced by the PLATFORM, not by promises:** the
   account is configured `no_shorting` (long-only), margin multiplier 1
   (no leverage), options Level 3 maximum — Alpaca has no naked-call tier,
   and multi-leg orders are accepted only with every leg covered. The
   defined-risk charter is now a property of the account itself.
5. **Paper-only by construction.** The runtime holds ONLY the paper
   account's trade keys (`EDGEFINDER_ALPACA_TRADE_KEY/_SECRET`); paper
   keys cannot authenticate against the live API, so live trading is
   unreachable no matter what the code does. The data keys
   (`EDGEFINDER_ALPACA_API_KEY/_SECRET`, Algo Trader Plus) are read-only
   by design — `agent/broker.py` still has no write methods; the ONLY
   module that submits orders is `agent/trade.py`, and a repo-wide test
   enforces that.
6. **Sacred, never drop/clear:** the market-data tables (`daily_bars`,
   `dividends`, `ticker_splits`, `fundamentals_pit`, `ticker_news`) and
   the R2 parquet archive — plus, since V4, the nightly R2 backup of the
   whole knowledge layer, so no hosted-database pause can ever strand
   what the desk has learned.
7. **The agent cannot teach itself into authority** (unchanged from V3):
   the tiered claims registry (`desk_claims`) with pre-registered
   promotion criteria, structured commitments machine-swept from stored
   closes, owner-approved proposals for cap raises and pivots,
   supersede-never-delete, `agent.knowledge lint` + `loop-report`.

## The era model

- **Era 1** (2026-07-07 → cutover): the V3 hand-rolled ledger. Frozen at
  cutover — `desk_trades`/`desk_positions`/`desk_equity` renamed to
  `era1_*`, every open outcome row closed with `exit_kind='cutover'` at
  the final mark, the full database exported to R2. Still rendered on
  the desk, read-only, forever.
- **Era 2** (cutover →): the Alpaca paper account, funded at Era 1's
  final marked equity so the all-time P&L story stays one line. The
  first Era-2 cycle re-entered only what the agent still believed in, at
  live prices, as fresh picks — carrying a position is itself a call.
- The knowledge layer (claims, wiki, journal, decisions, outcomes,
  commitments, proposals) is continuous across both eras.

## Runtime layout

| Piece | Where | Job |
|---|---|---|
| Paper account | Alpaca | THE BOOK: fills, positions, cash, equity, resting stops, expiry |
| Trading brain | Claude Code Routines (owner's subscription) | Wake-plan chain 15–60 min apart, fired by the Render dispatcher via the chain-wakes Routine's API trigger + hourly restart floor; runs the trading-agent skill; orders via `agent.trade` |
| Quote streamer | Render (always-on) | SIP WebSocket → QuoteCache → SSE live tape on `/desk` (display + research; fills don't depend on it) |
| Desk page | Render | Live ticks, the book (from Alpaca), decisions, thinking, claims, journal, What's New |
| Nightly data | Claude Code Routine | `data-refresh`: whole-market ingest + EDGAR + brief + **mirror sync, portfolio snapshot, split guard, R2 backup, DB size check** |
| Strategy Lab | Claude Code Routine | nightly 21y split-sample sweep → leaderboard |
| Reflection | Claude Code Routine (Fri) | grade the week, curate the wiki, lint the registry |
| App evolver | Claude Code Routine (Sat) | one small announced `/desk` improvement |
| Book mirror + knowledge | Supabase (free tier) | `desk_orders`/`desk_activities`/`desk_portfolio_history` + the knowledge layer; nightly R2 backup |
| Deep history | Cloudflare R2 | 21y parquet archive + `backups/` (knowledge + irreplaceable tables) |

There is no GitHub Actions execution arm, no wake-dispatcher, no tripwire
sweep, and no in-house hard-stop executor — protective exits are real GTC
stop orders resting at the broker (with the one caveat the skill manages:
Alpaca auto-cancels GTC orders at 90 days; `agent.trade reconcile` warns
from day 80).

## Credentials model (two key pairs, never interchangeable)

| Pair | Env vars | Powers | Cannot |
|---|---|---|---|
| DATA (ATP) | `EDGEFINDER_ALPACA_API_KEY/_SECRET` | SIP quotes, bars, news, OPRA chains, clock | place orders (no write methods exist in broker.py) |
| TRADE (paper) | `EDGEFINDER_ALPACA_TRADE_KEY/_SECRET` | orders/positions/account on the PAPER API | authenticate against the live API at all |

## Costs (the point of the exercise)

Alpaca ATP $99/mo (the live data — kept), Render (the always-on tape +
desk page), Supabase free tier (knowledge + mirror, guarded by the size
check and the R2 backup), R2 (archive + backups), and the owner's Claude
subscription (every agent session — no Actions minutes, no dispatcher
PAT, no SMTP).

## Rules for future sessions

- Never force-push `main`; never skip the test gate
  (`DATABASE_URL= python -m pytest tests/ -q -m "not integration"`).
- Version-bump `dashboard/app.py` on every functional merge.
- The trading skill may not touch UI files; the app-evolver may not touch
  `agent/trade.py`, `agent/grade.py`, the mirror tables, or any sacred
  table.
- `agent/trade.py` is the ONLY module that may call Alpaca order writes —
  `tests/test_live_fill.py`'s allowlist test pins it.
- Durable context lives in files (this charter, the skills, CLAUDE.md),
  not chat.
