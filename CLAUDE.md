# EdgeFinder — the autonomous AI paper-trading desk

EdgeFinder is a single **autonomous AI paper-trading agent**. An hourly
Claude Code Routine session *is* the trader: it observes live markets,
authors and evolves its **own** strategy, grounds ideas with backtests over
21 years of history, trades one **$100k long-only paper book** at **live
Alpaca SIP quotes** with full discretion, and explains every decision on the
public `/desk` trading-desk page.

**`REBUILD-V3.md` is the design charter** (honesty contract, runtime layout,
rules). The agent's operating manual is
`.claude/skills/trading-agent/SKILL.md`. This file is the map.

## The honesty contract (non-negotiable)

1. **Every fill prices off the live quote at the decision moment** — BUY at
   the live ask, SELL at the live bid (± slippage), with the quote snapshot
   `{bid, ask, mid, t}` stamped on the fill row (`fill_quote`). Never a daily
   close, never a past price, never an invented price. `agent.ledger fill`
   refuses to book when the market is closed or the quote fails sanity.
   Extended equity sessions (pre/post market) are allowed under a tighter 2%
   spread cap — a post-close BUY is an overnight hold by construction. Fills
   also clear friction gates: a quote with an unverifiable timestamp fails
   closed; a price >20% off the latest stored close, or an order over 1% of
   20-session average dollar volume, needs an explicit override flag; and
   option fills pay a flat $0.65/contract fee (settlement rows stay fee-free).
   The bands veto ENTRIES: protective exits (hard stops) traverse the same
   gates with both overrides passed explicitly and stamped on the fill's
   receipt — honesty by the receipt, never by refusing the exit. Missing,
   short, or stale (>5 sessions) reference history warns and allows.
2. **Alpaca is DATA-ONLY.** `agent/broker.py` has no write methods, by
   design — orders are never submitted to Alpaca; the book is ours
   (`desk_*` tables).
3. **Paper only. Equities long-only. Options DEFINED-RISK only** (long
   calls/puts, covered calls, cash-secured puts, vertical spreads). The
   ledger enforces every rule — naked short calls rejected, CSP cash
   reserved, spread coverage checked, covered-call shares unsellable, expiry
   settled honestly at the EXPIRY-DAY close (`agent.ledger settle`, which
   also books equity splits/dividends as explicit corp-action rows). A
   rejection is final. **Crypto is authorized** as a data-supported asset
   class outside the backtest loop — 24/7, no session gate, 3% spread cap,
   hard stops NOT available, and no lab/brief/benchmark evidence covers it;
   that limitation is stated, never papered over.
4. **Sacred, never drop/clear:** the market-data tables (`daily_bars`,
   `dividends`, `ticker_splits`, `fundamentals_snapshots`, `ticker_news`,
   `index_daily`) and the R2 parquet archive. Irreplaceable.
5. **The scoreboard cannot flatter:** SPY comparisons are TOTAL RETURN on
   both sides (lab and ledger), every equity snapshot records its mark
   provenance (live/close/cost — degraded marks recorded on every snapshot;
   desk surface in a later phase), every buy/add pick must register a
   falsifiable prediction + horizon + kill before the decision saves, and
   opt-in hard stops sell through the same fill gates as any trade (the
   entry-friction bands overridden explicitly, on the receipt).
6. **Learning is tier-gated (v9.13–v9.18, `SCHEMA.md`):** the structured
   claims registry (`desk_claims`, via `agent.knowledge`) is the source of
   truth for every behavior-influencing fact — the wiki is the narrative and
   must cite claims by `[C-n]` token. Prose can inform; ONLY claims can
   justify: a pick may cite only an active `established` (or
   `experimental`-flagged, exposure-capped) claim — enforced at decision
   save. Candidates promote only through pre-registered criteria evaluated
   against stats recomputed from `desk_outcomes` (no confidence floats
   anywhere; recorded sample sizes only). A trim/exit pick whose text makes
   a conditional promise must structure it as a commitment
   (`desk_commitments` — machine-swept by `grade`, surfaced until honored).
   Strategy pivots and cap RAISES need an owner-approved proposal
   (`desk_proposals`, `PROPOSAL-<id>` GitHub issue) or an audited
   `--no-learned-basis`; tightening is always free. Supersede, never
   delete; risk rules never decay; regime claims expire unless renewed.
   `agent.knowledge lint` + `loop-report` keep the loop observable.

## Runtime layout

| Piece | Where | Job |
|---|---|---|
| Quote streamer | Render (always-on) | One Alpaca SIP WebSocket → in-memory `QuoteCache` → SSE to `/desk` (`/api/desk/stream`) |
| Desk page | Render | Live ticks, the book, thinking feed, decisions, journal, What's New |
| Trading brain | GitHub Actions, **at the desk all day** (v9.12.0): the Render streamer machine-fires a cycle for every due wake/tripped wire; the agent runs a rolling 15–60-min wake chain from a ~9:00 ET prep cycle to a post-close wrap, with a rotating study block each cycle; a half-hour cron floor only restarts a dropped chain | Runs the `trading-agent` skill; fills via `agent.ledger fill` |
| Data refresh | Claude Code Routine, nightly | `data-refresh` skill — whole-market ingest, fresh top-N set |
| Strategy Lab | Claude Code Routine, nightly post-ingest | `strategy-lab` skill — mass backtest sweep (incl. mid-tier universe), split-sample scored, leaderboard → rebuilds the brief |
| App evolver | Claude Code Routine, nightly | `app-evolver` skill — one small announced `/desk` improvement |
| Reflection | Claude Code Routine, Friday post-close | `reflection-agent` skill — score aged ideas, prune the lessons wiki |
| Book + state | Supabase (`desk_*` tables) | Ledger (source of truth), strategy, journal, thinking, changelog |
| Deep history | Cloudflare R2 (21y parquet) + `daily_bars` hot set | Backtests via `agent.backtest_tool`; topped up nightly from Alpaca |

There is **no in-process scheduler** — scheduling is external via Claude
Code Routines running the skills above.

## Tech stack

Python 3.11+, FastAPI + Jinja2 (server-rendered desk, vanilla ES modules,
no build system), SQLAlchemy 2.0 + Alembic, pydantic/pydantic-settings,
alpaca-py (data only), websockets, boto3 (R2), SQLite (dev/tests) /
Supabase Postgres (prod). Deployed on Render
(`https://edgefinder-pm8h.onrender.com` — `edgefinder.onrender.com` belongs
to someone else, never probe it).

## Directory structure

```
edgefinder/
├── agent/                      # THE AGENT'S TOOLS (CLI, JSON out, called via Bash)
│   ├── market.py               #   observe: regime/quote/history/news/universe (local data)
│   ├── data.py                 #   the one data-access seam over the kept layer
│   ├── backtest_tool.py        #   ground ideas: parametric rules vs SPY, net of costs
│   ├── ledger.py               #   the paper book: live-quote fills, settle, mark, outcomes
│   ├── brain.py                #   strategy state, journal, thinking feed, decision, wiki
│   ├── knowledge.py            #   the claims registry: tiers/promotion, commitments,
│   │                           #   owner proposals, lint, loop-report (SCHEMA.md)
│   ├── broker.py               #   Alpaca DATA-READER (read-only): quotes, clock, chains
│   ├── streamer.py             #   the always-on SIP WebSocket → QuoteCache (Render)
│   ├── refresh.py              #   Alpaca bar ingest: hourly top-up / nightly full-market
│   ├── edgar.py                #   SEC EDGAR PIT fundamentals: ingest/coverage/validate
│   ├── options_data.py         #   chain summary (ATM IV, expected move) + IV data bank
│   ├── occ.py                  #   OCC option-symbol parse/format (pure)
│   ├── store.py                #   transport-agnostic table access (pg | rest)
│   ├── rest.py                 #   stdlib PostgREST client (HTTPS/443 for the web sandbox)
│   ├── preflight.py            #   fast readiness check before a cycle
│   ├── announce.py             #   "What's New" changelog writer (app-evolver's tool)
│   └── models.py               #   the desk_* ORM tables
├── .claude/skills/             # trading-agent (charter), data-refresh,
│                               # app-evolver, reflection-agent
├── dashboard/
│   ├── app.py                  # FastAPI app — __version__ lives here
│   ├── routers/                # pages.py (/desk, /symbol), desk.py (/api/desk/*),
│   │                           # symbols.py (/api/symbols/*)
│   ├── templates/ static/      # dark-terminal design system, no CDN, no inline styles
│   └── symbol_service.py       # chart bars (DB/R2 seam + TTL cache)
├── edgefinder/                 # KEPT data + backtest layer (audited, reused)
│   ├── core/                   #   models, interfaces, logging
│   ├── data/                   #   barstore.py (R2 archive), indicator_engine, market_data
│   ├── engine/                 #   backtest.py (pure), data.py (bars + adjustments), strategy.py
│   ├── backtest/costs.py       #   realistic cost model
│   └── db/                     #   engine.py, models.py (market-data ORM), migrations/
├── config/settings.py          # EDGEFINDER_-prefixed runtime settings
├── scripts/                    # render_start.py, bootstrap.sh (Routine SessionStart),
│                               # setup_db.py, smoke_dashboard.py, cutover/retire history
└── tests/                      # 150+ unit tests, SQLite-pinned, no prod creds
```

## Database

One Supabase Postgres database, two namespaces:

- **The agent's own tables** (`agent/models.py`): `desk_trades` (append-only
  fill ledger — **the source of truth for cash**), `desk_positions`
  (projection rebuilt from the ledger), `desk_equity`, `desk_strategy_state`,
  `desk_journal`, `desk_thinking`, `desk_decisions` (picks carry a
  prediction registry — `prediction`/`horizon_days`/`kill` — plus a
  `rejected` candidates list), `desk_backtests`, `desk_changelog`,
  `desk_options_snap`, `desk_wiki`, `desk_briefs` (the nightly research
  pack the trading cycle reads first), `desk_watch` (tripwires the
  always-on streamer sweeps against the live tape), `desk_wakes` (the
  budget ledger for self-scheduled check-ins — the brain owns its own
  attention: since v9.12.0 it runs a ROLLING CHAIN all session (every
  market-hours cycle plans the next, 15–60 min out; prep ~9:00 ET, wrap
  post-close; max 30 wakes/ET-day, 15-min gap), and the always-on
  streamer MACHINE-FIRES a GitHub Actions trading cycle when a wake
  comes due or a tripwire trips (≤45 dispatches/ET-day, ≥5-min gap) —
  `desk_dispatches` is that loop's at-most-once ledger; a half-hour
  cron floor in the workflow re-seeds a dropped chain),
  `desk_outcomes` (machine-graded pick facts written by `agent.ledger
  grade`; the weekly reflection's verdicts live here via `agent.brain
  verdict`, next to the numbers they judged), and the knowledge layer
  (v9.13+, written only by `agent.knowledge`): `desk_claims` (the tiered
  claims registry — see honesty-contract §6), `desk_claim_events`
  (append-only lifecycle audit), `desk_commitments` (structured
  trim/exit falsification clauses, machine-swept by `grade`), and
  `desk_proposals` (the owner-approval queue for learned-behavior
  changes).
- **Kept market-data tables** (`edgefinder/db/models.py`, read-only inputs):
  `daily_bars` (raw bars; splits applied at load), `index_daily` (FROZEN at
  the 2026-06 cutover — SPY benchmarks read `daily_bars` instead),
  `ticker_news`, `ticker_splits`, `dividends`, `fundamentals_pit` (SEC
  EDGAR point-in-time fundamentals, one row per FILING — written only by
  `agent.edgar`; validation gate PASSED 2026-07-14, see
  `docs/fundamentals-validation.md`), and `fundamentals_snapshots` (the
  frozen Polygon-era table, final snapshot 2026-06-10 — kept as the
  validation reference, never updated).

**Account integrity (CRITICAL):** cash is always recomputed from the
append-only `desk_trades` ledger; positions are a rebuilt projection, so the
account can never silently drift. Fills refuse to book on closed markets,
stale/degenerate quotes, or insufficient cash. The integrity logic lives
above the transport (`agent/store.py`) and is unit-tested on SQLite.

## Data sources & transports

- **Alpaca (Algo Trader Plus) is the sole live MARKET data source** — SIP
  quotes, market clock/calendar, daily bars, Benzinga news, OPRA option
  chains. Polygon was retired; nothing calls it.
- **SEC EDGAR is the fundamentals source** (`agent/edgar.py` →
  `fundamentals_pit`): free, public domain (display anything), and
  point-in-time by construction — every fact carries its `filed` date.
  ≤10 req/s with the declared User-Agent (`settings.edgar_user_agent`).
  XBRL floor ~2009. Decision + validation records:
  `docs/fundamentals-sources.md`, `docs/fundamentals-validation.md`.
- **R2 archive** — grow-only parquet mirror, one object per symbol +
  manifest, 21 years deep; backtests read it. `agent.refresh` merge-syncs
  fresh Alpaca bars into it.
- **Two DB transports**, selected by `EDGEFINDER_DB_TRANSPORT`
  (`agent/store.py`): `pg` (SQLAlchemy — Render, Codespaces, CI) and `rest`
  (Supabase PostgREST over HTTPS/443 — the Claude-web Routine sandbox blocks
  the Postgres ports; bars come from R2 over 443 on this lane). `auto`
  picks `rest` iff `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set.

## Configuration

All settings in `config/settings.py`, overridable via env with the
`EDGEFINDER_` prefix: `ALPACA_API_KEY` / `ALPACA_API_SECRET` /
`ALPACA_PAPER=true` / `ALPACA_DATA_FEED=sip`, `STREAM_SYMBOLS`,
`STREAM_STALE_SECS` (5 — staler quotes never price a fill),
`DATABASE_URL`, `DB_TRANSPORT`, `STARTING_CAPITAL` (100000). Plus the
non-prefixed `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `R2_*`
(access key / secret / endpoint / bucket) for the rest lane and the archive.

## Quick start

```bash
pip install -e ".[dev]"            # Render uses ".[live]"; Routines run scripts/bootstrap.sh
python -m agent.preflight          # DB reachable + data fresh? run before anything

# The agent's own tools (JSON out) — the skill drives these via Bash
python -m agent.brain context                     # the cycle's working memory in ONE read
python -m agent.ledger state                      # cash, positions, equity, P&L
python -m agent.market brief                      # the nightly research pack
python -m agent.market regime                     # SPY/QQQ/IWM trend + regime tag
python -m agent.market universe --top 40          # most-liquid names
python -m agent.broker quote --symbols NVDA,SPY   # LIVE bid/ask
python -m agent.broker bars --symbols NVDA --timeframe 15Min  # intraday glance
python -m agent.brain watch-set --symbol AMD --below 540 --reason "..."  # tripwire
python -m agent.brain wake-plan --at 2026-07-10T19:45:00Z --reason "..."  # budget gate
python -m agent.backtest_tool --symbols A,B,C --rule momentum:5
python -m agent.lab sweep --max-combos 80    # nightly strategy search (21y, split-sample)
python -m agent.lab leaderboard              # current honest winners
python -m agent.ledger fill --symbol NVDA --side buy --notional 5000 \
    --rationale "..." --run-id 2026-07-07T14:30   # books at the LIVE quote
python -m agent.ledger outcomes --days 14         # picks vs predictions vs SPY (alpha)
python -m agent.ledger grade                      # machine facts per pick → desk_outcomes
python -m agent.knowledge claim-list              # the claims registry (tiers = authority)
python -m agent.knowledge claim-promote --claim-id N  # pre-registered gate, code-evaluated
python -m agent.knowledge lint                    # registry integrity (run at reflection)
python -m agent.knowledge loop-report --days 7    # was knowledge written/promoted/READ?

# Dashboard
uvicorn dashboard.app:app --reload   # http://localhost:8000/ → /desk
python scripts/smoke_dashboard.py

# Tests — THE pre-commit gate
DATABASE_URL= python -m pytest tests/ -q -m "not integration"
```

## API endpoints

Pages: `/` → redirects to `/desk` · `/symbol/{sym}` chart page.

`/api/desk/*` (read-only projections of the `desk_*` tables —
`dashboard/routers/desk.py`): `portfolio` (incl. `vs_spy` since-inception
alpha), `equity`, `decision/latest`, `thinking`, `backtests`, `strategy`,
`wiki`, `regime`, `movers`, `holding-stats`, `dividends`, `quotes`,
`stream` (SSE live ticks), `options/{symbol}`, `options/{symbol}/history`,
`broker-health`, `data-health` (bar-coverage freshness — the desk pill),
`brief` (the nightly research pack), `watch` (tripwires + planned wakes —
the attention system), `claims` (the tiered claims registry), `proposals`
(the owner-approval queue), `whatsnew`, `trades`.

`/api/symbols/{sym}/bars?range=&indicators=` and `/api/symbols/{sym}/events`
power the chart page. `/api/health` returns status + version.

## Rules for every session

- **Never force-push `main`; never skip the test gate**
  (`DATABASE_URL= python -m pytest tests/ -q -m "not integration"`).
- **Version-bump `dashboard/app.py`** (`__version__`) on every functional
  merge; commit format `[vX.Y.Z] short description` (`[docs]`/`[cleanup]`/
  `[ops]` for non-functional).
- **Skill boundaries:** the trading skill may not touch UI files; the
  app-evolver may not touch the ledger, fill model, or any sacred table.
- The agent tools are the only write path to `desk_*` — never raw SQL, and
  never write to the market-data tables outside `agent.refresh`.
- Durable context lives in files (`REBUILD-V3.md`, the skills, this file),
  not chat — conversation history does not survive a session reset.

## History

The original EdgeFinder was a strategy-research workbench (walk-forward
validation, a 12-strategy fleet, per-strategy paper accounts, GitHub-Actions
agents, Polygon data). It was retired in stages — the per-ticker arena in
v5.47, the whole workbench in the v6.0 greenfield cutover (2026-06-22), the
live-desk model in REBUILD-V3 (v8.x), and the last Polygon/workbench code
purged around v8.10. Only the market-data asset and the thin
data/backtest layer survived. The record lives in git history, `HANDOFF.md`,
and `reviews/` — none of it is current guidance.
