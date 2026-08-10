# EdgeFinder — the autonomous AI paper-trading desk

EdgeFinder is a single **autonomous AI paper-trading agent**. Claude Code
Routine sessions *are* the trader: a self-scheduling chain of cycles
observes live markets, authors and evolves its **own** strategy, grounds
ideas with backtests over 21 years of history, and trades a real broker's
paper account — **Alpaca's paper brokerage is the book of record** — with
full discretion, explaining every decision on the public `/desk` page.

**`REBUILD-V4.md` is the design charter** (honesty contract V2, era model,
runtime layout, credentials model). The agent's operating manual is
`.claude/skills/trading-agent/SKILL.md`. This file is the map.

## The honesty contract V2 (non-negotiable — full text in REBUILD-V4.md)

1. **A real broker executes every fill** — orders go to Alpaca's paper API
   and fill against the live NBBO; our mirror (`desk_orders`/
   `desk_activities`) is a cache, never the arbiter. What the simulator
   does NOT simulate (market impact, commissions, **dividends**, same-day
   option-event records) is disclosed on the desk, never papered over.
2. **Risk shape is platform-enforced**: the paper account runs
   `no_shorting`, margin multiplier 1, options Level 3 max (no naked tier
   exists; multi-leg orders must be fully covered). A broker rejection is
   final.
3. **Paper-only by construction**: the runtime holds only the paper
   account's trade keys, which cannot authenticate against the live API.
   `agent/trade.py` is the ONLY module that submits orders (test-pinned);
   `agent/broker.py` stays a data-reader with no write methods.
4. **Sacred, never drop/clear:** the market-data tables (`daily_bars`,
   `dividends`, `ticker_splits`, `fundamentals_pit`, `ticker_news`), the
   R2 parquet archive, the R2 `backups/` prefix, and the frozen `era1_*`
   tables.
5. **The scoreboard cannot flatter:** SPY comparisons are **symmetric
   price return** both sides (the paper book earns no dividends — the
   benchmark doesn't get to carry its own); every buy/add pick registers
   a falsifiable prediction + horizon + kill before the decision saves;
   grading runs from machine facts (`agent.grade` → `desk_outcomes`).
6. **Learning is tier-gated** (`SCHEMA.md`): the claims registry
   (`desk_claims`) is the source of truth for behavior-influencing facts;
   prose can inform, ONLY claims can justify (enforced at decision save);
   candidates promote through pre-registered criteria; trim/exit
   conditional promises must be structured commitments (machine-swept from
   stored closes); pivots and cap RAISES need an owner-approved proposal;
   supersede, never delete.

## Runtime layout

| Piece | Where | Job |
|---|---|---|
| Paper account | Alpaca | THE BOOK: fills, positions, cash, equity, resting GTC stops, expiry |
| Trading brain | Claude Code Routines | Each market-hours cycle plans the next 15–60 min out (`brain wake-plan` budget gate — the row is the whole job; fired sessions have NO scheduler tools); the dispatcher's restart branch revives a dropped chain (desk hours + no cycle in 25 min, paced 25 min apart — the retired hourly floor Routine's job, absorbed in V4.1.1) |
| Chain dispatcher | Render (always-on, in `agent/streamer.py`) | Polls `desk_wakes` every 60s, fires the "EdgeFinder chain wakes" Routine's API `/fire` trigger when a plan is due; `desk_dispatches` CAS ledger = at-most-once, ≤60/day, ≤3 tries/wake then `missed:auto` |
| Quote streamer | Render (always-on) | SIP WebSocket → in-memory `QuoteCache` → SSE live tape (`/api/desk/stream`) — display + research; fills don't depend on it |
| Desk page | Render | Live ticks, the book (from Alpaca), thinking feed, decisions, journal, claims, What's New |
| Nightly data | Claude Code Routine | `data-refresh` skill — whole-market ingest + EDGAR + brief + mirror sync, portfolio snapshot, split guard, R2 backup, DB size check |
| Strategy Lab | Claude Code Routine, nightly | `strategy-lab` skill — 21y split-sample sweep → leaderboard → brief |
| Reflection | Claude Code Routine, Friday | `reflection-agent` skill — grade the week, curate the wiki, lint the registry |
| App evolver | on-demand (no Routine) | `app-evolver` skill — one small announced `/desk` improvement when the owner asks |
| Mirror + knowledge | Supabase Postgres (free tier) | `desk_*` tables; nightly R2 backup; size check vs the 500MB cap |
| Deep history | Cloudflare R2 | 21y parquet + `backups/` |

There is no GitHub Actions execution arm and no in-house stop watcher —
protection rests on Alpaca's own book. The one in-process clock is the
Render dispatcher above (V4.1): it fires trading *sessions*, never
orders, because fired Routine sessions cannot create their own triggers
(probed 2026-07-13, re-proven 2026-08-10).

## Tech stack

Python 3.11+, FastAPI + Jinja2 (server-rendered desk, vanilla ES modules,
no build system), SQLAlchemy 2.0, pydantic/pydantic-settings, alpaca-py
(data + PAPER trading), websockets, boto3 (R2), SQLite (dev/tests) /
Supabase Postgres (prod). Deployed on Render
(`https://edgefinder-pm8h.onrender.com` — `edgefinder.onrender.com`
belongs to someone else, never probe it).

## Directory structure

```
edgefinder/
├── agent/                      # THE AGENT'S TOOLS (CLI, JSON out, called via Bash)
│   ├── trade.py                #   THE ORDER PATH: submit/cancel/arm-stop/reconcile/
│   │                           #   state/probe against the Alpaca PAPER account;
│   │                           #   client_order_id="<run_id>:<seq>" attribution;
│   │                           #   mirrors into desk_orders/desk_activities
│   ├── grade.py                #   outcomes + grading on the mirror: (run_id,symbol)
│   │                           #   joins, price-return SPY, kill/commitment sweeps,
│   │                           #   exit_kind incl. hardstop/settlement/cutover
│   ├── market.py               #   observe: regime/quote/history/news/universe (local data)
│   ├── data.py                 #   the one data-access seam over the kept layer
│   ├── backtest_tool.py        #   ground ideas: parametric rules vs SPY, net of costs
│   ├── brain.py                #   strategy state, journal, thinking, decision registry,
│   │                           #   wiki, wake budget (wake-plan/due/honor), chain-health
│   ├── knowledge.py            #   claims registry, commitments, proposals, lint (SCHEMA.md)
│   ├── broker.py               #   Alpaca DATA-READER (read-only): quotes, clock, chains
│   ├── streamer.py             #   the always-on SIP WebSocket → QuoteCache (Render)
│   ├── refresh.py              #   bar ingest + the V4 nightly duties (sync/snapshot/
│   │                           #   split-guard/backup/size-check)
│   ├── backup.py               #   R2 export of knowledge + irreplaceable tables; size-check
│   ├── edgar.py                #   SEC EDGAR PIT fundamentals: ingest/coverage/validate
│   ├── options_data.py         #   chain summary (ATM IV, expected move) + IV data bank
│   ├── occ.py                  #   OCC option-symbol parse/format (pure)
│   ├── store.py                #   transport-agnostic table access (pg | rest)
│   ├── rest.py                 #   stdlib PostgREST client (+/rpc) for the 443-only lane
│   ├── preflight.py            #   readiness: DB, data freshness, paper account
│   ├── announce.py             #   "What's New" changelog writer (app-evolver's tool)
│   └── models.py               #   the desk_* ORM tables + idempotent DDL
├── .claude/skills/             # trading-agent (the operating manual), data-refresh,
│                               # strategy-lab, reflection-agent, app-evolver
├── dashboard/                  # FastAPI app — __version__ in app.py; routers/ (pages,
│                               # desk API, symbols API); dark-terminal design system
├── edgefinder/                 # KEPT data + backtest layer (audited, reused)
├── config/settings.py          # EDGEFINDER_-prefixed runtime settings
├── scripts/                    # render_start.py, bootstrap.sh, setup_db.py,
│                               # smoke_dashboard.py, mobile_audit.py
├── docs/                       # ROUTINES.md, CUTOVER-V4.md, fundamentals docs,
│                               # history/ (REBUILD-V3.md and earlier)
└── tests/                      # SQLite-pinned unit tests, FakeTradingClient seam
```

## Database

One Supabase Postgres database (free tier — nightly R2 backup + size
check), two namespaces:

- **The agent's own tables** (`agent/models.py`): the Alpaca mirror
  (`desk_orders` — one row per order and per mleg leg, attribution via
  `client_order_id`; `desk_activities` — append-only, cursor-synced,
  T+1-aware; `desk_portfolio_history` — nightly equity+positions
  snapshot, the split guard's baseline), the decision layer
  (`desk_decisions` with the prediction registry, `desk_thinking`,
  `desk_journal`, `desk_strategy_state`, `desk_backtests`,
  `desk_changelog`, `desk_options_snap`, `desk_wiki`(+history),
  `desk_briefs`, `desk_wakes` — the wake-budget ledger), the outcomes
  layer (`desk_outcomes` — grade's machine facts + reflection verdicts),
  and the knowledge layer (`desk_claims`, `desk_claim_events`,
  `desk_commitments`, `desk_proposals`). After cutover: frozen
  `era1_trades`/`era1_positions`/`era1_equity` (the V3 book, read-only
  forever).
- **Kept market-data tables** (`edgefinder/db/models.py`, read-only
  inputs): `daily_bars`, `ticker_news`, `ticker_splits`, `dividends`,
  `fundamentals_pit` (SEC EDGAR PIT), `fundamentals_snapshots` (frozen
  validation reference).

**Account integrity (CRITICAL):** cash/positions/equity live at the
broker; the mirror re-converges at every cycle start
(`agent.trade reconcile`) and Alpaca wins on conflict. The agent tools
are the ONLY write path to `desk_*` — never raw SQL, and never write to
the market-data tables outside `agent.refresh`.

## Data sources & transports

- **Alpaca is the sole live market-data source** (Algo Trader Plus: SIP
  quotes, clock/calendar, daily bars, Benzinga news, OPRA chains) AND the
  paper broker (separate paper trade keys).
- **SEC EDGAR is the fundamentals source** (`agent/edgar.py` →
  `fundamentals_pit`), point-in-time by construction, ≤10 req/s with the
  declared User-Agent.
- **R2**: grow-only 21y parquet archive + `backups/` (knowledge + market
  tables, written nightly by `agent.backup`).
- **Two DB transports** (`agent/store.py`): `pg` (SQLAlchemy) and `rest`
  (Supabase PostgREST over HTTPS/443 — the sandbox lane); `auto` picks
  `rest` iff `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set.

## Configuration

All settings in `config/settings.py` (`EDGEFINDER_` prefix):
`ALPACA_API_KEY`/`_SECRET` (data), `ALPACA_TRADE_KEY`/`_SECRET` (the
paper account — no fallback between the pairs, by design),
`ALPACA_DATA_FEED=sip`, `ALPACA_PAPER=true` (trade.py refuses to
construct when false), `STREAM_SYMBOLS`, `STREAM_STALE_SECS`,
`DATABASE_URL`, `DB_TRANSPORT`, `STARTING_CAPITAL` (set at cutover to
Era-1's final equity), `GITHUB_REPO` (proposal issues). Plus non-prefixed
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `R2_*`.

## Quick start

```bash
pip install -e ".[dev]"            # Render uses ".[live]"; Routines run scripts/bootstrap.sh
python -m agent.preflight          # DB + data + paper account? run before anything

# The agent's own tools (JSON out) — the skill drives these via Bash
python -m agent.brain context                     # the cycle's working memory in ONE read
python -m agent.trade state                       # cash, equity, positions (LIVE from Alpaca)
python -m agent.trade reconcile                   # cycle-start mirror sync + what filled
python -m agent.trade submit --symbol NVDA --side buy --notional 5000 \
    --type market --run-id 2026-08-17T14:30-r7kq  # a REAL paper order
python -m agent.trade arm-stop --symbol NVDA --stop-price 150 --run-id <RID>
python -m agent.trade orders --status open        # resting stops + working limits
python -m agent.grade run                         # machine facts → desk_outcomes
python -m agent.grade outcomes --days 14          # picks vs predictions vs SPY (price-return)
python -m agent.market brief                      # the nightly research pack
python -m agent.market universe --top 40          # most-liquid names
python -m agent.broker quote --symbols NVDA,SPY   # LIVE bid/ask (data keys)
python -m agent.brain wake-plan --at 2026-08-17T19:45:00Z --reason "..."  # budget gate
python -m agent.brain chain-health                # is the chain alive? (floor sessions)
python -m agent.backtest_tool --symbols A,B,C --rule momentum:5
python -m agent.lab sweep --max-combos 80         # nightly strategy search
python -m agent.knowledge claim-list              # the claims registry
python -m agent.backup run                        # knowledge + market tables → R2
python -m agent.backup size-check                 # DB vs the 500MB free cap

# Dashboard
uvicorn dashboard.app:app --reload   # http://localhost:8000/ → /desk
python scripts/smoke_dashboard.py

# Tests — THE pre-commit gate
DATABASE_URL= python -m pytest tests/ -q -m "not integration"
```

## API endpoints

Pages: `/` → `/desk` · `/trades` · `/symbol/{sym}`.

`/api/desk/*` (read-only projections): `portfolio` (LIVE Alpaca account +
positions + price-return `vs_spy`), `equity` (Era-1 + Era-2 stitched
curve), `open-orders` (resting stops + working limits), `decision/latest`,
`decisions`, `outcomes`, `thinking`, `backtests`, `strategy`, `wiki`,
`claims`, `proposals`, `regime`, `movers`, `holding-stats`, `dividends`
(incl. the estimated-missed-dividends counter), `quotes`, `stream` (SSE
live tape), `options/{symbol}`(+`/history`), `broker-health` (paper
account + clock + last reconcile), `data-health`, `brief`, `whatsnew`,
`trades` (era-tagged fills), `trade-history`.

`/api/symbols/{sym}/bars|events|fundamentals` power the chart page.
`/api/health` returns status + version.

## Rules for every session

- **Never force-push `main`; never skip the test gate**
  (`DATABASE_URL= python -m pytest tests/ -q -m "not integration"`).
- **Version-bump `dashboard/app.py`** on every functional merge; commit
  format `[vX.Y.Z] short description` (`[docs]`/`[cleanup]`/`[ops]` for
  non-functional).
- **Skill boundaries:** the trading skill may not touch UI files; the
  app-evolver may not touch `agent/trade.py`, `agent/grade.py`, the
  mirror tables, or any sacred table.
- The agent tools are the only write path to `desk_*`; `agent/trade.py`
  is the only Alpaca order-write path (test-pinned in
  `tests/test_live_fill.py`).
- Durable context lives in files (`REBUILD-V4.md`, the skills, this
  file), not chat.

## History

EdgeFinder v1 was a strategy-research workbench (retired in the v6.0
greenfield cutover, 2026-06-22). REBUILD-V3 (v8–v9.28) was the live desk
with a hand-rolled ledger: fills priced off self-captured SIP quotes, a
GitHub-Actions execution arm, a Render wake-dispatcher, tripwires and an
in-house hard-stop sweep. V4 (v10.0.0) replaced the ledger with Alpaca's
paper brokerage and froze the V3 book as Era 1; its self-scheduling
design (each cycle `create_trigger`s its successor) assumed a tool fired
sessions never have and died on first contact (2026-08-10) — V4.1
(v10.1.0) brought back the V3 dispatcher pattern on Render, firing the
chain-wakes Routine's API trigger instead of GitHub Actions. The record
lives in git history,
`docs/history/REBUILD-V3.md`, `HANDOFF.md`, and `reviews/` — none of it
is current guidance.
