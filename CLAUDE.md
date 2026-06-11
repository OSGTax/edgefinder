# EdgeFinder v2 — Trading Workbench

## Project Overview
EdgeFinder is a general strategy workbench: drop in ANY portfolio strategy →
honest walk-forward validation → promote the survivors to self-running paper
trading → monitor everything on the dashboard. It runs on a permanent,
growing market-data asset and an ongoing wide hunt (Lynch-style + systematic)
for anything that beats the market.

**Key principles:**
- Polygon.io is the sole data source (no fallback chains)
- Two-tier storage (v5.35): Cloudflare R2 holds the permanent, GROW-ONLY
  bar history (Parquet per symbol, merge-sync nightly — never shrinks);
  Supabase holds the operational hot set only (protected ETFs full-history
  + trailing-365d top-1000/day; nightly fingerprint-guarded prune keeps it
  under the free-tier cap). Deep/breadth backtests read R2 via
  `engine.validate --bars-from r2` (works with --universe; equivalence
  vs the DB path proven bit-exact)
- Honesty before performance: parameters pre-registered in git BEFORE the
  first run, point-in-time universes/fundamentals, realistic costs,
  total-return prices, a sealed holdout (2025-12-05), null/random controls
  in every batch, adversarial re-checks before any "finalist" claim
- Every promoted strategy trades an isolated $100,000 paper account; the
  trades table (+ dividend_credits) is the source of truth for balances
- The old per-ticker arena (coward/gambler/degenerate, $5k accounts,
  intraday signal loop, coach) was **retired in v5.47** — the v2 engine is
  the only trading path

> ## 🏆 ACTIVE INITIATIVE — read `HANDOFF.md` first
> The hunt goal (**find 10 winning strategies**) was **REACHED on
> 2026-06-11: 12 confirmed finalists** across 4 rounds / 73 pre-registered
> candidates (reports `reviews/HUNT-ROUND-{1..4}.md`). Confirmation
> standard: criteria-passing + ALL THREE adversarial re-checks. The
> holdout (2025-12-05) is STILL SEALED — burning it on the finalists is
> the next owner decision, then live-universe mechanics + promotion.
> Full state in **`HANDOFF.md` → "CURRENT INITIATIVE"**. If asked to
> "continue," read that first. Conversation history does NOT survive a
> Codespace rebuild — durable context lives in `HANDOFF.md`, not chat.

## Architecture
```
Dashboard (FastAPI) — Portfolio | Symbol | Trades | Strategies | Screener | Lab | Ops
        |
        ├─ Validation lab (offline): engine.validate → walk-forward folds
        |     + sealed holdout → validation_runs (scoreboard on /lab)
        ├─ Promotion: engine.promote → promoted_strategies
        ├─ Live paper trading: engine.live (9:45 ET daily cycle)
        |     + 30-min account marks → trades / strategy_snapshots
        └─ Data collection (nightly): scanner (fundamentals + PIT snapshot),
              benchmarks, news/dividends/splits, R2 bar sync + DB prune
                          |
                  Data Layer (Polygon.io · Supabase hot set · R2 archive)
```

## Tech Stack
- Python 3.11+, FastAPI, SQLAlchemy 2.0, Alembic
- Polygon.io (bars, fundamentals, universe)
- pydantic + pydantic-settings (domain models, typed config)
- APScheduler (ET timezone scheduling)
- SQLite (dev) / PostgreSQL (production via Render + Supabase)
- Cloudflare R2 (Parquet bar archive, boto3/s3)
- Frontend: vanilla ES modules + CSS tokens, vendored lightweight-charts
  (no build system, no CDN, zero inline styles — guard-tested)

## Directory Structure
```
edgefinder/
├── pyproject.toml              # Dependencies and build config
├── config/settings.py          # Tunable parameters (EDGEFINDER_ env prefix)
├── edgefinder/                 # Main package
│   ├── core/                   # models.py, interfaces.py (DataProvider/DataHub)
│   ├── data/                   # polygon.py, provider.py, cache.py,
│   │                           # barstore.py (R2 sync/prune), market_data.py,
│   │                           # indicator_engine.py, pit_fundamentals.py,
│   │                           # accumulator.py (news/dividends/splits)
│   ├── db/                     # engine.py, models.py (ORM), migrations/
│   ├── engine/                 # THE v2 ENGINE:
│   │   ├── strategy.py         #   AssetView/RebalanceContext + base strategies
│   │   ├── strategies.py       #   make_strategy_factory (spec -> factory)
│   │   ├── hunt_r1.py/hunt_r2.py  # pre-registered hunt rosters
│   │   ├── backtest.py         #   pure run_backtest (costed, div-adjusted)
│   │   ├── walkforward.py      #   folds + regime tags + criteria
│   │   ├── validate.py         #   CLI: walk-forward + sealed holdout
│   │   ├── record.py           #   persists scorecards to validation_runs
│   │   ├── promote.py          #   CLI: validation-gated promotion
│   │   ├── data.py             #   bar loading (DB/R2), PIT universe ranking
│   │   └── live.py             #   daily paper-trading cycle ($100k accounts)
│   ├── backtest/costs.py       # cost model (corwin-schultz spread est.)
│   ├── trading/                # journal.py (TradeJournal), integrity.py
│   ├── scanner/                # unified_scanner.py — nightly DATA collector
│   ├── signals/engine.py       # indicators + pattern detectors (research)
│   ├── analytics/              # live_scorecard.py (live-vs-lab proof)
│   ├── market/                 # snapshot.py, benchmarks.py, sector_rotation.py
│   ├── research/research.py    # per-ticker deep-dive aggregation
│   ├── agents/                 # watchdog.py, reasoning.py, alerts.py
│   └── scheduler/scheduler.py  # APScheduler (ET)
├── dashboard/
│   ├── app.py                  # FastAPI app (__version__ lives here)
│   ├── services.py             # provider/scheduler singletons + jobs
│   ├── symbol_service.py       # chart bars (DB/R2 seam + TTL cache)
│   ├── routers/                # trades strategies research benchmarks
│   │                           # market admin ops symbols lab pages
│   ├── templates/              # 7 pages (dark-terminal design system)
│   └── static/                 # tokens/components CSS + ES modules
├── hunt/queue.json             # wave manifest -> hunt-batch.yml matrix
├── ops/                        # flag files: slim.flag, r2-check.flag,
│                               # retire-arena.flag (push-triggered workflows)
├── scripts/                    # setup_db, run_scanner, seed_demo_data,
│                               # smoke_dashboard, slim_daily_bars,
│                               # backfill_daily_bars, retire_arena_db
└── tests/                      # 553+ tests
```

## Quick Start
```bash
pip install -e ".[dev]"
echo 'EDGEFINDER_POLYGON_API_KEY=your_key' > .env
python scripts/setup_db.py

# Validate a strategy (the core workflow)
python -m edgefinder.engine.validate --strategy equal_weight \
    --symbols SPY,QQQ,IWM --schedule monthly
# Universe-scale, honest everything, straight from R2:
python -m edgefinder.engine.validate --strategy mom_6m_k20 \
    --universe top:500 --start 2021-06-01 --costed --div-adjust \
    --bars-from r2 --holdout-start 2025-12-05 --record

# Dashboard (demo data for offline dev)
python scripts/seed_demo_data.py --db-url sqlite:///data/demo.db
uvicorn dashboard.app:app --reload     # http://localhost:8000/
python scripts/smoke_dashboard.py      # 35-endpoint smoke

# Nightly data collection, on demand
python scripts/run_scanner.py --quick

# Tests
python -m pytest tests/ -v -m "not integration"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check (+ version) |
| GET | `/api/trades?strategy=X&status=Y` | List trades (filterable) |
| GET | `/api/trades/stats` · `/wins` · `/losses` · `/integrity` | Trade stats + hash-chain audit |
| GET | `/api/strategies` | Promoted strategies (from DB) |
| GET | `/api/strategies/accounts` · `/positions` · `/equity-curve` | Account states, open positions, curves |
| GET | `/api/strategies/summary` · `/promoted` · `/meta` · `/dividends` · `/params` | Rollups, registry, metadata, ledgers |
| GET | `/api/strategies/scorecard` · `/validation` · `/scheduler` | Live-proof, latest validation, job times |
| GET | `/api/lab/runs` · `/runs/{id}` · `/scoreboard` · `/labels` | Validation-runs browser + 10-finalist scoreboard |
| GET | `/api/symbols/{sym}/bars?range=&indicators=` | Chart bars + indicators (DB/R2 seam, epoch times) |
| GET | `/api/symbols/{sym}/events` | Dividend/split/news chart markers |
| GET | `/api/research/ticker/{symbol}` · `/search?q=` · `/active` | Ticker reports, search |
| POST | `/api/research/scan` | On-demand data-collection scan |
| GET | `/api/benchmarks/comparison?days=90` · `/sectors` | Index/sector comparison series |
| POST | `/api/benchmarks/collect` · `/backfill` | Collect/backfill index data |
| GET | `/api/market/regime` · `/sectors/history` | Market snapshot + sector series |
| GET | `/api/ops/health` · `/activity` · `/storage` | Heartbeats; agent timeline; DB vs R2 panel |
| POST | `/api/admin/run-eod` | Token-guarded on-demand EOD pipeline |

Pages: `/` (portfolio) · `/symbol/{sym}` · `/trades` · `/strategies` ·
`/screener` · `/lab` · `/ops` (`/research` → `/symbol`, `/backtest` → `/lab`).

## Paper Account Rules (v2)
- $100,000 starting capital per promoted strategy, isolated accounts
- Long-only target weights; whole-share fills; no margin/leverage
- Costs modeled at fill time; dividends credited on ex-date while held
  (`dividend_credits` — the live counterpart of the lab's total-return bar)

### Account Balance Integrity (CRITICAL)
The trades table is the **source of truth** for all account balances.
`engine/live._recalc_cash` recomputes cash on every cycle and mark:
```
correct_cash = starting_capital + sum(closed trade P&L) + sum(dividend credits)
             - sum(open position cost basis)
```
1. Account state persists to `strategy_accounts` after every cycle; the
   30-min snapshot job appends `strategy_snapshots` (equity-curve series)
2. Total Account Value = cash + market value of positions (not cost basis)
3. Total P&L = Total Account Value − starting capital (canonical formula)
4. The watchdog's cash-drift check uses the same formula

## Strategy Guide (v2 engine)

A strategy is a class with `name` and a pure `rebalance(ctx) -> weights`:

```python
from edgefinder.engine.strategy import RebalanceContext

class MyStrategy:
    @property
    def name(self) -> str:
        return "my_strategy"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        # ctx.assets: {symbol: AssetView(price, indicators, history,
        #              fundamentals, .ret(n), ...)}; ctx.date is PIT-safe.
        # Return target weights (>=0, sum <= 1.0); {} means all-cash.
        a = ctx.get("SPY")
        if a and a.indicators.ema_200 and a.price > a.indicators.ema_200:
            return {"SPY": 1.0}
        return {}
```

Wire a spec into `engine/strategies.make_strategy_factory` (or a hunt
roster dict like `hunt_r2.HUNT_R2_SPECS`) — the same spec string then works
in the validator, the promotion CLI, and the live runner.

**Hunt discipline (non-negotiable):** commit the roster BEFORE the first
run; run a null control (buy_and_hold:SPY) per batch; score both bars
(risk-adjusted criteria recorded + total-return computed from folds);
finalists need majority-of-folds wins AND all three adversarial re-checks
(`--is-days 357`, `--is-days 399`, late `--start 2022-06-01`); the holdout
(2025-12-05) stays sealed without owner sign-off. Waves run via
`hunt/queue.json` → `.github/workflows/hunt-batch.yml` (push-triggered).

**Promotion:** `python -m edgefinder.engine.promote --spec X --symbols ...`
— refuses specs whose latest validation_run did not pass. `--list`,
`--demote NAME`. The 9:45 ET cycle then trades it daily, unattended.

## Key Configuration (config/settings.py)
All parameters can be overridden via env vars with `EDGEFINDER_` prefix.
Key sections: scanner filters (price/market-cap bands, top-N universe),
technical indicator params (EMA/RSI/MACD/BB), scheduling times, Polygon
connection, DB/R2. Engine knobs (costs, folds, holdout) are CLI flags on
`engine.validate`, not settings.

## Testing
```bash
python -m pytest tests/ -v -m "not integration"   # Unit tests (553+)
python -m pytest tests/ -v -m integration          # Hits Polygon
# Pre-commit gate used in CI and by agents:
DATABASE_URL= python -m pytest tests/ -q -m "not integration" --ignore=tests/test_market.py
```

## Git Workflow
- Commit and push directly to main (feature branches for sessions)
- Run tests before every commit
- Commit format: `[vX.Y] short description`
- **Version bump required**: every functional merge to main updates
  `__version__` in `dashboard/app.py` (shown via `/api/health`)

## Production
- Render paid Starter instance (never idles): the in-process APScheduler
  is the production driver. **Real URL: `https://edgefinder-pm8h.onrender.com`**
  (`edgefinder.onrender.com` belongs to someone else — never probe it).
- Schedule (ET, weekdays): 9:45 portfolio cycle · every 30m 9:45–16:15
  v2 account snapshots · 16:05 market snapshot · 16:10 benchmarks ·
  16:15 sector rotation · 18:15 nightly scan + PIT snapshot · 18:30
  dividends/splits · 19:00 R2 sync + DB prune · hourly news (market hours).
- `EDGEFINDER_SCHEDULER_ENABLED=false` disables all jobs (CI/dev).

## Management Agents
Agents write findings to `agent_observations` and changes to
`agent_actions` — one timeline for postmortems.

- **watchdog** — two-phase: (1) deterministic SQL checks
  (`agents/watchdog.py`): cash drift (v2 formula), negative cash, paused
  accounts, high drawdown, **cycle liveness** — CRITICAL if the
  `v2_portfolio_cycle` heartbeat is missing/failed/staler than 26h
  (weekdays after 10:00 ET); (2) agentic reasoning (`agents/reasoning.py`)
  via `claude -p` over observations + persistent `agent_memory`.
  Active window Mon–Fri 08:30–17:00 ET; `--ignore-window` to override.
  Cron: `.github/workflows/watchdog.yml` (hourly), kill switch
  `WATCHDOG_ENABLED` repo variable.
- **alerts** (`agents/alerts.py`) — projects unresolved CRITICAL
  observations onto deduped GitHub issues (label `edgefinder-alert`),
  auto-closes on recovery. Cron: `liveness.yml` (15-min, market hours),
  kill switch `LIVENESS_ENABLED`. Uses the workflow `GITHUB_TOKEN`.
- Retired in v5.47: coach, weekly summary, intraday-cycle/keepalive crons.

### Ops workflows (push-triggered — no actions:write needed)
- `hunt/queue.json` → hunt-batch.yml (validation wave matrix, R2+DB secrets)
- `ops/slim.flag` → db-slim.yml (`EXECUTE` to run; anything else dry-run)
- `ops/retire-arena.flag` → retire-arena.yml (same EXECUTE convention)
- `ops/r2-check.flag` → r2-ops.yml (store verify)
- `bars-nightly.yml` (cron Tue–Sat 04:15 UTC): top-1000 daily-bars ingest
- `tests.yml`: full suite on PRs to main

### Running agents from a Codespace
Set Codespaces secrets `DATABASE_URL` + `CLAUDE_CODE_OAUTH_TOKEN`, then:
```bash
python -m edgefinder.agents.watchdog --force      # deterministic checks
python -m edgefinder.agents.reasoning --force     # LLM step
python -m edgefinder.agents.watchdog --dry-run    # preview
```
Unattended scheduling stays on GitHub Actions (Codespaces auto-stop).
