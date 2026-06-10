# EdgeFinder v2 — Trading Workbench

## Project Overview
EdgeFinder is a trading workbench for strategy research, paper trading, and performance analysis.
It combines fundamental data from Polygon.io, technical signal detection,
multi-source sentiment analysis, and multi-strategy competition in isolated virtual accounts.

**Key principles:**
- Polygon.io is the sole data source (no fallback chains)
- Two-tier storage (v5.35): Cloudflare R2 holds the permanent, GROW-ONLY
  bar history (Parquet per symbol, merge-sync nightly — never shrinks);
  Supabase holds the operational hot set only (protected ETFs full-history
  + trailing-365d top-1000/day; nightly fingerprint-guarded prune keeps it
  under the free-tier cap). Deep/breadth backtests read R2 via
  `engine.validate --bars-from r2` (works with --universe; equivalence
  vs the DB path proven bit-exact)
- Every strategy gets its own isolated $5,000 virtual account
- Every trade captures a market-wide snapshot (SPY, QQQ, IWM, DIA, VIX, sectors)
- Per-strategy views everywhere — no aggregate P&L
- AI meta-strategy interfaces are built in (future Phase 8)

> ## 🎯 ACTIVE INITIATIVE — read `HANDOFF.md` first
> The project has pivoted to a **clean-engine rebuild + general workbench** (drop in
> ANY strategy → honest backtest → auto-promote to self-running paper trading →
> monitor on the dashboard; a permanent, growing data asset; a wide Lynch-style hunt
> for anything that beats the market). **Phase 1 is committed (`48771df`, v5.22.0);
> resume at Phase 2.** The full vision, design principles, roadmap, working
> agreements, and the Cloudflare R2 storage plan are in **`HANDOFF.md` → "CURRENT
> INITIATIVE"**. If asked to "continue" or "finish autonomously," read that first.
> Note: conversation history does NOT survive a Codespace rebuild — only this repo
> does, so durable context lives in `HANDOFF.md`, not chat.

## Architecture
```
Dashboard (FastAPI) — Research | Trades | Strategies | Benchmarks | Inject
        |                                    |
  Research Service              Trading Engine (Arena)
  (per-ticker aggregation)      Executor → Virtual Accounts
        |                                    |
  Scanner | Sentiment | Market Snapshot | Strategies (plugins)
        |            |           |              |
                  Data Layer (Polygon.io)
```

## Tech Stack
- Python 3.11+, FastAPI, SQLAlchemy 2.0, Alembic
- Polygon.io (bars, fundamentals, universe, streaming)
- pydantic + pydantic-settings (domain models, typed config)
- APScheduler (ET timezone scheduling)
- SQLite (dev) / PostgreSQL (production via Render)

## Directory Structure
```
edgefinder/
├── pyproject.toml              # Dependencies and build config
├── alembic.ini                 # Database migration config
├── .env                        # Polygon API key (gitignored)
├── config/
│   └── settings.py             # All tunable parameters (EDGEFINDER_ env prefix)
├── edgefinder/                 # Main package
│   ├── core/
│   │   ├── models.py           # Pydantic domain models (Signal, Trade, MarketSnapshot, etc.)
│   │   ├── interfaces.py       # Protocols: DataProvider, StreamProvider, SentimentProvider
│   │   └── events.py           # In-process event bus (pub/sub)
│   ├── data/
│   │   ├── polygon.py          # Polygon.io REST (bars, fundamentals, universe, price)
│   │   ├── provider.py         # CachedDataProvider wrapper
│   │   ├── cache.py            # Filesystem cache (Parquet bars, JSON fundamentals)
│   │   └── stream.py           # Polygon WebSocket streaming
│   ├── db/
│   │   ├── engine.py           # SQLAlchemy engine/session (SQLite/PostgreSQL)
│   │   ├── models.py           # 10 ORM tables
│   │   └── migrations/         # Alembic migrations
│   ├── scanner/
│   │   └── scanner.py          # Nightly fundamental scan + strategy qualification
│   ├── signals/
│   │   └── engine.py           # Technical indicators + 9 signal pattern detectors
│   ├── strategies/
│   │   ├── base.py             # BaseStrategy ABC + StrategyRegistry
│   │   ├── coward.py           # Conservative swing — oversold dips, exit early
│   │   ├── gambler.py          # Balanced swing — MACD momentum, exit on fade
│   │   └── degenerate_v2.py    # Aggressive swing — volume spikes, ride the hype
│   ├── trading/
│   │   ├── account.py          # Per-strategy $5k virtual accounts
│   │   ├── executor.py         # Risk-based sizing, slippage, hash chain audit
│   │   ├── arena.py            # Multi-strategy orchestration
│   │   └── journal.py          # Trade persistence + stats
│   ├── market/
│   │   ├── snapshot.py         # Captures indices/VIX/sectors at trade time
│   │   └── benchmarks.py       # Daily index data for comparison charts
│   ├── sentiment/
│   │   ├── aggregator.py       # Weighted composite from all sources
│   │   ├── reddit.py           # Reddit API (r/wallstreetbets, r/stocks)
│   │   ├── twitter.py          # Twitter/X stub (future API integration)
│   │   ├── news_rss.py         # RSS keyword-based sentiment
│   │   └── provider.py         # Score-to-action mapping
│   ├── research/
│   │   └── research.py         # Per-ticker deep-dive aggregation
│   └── scheduler/
│       └── scheduler.py        # APScheduler (ET timezone)
├── dashboard/
│   ├── app.py                  # FastAPI application
│   ├── dependencies.py         # DB session dependency injection
│   ├── routers/
│   │   ├── trades.py           # Wins/losses/open/closed, strategy-filterable
│   │   ├── strategies.py       # Per-strategy accounts + equity curves
│   │   ├── research.py         # Ticker reports + search
│   │   ├── sentiment.py        # Sentiment scores + trending
│   │   ├── benchmarks.py       # Strategy vs index comparison
│   │   └── inject.py           # Manual ticker injection
│   ├── templates/              # 8 pages (v5.45 dark-terminal redesign)
│   └── static/                 # tokens/components CSS + ES-module JS, vendored charts
├── scripts/
│   ├── setup_db.py             # Initialize database
│   ├── run_scanner.py          # CLI scanner (--quick, --tickers)
│   └── render_start.py         # Render deployment startup
└── tests/                      # 244+ tests
```

## Quick Start
```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set Polygon API key
echo 'EDGEFINDER_POLYGON_API_KEY=your_key' > .env

# 3. Setup database
python scripts/setup_db.py

# 4. Run scanner
python scripts/run_scanner.py --quick    # 20 popular tickers
python scripts/run_scanner.py            # Full universe

# 5. Start dashboard
uvicorn dashboard.app:app --reload
# Visit http://localhost:8000/

# 6. Run tests
python -m pytest tests/ -v -m "not integration"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/trades?strategy=X&status=Y` | List trades (filterable) |
| GET | `/api/trades/stats?strategy=X` | Trade statistics |
| GET | `/api/trades/wins` | Winning trades |
| GET | `/api/trades/losses` | Losing trades |
| GET | `/api/strategies` | List registered strategies |
| GET | `/api/strategies/accounts` | Per-strategy account states |
| GET | `/api/strategies/equity-curve?days=90` | Equity curve data |
| GET | `/api/research/ticker/{symbol}` | Full ticker research report |
| GET | `/api/research/search?q=X` | Search tickers |
| GET | `/api/research/active` | Active watchlist tickers |
| GET | `/api/sentiment/ticker/{symbol}` | Aggregated sentiment |
| GET | `/api/sentiment/trending` | Trending tickers |
| GET | `/api/sentiment/history/{symbol}` | Sentiment time series |
| GET | `/api/benchmarks/comparison?days=90` | Strategy vs index data |
| POST | `/api/benchmarks/collect` | Collect daily benchmark data |
| POST | `/api/backtest` | Sync backtest over explicit symbols (≤25) |
| POST | `/api/backtest/jobs` | Start a universe-scale backtest (symbols/top-N/full) on the background worker |
| GET | `/api/backtest/jobs` | List recent backtest jobs |
| GET | `/api/backtest/jobs/{id}` | Poll a backtest job (status, progress, result) |
| POST | `/api/inject` | Inject ticker for evaluation |
| GET | `/api/symbols/{sym}/bars?range=&indicators=` | Chart bars + indicator series (DB/R2 seam, epoch times) |
| GET | `/api/symbols/{sym}/events` | Dividend/split/news chart markers |
| GET | `/api/lab/runs` · `/runs/{id}` · `/scoreboard` · `/labels` | Validation-runs browser + 10-finalist scoreboard |
| GET | `/api/strategies/summary` · `/promoted` · `/meta` · `/dividends` · `/params` | Lane rollups, v2 registry, metadata, ledgers |
| GET | `/api/ops/activity` · `/api/ops/storage` | Agent timeline; DB vs R2 storage panel |
| GET | `/api/research/qualifications` | Scanner's ranked watchlist |
| GET | `/api/market/sectors/history` | Sector price series |
| GET | `/api/inject` | List active injections |
| DELETE | `/api/inject/{id}` | Remove injection |

## Virtual Account Rules
- $5,000 starting capital per strategy
- Buying power = cash only (no margin/leverage)
- PDT mode: per-strategy toggle (3 day trades / 5 business days)
- Max risk per trade: 2% of equity
- Max concentration: 20% in single position
- Max open positions: 5
- Drawdown circuit breaker: 20%
- Revenge trade cooldown: 30 minutes after stop-out

### Account Balance Integrity (CRITICAL)
The trades table is the **source of truth** for all account balances. On every startup,
`_recalculate_account_balances()` recomputes cash from trades to self-heal any corruption:
```
correct_cash = starting_capital + sum(closed trade P&L) - sum(open position cost basis)
```
**v2 portfolio accounts** (engine/live) extend the formula with dividend cash credits
(`dividend_credits` table, written when an ex-date passes while lots are held —
the live counterpart of the lab's total-return adjustment):
```
correct_cash_v2 = starting_capital + sum(closed P&L) + sum(dividend credits)
                - sum(open cost basis)
```
The watchdog's cash-drift check uses the extended formula for ALL accounts
(old-arena strategies have no credit rows, so the extra term is zero for them).
**Rules for all strategies (existing and new):**
1. Every strategy uses the same `VirtualAccount` class — no custom account logic
2. Account state is persisted to DB immediately after every trade open/close AND on shutdown
3. On startup, cash and realized P&L are always recalculated from the trades table
4. Total Account Value on the dashboard = Cash + market value of positions (not cost basis)
5. Total P&L = Total Account Value - Starting Capital (canonical formula)
6. Realized P&L = sum of pnl_dollars from closed trades in DB
7. Unrealized P&L = sum of (current_price - entry_price) × shares for open positions

## Strategy Plugin Guide

### Creating a Strategy
```python
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry
from edgefinder.core.models import Signal, TickerFundamentals

@StrategyRegistry.register("my_strategy")
class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "my_strategy"

    @property
    def version(self) -> str: return "1.0"

    @property
    def preferred_signals(self) -> list[str]:
        return ["ema_crossover_bullish", "rsi_oversold"]

    def init(self) -> None: pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        # Each strategy defines its own qualification criteria
        return (fundamentals.earnings_growth or 0) > 0

    def generate_signals(self, ticker: str, bars) -> list[Signal]:
        from edgefinder.signals.engine import compute_indicators, detect_signals
        indicators = compute_indicators(bars)
        if not indicators: return []
        return [s for s in detect_signals(indicators, ticker)
                if s.metadata.get("pattern") in self.preferred_signals]

    def on_trade_executed(self, notification) -> None: pass
```

Then add the import to `edgefinder/strategies/__init__.py`.

### Available Signal Patterns
`ema_crossover_bullish`, `ema_crossover_bearish`, `rsi_oversold`, `rsi_overbought`,
`macd_bullish_cross`, `macd_bearish_cross`, `bb_lower_touch`, `volume_spike_bullish`,
`volume_spike_bearish`

## Key Configuration (config/settings.py)
All parameters can be overridden via environment variables with `EDGEFINDER_` prefix.
See `config/settings.py` for the full list. Key sections:
- Account ($5k capital, 5 max positions, 20% concentration)
- Risk (2% max risk, 20% drawdown breaker, 1.5 R:R minimum)
- Scanner filters ($300M-$200B market cap, $5-$500 price)
- Strategy qualification (per-strategy criteria using Polygon fundamentals)
- Technical signals (EMA 9/21/50/200, RSI 14, MACD 12/26/9, BB 20/2)
- Sentiment thresholds (BLOCK at -0.5, REDUCE at -0.2, BOOST at +0.2/+0.5)
- Scheduling (scanner 6:15 PM, signals every 15m, positions every 5m)
- Polygon.io connection (API key, retries, timeouts)

## Testing
```bash
python -m pytest tests/ -v -m "not integration"   # Unit tests (244+)
python -m pytest tests/ -v -m integration          # Integration tests (hits Polygon)
```

## Git Workflow
- Commit and push directly to main
- Run tests before every commit
- Commit format: `[vX.Y] short description`
- **Version bump required**: Every merge to main that changes functionality must update `__version__` in `dashboard/app.py`. This version is displayed in the top-right corner of the dashboard via `/api/health`.

## Management Agents

Agent infrastructure lives under `edgefinder/agents/`. Each agent writes
findings to `agent_observations` and changes to `agent_actions` so
postmortems read a single timeline alongside the trades table.

### Kill switches — two separate gates
- **Interactive / Render**: `.claude/agent-config.json` (gitignored).
  `is_agent_enabled("<name>")` reads it on every call. Missing file or
  `enabled: false` ⇒ disabled. Example template:
  `.claude/agent-config.example.json`.
- **GitHub Actions cron**: the `WATCHDOG_ENABLED` repo variable. The
  workflow's `if: vars.WATCHDOG_ENABLED == 'true'` skips the job when
  set to anything other than `'true'`. No redeploy needed.

### Agents
- **watchdog** — two-phase management agent:
  1. **Deterministic checks** (`edgefinder/agents/watchdog.py`) — DB-only
     SQL checks: cash drift, negative cash, paused accounts, high
     drawdown, **cycle liveness** (stalled intraday loop). Writes to
     `agent_observations`, reconciles against prior unresolved rows
     (dedup + auto-resolve).
  2. **Agentic reasoning** (`edgefinder/agents/reasoning.py`) — calls
     Claude (default `claude-opus-4-8`, adaptive thinking, prompt
     caching on system + memory) over the current observations + the
     agent's persistent memory (`agent_memory` table) + recent trades
     + recent trading-path commits. Returns structured decisions per
     observation (escalate/investigate/monitor/dismiss) and an
     optional memory update. Records AgentActions only for
     escalate/investigate decisions (no audit noise for routine
     ticks).

  Both steps respect an **active-window check**: Mon-Fri 08:30-17:00
  ET (one hour before market open to one hour after close). Outside
  the window they exit cleanly without work. Override with
  `--ignore-window`.

  Entry points:
  - Interactive: `/watchdog-tick` in a Claude Code session.
  - CLI deterministic: `python -m edgefinder.agents.watchdog [--dry-run] [--force] [--ignore-window]`.
  - CLI reasoning: `python -m edgefinder.agents.reasoning [--force] [--model MODEL]`.
  - Cron: `.github/workflows/watchdog.yml` (hourly, kill-switch via
    `WATCHDOG_ENABLED` repo variable).

  **Cycle liveness + heartbeat (v5.13.0).** The intraday cycle writes a
  `system_heartbeat` row (component `intraday_cycle`) at the end of every
  run — success, controlled skip (holiday/not-ready, written fresh + ok),
  or failure. `check_cycle_liveness` raises CRITICAL during market hours
  if the heartbeat goes stale (>`liveness_stale_minutes`, default 15),
  is missing, or the last run errored; it auto-resolves at the close.
  This is the detector for a stalled loop — the original "0 closed
  trades" failure mode. The holiday-skip-is-fresh trick means the check
  needs no holiday calendar of its own.

- **alerts** (`edgefinder/agents/alerts.py`) — projects unresolved
  CRITICAL observations onto GitHub issues: opens a deduped issue per
  finding (label `edgefinder-alert`, title keyed by `category/key`) and
  auto-closes it on recovery. The issue is a pure projection of DB state,
  so the loop is idempotent. Deterministic + quota-free (no Claude).
  - Cron: `.github/workflows/liveness.yml` (every 15 min during market
    hours, ET-gated, kill-switch `LIVENESS_ENABLED`). Runs the watchdog
    deterministic checks then `alerts`. `gh` is authenticated by the
    workflow `GITHUB_TOKEN` (`issues: write`); no PAT needed.
  - CLI: `python -m edgefinder.agents.alerts [--dry-run] [--force] [--ignore-window]`.

### Automation model — set it once, runs itself forever
The watchdog is designed to be fully unattended after a 5-minute
setup. After step 3 below, you never touch it again unless you want
to pause it or read what it's learned.

**What runs on its own, with no prompting:**
- GitHub Actions cron fires every hour Mon-Fri (covers both EDT +
  EST via a wide UTC range).
- Python skips cleanly outside the ET active window (08:30-17:00)
  so only ~10 ticks per day actually do work.
- Each tick: SQL checks → write observations → call Claude via
  `claude -p` → record decisions → update memory.
- DST rollovers are handled automatically by `zoneinfo` in the
  Python layer — nothing to adjust twice a year.
- Memory accumulates across ticks. Known false positives get
  suppressed without code changes.
- Kill switch is a GitHub repo variable, not code — flipping it
  off takes seconds and requires no deploy.

**What requires a human (one-time, ~5 minutes):**
1. **Mint a subscription token.** On a machine with Claude Code
   installed (including a Codespace):
   ```bash
   claude setup-token   # browser OAuth; prints a long-lived token
   ```
2. **Add three settings** to GitHub → Settings → Secrets and
   variables → Actions:
   - Secret `DATABASE_URL` = Supabase pooler URL (same as Render).
   - Secret `CLAUDE_CODE_OAUTH_TOKEN` = the token from step 1.
   - Variable `WATCHDOG_ENABLED` = `true`.
3. **Smoke test.** GitHub → Actions → Watchdog → Run workflow. If
   the run goes green and you see a `tick done` line in the logs,
   the cron will take over on the next scheduled hour.

**What requires a human (recurring):** nothing, by design. The
agent records its decisions to `agent_actions` but does not yet
auto-create GitHub issues on CRITICAL escalations — that is the
next automation step. Today you learn about escalations by
occasionally running `SELECT * FROM agent_actions WHERE status =
'pending' ORDER BY timestamp DESC` or by reading `agent_memory`.
See "Next automation step" below to close that loop.

### Pausing the watchdog
Flip the `WATCHDOG_ENABLED` repo variable to `false`. The workflow
`if:` gate skips the entire job — no deploy, no downtime, instant.

### Coach (daily strategy reviewer + tuner) — v4.9.0
Where the watchdog monitors data integrity, the **coach** monitors
**trade quality** for one strategy per weekday and proposes parameter
tweaks as PRs.

- Module: `edgefinder/agents/coach.py`. Cron:
  `.github/workflows/coach.yml` (Mon-Fri 5:30 PM ET).
- Rotation: round-robin over the live StrategyRegistry (currently coward,
  degenerate, gambler), cycled by day-of-year across weekdays — never
  drifts when strategies are added/removed. Weekends are skipped.
- Each run pulls 30 days of closed trades for that strategy from
  Supabase, the current `config/settings.py` text, and the last 3
  prior reviews of the same strategy. Hands all of it to `claude -p`,
  asks for a short review and (optionally) ONE parameter tweak.
- Always commits a `reviews/YYYY-MM-DD-<strategy>.md` markdown file.
  - No tune → committed straight to `main`.
  - Tune → opens a PR with both the review and the `settings.py` edit;
    `gh pr merge --auto --squash --delete-branch` enables auto-merge,
    so the PR lands on `main` once the test suite goes green.
- Kill switch: `COACH_ENABLED` repo variable.
- Setup checklist: `SETUP-COACH.md` at the repo root.

### Weekly portfolio summary
Cross-strategy synthesis Saturday morning. Reads the past 7 days of
`reviews/*.md` plus all closed trades, asks Claude for a portfolio-
level digest, commits to `reviews/WEEK-YYYY-WW.md`. No code changes.

- Module: `edgefinder/agents/weekly_summary.py`.
- Cron: `.github/workflows/weekly-summary.yml`.
- Kill switch: `WEEKLY_SUMMARY_ENABLED` repo variable.

### Tests workflow
`.github/workflows/tests.yml` runs `pytest -m "not integration"` (with
`test_market.py` excluded — known pre-existing flake) on every PR to
main. This is what gates the coach's auto-merge. Without it, the
auto-merge has no safety check.

### Model selection
Default reasoning model is `claude-opus-4-8`. Downgrade to Sonnet 4.6
via `WATCHDOG_REASONING_MODEL=claude-sonnet-4-6` in the workflow env
if you want to conserve subscription quota on a larger cron schedule.

### Live trading loop — in-process driver (decision updated 2026-06-05)
Both trade entry and exit run in the in-process intraday cycle
(`_signal_check_job` → `arena.run_intraday_cycle`).

**Production reality check (2026-06-05):** the Render service is a PAID
Starter instance and never idles (verified: multi-week uptime, no cold
starts). The free-tier-idling premise behind the cron-driven cutover was
wrong, so **the in-process APScheduler is the production driver**:
`intraday_external_driver` stays `false`, `INTRADAY_CYCLE_ENABLED` and
`KEEPALIVE_ENABLED` stay off. The cron machinery below is kept as a
**break-glass fallback** (flip the repo vars, no deploy). The liveness
watchdog (heartbeat → CRITICAL → GitHub issue) is the detector either
way — keep `LIVENESS_ENABLED=true`.

**The real service URL is `https://edgefinder-pm8h.onrender.com`**
(`edgefinder.onrender.com` belongs to another Render customer — never
point probes or `EDGEFINDER_URL` at the bare name). Historic root cause
of "0 closed trades" was pre-v5.10 engine bugs (unenforced caps, exits
never firing), not idling.

#### Cron-driven fallback (built v5.13.0, dormant)

- Endpoint: `POST /api/admin/run-intraday` (token-guarded by
  `eod_trigger_token`, 202 + background thread → `run_intraday_jobs`,
  which runs `_signal_check_job` then `_position_monitor_job` under a
  single-flight lock).
- Cron: `.github/workflows/intraday-cycle.yml` (every 5 min, ET-gated to
  09:30-16:00, 3×60s retry, kill-switch `INTRADAY_CYCLE_ENABLED`).
- Single driver: when `intraday_external_driver` is true
  (`EDGEFINDER_INTRADAY_EXTERNAL_DRIVER=true` on Render), `init_services`
  does NOT register the in-process intraday jobs, so the cron is the only
  driver (no double-execution). Default false ⇒ a deploy never silently
  changes the driver; local/dev keeps the in-process timer.
- `keepalive.yml` is superseded (set `KEEPALIVE_ENABLED=false`) — the
  intraday cron wakes the box itself. Kept for rollback.
- **Cutover (all config, no code):** deploy (inert) → set
  `EDGEFINDER_INTRADAY_EXTERNAL_DRIVER=true` + restart → set repo vars
  `INTRADAY_CYCLE_ENABLED=true`, `LIVENESS_ENABLED=true`,
  `KEEPALIVE_ENABLED=false`. Instant rollback = flip the repo vars off.
- **Known limits:** GitHub cron is best-effort (5-min granularity, can be
  delayed/dropped → up to a ~5-min gap); exits are evaluated on a ~5-min
  grid, not intrabar. An always-on Render worker is the upgrade path for
  real money.

### Auto-issue on CRITICAL — now built (v5.13.0)
The previously-missing "escalations reach you automatically" loop is
done: `edgefinder/agents/alerts.py` + `liveness.yml` open and auto-close
GitHub issues for unresolved CRITICAL observations (see the **alerts**
agent above). It projects from `agent_observations` (severity CRITICAL)
rather than `agent_actions`, and uses the workflow `GITHUB_TOKEN` via the
`gh` CLI — no GitHub MCP or PAT required.

### Running the watchdog from a Codespace (interactive / dev)
The Codespace is for running ticks on demand while you iterate, not
for replacing the cron. Unattended scheduling stays on GitHub
Actions because Codespaces auto-stop after 30 min idle.

1. **Set Codespaces secrets** (one-time, per-user): GitHub →
   Settings → Codespaces → Secrets. Add `DATABASE_URL` and
   `CLAUDE_CODE_OAUTH_TOKEN`, scoped to this repo.
2. **Open the repo in a Codespace** (green Code button → Codespaces).
   The committed `.devcontainer/devcontainer.json` installs Python +
   Node + the package + `@anthropic-ai/claude-code` on container
   create, so the environment is ready without manual setup.
3. **Run ticks on demand:**
   ```bash
   python -m edgefinder.agents.watchdog --force         # deterministic checks
   python -m edgefinder.agents.reasoning --force        # LLM step
   python -m edgefinder.agents.watchdog --dry-run       # preview without writes
   ```
