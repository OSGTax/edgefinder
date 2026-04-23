# EdgeFinder v2 — Trading Workbench

## Project Overview
EdgeFinder is a trading workbench for strategy research, paper trading, and performance analysis.
It combines fundamental data from Polygon.io, technical signal detection,
Polygon News AI-insight sentiment enrichment, and multi-strategy competition in isolated virtual accounts.

**Key principles:**
- Polygon.io is the sole data source (no fallback chains). The Python
  SDK ships as the `massive` package on PyPI (Polygon rebranded to
  "Massive" in 2025); the vendor and capabilities are unchanged.
- Every strategy gets its own isolated $5,000 virtual account.
- Every trade captures a market-wide snapshot (SPY, QQQ, IWM, DIA, VIX,
  sectors) in the SAME transaction as the trade row write. If Polygon
  snapshot capture fails, the trade still persists with
  `market_snapshot_id=NULL`; run `scripts/repair_orphan_snapshots.py` to
  backfill from nearest-in-time snapshots.
- Per-strategy views everywhere — no aggregate P&L.
- AI meta-strategy interfaces are built in (future Phase 8).

## Architecture
```
Dashboard (FastAPI) — Research | Trades | Strategies | Benchmarks | Inject
        |                                    |
  Research Service              Trading Engine (Arena)
  (per-ticker aggregation)      Executor → Virtual Accounts
        |                                    |
  UnifiedScanner  | Market Snapshot | Strategies (plugins)
        |                |                |
                  Data Layer (Polygon.io — `massive` SDK)
```

## Tech Stack
- Python 3.11+, FastAPI, SQLAlchemy 2.0, Alembic
- `massive>=2.4.0` — Polygon.io REST client (bars, fundamentals, universe)
- pydantic + pydantic-settings (domain models, typed config)
- APScheduler (ET timezone scheduling — Render only; Vercel is not
  supported and the config has been removed)
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
│   │   ├── interfaces.py       # Protocol: DataProvider (+ DataHub wrapper)
│   │   └── events.py           # In-process event bus — `trade.opened`, `trade.closed` live
│   ├── data/
│   │   ├── polygon.py          # Polygon REST (bars, fundamentals, universe, price, news sentiment)
│   │   ├── provider.py         # CachedDataProvider wrapper
│   │   └── cache.py            # Filesystem cache (Parquet bars, JSON fundamentals)
│   ├── db/
│   │   ├── engine.py           # SQLAlchemy engine/session (SQLite/PostgreSQL)
│   │   ├── models.py           # ORM tables (see live list via `SELECT name FROM sqlite_master`)
│   │   └── migrations/         # Alembic migrations
│   ├── scanner/
│   │   └── unified_scanner.py  # Nightly fundamental scan — evaluates all strategies in one pass
│   ├── signals/
│   │   └── engine.py           # Technical indicators + 9 signal pattern detectors
│   ├── strategies/
│   │   ├── base.py             # BaseStrategy ABC + StrategyRegistry
│   │   ├── alpha.py            # Momentum/EMA day trading
│   │   ├── bravo.py            # Mean reversion/BB swing trading
│   │   ├── charlie.py          # Deep value contrarian
│   │   ├── degenerate.py       # High-risk / high-concentration plays
│   │   └── echo.py             # Learning meta-strategy (uses edgefinder/analytics/)
│   ├── analytics/              # Regime + trade-feature analytics (consumed by echo.py)
│   ├── trading/
│   │   ├── account.py          # Per-strategy $5k virtual accounts, mark-to-market equity
│   │   ├── executor.py         # Risk-based sizing, slippage, hash chain audit + verify
│   │   ├── arena.py            # Multi-strategy orchestration
│   │   └── journal.py          # Trade persistence + stats
│   ├── market/
│   │   ├── snapshot.py         # Captures indices/VIX/sectors at trade time
│   │   └── benchmarks.py       # Daily index data for comparison charts
│   ├── research/
│   │   └── research.py         # Per-ticker deep-dive aggregation (reads injections for reports)
│   ├── agents/                 # Watchdog + reasoning agents (kill-switched via .claude/agent-config.json)
│   └── scheduler/
│       └── scheduler.py        # APScheduler (ET timezone)
├── dashboard/
│   ├── app.py                  # FastAPI application (bearer auth, CORS allowlist)
│   ├── services.py             # init_services, _load_watchlists, event handlers
│   ├── dependencies.py         # DB session dependency injection
│   ├── routers/
│   │   ├── trades.py           # Wins/losses/open/closed, strategy-filterable
│   │   ├── strategies.py       # Per-strategy accounts + equity curves + open positions
│   │   ├── research.py         # Ticker reports + search + POST /scan trigger
│   │   ├── benchmarks.py       # Strategy vs index comparison + sector data
│   │   └── inject.py           # Manual ticker injection (wired into watchlist)
│   └── templates/
│       └── index.html          # Dashboard frontend
├── scripts/
│   ├── setup_db.py                   # Initialize database
│   ├── run_scanner.py                # CLI scanner (UnifiedScanner, --quick, --tickers)
│   ├── repair_orphan_snapshots.py    # Backfill trades.market_snapshot_id for NULL rows
│   └── render_start.py               # Render deployment startup (provisions agent-config)
└── tests/                      # 347 tests (`pytest -m "not integration"`)
```

### Removed in the 2026-04 cleanup (no longer referenced)
- `vercel.json` + `api/index.py` — Vercel serverless cannot run
  APScheduler. Render is the sole deploy target.
- `edgefinder/data/stream.py` + `StreamProvider`, `SupplementalProvider`
  protocols — streaming was never wired; system is polling-only.
- `edgefinder/scanner/strategy_scanner.py` — replaced by `unified_scanner.py`.
- `edgefinder/sentiment/` — never existed in code. Sentiment comes from
  Polygon News AI insights populating `fundamentals.news_sentiment`,
  which `on_trade_opened` copies onto `trades.sentiment_data`.
- `trades.sentiment_score`, `strategy_parameters`, and `trade_context`
  tables/columns — dropped via Alembic migrations
  `a1d9f3c0b2e4_drop_strategy_parameters` and
  `b7e2f8a3c5d1_drop_sentiment_score_and_trade_context`.

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

### Optional production env vars
- `EDGEFINDER_DASHBOARD_TOKEN` — when set, every non-exempt route
  requires `Authorization: Bearer <token>`. Exempt: `/`, `/api/health`,
  `/docs`, `/redoc`, `/openapi.json`.
- `EDGEFINDER_CORS_ORIGINS` — comma-separated allowlist. Defaults to
  `http://localhost:8000,http://127.0.0.1:8000` so prod isn't wide-open
  unless you opt in. Setting to `*` restores unrestricted CORS but
  disables credentialed requests.
- `AGENT_CONFIG_JSON` — on Render, the JSON body written to
  `.claude/agent-config.json` at boot. Required to enable the
  in-process management agents; without it, `is_agent_enabled()` returns
  False and the agent layer no-ops.

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
| GET | `/api/strategies/positions` | All open positions across strategies |
| GET | `/api/strategies/equity-curve?days=90` | Equity curve data |
| GET | `/api/research/ticker/{symbol}` | Full ticker research report |
| GET | `/api/research/search?q=X` | Search tickers |
| GET | `/api/research/active` | Active watchlist tickers |
| POST | `/api/research/scan` | Trigger a unified multi-strategy scan in the background |
| GET | `/api/benchmarks/comparison?days=90` | Strategy vs index data |
| GET | `/api/benchmarks/sectors` | Current sector rotation snapshot |
| POST | `/api/benchmarks/collect` | Collect daily benchmark data |
| POST | `/api/benchmarks/backfill` | Backfill historical benchmark bars |
| POST | `/api/inject` | Inject ticker for evaluation — reaches the arena watchlist on the next signal check |
| GET | `/api/inject` | List active injections |
| DELETE | `/api/inject/{id}` | Remove injection |

Sentiment-only endpoints that appeared in earlier versions of this doc
(`/api/sentiment/*`) were never implemented — the multi-source sentiment
module they referenced doesn't exist. Sentiment is surfaced indirectly
via `trades.sentiment_data.news_sentiment`.

## Virtual Account Rules
- $5,000 starting capital per strategy
- Buying power = cash only (no margin/leverage)
- PDT mode: per-strategy toggle (3 day trades / 5 **business** days,
  counted via `numpy.busday_offset`)
- Max risk per trade: 2% of equity
- Max concentration: 20% in single position
- Max open positions: 5
- Max positions per sector: 3 (enforced via `Position.sector` looked up
  from `Fundamental.sector`)
- Drawdown circuit breaker: 20%, computed on **mark-to-market** equity
- Revenge trade cooldown: 30 minutes after stop-out — state persisted
  via `_last_stop_out` and restored on every boot from the most recent
  `STOP_HIT` trade per strategy

### Account Balance Integrity (CRITICAL)
The trades table is the **source of truth** for all account balances. On every startup,
`_recalculate_account_balances()` recomputes cash from trades to self-heal any corruption:
```
correct_cash = starting_capital + sum(closed trade P&L) - sum(open position cost basis)
```
Additionally, every executor's integrity hash chain is restored from
`trades.sequence_num`/`integrity_hash` and then verified end-to-end by
`verify_chain()`. A mismatch logs a loud WARNING but does not halt
trading — investigate `agent_observations` and the trades table before
assuming the chain is corrupt rather than legitimately extended.

**Rules for all strategies (existing and new):**
1. Every strategy uses the same `VirtualAccount` class — no custom account logic.
2. Account state is persisted to DB immediately after every trade open/close AND on shutdown.
3. On startup, cash and realized P&L are always recalculated from the trades table.
4. **Total Account Value = Cash + market value of positions** (mark-to-market,
   not cost basis). `VirtualAccount.total_equity` uses
   `Position.current_price` stamped by `Executor.check_positions` each
   tick; before the first tick it falls back to `entry_price`.
5. Total P&L = Total Account Value − Starting Capital (canonical formula).
6. Realized P&L = sum of `pnl_dollars` from closed trades in DB.
7. Unrealized P&L = sum of `(current_price − entry_price) × shares` for open positions.
8. `Peak_equity` is updated on every tick (not just on close) so
   drawdown tracks real account value.

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
- Scheduling (scanner 6:15 PM ET, signals every 5m, positions every 5m)
- Polygon.io connection (API key, retries, timeouts)

## Testing
```bash
python -m pytest tests/ -v -m "not integration"   # Unit tests (347)
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
     drawdown. Writes to `agent_observations`, reconciles against
     prior unresolved rows (dedup + auto-resolve).
  2. **Agentic reasoning** (`edgefinder/agents/reasoning.py`) — calls
     Claude (default `claude-opus-4-7`, adaptive thinking, prompt
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

### Model selection
Default reasoning model is `claude-opus-4-7`. Downgrade to Sonnet 4.6
via `WATCHDOG_REASONING_MODEL=claude-sonnet-4-6` in the workflow env
if you want to conserve subscription quota on a larger cron schedule.

### Next automation step (not yet built)
To make the loop fully hands-off: add a step at the end of the
workflow that reads `agent_actions` rows with `status='pending'` and
`action_type='diagnose'` from the current tick, then calls
`mcp__github__issue_write` (GitHub MCP) to create an issue for each
one, and updates the row to `status='submitted'`. Then every
escalation reaches you as a GitHub notification with no manual
checking. Ping in a future session to build this.

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
