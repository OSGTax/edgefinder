# EdgeFinder v2 — Trading Workbench

## Project Overview
EdgeFinder is a trading workbench for strategy research, paper trading, and performance analysis.
It combines fundamental data from Polygon.io, technical signal detection,
multi-source sentiment analysis, and multi-strategy competition in isolated virtual accounts.

**Key principles:**
- Polygon.io is the sole data source (no fallback chains)
- Every strategy gets its own isolated $5,000 virtual account
- Every trade captures a market-wide snapshot (SPY, QQQ, IWM, DIA, VIX, sectors)
- Per-strategy views everywhere — no aggregate P&L
- AI meta-strategy interfaces are built in (future Phase 8)

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
│   │   ├── alpha.py            # Momentum/EMA day trading
│   │   ├── bravo.py            # Mean reversion/BB swing trading
│   │   └── charlie.py          # Deep value contrarian
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
│   └── templates/
│       └── index.html          # Dashboard frontend
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
| POST | `/api/inject` | Inject ticker for evaluation |
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
- **watchdog** (`edgefinder/agents/watchdog.py`) — DB-only health
  monitor. Checks cash drift, negative cash, paused accounts, high
  drawdown. Reconciles against unresolved observations so conditions
  that clear are auto-resolved and persisting conditions don't spam.
  Run one tick:
  - Interactive: `/watchdog-tick` in a Claude Code session.
  - CLI: `python -m edgefinder.agents.watchdog [--dry-run] [--force]`.
  - Cron: `.github/workflows/watchdog.yml` (every 15 min when
    `WATCHDOG_ENABLED=true`).

### Enabling the watchdog cron for the first time
1. Repo → Settings → Secrets and variables → Actions:
   - Secret `DATABASE_URL` = Supabase pooler URL (same as Render).
   - Variable `WATCHDOG_ENABLED` = `true`.
2. Actions → Watchdog → Run workflow (smoke test the cron).
3. Check `agent_observations` for the first findings. A clean system
   should produce 0 rows on the first tick.
