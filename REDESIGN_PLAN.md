# EdgeFinder v2 — Complete Redesign Plan

## Context

EdgeFinder v1 is a fully-built paper trading system (~80KB arena, 21 strategies, 15+ indicators, dashboard). However, the design evolved organically and now has fundamental issues: data sources are too limited for complex strategies, the watchlist is redundant, there's no market benchmarking, sentiment is basic (VADER on RSS only), no social media integration, no market context captured with trades, and the dashboard can't filter by strategy. The user wants to **reset the repo and rewrite from scratch** with a clearer vision — shifting from "automated trading pipeline" to a **trading workbench** for research, strategy development, and eventually an AI meta-strategy.

**Key user decisions:**
- Real-time data required (not just delayed/free tier)
- Social sentiment from Reddit + Twitter/X
- Per-strategy views everywhere (no aggregate P&L)
- Research tool replaces watchlist
- Market-wide snapshot on every trade
- AI meta-strategy is future build — but design interfaces for it now

---

## Implementation Approach: Clean Slate

**ALL existing code will be deleted.** The repo will be reset to contain only:
- This plan document (saved as `REDESIGN_PLAN.md` in repo root)
- `CLAUDE.md` (updated for v2)
- `.gitignore`

Every file will be written fresh — no porting, no legacy baggage. If unsure about how something should work, **ask before building**. The goal is solid code from line 1.

---

## New Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard (FastAPI)                    │
│  Research | Trades | Strategies | Benchmarks | Inject    │
└────────────┬────────────────────────────────┬───────────┘
             │                                │
┌────────────▼────────┐         ┌─────────────▼──────────┐
│   Research Service   │         │   Trading Engine       │
│ (aggregates all data │         │ Arena → Executor →     │
│  per-ticker)         │         │ Virtual Accounts       │
└──────────┬──────────┘         └──────┬─────────────────┘
           │                           │
    ┌──────┴──────┬──────────┬────────┴────────┐
    │             │          │                  │
┌───▼───┐  ┌─────▼────┐ ┌───▼──────┐  ┌───────▼───────┐
│Scanner│  │Sentiment  │ │ Market   │  │  Strategies   │
│       │  │Aggregator │ │ Snapshot │  │  (plugins)    │
└───┬───┘  └─────┬────┘ └───┬──────┘  └───────┬───────┘
    │            │           │                  │
    └────────────┴───────────┴──────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Data Layer         │
              │  (Protocol-based)   │
              │  Polygon > Alpaca > │
              │  yfinance fallback  │
              └─────────────────────┘
```

---

## Virtual Account Rules

Each strategy operates its own **isolated $5,000 virtual account** with these rules:

| Rule | Detail |
|------|--------|
| Starting capital | $5,000 per strategy |
| Buying power | Cash balance only — cannot trade more than available cash minus open position value |
| PDT mode | **Per-strategy toggle** — some strategies day trade, some swing only |
| Open trades | Visible in account view with unrealized P&L |
| Account view | Shows: open trades + unrealized gains/losses, cash balance, total equity |
| No margin/leverage | Account value is the hard ceiling |

**Dashboard per-strategy account card:**
```
┌─ Alpha Strategy ──────────────────────┐
│ Cash:    $3,200    Equity:  $4,850    │
│ Open:    2 trades  PDT:     Enabled   │
│ ┌──────┬────────┬────────┬──────────┐ │
│ │ AAPL │ +$120  │ +2.4%  │ 50 shr   │ │
│ │ MSFT │ -$70   │ -1.1%  │ 30 shr   │ │
│ └──────┴────────┴────────┴──────────┘ │
│ Realized P&L: +$650  Win Rate: 62%    │
└───────────────────────────────────────┘
```

---

## Directory Structure

```
edgefinder/
├── pyproject.toml                     # Replaces setup.py + requirements.txt
├── alembic.ini                        # DB migrations
├── config/
│   ├── settings.py                    # All tunable parameters
│   └── secrets.env.example
├── edgefinder/                        # Main package
│   ├── core/
│   │   ├── models.py                  # Pydantic domain models
│   │   ├── interfaces.py             # Protocols: DataProvider, SentimentProvider, etc.
│   │   └── events.py                  # In-process event bus (pub/sub)
│   ├── data/
│   │   ├── provider.py               # DataProvider factory + CachedDataProvider wrapper
│   │   ├── polygon.py                # Polygon.io (primary, real-time)
│   │   ├── alpaca.py                 # Alpaca (fallback)
│   │   ├── yfinance_provider.py      # yfinance (last resort)
│   │   ├── cache.py                  # Local cache layer
│   │   └── stream.py                 # WebSocket streaming abstraction
│   ├── db/
│   │   ├── engine.py                 # SQLAlchemy engine/session (SQLite/PostgreSQL)
│   │   ├── models.py                 # ORM models (new schema)
│   │   └── migrations/               # Alembic migrations
│   ├── scanner/
│   │   └── scanner.py                # Nightly fundamental scan
│   ├── research/
│   │   └── research.py               # Per-ticker research aggregation (replaces watchlist)
│   ├── sentiment/
│   │   ├── provider.py               # SentimentProvider protocol
│   │   ├── reddit.py                 # Reddit API (r/wallstreetbets, r/stocks)
│   │   ├── twitter.py                # Twitter/X financial accounts
│   │   ├── news_rss.py               # RSS feeds (upgraded from v1)
│   │   └── aggregator.py             # Combines all sources, weighted scoring
│   ├── market/
│   │   ├── snapshot.py               # Captures SPY/QQQ/IWM/DIA/VIX/sectors at trade time
│   │   └── benchmarks.py             # Daily index data for comparison charts
│   ├── signals/
│   │   └── engine.py                 # Technical indicator computation + signal generation
│   ├── strategies/
│   │   ├── base.py                   # BaseStrategy ABC + StrategyRegistry + AI agent hooks
│   │   ├── alpha.py, bravo.py, ...   # Strategy plugins
│   │   └── (ai_agent.py)             # Future: AI meta-strategy
│   ├── trading/
│   │   ├── arena.py                  # Multi-strategy orchestration
│   │   ├── executor.py               # Execution with slippage modeling
│   │   ├── account.py                # Per-strategy virtual account
│   │   └── journal.py                # Trade logging with market context
│   └── scheduler/
│       └── scheduler.py              # APScheduler (ET timezone)
├── dashboard/
│   ├── app.py                        # FastAPI app + lifespan
│   ├── routers/
│   │   ├── trades.py                 # Wins/losses/open/closed, strategy-filterable
│   │   ├── strategies.py             # Per-strategy equity curves + comparison
│   │   ├── research.py               # Research tool endpoints
│   │   ├── sentiment.py              # Sentiment data endpoints
│   │   ├── benchmarks.py             # Strategy vs index comparison
│   │   └── inject.py                 # Manual ticker injection
│   └── templates/
├── scripts/
│   ├── setup_db.py
│   ├── run_scanner.py
│   └── render_start.py
└── tests/
```

---

## Database Schema (Key Tables)

**`tickers`** — Master registry (replaces watchlist)
- `id`, `symbol`, `company_name`, `sector`, `industry`, `market_cap`, `last_price`
- `source` (scanner / manual / injected), `is_active`, timestamps

**`fundamentals`** — Latest scan data per ticker (one row, overwritten each scan)
- All Lynch/Burry metrics + `raw_data` JSON for research tool deep dive

**`trades`** — Immutable trade log (per-strategy, FK to market snapshot)
- `strategy_name`, `symbol`, `direction`, `trade_type`
- Entry/exit prices, shares, stop/target, confidence
- `pnl_dollars`, `pnl_percent`, `r_multiple`, `exit_reason`, `status`
- **`market_snapshot_id` FK** — links trade to broad market state at execution
- `sentiment_score`, `sentiment_data` JSON, `technical_signals` JSON
- `sequence_num` + `integrity_hash` (audit chain)

**`market_snapshots`** — Captured at every trade + periodically
- SPY/QQQ/IWM/DIA prices + change %, VIX level
- `market_regime` (bull/bear/sideways), sector ETF performance JSON, advance/decline ratio

**`strategy_accounts`** — Per-strategy virtual account state (live, updated on every trade)
- `strategy_name`, `starting_capital` ($5,000), `cash_balance`, `open_positions_value`
- `buying_power` = cash_balance (cannot exceed cash minus open positions)
- `pdt_enabled` BOOLEAN — togglable per strategy (day trading rules on/off)
- `total_equity`, `peak_equity`, `drawdown_pct`
- `is_paused`, timestamps

**`strategy_snapshots`** — Per-strategy equity curve (daily)
- `strategy_name`, cash, positions_value, total_equity, drawdown, returns

**`index_daily`** — Daily index closes for benchmark charts
- symbol (SPY/QQQ/IWM/DIA), date, close, change_pct

**`sentiment_readings`** — Per-ticker sentiment time series
- `symbol`, `source` (reddit/twitter/news), score (-1 to +1), mention_count, is_trending, timestamp

**`manual_injections`** — User-injected tickers
- `symbol`, `target_strategy` (null = all), `expires_at`, `notes`

**`strategy_parameters`** — Parameter change audit log
- `strategy_name`, param name/old/new, `changed_by` (optimizer/user/ai_agent)

---

## Key Interfaces

### Data Provider (Protocol-based, swappable)
```python
class DataProvider(Protocol):
    def get_bars(self, ticker, timeframe, start, end) -> DataFrame | None
    def get_latest_price(self, ticker) -> float | None
    def get_fundamentals(self, ticker) -> dict | None
    def is_market_open(self) -> bool

class StreamProvider(Protocol):  # For real-time
    async def subscribe(self, tickers, callback) -> None
    async def connect(self) -> None
```
Factory reads config -> instantiates Polygon (if key set) > Alpaca > yfinance. Consuming code never imports a specific provider.

### Strategy Plugin (v2 — adds AI hooks)
```python
class BaseStrategy(ABC):
    # Existing (unchanged):
    name, version, preferred_signals, init(), generate_signals(),
    on_trade_executed(), qualifies_stock(), get_watchlist(),
    on_market_regime_change(), on_strategy_pause()

    # NEW — market context:
    def on_market_snapshot(self, snapshot: MarketSnapshot) -> None

    # NEW — AI agent hooks (future):
    def get_state(self) -> dict          # AI reads strategy's internal state
    def apply_suggestion(self, suggestion: dict) -> bool  # AI tunes strategy
```

### Sentiment Aggregator
```python
class SentimentProvider(Protocol):
    source_name: str
    def get_sentiment(self, ticker) -> TickerSentiment
    def get_trending(self) -> list[TickerSentiment]

class SentimentAggregator:
    # Combines Reddit + Twitter + News, weighted by source reliability
    # Stores readings to DB, exposes per-ticker score for trade gating
```

### Research Service (read-only aggregation)
```python
class ResearchService:
    def get_ticker_report(self, symbol) -> TickerReport
    # Aggregates: fundamentals + current price + indicators + sentiment
    # + trade history + which strategies qualify this stock
```

### AI Agent Data Access (design now, build later)
```python
class AIAgentDataAccess(Protocol):
    def get_all_trades(strategy, status, since) -> list[Trade]
    def get_market_snapshots(since) -> list[MarketSnapshot]
    def get_sentiment_history(symbol, since) -> list[SentimentReading]
    def get_strategy_states() -> dict[str, dict]  # calls get_state() on each
    def suggest_to_strategy(name, suggestion) -> bool
```
The DB schema already supports every query this interface needs.

---

## Build Order

### Phase 1: Foundation
- `pyproject.toml`, package structure, config
- `core/models.py` (Pydantic domain models), `core/interfaces.py` (Protocols)
- `db/engine.py` + `db/models.py` (new schema) + Alembic setup
- `data/provider.py` + `yfinance_provider.py` + `cache.py` (start with free data)
- Tests for data layer and DB

### Phase 2: Scanner + Strategy System
- `scanner/scanner.py` (rewrite for new schema)
- `strategies/base.py` (v2 with AI hooks)
- Port 2-3 strategies (alpha, bravo, charlie) to validate interface
- `signals/engine.py` (port signal engine)

### Phase 3: Trading Engine
- `trading/account.py`, `trading/executor.py`, `trading/arena.py`
- `trading/journal.py` (with market context integration)
- Full pipeline test: scan -> signal -> execute -> journal

### Phase 4: Market Context + Benchmarks
- `market/snapshot.py` (captures indices, VIX, sectors at trade time)
- `market/benchmarks.py` (daily index data collection)
- Wire snapshot into trade execution

### Phase 5: Sentiment System
- `sentiment/news_rss.py` (upgraded from v1)
- `sentiment/reddit.py` (Reddit API)
- `sentiment/twitter.py` (Twitter/X)
- `sentiment/aggregator.py` (weighted combination)
- Wire into trade gating

### Phase 6: Research Tool + Dashboard
- `research/research.py` (per-ticker aggregation)
- Dashboard routers: trades (wins/losses/open/closed), strategies (equity curves), research, benchmarks, inject
- Strategy selector/filter, strategy-vs-index charts
- Manual ticker injection endpoint

### Phase 7: Real-Time Data Upgrade
- `data/polygon.py` or `data/alpaca.py` (paid tier)
- `data/stream.py` (WebSocket streaming)
- Port remaining strategies

### Phase 8: Scheduler + Deployment
- APScheduler jobs for all features
- Render deployment config
- End-to-end integration testing

### Phase 9 (Future): AI Meta-Strategy
- `strategies/ai_agent.py` as a strategy plugin
- Uses `AIAgentDataAccess` to query trade history + market snapshots
- Calls `get_state()`/`apply_suggestion()` on other strategies
- Backtests against historical data

---

## Verification Plan

After each phase, verify by:
1. `python -m pytest tests/ -v -m "not integration"` — all unit tests pass
2. Phase 1: Can fetch bars and fundamentals through abstracted data provider
3. Phase 2: `python scripts/run_scanner.py --quick` produces scored results
4. Phase 3: Simulated trade pipeline executes end-to-end
5. Phase 4: Trades in DB have non-null `market_snapshot_id`
6. Phase 5: `SentimentAggregator` returns scores from multiple sources
7. Phase 6: Dashboard loads, strategy filter works, benchmark charts render
8. Phase 7: Real-time prices stream via WebSocket
9. Phase 8: Full system runs on schedule, deployed to Render
