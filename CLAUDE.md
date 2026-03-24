# EdgeFinder — Intelligent Paper Trading System

## Project Overview
EdgeFinder is a self-improving paper trading system that combines fundamental analysis
(Peter Lynch + Michael Burry criteria), technical signals, and news sentiment filtering
to simulate intelligent stock trading. It starts rule-based and graduates to ML.

## Architecture
```
Layer 1: Fundamental Scanner  → Nightly scan, ~8000 stocks → ~50-100 candidates
Layer 2: Technical Signal Engine → Every 15-30 min, monitors candidates for entries
Layer 2.5: News Sentiment Gate  → Go/no-go filter before trade execution
Layer 3: Paper Trader           → Simulated execution with risk management
Layer 4: Trade Journal          → Logs every trade + skipped signal with full context
Layer 5: Strategy Optimizer     → Weekly analysis, adjusts parameters
```

## Tech Stack
- Python 3.11+, FastAPI, SQLite, yfinance, pandas-ta, VADER sentiment
- Free hosting (Render / PythonAnywhere)
- Claude Code for strategy analysis and code generation

## Directory Structure
```
edgefinder/
├── CLAUDE.md              ← YOU ARE HERE — master instructions
├── README.md              ← Human-readable project overview
├── requirements.txt       ← Python dependencies
├── setup.py               ← Package setup
├── config/
│   ├── settings.py        ← All tunable parameters (thresholds, weights, etc.)
│   └── secrets.env.example ← Template for API keys (if needed later)
├── modules/
│   ├── __init__.py
│   ├── scanner.py         ← Module 1: Fundamental Scanner
│   ├── signals.py         ← Module 2: Technical Signal Engine
│   ├── sentiment.py       ← Module 2.5: News Sentiment Gate
│   ├── trader.py          ← Module 3: Paper Trader
│   ├── journal.py         ← Module 4: Trade Journal
│   ├── optimizer.py       ← Module 5: Strategy Optimizer
│   └── database.py        ← Database models and helpers
├── tests/
│   ├── __init__.py
│   ├── test_scanner.py    ← Module 1 tests (BUILT)
│   ├── test_signals.py    ← Module 2 tests (PLACEHOLDER)
│   ├── test_sentiment.py  ← Module 2.5 tests (PLACEHOLDER)
│   ├── test_trader.py     ← Module 3 tests (PLACEHOLDER)
│   ├── test_journal.py    ← Module 4 tests (PLACEHOLDER)
│   ├── test_optimizer.py  ← Module 5 tests (PLACEHOLDER)
│   └── conftest.py        ← Shared test fixtures
├── scripts/
│   ├── run_scanner.py     ← CLI: run nightly scan
│   ├── run_tests.py       ← CLI: run all tests with summary
│   ├── setup_db.py        ← CLI: initialize database
│   └── verify_install.py  ← CLI: check all dependencies
├── data/
│   └── edgefinder.db      ← SQLite database (created at runtime)
├── dashboard/             ← Web UI (Phase 5)
└── docs/
    └── EdgeFinder_Project_Spec.docx
```

## Git & GitHub Workflow
This project lives in a GitHub repo. Follow these conventions:

### Branch Strategy
```
main              ← stable, tested, working code only
├── module-1      ← fundamental scanner (COMPLETE)
├── module-2      ← technical signal engine
├── module-2.5    ← news sentiment gate
├── module-3-4    ← paper trader + journal
├── module-5      ← strategy optimizer
└── dashboard     ← web UI
```

### Rules
- **NEVER push directly to main.** Always work on a feature branch.
- Each module gets its own branch off `main`.
- A module branch merges to `main` ONLY after ALL its tests pass.
- Commit messages follow: `[module-N] short description`
  - Examples: `[module-1] fix normalization bug in Lynch scoring`
  - Examples: `[module-2] add RSI oversold signal detection`
- Run `python -m pytest tests/ -v -m "not integration"` before every commit.

### Starting a New Module
```bash
git checkout main
git pull origin main
git checkout -b module-2
# ... build and test ...
git add -A
git commit -m "[module-2] implement technical signal engine"
git push origin module-2
# Create PR on GitHub → merge after review
```

### First-Time Setup (after cloning)
```bash
git clone <repo-url>
cd edgefinder
pip install -r requirements.txt
python scripts/verify_install.py
python scripts/setup_db.py
python -m pytest tests/test_scanner.py -v
```

## Build Order — FOLLOW THIS EXACTLY
Each module MUST pass its tests before moving to the next.

### Phase 1: Foundation + Module 1 (Fundamental Scanner)
1. Run `python scripts/verify_install.py` — installs and verifies all dependencies
2. Run `python scripts/setup_db.py` — creates SQLite database
3. Run `python -m pytest tests/test_scanner.py -v` — Module 1 tests
4. Run `python scripts/run_scanner.py` — live scan (uses real yfinance data)
5. Review output: does the watchlist look reasonable? (HUMAN CHECK)

### Phase 2: Module 2 (Technical Signal Engine)
1. Build `modules/signals.py` following the spec in config/settings.py
2. Write tests in `tests/test_signals.py`
3. Run `python -m pytest tests/test_signals.py -v`
4. Integration test: scanner → signals pipeline

### Phase 3: Module 2.5 (News Sentiment Gate)
1. Build `modules/sentiment.py`
2. Write tests in `tests/test_sentiment.py`
3. Run `python -m pytest tests/test_sentiment.py -v`

### Phase 4: Modules 3-4 (Paper Trader + Journal)
1. Build `modules/trader.py` and `modules/journal.py`
2. Write tests
3. Integration test: full pipeline scan → signal → sentiment → trade → log

### Phase 5: Dashboard
1. Build FastAPI backend in `dashboard/`
2. Build lightweight frontend

### Phase 6: Module 5 (Strategy Optimizer)
1. Build `modules/optimizer.py`
2. Requires 50+ logged trades to be meaningful

## Coding Standards
- Type hints on all functions
- Docstrings on all public methods
- All database operations use context managers
- Never hardcode thresholds — always reference config/settings.py
- Log everything with Python logging module (level=INFO default)
- Handle yfinance failures gracefully (it WILL return None/NaN randomly)

## HUMAN_ACTION_REQUIRED Convention
Any time the human needs to do something, mark it clearly:
```
# ============================================================
# HUMAN_ACTION_REQUIRED
# What: [description of what they need to do]
# Why: [why this can't be automated]
# How: [step-by-step instructions]
# ============================================================
```

## Testing Convention
- Every module has a matching test file
- Tests use pytest with fixtures from conftest.py
- Mock external APIs (yfinance) in unit tests — don't hit real APIs in CI
- Include at least one integration test per module that DOES hit real APIs
  (marked with @pytest.mark.integration so they can be skipped)
- Test files include a `TestResults` summary block at the bottom

## Key Parameters (from config/settings.py)
- Starting capital: $2,500
- Max risk per trade: 2% ($50)
- Max open positions: 5
- PDT limit: 3 day trades / 5 rolling business days
- Watchlist gate: strategy-driven (each strategy's `qualifies_stock()` decides)
- Composite score: still computed for display/sorting (Lynch 50% + Burry 50%)
- Fallback threshold: composite >= 60 (only used if no strategies are loaded)

## Strategy Plugin Guide
When someone says "I want to create a trading strategy" or "I'm building a strategy plugin,"
use this section to guide them through the process.

### Available Data
The scanner collects and stores the following fundamental data for every stock that passes
basic pre-screening (market cap, price, volume). Your strategy can use any of these fields:

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | str | Stock symbol (e.g., "AAPL") |
| `company_name` | str | Company name |
| `sector` | str | GICS sector (e.g., "Technology") |
| `industry` | str | Industry classification |
| `market_cap` | float | Market capitalization in dollars |
| `price` | float | Current stock price |
| `peg_ratio` | float | Price/Earnings-to-Growth ratio |
| `earnings_growth` | float | Annual earnings growth rate (0.25 = 25%) |
| `debt_to_equity` | float | Debt-to-equity ratio |
| `revenue_growth` | float | Annual revenue growth rate |
| `institutional_pct` | float | Institutional ownership percentage (0.45 = 45%) |
| `lynch_score` | float | Peter Lynch composite score (0-100) |
| `lynch_category` | str | Lynch classification: fast_grower, stalwart, turnaround, asset_play, cyclical, slow_grower |
| `fcf_yield` | float | Free cash flow yield (0.08 = 8%) |
| `price_to_tangible_book` | float | Price to tangible book value |
| `short_interest` | float | Short interest as percentage of float |
| `ev_to_ebitda` | float | Enterprise Value / EBITDA |
| `current_ratio` | float | Current assets / Current liabilities |
| `burry_score` | float | Michael Burry composite score (0-100) |
| `composite_score` | float | Weighted average of Lynch + Burry scores |

### How to Build a Strategy Plugin
1. Create a file: `modules/strategies/my_strategy.py`
2. Subclass `BaseStrategy` from `modules/strategies/base.py`
3. Register with `@StrategyRegistry.register("my_strategy")`

### Required Methods
```python
from modules.strategies.base import BaseStrategy, StrategyRegistry, Signal, TradeNotification

@StrategyRegistry.register("my_strategy")
class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def init(self) -> None:
        """Setup before trading begins."""
        pass

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Return True if this strategy wants the stock on its watchlist.
        The scanner uses this to decide which stocks stay active.
        stock_data contains all fields from the Available Data table above."""
        return stock_data.get("revenue_growth", 0) >= 0.20  # example

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate trading signals from OHLCV data.
        bars: dict of ticker -> DataFrame with Open, High, Low, Close, Volume columns."""
        return []

    def on_trade_executed(self, notification: TradeNotification) -> None:
        """Called after a trade is opened or closed."""
        pass
```

### Optional Methods
- `set_watchlist(scored_stocks: list[dict])` — receive pre-scored stocks from the scanner
- `get_watchlist() -> list[str]` — return tickers this strategy wants data for
- `on_market_regime_change(regime: MarketRegime)` — react to bull/bear/sideways changes
- `on_strategy_pause(reason: str)` — handle auto-pause due to drawdown

### Post-Build Checklist
- [ ] Strategy file in `modules/strategies/`
- [ ] Registered via `@StrategyRegistry.register()` decorator
- [ ] `qualifies_stock()` returns True for stocks your strategy wants
- [ ] `generate_signals()` returns `Signal` objects with valid stop_loss and target
- [ ] Tests in `tests/test_<strategy_name>.py`
- [ ] Run `python -m pytest tests/ -v -m "not integration"` — all pass
- [ ] Strategy appears in arena when `StrategyRegistry.list_strategies()` is called
