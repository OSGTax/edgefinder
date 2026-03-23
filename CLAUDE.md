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
- Composite score threshold: 60/100 to make watchlist
- Lynch weight: 50%, Burry weight: 50%
