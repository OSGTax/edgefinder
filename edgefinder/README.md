# EdgeFinder — Intelligent Paper Trading System

A self-improving paper trading system that combines Peter Lynch and Michael Burry's
fundamental analysis with technical signals and news sentiment to simulate intelligent
stock trading. Built to learn from its own mistakes.

## Quick Start

```bash
# ============================================================
# HUMAN_ACTION_REQUIRED
# What: Create a GitHub repo and clone it
# Why: This is your project's home
# How: 1. Create a new repo on GitHub (private recommended)
#      2. Clone it and copy these files in, OR
#         init git in this directory and push
# ============================================================

# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/edgefinder.git
cd edgefinder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify everything works
python scripts/verify_install.py

# 4. Initialize the database
python scripts/setup_db.py

# 5. Run Module 1 tests (should be 63 passed)
python -m pytest tests/test_scanner.py -v

# 6. Try a live scan
python scripts/run_scanner.py --quick
```

## Using with Claude Code

This project is designed to work with Claude Code. Here's how:

### First Session — Verify Module 1
```
Open a terminal in the edgefinder/ directory, then run Claude Code.

Tell Claude Code:
> Read CLAUDE.md, then run the Module 1 tests and fix any failures.
> After tests pass, run a quick scan and show me the results.
```

### Building New Modules
```
Tell Claude Code:
> Read CLAUDE.md. Create a branch called module-2, then build
> Module 2 (Technical Signal Engine). Write tests first in
> tests/test_signals.py, then implement modules/signals.py.
> Run all tests before committing. When done, push the branch.
```

### Weekly Strategy Reviews (once trading is active)
```
Tell Claude Code:
> Read CLAUDE.md, then analyze the trade journal for the past week.
> What setups are winning? What's losing? Draft parameter adjustments.
> Commit changes to a branch called optimizer-adjustments.
```

## Git Workflow

Each module is built on its own branch and merged to `main` only after all tests pass.

```
main ← stable code only
├── module-2       ← technical signals
├── module-2.5     ← news sentiment
├── module-3-4     ← trader + journal
├── module-5       ← optimizer
└── dashboard      ← web UI
```

Always run tests before committing: `python -m pytest tests/ -v -m "not integration"`

## Build Order

Each module must pass its tests before moving to the next.

| Phase | Module | Test Command | Status |
|-------|--------|-------------|--------|
| 1 | Fundamental Scanner | `python -m pytest tests/test_scanner.py -v` | ✅ Built |
| 2 | Technical Signal Engine | `python -m pytest tests/test_signals.py -v` | 🔲 Next |
| 3 | News Sentiment Gate | `python -m pytest tests/test_sentiment.py -v` | 🔲 Queued |
| 4 | Paper Trader + Journal | `python -m pytest tests/test_trader.py -v` | 🔲 Queued |
| 5 | Web Dashboard | Manual testing | 🔲 Queued |
| 6 | Strategy Optimizer | `python -m pytest tests/test_optimizer.py -v` | 🔲 Queued |
| 7 | ML Graduation | TBD | 🔲 Future |

## Running Scans

```bash
# Quick scan (50 popular tickers, ~2 minutes)
python scripts/run_scanner.py --quick

# Full scan (S&P 500/400/600, ~15-30 minutes)
python scripts/run_scanner.py

# Specific tickers
python scripts/run_scanner.py --tickers AAPL MSFT GOOGL JPM

# Without saving to database
python scripts/run_scanner.py --quick --no-save
```

## Running Tests

```bash
# All unit tests
python scripts/run_tests.py

# Specific module
python scripts/run_tests.py --module scanner

# Include integration tests (hits real APIs)
python scripts/run_tests.py --integration

# With coverage
python scripts/run_tests.py --coverage
```

## Key Configuration

All tunable parameters live in `config/settings.py`. Never hardcode values in modules.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Starting Capital | $2,500 | Simulated account balance |
| Max Risk/Trade | 2% | Maximum loss per trade |
| Watchlist Threshold | 60/100 | Minimum composite score |
| Lynch Weight | 50% | Weight in composite score |
| Burry Weight | 50% | Weight in composite score |
| PDT Limit | 3 day trades / 5 days | Pattern Day Trader compliance |

## Architecture

```
                    ┌──────────────────┐
                    │  Ticker Universe │
                    │ S&P 500/400/600  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Module 1: Scanner │  ← Nightly
                    │ Lynch + Burry     │
                    │ fundamentals      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Watchlist (DB)    │  50-100 candidates
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Module 2: Signals │  ← Every 15 min
                    │ EMA, RSI, MACD    │
                    │ Volume, S/R       │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Module 2.5: News  │  ← Before each trade
                    │ Sentiment gate    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Module 3: Trader  │  ← When signals fire
                    │ Position sizing   │
                    │ Risk management   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Module 4: Journal │  ← Per trade
                    │ Full context log  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Module 5: Optim.  │  ← Weekly
                    │ Analyze & adjust  │
                    └──────────────────┘
```

## Disclaimer

EdgeFinder is a paper trading and learning system. All trades are simulated.
This system is for educational purposes only and should not be relied upon as
the sole basis for real trading decisions. Always do your own research and
consider consulting a financial advisor before trading with real money.
