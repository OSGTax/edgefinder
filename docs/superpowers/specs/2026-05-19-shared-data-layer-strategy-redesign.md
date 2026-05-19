# Shared Data Layer & Strategy Redesign

## Problem

The current architecture has every strategy independently calling the same signal engine, filtering pre-baked signal patterns by name. All strategies see the same 5-minute bars, compute the same indicators, and differ only in which pattern names they accept. This produces nearly identical trades across strategies and generates results that don't teach anything about which approach actually works.

Additionally:
- 5-minute bars are meaningless for swing strategies (EMA 50/200 on 5-min bars ≠ daily EMA 50/200)
- The risk system is too conservative for a $5,000 account (2% risk = $100 max loss per trade)
- Strategies don't have access to raw data — they can only pick from a menu of pre-detected signals
- No rich logging of trade reasoning, making it impossible to learn from results

## Solution Overview

Replace the entire strategy layer with:
1. A shared data layer that computes indicators once on daily bars and makes all data available to all strategies
2. A new strategy interface where strategies write their own entry/exit logic against raw data
3. A revised risk system with per-strategy risk budgets and fixed percentage stops
4. Three new strategies (Coward, Gambler, Degenerate) replacing all existing ones
5. Rich trade logging capturing reasoning, indicator state, and market context at every decision point

## Shared Data Layer

### Data available to every strategy

**Technical indicators (current + 30-day history):**
EMA 9/21/50/200, RSI 14, MACD (line/signal/histogram), Bollinger Bands (upper/middle/lower/width), ADX, ATR, Stochastic RSI, Williams %R, volume ratio (time-normalized)

**Fundamentals (from DB, refreshed nightly):**
Earnings growth, revenue growth, P/E, P/B, D/E, current ratio, FCF yield, ROE, ROA, short interest, dividend yield, market cap, sector

**Market context (shared across all tickers):**
SPY/QQQ/IWM/DIA prices and daily changes, VIX proxy level, market regime (bull/bear/sideways), sector rotation data

### Two computation cycles

**Daily cycle (6:15 PM ET, after market close):**
- Fetch actual daily bars for all watchlist tickers
- Compute all indicators on full daily bar history
- Save today's indicator snapshot as a permanent entry in the 30-day history buffer
- Refresh fundamentals via nightly scan (500-1000 tickers, unlimited API calls, completes well before market open)

**Intraday cycle (every 5 minutes, 9:30 AM - 4:00 PM ET):**
- Call `get_all_snapshots()` — one API call returns price + today's running volume for all stocks
- For each watchlist ticker:
  - Append current price as provisional close to 30-day daily closes
  - Recompute all indicators on the 31-bar series (30 real + 1 provisional)
  - Compute time-normalized volume ratio: `(today_volume / avg_daily_volume) / (minutes_since_open / 390)`
  - Package into a MarketData object: current indicators, 30-day history, fundamentals, market context
- Provisional computations are never saved to the history buffer — discarded and recomputed each cycle

### Startup safeguard

If the server starts within 2 hours of market open, skip the initial scan and use existing DB watchlists from the last nightly scan. Only run the initial scan if starting well before or after market hours. This prevents the scan from bleeding into trading time.

## Risk System

### Position sizing

Each strategy defines its own risk percentage — the max percentage of equity it will lose on a single trade:
- Coward: 5% of equity per trade
- Gambler: 10% of equity per trade
- Degenerate: 20% of equity per trade

Position size formula: `shares = max_loss / stop_distance` where `max_loss = equity * risk_pct` and `stop_distance = entry_price * 0.20`.

### Stop loss

Fixed at 20% below entry price for all strategies. Non-negotiable — always active, fires regardless of strategy opinion. This is the universal safety net.

### Profit targets

Per-strategy fixed percentage above entry price:
- Coward: 15%
- Gambler: 25%
- Degenerate: 50%

### Position limits

No hard limit on number of positions. Cash is the natural constraint. The math creates natural limits:
- Coward at 5% risk: can hold ~4 positions
- Gambler at 10% risk: can hold ~2 positions
- Degenerate at 20% risk: effectively 1 position at a time

### Exit priority

Checked every 5 minutes, whichever fires first:
1. 20% stop loss (always, non-negotiable)
2. Profit target hit
3. Strategy exit condition (strategy-defined logic based on MarketData)

### PDT tracking

Every same-day close is flagged as a day trade in the trade journal. Logged for awareness — does not prevent the stop from firing. Allows post-analysis of "how often would this strategy trigger PDT on a real account."

## Strategy Interface

Each strategy has two responsibilities:

**1. Entry evaluation:** Receives a MarketData object for a ticker. Returns a trade intent (ticker, direction, reasoning) or nothing. The risk system handles sizing, stop placement, target setting, and account checks.

**2. Exit evaluation:** For each open position, receives MarketData. Returns an exit intent (reasoning) or nothing. The 20% stop and profit target are checked separately by the risk system — the strategy only defines supplementary exit conditions.

### Watchlist filters

Each strategy defines fundamental criteria that determine which tickers it watches. Applied during the nightly scan. Tickers that don't pass never appear on that strategy's watchlist.

## Three Strategies

### Coward — conservative swing trading

**Risk:** 5% of equity per trade
**Stop:** 20% below entry
**Profit target:** 15%

**Watchlist filter:**
- Positive earnings growth
- Current ratio above 1.5

**Entry (either condition, not both required):**
- RSI below 35
- Price within 1% of BB lower band

**Exit:**
- RSI crosses above 70
- (Plus 20% stop and 15% target always active)

**Philosophy:** Watches quality stocks, enters when they dip, leaves at the first sign of a top. Wins often, wins small.

### Gambler — balanced swing trading

**Risk:** 10% of equity per trade
**Stop:** 20% below entry
**Profit target:** 25%

**Watchlist filter:**
- Positive earnings growth OR positive revenue growth (only needs one)

**Entry (both required):**
- MACD histogram crosses from negative to positive (momentum shifting bullish)
- RSI between 40 and 60 (not overbought, not oversold — mid-trend)

Cross detection: compare today's provisional MACD histogram to yesterday's actual value from the 30-day history buffer.

**Exit:**
- MACD histogram crosses from positive to negative (momentum fading)
- (Plus 20% stop and 25% target always active)

**Philosophy:** Rides momentum in the middle of moves. Doesn't try to catch bottoms or tops. More trades, bigger positions, holds for bigger targets.

### Degenerate — aggressive swing trading

**Risk:** 20% of equity per trade
**Stop:** 20% below entry
**Profit target:** 50%

**Watchlist filter:**
- Any stock with sufficient data (no fundamental requirements)
- Capped at top 200 by dollar volume from nightly scan
- PLUS dynamic additions: any stock showing 3x+ time-normalized volume in the `get_all_snapshots()` data gets temporarily added

**Entry (both required):**
- Time-normalized volume ratio above 2x average
- RSI above 50 AND price above EMA 21 (bullish momentum)

**Exit (both required to trigger):**
- Volume ratio drops below 1x average
- AND RSI crosses above 80 (overbought + hype dying)
- (Plus 20% stop and 50% target always active)

**Philosophy:** Something big is happening, jump in. Ride it until the music stops. One position at a time, lives or dies on single trades.

## Trade Logging

Every trade captures rich context for post-analysis:

**At entry:**
- Ticker, direction, shares, entry price, stop, target
- Strategy name and reasoning (the conditions that triggered entry)
- Full indicator snapshot (all indicators at entry moment)
- Fundamentals snapshot (earnings, ratios, etc.)
- Market context (SPY, VIX, regime, sector rotation)

**At exit:**
- Exit price, P&L (dollars and percent), R-multiple
- Exit reason (stop/target/strategy signal — which one fired)
- Full indicator snapshot at exit
- Hold duration
- PDT flag (was this a same-day close)

## Universe & Watchlists

**Nightly scan:** 500-1000 tickers (top by dollar volume). Runs at 6:15 PM ET, completes well before midnight. Fetches fundamentals (2 API calls per ticker, unlimited on Starter plan).

**Per-strategy watchlist sizes:**
- Coward: top 50 by qualification score
- Gambler: top 100 by qualification score
- Degenerate: top 200 by dollar volume + dynamic volume anomaly additions

**Dynamic volume screening (Degenerate only):** Each intraday cycle, the `get_all_snapshots()` data is scanned for stocks showing 3x+ time-normalized volume that aren't already on any watchlist. These are temporarily added to Degenerate's watchlist for that cycle.

## Old Strategies — Removal

Alpha, Bravo, Charlie, old Degenerate, and Echo are all removed. Their strategy files are deleted. The strategy registry is cleared and only the three new strategies are registered. Existing trade history in the DB is preserved (different strategy names, no conflict).

## Files Changed

### New files
- `edgefinder/data/market_data.py` — MarketData object, indicator history buffer
- `edgefinder/data/snapshot_provider.py` — enriched `get_all_snapshots()` returning price + volume
- `edgefinder/strategies/coward.py` — Coward strategy
- `edgefinder/strategies/gambler.py` — Gambler strategy
- `edgefinder/strategies/degenerate_v2.py` — new Degenerate strategy
- `edgefinder/strategies/strategy_interface.py` — new base class (evaluate + should_exit)
- `edgefinder/trading/risk.py` — centralized risk system (sizing, stops, targets)

### Modified files
- `edgefinder/trading/arena.py` — new intraday cycle using shared data layer
- `edgefinder/trading/executor.py` — accept trade intents instead of signals
- `edgefinder/signals/engine.py` — refactored as shared indicator computation (no signal detection)
- `edgefinder/trading/account.py` — remove hard position limit, update PDT tracking
- `dashboard/services.py` — wire new strategies, new scan logic, startup safeguard
- `edgefinder/scheduler/scheduler.py` — add daily indicator cycle, restrict intraday to market hours
- `config/settings.py` — new risk parameters, remove old signal settings
- `edgefinder/trading/journal.py` — rich trade context logging
- `edgefinder/strategies/__init__.py` — register new strategies only

### Removed files
- `edgefinder/strategies/alpha.py`
- `edgefinder/strategies/bravo.py`
- `edgefinder/strategies/charlie.py`
- `edgefinder/strategies/echo.py`
- `edgefinder/strategies/degenerate.py` (replaced by degenerate_v2.py)
