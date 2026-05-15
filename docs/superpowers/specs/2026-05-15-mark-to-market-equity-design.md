# Mark-to-Market Equity

## Problem

`VirtualAccount.total_equity` uses cost basis (`shares x entry_price`) instead of current market value. This means:

- The drawdown circuit breaker doesn't see unrealized losses and triggers late or never.
- Position sizing is based on inflated equity during losing streaks.
- Strategy comparison on the dashboard shows fictional equity values.
- The system records strategies surviving drawdowns they wouldn't survive in real trading.

## Constraint

All price data is 15-minute delayed (Massive Stocks Starter plan, $29/mo). Market prices are fetched via REST API during the existing 5-minute position monitor cycle. No additional API calls are introduced.

## Design

### Position: new `market_price` field

Add `market_price: float | None = None` to the `Position` dataclass.

`market_value` property uses the best available price:
- `market_price` if set (updated every 5 min by position monitor)
- `entry_price` as last resort if `market_price` has never been fetched

A stale `market_price` from a previous cycle is never cleared. A single failed price fetch does not regress to entry price — the previous market price persists.

`cost_basis` remains unchanged. It is used for cash accounting (`_recalculate_account_balances`) and P&L calculations, which are transaction-based and unrelated to current market value.

### VirtualAccount: equity from market values

`total_equity` = `cash + sum(position.market_value for all positions)`

`open_positions_value` = `sum(position.market_value for all positions)`

Everything downstream that uses `total_equity` automatically gets correct values:
- `drawdown_pct` reflects real unrealized losses
- `_size_position()` in the executor uses real equity for risk math
- Concentration checks use real equity

No changes to `cash`, `buying_power`, `open_position()`, or `close_position()`. These are cash-flow operations based on actual transaction prices.

### Price updates: when and where

**Position monitor** (`arena.py:check_positions`): Already fetches prices for every open position to check stops/targets. After checking, write the fetched price onto `pos.market_price`. One-line addition.

**Startup** (`services.py`): After `_restore_open_positions()`, fetch prices once for all restored positions. If a price fetch fails, log a warning — `market_price` stays `None` and `market_value` falls back to entry price until the first monitor cycle.

**New position open** (`arena.py:run_signal_check`): Set `market_price` on the new position to `fresh_price` or `execution_price` so it has a market value immediately.

### Dashboard and persistence

`account.to_dict()` already serializes `total_equity` and `open_positions_value`. No router changes needed — the dashboard gets correct numbers automatically.

`_snapshot_job()` records equity curves using `total_equity`. Automatically uses mark-to-market now.

`_recalculate_account_balances()` is unchanged. It validates cash accounting from the trades table using `entry_price * shares` for open positions. This is correct — cash was deducted at entry price, not market price.

`_persist_account_state()` writes market-value-based `total_equity` to DB. On restart, `_restore_account_state()` loads it back, but `_recalculate_account_balances()` immediately corrects cash from trades. The market-value fields are informational between restarts.

## Files changed

| File | Change |
|------|--------|
| `edgefinder/trading/account.py` | Add `market_price` to Position, add `market_value` property, change `total_equity` and `open_positions_value` to use `market_value` |
| `edgefinder/trading/arena.py` | Write `market_price` onto positions in `check_positions()`, set `market_price` on new positions in `run_signal_check()` |
| `dashboard/services.py` | Fetch prices for restored positions after `_restore_open_positions()` |
| `tests/` | New tests for mark-to-market behavior |

## Tests

1. `Position.market_value` returns `shares x market_price` when set, `shares x entry_price` when `None`.
2. `VirtualAccount.total_equity` reflects unrealized gains (position up 10% = equity increases).
3. `VirtualAccount.total_equity` reflects unrealized losses (position down 10% = equity decreases).
4. `VirtualAccount.drawdown_pct` triggers on mark-to-market losses, not just realized.
5. Position monitor writes `market_price` onto positions after price fetch.
6. New positions get `market_price` set at open time.
7. Failed price fetch preserves previous `market_price` (does not revert to `None`).
