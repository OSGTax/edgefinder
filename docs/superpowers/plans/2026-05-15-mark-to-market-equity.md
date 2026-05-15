# Mark-to-Market Equity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch equity calculations from cost basis to mark-to-market so drawdown breakers, position sizing, and dashboard values reflect real unrealized P&L.

**Architecture:** Add `market_price` field to `Position`, update `VirtualAccount.total_equity` and `open_positions_value` to use market prices, write market prices from the position monitor cycle and at startup.

**Tech Stack:** Python, pytest, existing `VirtualAccount`/`Position`/`ArenaEngine` classes.

---

### Task 1: Add `market_price` to Position and `market_value` property

**Files:**
- Modify: `edgefinder/trading/account.py:22-57`
- Test: `tests/test_trading.py`

- [ ] **Step 1: Write failing tests for `market_value`**

Add to `TestPosition` class in `tests/test_trading.py`:

```python
def test_market_value_with_market_price(self):
    pos = Position(
        symbol="AAPL", shares=10, entry_price=100.0,
        stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
    )
    pos.market_price = 110.0
    assert pos.market_value == 1100.0

def test_market_value_falls_back_to_entry_price(self):
    pos = Position(
        symbol="AAPL", shares=10, entry_price=100.0,
        stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
    )
    # market_price is None by default
    assert pos.market_value == 1000.0  # falls back to entry_price

def test_market_value_preserves_stale_price(self):
    pos = Position(
        symbol="AAPL", shares=10, entry_price=100.0,
        stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
    )
    pos.market_price = 105.0
    # Simulate a failed price fetch — market_price stays at 105
    # (we never clear it)
    assert pos.market_value == 1050.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading.py::TestPosition::test_market_value_with_market_price tests/test_trading.py::TestPosition::test_market_value_falls_back_to_entry_price tests/test_trading.py::TestPosition::test_market_value_preserves_stale_price -v`

Expected: FAIL — `Position` has no attribute `market_price` or `market_value`

- [ ] **Step 3: Add `market_price` field and `market_value` property to Position**

In `edgefinder/trading/account.py`, add to the `Position` dataclass:

```python
market_price: float | None = None  # latest price, updated by position monitor

@property
def market_value(self) -> float:
    """Current market value using best available price.
    Priority: market_price (updated every 5 min) > entry_price (last resort).
    """
    price = self.market_price if self.market_price is not None else self.entry_price
    return self.shares * price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trading.py::TestPosition -v`

Expected: All Position tests PASS (including existing ones)

- [ ] **Step 5: Commit**

```bash
git add edgefinder/trading/account.py tests/test_trading.py
git commit -m "feat: add market_price and market_value to Position"
```

---

### Task 2: Switch `VirtualAccount.total_equity` and `open_positions_value` to market value

**Files:**
- Modify: `edgefinder/trading/account.py:96-109`
- Modify: `tests/test_trading.py`

- [ ] **Step 1: Write failing tests for mark-to-market equity**

Add a new test class in `tests/test_trading.py`:

```python
class TestMarkToMarketEquity:
    def test_total_equity_reflects_unrealized_gain(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)  # cash = 4000, cost = 1000
        pos.market_price = 110.0  # position now worth 1100
        assert acct.total_equity == 5100.0  # 4000 + 1100

    def test_total_equity_reflects_unrealized_loss(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)  # cash = 4000
        pos.market_price = 90.0  # position now worth 900
        assert acct.total_equity == 4900.0  # 4000 + 900

    def test_open_positions_value_uses_market_price(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)
        pos.market_price = 115.0
        assert acct.open_positions_value == 1150.0

    def test_drawdown_triggers_on_unrealized_loss(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        acct.peak_equity = 5000.0
        pos = Position(
            symbol="AAPL", shares=40, entry_price=100.0,
            stop_loss=70.0, target=130.0, direction="LONG", trade_type="SWING",
        )
        acct.open_position(pos)  # cash = 1000, cost = 4000
        # Position drops 30%: 40 * 70 = 2800
        pos.market_price = 70.0
        # Equity = 1000 + 2800 = 3800. Drawdown = (5000-3800)/5000 = 24%
        assert acct.drawdown_pct == pytest.approx(0.24, abs=0.01)
        allowed, reason = acct.can_open_position(100.0)
        assert allowed is False
        assert "circuit breaker" in reason.lower()

    def test_equity_without_market_price_uses_entry(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)
        # market_price is None — falls back to entry_price
        assert acct.total_equity == 5000.0  # 4000 + 1000 (cost basis)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_trading.py::TestMarkToMarketEquity -v`

Expected: FAIL — `total_equity` still uses `cost_basis` so gain/loss tests fail

- [ ] **Step 3: Update `total_equity` and `open_positions_value`**

In `edgefinder/trading/account.py`, change the two properties:

```python
@property
def open_positions_value(self) -> float:
    """Mark-to-market value of all open positions."""
    return sum(p.market_value for p in self.positions)

@property
def total_equity(self) -> float:
    """Cash + mark-to-market value of open positions."""
    return self.cash + self.open_positions_value
```

- [ ] **Step 4: Fix the existing test that asserts cost-basis equity**

In `tests/test_trading.py`, update `TestAccountFinancialIntegrity::test_multi_position_cash_tracking`. After closing AAPL, MSFT is still open with no `market_price`, so `market_value` falls back to `entry_price` (cost basis). The assertions stay the same:

```python
# Line 444 — MSFT has no market_price, falls back to entry_price * shares
assert acct.open_positions_value == 1000.0
# Line 446 — cash 4050 + MSFT market_value 1000 = 5050
assert acct.total_equity == 5050.0
```

No change needed — the fallback to `entry_price` preserves the existing behavior when `market_price` is `None`.

- [ ] **Step 5: Run ALL trading tests**

Run: `pytest tests/test_trading.py -v`

Expected: ALL tests PASS (32 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add edgefinder/trading/account.py tests/test_trading.py
git commit -m "feat: switch total_equity to mark-to-market valuation"
```

---

### Task 3: Position monitor writes `market_price` onto positions

**Files:**
- Modify: `edgefinder/trading/arena.py:197-264`
- Test: `tests/test_trading.py`

- [ ] **Step 1: Write failing test**

Add to `TestArena` class in `tests/test_trading.py`:

```python
def test_check_positions_updates_market_price(self, mock_provider):
    import importlib
    from edgefinder.strategies import alpha, bravo, charlie
    from edgefinder.strategies.base import StrategyRegistry
    StrategyRegistry.clear()
    importlib.reload(alpha)
    importlib.reload(bravo)
    importlib.reload(charlie)

    arena = ArenaEngine(mock_provider)
    arena.load_strategies()

    # Manually add a position to alpha's account
    acct = arena.get_account("alpha")
    pos = Position(
        symbol="AAPL", shares=10, entry_price=100.0,
        stop_loss=50.0, target=200.0, direction="LONG", trade_type="SWING",
        trade_id="test-mtm-1",
    )
    acct.open_position(pos)

    # Position monitor checks prices — mock returns 105.0
    mock_provider.get_latest_price.return_value = 105.0
    arena.check_positions()

    # market_price should now be set
    assert pos.market_price == 105.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading.py::TestArena::test_check_positions_updates_market_price -v`

Expected: FAIL — `pos.market_price` is still `None`

- [ ] **Step 3: Write market prices in `check_positions`**

In `edgefinder/trading/arena.py`, in the `check_positions` method, after fetching prices for each position (around line 218), add a line to write the price onto the position:

```python
# Get current prices for all symbols with open positions
prices: dict[str, float] = {}
for pos in slot.account.positions:
    price = self._provider.get_latest_price(pos.symbol)
    if price is not None:
        prices[pos.symbol] = price
        pos.market_price = price  # <-- ADD THIS LINE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading.py::TestArena::test_check_positions_updates_market_price -v`

Expected: PASS

- [ ] **Step 5: Run all trading tests**

Run: `pytest tests/test_trading.py -v`

Expected: ALL tests PASS

- [ ] **Step 6: Commit**

```bash
git add edgefinder/trading/arena.py tests/test_trading.py
git commit -m "feat: position monitor writes market_price onto positions"
```

---

### Task 4: New positions get `market_price` at open time

**Files:**
- Modify: `edgefinder/trading/arena.py:88-195`
- Test: `tests/test_trading.py`

- [ ] **Step 1: Write failing test**

Add to `TestArena` class in `tests/test_trading.py`:

```python
def test_new_position_gets_market_price(self, mock_provider):
    import importlib
    from edgefinder.strategies import alpha, bravo, charlie
    from edgefinder.strategies.base import StrategyRegistry
    StrategyRegistry.clear()
    importlib.reload(alpha)
    importlib.reload(bravo)
    importlib.reload(charlie)

    arena = ArenaEngine(mock_provider)
    arena.load_strategies()
    arena.set_watchlists({"alpha": ["AAPL"]})

    # Run signal check — if a trade opens, the position should have market_price
    mock_provider.get_latest_price.return_value = 100.0
    trades = arena.run_signal_check()

    if trades:
        # Check the position in the account
        acct = arena.get_account(trades[0].strategy_name)
        pos = acct.get_position("AAPL")
        assert pos is not None
        assert pos.market_price is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trading.py::TestArena::test_new_position_gets_market_price -v`

Expected: FAIL — `pos.market_price` is `None`

- [ ] **Step 3: Set `market_price` after opening position**

In `edgefinder/trading/arena.py`, in `run_signal_check`, after the position is opened via `slot.executor.execute_signal()` (around line 176), set the market price on the newly created position:

```python
for signal in signals:
    trade = slot.executor.execute_signal(signal, fresh_price=fresh_price)
    if trade:
        all_trades.append(trade)
        opened_here += 1
        # Set market price on the new position
        new_pos = slot.account.get_position(signal.ticker)
        if new_pos:
            new_pos.market_price = fresh_price or trade.entry_price
        slot.strategy.on_trade_executed(
            TradeNotification(trade=trade, event="opened")
        )
        break  # one position per ticker per signal check
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trading.py::TestArena::test_new_position_gets_market_price -v`

Expected: PASS

- [ ] **Step 5: Run all trading tests**

Run: `pytest tests/test_trading.py -v`

Expected: ALL tests PASS

- [ ] **Step 6: Commit**

```bash
git add edgefinder/trading/arena.py tests/test_trading.py
git commit -m "feat: set market_price on new positions at open time"
```

---

### Task 5: Fetch market prices for restored positions at startup

**Files:**
- Modify: `dashboard/services.py:263-310`
- Test: `tests/test_trading.py`

- [ ] **Step 1: Write failing test**

Add a new test class in `tests/test_trading.py`:

```python
class TestStartupPriceFetch:
    def test_restored_positions_get_market_price(self):
        """Simulates what services.py should do after restoring positions."""
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="SWING",
            trade_id="restore-001",
        )
        # Simulate restore: add position without deducting cash
        # (mirrors _restore_open_positions behavior)
        acct.positions.append(pos)
        acct.cash = 4000.0  # already reflects the open position

        # Before price fetch — falls back to entry
        assert pos.market_price is None
        assert acct.total_equity == 5000.0  # 4000 + 1000 (entry fallback)

        # Simulate startup price fetch
        mock_price = 112.0
        pos.market_price = mock_price

        # After price fetch — uses real market value
        assert acct.total_equity == 5120.0  # 4000 + 1120
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_trading.py::TestStartupPriceFetch -v`

Expected: PASS (this test validates the behavior our Position/Account changes already support)

- [ ] **Step 3: Add price fetch to `_restore_open_positions` in services.py**

In `dashboard/services.py`, at the end of `_restore_open_positions()`, after restoring all positions, add a price fetch loop:

```python
        if restored:
            logger.info("Restored %d open positions from DB", restored)
            # Fetch market prices for restored positions
            _fetch_startup_prices()
    except Exception:
        logger.exception("Failed to restore open positions")
    finally:
        session.close()
```

Add the new function after `_restore_open_positions`:

```python
def _fetch_startup_prices() -> None:
    """Fetch current market prices for all open positions at startup.

    Ensures mark-to-market equity is correct before the first
    position monitor cycle runs.
    """
    if not _arena or not _provider:
        return
    fetched = 0
    failed = 0
    for name in _arena.get_strategy_names():
        account = _arena.get_account(name)
        if not account:
            continue
        for pos in account.positions:
            price = _provider.get_latest_price(pos.symbol)
            if price is not None:
                pos.market_price = price
                fetched += 1
            else:
                logger.warning(
                    "Startup: no price for %s — using entry price $%.2f until next monitor cycle",
                    pos.symbol, pos.entry_price,
                )
                failed += 1
    if fetched or failed:
        logger.info("Startup price fetch: %d fetched, %d failed", fetched, failed)
```

- [ ] **Step 4: Run all trading tests**

Run: `pytest tests/test_trading.py -v`

Expected: ALL tests PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/services.py tests/test_trading.py
git commit -m "feat: fetch market prices for restored positions at startup"
```

---

### Task 6: Final integration test and full test suite

**Files:**
- Test: `tests/test_trading.py`

- [ ] **Step 1: Write an end-to-end mark-to-market integration test**

Add to `TestMarkToMarketEquity` in `tests/test_trading.py`:

```python
def test_full_lifecycle_mark_to_market(self):
    """Open position, market moves, check equity, close position."""
    acct = VirtualAccount("alpha", starting_capital=5000.0)
    executor = Executor(acct)

    # Open a position
    signal = Signal(
        ticker="AAPL", action=SignalAction.BUY,
        entry_price=100.0, stop_loss=95.0, target=110.0,
        confidence=70.0, trade_type=TradeType.DAY, strategy_name="alpha",
    )
    trade = executor.execute_signal(signal)
    assert trade is not None
    shares = trade.shares
    entry = trade.entry_price  # includes slippage

    # Simulate position monitor updating market price — stock goes up
    pos = acct.get_position("AAPL")
    pos.market_price = 108.0
    equity_up = acct.total_equity
    assert equity_up > 5000.0  # unrealized gain reflected

    # Stock drops
    pos.market_price = 92.0
    equity_down = acct.total_equity
    assert equity_down < 5000.0  # unrealized loss reflected

    # Close the position
    result = acct.close_position(pos, 92.0, "STOP_HIT")
    # After close, equity is just cash (no positions)
    assert acct.position_count == 0
    assert acct.total_equity == acct.cash
    assert result["pnl_dollars"] < 0  # loss
```

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_trading.py::TestMarkToMarketEquity::test_full_lifecycle_mark_to_market -v`

Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v -m "not integration" --tb=short 2>&1 | tail -20`

Expected: ALL tests PASS — no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_trading.py
git commit -m "test: add mark-to-market integration test"
```

- [ ] **Step 5: Final version bump commit**

Update `__version__` in `dashboard/app.py` from `"4.6.3"` to `"4.7.0"` (mark-to-market is a meaningful behavior change).

```bash
git add dashboard/app.py
git commit -m "[v4.7.0] mark-to-market equity for virtual accounts"
```
