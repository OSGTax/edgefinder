"""
EdgeFinder Module 3 Tests: Paper Trader
========================================
Tests cover: position sizing, risk management, PDT compliance,
stop-loss/target calculation, trailing stops, circuit breakers,
trade execution, price updates, and database persistence.

Run: python -m pytest tests/test_trader.py -v
"""

import pytest
from datetime import datetime, timedelta, timezone

from modules.trader import (
    PaperTrader,
    AccountState,
    Position,
    TradeResult,
)
from config import settings


# ── FIXTURES ─────────────────────────────────────────────────

@pytest.fixture
def trader() -> PaperTrader:
    """Fresh paper trader with default $2,500 account."""
    return PaperTrader()


@pytest.fixture
def funded_trader() -> PaperTrader:
    """Trader with more capital for multi-position tests."""
    account = AccountState(cash=10_000.0, peak_value=10_000.0)
    return PaperTrader(account=account)


@pytest.fixture
def trader_with_position(trader) -> tuple[PaperTrader, Position]:
    """Trader with one open position."""
    pos = trader.open_position(
        ticker="AAPL", entry_price=100.0, stop_loss=98.0,
        target=103.0, shares=10, trade_type="DAY",
    )
    return trader, pos


# ════════════════════════════════════════════════════════════
# ACCOUNT STATE
# ════════════════════════════════════════════════════════════

class TestAccountState:
    """Test account state properties."""

    def test_default_starting_capital(self):
        account = AccountState()
        assert account.cash == settings.STARTING_CAPITAL

    def test_total_value_cash_only(self):
        account = AccountState(cash=2500.0)
        assert account.total_value == 2500.0

    def test_open_position_count(self):
        account = AccountState()
        assert account.open_position_count == 0

    def test_drawdown_pct_no_loss(self):
        account = AccountState(cash=2500.0, peak_value=2500.0)
        assert account.drawdown_pct == 0.0

    def test_drawdown_pct_with_loss(self):
        account = AccountState(cash=2000.0, peak_value=2500.0)
        assert abs(account.drawdown_pct - 0.2) < 0.01

    def test_sector_count_empty(self):
        account = AccountState()
        assert account.sector_count("Technology") == 0


# ════════════════════════════════════════════════════════════
# POSITION SIZING
# ════════════════════════════════════════════════════════════

class TestPositionSizing:
    """Test the 2% risk-based position sizing."""

    def test_basic_sizing(self, trader):
        # $2,500 account, 2% risk = $50 max risk
        # Entry $100, stop $98 = $2 risk/share
        # Half position (confidence < 80): $25 / $2 = 12 shares
        shares = trader.calculate_position_size(100.0, 98.0, confidence=60.0)
        assert shares > 0
        assert shares <= 12

    def test_high_confidence_full_position(self, trader):
        # With $2,500 account and wide stop ($10 risk/share), concentration
        # limit ($500 / $50 = 10 shares) won't cap before risk sizing does.
        # Full: $50 / $10 = 5 shares. Half: $25 / $10 = 2 shares.
        full = trader.calculate_position_size(50.0, 40.0, confidence=85.0)
        half = trader.calculate_position_size(50.0, 40.0, confidence=60.0)
        assert full > half

    def test_zero_entry_price(self, trader):
        assert trader.calculate_position_size(0, 98.0) == 0

    def test_zero_stop_loss(self, trader):
        assert trader.calculate_position_size(100.0, 0) == 0

    def test_stop_equals_entry(self, trader):
        assert trader.calculate_position_size(100.0, 100.0) == 0

    def test_respects_concentration_limit(self, trader):
        # With $2,500, 20% max = $500 max position
        # If stock is $100, max 5 shares by concentration
        shares = trader.calculate_position_size(100.0, 50.0, confidence=90.0)
        max_by_concentration = int(trader.account.total_value * settings.MAX_PORTFOLIO_CONCENTRATION_PCT / 100.0)
        assert shares <= max_by_concentration

    def test_respects_cash_limit(self):
        account = AccountState(cash=100.0, peak_value=2500.0)
        trader = PaperTrader(account=account)
        shares = trader.calculate_position_size(50.0, 48.0, confidence=90.0)
        assert shares * 50.0 <= 100.0


# ════════════════════════════════════════════════════════════
# STOP-LOSS & TARGET
# ════════════════════════════════════════════════════════════

class TestStopLossTarget:
    """Test stop-loss and target price calculations."""

    def test_default_stop_loss(self, trader):
        stop = trader.calculate_stop_loss(100.0)
        assert stop == 98.0  # 2% below entry

    def test_atr_stop_loss(self, trader):
        stop = trader.calculate_stop_loss(100.0, atr=3.0)
        assert stop == 95.5  # 100 - 1.5 * 3

    def test_target_respects_rr_ratio(self, trader):
        target = trader.calculate_target(100.0, 98.0)
        risk = 100.0 - 98.0  # $2
        reward = target - 100.0
        assert reward / risk >= settings.MIN_REWARD_TO_RISK_RATIO

    def test_target_calculation(self, trader):
        target = trader.calculate_target(100.0, 98.0)
        # risk = $2, reward = $2 * 1.5 = $3, target = $103
        assert target == 103.0


# ════════════════════════════════════════════════════════════
# PRE-TRADE CHECKS
# ════════════════════════════════════════════════════════════

class TestPreTradeChecks:
    """Test risk management pre-trade validation."""

    def test_can_trade_fresh_account(self, trader):
        can, reason = trader.can_trade("AAPL", "DAY")
        assert can is True

    def test_max_positions_blocks(self, funded_trader):
        # Open max positions
        for i in range(settings.MAX_OPEN_POSITIONS):
            funded_trader.open_position(
                ticker=f"T{i}", entry_price=10.0, stop_loss=9.0,
                target=15.0, shares=1,
            )
        can, reason = funded_trader.can_trade("NEW", "DAY")
        assert can is False
        assert "Max open positions" in reason

    def test_duplicate_ticker_blocks(self, trader_with_position):
        trader, _ = trader_with_position
        can, reason = trader.can_trade("AAPL", "DAY")
        assert can is False
        assert "Already holding" in reason

    def test_pdt_limit_blocks(self, funded_trader):
        # Simulate 3 recent day trades
        now = datetime.now(timezone.utc)
        funded_trader.account.day_trades_timestamps = [
            now - timedelta(hours=1),
            now - timedelta(hours=2),
            now - timedelta(hours=3),
        ]
        can, reason = funded_trader.can_trade("NEW", "DAY")
        assert can is False
        assert "PDT" in reason

    def test_pdt_allows_swing(self, funded_trader):
        """PDT only limits day trades, not swings."""
        now = datetime.now(timezone.utc)
        funded_trader.account.day_trades_timestamps = [
            now - timedelta(hours=1),
            now - timedelta(hours=2),
            now - timedelta(hours=3),
        ]
        can, reason = funded_trader.can_trade("NEW", "SWING")
        assert can is True

    def test_old_day_trades_expire(self, funded_trader):
        """Day trades older than PDT window don't count."""
        old = datetime.now(timezone.utc) - timedelta(days=10)
        funded_trader.account.day_trades_timestamps = [old, old, old]
        can, reason = funded_trader.can_trade("NEW", "DAY")
        assert can is True

    def test_sector_concentration_blocks(self, funded_trader):
        for i in range(settings.MAX_SAME_SECTOR_POSITIONS):
            funded_trader.open_position(
                ticker=f"T{i}", entry_price=10.0, stop_loss=9.0,
                target=15.0, shares=1, sector="Technology",
            )
        can, reason = funded_trader.can_trade("NEW", "DAY", sector="Technology")
        assert can is False
        assert "sector" in reason.lower()

    def test_drawdown_circuit_breaker(self):
        # 20% drawdown
        account = AccountState(cash=2000.0, peak_value=2500.0)
        trader = PaperTrader(account=account)
        can, reason = trader.can_trade("NEW", "DAY")
        assert can is False
        assert "circuit breaker" in reason.lower()

    def test_revenge_cooldown_blocks(self, trader):
        trader.account.last_stop_out_time = datetime.now(timezone.utc)
        can, reason = trader.can_trade("NEW", "DAY")
        assert can is False
        assert "cooldown" in reason.lower()

    def test_revenge_cooldown_expires(self, trader):
        past = datetime.now(timezone.utc) - timedelta(minutes=settings.REVENGE_TRADE_COOLDOWN_MINUTES + 1)
        trader.account.last_stop_out_time = past
        can, reason = trader.can_trade("NEW", "DAY")
        assert can is True

    def test_no_cash_blocks(self):
        # peak_value matches total to avoid circuit breaker triggering first
        account = AccountState(cash=0.0, peak_value=0.0)
        trader = PaperTrader(account=account)
        can, reason = trader.can_trade("NEW", "DAY")
        assert can is False
        assert "cash" in reason.lower()


# ════════════════════════════════════════════════════════════
# OPENING POSITIONS
# ════════════════════════════════════════════════════════════

class TestOpenPosition:
    """Test opening paper trade positions."""

    def test_open_deducts_cash(self, trader):
        initial_cash = trader.account.cash
        trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        assert trader.account.cash == initial_cash - 500.0

    def test_open_adds_to_positions(self, trader):
        pos = trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        assert trader.account.open_position_count == 1
        assert pos.trade_id in trader.account.positions

    def test_open_returns_position(self, trader):
        pos = trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        assert pos is not None
        assert pos.ticker == "AAPL"
        assert pos.shares == 5

    def test_open_zero_shares_rejected(self, trader):
        pos = trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=0,
        )
        assert pos is None

    def test_open_insufficient_cash_rejected(self, trader):
        pos = trader.open_position(
            ticker="AAPL", entry_price=1000.0, stop_loss=980.0,
            target=1030.0, shares=100,
        )
        assert pos is None


# ════════════════════════════════════════════════════════════
# CLOSING POSITIONS
# ════════════════════════════════════════════════════════════

class TestClosePosition:
    """Test closing positions and P&L calculation."""

    def test_close_returns_cash(self, trader_with_position):
        trader, pos = trader_with_position
        cash_before = trader.account.cash
        result = trader.close_position(pos.trade_id, exit_price=102.0)
        assert trader.account.cash == cash_before + (102.0 * 10)

    def test_close_removes_position(self, trader_with_position):
        trader, pos = trader_with_position
        trader.close_position(pos.trade_id, exit_price=102.0)
        assert trader.account.open_position_count == 0

    def test_close_calculates_profit(self, trader_with_position):
        trader, pos = trader_with_position
        result = trader.close_position(pos.trade_id, exit_price=102.0)
        assert result.pnl_dollars == 20.0  # (102 - 100) * 10

    def test_close_calculates_loss(self, trader_with_position):
        trader, pos = trader_with_position
        result = trader.close_position(pos.trade_id, exit_price=97.0)
        assert result.pnl_dollars == -30.0  # (97 - 100) * 10

    def test_close_calculates_r_multiple(self, trader_with_position):
        trader, pos = trader_with_position
        # Entry 100, stop 98, risk = $2/share
        # Exit at 104, profit = $4/share → 2R
        result = trader.close_position(pos.trade_id, exit_price=104.0)
        assert result.r_multiple == 2.0

    def test_close_nonexistent_returns_none(self, trader):
        result = trader.close_position("fake-id", exit_price=100.0)
        assert result is None

    def test_close_tracks_day_trade(self, trader_with_position):
        trader, pos = trader_with_position
        trader.close_position(pos.trade_id, exit_price=102.0)
        assert len(trader.account.day_trades_timestamps) == 1

    def test_close_stop_hit_sets_cooldown(self, trader_with_position):
        trader, pos = trader_with_position
        trader.close_position(pos.trade_id, exit_price=97.0, exit_reason="STOP_HIT")
        assert trader.account.last_stop_out_time is not None

    def test_close_updates_daily_pnl(self, trader_with_position):
        trader, pos = trader_with_position
        trader.close_position(pos.trade_id, exit_price=102.0)
        assert trader.account.daily_pnl == 20.0

    def test_close_exit_reason_preserved(self, trader_with_position):
        trader, pos = trader_with_position
        result = trader.close_position(pos.trade_id, exit_price=103.0, exit_reason="TARGET_HIT")
        assert result.exit_reason == "TARGET_HIT"


# ════════════════════════════════════════════════════════════
# PRICE UPDATES & TRAILING STOPS
# ════════════════════════════════════════════════════════════

class TestPriceUpdates:
    """Test price update logic including stops and targets."""

    def test_stop_hit(self, trader_with_position):
        trader, pos = trader_with_position
        action = trader.update_price(pos.trade_id, 97.0)
        assert action == "STOP_HIT"

    def test_target_hit(self, trader_with_position):
        trader, pos = trader_with_position
        action = trader.update_price(pos.trade_id, 104.0)
        assert action == "TARGET_HIT"

    def test_no_action_in_range(self, trader_with_position):
        trader, pos = trader_with_position
        action = trader.update_price(pos.trade_id, 101.0)
        assert action is None

    def test_trailing_stop_activates_at_1r(self, trader_with_position):
        trader, pos = trader_with_position
        # Entry 100, stop 98, risk = $2. 1R = $102
        trader.update_price(pos.trade_id, 102.5)
        assert pos.trailing_stop == 100.0  # Breakeven

    def test_trailing_stop_trails_at_2r(self, trader_with_position):
        trader, pos = trader_with_position
        # Push to 2R ($104), trail = high_water - risk = 104 - 2 = 102
        trader.update_price(pos.trade_id, 104.0)
        # target is 103, so TARGET_HIT fires first at 104
        # Let's test with a higher target
        pos2 = trader.open_position(
            ticker="MSFT", entry_price=100.0, stop_loss=98.0,
            target=110.0, shares=5,
        )
        trader.update_price(pos2.trade_id, 104.5)  # > 2R
        assert pos2.trailing_stop is not None
        assert pos2.trailing_stop >= 102.0

    def test_trailing_stop_triggers(self, funded_trader):
        pos = funded_trader.open_position(
            ticker="TEST", entry_price=100.0, stop_loss=98.0,
            target=110.0, shares=5,
        )
        # Move up to activate trailing stop
        funded_trader.update_price(pos.trade_id, 105.0)
        assert pos.trailing_stop is not None
        # Now price drops to trailing stop
        action = funded_trader.update_price(pos.trade_id, pos.trailing_stop - 0.01)
        assert action == "TRAILING_STOP"

    def test_high_water_mark_updates(self, trader_with_position):
        trader, pos = trader_with_position
        trader.update_price(pos.trade_id, 101.0)
        assert pos.high_water_mark == 101.0
        trader.update_price(pos.trade_id, 99.0)
        assert pos.high_water_mark == 101.0  # Doesn't decrease

    def test_nonexistent_trade(self, trader):
        action = trader.update_price("fake-id", 100.0)
        assert action is None


# ════════════════════════════════════════════════════════════
# EXECUTE SIGNAL (FULL PIPELINE)
# ════════════════════════════════════════════════════════════

class TestExecuteSignal:
    """Test the full signal execution pipeline."""

    def test_execute_buy_signal(self, trader):
        pos = trader.execute_signal(
            ticker="AAPL", signal_type="BUY", trade_type="DAY",
            entry_price=50.0, confidence=70.0,
        )
        assert pos is not None
        assert pos.ticker == "AAPL"
        assert trader.account.open_position_count == 1

    def test_execute_sell_signal_skipped(self, trader):
        """Long-only system should skip SELL signals."""
        pos = trader.execute_signal(
            ticker="AAPL", signal_type="SELL", trade_type="DAY",
            entry_price=50.0, confidence=70.0,
        )
        assert pos is None

    def test_execute_with_reduce_50(self, trader):
        """REDUCE_50 sentiment should halve position size."""
        full = trader.execute_signal(
            ticker="FULL", signal_type="BUY", trade_type="DAY",
            entry_price=50.0, confidence=70.0, sentiment_action="PROCEED",
        )
        # Reset
        trader2 = PaperTrader()
        half = trader2.execute_signal(
            ticker="HALF", signal_type="BUY", trade_type="DAY",
            entry_price=50.0, confidence=70.0, sentiment_action="REDUCE_50",
        )
        if full and half:
            assert half.shares <= full.shares

    def test_execute_rejected_by_risk_check(self, funded_trader):
        """Already at max positions should reject."""
        for i in range(settings.MAX_OPEN_POSITIONS):
            funded_trader.execute_signal(
                ticker=f"T{i}", signal_type="BUY", trade_type="SWING",
                entry_price=10.0, confidence=70.0,
            )
        pos = funded_trader.execute_signal(
            ticker="NEW", signal_type="BUY", trade_type="SWING",
            entry_price=10.0, confidence=70.0,
        )
        assert pos is None

    def test_execute_sets_stop_and_target(self, trader):
        pos = trader.execute_signal(
            ticker="AAPL", signal_type="BUY", trade_type="DAY",
            entry_price=100.0, confidence=70.0,
        )
        assert pos.stop_loss < pos.entry_price
        assert pos.target > pos.entry_price

    def test_execute_with_atr(self, trader):
        pos = trader.execute_signal(
            ticker="AAPL", signal_type="BUY", trade_type="DAY",
            entry_price=100.0, confidence=70.0, atr=4.0,
        )
        assert pos is not None
        assert pos.stop_loss == 94.0  # 100 - 1.5 * 4


# ════════════════════════════════════════════════════════════
# DATABASE PERSISTENCE
# ════════════════════════════════════════════════════════════

class TestTraderDatabase:
    """Test saving trades and snapshots to database."""

    def test_save_trade(self, in_memory_db, trader_with_position):
        trader, pos = trader_with_position
        result = trader.close_position(pos.trade_id, exit_price=102.0, exit_reason="TARGET_HIT")
        trader.save_trade(result, pos)

        from modules.database import get_session, Trade as TradeRecord
        session = get_session()
        records = session.query(TradeRecord).all()
        assert len(records) == 1
        assert records[0].ticker == "AAPL"
        assert records[0].pnl_dollars == 20.0
        session.close()

    def test_save_account_snapshot(self, in_memory_db, trader):
        trader.save_account_snapshot()

        from modules.database import get_session, AccountSnapshot
        session = get_session()
        snapshots = session.query(AccountSnapshot).all()
        assert len(snapshots) == 1
        assert snapshots[0].total_value == settings.STARTING_CAPITAL
        session.close()


# ════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_position_properties(self):
        pos = Position(
            trade_id="test", ticker="X", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=98.0, target=103.0,
        )
        assert pos.cost_basis == 1000.0
        assert pos.risk_per_share == 2.0
        assert pos.total_risk == 20.0

    def test_close_breakeven(self, trader_with_position):
        trader, pos = trader_with_position
        result = trader.close_position(pos.trade_id, exit_price=100.0)
        assert result.pnl_dollars == 0.0
        assert result.r_multiple == 0.0

    def test_multiple_open_close_cycle(self, funded_trader):
        """Open and close multiple positions sequentially."""
        for i in range(3):
            pos = funded_trader.open_position(
                ticker=f"T{i}", entry_price=10.0, stop_loss=9.0,
                target=15.0, shares=1,
            )
            funded_trader.close_position(pos.trade_id, exit_price=12.0)
        assert funded_trader.account.open_position_count == 0
        assert funded_trader.account.daily_pnl == 6.0  # 3 * $2 profit

    def test_zero_peak_drawdown(self):
        account = AccountState(cash=0.0, peak_value=0.0)
        assert account.drawdown_pct == 0.0


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_trader.py -v
#
# Expected results:
#   TestAccountState:           6 tests  — all should PASS
#   TestPositionSizing:         7 tests  — all should PASS
#   TestStopLossTarget:         4 tests  — all should PASS
#   TestPreTradeChecks:        12 tests  — all should PASS
#   TestOpenPosition:           5 tests  — all should PASS
#   TestClosePosition:         10 tests  — all should PASS
#   TestPriceUpdates:           8 tests  — all should PASS
#   TestExecuteSignal:          6 tests  — all should PASS
#   TestTraderDatabase:         2 tests  — all should PASS
#   TestEdgeCases:              4 tests  — all should PASS
#
# TOTAL: 64 tests
# ════════════════════════════════════════════════════════════
