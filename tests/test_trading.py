"""Tests for edgefinder/trading/ — account, executor, arena, journal."""

from datetime import datetime, timedelta, timezone

import pytest

from edgefinder.core.models import (
    Direction, Signal, SignalAction, Trade, TradeStatus, TradeType,
)
from edgefinder.db.models import TradeRecord
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.executor import Executor
from edgefinder.trading.journal import TradeJournal


# ── Virtual Account Tests ────────────────────────────────


class TestVirtualAccount:
    def test_initial_state(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        assert acct.cash == 5000.0
        assert acct.buying_power == 5000.0
        assert acct.position_count == 0
        assert acct.is_paused is False

    def test_open_position_deducts_cash(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=150.0,
            stop_loss=145.0, target=160.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)
        assert acct.cash == 5000.0 - 1500.0
        assert acct.position_count == 1

    def test_close_position_returns_cash(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=150.0,
            stop_loss=145.0, target=160.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)
        result = acct.close_position(pos, 155.0, "TARGET_HIT")
        assert result["pnl_dollars"] == 50.0
        assert acct.cash == 5000.0 + 50.0
        assert acct.position_count == 0

    def test_insufficient_buying_power(self):
        acct = VirtualAccount("alpha", starting_capital=100.0)
        allowed, reason = acct.can_open_position(500.0)
        assert allowed is False
        assert "Insufficient" in reason

    def test_drawdown_circuit_breaker(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        acct.cash = 3500.0  # 30% drawdown
        acct.peak_equity = 5000.0
        allowed, reason = acct.can_open_position(100.0)
        assert allowed is False
        assert "circuit breaker" in reason.lower()

    def test_pdt_enforcement(self):
        acct = VirtualAccount("alpha", pdt_enabled=True)
        # Simulate 3 recent day trades
        acct._day_trades = [
            datetime.now(timezone.utc) - timedelta(hours=i) for i in range(3)
        ]
        allowed, reason = acct.can_open_position(100.0, "DAY")
        assert allowed is False
        assert "PDT" in reason

    def test_pdt_disabled_allows_day_trades(self):
        acct = VirtualAccount("alpha", pdt_enabled=False)
        acct._day_trades = [
            datetime.now(timezone.utc) - timedelta(hours=i) for i in range(5)
        ]
        allowed, _ = acct.can_open_position(100.0, "DAY")
        assert allowed is True

    def test_revenge_trade_cooldown(self):
        acct = VirtualAccount("alpha")
        acct._last_stop_out = datetime.now(timezone.utc)
        allowed, reason = acct.can_open_position(100.0)
        assert allowed is False
        assert "cooldown" in reason.lower()


class TestPosition:
    def test_unrealized_pnl_long(self):
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        assert pos.unrealized_pnl(105.0) == 50.0
        assert pos.unrealized_pnl(95.0) == -50.0

    def test_unrealized_pnl_short(self):
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=105.0, target=90.0, direction="SHORT", trade_type="DAY",
        )
        assert pos.unrealized_pnl(95.0) == 50.0
        assert pos.unrealized_pnl(105.0) == -50.0

    def test_stop_out_long(self):
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        assert pos.should_stop_out(94.0) is True
        assert pos.should_stop_out(96.0) is False

    def test_take_profit_long(self):
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        assert pos.should_take_profit(111.0) is True
        assert pos.should_take_profit(109.0) is False

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


# ── Executor Tests ───────────────────────────────────────


class TestExecutor:
    def _make_signal(self, **overrides) -> Signal:
        defaults = dict(
            ticker="AAPL",
            action=SignalAction.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            target=110.0,
            confidence=70.0,
            trade_type=TradeType.DAY,
            strategy_name="alpha",
        )
        defaults.update(overrides)
        return Signal(**defaults)

    def test_execute_signal_opens_position(self):
        acct = VirtualAccount("alpha")
        executor = Executor(acct)
        signal = self._make_signal()
        trade = executor.execute_signal(signal)
        assert trade is not None
        assert trade.status == TradeStatus.OPEN
        assert acct.position_count == 1

    def test_execute_signal_applies_slippage(self):
        acct = VirtualAccount("alpha")
        executor = Executor(acct)
        signal = self._make_signal(entry_price=100.0)
        trade = executor.execute_signal(signal)
        assert trade.entry_price > 100.0  # BUY slippage increases price

    def test_execute_signal_sizes_by_risk(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        executor = Executor(acct)
        # Risk per share = 100 - 95 = $5. Max risk = 5000 * 0.02 = $100. Shares = 20.
        # But concentration cap: 20% of $5000 = $1000 / $100.05 (slippage) = 9 shares.
        signal = self._make_signal(entry_price=100.0, stop_loss=95.0)
        trade = executor.execute_signal(signal)
        assert trade.shares == 9  # limited by concentration cap (slippage-adjusted)

    def test_rejects_when_insufficient_funds(self):
        acct = VirtualAccount("alpha", starting_capital=50.0)
        executor = Executor(acct)
        signal = self._make_signal(entry_price=100.0)
        trade = executor.execute_signal(signal)
        assert trade is None

    def test_check_positions_stop_hit(self):
        acct = VirtualAccount("alpha")
        executor = Executor(acct)
        signal = self._make_signal()
        executor.execute_signal(signal)
        closed = executor.check_positions({"AAPL": 90.0})
        assert len(closed) == 1
        assert closed[0].exit_reason == "STOP_HIT"
        assert closed[0].status == TradeStatus.CLOSED

    def test_check_positions_target_hit(self):
        acct = VirtualAccount("alpha")
        executor = Executor(acct)
        signal = self._make_signal()
        executor.execute_signal(signal)
        closed = executor.check_positions({"AAPL": 115.0})
        assert len(closed) == 1
        assert closed[0].exit_reason == "TARGET_HIT"

    def test_integrity_hash_chain(self):
        acct = VirtualAccount("alpha")
        executor = Executor(acct)
        s1 = self._make_signal(ticker="AAPL")
        s2 = self._make_signal(ticker="MSFT")
        t1 = executor.execute_signal(s1)
        t2 = executor.execute_signal(s2)
        assert t1.integrity_hash != t2.integrity_hash
        assert t1.sequence_num == 1
        assert t2.sequence_num == 2


# ── Arena Tests ──────────────────────────────────────────
# Old TestArena class removed — new arena tested in tests/test_arena_new.py


# ── Journal Tests ────────────────────────────────────────


class TestTradeJournal:
    def test_log_open_trade(self, db_session):
        journal = TradeJournal(db_session)
        trade = Trade(
            trade_id="test-001",
            strategy_name="alpha",
            symbol="AAPL",
            direction=Direction.LONG,
            trade_type=TradeType.DAY,
            entry_price=150.0,
            shares=10,
            stop_loss=145.0,
            target=160.0,
            confidence=75.0,
        )
        journal.log_trade(trade)
        records = journal.get_open_trades("alpha")
        assert len(records) == 1
        assert records[0].symbol == "AAPL"

    def test_log_closed_trade_updates(self, db_session):
        journal = TradeJournal(db_session)
        # Open
        trade = Trade(
            trade_id="test-002",
            strategy_name="alpha",
            symbol="MSFT",
            direction=Direction.LONG,
            trade_type=TradeType.SWING,
            entry_price=300.0,
            shares=5,
            stop_loss=290.0,
            target=320.0,
            confidence=80.0,
        )
        journal.log_trade(trade)
        # Close
        trade.status = TradeStatus.CLOSED
        trade.exit_price = 315.0
        trade.pnl_dollars = 75.0
        trade.pnl_percent = 5.0
        trade.r_multiple = 1.5
        trade.exit_reason = "TARGET_HIT"
        trade.exit_time = datetime.now(timezone.utc)
        journal.log_trade(trade)

        records = journal.get_closed_trades("alpha")
        assert len(records) == 1
        assert records[0].pnl_dollars == 75.0

    def test_compute_stats(self, db_session):
        journal = TradeJournal(db_session)
        # Log some closed trades
        for i, (pnl, r) in enumerate([(50, 1.0), (100, 2.0), (-30, -0.6)]):
            trade = Trade(
                trade_id=f"stat-{i}",
                strategy_name="alpha",
                symbol="AAPL",
                direction=Direction.LONG,
                trade_type=TradeType.DAY,
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                target=110.0,
                confidence=70.0,
                status=TradeStatus.CLOSED,
                pnl_dollars=float(pnl),
                r_multiple=r,
                exit_reason="TARGET_HIT" if pnl > 0 else "STOP_HIT",
            )
            journal.log_trade(trade)

        stats = journal.compute_stats("alpha")
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["total_pnl"] == 120.0

    def test_compute_stats_empty(self, db_session):
        journal = TradeJournal(db_session)
        stats = journal.compute_stats("empty_strategy")
        assert stats["total_trades"] == 0

    def test_filter_by_symbol(self, db_session):
        journal = TradeJournal(db_session)
        for sym in ["AAPL", "MSFT", "AAPL"]:
            trade = Trade(
                trade_id=f"sym-{sym}-{datetime.now(timezone.utc).timestamp()}",
                strategy_name="alpha",
                symbol=sym,
                direction=Direction.LONG,
                trade_type=TradeType.DAY,
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                target=110.0,
                confidence=70.0,
            )
            journal.log_trade(trade)
        aapl_trades = journal.get_trades(symbol="AAPL")
        assert len(aapl_trades) == 2


# ── Financial Account Verification Tests ────────────────


class TestAccountFinancialIntegrity:
    """Verify core financial account invariants: cash tracking, balance
    preservation, negative-balance prevention, and self-healing."""

    def test_multi_position_cash_tracking(self):
        """Open 2 positions, close 1 — cash must equal
        starting - remaining_cost + closed_proceeds."""
        acct = VirtualAccount("alpha", starting_capital=5000.0)

        pos1 = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="SWING",
        )
        pos2 = Position(
            symbol="MSFT", shares=5, entry_price=200.0,
            stop_loss=190.0, target=220.0, direction="LONG", trade_type="SWING",
        )
        acct.open_position(pos1)  # cost 1000
        acct.open_position(pos2)  # cost 1000
        assert acct.cash == 3000.0

        # Close AAPL at a profit
        result = acct.close_position(pos1, 105.0, "TARGET_HIT")
        proceeds = pos1.shares * 100.0 + result["pnl_dollars"]  # cost_basis + pnl
        assert result["pnl_dollars"] == 50.0
        # Cash = 3000 + 1050 (cost 1000 + profit 50)
        assert acct.cash == 4050.0
        # MSFT still open — remaining cost basis
        assert acct.open_positions_value == 1000.0
        # Total equity = cash + open cost
        assert acct.total_equity == 5050.0

    def test_cannot_go_negative_through_trades(self):
        """Executor must reject a trade when cost exceeds buying power."""
        acct = VirtualAccount("alpha", starting_capital=500.0)
        executor = Executor(acct)
        # Signal for a $100 stock — risk sizing will try to buy shares
        # but even 1 share at $100.05 (after slippage) fits in $500
        # Open a big position first to eat up most cash
        pos = Position(
            symbol="TSLA", shares=4, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="SWING",
        )
        acct.open_position(pos)  # cost = 400, cash = 100

        # Now try to open another position that costs more than remaining cash
        signal = Signal(
            ticker="NVDA", action=SignalAction.BUY,
            entry_price=200.0, stop_loss=190.0, target=220.0,
            confidence=70.0, trade_type=TradeType.DAY, strategy_name="alpha",
        )
        trade = executor.execute_signal(signal)
        assert trade is None  # rejected — insufficient buying power
        assert acct.cash == 100.0  # unchanged

    def test_peak_equity_only_updates_on_new_highs(self):
        """Peak equity stays at previous high after a losing trade,
        and updates only when equity exceeds the old peak."""
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        assert acct.peak_equity == 5000.0

        # Open and close at a loss
        pos1 = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="SWING",
        )
        acct.open_position(pos1)
        acct.close_position(pos1, 95.0, "STOP_HIT")  # pnl = -50
        assert acct.cash == 4950.0
        assert acct.peak_equity == 5000.0  # unchanged — no new high

        # Open and close at a big profit
        pos2 = Position(
            symbol="MSFT", shares=10, entry_price=100.0,
            stop_loss=95.0, target=120.0, direction="LONG", trade_type="SWING",
        )
        acct.open_position(pos2)
        acct.close_position(pos2, 120.0, "TARGET_HIT")  # pnl = +200
        assert acct.cash == 5150.0
        assert acct.peak_equity == 5150.0  # updated — new high

    def test_account_recalculation_self_healing(self, db_session):
        """Simulate corrupted cash, then verify recalculation from trades
        table restores the correct balance."""
        from sqlalchemy import func

        journal = TradeJournal(db_session)

        # Log a closed winning trade
        trade1 = Trade(
            trade_id="heal-001", strategy_name="alpha", symbol="AAPL",
            direction=Direction.LONG, trade_type=TradeType.DAY,
            entry_price=100.0, shares=10, stop_loss=95.0, target=110.0,
            confidence=70.0, status=TradeStatus.CLOSED,
            pnl_dollars=50.0, pnl_percent=5.0, r_multiple=1.0,
            exit_price=105.0, exit_reason="TARGET_HIT",
            exit_time=datetime.now(timezone.utc),
        )
        journal.log_trade(trade1)

        # Log an open position
        trade2 = Trade(
            trade_id="heal-002", strategy_name="alpha", symbol="MSFT",
            direction=Direction.LONG, trade_type=TradeType.SWING,
            entry_price=200.0, shares=5, stop_loss=190.0, target=220.0,
            confidence=75.0, status=TradeStatus.OPEN,
        )
        journal.log_trade(trade2)

        # Recalculate from trades table (same formula as services.py)
        realized = (
            db_session.query(func.coalesce(func.sum(TradeRecord.pnl_dollars), 0.0))
            .filter(TradeRecord.strategy_name == "alpha", TradeRecord.status == "CLOSED")
            .scalar()
        )
        open_cost = (
            db_session.query(
                func.coalesce(func.sum(TradeRecord.entry_price * TradeRecord.shares), 0.0)
            )
            .filter(TradeRecord.strategy_name == "alpha", TradeRecord.status == "OPEN")
            .scalar()
        )
        starting_capital = 5000.0
        correct_cash = round(starting_capital + realized - open_cost, 2)

        # correct_cash = 5000 + 50 - 1000 = 4050
        assert correct_cash == 4050.0

        # Simulate corrupted account with wrong cash
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        acct.cash = 9999.99  # obviously wrong

        # Apply the self-healing formula
        acct.cash = correct_cash
        acct.realized_pnl = round(realized, 2)
        assert acct.cash == 4050.0
        assert acct.realized_pnl == 50.0


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

    def test_equity_without_market_price_uses_entry(self):
        acct = VirtualAccount("alpha", starting_capital=5000.0)
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=95.0, target=110.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)
        # market_price is None — falls back to entry_price
        assert acct.total_equity == 5000.0  # 4000 + 1000 (cost basis)


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
