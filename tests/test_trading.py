"""Tests for edgefinder/trading/ — account, executor, arena, journal."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from edgefinder.core.models import (
    Direction, Signal, SignalAction, Trade, TradeStatus, TradeType,
)
from edgefinder.db.models import TradeRecord
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.arena import ArenaEngine
from edgefinder.trading.executor import Executor
from edgefinder.trading.journal import TradeJournal


# ── Virtual Account Tests ────────────────────────────────


class TestVirtualAccount:
    def test_initial_state(self):
        acct = VirtualAccount("alpha")
        assert acct.cash == 5000.0
        assert acct.buying_power == 5000.0
        assert acct.position_count == 0
        assert acct.is_paused is False

    def test_open_position_deducts_cash(self):
        acct = VirtualAccount("alpha")
        pos = Position(
            symbol="AAPL", shares=10, entry_price=150.0,
            stop_loss=145.0, target=160.0, direction="LONG", trade_type="DAY",
        )
        acct.open_position(pos)
        assert acct.cash == 5000.0 - 1500.0
        assert acct.position_count == 1

    def test_close_position_returns_cash(self):
        acct = VirtualAccount("alpha")
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

    def test_max_positions(self):
        acct = VirtualAccount("alpha")
        for i in range(5):
            pos = Position(
                symbol=f"T{i}", shares=1, entry_price=10.0,
                stop_loss=9.0, target=12.0, direction="LONG", trade_type="SWING",
            )
            acct.open_position(pos)
        allowed, reason = acct.can_open_position(10.0)
        assert allowed is False
        assert "Max positions" in reason

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
            datetime.utcnow() - timedelta(hours=i) for i in range(3)
        ]
        allowed, reason = acct.can_open_position(100.0, "DAY")
        assert allowed is False
        assert "PDT" in reason

    def test_pdt_disabled_allows_day_trades(self):
        acct = VirtualAccount("alpha", pdt_enabled=False)
        acct._day_trades = [
            datetime.utcnow() - timedelta(hours=i) for i in range(5)
        ]
        allowed, _ = acct.can_open_position(100.0, "DAY")
        assert allowed is True

    def test_revenge_trade_cooldown(self):
        acct = VirtualAccount("alpha")
        acct._last_stop_out = datetime.utcnow()
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
        # But concentration cap: 20% of $5000 = $1000 / $100 = 10 shares.
        signal = self._make_signal(entry_price=100.0, stop_loss=95.0)
        trade = executor.execute_signal(signal)
        assert trade.shares == 10  # limited by concentration cap

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


class TestArena:
    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_latest_price.return_value = 100.0
        # Return a DataFrame with enough bars
        np.random.seed(42)
        n = 250
        close = 100 + np.random.normal(0, 1, n).cumsum()
        df = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        }, index=pd.date_range("2023-01-01", periods=n, freq="D", name="timestamp"))
        provider.get_bars.return_value = df
        return provider

    def test_load_strategies(self, mock_provider):
        # Ensure strategies are registered
        import importlib
        from edgefinder.strategies import alpha, bravo, charlie
        from edgefinder.strategies.base import StrategyRegistry
        StrategyRegistry.clear()
        importlib.reload(alpha)
        importlib.reload(bravo)
        importlib.reload(charlie)

        arena = ArenaEngine(mock_provider)
        arena.load_strategies(pdt_config={"alpha": True})
        assert len(arena.get_strategy_names()) == 3
        alpha_acct = arena.get_account("alpha")
        assert alpha_acct.pdt_enabled is True

    def test_get_all_accounts(self, mock_provider):
        import importlib
        from edgefinder.strategies import alpha, bravo, charlie
        from edgefinder.strategies.base import StrategyRegistry
        StrategyRegistry.clear()
        importlib.reload(alpha)
        importlib.reload(bravo)
        importlib.reload(charlie)

        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        accounts = arena.get_all_accounts()
        assert len(accounts) == 3
        for name, acct in accounts.items():
            assert acct["cash"] == 5000.0

    def test_set_global_watchlist(self, mock_provider):
        import importlib
        from edgefinder.strategies import alpha, bravo, charlie
        from edgefinder.strategies.base import StrategyRegistry
        StrategyRegistry.clear()
        importlib.reload(alpha)
        importlib.reload(bravo)
        importlib.reload(charlie)

        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        arena.set_global_watchlist(["AAPL", "MSFT"])
        # Signal check should not crash
        trades = arena.run_signal_check()
        assert isinstance(trades, list)


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
        trade.exit_time = datetime.utcnow()
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
                trade_id=f"sym-{sym}-{datetime.utcnow().timestamp()}",
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
