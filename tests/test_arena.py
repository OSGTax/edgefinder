"""
Tests for EdgeFinder Arena — Virtual Accounts, Executor, and Engine
====================================================================
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from config import settings
from modules.strategies.base import (
    BaseStrategy,
    StrategyRegistry,
    Signal,
    TradeNotification,
    MarketRegime,
)
from modules.arena.virtual_account import VirtualAccount, Position
from modules.arena.executor import Executor, AuditEntry
from modules.arena.engine import ArenaEngine


# ── HELPERS ──────────────────────────────────────────────────

class SimpleStrategy(BaseStrategy):
    """Simple test strategy that buys everything."""

    @property
    def name(self) -> str:
        return "simple"

    @property
    def version(self) -> str:
        return "1.0.0"

    def init(self) -> None:
        self.initialized = True
        self.trades = []

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for ticker, df in bars.items():
            if not df.empty:
                close = float(df.iloc[-1]["close"])
                signals.append(Signal(
                    ticker=ticker,
                    action="BUY",
                    entry_price=close,
                    stop_loss=close * 0.95,
                    target=close * 1.10,
                    confidence=80.0,
                ))
        return signals

    def on_trade_executed(self, notification: TradeNotification) -> None:
        self.trades.append(notification)


def make_bars(ticker: str, close: float = 100.0, periods: int = 5):
    dates = pd.date_range("2026-03-01", periods=periods, freq="B")
    return pd.DataFrame({
        "open": [close - 1] * periods,
        "high": [close + 1] * periods,
        "low": [close - 2] * periods,
        "close": [close] * periods,
        "volume": [1_000_000] * periods,
    }, index=dates)


@pytest.fixture(autouse=True)
def clean_registry():
    StrategyRegistry.clear()
    yield
    StrategyRegistry.clear()


# ── VIRTUAL ACCOUNT TESTS ───────────────────────────────────

class TestVirtualAccount:

    def test_initial_state(self):
        acc = VirtualAccount("test", starting_capital=10000)
        assert acc.cash == 10000
        assert acc.total_equity == 10000
        assert acc.open_position_count == 0
        assert acc.drawdown_pct == 0.0

    def test_open_position(self):
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        assert acc.open_position(pos) is True
        assert acc.cash == 9000.0
        assert acc.open_position_count == 1
        assert "T1" in acc.positions

    def test_insufficient_cash(self):
        acc = VirtualAccount("test", starting_capital=500)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        assert acc.open_position(pos) is False

    def test_max_positions(self):
        acc = VirtualAccount("test", starting_capital=100000, max_positions=2)
        for i in range(2):
            pos = Position(
                trade_id=f"T{i}", ticker=f"TICK{i}", direction="LONG",
                trade_type="DAY", entry_price=10.0, shares=1,
                stop_loss=9.0, target=11.0,
            )
            assert acc.open_position(pos) is True

        pos3 = Position(
            trade_id="T2", ticker="TICK2", direction="LONG",
            trade_type="DAY", entry_price=10.0, shares=1,
            stop_loss=9.0, target=11.0,
        )
        assert acc.open_position(pos3) is False

    def test_close_position(self):
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        acc.open_position(pos)

        result = acc.close_position("T1", exit_price=110.0, exit_reason="TARGET_HIT")
        assert result is not None
        assert result["pnl_dollars"] == 100.0  # (110-100)*10
        assert result["exit_reason"] == "TARGET_HIT"
        assert acc.open_position_count == 0
        assert acc.cash == 10000.0 + 100.0  # started with 10k, profit 100

    def test_close_losing_position(self):
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        acc.open_position(pos)

        result = acc.close_position("T1", exit_price=95.0, exit_reason="STOP_HIT")
        assert result["pnl_dollars"] == -50.0
        assert result["r_multiple"] == -1.0

    def test_close_nonexistent(self):
        acc = VirtualAccount("test", starting_capital=10000)
        assert acc.close_position("NOPE", 100.0, "STOP_HIT") is None

    def test_win_rate(self):
        acc = VirtualAccount("test", starting_capital=50000)
        # Open and close 3 trades: 2 wins, 1 loss
        for i, (exit_p, reason) in enumerate([
            (110.0, "TARGET_HIT"),
            (95.0, "STOP_HIT"),
            (105.0, "TARGET_HIT"),
        ]):
            pos = Position(
                trade_id=f"T{i}", ticker="AAPL", direction="LONG",
                trade_type="DAY", entry_price=100.0, shares=10,
                stop_loss=95.0, target=110.0,
            )
            acc.open_position(pos)
            acc.close_position(f"T{i}", exit_p, reason)

        assert acc.win_rate == pytest.approx(66.67, rel=0.01)

    def test_drawdown_calculation(self):
        acc = VirtualAccount("test", starting_capital=10000)
        acc.peak_equity = 10000
        # Simulate loss
        acc.cash = 8500
        assert acc.drawdown_pct == pytest.approx(-15.0)

    def test_circuit_breaker(self):
        acc = VirtualAccount("test", starting_capital=10000)
        acc.peak_equity = 10000
        acc.cash = 8000  # -20%
        assert acc.check_drawdown_breaker(-15.0) is True
        assert acc.is_paused is True

    def test_unpause(self):
        acc = VirtualAccount("test", starting_capital=10000)
        acc.is_paused = True
        acc.pause_reason = "test"
        acc.unpause()
        assert acc.is_paused is False

    def test_paused_cannot_open(self):
        acc = VirtualAccount("test", starting_capital=10000)
        acc.is_paused = True
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=1,
            stop_loss=95.0, target=110.0,
        )
        assert acc.open_position(pos) is False

    def test_position_unrealized_pnl(self):
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        pos.last_known_price = 105.0
        assert pos.unrealized_pnl == 50.0
        assert pos.r_multiple == 1.0  # 5/5

    def test_calculate_shares(self):
        acc = VirtualAccount("test", starting_capital=10000, max_risk_pct=0.02)
        shares = acc.calculate_shares(entry_price=100.0, stop_loss=95.0)
        # max risk = 10000 * 0.02 = 200. risk/share = 5. -> 40 shares
        # but concentration limit = 10000 * 0.20 = 2000, so 20 shares
        assert shares == 20

    def test_calculate_shares_zero_risk(self):
        acc = VirtualAccount("test", starting_capital=10000)
        assert acc.calculate_shares(100.0, 100.0) == 0

    def test_take_snapshot(self):
        acc = VirtualAccount("test", starting_capital=10000)
        snap = acc.take_snapshot()
        assert snap["total_equity"] == 10000
        assert snap["strategy_name"] == "test"
        assert snap["drawdown_pct"] == 0.0

    def test_day_trade_tracking(self):
        acc = VirtualAccount("test", starting_capital=50000)
        assert acc.day_trades_remaining() == settings.PDT_DAY_TRADE_LIMIT

        # Open and close day trades
        for i in range(settings.PDT_DAY_TRADE_LIMIT):
            pos = Position(
                trade_id=f"T{i}", ticker="AAPL", direction="LONG",
                trade_type="DAY", entry_price=100.0, shares=1,
                stop_loss=95.0, target=110.0,
            )
            acc.open_position(pos)
            acc.close_position(f"T{i}", 105.0, "TARGET_HIT")

        assert acc.day_trades_remaining() == 0
        assert acc.can_day_trade() is False

    def test_to_dict(self):
        acc = VirtualAccount("test", starting_capital=10000)
        d = acc.to_dict()
        assert d["strategy_name"] == "test"
        assert d["cash"] == 10000
        assert d["total_equity"] == 10000
        assert "positions" in d

    def test_update_position_price(self):
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        acc.open_position(pos)
        acc.update_position_price("T1", 107.0)
        assert acc.positions["T1"].last_known_price == 107.0
        assert acc.positions["T1"].high_water_mark == 107.0


# ── EXECUTOR TESTS ───────────────────────────────────────────

class TestExecutor:

    def test_slippage_calculation(self):
        ex = Executor(base_slippage=0.001, volume_factor=1.0)
        slippage = ex.calculate_slippage(
            price=100.0, shares=100, avg_daily_volume=1_000_000
        )
        assert slippage > 0
        assert slippage < 1.0  # Reasonable for $100 stock

    def test_higher_volume_ratio_more_slippage(self):
        ex = Executor(base_slippage=0.001, volume_factor=1.0)
        small_order = ex.calculate_slippage(100.0, 100, 1_000_000)
        large_order = ex.calculate_slippage(100.0, 100_000, 1_000_000)
        assert large_order > small_order

    def test_low_volume_penalty(self):
        ex = Executor(base_slippage=0.001, volume_factor=1.0)
        normal = ex.calculate_slippage(100.0, 100, 1_000_000)
        low_vol = ex.calculate_slippage(100.0, 100, 10_000)
        assert low_vol > normal

    def test_execute_signal_basic(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.1)
        acc = VirtualAccount("test", starting_capital=10000)
        signal = Signal(
            ticker="AAPL", action="BUY", entry_price=100.0,
            stop_loss=95.0, target=110.0, confidence=80.0,
        )
        audit = ex.execute_signal(
            signal=signal, account=acc, avg_daily_volume=1_000_000,
            price_source="test",
        )
        assert audit is not None
        assert audit.ticker == "AAPL"
        assert audit.execution_price > signal.entry_price  # Slippage added
        assert len(ex.audit_log) == 1
        assert acc.open_position_count == 1

    def test_execute_sell_skipped(self):
        ex = Executor()
        acc = VirtualAccount("test", starting_capital=10000)
        signal = Signal(
            ticker="AAPL", action="SELL", entry_price=100.0,
            stop_loss=105.0, target=90.0,
        )
        audit = ex.execute_signal(signal=signal, account=acc)
        assert audit is None

    def test_execute_bad_rr_rejected(self):
        ex = Executor()
        acc = VirtualAccount("test", starting_capital=10000)
        signal = Signal(
            ticker="AAPL", action="BUY", entry_price=100.0,
            stop_loss=95.0, target=101.0,  # R:R = 0.2 < 1.5 minimum
            confidence=80.0,
        )
        audit = ex.execute_signal(signal=signal, account=acc)
        assert audit is None

    def test_execute_paused_account_rejected(self):
        ex = Executor()
        acc = VirtualAccount("test", starting_capital=10000)
        acc.is_paused = True
        signal = Signal(
            ticker="AAPL", action="BUY", entry_price=100.0,
            stop_loss=95.0, target=110.0, confidence=80.0,
        )
        audit = ex.execute_signal(signal=signal, account=acc)
        assert audit is None

    def test_execute_pdt_limit(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc = VirtualAccount("test", starting_capital=100000)

        # Use up PDT limit
        for i in range(settings.PDT_DAY_TRADE_LIMIT):
            pos = Position(
                trade_id=f"T{i}", ticker="AAPL", direction="LONG",
                trade_type="DAY", entry_price=100.0, shares=1,
                stop_loss=95.0, target=110.0,
            )
            acc.open_position(pos)
            acc.close_position(f"T{i}", 105.0, "TARGET_HIT")

        signal = Signal(
            ticker="MSFT", action="BUY", entry_price=300.0,
            stop_loss=285.0, target=330.0, confidence=80.0,
            trade_type="DAY",
        )
        audit = ex.execute_signal(signal=signal, account=acc)
        assert audit is None

    def test_audit_trail_completeness(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc = VirtualAccount("test", starting_capital=10000)
        signal = Signal(
            ticker="AAPL", action="BUY", entry_price=100.0,
            stop_loss=95.0, target=110.0, confidence=80.0,
        )
        audit = ex.execute_signal(
            signal=signal, account=acc, avg_daily_volume=1_000_000,
            price_source="alpaca",
            bar_data={"open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000000},
            market_regime=MarketRegime(trend="bull"),
            signal_overlap=2,
            position_overlap=1,
        )
        d = audit.to_dict()
        assert d["price_source"] == "alpaca"
        assert d["market_regime"] == "bull"
        assert d["signal_overlap"] == 2
        assert d["position_overlap"] == 1
        assert d["bar_data_at_decision"]["close"] == 100

    def test_close_on_stop(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        acc.open_position(pos)

        result = ex.close_on_stop_or_target(acc, "T1", 94.0)
        assert result is not None
        assert result["exit_reason"] == "STOP_HIT"

    def test_close_on_target(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        acc.open_position(pos)

        result = ex.close_on_stop_or_target(acc, "T1", 111.0)
        assert result is not None
        assert result["exit_reason"] == "TARGET_HIT"

    def test_trailing_stop_activation(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=120.0,
        )
        acc.open_position(pos)

        # Price hits 1R (105 = entry + risk), should activate trailing
        result = ex.close_on_stop_or_target(acc, "T1", 105.0)
        assert result is None  # Not closed yet
        assert acc.positions["T1"].trailing_stop == 100.0  # Breakeven

    def test_no_close_when_price_in_range(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc = VirtualAccount("test", starting_capital=10000)
        pos = Position(
            trade_id="T1", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=95.0, target=110.0,
        )
        acc.open_position(pos)

        result = ex.close_on_stop_or_target(acc, "T1", 102.0)
        assert result is None
        assert acc.positions["T1"].last_known_price == 102.0

    def test_get_audit_log_filtered(self):
        ex = Executor(base_slippage=0.0001, volume_factor=0.01)
        acc1 = VirtualAccount("strat_a", starting_capital=10000)
        acc2 = VirtualAccount("strat_b", starting_capital=10000)

        for acc in [acc1, acc2]:
            signal = Signal(
                ticker="AAPL", action="BUY", entry_price=100.0,
                stop_loss=95.0, target=110.0, confidence=80.0,
            )
            ex.execute_signal(signal=signal, account=acc)

        all_logs = ex.get_audit_log()
        assert len(all_logs) == 2
        filtered = ex.get_audit_log(strategy_name="strat_a")
        assert len(filtered) == 1


# ── ARENA ENGINE TESTS ──────────────────────────────────────

class TestArenaEngine:

    def test_add_strategy(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)
        assert "simple" in engine.strategies
        assert "simple" in engine.accounts
        assert engine.is_enabled("simple") is True

    def test_enable_disable(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        engine.disable_strategy("simple")
        assert engine.is_enabled("simple") is False

        engine.enable_strategy("simple")
        assert engine.is_enabled("simple") is True

    def test_run_signal_check(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        executed = engine.run_signal_check(bars, price_source="test")
        assert len(executed) == 1
        assert executed[0]["ticker"] == "AAPL"
        assert engine.accounts["simple"].open_position_count == 1

    def test_disabled_strategy_skipped(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)
        engine.disable_strategy("simple")

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        executed = engine.run_signal_check(bars)
        assert len(executed) == 0

    def test_multiple_strategies(self):
        engine = ArenaEngine(starting_capital=10000)

        s1 = SimpleStrategy()
        s1.init()
        s2 = SimpleStrategy()
        s2.init()

        engine.add_strategy("strat_a", s1)
        engine.add_strategy("strat_b", s2)

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        executed = engine.run_signal_check(bars)

        # Both strategies should have opened positions
        assert engine.accounts["strat_a"].open_position_count == 1
        assert engine.accounts["strat_b"].open_position_count == 1

    def test_signal_overlap_tracking(self):
        engine = ArenaEngine(starting_capital=10000)

        s1 = SimpleStrategy()
        s1.init()
        s2 = SimpleStrategy()
        s2.init()

        engine.add_strategy("strat_a", s1)
        engine.add_strategy("strat_b", s2)

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        executed = engine.run_signal_check(bars)

        # Both signaled AAPL, so overlap = 1 (other strategies)
        for entry in executed:
            assert entry["signal_overlap"] == 1

    def test_monitor_positions_stop_hit(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        # Open a position
        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        engine.run_signal_check(bars)

        # Price drops below stop
        closed = engine.monitor_positions({"AAPL": 90.0})
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "STOP_HIT"
        assert engine.accounts["simple"].open_position_count == 0

    def test_monitor_positions_target_hit(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        engine.run_signal_check(bars)

        closed = engine.monitor_positions({"AAPL": 115.0})
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "TARGET_HIT"

    def test_take_snapshots(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        snapshots = engine.take_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["strategy_name"] == "simple"
        assert snapshots[0]["total_equity"] == 10000

    def test_get_leaderboard(self):
        engine = ArenaEngine(starting_capital=10000)
        s1 = SimpleStrategy()
        s1.init()
        s2 = SimpleStrategy()
        s2.init()
        engine.add_strategy("winner", s1)
        engine.add_strategy("loser", s2)

        # Simulate: winner makes money
        engine.accounts["winner"].cash = 11000
        engine.accounts["winner"].peak_equity = 11000
        engine.accounts["loser"].cash = 9000

        board = engine.get_leaderboard()
        assert board[0]["strategy_name"] == "winner"
        assert board[1]["strategy_name"] == "loser"

    def test_get_overlap_report(self):
        engine = ArenaEngine(starting_capital=10000)
        s1 = SimpleStrategy()
        s1.init()
        s2 = SimpleStrategy()
        s2.init()
        engine.add_strategy("a", s1)
        engine.add_strategy("b", s2)

        # Both hold AAPL
        for name in ["a", "b"]:
            pos = Position(
                trade_id=f"T_{name}", ticker="AAPL", direction="LONG",
                trade_type="DAY", entry_price=100.0, shares=1,
                stop_loss=95.0, target=110.0,
            )
            engine.accounts[name].open_position(pos)

        report = engine.get_overlap_report()
        assert report["overlap_count"] == 1
        assert "AAPL" in report["overlapping_positions"]

    def test_update_market_regime(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        regime = MarketRegime(trend="bear", volatility="high", vix_level=35.0)
        engine.update_market_regime(regime)
        assert engine.market_regime.trend == "bear"

    def test_get_status(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        status = engine.get_status()
        assert status["strategies"] == 1
        assert status["enabled"] == 1
        assert "leaderboard" in status
        assert "overlap" in status

    def test_drawdown_auto_pause(self):
        engine = ArenaEngine(starting_capital=10000, drawdown_pause_pct=-10.0)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        # Force drawdown
        engine.accounts["simple"].cash = 8000
        engine.accounts["simple"].peak_equity = 10000

        # Monitor should trigger circuit breaker
        engine.monitor_positions({})
        assert engine.accounts["simple"].is_paused is True

    def test_load_strategies_from_registry(self):
        StrategyRegistry.register("simple")(SimpleStrategy)
        engine = ArenaEngine(starting_capital=10000)
        loaded = engine.load_strategies()
        assert "simple" in loaded
        assert "simple" in engine.accounts

    def test_load_specific_strategies(self):
        StrategyRegistry.register("simple")(SimpleStrategy)

        class AnotherStrategy(SimpleStrategy):
            @property
            def name(self):
                return "another"

        StrategyRegistry.register("another")(AnotherStrategy)

        engine = ArenaEngine(starting_capital=10000)
        loaded = engine.load_strategies(names=["simple"])
        assert "simple" in loaded
        assert "another" not in loaded

    def test_strategy_receives_trade_notification(self):
        engine = ArenaEngine(starting_capital=10000)
        strategy = SimpleStrategy()
        strategy.init()
        engine.add_strategy("simple", strategy)

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        engine.run_signal_check(bars)

        assert len(strategy.trades) == 1
        assert strategy.trades[0].ticker == "AAPL"
        assert strategy.trades[0].action == "BUY"

    def test_strategy_error_doesnt_crash_engine(self):
        class CrashStrategy(SimpleStrategy):
            def generate_signals(self, bars):
                raise RuntimeError("oops")

        engine = ArenaEngine(starting_capital=10000)
        s = CrashStrategy()
        s.init()
        engine.add_strategy("crash", s)

        bars = {"AAPL": make_bars("AAPL", close=100.0)}
        # Should not raise
        executed = engine.run_signal_check(bars)
        assert len(executed) == 0
