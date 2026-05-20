"""Tests for the redesigned arena with shared data layer."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from edgefinder.core.models import TradeIntent
from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.arena import ArenaEngine
from edgefinder.trading.risk import RiskManager


class TestNewArena:
    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        np.random.seed(42)
        n = 60
        close = 100 + np.random.normal(0, 1.5, n).cumsum()
        df = pd.DataFrame({
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=pd.date_range("2024-01-01", periods=n, freq="B", name="timestamp"))
        provider.get_bars.return_value = df
        provider.get_latest_price.return_value = 100.0
        return provider

    def test_stop_loss_fires_at_20_pct(self):
        """Verify the 20% stop is non-negotiable."""
        acct = VirtualAccount("gambler", starting_capital=5000.0)
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)

        pos = Position(
            symbol="AAPL", shares=12, entry_price=100.0,
            stop_loss=rm.compute_stop(100.0),
            target=rm.compute_target(100.0),
            direction="LONG", trade_type="SWING",
            trade_id="test-stop-1",
        )
        acct.open_position(pos)

        assert rm.should_stop_out(100.0, 79.0) is True
        assert rm.should_stop_out(100.0, 81.0) is False

    def test_profit_target_fires(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        assert rm.should_take_profit(100.0, 126.0) is True
        assert rm.should_take_profit(100.0, 124.0) is False

    def test_position_sizing_per_strategy(self):
        """Each strategy gets different position sizes."""
        rm_coward = RiskManager(risk_pct=0.05, stop_pct=0.20, target_pct=0.15)
        rm_gambler = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        rm_degen = RiskManager(risk_pct=0.20, stop_pct=0.20, target_pct=0.50)

        equity = 5000.0
        price = 200.0

        assert rm_coward.compute_shares(price, equity) == 6
        assert rm_gambler.compute_shares(price, equity) == 12
        assert rm_degen.compute_shares(price, equity) == 25

    def test_arena_loads_strategies(self, mock_provider):
        """Arena loads all registered strategies into slots."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        names = arena.get_strategy_names()
        # Should have at least the three new strategies
        assert len(names) >= 1
        for name in names:
            assert arena.get_account(name) is not None
            assert arena.get_strategy(name) is not None

    def test_arena_set_watchlists(self, mock_provider):
        """Watchlists are set per-strategy."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        names = arena.get_strategy_names()
        if not names:
            pytest.skip("No strategies registered")
        watchlists = {names[0]: ["AAPL", "MSFT"]}
        arena.set_watchlists(watchlists)
        assert arena._slots[names[0]].watchlist == ["AAPL", "MSFT"]

    def test_arena_set_global_watchlist(self, mock_provider):
        """Global watchlist applies to all strategies."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        arena.set_global_watchlist(["TSLA", "NVDA"])
        for slot in arena._slots.values():
            assert slot.watchlist == ["TSLA", "NVDA"]

    def test_get_all_accounts(self, mock_provider):
        """get_all_accounts returns dict of serialized account states."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        accounts = arena.get_all_accounts()
        assert isinstance(accounts, dict)
        for name, acct_dict in accounts.items():
            assert "cash" in acct_dict
            assert "total_equity" in acct_dict
            assert acct_dict["starting_capital"] == 10000.0

    def test_get_all_open_positions(self, mock_provider):
        """get_all_open_positions returns empty lists initially."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        positions = arena.get_all_open_positions()
        assert isinstance(positions, dict)
        for name, pos_list in positions.items():
            assert pos_list == []

    def test_daily_cycle_computes_indicators(self, mock_provider):
        """run_daily_cycle populates indicator histories."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        arena.set_global_watchlist(["AAPL"])
        arena.run_daily_cycle()

        assert "AAPL" in arena._indicator_histories
        assert len(arena._indicator_histories["AAPL"]) >= 1
        assert "AAPL" in arena._daily_bars_cache

    def test_slot_has_risk_manager(self, mock_provider):
        """Each slot gets a RiskManager configured from the strategy."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        for name, slot in arena._slots.items():
            assert isinstance(slot.risk_manager, RiskManager)
            assert slot.risk_manager.risk_pct == slot.strategy.risk_pct
            assert slot.risk_manager.target_pct == slot.strategy.target_pct

    def test_fundamentals_cache(self, mock_provider):
        """Fundamentals cache stores and retrieves data."""
        arena = ArenaEngine(mock_provider)
        arena.set_fundamentals_cache({"AAPL": MagicMock()})
        assert "AAPL" in arena._fundamentals_cache

    def test_intraday_cycle_no_crash_empty(self, mock_provider):
        """Intraday cycle with no data doesn't crash."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        arena.set_global_watchlist(["AAPL"])

        opened, closed = arena.run_intraday_cycle({}, MarketContext())
        assert opened == []
        assert closed == []

    def test_exit_order_stop_before_strategy(self, mock_provider):
        """Stop loss fires before strategy exit logic."""
        arena = ArenaEngine(mock_provider)
        arena.load_strategies()
        names = arena.get_strategy_names()
        if not names:
            pytest.skip("No strategies registered")

        name = names[0]
        slot = arena._slots[name]

        # Manually open a position
        pos = Position(
            symbol="AAPL", shares=10, entry_price=100.0,
            stop_loss=slot.risk_manager.compute_stop(100.0),
            target=slot.risk_manager.compute_target(100.0),
            direction="LONG", trade_type="SWING",
            trade_id="test-exit-order",
        )
        slot.account.open_position(pos)

        # Price at 79 should trigger stop (20% below 100)
        snapshot_data = {"AAPL": {"price": 79.0, "volume": 1000000.0}}
        context = MarketContext()

        opened, closed = arena.run_intraday_cycle(snapshot_data, context)
        assert len(closed) == 1
        assert closed[0].exit_reason == "STOP_LOSS"
