"""Mark-to-market equity and drawdown invariants (plan item C1).

Before this refactor, VirtualAccount.total_equity used cost basis. That
meant underwater positions didn't lower equity or trigger the drawdown
circuit breaker, and equity curves persisted to StrategySnapshot were
cost-basis — contradicting the invariant in CLAUDE.md:
    Total Account Value = Cash + market value of positions
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from edgefinder.core.models import Signal, SignalAction, TradeType
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.executor import Executor


@pytest.fixture
def long_account():
    """Account with one long position at 100 bought with 10 shares ($1000)."""
    acct = VirtualAccount("alpha")
    pos = Position(
        symbol="XYZ", shares=10, entry_price=100.0,
        stop_loss=90.0, target=120.0, direction="LONG", trade_type="SWING",
    )
    acct.open_position(pos)
    return acct, pos


class TestMarkToMarketEquity:
    def test_before_mark_equity_uses_entry_price(self, long_account):
        acct, _ = long_account
        # cash = 5000 - 1000 = 4000; no mark yet → fallback entry_price
        assert acct.total_equity == pytest.approx(5000.0)

    def test_equity_follows_price_up(self, long_account):
        acct, pos = long_account
        pos.current_price = 110.0  # +10%
        # cash 4000 + market_value 10 * 110 = 5100
        assert acct.total_equity == pytest.approx(5100.0)

    def test_equity_follows_price_down(self, long_account):
        acct, pos = long_account
        pos.current_price = 80.0  # -20%
        # cash 4000 + market_value 10 * 80 = 4800
        assert acct.total_equity == pytest.approx(4800.0)

    def test_drawdown_uses_market_value(self, long_account):
        acct, pos = long_account
        # peak starts at starting_capital 5000; before mark drawdown=0
        assert acct.drawdown_pct == pytest.approx(0.0)
        pos.current_price = 80.0  # underwater 20%
        # equity 4800 vs peak 5000 → 4% drawdown
        assert acct.drawdown_pct == pytest.approx(0.04)

    def test_check_positions_stamps_current_price(self, long_account):
        acct, pos = long_account
        executor = Executor(acct)
        # Price moves up but doesn't hit target (120) or stop (90).
        executor.check_positions({"XYZ": 108.0})
        assert pos.current_price == 108.0
        assert acct.total_equity == pytest.approx(4000.0 + 108.0 * 10)

    def test_peak_equity_rises_without_close(self, long_account):
        acct, pos = long_account
        executor = Executor(acct)
        executor.check_positions({"XYZ": 115.0})
        # 4000 + 1150 = 5150. That's the new peak.
        assert acct.peak_equity == pytest.approx(5150.0)
        # Price drops back — peak stays at 5150.
        executor.check_positions({"XYZ": 105.0})
        assert acct.peak_equity == pytest.approx(5150.0)
        assert acct.drawdown_pct == pytest.approx((5150.0 - 5050.0) / 5150.0)

    def test_short_position_market_value_is_mark_sensitive(self):
        acct = VirtualAccount("alpha")
        pos = Position(
            symbol="XYZ", shares=10, entry_price=100.0,
            stop_loss=110.0, target=90.0, direction="SHORT", trade_type="SWING",
        )
        acct.open_position(pos)
        # No mark → equity = cash + cost_basis (no P&L yet)
        assert acct.total_equity == pytest.approx(5000.0)
        pos.current_price = 90.0  # short is up 10% = +$100
        # market_value = cost_basis 1000 + profit (100-90)*10 = 1100
        # equity = cash 4000 + 1100 = 5100
        assert acct.total_equity == pytest.approx(5100.0)
        pos.current_price = 110.0  # short underwater 10% = -$100
        assert acct.total_equity == pytest.approx(4900.0)

    def test_cost_basis_helper_unchanged_for_cash_recalc(self, long_account):
        """_recalculate_account_balances uses cost basis, not mark value —
        cash out on open is always entry_price * shares. Guarding the
        helper we kept for that purpose."""
        acct, pos = long_account
        pos.current_price = 200.0  # mark way up
        assert acct.open_positions_value != acct.open_positions_cost_basis
        assert acct.open_positions_cost_basis == pytest.approx(1000.0)


class TestSectorConcentration:
    def test_sector_cap_blocks_fourth_position_in_sector(self):
        from config.settings import settings
        acct = VirtualAccount("alpha")
        cap = settings.max_same_sector_positions  # 3 by default
        for i in range(cap):
            acct.open_position(Position(
                symbol=f"T{i}", shares=1, entry_price=10.0,
                stop_loss=9.0, target=11.0, direction="LONG",
                trade_type="SWING", sector="semiconductors",
            ))
        allowed, reason = acct.can_open_position(
            cost=10.0, trade_type="SWING", symbol="T4",
            sector="semiconductors",
        )
        assert allowed is False
        assert "semiconductors" in reason

    def test_sector_cap_independent_across_sectors(self):
        acct = VirtualAccount("alpha")
        acct.open_position(Position(
            symbol="SEMI1", shares=1, entry_price=10.0,
            stop_loss=9.0, target=11.0, direction="LONG",
            trade_type="SWING", sector="semiconductors",
        ))
        allowed, _ = acct.can_open_position(
            cost=10.0, trade_type="SWING", symbol="FIN1", sector="financials",
        )
        assert allowed is True

    def test_unknown_sector_is_not_capped(self):
        acct = VirtualAccount("alpha")
        allowed, _ = acct.can_open_position(
            cost=10.0, trade_type="SWING", symbol="AAA", sector=None,
        )
        assert allowed is True
