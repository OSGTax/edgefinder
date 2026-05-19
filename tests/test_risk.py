"""Tests for the centralized risk manager."""

import pytest
from edgefinder.trading.risk import RiskManager
from edgefinder.trading.account import VirtualAccount


class TestRiskManager:
    def test_compute_stop(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        stop = rm.compute_stop(entry_price=200.0)
        assert stop == 160.0  # 200 * (1 - 0.20)

    def test_compute_target(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        target = rm.compute_target(entry_price=200.0)
        assert target == 250.0  # 200 * (1 + 0.25)

    def test_compute_shares(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        acct = VirtualAccount("gambler", starting_capital=5000.0)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0)
        # max_loss = 5000 * 0.10 = 500
        # stop_distance = 200 * 0.20 = 40
        # shares = 500 / 40 = 12
        assert shares == 12

    def test_compute_shares_limited_by_cash(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        # Only $300 cash — can afford 1 share at $200
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0, available_cash=300.0)
        assert shares == 1

    def test_compute_shares_zero_when_no_cash(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0, available_cash=50.0)
        assert shares == 0

    def test_should_stop_out(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        assert rm.should_stop_out(entry_price=200.0, current_price=159.0) is True
        assert rm.should_stop_out(entry_price=200.0, current_price=161.0) is False

    def test_should_take_profit(self):
        rm = RiskManager(risk_pct=0.10, stop_pct=0.20, target_pct=0.25)
        assert rm.should_take_profit(entry_price=200.0, current_price=251.0) is True
        assert rm.should_take_profit(entry_price=200.0, current_price=249.0) is False

    def test_coward_risk_profile(self):
        rm = RiskManager(risk_pct=0.05, stop_pct=0.20, target_pct=0.15)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0)
        # max_loss = 250, stop_distance = 40, shares = 6
        assert shares == 6

    def test_degenerate_risk_profile(self):
        rm = RiskManager(risk_pct=0.20, stop_pct=0.20, target_pct=0.50)
        shares = rm.compute_shares(entry_price=200.0, equity=5000.0)
        # max_loss = 1000, stop_distance = 40, shares = 25
        assert shares == 25
