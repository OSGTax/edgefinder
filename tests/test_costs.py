"""Unit tests for the microcap transaction-cost model (edgefinder/backtest/costs.py).

These are the load-bearing assumptions that keep a microcap backtest honest, so
they are tested tightly: the spread estimator's sign/monotonicity, square-root
impact scaling, and the participation cap that makes thin names untradeable.
"""

from __future__ import annotations

import math

import pytest

from edgefinder.backtest.costs import CostModel, corwin_schultz_spread


class TestCorwinSchultz:
    def test_flat_days_zero_spread(self):
        # No intraday range at all → estimator is exactly 0.
        assert corwin_schultz_spread(100, 100, 100, 100) == 0.0

    def test_degenerate_prices_return_zero(self):
        assert corwin_schultz_spread(0, 0, 0, 0) == 0.0
        assert corwin_schultz_spread(100, -1, 100, 99) == 0.0

    def test_high_intraday_range_no_drift_gives_positive_spread(self):
        # Big daily bounce (5% range) with no net 2-day move = classic
        # bid-ask-bounce signature → a meaningful positive spread estimate.
        s = corwin_schultz_spread(105, 100, 105, 100)
        assert 0.03 < s < 0.07

    def test_spread_monotonic_in_bounce(self):
        narrow = corwin_schultz_spread(102, 100, 102, 100)
        wide = corwin_schultz_spread(110, 100, 110, 100)
        assert wide > narrow >= 0.0

    def test_negative_estimate_clamps_to_zero(self):
        # Mild trending days where the 2-day range dominates the single-day
        # ranges drive the raw estimate negative → clamped to 0, never < 0.
        s = corwin_schultz_spread(101, 100, 108, 107)
        assert s == 0.0


class TestImpact:
    def test_square_root_scaling(self):
        m = CostModel(impact_coef=1.0)
        # participation 0.01, vol 0.05 → 1 * 0.05 * sqrt(0.01) = 0.005
        assert m.impact(1_000, 100_000, 0.05) == pytest.approx(0.005)

    def test_impact_monotonic_in_size_and_vol(self):
        m = CostModel(impact_coef=1.0)
        assert m.impact(2_000, 100_000, 0.05) > m.impact(1_000, 100_000, 0.05)
        assert m.impact(1_000, 100_000, 0.08) > m.impact(1_000, 100_000, 0.05)

    def test_quadrupling_size_doubles_impact(self):
        m = CostModel(impact_coef=1.0)
        small = m.impact(1_000, 100_000, 0.05)
        big = m.impact(4_000, 100_000, 0.05)
        assert big == pytest.approx(2 * small)

    def test_zero_adv_no_impact(self):
        assert CostModel().impact(1_000, 0, 0.05) == 0.0


class TestHalfSpreadAndFloor:
    def test_floor_applies_when_estimate_is_thin(self):
        m = CostModel(spread_floor=0.005)
        # estimate 0 → floored to 0.005, halved to 0.0025
        assert m.half_spread(0.0) == pytest.approx(0.0025)

    def test_estimate_above_floor_used_directly(self):
        m = CostModel(spread_floor=0.005)
        assert m.half_spread(0.04) == pytest.approx(0.02)


class TestFillPrice:
    def test_buy_pays_up_sell_receives_less_symmetrically(self):
        m = CostModel(impact_coef=1.0, spread_floor=0.0)
        kw = dict(order_dollars=1_000, adv_dollars=100_000, volatility=0.05,
                  spread_frac=0.02)
        buy = m.fill_price(100.0, "BUY", **kw)
        sell = m.fill_price(100.0, "SELL", **kw)
        frac = m.cost_fraction(1_000, 100_000, 0.05, 0.02)
        assert buy == pytest.approx(100.0 * (1 + frac))
        assert sell == pytest.approx(100.0 * (1 - frac))
        assert buy > 100.0 > sell

    def test_cost_fraction_is_half_spread_plus_impact(self):
        m = CostModel(impact_coef=1.0, spread_floor=0.0)
        total = m.cost_fraction(1_000, 100_000, 0.05, 0.02)
        assert total == pytest.approx(m.half_spread(0.02) + m.impact(1_000, 100_000, 0.05))


class TestParticipationCap:
    def test_thin_name_below_min_adv_is_untradeable(self):
        m = CostModel(min_adv_dollars=25_000)
        assert m.cap_shares(1_000, price=10.0, adv_dollars=10_000) == 0

    def test_cap_clips_to_participation_limit(self):
        m = CostModel(max_participation=0.05, min_adv_dollars=0)
        # 5% of $100k ADV = $5k; at $10 → 500 shares max
        assert m.cap_shares(1_000, price=10.0, adv_dollars=100_000) == 500

    def test_small_order_passes_through_uncapped(self):
        m = CostModel(max_participation=0.05, min_adv_dollars=0)
        assert m.cap_shares(100, price=10.0, adv_dollars=100_000) == 100

    def test_nonpositive_inputs_return_zero(self):
        m = CostModel()
        assert m.cap_shares(0, 10.0, 100_000) == 0
        assert m.cap_shares(100, 0.0, 100_000) == 0
