"""Unit smoke tests for the 2026-06-05 research candidates.

Pre-registered defaults are exercised exactly as the screen will run them.
"""

from __future__ import annotations

from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.strategies.gap_drift import GapDriftStrategy
from edgefinder.strategies.pullback_rider import PullbackRiderStrategy
from edgefinder.strategies.turtle_adx import TurtleAdxStrategy


def _snap(**kw) -> IndicatorSnapshot:
    s = IndicatorSnapshot()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _md(current: IndicatorSnapshot, history_snaps: list[IndicatorSnapshot]) -> MarketData:
    h = IndicatorHistory(max_days=30)
    for s in history_snaps:
        h.add(s)
    return MarketData(
        ticker="TEST", current=current, history=h, fundamentals=None,
        context=MarketContext(), current_price=current.close,
    )


class TestPullbackRider:
    def _uptrend_cur(self, **over):
        base = dict(close=101.0, ema_21=100.0, ema_50=98.0, ema_200=90.0, rsi=50.0)
        base.update(over)
        return _snap(**base)

    def test_reclaim_in_uptrend_fires(self):
        prev = _snap(close=99.0, ema_21=100.0)
        intent = PullbackRiderStrategy().evaluate("TEST", _md(self._uptrend_cur(), [prev]))
        assert intent is not None and "reclaim" in intent.reasoning.lower()

    def test_no_entry_without_trend_stack(self):
        prev = _snap(close=99.0, ema_21=100.0)
        cur = self._uptrend_cur(ema_50=85.0)  # 50 < 200 → no stack
        assert PullbackRiderStrategy().evaluate("TEST", _md(cur, [prev])) is None

    def test_no_entry_if_dip_still_in_progress(self):
        prev = _snap(close=99.0, ema_21=100.0)
        cur = self._uptrend_cur(close=99.5)  # still below ema_21
        assert PullbackRiderStrategy().evaluate("TEST", _md(cur, [prev])) is None

    def test_exit_on_50ema_loss_and_rsi_strength(self):
        strat = PullbackRiderStrategy()
        cut = _snap(close=97.0, ema_50=98.0, rsi=50.0)
        assert strat.should_exit("TEST", _md(cut, []), 100.0) is not None
        strength = _snap(close=110.0, ema_50=98.0, rsi=75.0)
        assert strat.should_exit("TEST", _md(strength, []), 100.0) is not None
        hold = _snap(close=103.0, ema_50=98.0, rsi=60.0)
        assert strat.should_exit("TEST", _md(hold, []), 100.0) is None


class TestGapDrift:
    def _held_gap_cur(self, **over):
        base = dict(open=106.0, close=107.0, high=107.2, low=105.0,
                    ema_200=90.0, volume_ratio=2.0)
        base.update(over)
        return _snap(**base)

    def test_held_gap_fires(self):
        prev = _snap(close=100.0)
        intent = GapDriftStrategy().evaluate("TEST", _md(self._held_gap_cur(), [prev]))
        assert intent is not None and "gap" in intent.reasoning.lower()

    def test_small_gap_blocked(self):
        prev = _snap(close=103.0)  # ~2.9% gap < 5% default
        assert GapDriftStrategy().evaluate("TEST", _md(self._held_gap_cur(), [prev])) is None

    def test_weak_close_blocked(self):
        prev = _snap(close=100.0)
        cur = self._held_gap_cur(close=105.3, low=105.0, high=108.0)  # bottom of range
        assert GapDriftStrategy().evaluate("TEST", _md(cur, [prev])) is None

    def test_fail_stop_exit(self):
        strat = GapDriftStrategy()
        failed = _snap(close=99.0)  # 106 fill → -6.6% < -6% default
        assert strat.should_exit("TEST", _md(failed, []), 106.0) is not None
        holding = _snap(close=103.0)
        assert strat.should_exit("TEST", _md(holding, []), 106.0) is None


class TestTurtleAdx:
    def _history(self):
        # 26 days: high of 90 mid-window; yesterday (85) NOT a 30d high.
        closes = [80.0] * 10 + [90.0] + [85.0] * 15
        return [_snap(close=c) for c in closes]

    def _breakout_cur(self, **over):
        base = dict(close=91.0, adx=25.0, plus_di=30.0, minus_di=15.0,
                    volume_ratio=1.5, ema_50=80.0, ema_21=86.0)
        base.update(over)
        return _snap(**base)

    def test_breakout_cross_fires(self):
        intent = TurtleAdxStrategy().evaluate(
            "TEST", _md(self._breakout_cur(), self._history()))
        assert intent is not None and "breakout" in intent.reasoning.lower()

    def test_weak_adx_blocked(self):
        cur = self._breakout_cur(adx=15.0)
        assert TurtleAdxStrategy().evaluate("TEST", _md(cur, self._history())) is None

    def test_no_refire_when_yesterday_was_high(self):
        # Yesterday already the 30d high → not a cross event.
        closes = [80.0] * 25 + [90.0]
        hist = [_snap(close=c) for c in closes]
        cur = self._breakout_cur(close=92.0)
        assert TurtleAdxStrategy().evaluate("TEST", _md(cur, hist)) is None

    def test_exit_on_21ema_loss(self):
        strat = TurtleAdxStrategy()
        cur = _snap(close=84.0, ema_21=86.0)
        assert strat.should_exit("TEST", _md(cur, []), 91.0) is not None
        riding = _snap(close=95.0, ema_21=86.0)
        assert strat.should_exit("TEST", _md(riding, []), 91.0) is None


class TestTrendDip:
    def _stretch_md(self, *, closes_hist, cur_close, wr=-95.0, ema_200=90.0, rsi=25.0):
        from edgefinder.strategies.trend_dip import TrendDipStrategy  # noqa: F401
        hist = [_snap(close=c) for c in closes_hist]
        cur = _snap(close=cur_close, ema_200=ema_200, williams_r=wr, rsi=rsi)
        return _md(cur, hist)

    def test_three_day_stretch_in_uptrend_fires(self):
        from edgefinder.strategies.trend_dip import TrendDipStrategy
        md = self._stretch_md(closes_hist=[100.0, 99.0, 98.0], cur_close=97.0)
        intent = TrendDipStrategy().evaluate("TEST", md)
        assert intent is not None and "stretch" in intent.reasoning.lower()

    def test_below_200dma_blocked(self):
        from edgefinder.strategies.trend_dip import TrendDipStrategy
        md = self._stretch_md(closes_hist=[100.0, 99.0, 98.0], cur_close=97.0,
                              ema_200=120.0)  # below trend
        assert TrendDipStrategy().evaluate("TEST", md) is None

    def test_shallow_wr_blocked(self):
        from edgefinder.strategies.trend_dip import TrendDipStrategy
        md = self._stretch_md(closes_hist=[100.0, 99.0, 98.0], cur_close=97.0,
                              wr=-50.0)  # not stretched
        assert TrendDipStrategy().evaluate("TEST", md) is None

    def test_too_few_down_days_blocked(self):
        from edgefinder.strategies.trend_dip import TrendDipStrategy
        md = self._stretch_md(closes_hist=[100.0, 99.0, 99.5], cur_close=97.0)
        assert TrendDipStrategy().evaluate("TEST", md) is None  # only 1 down day

    def test_recovery_exit(self):
        from edgefinder.strategies.trend_dip import TrendDipStrategy
        strat = TrendDipStrategy()
        recovered = _snap(close=102.0, rsi=65.0)
        assert strat.should_exit("TEST", _md(recovered, []), 100.0) is not None
        waiting = _snap(close=98.0, rsi=40.0)
        assert strat.should_exit("TEST", _md(waiting, []), 100.0) is None


class TestGapDriftV2:
    def _cur(self, **over):
        # prev: close 100, ATR 2 → 2.5 ATRs = $5 gap needed (and >= 2% floor)
        base = dict(open=106.0, close=107.0, high=107.2, low=105.0,
                    ema_200=90.0, volume_ratio=2.0)
        base.update(over)
        return _snap(**base)

    def _prev(self, atr=2.0):
        return _snap(close=100.0, atr=atr)

    def test_atr_sized_gap_fires(self):
        from edgefinder.strategies.gap_drift_v2 import GapDriftV2Strategy
        intent = GapDriftV2Strategy().evaluate("TEST", _md(self._cur(), [self._prev()]))
        assert intent is not None and "atr" in intent.reasoning.lower()

    def test_same_gap_blocked_on_volatile_name(self):
        # Same $6 gap, but the stock's normal day is $4 → only 1.5 ATRs < 2.5.
        from edgefinder.strategies.gap_drift_v2 import GapDriftV2Strategy
        md = _md(self._cur(), [self._prev(atr=4.0)])
        assert GapDriftV2Strategy().evaluate("TEST", md) is None

    def test_absolute_floor_blocks_micro_gaps(self):
        # Quiet name: ATR 0.5 → 2.5 ATRs = $1.25 gap = 1.25% < 2% floor.
        from edgefinder.strategies.gap_drift_v2 import GapDriftV2Strategy
        cur = self._cur(open=101.3, close=101.8, high=101.9, low=101.0)
        md = _md(cur, [self._prev(atr=0.5)])
        assert GapDriftV2Strategy().evaluate("TEST", md) is None


class TestGapCarry:
    """gap_carry: gap_drift v1's entry FIXED, exits = fail-stop + 21EMA trail."""

    def _cur(self, **over):
        base = dict(open=106.0, close=107.0, high=107.2, low=105.0,
                    ema_200=90.0, volume_ratio=2.0)
        base.update(over)
        return _snap(**base)

    def test_held_gap_fires(self):
        from edgefinder.strategies.gap_carry import GapCarryStrategy
        prev = _snap(close=100.0)
        intent = GapCarryStrategy().evaluate("TEST", _md(self._cur(), [prev]))
        assert intent is not None and "gap" in intent.reasoning.lower()

    def test_entry_identical_to_v1(self):
        # The whole design: any screen difference is attributable to exits.
        from edgefinder.strategies.gap_carry import GapCarryStrategy
        for prev_close, cur_kw in [
            (103.0, {}),                                    # small gap
            (100.0, dict(close=105.3, low=105.0, high=108.0)),  # weak close
            (100.0, dict(volume_ratio=1.2)),                # quiet volume
            (100.0, dict(ema_200=120.0)),                   # below 200dma
        ]:
            md = _md(self._cur(**cur_kw), [_snap(close=prev_close)])
            assert GapDriftStrategy().evaluate("TEST", md) is None
            assert GapCarryStrategy().evaluate("TEST", md) is None

    def test_fail_stop_exit(self):
        from edgefinder.strategies.gap_carry import GapCarryStrategy
        strat = GapCarryStrategy()
        failed = _snap(close=99.0, ema_21=95.0)  # 106 fill → -6.6% < -6% default
        assert strat.should_exit("TEST", _md(failed, []), 106.0) is not None

    def test_ema21_trail_exit(self):
        from edgefinder.strategies.gap_carry import GapCarryStrategy
        strat = GapCarryStrategy()
        lost = _snap(close=110.0, ema_21=112.0)  # well above fail-stop, lost trail
        exit_intent = strat.should_exit("TEST", _md(lost, []), 106.0)
        assert exit_intent is not None and "drift over" in exit_intent.reasoning.lower()
        riding = _snap(close=120.0, ema_21=112.0)
        assert strat.should_exit("TEST", _md(riding, []), 106.0) is None

    def test_wide_target_and_long_hold(self):
        from edgefinder.strategies.gap_carry import GapCarryStrategy
        strat = GapCarryStrategy()
        assert strat.target_pct == 0.30
        assert strat.max_hold_days == 45
