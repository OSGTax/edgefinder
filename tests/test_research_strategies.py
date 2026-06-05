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
