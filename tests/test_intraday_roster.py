"""Tests for the PRE-REGISTERED intraday hunt roster (intraday_roster.py).

All offline: synthetic IntradayContext fixtures built directly (no engine, no
R2, no DB). The contract under test for EVERY spec: it builds, decide() returns
non-negative weights summing to <= 1 over symbols that exist in the context, and
returns {} on the last decision bar (let the engine flatten). Plus per-family
behavior checks on constructed data, and RandomBasket determinism.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from edgefinder.engine.intraday_roster import (
    INTRADAY_R1_SPECS,
    GapFade,
    HighBreak,
    IntradayMomentum,
    IntradayReversal,
    OpeningRangeBreakout,
    RandomBasket,
)
from edgefinder.engine.intraday_strategy import (
    IntradayAssetView,
    IntradayContext,
)


# ── synthetic context builder ───────────────────────────────────────────


def _make_view(symbol, *, opens, highs, lows, closes, volumes=None, i=None,
               session_start=0, prev_close=None):
    """Build one IntradayAssetView from explicit OHLC arrays.

    ``prev_close`` (if given) is prepended as a bar BEFORE session_start so
    ``view.prev_close`` returns it (and session_start shifts to 1)."""
    o = list(opens)
    h = list(highs)
    l = list(lows)
    c = list(closes)
    ss = session_start
    if prev_close is not None:
        o = [prev_close] + o
        h = [prev_close] + h
        l = [prev_close] + l
        c = [prev_close] + c
        ss = session_start + 1
    n = len(c)
    if volumes is None:
        v = np.ones(n, dtype=float)
    else:
        vv = list(volumes)
        if prev_close is not None:
            vv = [1.0] + vv
        v = np.array(vv, dtype=float)
    if i is None:
        i = n - 1
    return IntradayAssetView(
        symbol=symbol,
        _o=np.array(o, dtype=float), _h=np.array(h, dtype=float),
        _l=np.array(l, dtype=float), _c=np.array(c, dtype=float),
        _v=v, _ts=np.arange(n, dtype=np.int64), i=i, session_start=ss)


def _ctx(assets, *, bars_since_open, bars_until_close=200,
         is_last_decision_bar=False, session_date=date(2024, 1, 2),
         minute_of_day=9 * 60 + 30):
    return IntradayContext(
        ts=0, session_date=session_date, minute_of_day=minute_of_day,
        bars_since_open=bars_since_open, bars_until_close=bars_until_close,
        assets=assets, is_last_decision_bar=is_last_decision_bar)


def _flat_view(symbol, n=40, price=100.0, prev_close=None):
    """A featureless flat session of ``n`` bars at ``price``."""
    arr = [price] * n
    return _make_view(symbol, opens=arr, highs=[p * 1.001 for p in arr],
                      lows=[p * 0.999 for p in arr], closes=arr,
                      prev_close=prev_close)


def _trend_view(symbol, n=40, start=100.0, slope=0.0, prev_close=None):
    """A monotone session: close[k] = start*(1+slope*k)."""
    closes = [start * (1 + slope * k) for k in range(n)]
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * 1.0005 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.9995 for o, c in zip(opens, closes)]
    return _make_view(symbol, opens=opens, highs=highs, lows=lows,
                      closes=closes, prev_close=prev_close)


# ── universal contract: every spec ──────────────────────────────────────


def _diverse_assets(prev_close=None):
    """A mix of risers, fallers, flats so every family has candidates."""
    return {
        "UP1": _trend_view("UP1", slope=0.001, prev_close=prev_close),
        "UP2": _trend_view("UP2", slope=0.0008, prev_close=prev_close),
        "DN1": _trend_view("DN1", slope=-0.001, prev_close=prev_close),
        "DN2": _trend_view("DN2", slope=-0.0008, prev_close=prev_close),
        "FLAT": _flat_view("FLAT", prev_close=prev_close),
        "GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0,
                             prev_close=100.0),
        "GAPUP": _trend_view("GAPUP", start=110.0, slope=0.0005,
                             prev_close=100.0),
    }


class TestUniversalContract:
    def test_every_spec_builds_and_returns_valid_weights(self):
        for name, factory in INTRADAY_R1_SPECS.items():
            strat = factory()
            assert strat.name == name, f"{name}: name mismatch ({strat.name})"
            assets = _diverse_assets()
            ctx = _ctx(assets, bars_since_open=35, bars_until_close=30)
            w = strat.decide(ctx)
            assert isinstance(w, dict), name
            assert all(v >= 0 for v in w.values()), f"{name}: negative weight"
            assert sum(w.values()) <= 1.0 + 1e-9, f"{name}: weights sum > 1"
            assert all(s in assets for s in w), f"{name}: phantom symbol"

    def test_every_spec_flat_on_last_decision_bar(self):
        for name, factory in INTRADAY_R1_SPECS.items():
            strat = factory()
            assets = _diverse_assets()
            ctx = _ctx(assets, bars_since_open=35, bars_until_close=0,
                       is_last_decision_bar=True)
            assert strat.decide(ctx) == {}, f"{name}: not flat at close"

    def test_every_spec_handles_empty_universe(self):
        for name, factory in INTRADAY_R1_SPECS.items():
            strat = factory()
            ctx = _ctx({}, bars_since_open=35, bars_until_close=30)
            assert strat.decide(ctx) == {}, f"{name}: nonempty on empty universe"


# ── REVERSAL vs MOMENTUM: opposite picks on the same data ────────────────


class TestReversalVsMomentum:
    def test_reversal_picks_losers_momentum_picks_winners(self):
        # 6 names with distinct trailing-30 returns
        assets = {
            "BIGUP": _trend_view("BIGUP", slope=0.002),
            "UP": _trend_view("UP", slope=0.001),
            "MID": _trend_view("MID", slope=0.0),
            "DN": _trend_view("DN", slope=-0.001),
            "BIGDN": _trend_view("BIGDN", slope=-0.002),
            "WORST": _trend_view("WORST", slope=-0.003),
        }
        ctx = _ctx(assets, bars_since_open=35)
        rev = IntradayReversal().decide(ctx)
        mom = IntradayMomentum().decide(ctx)
        # reversal must hold the worst loser; momentum must hold the best winner
        assert "WORST" in rev and "BIGUP" not in rev
        assert "BIGUP" in mom and "WORST" not in mom
        # and they are genuinely opposite at the extremes
        assert "BIGUP" not in rev
        assert "WORST" not in mom


# ── BREAKOUT: only after the range forms ─────────────────────────────────


class TestORB:
    def test_no_action_before_range_forms(self):
        assets = {"A": _trend_view("A", slope=0.001)}
        # bars_since_open < m (30)
        ctx = _ctx(assets, bars_since_open=10)
        assert OpeningRangeBreakout().decide(ctx) == {}

    def test_fires_on_break_above_opening_range(self):
        # build a name that breaks above its opening-range high late in session
        n = 40
        # first 30 bars range ~ [100, 101]; then a clear break up to 105
        closes = [100.0 + (k % 2) for k in range(30)] + [105.0] * (n - 30)
        opens = [closes[0]] + closes[:-1]
        highs = [max(o, c) for o, c in zip(opens, closes)]
        lows = [min(o, c) for o, c in zip(opens, closes)]
        breaker = _make_view("BRK", opens=opens, highs=highs, lows=lows,
                             closes=closes)
        flat = _flat_view("FLAT", n=n, price=100.5)
        ctx = _ctx({"BRK": breaker, "FLAT": flat}, bars_since_open=n)
        w = OpeningRangeBreakout().decide(ctx)
        assert "BRK" in w        # broke above OR high
        assert "FLAT" not in w   # stayed inside its range


# ── GAP: only fires on gap-downs (fade) ──────────────────────────────────


class TestGapFade:
    def test_only_gap_downs_qualify(self):
        assets = {
            "GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0,
                                 prev_close=100.0),    # -10% gap
            "GAPUP": _trend_view("GAPUP", start=110.0, slope=0.0,
                                 prev_close=100.0),    # +10% gap (must NOT qualify)
            "FLAT": _flat_view("FLAT", price=100.0, prev_close=100.0),
        }
        ctx = _ctx(assets, bars_since_open=5)   # early session
        w = GapFade().decide(ctx)
        assert "GAPDN" in w
        assert "GAPUP" not in w
        assert "FLAT" not in w

    def test_goes_cash_after_first_half_hour(self):
        assets = {"GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0,
                                       prev_close=100.0)}
        late = _ctx(assets, bars_since_open=45)   # > 30
        assert GapFade().decide(late) == {}


# ── HighBreak needs a settled session ────────────────────────────────────


class TestHighBreak:
    def test_waits_for_session_to_settle(self):
        assets = {"UP": _trend_view("UP", slope=0.001)}
        early = _ctx(assets, bars_since_open=5)   # < 10
        assert HighBreak().decide(early) == {}


# ── CLOSE-EFFECT: only last hour ─────────────────────────────────────────


class TestLateDayMomentum:
    def test_silent_outside_last_hour(self):
        from edgefinder.engine.intraday_roster import LateDayMomentum
        assets = _diverse_assets()
        ctx = _ctx(assets, bars_since_open=35, bars_until_close=120)
        assert LateDayMomentum().decide(ctx) == {}

    def test_active_in_last_hour_picks_day_winners(self):
        from edgefinder.engine.intraday_roster import LateDayMomentum
        # > K=5 names so the worst day-returns are genuinely excluded
        assets = {
            "WIN1": _trend_view("WIN1", slope=0.003),
            "WIN2": _trend_view("WIN2", slope=0.0025),
            "WIN3": _trend_view("WIN3", slope=0.002),
            "WIN4": _trend_view("WIN4", slope=0.0015),
            "WIN5": _trend_view("WIN5", slope=0.001),
            "LOSE": _trend_view("LOSE", slope=-0.002),
            "WORST": _trend_view("WORST", slope=-0.003),
        }
        ctx = _ctx(assets, bars_since_open=35, bars_until_close=30)
        w = LateDayMomentum().decide(ctx)
        assert "WIN1" in w
        assert "LOSE" not in w and "WORST" not in w


# ── CONTROLS: deterministic + seed-dependent ─────────────────────────────


class TestRandomBasket:
    def test_deterministic_per_seed_and_date(self):
        assets = _diverse_assets()
        ctx = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 5))
        a = RandomBasket(101).decide(ctx)
        b = RandomBasket(101).decide(ctx)
        assert a == b and a != {}

    def test_seed_dependent(self):
        # a universe big enough that two seeds pick different baskets
        assets = {f"S{k:02d}": _flat_view(f"S{k:02d}") for k in range(30)}
        ctx = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 5))
        a = RandomBasket(101).decide(ctx)
        b = RandomBasket(103).decide(ctx)
        assert set(a) != set(b)

    def test_date_dependent(self):
        assets = {f"S{k:02d}": _flat_view(f"S{k:02d}") for k in range(30)}
        c1 = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 5))
        c2 = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 6))
        assert set(RandomBasket(101).decide(c1)) != \
            set(RandomBasket(101).decide(c2))

    def test_flattens_at_close(self):
        assets = _diverse_assets()
        ctx = _ctx(assets, bars_since_open=20, is_last_decision_bar=True)
        assert RandomBasket(101).decide(ctx) == {}


# ── factory wiring ───────────────────────────────────────────────────────


class TestFactoryWiring:
    def test_make_intraday_factory_resolves_roster_specs(self):
        from edgefinder.engine.intraday_validate import make_intraday_factory
        for name in INTRADAY_R1_SPECS:
            strat = make_intraday_factory(name)()
            assert strat.name == name

    def test_unknown_spec_mentions_roster(self):
        from edgefinder.engine.intraday_validate import make_intraday_factory
        import pytest
        with pytest.raises(ValueError) as ei:
            make_intraday_factory("definitely_not_a_spec")
        assert "ir_reversal" in str(ei.value)
