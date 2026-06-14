"""Tests for the PRE-REGISTERED intraday hunt ROUND 2 roster
(intraday_roster_r2.py) plus the engine's new no-trade rebalance band.

All offline: synthetic IntradayContext fixtures built directly (no R2, no DB).
The Round-2 contract under test for EVERY spec: it builds, decide() returns
non-negative weights summing to <= 1 over symbols that exist in the context, and
returns {} on the last decision bar. Plus the ENTER-ONCE property: the basket is
STABLE across consecutive decision bars when the (fixed) signal inputs are
unchanged. And the engine band tests: band=0.01 generates ~2 trades/day for an
enter-once strategy; band never blocks the opening buy or MOC flatten; band=0.0
reproduces the no-band result bit-for-bit.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edgefinder.engine.intraday_backtest import run_intraday_backtest
from edgefinder.engine.intraday_roster_r2 import (
    INTRADAY_R2_SPECS,
    IroGapFade,
    IroGapGo,
    IroMorningTrend,
    IroOrb,
    IroRandomHold,
)
from edgefinder.engine.intraday_strategy import (
    BuyHoldFromOpen,
    IntradayAssetView,
    IntradayContext,
)

ET = ZoneInfo("America/New_York")


# ── synthetic AssetView / context builders (mirror test_intraday_roster) ──


def _make_view(symbol, *, opens, highs, lows, closes, volumes=None, i=None,
               session_start=0, prev_close=None):
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
    arr = [price] * n
    return _make_view(symbol, opens=arr, highs=[p * 1.001 for p in arr],
                      lows=[p * 0.999 for p in arr], closes=arr,
                      prev_close=prev_close)


def _trend_view(symbol, n=40, start=100.0, slope=0.0, prev_close=None):
    closes = [start * (1 + slope * k) for k in range(n)]
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * 1.0005 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.9995 for o, c in zip(opens, closes)]
    return _make_view(symbol, opens=opens, highs=highs, lows=lows,
                      closes=closes, prev_close=prev_close)


def _diverse_assets(prev_close=None):
    return {
        "UP1": _trend_view("UP1", slope=0.001, prev_close=prev_close),
        "UP2": _trend_view("UP2", slope=0.0008, prev_close=prev_close),
        "DN1": _trend_view("DN1", slope=-0.001, prev_close=prev_close),
        "DN2": _trend_view("DN2", slope=-0.0008, prev_close=prev_close),
        "FLAT": _flat_view("FLAT", prev_close=prev_close),
        "GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0, prev_close=100.0),
        "GAPUP": _trend_view("GAPUP", start=110.0, slope=0.0005, prev_close=100.0),
    }


def _view_at(view: IntradayAssetView, i: int) -> IntradayAssetView:
    """Same arrays/session_start as ``view`` but with the decision index moved
    to ``i`` — lets us simulate the SAME asset at a later decision bar."""
    return IntradayAssetView(
        symbol=view.symbol, _o=view._o, _h=view._h, _l=view._l, _c=view._c,
        _v=view._v, _ts=view._ts, i=i, session_start=view.session_start)


# ── universal contract: every R2 spec ────────────────────────────────────


class TestUniversalContract:
    def test_every_spec_builds_and_returns_valid_weights(self):
        for name, factory in INTRADAY_R2_SPECS.items():
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
        for name, factory in INTRADAY_R2_SPECS.items():
            strat = factory()
            assets = _diverse_assets()
            ctx = _ctx(assets, bars_since_open=35, bars_until_close=0,
                       is_last_decision_bar=True)
            assert strat.decide(ctx) == {}, f"{name}: not flat at close"

    def test_every_spec_handles_empty_universe(self):
        for name, factory in INTRADAY_R2_SPECS.items():
            strat = factory()
            ctx = _ctx({}, bars_since_open=35, bars_until_close=30)
            assert strat.decide(ctx) == {}, f"{name}: nonempty on empty universe"

    def test_factory_wiring_resolves_r2_specs(self):
        from edgefinder.engine.intraday_validate import make_intraday_factory
        for name in INTRADAY_R2_SPECS:
            strat = make_intraday_factory(name)()
            assert strat.name == name


# ── GAP: only gap-downs / gap-ups qualify, and the basket is STABLE ──────


class TestGapFade:
    def test_only_gap_downs_qualify(self):
        assets = {
            "GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0, prev_close=100.0),
            "GAPUP": _trend_view("GAPUP", start=110.0, slope=0.0, prev_close=100.0),
            "FLAT": _flat_view("FLAT", price=100.0, prev_close=100.0),
        }
        w = IroGapFade().decide(_ctx(assets, bars_since_open=5))
        assert "GAPDN" in w
        assert "GAPUP" not in w
        assert "FLAT" not in w

    def test_no_time_gate_holds_late_in_session(self):
        # Round-1 GapFade went cash after 30 bars; the enter-once version holds.
        assets = {"GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0,
                                       prev_close=100.0)}
        late = _ctx(assets, bars_since_open=300)
        assert "GAPDN" in IroGapFade().decide(late)

    def test_basket_stable_across_consecutive_bars(self):
        # the ENTER-ONCE property: session_open/prev_close are fixed, so the
        # same basket comes back at bar t and bar t+1.
        names = {f"GD{k}": _trend_view(f"GD{k}", start=100.0 - k, slope=0.0,
                                       prev_close=100.0) for k in range(1, 8)}
        # bar t (early) vs bar t+1 (later) — move the decision index forward.
        ctx_t = _ctx(names, bars_since_open=5)
        later = {s: _view_at(v, v.i) for s, v in names.items()}  # same i (flat trend)
        ctx_t1 = _ctx(later, bars_since_open=6)
        a = IroGapFade().decide(ctx_t)
        b = IroGapFade().decide(ctx_t1)
        assert set(a) == set(b) and a != {}


class TestGapGo:
    def test_only_gap_ups_qualify(self):
        assets = {
            "GAPDN": _trend_view("GAPDN", start=90.0, slope=0.0, prev_close=100.0),
            "GAPUP": _trend_view("GAPUP", start=110.0, slope=0.0, prev_close=100.0),
            "FLAT": _flat_view("FLAT", price=100.0, prev_close=100.0),
        }
        w = IroGapGo().decide(_ctx(assets, bars_since_open=5))
        assert "GAPUP" in w
        assert "GAPDN" not in w
        assert "FLAT" not in w

    def test_holds_even_when_price_falls_below_open(self):
        # the dropped price>open gate: a gapper that fades still stays held
        # (the gap is fixed; only the OPEN matters).
        closes = [110.0] + [105.0] * 39   # gapped up, then faded below open? open=110
        opens = [closes[0]] + closes[:-1]
        highs = [max(o, c) for o, c in zip(opens, closes)]
        lows = [min(o, c) for o, c in zip(opens, closes)]
        faded = _make_view("FADE", opens=opens, highs=highs, lows=lows,
                           closes=closes, prev_close=100.0)
        w = IroGapGo().decide(_ctx({"FADE": faded}, bars_since_open=20))
        assert "FADE" in w   # gap was +10% vs prev_close; held despite the fade


# ── BREAKOUT: ORB membership is MONOTONE (once in, stays in) ─────────────


class TestOrb:
    def test_no_action_before_range_forms(self):
        assets = {"A": _trend_view("A", slope=0.001)}
        assert IroOrb().decide(_ctx(assets, bars_since_open=10)) == {}

    def test_fires_after_breakout(self):
        n = 40
        closes = [100.0 + (k % 2) for k in range(30)] + [105.0] * (n - 30)
        opens = [closes[0]] + closes[:-1]
        highs = [max(o, c) for o, c in zip(opens, closes)]
        lows = [min(o, c) for o, c in zip(opens, closes)]
        brk = _make_view("BRK", opens=opens, highs=highs, lows=lows, closes=closes)
        flat = _flat_view("FLAT", n=n, price=100.5)
        w = IroOrb().decide(_ctx({"BRK": brk, "FLAT": flat}, bars_since_open=n))
        assert "BRK" in w
        assert "FLAT" not in w

    def test_membership_monotone_stays_in_after_pullback(self):
        # a name breaks out at bar 30 (high 105), then price PULLS BACK below
        # the OR high. Because session_high is monotone, it STAYS in the set.
        n = 40
        closes = ([100.0 + (k % 2) for k in range(30)]
                  + [105.0]                       # breakout bar
                  + [100.5] * (n - 31))           # pulls back below OR high
        opens = [closes[0]] + closes[:-1]
        highs = [max(o, c) for o, c in zip(opens, closes)]
        lows = [min(o, c) for o, c in zip(opens, closes)]
        brk = _make_view("BRK", opens=opens, highs=highs, lows=lows, closes=closes)
        # at the breakout bar (i=30): in the set
        at_break = IroOrb().decide(_ctx({"BRK": _view_at(brk, 30)},
                                        bars_since_open=31))
        # later, after the pullback (i=38): still in the set (monotone)
        after = IroOrb().decide(_ctx({"BRK": _view_at(brk, 38)},
                                     bars_since_open=39))
        assert "BRK" in at_break
        assert "BRK" in after   # the enter-once / monotone guarantee


# ── MORNING-TREND: ranking is FIXED at the m-th bar (stable hold) ────────


class TestMorningTrend:
    def test_no_action_before_window_forms(self):
        assets = {"A": _trend_view("A", slope=0.001)}
        assert IroMorningTrend().decide(_ctx(assets, bars_since_open=10)) == {}

    def test_picks_highest_morning_return(self):
        assets = {
            "WIN1": _trend_view("WIN1", slope=0.003),
            "WIN2": _trend_view("WIN2", slope=0.0025),
            "WIN3": _trend_view("WIN3", slope=0.002),
            "WIN4": _trend_view("WIN4", slope=0.0015),
            "WIN5": _trend_view("WIN5", slope=0.001),
            "LOSE": _trend_view("LOSE", slope=-0.002),
            "WORST": _trend_view("WORST", slope=-0.003),
        }
        w = IroMorningTrend().decide(_ctx(assets, bars_since_open=35))
        assert "WIN1" in w
        assert "LOSE" not in w and "WORST" not in w

    def test_ranking_stable_after_window(self):
        # the m-th bar close is fixed once it exists; the basket at bar 35 and
        # bar 39 must match even though current price keeps moving.
        assets = {
            "A": _trend_view("A", slope=0.003),
            "B": _trend_view("B", slope=0.002),
            "C": _trend_view("C", slope=0.001),
            "D": _trend_view("D", slope=0.0005),
            "E": _trend_view("E", slope=0.0003),
            "F": _trend_view("F", slope=-0.001),
        }
        at_35 = IroMorningTrend().decide(
            _ctx({s: _view_at(v, 35) for s, v in assets.items()},
                 bars_since_open=36))
        at_39 = IroMorningTrend().decide(
            _ctx({s: _view_at(v, 39) for s, v in assets.items()},
                 bars_since_open=40))
        assert set(at_35) == set(at_39) and at_35 != {}


# ── CONTROLS: deterministic per (seed, date), held all session ──────────


class TestRandomHold:
    def test_deterministic_per_seed_and_date(self):
        assets = _diverse_assets()
        ctx = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 5))
        a = IroRandomHold(201).decide(ctx)
        b = IroRandomHold(201).decide(ctx)
        assert a == b and a != {}

    def test_seed_dependent(self):
        assets = {f"S{k:02d}": _flat_view(f"S{k:02d}") for k in range(30)}
        ctx = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 5))
        assert set(IroRandomHold(201).decide(ctx)) != \
            set(IroRandomHold(203).decide(ctx))

    def test_stable_across_bars_same_date(self):
        # the enter-once property for the control: same date => same basket.
        assets = {f"S{k:02d}": _flat_view(f"S{k:02d}") for k in range(30)}
        c1 = _ctx(assets, bars_since_open=20, session_date=date(2024, 3, 5))
        c2 = _ctx(assets, bars_since_open=200, session_date=date(2024, 3, 5))
        assert IroRandomHold(201).decide(c1) == IroRandomHold(201).decide(c2)

    def test_flattens_at_close(self):
        assets = _diverse_assets()
        ctx = _ctx(assets, bars_since_open=20, is_last_decision_bar=True)
        assert IroRandomHold(201).decide(ctx) == {}


# ════ ENGINE: the no-trade rebalance band (Step 1) ════════════════════════


def _session_ts(d: date, n_bars: int) -> list[int]:
    base = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
    return [int(base.timestamp()) + 60 * k for k in range(n_bars)]


def _us_trading_days(start: date, count: int) -> list[date]:
    out, d = [], start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    return out


def _drift_session(symbols, days, *, bars_per_day=30, start=date(2024, 1, 2),
                   drift_per_day=0.0, drifts=None):
    """Deterministic RTH minute frames with a linear open->close drift (so the
    equity mark wanders within the session — the conditions under which a
    constant-target strategy would integer-share churn WITHOUT a band).

    ``drifts`` (optional {symbol: drift}) overrides ``drift_per_day`` per name;
    DIVERGENT per-symbol drifts pull an equal-weight basket's weights apart
    intra-session, which is exactly what forces an exact-re-true engine to churn
    each name back toward equal weight — and what the band suppresses."""
    out: dict[str, pd.DataFrame] = {}
    for si, sym in enumerate(symbols):
        drift_per_day = (drifts or {}).get(sym, drift_per_day)
        rows, price = [], 100.0 * (1 + 0.01 * si)
        for d in _us_trading_days(start, days):
            ts = _session_ts(d, bars_per_day)
            day_open = price
            prev_close = day_open
            for bi in range(bars_per_day):
                frac = bi / max(1, bars_per_day - 1)
                c = day_open * (1 + drift_per_day * frac)
                o = day_open if bi == 0 else prev_close
                h = max(o, c) * 1.0005
                l = min(o, c) * 0.9995
                rows.append((ts[bi], o, h, l, c, 1_000_000.0))
                prev_close = c
            price = prev_close
        out[sym] = pd.DataFrame(
            rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return out


class _ConstantBasket:
    """Enter-once: returns the SAME stable target basket every decision bar
    (until the engine flattens at close)."""

    def __init__(self, symbols):
        self.basket = {s: 1.0 / len(symbols) for s in symbols}

    name = "const_basket"

    def decide(self, ctx):
        if ctx.is_last_decision_bar:
            return {}
        return {s: w for s, w in self.basket.items() if s in ctx.assets}


class TestRebalanceBand:
    def test_band_cuts_churn_to_about_two_trades_per_day(self):
        # WITH a band, a constant-target enter-once strategy trades ~2/day
        # (entry + MOC flatten); WITHOUT it, DIVERGENT per-name drift pulls the
        # equal-weight basket apart and the exact-re-true engine churns each
        # name back to weight every bar.
        syms = ["A", "B", "C", "D", "E"]
        # divergent drifts: some rip up, some fall -> weights diverge intra-day
        drifts = {"A": 0.20, "B": 0.10, "C": 0.0, "D": -0.10, "E": -0.20}
        bars = _drift_session(syms, 3, bars_per_day=40, drifts=drifts)
        no_band = run_intraday_backtest(
            {k: v.copy() for k, v in bars.items()}, _ConstantBasket(syms),
            cost_bps=0.0, flatten_at_close=True, rebalance_band=0.0)
        with_band = run_intraday_backtest(
            {k: v.copy() for k, v in bars.items()}, _ConstantBasket(syms),
            cost_bps=0.0, flatten_at_close=True, rebalance_band=0.01)
        # the band cuts churn DRAMATICALLY (the exact-re-true engine churns
        # ~every bar; the band re-trues only on >=1% drift).
        assert with_band.stats["num_trades"] < no_band.stats["num_trades"] / 2

    def test_band_holds_a_stable_basket_to_about_two_tolls_per_day(self):
        # a flat session (no drift -> no >=1% weight drift) is the pure
        # enter-once case: with the band the basket holds, costing exactly the
        # entry + MOC flatten (~2 tolls/day per name), with NO mid-session
        # re-trues; without the band, integer-share dust re-trues sneak in.
        syms = ["A", "B", "C", "D", "E"]
        bars = _drift_session(syms, 3, bars_per_day=40, drift_per_day=0.0)
        with_band = run_intraday_backtest(
            {k: v.copy() for k, v in bars.items()}, _ConstantBasket(syms),
            cost_bps=0.0, flatten_at_close=True, rebalance_band=0.01)
        # exactly entry (<=5 buys) + MOC flatten (<=5 sells) per session.
        assert with_band.stats["avg_trades_per_day"] <= 2.0 * len(syms) + 1e-9
        buys = [t for t in with_band.trades if t["side"] == "BUY"]
        mocs = [t for t in with_band.trades if t.get("reason") == "MOC_FLATTEN"]
        # no SELLs other than the MOC flatten -> no mid-session re-true churn.
        non_moc_sells = [t for t in with_band.trades
                         if t["side"] == "SELL" and t.get("reason") != "MOC_FLATTEN"]
        assert non_moc_sells == []
        assert len(buys) == 5 * 3      # 5 names entered each of 3 sessions
        assert len(mocs) == 5 * 3      # 5 names flattened each session

    def test_band_never_blocks_opening_buy_or_moc_flatten(self):
        # even with a large band, the opening entry and the closing flatten
        # must always fire (opens_or_closes is exempt from the band).
        syms = ["A", "B"]
        bars = _drift_session(syms, 2, bars_per_day=20, drift_per_day=0.0)
        res = run_intraday_backtest(
            bars, _ConstantBasket(syms), cost_bps=0.0, flatten_at_close=True,
            rebalance_band=0.5)   # huge band — would block everything but open/close
        buys = [t for t in res.trades if t["side"] == "BUY"]
        mocs = [t for t in res.trades if t.get("reason") == "MOC_FLATTEN"]
        # one entry per name per session (2 names * 2 sessions = 4 buys)
        assert len(buys) == 4
        # one MOC flatten per held name per session
        assert len(mocs) == 4
        assert res.stats["open_positions"] == 0

    def test_band_zero_reproduces_no_band_result_bit_for_bit(self):
        # the DEFAULT (0.0) must preserve every prior result exactly — run a
        # known strategy with the explicit default vs the implicit default.
        bars = _drift_session(["AAA"], 4, bars_per_day=30, drift_per_day=0.01)
        from edgefinder.backtest.costs import CostModel
        implicit = run_intraday_backtest(
            {k: v.copy() for k, v in bars.items()}, BuyHoldFromOpen("AAA"),
            cost_model=CostModel(), flatten_at_close=True)
        explicit_zero = run_intraday_backtest(
            {k: v.copy() for k, v in bars.items()}, BuyHoldFromOpen("AAA"),
            cost_model=CostModel(), flatten_at_close=True, rebalance_band=0.0)
        assert implicit.daily_equity_curve == explicit_zero.daily_equity_curve
        assert implicit.trades == explicit_zero.trades
        assert implicit.stats == explicit_zero.stats

    def test_band_threads_through_walkforward(self):
        from edgefinder.engine.intraday_walkforward import run_intraday_walkforward
        bars = _drift_session(["AAA", "SPY"], 120, bars_per_day=20,
                              drift_per_day=0.001)
        card = run_intraday_walkforward(
            {"AAA": bars["AAA"]}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=bars["SPY"], is_days=50, oos_days=21, step_days=21,
            warmup_days=3, rebalance_band=0.01)
        assert card["config"]["rebalance_band"] == 0.01
