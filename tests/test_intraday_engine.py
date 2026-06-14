"""Tests for the intraday (minute-bar) backtest engine + walk-forward + CLI.

All offline: synthetic deterministic RTH minute frames, no network, no R2, no DB
(except sqlite for the record test). The honesty invariants — next-bar fills,
flatten-at-close MOC, look-ahead guards, O(1) history access — are the contract
under test, mirroring tests/test_engine.py for the daily lane.
"""

from __future__ import annotations

import time as _time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edgefinder.db.models import ValidationRun  # registers the table on Base
from edgefinder.engine.intraday_backtest import (
    IntradayBacktestResult,
    run_intraday_backtest,
)
from edgefinder.engine.intraday_strategy import (
    BuyHoldFromOpen,
    IntradayContext,
    IntradayFlat,
    IntradayMeanReversion,
)
from edgefinder.engine.intraday_walkforward import run_intraday_walkforward

ET = ZoneInfo("America/New_York")


# ── synthetic minute-frame helper ──────────────────────────────────────


def _session_ts(d: date, n_bars: int) -> list[int]:
    """``n_bars`` consecutive 1-min UTC-epoch ts starting 09:30 ET on ``d``."""
    base = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
    base_epoch = int(base.timestamp())
    return [base_epoch + 60 * k for k in range(n_bars)]


def _us_trading_days(start: date, count: int) -> list[date]:
    """``count`` weekday dates from ``start`` (good enough — no holidays in the
    synthetic windows)."""
    out: list[date] = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    return out


def _synthetic_session(symbols, days, *, bars_per_day=390, start=date(2024, 1, 2),
                       price0=100.0, drift_per_day=0.0, intrabar=None, seed=0):
    """Deterministic RTH minute frames {symbol: DataFrame[ts,o,h,l,c,v]}.

    ``drift_per_day``: fractional open->close move applied LINEARLY across each
    session (so the day's open->close return is exactly drift_per_day).
    ``intrabar(day_i, bar_i, base) -> (o,h,l,c)`` optional override for crafting
    specific shapes (gaps, dips). ``v`` is constant (deterministic costs)."""
    rng = np.random.default_rng(seed)
    trading_days = _us_trading_days(start, days)
    out: dict[str, pd.DataFrame] = {}
    for si, sym in enumerate(symbols):
        rows = []
        price = price0 * (1 + 0.001 * si)   # tiny per-symbol offset
        for di, d in enumerate(trading_days):
            ts = _session_ts(d, bars_per_day)
            day_open = price
            for bi in range(bars_per_day):
                frac = bi / max(1, bars_per_day - 1)
                close = day_open * (1 + drift_per_day * frac)
                o = day_open if bi == 0 else prev_close
                c = close
                h = max(o, c) * 1.0005
                l = min(o, c) * 0.9995
                if intrabar is not None:
                    ov = intrabar(di, bi, day_open)
                    if ov is not None:
                        o, h, l, c = ov
                rows.append((ts[bi], o, h, l, c, 1_000_000.0))
                prev_close = c
            price = prev_close * 1.0   # carry close to next session open (no gap)
            price *= (1 + 0.0)         # explicit: no overnight gap by default
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        out[sym] = df
    return out


# ── IntradayAssetView.opening_range ─────────────────────────────────────


class TestOpeningRange:
    def _view(self, highs, lows, i, session_start=0):
        """Build an IntradayAssetView at index ``i`` from explicit hi/lo arrays."""
        from edgefinder.engine.intraday_strategy import IntradayAssetView
        n = len(highs)
        h = np.array(highs, dtype=float)
        l = np.array(lows, dtype=float)
        c = (h + l) / 2.0
        o = c.copy()
        v = np.ones(n)
        ts = np.arange(n, dtype=np.int64)
        return IntradayAssetView(symbol="AAA", _o=o, _h=h, _l=l, _c=c, _v=v,
                                 _ts=ts, i=i, session_start=session_start)

    def test_equals_first_m_bar_hi_lo(self):
        highs = [10, 11, 9, 15, 8, 7]
        lows = [5, 6, 4, 7, 3, 2]
        # fully past the range (i well beyond session_start+m): first 3 bars
        v = self._view(highs, lows, i=5, session_start=0)
        assert v.opening_range(3) == (11.0, 4.0)   # max(10,11,9), min(5,6,4)

    def test_partial_when_range_not_yet_formed(self):
        highs = [10, 11, 9, 15, 8, 7]
        lows = [5, 6, 4, 7, 3, 2]
        # bars_since_open (i+1) < m: only the bars seen so far count
        v = self._view(highs, lows, i=1, session_start=0)   # 2 bars seen, m=3
        assert v.opening_range(3) == (11.0, 5.0)   # max(10,11), min(5,6)

    def test_never_reads_future_bars(self):
        # a huge spike LATER in the session must not leak into the OR
        highs = [10, 11, 9, 999, 8, 7]
        lows = [5, 6, 4, -50, 3, 2]
        v = self._view(highs, lows, i=2, session_start=0)   # 3 bars seen, m=3
        assert v.opening_range(3) == (11.0, 4.0)   # spike at idx 3 ignored

    def test_respects_session_start(self):
        # opening range is THIS session's first m bars, not the array's
        highs = [99, 99, 10, 11, 9, 15]
        lows = [1, 1, 5, 6, 4, 7]
        v = self._view(highs, lows, i=5, session_start=2)   # session starts idx 2
        assert v.opening_range(3) == (11.0, 4.0)   # bars 2,3,4 -> max(10,11,9)


# ── basic plumbing ─────────────────────────────────────────────────────


class TestSynthHelper:
    def test_correct_et_timestamps(self):
        bars = _synthetic_session(["AAA"], 1, bars_per_day=390)
        df = bars["AAA"]
        from edgefinder.data.minutestore import to_et
        et = to_et(df["ts"])
        mins = et.dt.hour * 60 + et.dt.minute
        assert mins.iloc[0] == 9 * 60 + 30      # 09:30
        assert mins.iloc[-1] == 15 * 60 + 59    # 15:59
        assert len(df) == 390


class TestFlatNull:
    def test_flat_is_exactly_flat(self):
        bars = _synthetic_session(["AAA"], 4, bars_per_day=60, drift_per_day=0.02)
        res = run_intraday_backtest(bars, IntradayFlat(), start_cash=1_000_000.0)
        assert isinstance(res, IntradayBacktestResult)
        assert len(res.daily_equity_curve) == 4
        for _, eq in res.daily_equity_curve:
            assert eq == pytest.approx(1_000_000.0)
        assert res.trades == []
        assert res.stats["num_trades"] == 0


# ── anchor: BuyHoldFromOpen == session open->close ──────────────────────


class TestBuyHoldAnchor:
    def test_daily_return_equals_open_to_close_no_costs(self):
        drift = 0.01
        bars = _synthetic_session(["AAA"], 5, bars_per_day=60, drift_per_day=drift)
        res = run_intraday_backtest(
            bars, BuyHoldFromOpen("AAA"), start_cash=1_000_000.0,
            flatten_at_close=True, cost_model=None, cost_bps=0.0)
        # buy at first decision bar -> fills at bar 1's open == day open (no gap),
        # MOC flatten at last bar close. Each session's return ~ open->close.
        prev = 1_000_000.0
        for (_, eq) in res.daily_equity_curve:
            day_ret = eq / prev - 1.0
            assert day_ret == pytest.approx(drift, rel=1e-3, abs=5.0e-4)
            prev = eq

    def test_costed_one_entry_one_moc_per_day(self):
        bars = _synthetic_session(["AAA"], 3, bars_per_day=30, drift_per_day=0.005)
        from edgefinder.backtest.costs import CostModel
        res = run_intraday_backtest(
            bars, BuyHoldFromOpen("AAA"), cost_model=CostModel(),
            flatten_at_close=True)
        buys = [t for t in res.trades if t["side"] == "BUY"]
        mocs = [t for t in res.trades if t.get("reason") == "MOC_FLATTEN"]
        assert len(buys) == 3      # one entry per session
        assert len(mocs) == 3      # one MOC flatten per session
        assert res.stats["avg_trades_per_day"] == pytest.approx(2.0)


# ── next-bar fill (no look-ahead) ───────────────────────────────────────


class TestNextBarFill:
    def test_decision_fills_at_next_bar_open(self):
        # hand-built single session, distinct opens so the fill price is provable
        d = date(2024, 1, 2)
        ts = _session_ts(d, 5)
        opens = [100.0, 101.0, 102.0, 103.0, 104.0]
        rows = [(ts[i], opens[i], opens[i] + 1, opens[i] - 1, opens[i] + 0.5, 1e6)
                for i in range(5)]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])

        class BuyAtBar0:
            name = "buy_at_bar0"
            def decide(self, ctx):
                # buy only on the very first decision bar of the session
                return {"AAA": 1.0} if ctx.bars_since_open == 1 else (
                    {} if ctx.is_last_decision_bar else {"AAA": 1.0})

        res = run_intraday_backtest(
            {"AAA": df}, BuyAtBar0(), start_cash=1_000_000.0,
            flatten_at_close=True, cost_bps=0.0)
        first_buy = next(t for t in res.trades if t["side"] == "BUY")
        # decision at bar 0 -> fill at bar 1's OPEN == 101.0
        assert first_buy["price"] == pytest.approx(101.0)

    def test_cheater_earns_no_excess(self):
        # a strategy whose "edge" needs the next bar gets it at the next OPEN,
        # not the decision close -> no free money. Compare to buy-hold: same fills.
        bars = _synthetic_session(["AAA"], 4, bars_per_day=40, drift_per_day=0.01)

        class Cheater:
            name = "cheater"
            def decide(self, ctx):
                # "always go long" — but cannot peek; fills are next-bar regardless
                return {} if ctx.is_last_decision_bar else {"AAA": 1.0}

        cheat = run_intraday_backtest({k: v.copy() for k, v in bars.items()},
                                      Cheater(), flatten_at_close=True, cost_bps=0.0)
        anchor = run_intraday_backtest({k: v.copy() for k, v in bars.items()},
                                       BuyHoldFromOpen("AAA"), flatten_at_close=True,
                                       cost_bps=0.0)
        # both are long from bar1 open to close every day -> identical curves
        assert [e for _, e in cheat.daily_equity_curve] == \
            pytest.approx([e for _, e in anchor.daily_equity_curve], rel=1e-9)


# ── tolls: churn bleeds more than buy-once ──────────────────────────────


class TestTolls:
    def test_churn_bleeds_more_than_trade_once(self):
        bars = _synthetic_session(["AAA"], 3, bars_per_day=30, drift_per_day=0.0)
        from edgefinder.backtest.costs import CostModel

        class TradeEveryBar:
            name = "churn"
            def __init__(self):
                self._flip = False
            def decide(self, ctx):
                if ctx.is_last_decision_bar:
                    return {}
                self._flip = not self._flip
                return {"AAA": 1.0} if self._flip else {}

        churn = run_intraday_backtest({k: v.copy() for k, v in bars.items()},
                                      TradeEveryBar(), cost_model=CostModel())
        once = run_intraday_backtest({k: v.copy() for k, v in bars.items()},
                                     BuyHoldFromOpen("AAA"), cost_model=CostModel())
        assert churn.stats["final_equity"] < once.stats["final_equity"]
        assert churn.stats["num_trades"] > once.stats["num_trades"]


# ── session mechanics ───────────────────────────────────────────────────


class TestSessionMechanics:
    def test_context_counters(self):
        bars = _synthetic_session(["AAA"], 1, bars_per_day=390, drift_per_day=0.0)
        seen: list[IntradayContext] = []

        class Recorder:
            name = "rec"
            def decide(self, ctx):
                seen.append(ctx)
                return {}

        run_intraday_backtest(bars, Recorder(), flatten_at_close=True)
        # first decision bar
        first = seen[0]
        assert first.bars_since_open == 1
        assert first.minute_of_day == 9 * 60 + 30
        assert not first.is_last_decision_bar
        # bars_until_close from the ET clock at 09:30 (minutes to 16:00 / 1) - 1
        assert first.bars_until_close == (16 * 60 - (9 * 60 + 30)) - 1   # 389
        # the last decision bar (flatten -> engine skips deciding ON the last
        # bar, so the last recorded ctx is the 2nd-to-last bar)
        last = seen[-1]
        assert last.minute_of_day == 15 * 60 + 58
        # is_last_decision_bar flips on the true session-last bar; with flatten
        # the strategy isn't asked there, but the flag is computable on any bar
        mins = [c.minute_of_day for c in seen]
        assert mins == list(range(9 * 60 + 30, 15 * 60 + 59))   # 09:30..15:58

    def test_flatten_zero_open_positions_each_close(self):
        bars = _synthetic_session(["AAA"], 4, bars_per_day=30, drift_per_day=0.01)
        res = run_intraday_backtest(bars, BuyHoldFromOpen("AAA"),
                                    flatten_at_close=True, cost_bps=0.0)
        assert res.stats["open_positions"] == 0

    def test_hold_overnight_carries_and_marks_gap(self):
        # two sessions WITH an overnight gap; hold-overnight carries the lot and
        # the gap is marked into the next session's curve.
        d0, d1 = _us_trading_days(date(2024, 1, 2), 2)
        rows = []
        # day 0: flat at 100, close 100
        for t in _session_ts(d0, 5):
            rows.append((t, 100.0, 100.1, 99.9, 100.0, 1e6))
        # day 1: GAP UP to 110 at open, flat after
        for t in _session_ts(d1, 5):
            rows.append((t, 110.0, 110.1, 109.9, 110.0, 1e6))
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])

        class BuyDay0Hold:
            name = "buy_d0_hold"
            def decide(self, ctx):
                return {"AAA": 1.0}      # always want to be long; never flatten

        res = run_intraday_backtest({"AAA": df}, BuyDay0Hold(),
                                    flatten_at_close=False, cost_bps=0.0,
                                    start_cash=1_000_000.0)
        # day 0 ~flat; day 1 jumps ~+10% because the held lot re-prices at the gap
        c0 = res.daily_equity_curve[0][1]
        c1 = res.daily_equity_curve[1][1]
        assert c0 == pytest.approx(1_000_000.0, rel=1e-3)
        assert c1 / c0 - 1.0 == pytest.approx(0.10, rel=0.02)
        assert res.stats["open_positions"] == 1   # still holding at the end


# ── daily curve + intraday drawdown ─────────────────────────────────────


class TestDailyCurveAndDrawdown:
    def test_one_point_per_session(self):
        bars = _synthetic_session(["AAA"], 7, bars_per_day=20, drift_per_day=0.0)
        res = run_intraday_backtest(bars, IntradayFlat())
        assert len(res.daily_equity_curve) == 7
        dates = [d for d, _ in res.daily_equity_curve]
        assert dates == sorted(set(dates))     # unique, ascending

    def test_intraday_dd_ge_close_to_close(self):
        # craft an intra-session DIP that recovers by the close, while holding.
        def intrabar(di, bi, day_open):
            # one deep dip mid-session on day 1
            if di == 1 and bi == 10:
                return (day_open, day_open, day_open * 0.90, day_open * 0.99)
            return None
        bars = _synthetic_session(["AAA"], 3, bars_per_day=30, drift_per_day=0.0,
                                  intrabar=intrabar)

        class AlwaysLong:
            name = "always_long"
            def decide(self, ctx):
                return {"AAA": 1.0}      # hold through the dip (no flatten)

        res = run_intraday_backtest(bars, AlwaysLong(), flatten_at_close=False,
                                    cost_bps=0.0)
        assert res.stats["intraday_max_drawdown_pct"] >= res.stats["max_drawdown_pct"]
        assert res.stats["intraday_max_drawdown_pct"] > 0


# ── walk-forward ────────────────────────────────────────────────────────


def _daily_keys_subset():
    return {"strategy", "config", "oos", "criteria", "holdout", "by_regime",
            "folds", "verdict"}


class TestWalkforward:
    def _bars(self, days=140, syms=("AAA", "SPY")):
        return _synthetic_session(list(syms), days, bars_per_day=20,
                                  drift_per_day=0.001)

    def test_scorecard_shape_matches_daily(self):
        bars = self._bars()
        spy = bars["SPY"]
        card = run_intraday_walkforward(
            {"AAA": bars["AAA"]}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=spy, is_days=60, oos_days=21, step_days=21, warmup_days=3)
        assert _daily_keys_subset().issubset(set(card.keys()))
        assert card["config"]["engine"] == "intraday"
        assert card["config"]["bar"] == "1min"
        assert "flatten_at_close" in card["config"]
        assert "decision_interval" in card["config"]
        assert card["verdict"] in ("PASS", "FAIL")
        assert "criteria" in card and "all_met" in card["criteria"]
        # folds planned on DAYS
        assert card["config"]["num_folds"] >= 1

    def test_benchmark_scored_over_same_sessions(self):
        bars = self._bars()
        card = run_intraday_walkforward(
            {"AAA": bars["AAA"]}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=bars["SPY"], is_days=60, oos_days=21, step_days=21,
            warmup_days=3)
        for f in card["folds"]:
            assert f["excess_vs_spy_pct"] is not None

    def test_holdout_carving(self):
        bars = self._bars(days=160)
        days = sorted({d for d in
                       __import__("edgefinder.data.minutestore", fromlist=["to_et"])
                       .to_et(bars["AAA"]["ts"]).dt.date})
        hstart = days[-25]
        card = run_intraday_walkforward(
            {"AAA": bars["AAA"]}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=bars["SPY"], is_days=50, oos_days=21, step_days=21,
            warmup_days=3, holdout_start=hstart, holdout_eval=True)
        assert card["holdout"] is not None
        assert "passes" in card["holdout"]
        assert card["config"]["holdout_evaluated"] is True

    def test_determinism(self):
        bars = self._bars()
        a = run_intraday_walkforward(
            {"AAA": bars["AAA"].copy()}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=bars["SPY"].copy(), is_days=60, oos_days=21, step_days=21,
            warmup_days=3)
        b = run_intraday_walkforward(
            {"AAA": bars["AAA"].copy()}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=bars["SPY"].copy(), is_days=60, oos_days=21, step_days=21,
            warmup_days=3)
        assert a == b


# ── performance guard ───────────────────────────────────────────────────


class TestPerformanceGuard:
    def test_5sym_10session_is_fast(self):
        syms = ["S0", "S1", "S2", "S3", "S4"]
        bars = _synthetic_session(syms, 10, bars_per_day=390, drift_per_day=0.0)
        strat = IntradayMeanReversion("S0", lookback=20, z=1.0)
        t0 = _time.perf_counter()
        res = run_intraday_backtest(bars, strat, flatten_at_close=True)
        elapsed = _time.perf_counter() - t0
        # 5 * 3900 = 19,500 bars; O(1) history access -> a few seconds at most.
        assert elapsed < 10.0
        assert len(res.daily_equity_curve) == 10


# ── record (sqlite via db_session fixture) ──────────────────────────────


class TestRecord:
    def test_scorecard_round_trips_to_validation_runs(self, db_session):
        from edgefinder.engine.record import record_validation_run

        bars = _synthetic_session(["AAA", "SPY"], 120, bars_per_day=20,
                                  drift_per_day=0.001)
        card = run_intraday_walkforward(
            {"AAA": bars["AAA"]}, lambda: BuyHoldFromOpen("AAA"),
            spy_bars=bars["SPY"], is_days=50, oos_days=21, step_days=21,
            warmup_days=3)
        rid = record_validation_run(db_session, card, universe="intraday:1syms")
        row = db_session.get(ValidationRun, rid)
        assert row.config["engine"] == "intraday"
        assert row.universe == "intraday:1syms"
        assert row.verdict in ("PASS", "FAIL")


# ── CLI frame-split (the SPY-as-traded-symbol regression) ───────────────


class TestCLISplitFrames:
    def test_spy_as_traded_symbol_is_not_dropped(self):
        # SPY is the benchmark AND a tradable name; splitting must keep it in
        # bars (the pop() bug returned a false "no minute bars for SPY").
        from edgefinder.engine.intraday_validate import _split_frames

        frames = {"SPY": "spy_df", "AAA": "aaa_df"}
        bars, spy = _split_frames(frames, ["SPY"])
        assert spy == "spy_df"
        assert bars == {"SPY": "spy_df"}          # SPY survives as tradable

    def test_spy_benchmark_only_excluded_from_bars(self):
        from edgefinder.engine.intraday_validate import _split_frames

        frames = {"SPY": "spy_df", "AAA": "aaa_df"}
        bars, spy = _split_frames(frames, ["AAA"])
        assert spy == "spy_df"                     # benchmark available
        assert bars == {"AAA": "aaa_df"}           # SPY not traded

    def test_missing_spy_returns_none(self):
        from edgefinder.engine.intraday_validate import _split_frames

        bars, spy = _split_frames({"AAA": "aaa_df"}, ["AAA"])
        assert spy is None and bars == {"AAA": "aaa_df"}


# ── record numpy-coercion (intraday-r1 wave-1 write bug) ────────────────


class TestRecordNumpyCoercion:
    def test_numpy_scalars_in_scorecard_round_trip(self, db_session):
        """np.float64/np.bool_ in a scorecard must not break the JSON write
        (9/12 intraday-r1 jobs computed fine but failed here, 2026-06-14)."""
        import numpy as np
        from edgefinder.engine.record import record_validation_run

        card = {
            "strategy": "np_probe",
            "config": {"engine": "intraday", "costed": np.True_},
            "oos": {"mean_excess_vs_spy_pct": np.float64(-12.34),
                    "nan_metric": np.float64("nan"),
                    "n": np.int64(7)},
            "criteria": {"mean_excess_positive": np.False_, "all_met": np.False_},
            "holdout": None,
            "verdict": "FAIL",
        }
        rid = record_validation_run(db_session, card, universe="intraday-r1:np_probe")
        row = db_session.get(ValidationRun, rid)
        assert row.criteria["mean_excess_positive"] is False
        assert row.config["costed"] is True
        assert row.oos["mean_excess_vs_spy_pct"] == -12.34
        assert row.oos["nan_metric"] is None      # non-finite nulled
        assert row.oos["n"] == 7
