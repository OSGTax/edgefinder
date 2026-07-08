"""Tests for split adjustment — the fidelity-audit fix (fake split cliffs)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.data.splits_backfill import _as_int_ratio
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import DailyBar, TickerSplit
from edgefinder.engine.data import (
    adjust_dividends_for_splits,
    adjust_for_splits,
    load_bars,
)


def _bars(n: int, price: float, start: date = date(2022, 1, 3)) -> pd.DataFrame:
    dates = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "date": dates, "open": [price] * n, "high": [price] * n,
        "low": [price] * n, "close": [price] * n, "volume": [1e6] * n,
    })


class TestAdjustForSplits:
    def test_forward_split_removes_the_cliff(self):
        # 3:1 split on day 5: raw pre-split prices 900, post-split 300
        df = _bars(10, 900.0)
        df.loc[5:, ["open", "high", "low", "close"]] = 300.0
        ex = df["date"].iloc[5]
        out = adjust_for_splits({"TSLA": df}, {"TSLA": [(ex, 3.0)]})["TSLA"]
        assert out["close"].iloc[4] == pytest.approx(300.0)   # 900/3
        assert out["close"].iloc[5] == 300.0                  # untouched
        assert out["volume"].iloc[4] == pytest.approx(3e6)    # x3
        # dollar volume invariant: 900*1e6 == 300*3e6
        assert (out["close"].iloc[4] * out["volume"].iloc[4]
                == pytest.approx(900.0 * 1e6))

    def test_reverse_split_removes_fake_gain(self):
        # 1:8 reverse: raw pre-split 12.95, post-split ~103
        df = _bars(10, 12.95)
        df.loc[5:, ["open", "high", "low", "close"]] = 103.6
        ex = df["date"].iloc[5]
        out = adjust_for_splits({"GE": df}, {"GE": [(ex, 1 / 8)]})["GE"]
        assert out["close"].iloc[4] == pytest.approx(103.6)   # 12.95/(1/8)

    def test_multiple_splits_compound(self):
        # NVDA-style: 4:1 then 10:1 -> oldest bars divided by 40
        df = _bars(30, 751.19)
        df.loc[10:, ["open", "high", "low", "close"]] = 187.80   # post 4:1
        df.loc[20:, ["open", "high", "low", "close"]] = 121.79   # post 10:1... raw basis
        # raw basis: middle segment is as-traded (187.80), last is as-traded
        ex1, ex2 = df["date"].iloc[10], df["date"].iloc[20]
        out = adjust_for_splits(
            {"NVDA": df}, {"NVDA": [(ex1, 4.0), (ex2, 10.0)]})["NVDA"]
        assert out["close"].iloc[0] == pytest.approx(751.19 / 40)
        assert out["close"].iloc[15] == pytest.approx(187.80 / 10)
        assert out["close"].iloc[25] == 121.79

    def test_split_before_history_is_noop(self):
        df = _bars(10, 100.0)
        out = adjust_for_splits(
            {"AAA": df}, {"AAA": [(date(2020, 1, 1), 10.0)]})["AAA"]
        assert out["close"].iloc[0] == 100.0

    def test_no_splits_identity(self):
        df = _bars(5, 100.0)
        assert adjust_for_splits({"AAA": df}, {})["AAA"] is df


class TestDividendSplitInteraction:
    def test_pre_split_dividend_scaled(self):
        # $0.57/share declared pre-3:1-split == $0.19 on the adjusted basis
        divs = {"WMT": [(date(2023, 3, 10), 0.57), (date(2024, 6, 10), 0.21)]}
        splits = {"WMT": [(date(2024, 2, 26), 3.0)]}
        out = adjust_dividends_for_splits(divs, splits)["WMT"]
        assert out[0][1] == pytest.approx(0.57 / 3)   # pre-split: scaled
        assert out[1][1] == 0.21                      # post-split: as-is


class TestLoadBarsIntegration:
    def test_load_bars_split_adjusts_and_quarantines(self):
        engine = get_engine(url="sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = get_session_factory(engine)()
        d0 = date(2022, 1, 3)
        for i in range(10):
            px = 900.0 if i < 5 else 300.0
            session.add(DailyBar(symbol="TSLA", date=d0 + timedelta(days=i),
                                 open=px, high=px, low=px, close=px,
                                 volume=1e6, source="test"))
        session.add(TickerSplit(symbol="TSLA",
                                execution_date=str(d0 + timedelta(days=5)),
                                split_from=1, split_to=3))
        # META contamination: rows before 2022-06-09 must be dropped
        session.add(DailyBar(symbol="META", date=date(2022, 1, 5), open=15,
                             high=15, low=15, close=15, volume=1e6, source="t"))
        session.add(DailyBar(symbol="META", date=date(2022, 7, 1), open=170,
                             high=170, low=170, close=170, volume=1e6, source="t"))
        session.commit()

        bars = load_bars(session, ["TSLA", "META"])
        assert bars["TSLA"]["close"].iloc[0] == pytest.approx(300.0)
        assert len(bars["META"]) == 1                 # ETF-era row gone
        assert bars["META"]["date"].iloc[0] == date(2022, 7, 1)

        raw = load_bars(session, ["TSLA"], split_adjust=False)
        assert raw["TSLA"]["close"].iloc[0] == 900.0  # opt-out works
        session.close()

    def test_load_bars_sorts_split_free_symbol_by_date(self):
        """A split-free name (SPY/QQQ/IWM) inserted out of date order must come
        back ascending. `adjust_for_splits` only sorts symbols WITH splits, so
        without an unconditional sort in load_bars the frame renders scrambled —
        the chart's right edge showed a two-week-old bar (June 26/30) even though
        the newest bar was present, just mis-positioned."""
        engine = get_engine(url="sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = get_session_factory(engine)()
        # Insert the RECENT bars first, then older ones — mimicking a
        # trailing-window delete-then-insert top-up reshuffling physical order.
        recent = [date(2026, 7, 7), date(2026, 7, 6), date(2026, 7, 2)]
        older = [date(2026, 6, 30), date(2026, 6, 29), date(2026, 6, 26)]
        for d in recent + older:
            session.add(DailyBar(symbol="SPY", date=d, open=1, high=1, low=1,
                                 close=1, volume=1e6, source="test"))
        session.commit()

        frame = load_bars(session, ["SPY"])["SPY"]   # SPY has no splits
        dates = list(frame["date"])
        assert dates == sorted(dates)                # ascending
        assert dates[-1] == date(2026, 7, 7)         # newest bar is the RIGHT edge
        assert dates[0] == date(2026, 6, 26)
        session.close()


class TestBackfillRatio:
    def test_int_ratio_handling(self):
        assert _as_int_ratio(1, 3) == (1, 3)
        assert _as_int_ratio(8, 1) == (8, 1)          # reverse
        assert _as_int_ratio(2, 3) == (2, 3)
        assert _as_int_ratio(1, 1.5) == (2, 3)        # fractional -> exact
        assert _as_int_ratio(1000, 1281) == (1000, 1281)
        assert _as_int_ratio(0, 3) is None
        assert _as_int_ratio("x", 3) is None
