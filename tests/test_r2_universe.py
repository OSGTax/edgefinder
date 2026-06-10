"""R2-backed PIT universe resolution — parity with the SQL path.

rank_top_universe must implement EXACTLY resolve_universe('top') semantics
(PIT cut, graveyard liveness gate, trailing rank window, offset band) so a
hunt lane reads identically from the store after the DB sheds its breadth
history. Both implementations are run over the same seeded data here.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.db.models import DailyBar
from edgefinder.engine.data import (
    load_bars_from_store,
    rank_top_universe,
    resolve_universe,
)

START = date(2024, 1, 1)


def _seed(db_session):
    """Same scenario as the SQL tests: a future winner, a steady name, a
    dead name, and a faded-glory name — in BOTH the DB and frame form."""
    frames: dict[str, pd.DataFrame] = {}

    def add(sym, spec):
        rows = []
        for day_offset, vol in spec:
            d = START + timedelta(days=day_offset)
            rows.append({"date": d, "open": 50.0, "high": 50.5, "low": 49.5,
                         "close": 50.0, "volume": vol})
            db_session.add(DailyBar(symbol=sym, date=d, open=50.0, high=50.5,
                                    low=49.5, close=50.0, volume=vol))
        frames[sym] = pd.DataFrame(rows)

    add("LATER", [(i, 100.0 if i < 20 else 50_000_000.0) for i in range(40)])
    add("STEADY", [(i, 1_000_000.0) for i in range(40)])
    add("DEAD", [(i - 200, 90_000_000.0) for i in range(5)])
    add("FADED", [(i, 50_000_000.0 if i < 20 else 1_000.0) for i in range(40)])
    db_session.commit()
    return frames


@pytest.mark.parametrize("kwargs", [
    dict(top_n=2),
    dict(top_n=2, rank_offset=1),
    dict(top_n=3),
    dict(top_n=2, rank_start=START + timedelta(days=20)),
    dict(top_n=10),
])
def test_rank_top_universe_matches_sql_path(db_session, kwargs):
    frames = _seed(db_session)
    as_of = START + timedelta(days=39)
    sql = resolve_universe(db_session, "top", [], kwargs["top_n"],
                           as_of=as_of,
                           rank_offset=kwargs.get("rank_offset", 0),
                           rank_start=kwargs.get("rank_start"))
    mem = rank_top_universe(frames, as_of, kwargs["top_n"],
                            rank_offset=kwargs.get("rank_offset", 0),
                            rank_start=kwargs.get("rank_start"))
    assert mem == sql


def test_rank_top_universe_pit_cut_hides_future(db_session):
    frames = _seed(db_session)
    cut = START + timedelta(days=19)   # before LATER's volume explosion
    assert rank_top_universe(frames, cut, 2) == \
        resolve_universe(db_session, "top", [], 2, as_of=cut)
    assert "DEAD" not in rank_top_universe(frames, cut, 10)


def test_store_loader_applies_contamination_quarantine(monkeypatch):
    """META rows before 2022-06-09 are a different security and must be
    dropped by the store loader exactly as the DB loader drops them."""
    meta = pd.DataFrame({
        "date": [date(2022, 6, 8), date(2022, 6, 9), date(2022, 6, 10)],
        "open": [10.0, 160.0, 161.0], "high": [10.0, 161.0, 162.0],
        "low": [10.0, 159.0, 160.0], "close": [10.0, 160.0, 161.0],
        "volume": [1e6, 2e7, 2e7]})

    from edgefinder.data import barstore

    monkeypatch.setattr(barstore.BarStore, "__init__", lambda self: None)
    monkeypatch.setattr(barstore.BarStore, "read_manifest",
                        lambda self: {"META": {}})
    monkeypatch.setattr(barstore.BarStore, "load",
                        lambda self, symbols, max_workers=8: {"META": meta.copy()})

    bars = load_bars_from_store(None)
    assert list(bars["META"]["date"]) == [date(2022, 6, 9), date(2022, 6, 10)]


class TestResolveUniverseSQL:
    """The SQL-path resolve_universe tests, relocated from the retired
    backtest-jobs test module (the function now lives in engine/data.py)."""

    def _seed_multi(self, db_session):
        start = date(2024, 1, 1)
        prices = [120.0]
        for _ in range(40):
            prices.append(prices[-1] * 0.988)
        for _ in range(25):
            prices.append(prices[-1] * 1.02)
        for i, p in enumerate(prices):
            db_session.add(DailyBar(symbol="TEST", date=start + timedelta(days=i),
                open=p, high=p * 1.01, low=p * 0.99, close=p, volume=1_000_000.0))
        # Two flat symbols with different dollar-volume for top-N ordering.
        for sym, vol in (("HIGHV", 5_000_000.0), ("LOWV", 100_000.0)):
            for i in range(40):
                db_session.add(DailyBar(symbol=sym, date=start + timedelta(days=i),
                    open=50.0, high=50.5, low=49.5, close=50.0, volume=vol))
        db_session.commit()

    def test_resolve_universe_modes(self, db_session):
        self._seed_multi(db_session)
        assert resolve_universe(db_session, "symbols", ["test", "msft"], 0) == ["MSFT", "TEST"]
        assert set(resolve_universe(db_session, "full", [], 0)) == {"TEST", "HIGHV", "LOWV"}
        assert resolve_universe(db_session, "top", [], 1) == ["HIGHV"]  # top dollar-volume

    def test_resolve_universe_point_in_time(self, db_session):
        """as_of must hide future dollar-volume (no future-selection bias)
        and exclude names already dead at the cut."""
        start = date(2024, 1, 1)
        # LATER2: tiny volume before the cut, explodes after (future winner).
        for i in range(40):
            vol = 100.0 if i < 20 else 50_000_000.0
            db_session.add(DailyBar(symbol="LATER2", date=start + timedelta(days=i),
                open=50.0, high=50.5, low=49.5, close=50.0, volume=vol))
        # STEADY2: solid volume throughout.
        for i in range(40):
            db_session.add(DailyBar(symbol="STEADY2", date=start + timedelta(days=i),
                open=50.0, high=50.5, low=49.5, close=50.0, volume=1_000_000.0))
        # DEAD2: huge volume but last bar long before the cut (delisted).
        for i in range(5):
            db_session.add(DailyBar(symbol="DEAD2", date=start - timedelta(days=200 - i),
                open=50.0, high=50.5, low=49.5, close=50.0, volume=90_000_000.0))
        db_session.commit()

        cut = start + timedelta(days=19)
        # Full-table ranking picks the long-dead name AND the future winner
        # over the genuinely tradable steady name...
        assert resolve_universe(db_session, "top", [], 2) == ["DEAD2", "LATER2"]
        # ...the point-in-time cut ranks only what was knowable and alive.
        assert resolve_universe(db_session, "top", [], 2, as_of=cut) == ["STEADY2", "LATER2"]
        assert "DEAD2" not in resolve_universe(db_session, "top", [], 10, as_of=cut)

    def test_resolve_universe_trailing_rank_window(self, db_session):
        """rank_start ranks on TRAILING dollar volume: a name that was huge
        long ago but thin lately must rank below a recently-active one."""
        start = date(2024, 1, 1)
        # FADED2: enormous volume in days 0-19, near-dead in days 20-39
        # (still alive — trades a trickle, so the liveness gate keeps it).
        for i in range(40):
            vol = 50_000_000.0 if i < 20 else 1_000.0
            db_session.add(DailyBar(symbol="FADED2", date=start + timedelta(days=i),
                open=50.0, high=50.5, low=49.5, close=50.0, volume=vol))
        # CURRENT2: steady real volume throughout.
        for i in range(40):
            db_session.add(DailyBar(symbol="CURRENT2", date=start + timedelta(days=i),
                open=50.0, high=50.5, low=49.5, close=50.0, volume=1_000_000.0))
        db_session.commit()

        as_of = start + timedelta(days=39)
        window_start = start + timedelta(days=20)
        # lifetime ranking still crowns the faded name on its glory days...
        assert resolve_universe(
            db_session, "top", [], 1, as_of=as_of) == ["FADED2"]
        # ...the trailing window ranks what is liquid NOW.
        assert resolve_universe(
            db_session, "top", [], 1, as_of=as_of,
            rank_start=window_start) == ["CURRENT2"]

    def test_resolve_universe_rank_offset_band(self, db_session):
        # rank_offset skips the most-liquid names -> the lower-liquidity band.
        self._seed_multi(db_session)
        # full ranking by dollar volume: HIGHV > TEST > LOWV
        assert resolve_universe(db_session, "top", [], 3) == ["HIGHV", "TEST", "LOWV"]
        # skip the top 1 -> band starts at rank 1
        assert resolve_universe(db_session, "top", [], 2, rank_offset=1) == ["TEST", "LOWV"]
        assert resolve_universe(db_session, "top", [], 1, rank_offset=2) == ["LOWV"]
