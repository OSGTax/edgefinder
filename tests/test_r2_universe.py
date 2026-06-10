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

from edgefinder.backtest.jobs import resolve_universe
from edgefinder.db.models import DailyBar
from edgefinder.engine.data import load_bars_from_store, rank_top_universe

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
