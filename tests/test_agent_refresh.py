"""Unit-test the pure bar-ingestion logic of the data-refresh tool."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from agent.refresh import aggs_to_rows


def _bar(ticker, close, volume, *, open=None):
    return SimpleNamespace(ticker=ticker, open=open if open is not None else close,
                           high=close * 1.01, low=close * 0.99, close=close,
                           volume=volume, transactions=100)


def test_top_n_and_keep_and_filters():
    aggs = [
        _bar("AAA", 100.0, 1_000_000),   # dv 100M  (rank 1)
        _bar("BBB", 50.0, 1_000_000),    # dv 50M   (rank 2)
        _bar("CCC", 10.0, 1_000_000),    # dv 10M   (rank 3, cut by top_n=2)
        _bar("SPY", 700.0, 100),         # tiny dv but in keep-set → forced in
        _bar("BRK.A", 600000.0, 5),      # has '.' → universe-filtered (not kept)
        _bar("PENNY", 0.5, 9_999_999),   # below min_price → filtered
    ]
    rows = aggs_to_rows(aggs, date(2026, 6, 19), top_n=2, keep={"SPY"},
                        min_price=1.0, max_price=100_000.0)
    syms = {r["symbol"] for r in rows}
    assert syms == {"AAA", "BBB", "SPY"}          # top-2 + forced benchmark
    assert "CCC" not in syms and "PENNY" not in syms and "BRK.A" not in syms
    spy = next(r for r in rows if r["symbol"] == "SPY")
    assert spy["date"] == date(2026, 6, 19) and spy["source"] == "grouped_daily"
    assert spy["close"] == 700.0


def test_dedup_when_kept_also_ranks_top():
    aggs = [_bar("AAA", 100.0, 1_000_000), _bar("SPY", 700.0, 1_000_000)]
    rows = aggs_to_rows(aggs, date(2026, 6, 19), top_n=5, keep={"SPY"},
                        min_price=1.0, max_price=100_000.0)
    assert sorted(r["symbol"] for r in rows) == ["AAA", "SPY"]  # SPY once, not twice


def test_module_imports():
    import agent.refresh as r
    assert callable(r.refresh) and callable(r.main)


def test_merge_bar_frames_grow_only_db_wins():
    import pandas as pd
    from agent.refresh import merge_bar_frames

    r2 = pd.DataFrame([
        {"date": "2026-06-17", "open": 1.0, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        {"date": "2026-06-18", "open": 1.0, "high": 2, "low": 0.5, "close": 1.6, "volume": 11},
    ])
    db = [  # overlaps the 18th (correction: close 1.65 wins) + extends to the 19th
        {"date": "2026-06-18", "open": 1.0, "high": 2, "low": 0.5, "close": 1.65, "volume": 11},
        {"date": "2026-06-19", "open": 1.1, "high": 2, "low": 0.6, "close": 1.7, "volume": 12},
    ]
    m = merge_bar_frames(r2, db)
    assert list(m["date"]) == ["2026-06-17", "2026-06-18", "2026-06-19"]  # grow-only
    assert m[m["date"] == "2026-06-18"]["close"].iloc[0] == 1.65          # DB wins
    # empty inputs behave
    assert len(merge_bar_frames(None, db)) == 2
    assert list(merge_bar_frames(r2, []) ["date"]) == ["2026-06-17", "2026-06-18"]


def test_select_universe_symbols_ranks_and_forces_keep():
    from agent.refresh import select_universe_symbols

    rows = [
        {"symbol": "AAA", "close": 100.0, "volume": 1_000_000},  # dv 100M (1)
        {"symbol": "BBB", "close": 50.0, "volume": 1_000_000},   # dv 50M  (2)
        {"symbol": "CCC", "close": 10.0, "volume": 1_000_000},   # dv 10M  (cut by top_n=2)
        {"symbol": "SPY", "close": 700.0, "volume": 100},        # tiny dv but in keep
        {"symbol": "BRK.A", "close": 600000.0, "volume": 5},     # '.' → shape-filtered
        {"symbol": "PENNY", "close": 0.5, "volume": 9_999_999},  # below min_price
    ]
    out = select_universe_symbols(rows, top_n=2, keep={"SPY"},
                                  min_price=1.0, max_price=100_000.0)
    assert set(out) == {"AAA", "BBB", "SPY"}       # top-2 by dv + forced keep
    assert out[:2] == ["AAA", "BBB"]               # rank order preserved
    assert "CCC" not in out and "PENNY" not in out and "BRK.A" not in out


def test_select_universe_symbols_dedup_keep_that_also_ranks():
    from agent.refresh import select_universe_symbols

    rows = [
        {"symbol": "AAA", "close": 100.0, "volume": 1_000_000},
        {"symbol": "SPY", "close": 700.0, "volume": 1_000_000},
    ]
    out = select_universe_symbols(rows, top_n=5, keep={"SPY"},
                                  min_price=1.0, max_price=100_000.0)
    assert sorted(out) == ["AAA", "SPY"]           # SPY once, not twice
    assert len(out) == 2
