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


def test_today_bar_final_gate():
    """Today's bar counts as settled only on a weekday after 16:15 ET — never
    intraday (a partial bar) and never on weekends. This is the honesty gate:
    fills only ever price off a final close."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from agent.refresh import _today_bar_is_final

    et = ZoneInfo("America/New_York")
    # Wednesday
    assert _today_bar_is_final(datetime(2026, 6, 17, 16, 30, tzinfo=et)) is True   # after close
    assert _today_bar_is_final(datetime(2026, 6, 17, 16, 14, tzinfo=et)) is False  # 1 min early
    assert _today_bar_is_final(datetime(2026, 6, 17, 10, 0, tzinfo=et)) is False   # mid-session
    # Saturday afternoon — no trading day
    assert _today_bar_is_final(datetime(2026, 6, 20, 18, 0, tzinfo=et)) is False
