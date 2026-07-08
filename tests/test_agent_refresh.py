"""Unit-test the pure bar-ingestion logic of the data-refresh tool."""

from __future__ import annotations


def test_module_imports():
    import agent.refresh as r
    assert callable(r.refresh_alpaca) and callable(r.refresh_alpaca_market)
    assert callable(r.main)


def test_bound_network_sets_socket_timeout():
    """A hung TLS handshake must fail fast, not block the whole refresh: the
    guard installs a finite process-wide socket timeout."""
    import socket

    import agent.refresh as r

    prev = socket.getdefaulttimeout()
    try:
        r._bound_network(7.5)
        assert socket.getdefaulttimeout() == 7.5
        r._bound_network()  # default uses the module constant
        assert socket.getdefaulttimeout() == r.NET_TIMEOUT_S
        assert 0 < r.NET_TIMEOUT_S < 120  # finite + snappy, not the OS default
    finally:
        socket.setdefaulttimeout(prev)


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
