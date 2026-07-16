"""Unit-test the pure bar-ingestion logic of the data-refresh tool."""

from __future__ import annotations

from datetime import date, timedelta

import pytest


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


# ── corporate-actions ingest (fake broker, SQLite store) ──


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'refresh.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


class _FakeCABroker:
    """Stands in for broker.Broker(): records every CA query window and
    answers with the canned announcements whose ex-date falls inside it."""

    def __init__(self, announcements):
        self.calls: list[tuple] = []
        self._anns = announcements

    def corporate_announcements(self, *, since=None, until=None, symbol=None):
        self.calls.append((since, until))
        return [a for a in self._anns
                if since <= date.fromisoformat(a["ex_date"]) <= until]


def _split(sym, ex_date, old=1.0, new=10.0):
    return {"ca_type": "split", "symbol": sym, "ex_date": str(ex_date),
            "cash": None, "old_rate": old, "new_rate": new}


def _dividend(sym, ex_date, cash=0.5):
    return {"ca_type": "dividend", "symbol": sym, "ex_date": str(ex_date),
            "cash": cash, "old_rate": None, "new_rate": None}


def _patch_broker(monkeypatch, fake):
    import agent.broker as broker

    monkeypatch.setattr(broker, "enabled", lambda: True)
    monkeypatch.setattr(broker, "Broker", lambda: fake)


def test_corp_actions_windows_cover_long_lookbacks(store, monkeypatch):
    """The nightly passes its ~400-day bar window; Alpaca caps one CA call at
    90 days — the range must be tiled in capped, gap-free windows."""
    from agent.refresh import CA_WINDOW_DAYS, _corp_actions_alpaca

    today = date.today()
    fake = _FakeCABroker([_split("OLD", today - timedelta(days=300))])
    _patch_broker(monkeypatch, fake)

    out = _corp_actions_alpaca(store, ["OLD"], back_days=400, fwd_days=45)
    assert all((hi - lo).days <= CA_WINDOW_DAYS for lo, hi in fake.calls)
    assert fake.calls[0][0] == today - timedelta(days=400)
    assert fake.calls[-1][1] == today + timedelta(days=45)
    for (_, prev_hi), (next_lo, _) in zip(fake.calls, fake.calls[1:]):
        assert next_lo == prev_hi + timedelta(days=1)  # no gaps, no overlap
    # a split 300 days back — far outside the old single 90-day call — lands
    assert out["splits"] == 1 and out["window_errors"] == 0
    assert len(store.select("ticker_splits", filters={"symbol": "OLD"})) == 1


def test_corp_actions_hourly_defaults_stay_one_window(store, monkeypatch):
    from agent.refresh import _corp_actions_alpaca

    fake = _FakeCABroker([])
    _patch_broker(monkeypatch, fake)
    out = _corp_actions_alpaca(store, ["AAA"], back_days=45, fwd_days=45)
    assert len(fake.calls) == 1                    # the pre-fix single call
    assert out == {"dividends": 0, "splits": 0, "dupes_skipped": 0,
                   "windows": 1, "window_errors": 0}


def test_corp_actions_dedup_is_idempotent_across_nights(store, monkeypatch):
    from agent.refresh import _corp_actions_alpaca

    today = date.today()
    fake = _FakeCABroker([_split("SPL", today - timedelta(days=10)),
                          _dividend("DIV", today - timedelta(days=5))])
    _patch_broker(monkeypatch, fake)

    first = _corp_actions_alpaca(store, ["SPL", "DIV"])
    assert first["splits"] == 1 and first["dividends"] == 1
    # night two: the same announcements come back — nothing new is written
    second = _corp_actions_alpaca(store, ["SPL", "DIV"])
    assert second["splits"] == 0 and second["dividends"] == 0
    assert len(store.select("ticker_splits", filters={"symbol": "SPL"})) == 1
    assert len(store.select("dividends", filters={"symbol": "DIV"})) == 1


def test_corp_actions_only_ingests_requested_symbols(store, monkeypatch):
    from agent.refresh import _corp_actions_alpaca

    today = date.today()
    fake = _FakeCABroker([_split("MINE", today - timedelta(days=3)),
                          _split("OTHER", today - timedelta(days=3))])
    _patch_broker(monkeypatch, fake)

    out = _corp_actions_alpaca(store, ["MINE"])
    assert out["splits"] == 1
    assert store.select("ticker_splits", filters={"symbol": "OTHER"}) == []


def test_corp_actions_one_bad_window_degrades_not_aborts(store, monkeypatch):
    from agent.refresh import _corp_actions_alpaca

    today = date.today()

    class _Flaky(_FakeCABroker):
        def corporate_announcements(self, *, since=None, until=None,
                                    symbol=None):
            if not self.calls:  # the first (oldest) window blows up
                self.calls.append((since, until))
                raise RuntimeError("proxy reset")
            return super().corporate_announcements(since=since, until=until)

    fake = _Flaky([_split("SPL", today - timedelta(days=10))])
    _patch_broker(monkeypatch, fake)

    out = _corp_actions_alpaca(store, ["SPL"], back_days=400, fwd_days=45)
    assert out["window_errors"] == 1
    assert out["splits"] == 1  # the recent window's split still landed


def test_corp_actions_out_of_window_dupe_skips_row_not_batch(store, monkeypatch):
    """M3 regression: Alpaca tiles CA windows on DECLARATION date, so a
    correction row can carry an ex-date OLDER than the dedup pre-load window.
    It escapes the dedup set and trips the unique constraint — which must
    cost that ONE row, never the whole night's batch (the pre-fix behavior
    rolled back every new dividend AND split, every night)."""
    from agent.refresh import _corp_actions_alpaca

    today = date.today()
    old_ex = today - timedelta(days=100)   # outside the default 45d window
    # already stored by an earlier long-lookback run
    store.insert("dividends", {"symbol": "OLDDIV", "ex_date": old_ex,
                               "cash_amount": 0.5}, returning=False)
    store.insert("ticker_splits", {"symbol": "OLDSPL",
                                   "execution_date": old_ex,
                                   "split_from": 1, "split_to": 10},
                 returning=False)

    class _Unwindowed(_FakeCABroker):
        """Returns every announcement regardless of the query window —
        modeling declaration-date tiling handing back an old ex-date."""

        def corporate_announcements(self, *, since=None, until=None,
                                    symbol=None):
            self.calls.append((since, until))
            return list(self._anns)

    fake = _Unwindowed([
        _dividend("OLDDIV", old_ex),                     # escapes the dedup
        _dividend("NEWDIV", today - timedelta(days=5)),
        _split("OLDSPL", old_ex),                        # escapes the dedup
        _split("NEWSPL", today - timedelta(days=3)),
    ])
    _patch_broker(monkeypatch, fake)

    out = _corp_actions_alpaca(store, ["OLDDIV", "NEWDIV", "OLDSPL", "NEWSPL"])
    assert out["dividends"] == 1 and out["splits"] == 1  # the new rows landed
    assert out["dupes_skipped"] == 2                     # skipped, not fatal
    assert len(store.select("dividends", filters={"symbol": "OLDDIV"})) == 1
    assert len(store.select("dividends", filters={"symbol": "NEWDIV"})) == 1
    assert len(store.select("ticker_splits", filters={"symbol": "OLDSPL"})) == 1
    assert len(store.select("ticker_splits", filters={"symbol": "NEWSPL"})) == 1


def test_corp_actions_dedup_bounded_by_symbols_when_small(store, monkeypatch):
    """L1: the hourly ~15-name pass must not scan the whole market's recent
    CA rows every hour — small symbol sets bound the dedup pre-load by
    symbol IN() as well as the date window; the ~1000-name nightly keeps the
    pure window scan."""
    from agent.refresh import DEDUP_SYMBOL_IN_MAX, _corp_actions_alpaca

    fake = _FakeCABroker([])
    _patch_broker(monkeypatch, fake)

    seen: list[tuple[str, dict]] = []
    orig = store.select

    def spy(table, **kw):
        seen.append((table, kw.get("filters") or {}))
        return orig(table, **kw)

    monkeypatch.setattr(store, "select", spy)

    _corp_actions_alpaca(store, ["AAA", "BBB"])
    small = dict(seen)
    assert small["dividends"]["symbol"] == ("in", ["AAA", "BBB"])
    assert "ex_date" in small["dividends"]              # window bound kept
    assert small["ticker_splits"]["symbol"] == ("in", ["AAA", "BBB"])

    seen.clear()
    _corp_actions_alpaca(store, [f"S{i:04d}" for i in range(DEDUP_SYMBOL_IN_MAX + 1)])
    large = dict(seen)
    assert "symbol" not in large["dividends"]           # nightly: window-only
    assert "symbol" not in large["ticker_splits"]


def test_nightly_market_refresh_covers_corp_actions_for_universe(monkeypatch):
    """The nightly's CA pass must cover the SAME symbol set as the bar ingest,
    with a lookback matching the bar window (the whole point of F3: splits for
    the researched ~1000-name fresh set, not just the ~15 hourly names)."""
    import agent.broker as broker
    import agent.edgar as edgar
    import agent.refresh as r

    class _MarketBroker:
        def list_assets(self, optionable=False):
            return [{"symbol": s} for s in ("AAA", "BBB", "SPY")]

    monkeypatch.setattr(broker, "enabled", lambda: True)
    monkeypatch.setattr(broker, "Broker", lambda: _MarketBroker())
    monkeypatch.setattr(r, "_alpaca_latest_daily",
                        lambda b, catalog, lookback_days=7: [
                            {"symbol": s, "close": 100.0, "volume": 1e6}
                            for s in ("AAA", "BBB", "SPY")])
    monkeypatch.setattr(r, "_keep_symbols_rest", lambda: {"SPY"})
    monkeypatch.setattr(r, "_ingest_history_batched",
                        lambda store, b, syms, *, max_days, chunk=100:
                        {"symbols": len(syms), "bars_upserted": 0,
                         "batches": 0, "errors": 0})
    monkeypatch.setattr(r, "_r2_merge_sync", lambda syms: "skipped (test)")
    monkeypatch.setattr(edgar, "ingest", lambda store, symbols=None: {})

    captured = {}

    def fake_ca(store, symbols, *, back_days=45, fwd_days=45):
        captured["symbols"] = sorted(symbols)
        captured["back_days"] = back_days
        return {"dividends": 0, "splits": 0, "windows": 5, "window_errors": 0}

    monkeypatch.setattr(r, "_corp_actions_alpaca", fake_ca)
    out = r.refresh_alpaca_market(top_n=2, max_days=400)
    assert captured["symbols"] == ["AAA", "BBB", "SPY"]  # bar-ingest set
    assert captured["back_days"] == 400                  # bar-window lookback
    assert out["corp_actions"]["windows"] == 5
