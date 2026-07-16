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

        def session(self):
            return "closed"  # the nightly's normal post-close window

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
    monkeypatch.setattr(r, "_r2_merge_sync",
                        lambda syms, **kw: "skipped (test)")
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


# ── F17: bounded bar writes (payload-date delete, hole reporting) ──


def _bar_row(sym, d, close=1.0, volume=100.0):
    return {"symbol": sym, "date": d, "open": close, "high": close,
            "low": close, "close": close, "volume": volume, "source": "test"}


class _WrapStore:
    """Delegating store wrapper for fault injection."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_write_bars_replaces_exactly_payload_dates(store):
    """The delete is scoped to the payload's OWN dates: history outside the
    fetched window is never touched (the pre-fix blanket `date >= start`
    delete put ~400 days per symbol at risk on every nightly write)."""
    from agent.refresh import _write_bars_with_retry

    d = date(2026, 6, 1)
    store.insert("daily_bars", [
        _bar_row("AAA", d, close=10.0),                       # outside payload
        _bar_row("AAA", d + timedelta(days=1), close=11.0),   # outside payload
        _bar_row("AAA", d + timedelta(days=2), close=99.0),   # stale, replaced
    ], returning=False)
    payload = [_bar_row("AAA", d + timedelta(days=2), close=12.0),
               _bar_row("AAA", d + timedelta(days=3), close=13.0)]
    assert _write_bars_with_retry(store, "AAA", payload) == (2, 0)
    rows = {str(r["date"])[:10]: r["close"]
            for r in store.select("daily_bars", filters={"symbol": "AAA"})}
    assert rows == {"2026-06-01": 10.0, "2026-06-02": 11.0,   # untouched
                    "2026-06-03": 12.0, "2026-06-04": 13.0}   # replaced/added


def test_ingest_history_reports_holes_and_preserves_history(store, monkeypatch):
    """A symbol whose write fails every attempt is a HOLE: counted in the
    summary (with the symbol named), and its rows OUTSIDE the payload survive
    — the pre-fix behavior silently vaporized them with the range delete."""
    import time as _time
    from datetime import datetime as _dt
    from types import SimpleNamespace

    import agent.refresh as r

    monkeypatch.setattr(_time, "sleep", lambda s: None)  # skip retry backoff

    class _FailSymbolStore(_WrapStore):
        def insert(self, table, rows, *, returning=True):
            rl = [rows] if isinstance(rows, dict) else list(rows)
            if table == "daily_bars" and any(x.get("symbol") == "BBB"
                                             for x in rl):
                raise ConnectionError("proxy reset")
            return self._inner.insert(table, rows, returning=returning)

    store.insert("daily_bars", [_bar_row("BBB", date(2026, 1, 5), close=5.0)],
                 returning=False)
    bar = SimpleNamespace(timestamp=_dt(2026, 6, 1, 5, 0), open=1.0, high=1.0,
                          low=1.0, close=1.0, volume=10.0, trade_count=1)
    b = SimpleNamespace(data=SimpleNamespace(
        get_stock_bars=lambda req: SimpleNamespace(
            data={"AAA": [bar], "BBB": [bar]})))
    out = r._ingest_history_batched(_FailSymbolStore(store), b,
                                    ["AAA", "BBB"], max_days=30)
    assert out["bars_upserted"] == 1 and out["errors"] == 1
    assert out["holes"] == 1 and out["hole_symbols"] == ["BBB"]
    assert len(store.select("daily_bars", filters={"symbol": "AAA"})) == 1
    # BBB's history outside the payload survived every failed attempt
    assert [str(x["date"])[:10] for x in
            store.select("daily_bars", filters={"symbol": "BBB"})] == \
        ["2026-01-05"]


def test_retry_after_partial_insert_survives_duplicate_keys(store, monkeypatch):
    """Worst case: half the batch lands, the connection dies, and the retry's
    delete silently misses — the duplicate-skipping insert must converge on
    exactly the payload (no duplicate-key death, no doubled rows)."""
    import time as _time

    from agent.refresh import _write_bars_with_retry

    monkeypatch.setattr(_time, "sleep", lambda s: None)

    class _PartialThenDeadDeletes(_WrapStore):
        def __init__(self, inner):
            super().__init__(inner)
            self.insert_calls = 0
            self.delete_calls = 0

        def insert(self, table, rows, *, returning=True):
            if table == "daily_bars" and isinstance(rows, list):
                self.insert_calls += 1
                if self.insert_calls == 1:  # half lands, then the wire drops
                    self._inner.insert(table, rows[:1], returning=returning)
                    raise ConnectionError("reset mid-batch")
            return self._inner.insert(table, rows, returning=returning)

        def delete(self, table, filters):
            self.delete_calls += 1
            if self.delete_calls > 1:
                return None  # later deletes silently miss (worst case)
            return self._inner.delete(table, filters)

    payload = [_bar_row("CCC", date(2026, 6, 1), close=1.0),
               _bar_row("CCC", date(2026, 6, 2), close=2.0)]
    # (1, 1): one fresh insert + one payload row retained from the partial
    # first attempt — the dupe-skip converged, and the miss is surfaced
    assert _write_bars_with_retry(_PartialThenDeadDeletes(store), "CCC",
                                  payload) == (1, 1)
    rows = store.select("daily_bars", filters={"symbol": "CCC"})
    assert sorted(str(x["date"])[:10] for x in rows) == \
        ["2026-06-01", "2026-06-02"]
    assert len(rows) == 2  # each date exactly once


def _fake_bars_broker(bars_by_symbol):
    """A broker whose data client answers every history batch with the canned
    per-symbol bar lists (SimpleNamespace bars, like alpaca-py returns)."""
    from types import SimpleNamespace

    return SimpleNamespace(data=SimpleNamespace(
        get_stock_bars=lambda req: SimpleNamespace(data=bars_by_symbol)))


def _fake_bar(d, close=1.0):
    from datetime import datetime as _dt, time as _time
    from types import SimpleNamespace

    return SimpleNamespace(timestamp=_dt.combine(d, _time(5, 0)), open=close,
                           high=close, low=close, close=close, volume=10.0,
                           trade_count=1)


def test_zombie_dates_detected_and_counted_never_deleted(store, caplog):
    """M2: exact-date deletes never prune a date the vendor RETRACTED — the
    ingest must log + count such zombies after a successful write, and must
    NOT delete them (a fetch hiccup can't be data destruction on the sacred
    table)."""
    import logging

    import agent.refresh as r

    zombie_day = date.today() - timedelta(days=3)
    payload_day = date.today() - timedelta(days=1)
    store.insert("daily_bars", [_bar_row("AAA", zombie_day, close=7.77)],
                 returning=False)
    b = _fake_bars_broker({"AAA": [_fake_bar(payload_day, close=2.0)]})
    with caplog.at_level(logging.WARNING, logger="agent.refresh"):
        out = r._ingest_history_batched(store, b, ["AAA"], max_days=30)
    assert out["bars_upserted"] == 1 and out["holes"] == 0
    assert out["zombie_dates"] == 1 and out["zombie_symbols"] == ["AAA"]
    assert f"ZOMBIE dates for AAA: ['{zombie_day}']" in caplog.text
    # detection only: the retracted-date row is untouched
    rows = store.select("daily_bars",
                        filters={"symbol": "AAA", "date": zombie_day})
    assert len(rows) == 1 and rows[0]["close"] == 7.77


def test_no_zombies_when_payload_covers_the_window(store):
    """The steady state is silent: every DB date in the window re-fetched →
    zero zombies, zero stale_retained."""
    import agent.refresh as r

    d = date.today() - timedelta(days=1)
    store.insert("daily_bars", [_bar_row("AAA", d, close=2.0)], returning=False)
    b = _fake_bars_broker({"AAA": [_fake_bar(d, close=2.5)]})
    out = r._ingest_history_batched(store, b, ["AAA"], max_days=30)
    assert out["zombie_dates"] == 0 and out["zombie_symbols"] == []
    assert out["stale_retained"] == 0 and out["bars_upserted"] == 1


def test_delete_ineffective_surfaces_stale_retained(store, caplog):
    """L1: rows still present immediately after their dates were deleted mean
    the delete silently did nothing — warn, count them as stale_retained, and
    keep bars_upserted honest (real inserts, not payload size)."""
    import logging

    import agent.refresh as r

    class _DeadDeletes(_WrapStore):
        def delete(self, table, filters):
            return None  # silently ineffective — the failure mode under test

    d1 = date.today() - timedelta(days=2)
    d2 = date.today() - timedelta(days=1)
    store.insert("daily_bars", [_bar_row("AAA", d1, close=99.0)],
                 returning=False)
    b = _fake_bars_broker({"AAA": [_fake_bar(d1, close=12.0),
                                   _fake_bar(d2, close=13.0)]})
    with caplog.at_level(logging.WARNING, logger="agent.refresh"):
        out = r._ingest_history_batched(_DeadDeletes(store), b, ["AAA"],
                                        max_days=30)
    assert out["bars_upserted"] == 1      # only the row that really inserted
    assert out["stale_retained"] == 1     # the one the dead delete kept
    assert out["holes"] == 0 and out["zombie_dates"] == 0
    assert "DELETE INEFFECTIVE for AAA" in caplog.text
    rows = {str(x["date"])[:10]: x["close"]
            for x in store.select("daily_bars", filters={"symbol": "AAA"})}
    assert rows == {str(d1): 99.0, str(d2): 13.0}  # stale row visibly retained


def test_hourly_summary_names_hole_symbols(store, monkeypatch):
    """L5: the hourly top-up summary names the hole symbols (same 25-cap list
    the nightly carries), not just their count."""
    import agent.broker as broker
    import agent.refresh as r

    monkeypatch.setattr(broker, "enabled", lambda: True)
    monkeypatch.setattr(broker, "Broker", lambda: object())
    monkeypatch.setattr(r, "_ingest_history_batched",
                        lambda store, b, syms, *, max_days, chunk=100: {
                            "bars_upserted": 5, "batches": 1, "errors": 1,
                            "holes": 1, "hole_symbols": ["BBB"]})
    monkeypatch.setattr(r, "_iv_snapshots_alpaca", lambda store, b, syms: {})
    monkeypatch.setattr(r, "_news_alpaca", lambda store, syms, **kw: {})
    monkeypatch.setattr(r, "_corp_actions_alpaca", lambda store, syms, **kw: {})
    monkeypatch.setattr(r, "_r2_merge_sync", lambda syms, **kw: "skipped")
    out = r.refresh_alpaca(symbols=["AAA", "BBB"])
    assert out["bar_holes"] == 1 and out["hole_symbols"] == ["BBB"]


# ── F18: the R2 archive never keeps a partial intraday bar ──


class _FakeS3:
    """The two S3 calls _r2_merge_sync makes, over a plain dict."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        import io

        if Key not in self.objects:
            raise KeyError(Key)  # the sync catches broadly: missing = fresh
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture()
def r2(monkeypatch):
    import socket

    import boto3

    prev = socket.getdefaulttimeout()
    for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
              "R2_ENDPOINT", "R2_BUCKET"):
        monkeypatch.setenv(k, "test")
    fake = _FakeS3()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
    yield fake
    socket.setdefaulttimeout(prev)  # _bound_network widened it for "R2"


def _r2_dates(fake, sym):
    import io

    import pandas as pd

    df = pd.read_parquet(io.BytesIO(fake.objects[f"bars/{sym}.parquet"]))
    return {str(d)[:10] for d in df["date"]}


def _r2_manifest(fake):
    import json as _json

    return _json.loads(fake.objects["manifest.json"])


def test_r2_sync_excludes_intraday_today(store, r2):
    from agent.refresh import _r2_merge_sync, _today_et

    today = _today_et()
    store.insert("daily_bars", [
        _bar_row("AAA", today - timedelta(days=2), close=1.0),
        _bar_row("AAA", today - timedelta(days=1), close=2.0),
        _bar_row("AAA", today, close=3.0),  # in-progress bar (hourly ingest)
    ], returning=False)
    out = _r2_merge_sync(["AAA"])
    assert out == {"synced": 1, "unchanged": 0, "errors": 0}
    assert str(today) not in _r2_dates(r2, "AAA")  # never enters the archive
    assert _r2_manifest(r2)["AAA"]["db_max"] == str(today - timedelta(days=1))
    # re-running intraday stays a no-op — the partial bar never flips db_max
    assert _r2_merge_sync(["AAA"]) == {"synced": 0, "unchanged": 1, "errors": 0}


def test_r2_sync_includes_today_when_flagged_post_close(store, r2):
    from agent.refresh import _r2_merge_sync, _today_et

    today = _today_et()
    store.insert("daily_bars", [
        _bar_row("AAA", today - timedelta(days=1), close=2.0),
        _bar_row("AAA", today, close=3.0),
    ], returning=False)
    out = _r2_merge_sync(["AAA"], include_today_et=True)  # the nightly path
    assert out == {"synced": 1, "unchanged": 0, "errors": 0}
    assert str(today) in _r2_dates(r2, "AAA")
    assert _r2_manifest(r2)["AAA"]["db_max"] == str(today)


def test_r2_sync_same_date_correction_triggers_resync(store, r2):
    """A changed close at an unchanged db_max must re-sync (the db_fp content
    fingerprint) — the pre-fix date-only check read it as 'unchanged' forever."""
    from agent.refresh import _r2_merge_sync, _today_et

    today = _today_et()
    d1 = today - timedelta(days=2)
    d2 = today - timedelta(days=1)
    store.insert("daily_bars", [_bar_row("AAA", d1, close=1.0),
                                _bar_row("AAA", d2, close=2.0)],
                 returning=False)
    assert _r2_merge_sync(["AAA"])["synced"] == 1
    assert _r2_merge_sync(["AAA"])["unchanged"] == 1  # steady state

    store.update("daily_bars", {"symbol": "AAA", "date": d2},
                 {"close": 9.99}, returning=False)     # same-date correction
    out = _r2_merge_sync(["AAA"])
    assert out == {"synced": 1, "unchanged": 0, "errors": 0}
    import io

    import pandas as pd

    df = pd.read_parquet(io.BytesIO(r2.objects["bars/AAA.parquet"]))
    assert float(df[df["date"] == str(d2)]["close"].iloc[0]) == 9.99
    assert _r2_manifest(r2)["AAA"]["db_fp"] == [9.99, 100.0]


def test_r2_sync_legacy_manifest_without_fingerprint(store, r2):
    """Pre-fingerprint manifest entries keep working: same-max stays unchanged
    (the old rule), and the next real sync backfills db_fp."""
    import json as _json

    from agent.refresh import _r2_merge_sync, _today_et

    today = _today_et()
    d1, d2, d3 = (today - timedelta(days=n) for n in (3, 2, 1))
    store.insert("daily_bars", [_bar_row("AAA", d1, close=1.0),
                                _bar_row("AAA", d2, close=2.0)],
                 returning=False)
    r2.objects["manifest.json"] = _json.dumps({
        "AAA": {"rows": 2, "max_date": str(d2),
                "db_rows": 2, "db_max": str(d2)}}).encode()  # no db_fp
    assert _r2_merge_sync(["AAA"]) == {"synced": 0, "unchanged": 1, "errors": 0}

    store.insert("daily_bars", [_bar_row("AAA", d3, close=3.0)],
                 returning=False)
    assert _r2_merge_sync(["AAA"])["synced"] == 1
    m = _r2_manifest(r2)["AAA"]
    assert m["db_max"] == str(d3) and m["db_fp"] == [3.0, 100.0]


def test_todays_session_over_truth_table(monkeypatch):
    """M1: the include-today archive gate keys off the trading CALENDAR
    (today's session has ended), not the wall-clock session tag — the fixed
    00:45 UTC nightly lands at 19:45 ET all winter, when session() still
    reads 'extended' until 20:00, which made the gate a seasonal no-op."""
    from datetime import datetime as _dt
    from types import SimpleNamespace

    import agent.refresh as r

    def _b(cal):
        def _cal_day(day):
            if isinstance(cal, Exception):
                raise cal
            return cal
        return SimpleNamespace(calendar_day=_cal_day)

    def _at(hh, mm):
        monkeypatch.setattr(r, "_now_et", lambda: _dt(2026, 1, 14, hh, mm))

    day = date(2026, 1, 14)  # a Wednesday under EST (winter)
    cal = {"date": str(day), "open": _dt(2026, 1, 14, 9, 30),
           "close": _dt(2026, 1, 14, 16, 0)}
    _at(19, 45)  # the winter nightly landing: post-close, before 20:00
    assert r._todays_session_over(_b(cal)) is True   # was False pre-fix
    _at(16, 0)   # exactly at the close — the session has ended
    assert r._todays_session_over(_b(cal)) is True
    _at(14, 0)   # mid-session — today's bar is still in progress
    assert r._todays_session_over(_b(cal)) is False
    _at(4, 30)   # pre-market
    assert r._todays_session_over(_b(cal)) is False
    _at(19, 45)
    assert r._todays_session_over(_b(None)) is True  # weekend/holiday: no session
    assert r._todays_session_over(
        _b(RuntimeError("calendar down"))) is False  # fail-closed on errors
    # half-day: the calendar's 13:00 close decides, not a hardcoded 16:00
    half = {"date": str(day), "open": _dt(2026, 1, 14, 9, 30),
            "close": _dt(2026, 1, 14, 13, 0)}
    _at(13, 30)
    assert r._todays_session_over(_b(half)) is True
    _at(12, 0)
    assert r._todays_session_over(_b(half)) is False


def test_nightly_passes_calendar_gated_today_flag(monkeypatch):
    """refresh_alpaca_market forwards include_today_et from the CALENDAR
    check: today's bar joins the archive once today's ET session has ended
    (including the 19:45-ET winter landing), never mid-session."""
    from datetime import datetime as _dt

    import agent.broker as broker
    import agent.edgar as edgar
    import agent.refresh as r

    class _MarketBroker:
        def __init__(self, cal):
            self._cal = cal

        def list_assets(self, optionable=False):
            return [{"symbol": "AAA"}]

        def calendar_day(self, day):
            return self._cal

    monkeypatch.setattr(broker, "enabled", lambda: True)
    monkeypatch.setattr(r, "_alpaca_latest_daily",
                        lambda b, catalog, lookback_days=7: [
                            {"symbol": "AAA", "close": 100.0, "volume": 1e6}])
    monkeypatch.setattr(r, "_keep_symbols_rest", lambda: set())
    monkeypatch.setattr(r, "_ingest_history_batched",
                        lambda store, b, syms, *, max_days, chunk=100: {})
    monkeypatch.setattr(r, "_corp_actions_alpaca",
                        lambda store, symbols, **kw: {})
    monkeypatch.setattr(edgar, "ingest", lambda store, symbols=None: {})

    captured = {}

    def fake_sync(syms, *, include_today_et=False):
        captured["include_today_et"] = include_today_et
        return "ok"

    monkeypatch.setattr(r, "_r2_merge_sync", fake_sync)
    cal = {"date": "2026-01-14", "open": _dt(2026, 1, 14, 9, 30),
           "close": _dt(2026, 1, 14, 16, 0)}
    for now, expect in ((_dt(2026, 1, 14, 19, 45), True),   # EST nightly
                        (_dt(2026, 1, 14, 14, 0), False)):  # midday manual run
        monkeypatch.setattr(r, "_now_et", lambda now=now: now)
        monkeypatch.setattr(broker, "Broker", lambda c=cal: _MarketBroker(c))
        r.refresh_alpaca_market(top_n=1)
        assert captured["include_today_et"] is expect


# ── F19: the IV data bank only snapshots during the regular session ──


class _SessionBroker:
    def __init__(self, sess):
        self._sess = sess

    def session(self):
        if isinstance(self._sess, Exception):
            raise self._sess
        return self._sess


def test_iv_pass_skipped_outside_regular_session(store, monkeypatch):
    """A 07:00 pre-market run must never lock crossed pre-open OPRA marks in
    as the day's canonical IV row — the chain is not even fetched."""
    import agent.options_data as od
    from agent.refresh import _iv_snapshots_alpaca

    calls = []
    monkeypatch.setattr(od, "get_summary",
                        lambda sym, dte_max=45: calls.append(sym))
    for sess in ("extended", "closed"):
        out = _iv_snapshots_alpaca(store, _SessionBroker(sess), ["SPY"])
        assert out == {"written": 0, "skipped_session": sess}
    assert calls == []
    assert store.select("desk_options_snap") == []


def test_iv_pass_fails_closed_on_unknown_session(store):
    from agent.refresh import _iv_snapshots_alpaca

    out = _iv_snapshots_alpaca(store,
                               _SessionBroker(RuntimeError("clock down")),
                               ["SPY"])
    assert out["written"] == 0
    assert out["skipped_session"].startswith("unknown")
    assert store.select("desk_options_snap") == []


def test_iv_pass_writes_in_regular_session_with_captured_at(store, monkeypatch):
    import agent.options_data as od
    from agent.refresh import _iv_snapshots_alpaca

    monkeypatch.setattr(od, "get_summary",
                        lambda sym, dte_max=45: {
                            "available": True, "symbol": sym, "spot": 100.0,
                            "atm_iv": 0.2, "expected_move_pct": 1.0,
                            "skew_25d": 0.01, "dte": 10,
                            "expiry": "2026-08-21"})
    out = _iv_snapshots_alpaca(store, _SessionBroker("regular"), ["SPY"])
    assert out == {"written": 1, "skipped_have_today": 0}
    rows = store.select("desk_options_snap", filters={"symbol": "SPY"})
    assert len(rows) == 1
    assert rows[0]["captured_at"] is not None    # the RTH receipt
    # rerun same day: first write of the day still wins, the fetch is skipped
    out2 = _iv_snapshots_alpaca(store, _SessionBroker("regular"), ["SPY"])
    assert out2 == {"written": 0, "skipped_have_today": 1}
