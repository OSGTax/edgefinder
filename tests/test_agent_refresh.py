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


# ── backfill_r2_history (deep-history one-time backfill) ──


def test_backfill_needed_pure_logic():
    from datetime import date

    from agent.refresh import _backfill_needed

    start = date(2016, 1, 4)
    assert _backfill_needed(None, start) is True              # no archive yet
    assert _backfill_needed("2021-06-02", start) is True       # archive too shallow
    assert _backfill_needed("2016-01-04", start) is False      # exactly covers start
    assert _backfill_needed("2005-01-03", start) is False      # legacy ETF, deeper already


def test_r2_client_none_without_creds(monkeypatch):
    from agent.refresh import _r2_client

    for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    assert _r2_client() is None


def test_backfill_errors_without_alpaca_keys(monkeypatch):
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    monkeypatch.setattr(broker_mod, "enabled", lambda: False)
    out = r.backfill_r2_history(["AAPL"], start=date(2016, 1, 4))
    assert "error" in out and "Alpaca" in out["error"]


def test_backfill_errors_without_r2_creds(monkeypatch):
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    monkeypatch.setattr(broker_mod, "enabled", lambda: True)
    monkeypatch.setattr(r, "_r2_client", lambda: None)
    out = r.backfill_r2_history(["AAPL"], start=date(2016, 1, 4))
    assert "error" in out and "R2" in out["error"]


def test_r2_client_degrades_on_construction_failure(monkeypatch):
    """A malformed endpoint / bad credential shape must not raise out of
    _r2_client — every caller treats None as "skip archival," never a crash."""
    import agent.refresh as r

    for k, v in (("R2_ACCESS_KEY_ID", "k"), ("R2_SECRET_ACCESS_KEY", "s"),
                ("R2_ENDPOINT", "https://bad"), ("R2_BUCKET", "b")):
        monkeypatch.setenv(k, v)

    class _BrokenBoto3:
        @staticmethod
        def client(*a, **kw):
            raise ValueError("malformed endpoint")

    import sys
    monkeypatch.setitem(sys.modules, "boto3", _BrokenBoto3)
    assert r._r2_client() is None


class _FakeS3:
    """In-memory stand-in for the boto3 R2 client — objects keyed by name."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):
        from types import SimpleNamespace

        if Key not in self.objects:
            raise KeyError(Key)  # any exception = "not found" to the caller
        return {"Body": SimpleNamespace(read=lambda: self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body if isinstance(Body, (bytes, bytearray)) else Body.encode()


def _seed_parquet(s3, symbol, rows):
    import io

    import pandas as pd

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    s3.objects[f"bars/{symbol}.parquet"] = buf.getvalue()


class _FakeStockData:
    """Stands in for Broker.data — returns fixtured bars per requested symbol.
    ``always_raise=True`` simulates every attempt of a batch failing (retry
    exhaustion)."""

    def __init__(self, bars_by_symbol, always_raise=False):
        self.bars_by_symbol = bars_by_symbol
        self.requested_batches: list[list[str]] = []
        self.always_raise = always_raise
        self.calls = 0

    def get_stock_bars(self, req):
        from types import SimpleNamespace

        syms = req.symbol_or_symbols
        syms = syms if isinstance(syms, list) else [syms]
        self.requested_batches.append(syms)
        self.calls += 1
        if self.always_raise:
            raise ConnectionError("simulated Alpaca outage")
        return SimpleNamespace(data={s: self.bars_by_symbol.get(s, []) for s in syms})


def _fake_bar(day, price=100.0):
    from datetime import datetime
    from types import SimpleNamespace

    return SimpleNamespace(timestamp=datetime(*day), open=price, high=price + 1,
                           low=price - 1, close=price, volume=1_000_000.0,
                           trade_count=100.0)


def test_backfill_skips_symbol_already_covered(monkeypatch):
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    s3 = _FakeS3()
    # SPY's archive already reaches back to 2016-01-04 — nothing to fetch.
    _seed_parquet(s3, "SPY", [
        {"date": "2016-01-04", "open": 200.0, "high": 201, "low": 199, "close": 200.5,
         "volume": 1_000_000},
        {"date": "2021-06-02", "open": 400.0, "high": 401, "low": 399, "close": 400.5,
         "volume": 2_000_000},
    ])
    fake_data = _FakeStockData({})
    monkeypatch.setattr(broker_mod, "enabled", lambda: True)
    monkeypatch.setattr(broker_mod, "Broker",
                        lambda: type("B", (), {"data": fake_data})())
    monkeypatch.setattr(r, "_r2_client", lambda: (s3, "test-bucket"))

    out = r.backfill_r2_history(["SPY"], start=date(2016, 1, 4))
    assert out["already_covered"] == 1
    assert out["to_backfill"] == 0
    assert fake_data.requested_batches == []  # no Alpaca call spent on it


def test_backfill_fetches_and_merges_for_a_shallow_symbol(monkeypatch):
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    s3 = _FakeS3()
    # AAPL's archive only reaches back to 2021-06-02 — needs the 2016-2021 gap.
    _seed_parquet(s3, "AAPL", [
        {"date": "2021-06-02", "open": 150.0, "high": 151, "low": 149, "close": 150.5,
         "volume": 1_000_000},
    ])
    fake_data = _FakeStockData({"AAPL": [_fake_bar((2016, 1, 4), 90.0),
                                         _fake_bar((2016, 1, 5), 91.0)]})
    monkeypatch.setattr(broker_mod, "enabled", lambda: True)
    monkeypatch.setattr(broker_mod, "Broker",
                        lambda: type("B", (), {"data": fake_data})())
    monkeypatch.setattr(r, "_r2_client", lambda: (s3, "test-bucket"))

    out = r.backfill_r2_history(["AAPL"], start=date(2016, 1, 4),
                                end=date(2021, 6, 1))
    assert out["to_backfill"] == 1 and out["synced"] == 1 and out["errors"] == 0
    assert fake_data.requested_batches == [["AAPL"]]

    import io

    import pandas as pd

    merged = pd.read_parquet(io.BytesIO(s3.objects["bars/AAPL.parquet"]))
    # grow-only: the new 2016 rows PLUS the pre-existing 2021 row survive.
    assert list(merged["date"]) == ["2016-01-04", "2016-01-05", "2021-06-02"]
    manifest = __import__("json").loads(s3.objects["manifest.json"])
    assert manifest["AAPL"]["min_date"] == "2016-01-04"


def test_backfill_dry_run_makes_no_alpaca_calls(monkeypatch):
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    s3 = _FakeS3()
    fake_data = _FakeStockData({})
    monkeypatch.setattr(broker_mod, "enabled", lambda: True)
    monkeypatch.setattr(broker_mod, "Broker",
                        lambda: type("B", (), {"data": fake_data})())
    monkeypatch.setattr(r, "_r2_client", lambda: (s3, "test-bucket"))

    out = r.backfill_r2_history(["AAPL", "MSFT"], start=date(2016, 1, 4), dry_run=True)
    assert out["dry_run"] is True and out["to_backfill"] == 2
    assert fake_data.requested_batches == []
    assert "manifest.json" not in s3.objects


def test_backfill_retry_exhaustion_degrades_without_crashing(monkeypatch):
    """All 3 Alpaca attempts failing must count as errors and finish the
    run, not raise (regression guard for the reviewed retry-loop path)."""
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    monkeypatch.setattr("time.sleep", lambda s: None)  # skip real backoff waits
    s3 = _FakeS3()
    fake_data = _FakeStockData({}, always_raise=True)
    monkeypatch.setattr(broker_mod, "enabled", lambda: True)
    monkeypatch.setattr(broker_mod, "Broker",
                        lambda: type("B", (), {"data": fake_data})())
    monkeypatch.setattr(r, "_r2_client", lambda: (s3, "test-bucket"))

    out = r.backfill_r2_history(["AAPL"], start=date(2016, 1, 4))
    assert fake_data.calls == 3            # exhausted all 3 attempts
    assert out["errors"] == 1 and out["synced"] == 0
    assert "bars/AAPL.parquet" not in s3.objects  # nothing written on total failure


def test_backfill_new_data_wins_on_overlapping_date():
    """merge_bar_frames conflict semantics, exercised through the actual
    backfill merge call: freshly-fetched rows must win over a stale existing
    row on the same date, not silently lose to it."""
    from datetime import date

    import pandas as pd

    from agent.refresh import merge_bar_frames

    existing = pd.DataFrame([
        {"date": "2018-05-01", "open": 1.0, "high": 2, "low": 0.5, "close": 1.5,
         "volume": 10},
    ])
    fresh_rows = [
        {"symbol": "X", "date": date(2018, 5, 1), "open": 1.0, "high": 2.2,
         "low": 0.5, "close": 1.65, "volume": 15, "source": "alpaca_daily"},
    ]
    merged = merge_bar_frames(existing, fresh_rows)
    assert len(merged) == 1
    assert merged["close"].iloc[0] == 1.65  # freshly-fetched row wins


def test_backfill_partial_batch_failure_counts_errors_and_continues(monkeypatch):
    """One symbol's R2 write failing must not stop siblings in the same
    batch from being written, and must be counted as an error."""
    from datetime import date

    import agent.broker as broker_mod
    import agent.refresh as r

    s3 = _FakeS3()
    real_put = s3.put_object

    def flaky_put(Bucket, Key, Body):
        if Key == "bars/BAD.parquet":
            raise ConnectionError("simulated R2 write failure")
        real_put(Bucket=Bucket, Key=Key, Body=Body)

    s3.put_object = flaky_put
    fake_data = _FakeStockData({
        "GOOD": [_fake_bar((2016, 1, 4), 50.0)],
        "BAD": [_fake_bar((2016, 1, 4), 60.0)],
    })
    monkeypatch.setattr(broker_mod, "enabled", lambda: True)
    monkeypatch.setattr(broker_mod, "Broker",
                        lambda: type("B", (), {"data": fake_data})())
    monkeypatch.setattr(r, "_r2_client", lambda: (s3, "test-bucket"))

    out = r.backfill_r2_history(["GOOD", "BAD"], start=date(2016, 1, 4),
                                end=date(2016, 2, 1))
    assert out["synced"] == 1 and out["errors"] == 1
    assert "bars/GOOD.parquet" in s3.objects
    assert "bars/BAD.parquet" not in s3.objects
