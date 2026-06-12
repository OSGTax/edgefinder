"""Tests for the resumable minute-bar backfill (stubbed client + fake S3)."""

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from edgefinder.data.minutestore import MinuteStore
from scripts.backfill_minute_bars import (
    MAX_BARS_PER_CALL,
    MONTHS_PER_CALL,
    est_worst_case_bars,
    fetch_minute_df,
    month_chunks,
    plan_chunks,
    run,
)
from tests.test_minutestore import FakeS3

ET = ZoneInfo("America/New_York")


class Agg:
    def __init__(self, ts_ms: int, px: float = 100.0):
        self.timestamp = ts_ms
        self.open = self.high = self.low = self.close = px
        self.volume = 1000.0


def make_session_bars(d: date) -> list[Agg]:
    """A tiny session: open, midday, last RTH minute + one POST-market bar."""
    out = []
    for hh, mm in ((9, 30), (12, 0), (15, 59), (16, 30)):
        dt = datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)
        out.append(Agg(int(dt.timestamp()) * 1000))
    return out


def weekdays(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)
            if (start + timedelta(days=i)).weekday() < 5]


class FakeClient:
    """get_aggs over pre-built per-symbol bars; records every call."""

    def __init__(self, symbols: list[str], start: date, end: date,
                 fail_symbols: set[str] = frozenset(),
                 truncate: bool = False):
        self.calls: list[tuple] = []
        self.fail_symbols = set(fail_symbols)
        self.truncate = truncate
        self.bars = {
            s: [a for d in weekdays(start, end) for a in make_session_bars(d)]
            for s in symbols
        }

    def get_aggs(self, ticker, multiplier, timespan, from_, to, limit):
        assert (multiplier, timespan) == (1, "minute")
        self.calls.append((ticker, from_, to))
        if ticker in self.fail_symbols:
            raise ConnectionError("boom")
        if self.truncate:
            return [Agg(0)] * MAX_BARS_PER_CALL
        lo = int(datetime.strptime(from_, "%Y-%m-%d")
                 .replace(tzinfo=ET).timestamp()) * 1000
        hi = int((datetime.strptime(to, "%Y-%m-%d") + timedelta(days=1))
                 .replace(tzinfo=ET).timestamp()) * 1000
        return [a for a in self.bars.get(ticker, []) if lo <= a.timestamp < hi]


@pytest.fixture()
def store():
    s = MinuteStore.__new__(MinuteStore)
    s.bucket = "test"
    s._s3 = FakeS3()
    return s


class TestChunkMath:
    def test_chunk_size_respects_the_50k_cap(self):
        # extended-session worst case: a bigger chunk could truncate
        assert est_worst_case_bars(MONTHS_PER_CALL) <= MAX_BARS_PER_CALL
        assert est_worst_case_bars(MONTHS_PER_CALL + 1) > MAX_BARS_PER_CALL

    def test_month_chunks_are_aligned_and_ordered(self):
        months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
        assert month_chunks(months, 2) == [
            ["2024-01", "2024-02"], ["2024-03", "2024-04"], ["2024-05"]]

    def test_plan_skips_only_complete_months(self):
        needed = ["2024-01", "2024-02", "2024-03", "2024-04"]
        manifest_months = {
            "2024-01": {"rows": 9, "complete": True},
            "2024-02": {"rows": 9, "complete": True},
            "2024-03": {"rows": 3, "complete": False},   # partial top-up
        }
        assert plan_chunks(needed, manifest_months, per_call=2) == [
            ["2024-03", "2024-04"]]
        # a month merely PRESENT (e.g. nightly-seeded) is not skippable
        assert plan_chunks(needed, {"2024-01": {"rows": 1}}, per_call=2) == [
            ["2024-01", "2024-02"], ["2024-03", "2024-04"]]


class TestBackfillRun:
    def test_backfill_writes_rth_only_and_is_resumable(self, store):
        start, end = date(2024, 1, 1), date(2024, 4, 30)
        client = FakeClient(["AAA"], start, end)
        out = run(client, store, ["AAA"], start, end)
        assert out["failed"] == []
        # 4 months / 2-month chunks = 2 calls
        assert len(client.calls) == 2
        bars = store.load_minute_bars(["AAA"], start, end)["AAA"]
        n_days = len(weekdays(start, end))
        assert len(bars) == n_days * 3          # the 16:30 bar was dropped
        months = store.months("AAA")
        assert sorted(months) == ["2024-01", "2024-02", "2024-03", "2024-04"]
        assert all(m["complete"] for m in months.values())

        # rerun: every month complete -> zero fetches (resume = free)
        client2 = FakeClient(["AAA"], start, end)
        out2 = run(client2, store, ["AAA"], start, end)
        assert client2.calls == [] and out2["rows"] == 0

    def test_partial_end_month_is_refetched_on_resume(self, store):
        start = date(2024, 1, 1)
        client = FakeClient(["AAA"], start, date(2024, 4, 30))
        run(client, store, ["AAA"], start, date(2024, 4, 15))
        assert store.months("AAA")["2024-04"]["complete"] is False
        # extend through month end: only the Mar-Apr chunk refetches
        client2 = FakeClient(["AAA"], start, date(2024, 4, 30))
        run(client2, store, ["AAA"], start, date(2024, 4, 30))
        assert client2.calls == [("AAA", "2024-03-01", "2024-04-30")]
        assert store.months("AAA")["2024-04"]["complete"] is True

    def test_failed_symbol_is_loud_but_does_not_stop_the_run(self, store):
        start, end = date(2024, 1, 1), date(2024, 2, 29)
        client = FakeClient(["AAA", "BBB"], start, end, fail_symbols={"AAA"})
        out = run(client, store, ["AAA", "BBB"], start, end)
        assert out["failed"] == ["AAA"]
        assert "BBB" in store.load_minute_bars(["BBB"], start, end)
        # nothing was recorded as complete for the failed symbol
        assert store.months("AAA") == {}

    def test_truncated_response_fails_loudly(self, store):
        start, end = date(2024, 1, 1), date(2024, 2, 29)
        client = FakeClient(["AAA"], start, end, truncate=True)
        with pytest.raises(RuntimeError, match="cap"):
            fetch_minute_df(client, "AAA", start, end)
        out = run(client, store, ["AAA"], start, end)
        assert out["failed"] == ["AAA"]

    def test_dry_run_makes_no_calls_and_no_writes(self, store):
        out = run(None, store, ["AAA"], date(2024, 1, 1), date(2024, 2, 29),
                  dry_run=True)
        assert out["dry_run"] is True and out["failed"] == []
        assert store._s3.objects == {}

    def test_empty_month_for_unlisted_symbol_marked_complete(self, store):
        # symbol with no bars at all (pre-listing): months recorded rows=0
        client = FakeClient([], date(2024, 1, 1), date(2024, 1, 2))
        run(client, store, ["NEW"], date(2024, 1, 1), date(2024, 2, 29))
        months = store.months("NEW")
        assert months["2024-01"] == {"rows": 0, "min_ts": None,
                                     "max_ts": None, "complete": True}
        # no parquet object was uploaded for the empty months
        assert all(not k.startswith("minute/NEW/")
                   for k in store._s3.objects)


class TestMainGates:
    def test_empty_menu_exits_green(self, tmp_path, capsys):
        from scripts import backfill_minute_bars as mb

        menu = tmp_path / "menu.json"
        menu.write_text(json.dumps({"symbols": []}))
        mb.main(["--menu", str(menu)])         # returns, no SystemExit
        assert "no symbols yet" in capsys.readouterr().out

    def test_failed_symbol_exits_nonzero(self, store, tmp_path, monkeypatch):
        from scripts import backfill_minute_bars as mb

        start, end = "2024-01-01", "2024-01-31"
        client = FakeClient(["AAA"], date(2024, 1, 1), date(2024, 1, 31),
                            fail_symbols={"AAA"})
        monkeypatch.setattr(mb, "MinuteStore", lambda: store)
        monkeypatch.setattr(mb, "_make_client", lambda: client)
        with pytest.raises(SystemExit) as exc:
            mb.main(["--symbols", "AAA", "--start", start, "--end", end])
        assert exc.value.code == 1

    def test_dry_run_without_r2_secrets_plans_from_empty(self, monkeypatch,
                                                         capsys):
        from scripts import backfill_minute_bars as mb

        for k in ("R2_BUCKET", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(k, raising=False)
        mb.main(["--symbols", "AAA,BBB", "--start", "2024-01-01",
                 "--end", "2024-02-29", "--dry-run"])
        out = capsys.readouterr().out
        assert "assumes an EMPTY store" in out
        assert "AAA" in out and "BBB" in out
