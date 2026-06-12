"""Tests for the R2 minute-bar store (against an in-memory fake S3)."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edgefinder.data.minutestore import (
    MANIFEST_KEY,
    MinuteStore,
    filter_rth,
    month_bounds,
    months_between,
    to_et,
)

ET = ZoneInfo("America/New_York")


class FakeS3:
    """The S3 calls MinuteStore makes, over a plain dict."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, **kw):
        contents = [{"Key": k, "Size": len(v)}
                    for k, v in self.objects.items() if k.startswith(Prefix)]
        return {"Contents": contents, "IsTruncated": False}


@pytest.fixture()
def store():
    s = MinuteStore.__new__(MinuteStore)      # skip env/boto3 __init__
    s.bucket = "test"
    s._s3 = FakeS3()
    return s


def ts_et(y, m, d, hh, mm) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=ET).timestamp())


def bars(times, close=100.0) -> pd.DataFrame:
    ts = [ts_et(*t) for t in times]
    return pd.DataFrame({
        "ts": ts,
        "open": [close] * len(ts), "high": [close + 1] * len(ts),
        "low": [close - 1] * len(ts), "close": [close] * len(ts),
        "volume": [1000.0] * len(ts),
    })


# a regular Tuesday session (2024-01-16)
RTH_DAY = [(2024, 1, 16, 9, 30), (2024, 1, 16, 12, 0), (2024, 1, 16, 15, 59)]
NON_RTH = [(2024, 1, 16, 4, 0), (2024, 1, 16, 9, 29),
           (2024, 1, 16, 16, 0), (2024, 1, 16, 19, 59)]


class TestHelpers:
    def test_filter_rth_drops_pre_and_post_market(self):
        df = bars(NON_RTH + RTH_DAY)
        out = filter_rth(df)
        assert len(out) == 3
        et = to_et(out["ts"])
        assert list(et.dt.strftime("%H:%M")) == ["09:30", "12:00", "15:59"]

    def test_to_et_roundtrip(self):
        df = bars([(2024, 7, 16, 9, 30)])      # EDT — DST must be honored
        et = to_et(df["ts"])
        assert et.iloc[0].hour == 9 and et.iloc[0].minute == 30

    def test_months_between_and_bounds(self):
        assert months_between(date(2023, 11, 15), date(2024, 2, 1)) == [
            "2023-11", "2023-12", "2024-01", "2024-02"]
        assert months_between(date(2024, 1, 1), date(2023, 1, 1)) == []
        assert month_bounds("2024-02") == (date(2024, 2, 1), date(2024, 2, 29))
        assert month_bounds("2023-12") == (date(2023, 12, 1), date(2023, 12, 31))


class TestMinuteStore:
    def test_sync_filters_rth_and_writes_month_object(self, store):
        store.sync_symbol_month(bars(NON_RTH + RTH_DAY), "AAA", "2024-01")
        assert "minute/AAA/2024-01.parquet" in store._s3.objects
        assert MANIFEST_KEY in store._s3.objects
        out = store.load_minute_bars(["AAA"], date(2024, 1, 1),
                                     date(2024, 1, 31))
        assert len(out["AAA"]) == 3            # pre/post-market dropped

    def test_sync_drops_rows_outside_the_month(self, store):
        df = bars(RTH_DAY + [(2024, 2, 6, 10, 0)])
        store.sync_symbol_month(df, "AAA", "2024-01")
        out = store.load_minute_bars(["AAA"], date(2024, 1, 1),
                                     date(2024, 12, 31))
        assert len(out["AAA"]) == 3            # the Feb bar was not smuggled in

    def test_merge_never_shrinks(self, store):
        store.sync_symbol_month(bars(RTH_DAY), "AAA", "2024-01")
        # a later sync carrying ONLY a new bar must preserve the old three
        store.sync_symbol_month(bars([(2024, 1, 17, 9, 30)]), "AAA", "2024-01")
        out = store.load_minute_bars(["AAA"], date(2024, 1, 1),
                                     date(2024, 1, 31))["AAA"]
        assert len(out) == 4
        assert list(out["ts"]) == sorted(out["ts"])
        # an EMPTY sync is a no-op, never a delete
        entry = store.sync_symbol_month(bars([]), "AAA", "2024-01")
        assert entry["rows"] == 4
        assert len(store.load_minute_bars(["AAA"], date(2024, 1, 1),
                                          date(2024, 1, 31))["AAA"]) == 4

    def test_dedupe_new_wins_on_ts_conflict(self, store):
        store.sync_symbol_month(bars(RTH_DAY, close=100.0), "AAA", "2024-01")
        store.sync_symbol_month(bars([RTH_DAY[1]], close=222.0), "AAA",
                                "2024-01")
        out = store.load_minute_bars(["AAA"], date(2024, 1, 1),
                                     date(2024, 1, 31))["AAA"]
        assert len(out) == 3                   # replaced, not added
        assert float(out[out["ts"] == ts_et(*RTH_DAY[1])]["close"].iloc[0]) == 222.0

    def test_manifest_counts_and_complete_latch(self, store):
        store.sync_symbol_month(bars(RTH_DAY), "AAA", "2024-01",
                                complete=True)
        m = store.read_manifest()["AAA"]["months"]["2024-01"]
        assert m["rows"] == 3
        assert m["min_ts"] == ts_et(*RTH_DAY[0])
        assert m["max_ts"] == ts_et(*RTH_DAY[-1])
        assert m["complete"] is True
        # a later partial top-up must not clear the complete latch
        store.sync_symbol_month(bars([(2024, 1, 17, 9, 30)]), "AAA",
                                "2024-01", complete=False)
        m = store.read_manifest()["AAA"]["months"]["2024-01"]
        assert m["rows"] == 4 and m["complete"] is True

    def test_empty_month_recorded_without_object(self, store):
        entry = store.sync_symbol_month(bars([]), "BBB", "2021-06",
                                        complete=True)
        assert entry == {"rows": 0, "min_ts": None, "max_ts": None,
                         "complete": True}
        assert "minute/BBB/2021-06.parquet" not in store._s3.objects

    def test_load_across_month_boundary_and_clip(self, store):
        store.sync_symbol_month(bars([(2024, 1, 31, 9, 30),
                                      (2024, 1, 31, 9, 31)]), "AAA", "2024-01")
        store.sync_symbol_month(bars([(2024, 2, 1, 9, 30),
                                      (2024, 2, 12, 9, 30)]), "AAA", "2024-02")
        out = store.load_minute_bars(["AAA", "MISSING"],
                                     date(2024, 1, 15), date(2024, 2, 5))
        assert set(out) == {"AAA"}
        assert len(out["AAA"]) == 3            # Feb 12 clipped away
        assert list(out["AAA"]["ts"]) == sorted(out["AAA"]["ts"])
        only_feb = store.load_minute_bars(["AAA"], date(2024, 2, 1),
                                          date(2024, 2, 28))["AAA"]
        assert len(only_feb) == 2

    def test_verify_ok_then_catches_corruption(self, store):
        store.sync_symbol_month(bars(RTH_DAY), "AAA", "2024-01")
        store.sync_symbol_month(bars([(2024, 2, 1, 9, 30)]), "AAA", "2024-02")
        assert store.verify(["AAA"]) == {"ok": True, "checked": 2,
                                         "mismatches": []}
        # corrupt one object: drop a row behind the manifest's back
        import io
        df = bars(RTH_DAY[:2])
        buf = io.BytesIO(); df.to_parquet(buf, index=False)
        store._s3.objects["minute/AAA/2024-01.parquet"] = buf.getvalue()
        result = store.verify(["AAA"])
        assert result["ok"] is False
        assert any("2024-01" in m for m in result["mismatches"])

    def test_verify_catches_stray_object(self, store):
        store.sync_symbol_month(bars(RTH_DAY), "AAA", "2024-01")
        store._s3.objects["minute/AAA/2024-03.parquet"] = b"orphan"
        result = store.verify(["AAA"])
        assert result["ok"] is False
        assert any("not in manifest" in m for m in result["mismatches"])

    def test_batched_manifest_writes(self, store):
        manifest = store.read_manifest()
        store.sync_symbol_month(bars(RTH_DAY), "AAA", "2024-01",
                                manifest=manifest)
        assert MANIFEST_KEY not in store._s3.objects   # caller owns the write
        store.write_manifest(manifest)
        assert store.read_manifest()["AAA"]["months"]["2024-01"]["rows"] == 3

    def test_missing_secrets_raise_clearly(self, monkeypatch):
        for k in ("R2_BUCKET", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="missing R2 secret"):
            MinuteStore()
