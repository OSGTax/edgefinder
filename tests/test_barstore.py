"""Tests for the R2 Parquet bar store (against an in-memory fake S3)."""

from datetime import date, timedelta

import pytest

from edgefinder.data import barstore
from edgefinder.data.barstore import MANIFEST_KEY, BarStore
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import DailyBar


class FakeS3:
    """The four S3 calls BarStore makes, over a plain dict."""

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
def store(monkeypatch):
    s = BarStore.__new__(BarStore)        # skip env/boto3 __init__
    s.bucket = "test"
    s._s3 = FakeS3()
    return s


@pytest.fixture()
def session():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = get_session_factory(engine)()
    d0 = date(2024, 1, 1)
    for sym, px in (("AAA", 100.0), ("BBB", 50.0)):
        for i in range(30):
            sess.add(DailyBar(symbol=sym, date=d0 + timedelta(days=i),
                              open=px + i, high=px + i, low=px + i,
                              close=px + i, volume=1e6, source="test"))
    sess.commit()
    return sess


class TestBarStore:
    def test_sync_verify_load_roundtrip(self, store, session):
        result = store.sync(session)
        assert result == {"changed": 2, "uploaded": 2, "failed": 0,
                          "migrated": 0, "bytes": result["bytes"],
                          "symbols_total": 2}
        assert store.verify(session, sample=2) == {
            "ok": True, "checked": 2, "mismatches": []}
        bars = store.load(["AAA", "BBB", "MISSING"])
        assert set(bars) == {"AAA", "BBB"}
        assert len(bars["AAA"]) == 30
        assert bars["AAA"]["date"].iloc[0] == date(2024, 1, 1)   # real date objs
        assert bars["AAA"]["close"].iloc[-1] == 129.0

    def test_sync_is_incremental_and_idempotent(self, store, session):
        store.sync(session)
        again = store.sync(session)
        assert again["changed"] == 0 and again["uploaded"] == 0
        # a new bar for one symbol -> only that symbol re-exports
        session.add(DailyBar(symbol="AAA", date=date(2024, 2, 15), open=1,
                             high=1, low=1, close=1, volume=1, source="test"))
        session.commit()
        third = store.sync(session)
        assert third["changed"] == 1 and third["uploaded"] == 1
        assert len(store.load(["AAA"])["AAA"]) == 31

    def test_verify_catches_corruption(self, store, session):
        store.sync(session)
        # corrupt AAA's object in the store
        import io

        import pandas as pd
        df = pd.DataFrame({"date": [date(2024, 1, 1)], "open": [1.0],
                           "high": [1.0], "low": [1.0], "close": [999.0],
                           "volume": [1.0]})
        buf = io.BytesIO(); df.to_parquet(buf, index=False)
        store._s3.objects["bars/AAA.parquet"] = buf.getvalue()
        result = store.verify(session, sample=2)
        assert result["ok"] is False
        assert "AAA" in result["mismatches"]

    def test_manifest_checkpointing(self, store, session):
        store.sync(session)
        assert MANIFEST_KEY in store._s3.objects
        manifest = store.read_manifest()
        assert manifest["AAA"]["rows"] == 30
        assert manifest["AAA"]["max_date"] == "2024-01-30"

    def test_sync_writes_content_fingerprint(self, store, session):
        """Every upload records db_fp = [close, volume] at db_max — the shared
        manifest field agent.refresh's merge-sync uses to catch same-date
        value corrections (this command's own diff stays (rows, max))."""
        store.sync(session)
        manifest = store.read_manifest()
        assert manifest["AAA"]["db_fp"] == [129.0, 1e6]   # last bar's close/vol
        assert manifest["BBB"]["db_fp"] == [79.0, 1e6]
        # a forced re-merge after a value correction refreshes the fingerprint
        row = (session.query(DailyBar)
               .filter(DailyBar.symbol == "AAA",
                       DailyBar.date == date(2024, 1, 30)).one())
        row.close = 777.0
        session.commit()
        store.sync(session, symbols=["AAA"])
        assert store.read_manifest()["AAA"]["db_fp"] == [777.0, 1e6]

    def test_missing_secrets_raise_clearly(self, monkeypatch):
        for k in ("R2_BUCKET", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="missing R2 secret"):
            BarStore()


class TestGrowOnlyStore:
    """The store is the permanent asset: a DB that sheds rows must never
    shrink it, and retention pruning is gated on the merge fingerprint."""

    def test_merge_preserves_rows_the_db_no_longer_has(self, store, session):
        store.sync(session)
        # the DB sheds its oldest 10 AAA rows (retention) and gains one new
        (session.query(DailyBar)
         .filter(DailyBar.symbol == "AAA",
                 DailyBar.date < date(2024, 1, 11))
         .delete(synchronize_session=False))
        session.add(DailyBar(symbol="AAA", date=date(2024, 2, 15), open=1,
                             high=1, low=1, close=1, volume=1, source="test"))
        session.commit()
        out = store.sync(session)
        assert out["uploaded"] == 1
        merged = store.load(["AAA"])["AAA"]
        assert len(merged) == 31                       # 30 originals + 1 new
        assert merged["date"].iloc[0] == date(2024, 1, 1)   # shed rows kept
        m = store.read_manifest()["AAA"]
        assert m["rows"] == 31 and m["db_rows"] == 21
        # subset verify passes even though DB < R2
        assert store.verify(session, sample=2)["ok"] is True

    def test_db_correction_wins_on_conflicting_date(self, store, session):
        store.sync(session)
        row = (session.query(DailyBar)
               .filter(DailyBar.symbol == "AAA",
                       DailyBar.date == date(2024, 1, 15)).one())
        row.close = 777.0
        session.commit()
        # a pure value edit doesn't move the (rows, max_date) fingerprint —
        # the documented escape hatch is a forced per-symbol sync
        store.sync(session, symbols=["AAA"])
        merged = store.load(["AAA"])["AAA"]
        assert float(merged[merged["date"] == date(2024, 1, 15)]["close"].iloc[0]) == 777.0
        assert len(merged) == 30                       # replaced, not added

    def test_legacy_manifest_entries_migrate_without_upload(self, store, session):
        store.sync(session)
        # strip fingerprints to simulate the pre-merge-era manifest
        manifest = store.read_manifest()
        legacy = {sym: {"rows": m["rows"], "max_date": m["max_date"]}
                  for sym, m in manifest.items()}
        store.write_manifest(legacy)
        out = store.sync(session)
        assert out["migrated"] == 2 and out["uploaded"] == 0
        assert store.read_manifest()["AAA"]["db_rows"] == 30

