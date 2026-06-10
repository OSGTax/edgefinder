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
        assert result == {"changed": 2, "uploaded": 2,
                          "bytes": result["bytes"], "symbols_total": 2}
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

    def test_missing_secrets_raise_clearly(self, monkeypatch):
        for k in ("R2_BUCKET", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="missing R2 secret"):
            BarStore()
