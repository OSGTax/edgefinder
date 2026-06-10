"""Symbol chart API — source selection, split adjustment, epoch times,
indicators, events, TTL-cache behavior, and the R2-unavailable fallback."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dashboard.symbol_service as ss
from edgefinder.db.models import DailyBar, TickerDividend, TickerNews, TickerSplit


@pytest.fixture
def client(db_engine, db_session):
    from dashboard.app import app
    from dashboard.dependencies import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    ss.clear_cache()
    with patch("dashboard.services.init_services"), \
         patch("dashboard.services.shutdown_services"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
    ss.clear_cache()


def _seed_db_bars(session, symbol="AAA", days=120, price=100.0):
    d = date.today() - timedelta(days=days)
    while d <= date.today():
        if d.weekday() < 5:
            session.add(DailyBar(symbol=symbol, date=d, open=price, high=price * 1.01,
                                 low=price * 0.99, close=price, volume=1e6,
                                 source="test"))
        d += timedelta(days=1)
    session.commit()


def _store_frame(days=900, price=50.0, start_offset=900):
    rows = []
    d = date.today() - timedelta(days=start_offset)
    n = 0
    while n < days:
        if d.weekday() < 5:
            rows.append({"date": d, "open": price, "high": price * 1.01,
                         "low": price * 0.99, "close": price, "volume": 2e6})
            n += 1
        d += timedelta(days=1)
    return pd.DataFrame(rows)


class TestBars:
    def test_recent_range_serves_from_db_with_epoch_times(self, client, db_session):
        _seed_db_bars(db_session)
        r = client.get("/api/symbols/AAA/bars?range=3m")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "db"
        assert body["truncated"] is False
        assert body["basis"] == "split-adjusted"
        bars = body["bars"]
        assert len(bars) > 40
        # times are UTC-midnight epoch seconds, ascending
        times = [b["time"] for b in bars]
        assert times == sorted(times)
        assert all(t % 86400 == 0 for t in times)

    def test_unknown_symbol_404s(self, client):
        assert client.get("/api/symbols/NOPE/bars?range=3m").status_code == 404

    def test_max_range_reads_store_and_split_adjusts(self, client, db_session):
        # split 4:1 executed 100 days ago — store frame is RAW, so bars
        # before the split must come back divided by 4
        exec_day = date.today() - timedelta(days=100)
        db_session.add(TickerSplit(symbol="BBB", execution_date=str(exec_day),
                                   split_from=1, split_to=4))
        db_session.commit()
        frame = _store_frame()
        with patch.object(ss, "load_bars_from_store",
                          return_value={"BBB": frame}) as loader:
            r = client.get("/api/symbols/BBB/bars?range=max")
            assert r.status_code == 200
            body = r.json()
            assert body["source"] == "r2"
            loader.assert_called_once()
        bars = body["bars"]
        pre = [b for b in bars if b["time"] < int(datetime(
            exec_day.year, exec_day.month, exec_day.day,
            tzinfo=timezone.utc).timestamp())]
        post = [b for b in bars if b not in pre]
        assert pre and post
        assert pre[0]["close"] == pytest.approx(12.5, rel=1e-6)   # 50 / 4
        assert post[-1]["close"] == pytest.approx(50.0, rel=1e-6)

    def test_store_result_is_ttl_cached(self, client):
        frame = _store_frame()
        with patch.object(ss, "load_bars_from_store",
                          return_value={"CCC": frame}) as loader:
            assert client.get("/api/symbols/CCC/bars?range=max").status_code == 200
            assert client.get("/api/symbols/CCC/bars?range=max").status_code == 200
            assert loader.call_count == 1   # second hit served from cache

    def test_r2_failure_falls_back_to_db_truncated(self, client, db_session):
        _seed_db_bars(db_session, symbol="DDD")
        with patch.object(ss, "load_bars_from_store",
                          side_effect=RuntimeError("missing R2 secret")):
            r = client.get("/api/symbols/DDD/bars?range=max")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "db"
        assert body["truncated"] is True
        assert len(body["bars"]) > 40

    def test_protected_symbol_never_touches_store(self, client, db_session):
        _seed_db_bars(db_session, symbol="SPY", days=200)
        with patch.object(ss, "load_bars_from_store",
                          side_effect=AssertionError("must not be called")):
            r = client.get("/api/symbols/SPY/bars?range=max")
        assert r.status_code == 200
        assert r.json()["source"] == "db"

    def test_indicators_series(self, client, db_session):
        _seed_db_bars(db_session, symbol="EEE", days=320)
        r = client.get("/api/symbols/EEE/bars?range=3m&indicators=true")
        assert r.status_code == 200
        body = r.json()
        ind = body["indicators"]
        assert "ema_21" in ind and "rsi" in ind
        first_bar_time = body["bars"][0]["time"]
        # indicator points are clipped to the requested range, not warmup
        assert ind["ema_21"][0]["time"] >= first_bar_time
        # and every value is finite JSON (no NaN leakage)
        assert all(isinstance(p["value"], (int, float)) for p in ind["rsi"])


class TestEvents:
    def test_events_shapes_and_epochs(self, client, db_session):
        d = date.today() - timedelta(days=30)
        db_session.add(TickerDividend(symbol="AAA", ex_dividend_date=str(d),
                                      pay_date=str(d + timedelta(days=14)),
                                      cash_amount=0.25))
        db_session.add(TickerSplit(symbol="AAA", execution_date=str(d),
                                   split_from=1, split_to=2))
        db_session.add(TickerNews(symbol="AAA", title="headline",
                                  published_utc=datetime.now(timezone.utc),
                                  article_url="https://x", publisher_name="pub"))
        db_session.commit()
        r = client.get("/api/symbols/AAA/events")
        assert r.status_code == 200
        body = r.json()
        assert body["dividends"][0]["cash_amount"] == 0.25
        assert body["splits"][0]["ratio"] == "2:1"
        assert body["news"][0]["title"] == "headline"
        for group in ("dividends", "splits", "news"):
            for e in body[group]:
                assert e["time"] % 86400 == 0
