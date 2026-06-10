"""Tests for dashboard API endpoints."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from edgefinder.db.models import (
    MarketSnapshotRecord, StrategyAccount, StrategySnapshot, Ticker,
    TradeContext, TradeRecord,
)


@pytest.fixture
def client(db_engine, db_session):
    """Test client with overridden DB dependency."""
    from dashboard.app import app
    from dashboard.dependencies import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with patch("dashboard.services.init_services"), \
         patch("dashboard.services.shutdown_services"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestTradesAPI:
    def _seed_trades(self, db_session):
        for i, (pnl, status) in enumerate([
            (50.0, "CLOSED"), (-30.0, "CLOSED"), (None, "OPEN"),
        ]):
            db_session.add(TradeRecord(
                trade_id=f"t-{i}",
                strategy_name="alpha",
                symbol="AAPL",
                direction="LONG",
                trade_type="DAY",
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                target=110.0,
                confidence=70.0,
                entry_time=datetime.now(timezone.utc),
                status=status,
                pnl_dollars=pnl,
                r_multiple=1.0 if pnl and pnl > 0 else -0.6 if pnl else None,
            ))
        db_session.commit()

    def test_list_trades(self, client, db_session):
        self._seed_trades(db_session)
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_filter_by_status(self, client, db_session):
        self._seed_trades(db_session)
        resp = client.get("/api/trades?status=OPEN")
        assert len(resp.json()) == 1

    def test_filter_by_strategy(self, client, db_session):
        self._seed_trades(db_session)
        resp = client.get("/api/trades?strategy=alpha")
        assert len(resp.json()) == 3
        resp = client.get("/api/trades?strategy=nonexistent")
        assert len(resp.json()) == 0

    def test_stats(self, client, db_session):
        self._seed_trades(db_session)
        resp = client.get("/api/trades/stats?strategy=alpha")
        data = resp.json()
        assert data["total_trades"] == 2
        assert data["wins"] == 1
        assert data["losses"] == 1

    def test_wins(self, client, db_session):
        self._seed_trades(db_session)
        resp = client.get("/api/trades/wins")
        assert len(resp.json()) == 1

    def test_losses(self, client, db_session):
        self._seed_trades(db_session)
        resp = client.get("/api/trades/losses")
        assert len(resp.json()) == 1


class TestStrategiesAPI:
    def test_list_strategies_lists_promoted(self, client, db_session):
        from edgefinder.db.models import PromotedStrategy
        assert client.get("/api/strategies").json() == []
        db_session.add(PromotedStrategy(
            strategy_name="equal_weight", spec="equal_weight:SPY",
            symbols=["SPY"], schedule="monthly", tier="paper", active=True))
        db_session.commit()
        rows = client.get("/api/strategies").json()
        assert [r["name"] for r in rows] == ["equal_weight"]
        assert rows[0]["schedule"] == "monthly"

    def test_accounts_empty(self, client):
        resp = client.get("/api/strategies/accounts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_accounts_with_data(self, client, db_session):
        db_session.add(StrategyAccount(strategy_name="alpha", cash_balance=4500.0))
        db_session.commit()
        resp = client.get("/api/strategies/accounts")
        assert len(resp.json()) == 1
        assert resp.json()[0]["cash"] == 4500.0
        assert resp.json()[0]["lane"] == "v2"

    def test_positions_groups_open_trades(self, client, db_session):
        from datetime import datetime, timezone
        db_session.add(TradeRecord(
            trade_id="p-1", strategy_name="alpha", symbol="AAPL",
            direction="LONG", trade_type="SWING", entry_price=100.0, shares=10,
            stop_loss=0.0, target=0.0, confidence=1.0,
            entry_time=datetime.now(timezone.utc), status="OPEN"))
        db_session.add(TradeRecord(
            trade_id="p-2", strategy_name="alpha", symbol="MSFT",
            direction="LONG", trade_type="SWING", entry_price=200.0, shares=5,
            stop_loss=0.0, target=0.0, confidence=1.0,
            entry_time=datetime.now(timezone.utc), status="CLOSED"))
        db_session.commit()
        data = client.get("/api/strategies/positions").json()
        assert set(data) == {"alpha"}
        assert [p["symbol"] for p in data["alpha"]] == ["AAPL"]
        assert data["alpha"][0]["shares"] == 10
        assert data["alpha"][0]["entry_price"] == 100.0
        assert data["alpha"][0]["direction"] == "LONG"
        assert data["alpha"][0]["entry_time"]

    def test_equity_curve_empty(self, client):
        resp = client.get("/api/strategies/equity-curve")
        assert resp.status_code == 200

    def test_equity_curve_keeps_intraday_points_distinct(self, client, db_session):
        from datetime import datetime, timedelta, timezone
        # Two snapshots on the SAME day, 5 min apart — must stay distinct on
        # the time axis (the old date-only key would have collapsed them).
        base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=2)
        for i in range(2):
            db_session.add(StrategySnapshot(
                strategy_name="coward",
                timestamp=base + timedelta(minutes=5 * i),
                cash=10000.0, positions_value=0.0, total_equity=10000.0 + i,
                drawdown_pct=0.0, total_return_pct=0.0,
            ))
        db_session.commit()

        resp = client.get("/api/strategies/equity-curve")
        pts = resp.json()["coward"]
        times = [p["time"] for p in pts]
        assert all(isinstance(t, int) for t in times)        # epoch seconds
        assert times == sorted(times)                         # ascending
        assert len(set(times)) == len(times) == 2             # not collapsed by date


def _snap(ts, *, regime="bull", vix=15.0, spy=500.0, spy_chg=0.5, sectors=None):
    return MarketSnapshotRecord(
        timestamp=ts, market_regime=regime, vix_level=vix,
        spy_price=spy, spy_change_pct=spy_chg,
        qqq_price=400.0, qqq_change_pct=0.6,
        iwm_price=200.0, iwm_change_pct=-0.1,
        dia_price=380.0, dia_change_pct=0.3,
        sector_performance=sectors, advance_decline_ratio=1.8,
    )


class TestMarketRegimeAPI:
    def test_regime_returns_latest_and_ordered_history(self, client, db_session):
        from datetime import datetime, timedelta, timezone
        base = datetime.now(timezone.utc) - timedelta(hours=3)
        db_session.add(_snap(base, regime="sideways", vix=20.0))
        db_session.add(_snap(base + timedelta(hours=1), regime="bear", vix=28.0))
        db_session.add(_snap(base + timedelta(hours=2), regime="bull", vix=14.0,
                             sectors={"XLK": 1.2, "XLE": -0.4}))
        db_session.commit()

        data = client.get("/api/market/regime").json()
        # latest = newest snapshot
        assert data["latest"]["regime"] == "bull"
        assert data["latest"]["vix"] == 14.0
        assert data["latest"]["sector_prices"] == {"XLK": 1.2, "XLE": -0.4}
        assert data["latest"]["indices"]["SPY"]["change_pct"] == 0.5
        # history oldest -> newest
        vix_series = [h["vix"] for h in data["history"]]
        assert vix_series == [20.0, 28.0, 14.0]

    def test_regime_empty(self, client):
        data = client.get("/api/market/regime").json()
        assert data["latest"] is None
        assert data["history"] == []

    def test_regime_at_trade_uses_snapshot_at_or_before_entry(self, client, db_session):
        from datetime import datetime, timedelta, timezone
        entry = datetime.now(timezone.utc)
        # A snapshot just before entry (the regime the trade was taken in)...
        db_session.add(_snap(entry - timedelta(minutes=2), regime="bear", vix=30.0))
        # ...and one after entry that must NOT be chosen.
        db_session.add(_snap(entry + timedelta(minutes=10), regime="bull", vix=12.0))
        db_session.add(TradeRecord(
            trade_id="trade-x", strategy_name="coward", symbol="DBC",
            direction="LONG", trade_type="SWING", entry_price=24.0, shares=10,
            stop_loss=22.0, target=28.0, confidence=60.0,
            entry_time=entry, status="OPEN",
        ))
        db_session.commit()  # parent must exist before TradeContext FK
        db_session.add(TradeContext(
            trade_id="trade-x",
            sector_prices={"XLE": 0.9},
            short_interest={"pct_float": 3.2},
        ))
        db_session.commit()

        data = client.get("/api/market/regime/trade/trade-x").json()
        assert data["symbol"] == "DBC"
        assert data["regime"]["regime"] == "bear"      # at-or-before entry, not the later bull
        assert data["regime"]["vix"] == 30.0
        assert data["context"]["short_interest"] == {"pct_float": 3.2}

    def test_regime_at_trade_404(self, client):
        assert client.get("/api/market/regime/trade/nope").status_code == 404


class TestResearchBars:
    def test_ticker_bars_returns_daily_ohlc_in_range(self, client, db_session):
        from datetime import date, timedelta
        from edgefinder.db.models import DailyBar
        today = date.today()
        # in-range + out-of-range + different symbol
        db_session.add(DailyBar(symbol="AAPL", date=today - timedelta(days=2),
                                open=10, high=11, low=9, close=10.5, volume=1000))
        db_session.add(DailyBar(symbol="AAPL", date=today - timedelta(days=1),
                                open=10.5, high=12, low=10, close=11.5, volume=1200))
        db_session.add(DailyBar(symbol="AAPL", date=today - timedelta(days=400),
                                open=5, high=6, low=4, close=5.5, volume=900))
        db_session.add(DailyBar(symbol="MSFT", date=today - timedelta(days=1),
                                open=200, high=205, low=199, close=204, volume=500))
        db_session.commit()

        bars = client.get("/api/research/ticker/AAPL/bars?days=365").json()
        assert [b["time"] for b in bars] == [
            (today - timedelta(days=2)).isoformat(),
            (today - timedelta(days=1)).isoformat(),
        ]  # in-range only, ascending, AAPL only
        assert bars[-1]["close"] == 11.5
        assert {"open", "high", "low", "close", "volume"} <= bars[0].keys()

    def test_ticker_bars_empty_before_backfill(self, client):
        assert client.get("/api/research/ticker/NOPE/bars").json() == []


class TestResearchAPI:
    def test_search_empty(self, client):
        resp = client.get("/api/research/search?q=AAPL")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_with_data(self, client, db_session):
        db_session.add(Ticker(symbol="AAPL", company_name="Apple Inc."))
        db_session.commit()
        resp = client.get("/api/research/search?q=AAPL")
        assert len(resp.json()) == 1

    def test_active_tickers(self, client, db_session):
        db_session.add(Ticker(symbol="AAPL", is_active=True))
        db_session.add(Ticker(symbol="MSFT", is_active=False))
        db_session.commit()
        resp = client.get("/api/research/active")
        assert len(resp.json()) == 1


class TestBenchmarksAPI:
    def test_comparison_empty(self, client):
        resp = client.get("/api/benchmarks/comparison")
        assert resp.status_code == 200
