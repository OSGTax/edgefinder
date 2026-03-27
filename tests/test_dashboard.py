"""Tests for dashboard API endpoints."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from edgefinder.core.models import Direction, Trade, TradeStatus, TradeType
from edgefinder.db.models import (
    IndexDaily, ManualInjection, SentimentReading, StrategyAccount,
    StrategySnapshot, Ticker, TradeRecord,
)


@pytest.fixture
def client(db_engine, db_session):
    """Test client with overridden DB dependency."""
    from dashboard.app import app
    from dashboard.dependencies import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
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
    def test_list_strategies(self, client):
        resp = client.get("/api/strategies")
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()}
        assert "alpha" in names

    def test_accounts_empty(self, client):
        resp = client.get("/api/strategies/accounts")
        assert resp.status_code == 200
        # Arena may return live accounts if Polygon key is set,
        # otherwise falls back to empty DB
        data = resp.json()
        assert isinstance(data, list)

    def test_accounts_with_data(self, client, db_session):
        from dashboard.services import get_arena
        arena = get_arena()
        if arena:
            # Live arena returns all strategy accounts
            resp = client.get("/api/strategies/accounts")
            data = resp.json()
            assert len(data) >= 1
            names = {a["strategy_name"] for a in data}
            assert "alpha" in names
        else:
            # No arena — DB fallback
            db_session.add(StrategyAccount(strategy_name="alpha", cash_balance=4500.0))
            db_session.commit()
            resp = client.get("/api/strategies/accounts")
            assert len(resp.json()) == 1
            assert resp.json()[0]["cash"] == 4500.0

    def test_equity_curve_empty(self, client):
        resp = client.get("/api/strategies/equity-curve")
        assert resp.status_code == 200


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


class TestSentimentAPI:
    def test_history_empty(self, client):
        resp = client.get("/api/sentiment/history/AAPL")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_with_data(self, client, db_session):
        db_session.add(SentimentReading(
            symbol="AAPL", source="reddit", score=0.5, mention_count=100,
        ))
        db_session.commit()
        resp = client.get("/api/sentiment/history/AAPL")
        assert len(resp.json()) == 1


class TestBenchmarksAPI:
    def test_comparison_empty(self, client):
        resp = client.get("/api/benchmarks/comparison")
        assert resp.status_code == 200


class TestInjectAPI:
    def test_inject_ticker(self, client):
        resp = client.post("/api/inject", json={
            "symbol": "GME",
            "notes": "Squeeze potential",
        })
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "GME"

    def test_list_injections(self, client, db_session):
        db_session.add(ManualInjection(symbol="GME", notes="test"))
        db_session.commit()
        resp = client.get("/api/inject")
        assert len(resp.json()) == 1

    def test_remove_injection(self, client, db_session):
        inj = ManualInjection(symbol="AMC", notes="test")
        db_session.add(inj)
        db_session.commit()
        resp = client.delete(f"/api/inject/{inj.id}")
        assert resp.status_code == 200
        assert "removed" in resp.json()["message"]
