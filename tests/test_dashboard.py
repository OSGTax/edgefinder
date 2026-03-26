"""
EdgeFinder Phase 5 Tests: Dashboard API
=========================================
Tests cover: all API endpoints, response format, filtering,
and HTML serving.

Run: python -m pytest tests/test_dashboard.py -v
"""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from modules.database import (
    init_db, reset_engine, get_session,
    WatchlistStock, Signal as SignalRecord, ArenaTradeLog, ArenaSnapshot,
)


@pytest.fixture(autouse=True)
def setup_db():
    """Fresh in-memory DB for each test."""
    reset_engine()
    init_db(":memory:")
    yield
    reset_engine()


@pytest.fixture
def client(setup_db):
    """FastAPI test client."""
    from unittest.mock import patch
    # Prevent scheduler from starting in tests
    with patch("modules.scheduler.start_scheduler"), \
         patch("modules.scheduler.stop_scheduler"):
        from dashboard.app import app
        return TestClient(app)


def _seed_watchlist():
    session = get_session()
    for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL"]):
        session.add(WatchlistStock(
            ticker=ticker, company_name=f"{ticker} Inc", sector="Technology",
            industry="Software", market_cap=1e12, price=100 + i * 50,
            composite_score=80 - i * 5, lynch_score=75 - i * 3,
            burry_score=85 - i * 7, lynch_category="stalwart",
            scan_date=datetime(2025, 1, 15, tzinfo=timezone.utc), is_active=True,
        ))
    session.commit()
    session.close()


def _seed_trades():
    session = get_session()
    session.add(ArenaTradeLog(
        trade_id="t-001", strategy_name="lynch", strategy_version="1.0",
        ticker="AAPL", action="BUY", direction="LONG", trade_type="DAY",
        signal_price=100.0, execution_price=100.0, exit_price=105.0,
        shares=10, stop_loss=98.0, target=103.0, confidence=80.0,
        pnl_dollars=50.0, pnl_percent=0.05, r_multiple=2.5,
        exit_reason="TARGET_HIT", price_source="yfinance",
        status="CLOSED", created_at=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
    ))
    session.add(ArenaTradeLog(
        trade_id="t-002", strategy_name="burry", strategy_version="1.0",
        ticker="MSFT", action="BUY", direction="LONG", trade_type="SWING",
        signal_price=200.0, execution_price=200.0, exit_price=195.0,
        shares=5, stop_loss=196.0, target=206.0, confidence=65.0,
        pnl_dollars=-25.0, pnl_percent=-0.025, r_multiple=-1.25,
        exit_reason="STOP_HIT", price_source="yfinance",
        status="CLOSED", created_at=datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc),
    ))
    session.commit()
    session.close()


def _seed_signals():
    session = get_session()
    session.add(SignalRecord(
        ticker="AAPL", signal_type="BUY", trade_type="DAY",
        confidence=75.0, indicators={"rsi_oversold": {"rsi": 25}},
        was_traded=True, trade_id="t-001",
        timestamp=datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc),
    ))
    session.add(SignalRecord(
        ticker="GOOGL", signal_type="BUY", trade_type="DAY",
        confidence=35.0, indicators={}, was_traded=False,
        reason_skipped="Below confidence threshold",
        timestamp=datetime(2025, 1, 15, 9, 35, tzinfo=timezone.utc),
    ))
    session.commit()
    session.close()


def _seed_snapshots():
    session = get_session()
    for i in range(5):
        session.add(ArenaSnapshot(
            strategy_name="lynch",
            timestamp=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
            cash=2500.0 + i * 20, positions_value=0.0,
            total_equity=2500.0 + i * 20, peak_equity=2500.0 + i * 20,
            drawdown_pct=0.0, open_positions=0,
            total_return_pct=round(i * 20 / 2500.0 * 100, 2),
        ))
    session.commit()
    session.close()


# ════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "scheduler_running" in data
        assert "arena_engine_initialized" in data


# ════════════════════════════════════════════════════════════
# DASHBOARD HOME
# ════════════════════════════════════════════════════════════

class TestHome:
    def test_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "EdgeFinder" in resp.text


# ════════════════════════════════════════════════════════════
# WATCHLIST API
# ════════════════════════════════════════════════════════════

class TestWatchlistAPI:
    def test_empty_watchlist(self, client):
        resp = client.get("/api/watchlist")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["watchlist"] == []

    def test_watchlist_with_data(self, client):
        _seed_watchlist()
        resp = client.get("/api/watchlist")
        data = resp.json()
        assert data["count"] == 3
        assert data["watchlist"][0]["ticker"] == "AAPL"  # Highest composite
        assert data["watchlist"][0]["composite_score"] == 80

    def test_watchlist_limit(self, client):
        _seed_watchlist()
        resp = client.get("/api/watchlist?limit=2")
        data = resp.json()
        assert data["count"] == 2

    def test_watchlist_fields(self, client):
        _seed_watchlist()
        resp = client.get("/api/watchlist")
        stock = resp.json()["watchlist"][0]
        required_fields = [
            "ticker", "company_name", "sector", "price",
            "composite_score", "lynch_score", "burry_score", "lynch_category",
        ]
        for f in required_fields:
            assert f in stock, f"Missing field: {f}"


# ════════════════════════════════════════════════════════════
# SIGNALS API
# ════════════════════════════════════════════════════════════

class TestSignalsAPI:
    def test_empty_signals(self, client):
        resp = client.get("/api/signals")
        assert resp.json()["count"] == 0

    def test_signals_with_data(self, client):
        _seed_signals()
        resp = client.get("/api/signals")
        data = resp.json()
        assert data["count"] == 2

    def test_signals_filter_ticker(self, client):
        _seed_signals()
        resp = client.get("/api/signals?ticker=AAPL")
        data = resp.json()
        assert data["count"] == 1
        assert data["signals"][0]["ticker"] == "AAPL"

    def test_signals_fields(self, client):
        _seed_signals()
        sig = client.get("/api/signals").json()["signals"][0]
        for f in ["ticker", "signal_type", "trade_type", "confidence", "was_traded"]:
            assert f in sig


# ════════════════════════════════════════════════════════════
# TRADES API
# ════════════════════════════════════════════════════════════

class TestTradesAPI:
    def test_empty_trades(self, client):
        resp = client.get("/api/trades")
        assert resp.json()["count"] == 0

    def test_trades_with_data(self, client):
        _seed_trades()
        resp = client.get("/api/trades")
        data = resp.json()
        assert data["count"] == 2

    def test_trades_filter_ticker(self, client):
        _seed_trades()
        resp = client.get("/api/trades?ticker=AAPL")
        data = resp.json()
        assert data["count"] == 1

    def test_trades_filter_strategy(self, client):
        _seed_trades()
        resp = client.get("/api/trades?strategy=burry")
        data = resp.json()
        assert data["count"] == 1
        assert data["trades"][0]["ticker"] == "MSFT"

    def test_trades_pnl_fields(self, client):
        _seed_trades()
        trade = client.get("/api/trades").json()["trades"][0]
        assert "pnl_dollars" in trade
        assert "pnl_percent" in trade
        assert "r_multiple" in trade


# ════════════════════════════════════════════════════════════
# TRADE STATS API
# ════════════════════════════════════════════════════════════

class TestTradeStatsAPI:
    def test_empty_stats(self, client):
        resp = client.get("/api/trades/stats")
        data = resp.json()
        assert data["total_trades"] == 0

    def test_stats_with_data(self, client):
        _seed_trades()
        resp = client.get("/api/trades/stats")
        data = resp.json()
        assert data["total_trades"] == 2
        assert data["winning_trades"] == 1
        assert data["losing_trades"] == 1
        assert data["total_pnl"] == 25.0
        assert data["win_rate"] == 50.0

    def test_stats_fields(self, client):
        _seed_trades()
        data = client.get("/api/trades/stats").json()
        required = [
            "total_trades", "winning_trades", "losing_trades", "win_rate",
            "total_pnl", "avg_pnl", "profit_factor", "avg_r_multiple",
        ]
        for f in required:
            assert f in data, f"Missing field: {f}"


# ════════════════════════════════════════════════════════════
# EQUITY CURVE API
# ════════════════════════════════════════════════════════════

class TestEquityCurveAPI:
    def test_empty_curve(self, client):
        resp = client.get("/api/equity-curve")
        assert resp.json()["count"] == 0

    def test_curve_with_data(self, client):
        _seed_snapshots()
        resp = client.get("/api/equity-curve")
        data = resp.json()
        assert data["count"] == 5
        assert "lynch" in data["strategies"]
        lynch_snaps = data["strategies"]["lynch"]
        assert lynch_snaps[0]["total_value"] <= lynch_snaps[-1]["total_value"]

    def test_curve_limit(self, client):
        _seed_snapshots()
        resp = client.get("/api/equity-curve?limit=3")
        data = resp.json()
        assert data["count"] == 3


# ════════════════════════════════════════════════════════════
# ACCOUNT API
# ════════════════════════════════════════════════════════════

class TestAccountAPI:
    def test_default_account(self, client):
        resp = client.get("/api/account")
        data = resp.json()
        # No arena engine running in tests — falls back to zero values
        assert data["total_value"] == 0.0
        assert data["open_positions"] == 0

    def test_account_fields(self, client):
        resp = client.get("/api/account")
        data = resp.json()
        for f in ["cash", "positions_value", "total_value", "open_positions",
                   "realized_pnl", "unrealized_pnl"]:
            assert f in data, f"Missing field: {f}"


# ════════════════════════════════════════════════════════════
# SKIPPED SIGNALS API
# ════════════════════════════════════════════════════════════

class TestSkippedSignalsAPI:
    def test_empty_skipped(self, client):
        resp = client.get("/api/skipped-signals")
        assert resp.json()["count"] == 0

    def test_skipped_with_data(self, client):
        _seed_signals()
        resp = client.get("/api/skipped-signals")
        data = resp.json()
        assert data["count"] == 1
        assert data["signals"][0]["ticker"] == "GOOGL"
        assert data["signals"][0]["reason_skipped"] == "Below confidence threshold"


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_dashboard.py -v
#
# Expected results:
#   TestHealth:             1 test
#   TestHome:               1 test
#   TestWatchlistAPI:        4 tests
#   TestSignalsAPI:          4 tests
#   TestTradesAPI:           5 tests
#   TestTradeStatsAPI:       3 tests
#   TestEquityCurveAPI:      3 tests
#   TestAccountAPI:          2 tests
#   TestSkippedSignalsAPI:   2 tests
#
# TOTAL: 25 tests (seeding ArenaTradeLog + ArenaSnapshot)
# ════════════════════════════════════════════════════════════
