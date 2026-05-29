"""Tests for dashboard API endpoints."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from edgefinder.core.models import Direction, Trade, TradeStatus, TradeType
from edgefinder.db.models import (
    IndexDaily, ManualInjection, MarketSnapshotRecord, StrategyAccount,
    StrategySnapshot, Ticker, TradeContext, TradeRecord,
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
    def test_list_strategies(self, client):
        resp = client.get("/api/strategies")
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()}
        assert "coward" in names

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
            assert "coward" in names
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


def _fake_arena(accounts, positions):
    from unittest.mock import MagicMock
    a = MagicMock()
    a.get_all_accounts.return_value = accounts
    a.get_all_open_positions.return_value = positions
    return a


class TestLiveAccountMarking:
    """Per-strategy value must be cash + securities at current market price.

    Regression guards for two bugs: (1) ``/accounts`` double-counted P&L by
    adding unrealized to an already-mark-to-market open_positions_value, and
    (2) the equity curve ended at the last daily-close snapshot instead of the
    live current value.
    """

    def _patch(self, monkeypatch, accounts, positions, price):
        import dashboard.routers.strategies as strat
        import dashboard.services as services
        from unittest.mock import MagicMock
        monkeypatch.setattr(strat, "get_arena", lambda: _fake_arena(accounts, positions))
        provider = MagicMock()
        provider.get_latest_price.return_value = price
        monkeypatch.setattr(services, "_provider", provider)
        return strat

    def test_marks_to_market_without_double_counting(self, monkeypatch):
        # to_dict()'s open_positions_value is already mark-to-market (5100),
        # but the live value must come from the current price (52*100=5200),
        # and total_equity must be cash + market value with NO P&L re-added.
        accounts = {"coward": {
            "strategy_name": "coward", "starting_capital": 10000.0,
            "cash": 5000.0, "open_positions_value": 5100.0, "total_equity": 10100.0,
        }}
        positions = {"coward": [{
            "symbol": "AAA", "shares": 100.0, "entry_price": 50.0,
            "direction": "LONG", "trade_type": "SWING", "trade_id": "t",
        }]}
        strat = self._patch(monkeypatch, accounts, positions, price=52.0)

        [acct] = strat._live_account_states()
        assert acct["open_positions_value"] == 5200.0          # 100 * live 52
        assert acct["total_equity"] == 10200.0                 # cash + market value
        assert acct["unrealized_pnl"] == 200.0                 # (52-50)*100
        # The double-count bug would have produced 5100+200=5300 / 10300.
        assert acct["total_equity"] == round(acct["cash"] + acct["open_positions_value"], 2)

    def test_falls_back_to_entry_price_when_no_live_price(self, monkeypatch):
        accounts = {"coward": {
            "strategy_name": "coward", "starting_capital": 10000.0,
            "cash": 5000.0, "open_positions_value": 5100.0, "total_equity": 10100.0,
        }}
        positions = {"coward": [{
            "symbol": "AAA", "shares": 100.0, "entry_price": 50.0,
            "direction": "LONG", "trade_type": "SWING", "trade_id": "t",
        }]}
        strat = self._patch(monkeypatch, accounts, positions, price=None)

        [acct] = strat._live_account_states()
        assert acct["open_positions_value"] == 5000.0          # 100 * entry 50
        assert acct["total_equity"] == 10000.0
        assert acct["unrealized_pnl"] == 0.0

    def test_equity_curve_ends_at_live_market_value(self, client, db_session, monkeypatch):
        from datetime import datetime, timedelta, timezone
        # A stale prior-day close snapshot...
        db_session.add(StrategySnapshot(
            strategy_name="coward",
            timestamp=datetime.now(timezone.utc) - timedelta(days=1),
            cash=10000.0, positions_value=0.0, total_equity=10000.0,
            drawdown_pct=0.0, total_return_pct=0.0,
        ))
        db_session.commit()

        accounts = {"coward": {
            "strategy_name": "coward", "starting_capital": 10000.0,
            "cash": 5000.0, "open_positions_value": 5100.0, "total_equity": 10100.0,
        }}
        positions = {"coward": [{
            "symbol": "AAA", "shares": 100.0, "entry_price": 50.0,
            "direction": "LONG", "trade_type": "SWING", "trade_id": "t",
        }]}
        self._patch(monkeypatch, accounts, positions, price=52.0)

        resp = client.get("/api/strategies/equity-curve")
        assert resp.status_code == 200
        series = resp.json()["coward"]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Last point is the live "now" value (cash + securities at 52), not the
        # stale 10000 close.
        assert series[-1]["date"] == today
        assert series[-1]["total_equity"] == 10200.0

    def test_equity_curve_keeps_intraday_points_distinct(self, client, db_session, monkeypatch):
        from datetime import datetime, timedelta, timezone
        import dashboard.routers.strategies as strat
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
        monkeypatch.setattr(strat, "get_arena", lambda: None)  # no live tail

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


def test_write_equity_snapshots_persists_timeseries(db_session, monkeypatch):
    """The intraday monitor and daily job share this writer — one row/strategy."""
    import dashboard.services as services
    from unittest.mock import MagicMock
    arena = MagicMock()
    arena.get_all_accounts.return_value = {
        "coward": {"cash": 5000.0, "open_positions_value": 5200.0,
                   "total_equity": 10200.0, "drawdown_pct": 0.0},
    }
    monkeypatch.setattr(services, "_arena", arena)

    n = services._write_equity_snapshots(db_session)
    db_session.commit()
    assert n == 1
    rows = db_session.query(StrategySnapshot).filter_by(strategy_name="coward").all()
    assert len(rows) == 1
    assert rows[0].total_equity == 10200.0
    assert rows[0].positions_value == 5200.0


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
