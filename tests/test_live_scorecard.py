"""Tests for the live-vs-SPY scorecard (edgefinder/analytics/live_scorecard.py)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from edgefinder.analytics.live_scorecard import (
    compute_all_scorecards,
    compute_scorecard,
)
from edgefinder.db.models import IndexDaily, StrategySnapshot, TradeRecord

NOW = datetime(2026, 5, 20, 18, 0, tzinfo=timezone.utc)  # Wed mid-session

# Ten consecutive weekdays ending before NOW (Mon 5/4 .. Fri 5/15).
WEEKDAYS = [
    date(2026, 5, 4), date(2026, 5, 5), date(2026, 5, 6), date(2026, 5, 7),
    date(2026, 5, 8), date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13),
    date(2026, 5, 14), date(2026, 5, 15),
]


def _snap(session, strategy: str, d: date, equity: float, hour_utc: int = 20):
    # Naive UTC timestamps, as the DB round-trips them.
    session.add(StrategySnapshot(
        strategy_name=strategy,
        timestamp=datetime(d.year, d.month, d.day, hour_utc, 5),
        cash=equity, positions_value=0.0, total_equity=equity,
        drawdown_pct=0.0, total_return_pct=0.0,
    ))


def _spy(session, d: date, close: float):
    session.add(IndexDaily(
        symbol="SPY", date=datetime(d.year, d.month, d.day), close=close,
        change_pct=0.0,
    ))


def _closed_trade(session, strategy: str, exit_day: date, pnl: float, i: int):
    session.add(TradeRecord(
        trade_id=f"{strategy}-{exit_day}-{i}", strategy_name=strategy,
        symbol=f"SYM{i}", direction="LONG", trade_type="SWING",
        entry_price=100.0, shares=1, stop_loss=95.0, target=110.0,
        confidence=0.7,
        entry_time=datetime(exit_day.year, exit_day.month, exit_day.day, 14, 0),
        exit_time=datetime(exit_day.year, exit_day.month, exit_day.day, 19, 0),
        status="CLOSED", pnl_dollars=pnl,
    ))


def _seed_outperformer(session, strategy="coward"):
    """Strategy +10% over ten days vs SPY +2%."""
    for i, d in enumerate(WEEKDAYS):
        _snap(session, strategy, d, 10_000.0 * (1 + 0.01 * (i + 1)))
        _spy(session, d, 100.0 * (1 + 0.002 * (i + 1)))
    session.commit()


class TestComputeScorecard:
    def test_outperformer_passes_with_low_trade_bar(self, db_session):
        _seed_outperformer(db_session)
        sc = compute_scorecard(db_session, "coward", days=30,
                               pass_min_trades=0, now=NOW)
        assert sc["status"] == "ok"
        assert sc["window"]["points"] == 10
        assert sc["sharpe"] is not None and sc["sharpe"] > 0
        assert sc["excess_vs_spy_pct"] > 0
        assert sc["criteria"] == {
            "sharpe_positive": True, "beats_spy": True,
            "min_trades_met": True, "all_met": True,
        }
        assert sc["verdict"] == "PASS"

    def test_default_trade_bar_fails_with_zero_closed(self, db_session):
        # The live day-one path: good curve, zero closed trades.
        _seed_outperformer(db_session)
        sc = compute_scorecard(db_session, "coward", days=30, now=NOW)
        assert sc["trades"] == 0
        assert sc["criteria"]["min_trades_met"] is False
        assert sc["criteria"]["all_met"] is False
        assert sc["verdict"] == "FAIL"
        assert sc["trade_stats"]["win_rate"] is None

    def test_trade_bar_met_with_30_closed(self, db_session):
        _seed_outperformer(db_session)
        for i in range(30):
            _closed_trade(db_session, "coward", WEEKDAYS[i % len(WEEKDAYS)],
                          pnl=5.0 if i % 3 else -3.0, i=i)
        db_session.commit()
        sc = compute_scorecard(db_session, "coward", days=30, now=NOW)
        assert sc["trades"] == 30
        assert sc["criteria"]["min_trades_met"] is True
        assert sc["criteria"]["all_met"] is True
        assert sc["trade_stats"]["profit_factor"] is not None

    def test_underperformer_fails_beats_spy(self, db_session):
        for i, d in enumerate(WEEKDAYS):
            _snap(db_session, "gambler", d, 10_000.0 * (1 - 0.005 * (i + 1)))
            _spy(db_session, d, 100.0 * (1 + 0.002 * (i + 1)))
        db_session.commit()
        sc = compute_scorecard(db_session, "gambler", days=30,
                               pass_min_trades=0, now=NOW)
        assert sc["criteria"]["beats_spy"] is False
        assert sc["verdict"] == "FAIL"

    def test_inner_join_drops_unmatched_days(self, db_session):
        # Equity marks all ten days; SPY missing two of them (incl. "today").
        for i, d in enumerate(WEEKDAYS):
            _snap(db_session, "coward", d, 10_000.0 + 100 * i)
        for d in WEEKDAYS[:8]:
            _spy(db_session, d, 100.0)
        db_session.commit()
        sc = compute_scorecard(db_session, "coward", days=30,
                               pass_min_trades=0, now=NOW)
        assert sc["window"]["points"] == 8
        assert sc["window"]["end"] == WEEKDAYS[7].isoformat()

    def test_last_mark_per_day_wins(self, db_session):
        d = WEEKDAYS[0]
        _snap(db_session, "coward", d, 9_000.0, hour_utc=14)   # intraday
        _snap(db_session, "coward", d, 10_500.0, hour_utc=20)  # close mark
        _snap(db_session, "coward", WEEKDAYS[1], 10_500.0)
        _spy(db_session, d, 100.0)
        _spy(db_session, WEEKDAYS[1], 100.0)
        db_session.commit()
        sc = compute_scorecard(db_session, "coward", days=30,
                               pass_min_trades=0, now=NOW)
        # Day 1 equity must be the 20:05 close mark, so the curve is flat.
        assert sc["return_pct"] == 0.0

    def test_insufficient_data(self, db_session):
        _snap(db_session, "coward", WEEKDAYS[0], 10_000.0)
        _spy(db_session, WEEKDAYS[0], 100.0)
        db_session.commit()
        sc = compute_scorecard(db_session, "coward", days=30, now=NOW)
        assert sc["status"] == "insufficient_data"
        assert sc["sharpe"] is None
        assert sc["verdict"] == "FAIL"

    def test_weekend_rows_excluded(self, db_session):
        sat = date(2026, 5, 9)
        for d in (WEEKDAYS[0], sat, WEEKDAYS[5]):
            _snap(db_session, "coward", d, 10_000.0)
            _spy(db_session, d, 100.0)
        db_session.commit()
        sc = compute_scorecard(db_session, "coward", days=30,
                               pass_min_trades=0, now=NOW)
        assert sc["window"]["points"] == 2  # Saturday dropped from the join


class TestComputeAllScorecards:
    def test_enumerates_registry_with_empty_db(self, db_session):
        cards = compute_all_scorecards(db_session, days=30, now=NOW)
        names = {c["strategy_name"] for c in cards}
        assert {"coward", "gambler", "degenerate"} <= names
        assert all(c["status"] == "insufficient_data" for c in cards)


class TestScorecardEndpoint:
    def test_endpoint_returns_cards(self, db_session, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from dashboard.dependencies import get_db
        from dashboard.routers import strategies as strategies_router

        _seed_outperformer(db_session)
        app = FastAPI()
        app.include_router(strategies_router.router, prefix="/api/strategies")
        app.dependency_overrides[get_db] = lambda: db_session

        client = TestClient(app)
        resp = client.get("/api/strategies/scorecard?days=90&strategy=coward")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["strategy_name"] == "coward"
        assert "criteria" in body[0] and "verdict" in body[0]
