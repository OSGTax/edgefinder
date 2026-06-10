"""Tests for validation-run persistence + the offline-verdict API + trade
timeline serialization (the P2/P4 dashboard-verifiability fixes)."""

from __future__ import annotations

from datetime import datetime, timezone

from edgefinder.engine.record import record_validation_run
from edgefinder.db.models import TradeRecord, ValidationRun


def _scorecard(strategy: str, *, all_met: bool, holdout_passes: bool | None,
               verdict: str = "FAIL") -> dict:
    return {
        "strategy": strategy,
        "config": {"is_days": 252, "oos_days": 63, "step_days": 126, "num_folds": 5},
        "oos": {"total_return_pct": 1.0, "mean_sharpe": 0.5, "total_trades": 40},
        "criteria": {
            "sharpe_positive": True, "beats_spy_majority_folds": True,
            "min_trades_met": True, "min_trades_threshold": 30, "all_met": all_met,
        },
        "holdout": None if holdout_passes is None else {
            "window": "2025-11-21..2026-05-26", "passes": holdout_passes,
            "sharpe": 1.0, "excess_vs_spy_pct": 2.0, "trades": 10,
        },
        "verdict": verdict,
    }


class TestRecordValidationRun:
    def test_round_trip(self, db_session):
        rid = record_validation_run(
            db_session, _scorecard("degenerate", all_met=True, holdout_passes=False,
                                   verdict="PASS"),
            universe="top-300", git_sha="abc1234",
        )
        row = db_session.get(ValidationRun, rid)
        assert row.strategy_name == "degenerate"
        assert row.universe == "top-300"
        assert row.criteria["all_met"] is True
        assert row.holdout["passes"] is False
        assert row.verdict == "PASS"


class TestValidationEndpoint:
    def _client(self, db_session):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from dashboard.dependencies import get_db
        from dashboard.routers import strategies as strategies_router

        app = FastAPI()
        app.include_router(strategies_router.router, prefix="/api/strategies")
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app)

    def test_latest_per_strategy_and_validated_flag(self, db_session):
        # Older run: criteria met, holdout failed → NOT validated.
        record_validation_run(
            db_session, _scorecard("degenerate", all_met=True, holdout_passes=False,
                                   verdict="PASS"),
            universe="top-300",
        )
        # coward: criteria unmet → not validated.
        record_validation_run(
            db_session, _scorecard("coward", all_met=False, holdout_passes=False),
            universe="top-200",
        )
        # A hypothetical fully-passing strategy (criteria + holdout).
        record_validation_run(
            db_session, _scorecard("gambler", all_met=True, holdout_passes=True,
                                   verdict="PASS"),
            universe="top-200",
        )

        body = self._client(db_session).get("/api/strategies/validation").json()
        by_name = {r["strategy_name"]: r for r in body}
        assert by_name["degenerate"]["validated"] is False  # holdout failed
        assert by_name["coward"]["validated"] is False
        assert by_name["gambler"]["validated"] is True
        assert by_name["degenerate"]["verdict"] == "PASS"  # raw verdict preserved

    def test_sealed_unevaluated_holdout_is_not_validated(self, db_session):
        # Research-stage runs reserve the holdout without evaluating it:
        # criteria can pass, but "validated" requires a PASSING holdout.
        record_validation_run(
            db_session, _scorecard("coward", all_met=True, holdout_passes=None,
                                   verdict="PASS"),
            universe="top-50",
        )
        body = self._client(db_session).get("/api/strategies/validation").json()
        assert body[0]["validated"] is False
        assert body[0]["verdict"] == "PASS"  # raw verdict still visible


class TestTradeTimelineSerialization:
    def test_enrich_trade_includes_timeline_fields(self, db_session):
        from dashboard.routers.trades import _enrich_trade

        t = TradeRecord(
            trade_id="t-timeline", strategy_name="coward", symbol="AAPL",
            direction="LONG", trade_type="SWING", entry_price=100.0, shares=5,
            stop_loss=95.0, target=110.0, confidence=0.7,
            entry_time=datetime(2026, 6, 1, 14, 30), status="CLOSED",
            pnl_dollars=25.0, exit_time=datetime(2026, 6, 3, 18, 0),
            entry_reasoning="RSI oversold at 28.4",
            exit_reasoning="Target hit",
            indicators_at_entry={"rsi": 28.4, "close": 100.0},
            indicators_at_exit={"rsi": 71.2, "close": 105.0},
            hold_duration_hours=51.5,
        )
        db_session.add(t)
        db_session.commit()

        out = _enrich_trade(t)
        assert out["entry_reasoning"] == "RSI oversold at 28.4"
        assert out["exit_reasoning"] == "Target hit"
        assert out["indicators_at_entry"]["rsi"] == 28.4
        assert out["indicators_at_exit"]["close"] == 105.0
        assert out["hold_duration_hours"] == 51.5

    def test_as_dict_tolerates_strings_and_garbage(self):
        from dashboard.routers.trades import _as_dict

        assert _as_dict(None) is None
        assert _as_dict({"a": 1}) == {"a": 1}
        assert _as_dict('{"rsi": 30.5}') == {"rsi": 30.5}
        assert _as_dict("not json") is None
        assert _as_dict("[1,2]") is None  # non-dict JSON rejected
