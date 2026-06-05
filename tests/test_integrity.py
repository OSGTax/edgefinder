"""Tests for the v2 trade hash chain (journal-anchored) + verifier."""

from __future__ import annotations

from datetime import datetime, timezone

from edgefinder.core.models import Direction, Trade, TradeStatus, TradeType
from edgefinder.db.models import TradeRecord
from edgefinder.trading.integrity import verify_chain
from edgefinder.trading.journal import TradeJournal


def _trade(trade_id: str, strategy: str = "coward", symbol: str = "AAPL") -> Trade:
    return Trade(
        trade_id=trade_id,
        strategy_name=strategy,
        symbol=symbol,
        direction=Direction.LONG,
        trade_type=TradeType.SWING,
        entry_price=100.0,
        shares=5,
        stop_loss=95.0,
        target=110.0,
        confidence=70,
        status=TradeStatus.OPEN,
        entry_time=datetime(2026, 6, 1, 14, 30, tzinfo=timezone.utc),
    )


class TestChainWriting:
    def test_sequential_links_per_strategy(self, db_session):
        j = TradeJournal(db_session)
        j.log_trade(_trade("t1"))
        j.log_trade(_trade("t2", symbol="MSFT"))
        j.log_trade(_trade("g1", strategy="gambler"))

        rows = {r.trade_id: r for r in db_session.query(TradeRecord).all()}
        assert rows["t1"].sequence_num == 1
        assert rows["t2"].sequence_num == 2
        assert rows["g1"].sequence_num == 1  # independent chain per strategy
        assert rows["t1"].integrity_hash != rows["t2"].integrity_hash

    def test_chain_survives_restart(self, db_session):
        # A fresh journal instance (new process) must continue the chain
        # from the DB, not reset it — the v1 bug this scheme replaces.
        TradeJournal(db_session).log_trade(_trade("t1"))
        TradeJournal(db_session).log_trade(_trade("t2", symbol="MSFT"))

        report = verify_chain(db_session)
        assert report["ok"] is True
        assert report["totals"]["verified"] == 2

    def test_close_update_does_not_touch_chain(self, db_session):
        j = TradeJournal(db_session)
        t = _trade("t1")
        j.log_trade(t)
        before = db_session.query(TradeRecord).filter_by(trade_id="t1").one()
        seq, h = before.sequence_num, before.integrity_hash

        t.status = TradeStatus.CLOSED
        t.exit_price = 110.0
        t.exit_time = datetime(2026, 6, 2, 18, 0, tzinfo=timezone.utc)
        t.pnl_dollars = 50.0
        j.log_trade(t)  # update path

        after = db_session.query(TradeRecord).filter_by(trade_id="t1").one()
        assert after.status == "CLOSED"
        assert (after.sequence_num, after.integrity_hash) == (seq, h)
        assert verify_chain(db_session)["ok"] is True


class TestVerifier:
    def test_tamper_breaks_row_and_successor(self, db_session):
        j = TradeJournal(db_session)
        for i in range(1, 5):
            j.log_trade(_trade(f"t{i}", symbol=f"SYM{i}"))

        # Tamper with row 2's stored hash.
        row2 = db_session.query(TradeRecord).filter_by(trade_id="t2").one()
        row2.integrity_hash = "0" * 64
        db_session.commit()

        report = verify_chain(db_session)
        strat = report["strategies"]["coward"]
        assert report["ok"] is False
        # Row 2 fails its own check; row 3 chained on the tampered stored
        # hash so it fails too; row 4 re-verifies against row 3's stored hash.
        broken = {b["trade_id"] for b in strat["breaks"]}
        assert broken == {"t2", "t3"}
        assert strat["verified"] == 2

    def test_v2_rows_chain_on_top_of_legacy(self, db_session):
        # Simulate legacy rows (pre-v2): arbitrary hashes that don't verify.
        db_session.add(TradeRecord(
            trade_id="legacy1", strategy_name="coward", symbol="OLD",
            direction="LONG", trade_type="SWING", entry_price=50.0, shares=1,
            stop_loss=45.0, target=60.0, confidence=0.5,
            entry_time=datetime(2026, 5, 1, 14, 0), status="OPEN",
            sequence_num=7, integrity_hash="f" * 64,
        ))
        db_session.commit()

        TradeJournal(db_session).log_trade(_trade("new1"))

        report = verify_chain(db_session)
        strat = report["strategies"]["coward"]
        new_row = db_session.query(TradeRecord).filter_by(trade_id="new1").one()
        assert new_row.sequence_num == 8  # continues above legacy numbering
        assert strat["legacy_unverified"] == 1
        assert strat["verified"] == 1     # the new row verifies
        assert strat["intact_tail"] is True

    def test_empty_db_is_ok(self, db_session):
        report = verify_chain(db_session)
        assert report["ok"] is True
        assert report["totals"]["total"] == 0


class TestIntegrityEndpoint:
    def test_endpoint_reports_chain(self, db_session):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from dashboard.dependencies import get_db
        from dashboard.routers import trades as trades_router

        TradeJournal(db_session).log_trade(_trade("t1"))
        TradeJournal(db_session).log_trade(_trade("t2", symbol="MSFT"))

        app = FastAPI()
        app.include_router(trades_router.router, prefix="/api/trades")
        app.dependency_overrides[get_db] = lambda: db_session
        body = TestClient(app).get("/api/trades/integrity").json()
        assert body["ok"] is True
        assert body["totals"]["verified"] == 2
        assert body["strategies"]["coward"]["intact_tail"] is True
