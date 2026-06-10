"""Tests for the liveness heartbeat writer in dashboard/services.py."""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

import dashboard.services as services
from edgefinder.db.models import SystemHeartbeat


def test_record_heartbeat_upserts_single_row(db_engine, monkeypatch):
    sf = sessionmaker(bind=db_engine, expire_on_commit=False)
    monkeypatch.setattr(services, "_session_factory", sf)

    services._record_heartbeat("v2_portfolio_cycle", ok=True, detail={"opened": 1})
    services._record_heartbeat("v2_portfolio_cycle", ok=False, detail={"error": "boom"})

    session = sf()
    try:
        rows = session.query(SystemHeartbeat).all()
        assert len(rows) == 1  # upsert, not append
        assert rows[0].component == "v2_portfolio_cycle"
        assert not rows[0].ok  # flipped True -> False
        assert rows[0].detail == {"error": "boom"}
    finally:
        session.close()


def test_record_heartbeat_no_factory_is_noop(monkeypatch):
    # Before init (no session factory) the writer must silently no-op,
    # never raise — heartbeat bookkeeping can't break the cycle.
    monkeypatch.setattr(services, "_session_factory", None)
    services._record_heartbeat("v2_portfolio_cycle", ok=True, detail={})


def test_v2_snapshot_job_marks_accounts_and_appends_snapshot(db_engine, monkeypatch):
    """The 30-min v2 snapshot: recompute cash from trades, mark the account,
    append one StrategySnapshot row, write the v2_snapshot heartbeat."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from edgefinder.db.models import (
        PromotedStrategy, StrategyAccount, StrategySnapshot, TradeRecord,
    )

    sf = sessionmaker(bind=db_engine, expire_on_commit=False)
    monkeypatch.setattr(services, "_session_factory", sf)
    provider = MagicMock()
    provider.get_latest_price.return_value = 110.0
    monkeypatch.setattr(services, "_provider", provider)

    session = sf()
    try:
        session.add(PromotedStrategy(
            strategy_name="equal_weight", spec="equal_weight",
            symbols=["AAA"], schedule="monthly", tier="paper", active=True))
        session.add(StrategyAccount(
            strategy_name="equal_weight", starting_capital=100_000.0,
            cash_balance=0.0, peak_equity=100_000.0))
        # one open lot: 10 shares @ $100
        session.add(TradeRecord(
            trade_id="lot-1", strategy_name="equal_weight", symbol="AAA",
            direction="LONG", trade_type="SWING", entry_price=100.0, shares=10,
            stop_loss=0.0, target=0.0, confidence=1.0,
            entry_time=datetime.now(timezone.utc), status="OPEN"))
        session.commit()
    finally:
        session.close()

    services._v2_snapshot_job()

    session = sf()
    try:
        acct = session.query(StrategyAccount).one()
        assert acct.cash_balance == 99_000.0          # 100k - 10*100 cost basis
        assert acct.open_positions_value == 1_100.0   # 10 * live 110
        assert acct.total_equity == 100_100.0

        [snap] = session.query(StrategySnapshot).all()
        assert snap.strategy_name == "equal_weight"
        assert snap.total_equity == 100_100.0
        assert snap.total_return_pct == 0.1           # vs $100k start

        from edgefinder.db.models import SystemHeartbeat
        hb = (session.query(SystemHeartbeat)
              .filter(SystemHeartbeat.component == "v2_snapshot").one())
        assert hb.ok
    finally:
        session.close()


def test_v2_snapshot_job_skips_cleanly_when_nothing_promoted(db_engine, monkeypatch):
    from unittest.mock import MagicMock

    from edgefinder.db.models import StrategySnapshot, SystemHeartbeat

    sf = sessionmaker(bind=db_engine, expire_on_commit=False)
    monkeypatch.setattr(services, "_session_factory", sf)
    monkeypatch.setattr(services, "_provider", MagicMock())

    services._v2_snapshot_job()

    session = sf()
    try:
        assert session.query(StrategySnapshot).count() == 0
        hb = (session.query(SystemHeartbeat)
              .filter(SystemHeartbeat.component == "v2_snapshot").one())
        assert hb.ok  # controlled skip writes a fresh ok heartbeat
    finally:
        session.close()
