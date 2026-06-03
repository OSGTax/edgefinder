"""Tests for the liveness heartbeat writer in dashboard/services.py."""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker

import dashboard.services as services
from edgefinder.db.models import SystemHeartbeat


def test_record_heartbeat_upserts_single_row(db_engine, monkeypatch):
    sf = sessionmaker(bind=db_engine, expire_on_commit=False)
    monkeypatch.setattr(services, "_session_factory", sf)

    services._record_heartbeat("intraday_cycle", ok=True, detail={"opened": 1})
    services._record_heartbeat("intraday_cycle", ok=False, detail={"error": "boom"})

    session = sf()
    try:
        rows = session.query(SystemHeartbeat).all()
        assert len(rows) == 1  # upsert, not append
        assert rows[0].component == "intraday_cycle"
        assert not rows[0].ok  # flipped True -> False
        assert rows[0].detail == {"error": "boom"}
    finally:
        session.close()


def test_record_heartbeat_no_factory_is_noop(monkeypatch):
    # Before init (no session factory) the writer must silently no-op,
    # never raise — heartbeat bookkeeping can't break the cycle.
    monkeypatch.setattr(services, "_session_factory", None)
    services._record_heartbeat("intraday_cycle", ok=True, detail={})
