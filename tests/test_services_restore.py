"""Regression tests for position restore from DB (dashboard/services.py).

The DB's DateTime columns round-trip naive datetimes, but the arena compares
position.entry_time against tz-aware UTC now (max-hold, hold-hours). A naive
entry_time crashed every live intraday cycle with "can't subtract offset-naive
and offset-aware datetimes" (caught by the system_heartbeat on 2026-06-05).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

import dashboard.services as services
from edgefinder.db.models import TradeRecord
from edgefinder.trading.account import VirtualAccount


class _StubArena:
    def __init__(self, account: VirtualAccount):
        self._account = account

    def get_account(self, name: str):
        return self._account if name == self._account.strategy_name else None


def test_restored_entry_time_is_coerced_to_aware_utc(db_engine, monkeypatch):
    sf = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = sf()
    session.add(TradeRecord(
        trade_id="t-naive-restore",
        strategy_name="alpha",
        symbol="AAPL",
        direction="LONG",
        trade_type="SWING",
        entry_price=100.0,
        shares=10,
        stop_loss=95.0,
        target=110.0,
        confidence=0.7,
        entry_time=datetime(2026, 5, 1, 14, 30),  # NAIVE — as the DB returns it
        status="OPEN",
    ))
    session.commit()
    session.close()

    acct = VirtualAccount("alpha", starting_capital=10_000.0)
    monkeypatch.setattr(services, "_arena", _StubArena(acct))
    monkeypatch.setattr(services, "_session_factory", sf)
    monkeypatch.setattr(services, "_fetch_startup_prices", lambda: None)

    services._restore_open_positions()

    assert len(acct.positions) == 1
    entry_time = acct.positions[0].entry_time
    assert entry_time.tzinfo is not None, "restored entry_time must be tz-aware"
    # The exact arithmetic the live cycle crashed on (arena max-hold check):
    age_days = (datetime.now(timezone.utc) - entry_time).days
    assert age_days >= 0
