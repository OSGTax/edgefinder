"""Plan item C3 — revenge-trade cooldown restore across restart.

Proves _restore_revenge_cooldowns reads the most recent STOP_HIT exit
per strategy and seeds account._last_stop_out so the 30-minute cooldown
survives process restarts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import sessionmaker

import dashboard.services as services
from edgefinder.db.models import TradeRecord
from edgefinder.trading.account import VirtualAccount


@pytest.fixture
def services_wired(db_engine):
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)

    fake_arena = MagicMock()
    accounts: dict[str, VirtualAccount] = {}

    def _get_account(name: str):
        if name not in accounts:
            accounts[name] = VirtualAccount(name)
        return accounts[name]

    fake_arena.get_account.side_effect = _get_account
    fake_arena.get_strategy_names.return_value = ["alpha", "bravo"]

    orig_factory = services._session_factory
    orig_arena = services._arena
    services._session_factory = factory
    services._arena = fake_arena
    try:
        yield services, factory, accounts, fake_arena
    finally:
        services._session_factory = orig_factory
        services._arena = orig_arena


def _seed_closed_trade(session, strategy_name: str, exit_reason: str,
                       exit_time: datetime):
    session.add(TradeRecord(
        trade_id=f"{strategy_name}-{exit_time.isoformat()}",
        strategy_name=strategy_name,
        symbol="XYZ",
        direction="LONG",
        trade_type="SWING",
        entry_price=100.0,
        shares=10,
        stop_loss=95.0,
        target=110.0,
        confidence=80,
        entry_time=exit_time - timedelta(minutes=30),
        exit_time=exit_time,
        exit_price=95.0,
        exit_reason=exit_reason,
        status="CLOSED",
        pnl_dollars=-50.0,
    ))
    session.flush()


class TestRevengeCooldownRestore:
    def test_restores_last_stop_out_from_recent_stop_hit(self, services_wired):
        services_mod, factory, accounts, _arena = services_wired
        recent = datetime.now(timezone.utc) - timedelta(minutes=10)
        session = factory()
        try:
            _seed_closed_trade(session, "alpha", "STOP_HIT", recent)
            session.commit()
        finally:
            session.close()

        services_mod._restore_revenge_cooldowns()
        assert accounts["alpha"]._last_stop_out is not None
        # Allow for sub-second DB tz rounding.
        assert abs(
            (accounts["alpha"]._last_stop_out - recent).total_seconds()
        ) < 2

    def test_ignores_target_hit_exits(self, services_wired):
        services_mod, factory, accounts, _ = services_wired
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        session = factory()
        try:
            _seed_closed_trade(session, "alpha", "TARGET_HIT", recent)
            session.commit()
        finally:
            session.close()

        services_mod._restore_revenge_cooldowns()
        assert "alpha" not in accounts or accounts["alpha"]._last_stop_out is None

    def test_ignores_stop_hits_outside_cooldown_window(self, services_wired):
        services_mod, factory, accounts, _ = services_wired
        stale = datetime.now(timezone.utc) - timedelta(hours=2)
        session = factory()
        try:
            _seed_closed_trade(session, "alpha", "STOP_HIT", stale)
            session.commit()
        finally:
            session.close()

        services_mod._restore_revenge_cooldowns()
        assert "alpha" not in accounts or accounts["alpha"]._last_stop_out is None

    def test_picks_most_recent_stop_hit_per_strategy(self, services_wired):
        services_mod, factory, accounts, _ = services_wired
        older = datetime.now(timezone.utc) - timedelta(minutes=25)
        newer = datetime.now(timezone.utc) - timedelta(minutes=5)
        session = factory()
        try:
            _seed_closed_trade(session, "alpha", "STOP_HIT", older)
            _seed_closed_trade(session, "alpha", "STOP_HIT", newer)
            session.commit()
        finally:
            session.close()

        services_mod._restore_revenge_cooldowns()
        assert abs(
            (accounts["alpha"]._last_stop_out - newer).total_seconds()
        ) < 2
