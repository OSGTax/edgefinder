"""Tests for plan items C2 (hash chain), C4 (RSI flat), C5 (PDT biz days)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from edgefinder.core.models import Direction, Trade, TradeStatus, TradeType
from edgefinder.db.models import TradeRecord
from edgefinder.signals.engine import _rsi
from edgefinder.trading.account import VirtualAccount
from edgefinder.trading.executor import Executor, _compute_chain_hash
from edgefinder.trading.journal import TradeJournal


@pytest.fixture
def session_factory(db_engine):
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _write_open_trade(journal: TradeJournal, account: VirtualAccount, trade_id: str,
                     sequence_num: int, prev_hash: str):
    h = _compute_chain_hash(trade_id, sequence_num, prev_hash)
    trade = Trade(
        trade_id=trade_id,
        strategy_name=account.strategy_name,
        symbol="AAA",
        direction=Direction.LONG,
        trade_type=TradeType.SWING,
        entry_price=100.0,
        shares=10,
        stop_loss=95.0,
        target=110.0,
        confidence=70,
        status=TradeStatus.OPEN,
        entry_time=datetime.now(timezone.utc),
        sequence_num=sequence_num,
        integrity_hash=h,
    )
    journal.log_trade(trade)
    return h


class TestHashChainRestore:
    def test_restore_seeds_sequence_and_prev_hash(self, session_factory):
        account = VirtualAccount("alpha")
        session = session_factory()
        try:
            journal = TradeJournal(session)
            _write_open_trade(journal, account, "t1", 1, "")
            h2 = _write_open_trade(
                journal, account, "t2", 2, _compute_chain_hash("t1", 1, ""),
            )
        finally:
            session.close()

        session = session_factory()
        try:
            executor = Executor(account)
            assert executor._sequence_num == 0
            assert executor._prev_hash == ""
            executor.restore_hash_chain(session)
            assert executor._sequence_num == 2
            assert executor._prev_hash == h2
        finally:
            session.close()

    def test_verify_chain_detects_tamper(self, session_factory):
        account = VirtualAccount("alpha")
        session = session_factory()
        try:
            journal = TradeJournal(session)
            _write_open_trade(journal, account, "t1", 1, "")
            h1 = _compute_chain_hash("t1", 1, "")
            _write_open_trade(journal, account, "t2", 2, h1)
            session.query(TradeRecord).filter_by(trade_id="t2").update(
                {"integrity_hash": "deadbeef" * 8}
            )
            session.commit()
        finally:
            session.close()

        session = session_factory()
        try:
            executor = Executor(account)
            ok, checked = executor.verify_chain(session)
            assert ok is False
            assert checked == 1  # t1 ok, t2 caught
        finally:
            session.close()

    def test_verify_passes_on_clean_chain(self, session_factory):
        account = VirtualAccount("alpha")
        session = session_factory()
        try:
            journal = TradeJournal(session)
            _write_open_trade(journal, account, "t1", 1, "")
            h1 = _compute_chain_hash("t1", 1, "")
            _write_open_trade(journal, account, "t2", 2, h1)
        finally:
            session.close()

        session = session_factory()
        try:
            executor = Executor(account)
            ok, checked = executor.verify_chain(session)
            assert ok is True
            assert checked == 2
        finally:
            session.close()

    def test_new_trade_chains_across_restart(self, session_factory):
        """The core guarantee: after a restart, sequence_num and prev_hash
        continue from where they left off, not from scratch."""
        account = VirtualAccount("alpha")
        session = session_factory()
        try:
            journal = TradeJournal(session)
            _write_open_trade(journal, account, "t1", 1, "")
        finally:
            session.close()

        # Simulate process restart: brand new executor.
        session = session_factory()
        try:
            executor = Executor(account)
            executor.restore_hash_chain(session)
            h1 = _compute_chain_hash("t1", 1, "")
            assert executor._prev_hash == h1
            # Now compute the next hash — should chain to h1, not ""
            executor._sequence_num += 1
            h2 = executor._compute_hash("t2")
            assert h2 == _compute_chain_hash("t2", 2, h1)
        finally:
            session.close()


class TestRsiFlatMarket:
    def test_flat_price_gives_rsi_50(self):
        close = pd.Series([100.0] * 50)
        rsi = _rsi(close, period=14)
        tail = rsi.iloc[20:]
        assert (tail == 50.0).all(), f"Flat RSI should be 50, got: {tail.unique()}"

    def test_all_gains_gives_rsi_100(self):
        close = pd.Series(range(100, 150))  # monotonic up
        rsi = _rsi(close, period=14).dropna()
        assert rsi.iloc[-1] == pytest.approx(100.0)

    def test_all_losses_gives_rsi_0(self):
        close = pd.Series(range(150, 100, -1))  # monotonic down
        rsi = _rsi(close, period=14).dropna()
        assert rsi.iloc[-1] == pytest.approx(0.0)


class TestPdtBusinessDays:
    def test_calendar_week_boundary_still_counted_as_business_days(self):
        """Three day trades on Mon, Tue, Wed of last week should still count
        against today (Tue this week) because that's only 4 business days.
        """
        acct = VirtualAccount("alpha", pdt_enabled=True)
        now = datetime.now(timezone.utc)
        # Pin three trades just inside the 5-business-day window.
        last_biz = np.busday_offset(
            np.datetime64(now.date()), -3, roll="backward"
        ).astype("datetime64[D]").astype(object)
        t = datetime.combine(last_biz, datetime.min.time(), tzinfo=timezone.utc)
        acct._day_trades = [t, t + timedelta(hours=1), t + timedelta(hours=2)]
        assert acct._can_day_trade() is False  # 3 trades in last 3 biz days

    def test_trade_outside_window_doesnt_count(self):
        acct = VirtualAccount("alpha", pdt_enabled=True)
        # Old trade, comfortably outside any 5-business-day window.
        old = datetime.now(timezone.utc) - timedelta(days=30)
        acct._day_trades = [old]
        assert acct._can_day_trade() is True
