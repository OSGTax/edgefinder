"""
EdgeFinder Scheduler Tests
============================
Tests cover: scheduler creation, job execution, signal-to-trade pipeline,
position monitoring, day trade closing, state persistence, cold-start
recovery, and edge cases.

Run: python -m pytest tests/test_scheduler.py -v
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from modules.scheduler import (
    create_scheduler,
    start_scheduler,
    stop_scheduler,
    get_scheduler_status,
    restore_trader_state,
    save_open_position,
    _remove_open_position,
    _fetch_current_price,
    job_nightly_scan,
    job_signal_check,
    job_position_monitor,
    job_close_day_trades,
    job_account_snapshot,
)
from modules.trader import PaperTrader, AccountState, Position
from modules.database import (
    Trade as TradeRecord,
    AccountSnapshot,
    get_session,
)
from config import settings
import modules.scheduler as sched_module


# ── FIXTURES ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_scheduler():
    """Reset scheduler module state between tests."""
    sched_module._scheduler = None
    sched_module._trader = PaperTrader()
    sched_module._journal = MagicMock()
    sched_module._status = {
        "running": False,
        "jobs_run": 0,
        "last_signal_check": None,
        "last_position_monitor": None,
        "last_scan": None,
        "open_positions": 0,
        "errors": [],
    }
    yield
    if sched_module._scheduler and sched_module._scheduler.running:
        sched_module._scheduler.shutdown(wait=False)


# ════════════════════════════════════════════════════════════
# SCHEDULER CREATION
# ════════════════════════════════════════════════════════════

class TestSchedulerCreation:
    """Test scheduler setup and configuration."""

    def test_create_scheduler_returns_scheduler(self):
        scheduler = create_scheduler()
        assert scheduler is not None
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "signal_check" in job_ids
        assert "position_monitor" in job_ids
        assert "close_day_trades" in job_ids
        assert "account_snapshot" in job_ids
        assert "nightly_scan" in job_ids

    def test_scheduler_has_10_jobs(self):
        scheduler = create_scheduler()
        assert len(scheduler.get_jobs()) == 10  # 5 v1 + 5 arena

    def test_start_scheduler_is_idempotent(self, in_memory_db):
        with patch("modules.scheduler.get_active_watchlist", return_value=[{"ticker": "X"}]):
            start_scheduler()
            start_scheduler()  # Should not raise
            assert sched_module._scheduler.running
            stop_scheduler()

    def test_get_status_when_not_running(self):
        status = get_scheduler_status()
        assert status["running"] is False
        assert status["jobs"] == []


# ════════════════════════════════════════════════════════════
# NIGHTLY SCAN JOB
# ════════════════════════════════════════════════════════════

class TestNightlyScan:
    """Test the nightly fundamental scan job."""

    def test_scan_uses_default_tickers(self, in_memory_db):
        with patch("modules.scheduler.run_scan", return_value=[]) as mock_scan:
            job_nightly_scan()
            mock_scan.assert_called_once()
            tickers_arg = mock_scan.call_args[1]["tickers"]
            assert len(tickers_arg) > 50  # Our curated list

    def test_scan_updates_status(self, in_memory_db):
        with patch("modules.scheduler.run_scan", return_value=[]):
            job_nightly_scan()
            assert sched_module._status["last_scan"] is not None
            assert sched_module._status["jobs_run"] == 1

    def test_scan_handles_errors(self, in_memory_db):
        with patch("modules.scheduler.run_scan", side_effect=Exception("API down")):
            job_nightly_scan()  # Should not raise
            assert len(sched_module._status["errors"]) == 1


# ════════════════════════════════════════════════════════════
# SIGNAL CHECK JOB
# ════════════════════════════════════════════════════════════

class TestSignalCheck:
    """Test the signal check → sentiment → trade pipeline."""

    def test_skips_when_no_watchlist(self, in_memory_db):
        with patch("modules.scheduler.get_active_watchlist", return_value=[]):
            job_signal_check()
            # Should log warning and return without error

    def test_signal_blocked_by_sentiment(self, in_memory_db):
        mock_signal = MagicMock()
        mock_signal.ticker = "BAD"
        mock_signal.signal_type = "BUY"
        mock_signal.trade_type = "DAY"
        mock_signal.confidence = 70.0
        mock_signal.indicators = {}
        mock_signal.price = 50.0

        mock_sentiment = MagicMock()
        mock_sentiment.reason = "Fraud news"
        mock_sentiment.avg_compound = -0.8

        with patch("modules.scheduler.get_active_watchlist", return_value=[{"ticker": "BAD", "sector": "Tech", "composite_score": 70}]), \
             patch("modules.scheduler.scan_watchlist", return_value=[mock_signal]), \
             patch("modules.scheduler.gate_trade", return_value=("BLOCK", 0.0, mock_sentiment)):
            job_signal_check()
            sched_module._journal.log_skipped_signal.assert_called_once()

    def test_signal_executes_trade(self, in_memory_db):
        mock_signal = MagicMock()
        mock_signal.ticker = "GOOD"
        mock_signal.signal_type = "BUY"
        mock_signal.trade_type = "DAY"
        mock_signal.confidence = 70.0
        mock_signal.indicators = {"rsi_oversold": {"rsi": 25}}
        mock_signal.price = 50.0

        mock_sentiment = MagicMock()
        mock_sentiment.reason = "Neutral"
        mock_sentiment.avg_compound = 0.1

        # Use a real trader
        sched_module._trader = PaperTrader()

        with patch("modules.scheduler.get_active_watchlist", return_value=[{"ticker": "GOOD", "sector": "Tech", "composite_score": 70}]), \
             patch("modules.scheduler.scan_watchlist", return_value=[mock_signal]), \
             patch("modules.scheduler.gate_trade", return_value=("PROCEED", 70.0, mock_sentiment)), \
             patch("modules.scheduler.save_open_position"):
            job_signal_check()
            assert sched_module._trader.account.open_position_count == 1

    def test_signal_check_handles_errors(self, in_memory_db):
        with patch("modules.scheduler.get_active_watchlist", side_effect=Exception("DB error")):
            job_signal_check()  # Should not raise
            assert len(sched_module._status["errors"]) == 1


# ════════════════════════════════════════════════════════════
# POSITION MONITOR JOB
# ════════════════════════════════════════════════════════════

class TestPositionMonitor:
    """Test position price monitoring and stop/target checking."""

    def test_no_positions_skips(self, in_memory_db):
        job_position_monitor()  # Should do nothing without error

    def test_stop_hit_closes_position(self, in_memory_db):
        trader = PaperTrader()
        pos = trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        sched_module._trader = trader

        with patch("modules.scheduler._fetch_current_price", return_value=97.0), \
             patch("modules.scheduler._remove_open_position"):
            job_position_monitor()
            assert trader.account.open_position_count == 0
            sched_module._journal.log_trade.assert_called_once()

    def test_target_hit_closes_position(self, in_memory_db):
        trader = PaperTrader()
        pos = trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        sched_module._trader = trader

        with patch("modules.scheduler._fetch_current_price", return_value=104.0), \
             patch("modules.scheduler._remove_open_position"):
            job_position_monitor()
            assert trader.account.open_position_count == 0

    def test_price_in_range_no_close(self, in_memory_db):
        trader = PaperTrader()
        trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        sched_module._trader = trader

        with patch("modules.scheduler._fetch_current_price", return_value=101.0):
            job_position_monitor()
            assert trader.account.open_position_count == 1

    def test_price_fetch_failure_skips(self, in_memory_db):
        trader = PaperTrader()
        trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=5,
        )
        sched_module._trader = trader

        with patch("modules.scheduler._fetch_current_price", return_value=None):
            job_position_monitor()
            assert trader.account.open_position_count == 1  # Still open


# ════════════════════════════════════════════════════════════
# CLOSE DAY TRADES JOB
# ════════════════════════════════════════════════════════════

class TestCloseDayTrades:
    """Test end-of-day day trade closing."""

    def test_closes_day_trades_only(self, in_memory_db):
        trader = PaperTrader(account=AccountState(cash=10_000.0, peak_value=10_000.0))
        trader.open_position(
            ticker="DAY1", entry_price=50.0, stop_loss=48.0,
            target=53.0, shares=5, trade_type="DAY",
        )
        trader.open_position(
            ticker="SWING1", entry_price=50.0, stop_loss=48.0,
            target=53.0, shares=5, trade_type="SWING",
        )
        sched_module._trader = trader

        with patch("modules.scheduler._fetch_current_price", return_value=51.0), \
             patch("modules.scheduler._remove_open_position"):
            job_close_day_trades()
            # Only swing should remain
            assert trader.account.open_position_count == 1
            remaining = list(trader.account.positions.values())[0]
            assert remaining.trade_type == "SWING"

    def test_no_day_trades_does_nothing(self, in_memory_db):
        trader = PaperTrader()
        trader.open_position(
            ticker="SWING", entry_price=50.0, stop_loss=48.0,
            target=53.0, shares=5, trade_type="SWING",
        )
        sched_module._trader = trader
        job_close_day_trades()
        assert trader.account.open_position_count == 1


# ════════════════════════════════════════════════════════════
# ACCOUNT SNAPSHOT JOB
# ════════════════════════════════════════════════════════════

class TestAccountSnapshot:
    def test_saves_snapshot(self, in_memory_db):
        trader = PaperTrader()
        sched_module._trader = trader
        with patch.object(trader, "save_account_snapshot") as mock_save:
            job_account_snapshot()
            mock_save.assert_called_once()
            assert sched_module._status["jobs_run"] == 1


# ════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ════════════════════════════════════════════════════════════

class TestStatePersistence:
    """Test saving/restoring trader state for cold-start recovery."""

    def test_save_open_position(self, in_memory_db):
        pos = Position(
            trade_id="test-001", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=98.0, target=103.0,
        )
        save_open_position(pos)

        session = get_session()
        record = session.query(TradeRecord).filter(
            TradeRecord.trade_id == "test-001"
        ).first()
        assert record is not None
        assert record.status == "OPEN"
        assert record.ticker == "AAPL"
        session.close()

    def test_save_duplicate_position_no_error(self, in_memory_db):
        pos = Position(
            trade_id="test-001", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=98.0, target=103.0,
        )
        save_open_position(pos)
        save_open_position(pos)  # Should not raise or duplicate

        session = get_session()
        count = session.query(TradeRecord).filter(
            TradeRecord.trade_id == "test-001"
        ).count()
        assert count == 1
        session.close()

    def test_remove_open_position(self, in_memory_db):
        pos = Position(
            trade_id="test-001", ticker="AAPL", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=98.0, target=103.0,
        )
        save_open_position(pos)
        _remove_open_position("test-001")

        session = get_session()
        record = session.query(TradeRecord).filter(
            TradeRecord.trade_id == "test-001"
        ).first()
        assert record.status == "CLOSED"
        session.close()

    def test_restore_empty_state(self, in_memory_db):
        trader = restore_trader_state()
        assert isinstance(trader, PaperTrader)
        assert trader.account.cash == settings.STARTING_CAPITAL
        assert trader.account.open_position_count == 0

    def test_restore_with_open_positions(self, in_memory_db):
        # Save an open position
        session = get_session()
        session.add(TradeRecord(
            trade_id="restore-001", ticker="MSFT", direction="LONG",
            trade_type="SWING", entry_price=200.0, shares=5,
            stop_loss=196.0, target=206.0, status="OPEN",
            entry_time=datetime(2025, 1, 15, tzinfo=timezone.utc),
        ))
        # Save a snapshot with cash
        session.add(AccountSnapshot(
            date=datetime(2025, 1, 15, tzinfo=timezone.utc),
            cash=1500.0, positions_value=1000.0, total_value=2500.0,
            open_positions=1, peak_value=2500.0, drawdown_pct=0.0,
        ))
        session.commit()
        session.close()

        trader = restore_trader_state()
        assert trader.account.open_position_count == 1
        pos = list(trader.account.positions.values())[0]
        assert pos.ticker == "MSFT"
        assert pos.shares == 5
        # Cash should be snapshot cash minus position cost
        assert trader.account.cash == 1500.0 - (200.0 * 5)

    def test_restore_ignores_closed_trades(self, in_memory_db):
        session = get_session()
        session.add(TradeRecord(
            trade_id="closed-001", ticker="OLD", direction="LONG",
            trade_type="DAY", entry_price=100.0, shares=10,
            stop_loss=98.0, target=103.0, status="CLOSED",
            entry_time=datetime(2025, 1, 10, tzinfo=timezone.utc),
        ))
        session.commit()
        session.close()

        trader = restore_trader_state()
        assert trader.account.open_position_count == 0


# ════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_stop_scheduler_when_not_running(self):
        stop_scheduler()  # Should not raise

    def test_fetch_price_handles_errors(self):
        with patch("modules.scheduler.yf.Ticker", side_effect=Exception("Network")):
            price = _fetch_current_price("FAIL")
            assert price is None

    def test_status_with_open_positions(self):
        trader = PaperTrader()
        trader.open_position(
            ticker="X", entry_price=50.0, stop_loss=48.0,
            target=53.0, shares=5,
        )
        sched_module._trader = trader
        status = get_scheduler_status()
        assert len(status["open_positions"]) == 1
        assert status["open_positions"][0]["ticker"] == "X"

    def test_status_account_info(self):
        status = get_scheduler_status()
        assert "cash" in status["account"]
        assert "total_value" in status["account"]


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_scheduler.py -v
#
# Expected results:
#   TestSchedulerCreation:    4 tests
#   TestNightlyScan:          3 tests
#   TestSignalCheck:          4 tests
#   TestPositionMonitor:      4 tests
#   TestCloseDayTrades:       2 tests
#   TestAccountSnapshot:      1 test
#   TestStatePersistence:     5 tests
#   TestEdgeCases:            4 tests
#
# TOTAL: 27 tests
# ════════════════════════════════════════════════════════════
