"""
EdgeFinder Arena Scheduler Tests
==================================
Tests cover: scheduler creation with arena jobs, lifecycle management.

Run: python -m pytest tests/test_scheduler.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from modules.scheduler import (
    create_scheduler,
    start_scheduler,
    stop_scheduler,
    get_scheduler_status,
)
from modules.database import init_db, reset_engine


@pytest.fixture(autouse=True)
def in_memory_db():
    """Use in-memory database for all scheduler tests."""
    reset_engine()
    init_db(":memory:")
    yield
    reset_engine()


class TestSchedulerCreation:

    def test_scheduler_has_5_arena_jobs(self):
        scheduler = create_scheduler()
        jobs = scheduler.get_jobs()
        assert len(jobs) == 5
        job_ids = {j.id for j in jobs}
        assert "arena_signal_check" in job_ids
        assert "arena_position_monitor" in job_ids
        assert "arena_close_day_trades" in job_ids
        assert "arena_snapshot" in job_ids
        assert "arena_nightly_scan" in job_ids

    def test_no_v1_jobs(self):
        scheduler = create_scheduler()
        job_ids = {j.id for j in scheduler.get_jobs()}
        assert "signal_check" not in job_ids
        assert "position_monitor" not in job_ids
        assert "close_day_trades" not in job_ids
        assert "account_snapshot" not in job_ids
        assert "nightly_scan" not in job_ids

    @patch("modules.scheduler.get_active_watchlist", return_value=[{"ticker": "AAPL"}])
    @patch("modules.arena.live.init_arena")
    def test_start_scheduler_is_idempotent(self, mock_init, mock_wl):
        start_scheduler()
        start_scheduler()  # Should not fail
        stop_scheduler()

    @patch("modules.scheduler.get_active_watchlist", return_value=[{"ticker": "AAPL"}])
    @patch("modules.arena.live.init_arena")
    def test_stop_scheduler(self, mock_init, mock_wl):
        start_scheduler()
        stop_scheduler()

    @patch("modules.arena.live.get_arena_status", return_value={"running": True, "strategies": 2})
    def test_get_scheduler_status_returns_arena_data(self, mock_status):
        status = get_scheduler_status()
        assert "strategies" in status
        assert status["strategies"] == 2

    @patch("modules.scheduler.get_active_watchlist", return_value=[])
    @patch("modules.arena.live.init_arena")
    def test_empty_watchlist_triggers_initial_scan(self, mock_init, mock_wl):
        """When watchlist is empty, scheduler runs initial scan in background."""
        with patch("modules.scheduler.threading") as mock_threading:
            start_scheduler()
            mock_threading.Thread.assert_called_once()
            stop_scheduler()
