"""Tests for edgefinder/scheduler/scheduler.py."""

from edgefinder.scheduler.scheduler import EdgeFinderScheduler


class TestScheduler:
    def test_create(self):
        s = EdgeFinderScheduler()
        assert s.running is False

    def test_setup_no_functions(self):
        s = EdgeFinderScheduler()
        s.setup()
        status = s.get_status()
        assert status["running"] is False
        assert len(status["jobs"]) == 0

    def test_setup_with_functions(self):
        s = EdgeFinderScheduler()
        s.setup(
            portfolio_rebalance_fn=lambda: None,
            nightly_scan_fn=lambda: None,
        )
        status = s.get_status()
        assert "v2_portfolio_rebalance" in status["jobs"]
        assert "nightly_scan" in status["jobs"]

    def test_setup_all_v2_jobs(self):
        s = EdgeFinderScheduler()
        s.setup(
            portfolio_rebalance_fn=lambda: None,
            v2_snapshot_fn=lambda: None,
            nightly_scan_fn=lambda: None,
            benchmark_collect_fn=lambda: None,
            market_snapshot_fn=lambda: None,
            sector_rotation_fn=lambda: None,
            news_accumulate_fn=lambda: None,
            dividend_split_fn=lambda: None,
            r2_sync_fn=lambda: None,
        )
        status = s.get_status()
        assert set(status["jobs"]) == {
            "v2_portfolio_rebalance", "v2_snapshot", "nightly_scan",
            "benchmark_collect", "market_snapshot", "sector_rotation",
            "news_accumulate", "dividend_split", "r2_sync",
        }

    def test_start_and_stop(self):
        s = EdgeFinderScheduler()
        s.setup(portfolio_rebalance_fn=lambda: None)
        s.start()
        assert s.running is True
        s.stop()
        assert s.running is False

    def test_get_status(self):
        s = EdgeFinderScheduler()
        s.setup(
            portfolio_rebalance_fn=lambda: None,
            v2_snapshot_fn=lambda: None,
            benchmark_collect_fn=lambda: None,
        )
        s.start()
        status = s.get_status()
        assert status["running"] is True
        assert len(status["next_runs"]) == 3
        s.stop()
