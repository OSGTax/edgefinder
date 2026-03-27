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
            signal_check_fn=lambda: None,
            nightly_scan_fn=lambda: None,
        )
        status = s.get_status()
        assert "signal_check" in status["jobs"]
        assert "nightly_scan" in status["jobs"]

    def test_start_and_stop(self):
        s = EdgeFinderScheduler()
        s.setup(signal_check_fn=lambda: None)
        s.start()
        assert s.running is True
        s.stop()
        assert s.running is False

    def test_get_status(self):
        s = EdgeFinderScheduler()
        s.setup(
            signal_check_fn=lambda: None,
            position_monitor_fn=lambda: None,
            benchmark_collect_fn=lambda: None,
        )
        s.start()
        status = s.get_status()
        assert status["running"] is True
        assert len(status["next_runs"]) == 3
        s.stop()
