"""EdgeFinder v2 — APScheduler job scheduling.

All scheduled jobs run in ET timezone:
- Signal check: every 15 min during market hours
- Position monitor: every 5 min during market hours
- Nightly scan: 6:15 PM ET
- Daily benchmark collection: 4:10 PM ET
- Daily strategy snapshots: 4:05 PM ET
"""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings

logger = logging.getLogger(__name__)


class EdgeFinderScheduler:
    """Manages all scheduled jobs."""

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone="US/Eastern")
        self._jobs: dict[str, str] = {}

    def setup(
        self,
        signal_check_fn=None,
        position_monitor_fn=None,
        nightly_scan_fn=None,
        benchmark_collect_fn=None,
        snapshot_fn=None,
        sector_rotation_fn=None,
        news_accumulate_fn=None,
        dividend_split_fn=None,
    ) -> None:
        """Register all scheduled jobs.

        Pass None for any function to skip that job.
        """
        open_h = settings.market_open_et.split(":")[0]
        close_h = settings.market_close_et.split(":")[0]
        market_hours = f"{open_h}-{close_h}"

        if signal_check_fn:
            self._scheduler.add_job(
                signal_check_fn,
                CronTrigger(
                    minute=f"*/{settings.signal_check_interval_minutes}",
                    hour=market_hours,
                    day_of_week="mon-fri",
                ),
                id="signal_check",
                name="Signal Check",
                replace_existing=True,
            )
            self._jobs["signal_check"] = f"Every {settings.signal_check_interval_minutes}m"

        if position_monitor_fn:
            self._scheduler.add_job(
                position_monitor_fn,
                CronTrigger(
                    minute=f"*/{settings.position_monitor_interval_minutes}",
                    hour=market_hours,
                    day_of_week="mon-fri",
                ),
                id="position_monitor",
                name="Position Monitor",
                replace_existing=True,
            )
            self._jobs["position_monitor"] = f"Every {settings.position_monitor_interval_minutes}m"

        if nightly_scan_fn:
            scan_hour, scan_min = settings.scanner_run_time.split(":")
            self._scheduler.add_job(
                nightly_scan_fn,
                CronTrigger(hour=int(scan_hour), minute=int(scan_min), day_of_week="mon-fri"),
                id="nightly_scan",
                name="Nightly Scan",
                replace_existing=True,
            )
            self._jobs["nightly_scan"] = f"Daily at {settings.scanner_run_time} ET"

        if benchmark_collect_fn:
            self._scheduler.add_job(
                benchmark_collect_fn,
                CronTrigger(hour=16, minute=10, day_of_week="mon-fri"),
                id="benchmark_collect",
                name="Benchmark Collection",
                replace_existing=True,
            )
            self._jobs["benchmark_collect"] = "Daily at 4:10 PM ET"

        if snapshot_fn:
            self._scheduler.add_job(
                snapshot_fn,
                CronTrigger(hour=16, minute=5, day_of_week="mon-fri"),
                id="daily_snapshot",
                name="Daily Strategy Snapshots",
                replace_existing=True,
            )
            self._jobs["daily_snapshot"] = "Daily at 4:05 PM ET"

        if sector_rotation_fn:
            self._scheduler.add_job(
                sector_rotation_fn,
                CronTrigger(hour=16, minute=15, day_of_week="mon-fri"),
                id="sector_rotation",
                name="Sector Rotation (RRG)",
                replace_existing=True,
            )
            self._jobs["sector_rotation"] = "Daily at 4:15 PM ET"

        if news_accumulate_fn:
            self._scheduler.add_job(
                news_accumulate_fn,
                CronTrigger(minute="30", hour="9-16", day_of_week="mon-fri"),
                id="news_accumulate",
                name="News Accumulation",
                replace_existing=True,
            )
            self._jobs["news_accumulate"] = "Hourly 9:30 AM - 4:30 PM ET"

        if dividend_split_fn:
            self._scheduler.add_job(
                dividend_split_fn,
                CronTrigger(hour=18, minute=30, day_of_week="mon-fri"),
                id="dividend_split",
                name="Dividends & Splits",
                replace_existing=True,
            )
            self._jobs["dividend_split"] = "Daily at 6:30 PM ET"

        logger.info("Scheduler configured with %d jobs", len(self._jobs))

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("Scheduler started")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    def get_status(self) -> dict:
        next_runs = {}
        for job in self._scheduler.get_jobs():
            nrt = getattr(job, "next_run_time", None)
            next_runs[job.id] = nrt.isoformat() if nrt else None
        return {
            "running": self._scheduler.running,
            "jobs": self._jobs,
            "next_runs": next_runs,
        }

    @property
    def running(self) -> bool:
        return self._scheduler.running
