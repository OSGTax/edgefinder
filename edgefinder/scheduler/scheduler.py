"""EdgeFinder v2 — APScheduler job scheduling.

All scheduled jobs run in ET wall-clock time:
- V2 portfolio rebalance: 9:45 AM ET
- V2 account snapshots: every 30 min, 9:45 AM - 4:15 PM ET
- Market snapshot: 4:05 PM ET
- Benchmark collection: 4:10 PM ET
- Sector rotation: 4:15 PM ET
- Nightly data scan: 6:15 PM ET (settings.scanner_run_time)
- Dividends & splits: 6:30 PM ET
- R2 bar-store sync: 7:00 PM ET
- News accumulator: hourly :30 from 9:30 AM to 4:30 PM ET

Render containers (and most prod environments) run as UTC. APScheduler 3.x
CronTrigger ignores the scheduler's default tz when none is passed
explicitly — it falls back to ``tzlocal.get_localzone()``, which on a UTC
container makes ``CronTrigger(hour=18, ...)`` fire at 18:00 UTC, four hours
earlier than 18:00 ET. We pin the ZoneInfo here and pass it to both the
scheduler and every trigger so jobs fire on the intended ET wall clock.
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class EdgeFinderScheduler:
    """Manages all scheduled jobs."""

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone=ET)
        self._jobs: dict[str, str] = {}

    def setup(
        self,
        portfolio_rebalance_fn=None,
        v2_snapshot_fn=None,
        nightly_scan_fn=None,
        benchmark_collect_fn=None,
        market_snapshot_fn=None,
        sector_rotation_fn=None,
        news_accumulate_fn=None,
        dividend_split_fn=None,
        r2_sync_fn=None,
    ) -> None:
        """Register all scheduled jobs.

        Pass None for any function to skip that job.
        """
        if portfolio_rebalance_fn:
            # Shortly after the open: yesterday's flat-file bars are long
            # since published, and fills land near the open the v2 engine
            # models (decide on data through yesterday, fill at today's open).
            self._scheduler.add_job(
                portfolio_rebalance_fn,
                CronTrigger(hour=9, minute=45, day_of_week="mon-fri", timezone=ET),
                id="v2_portfolio_rebalance",
                name="V2 Portfolio Rebalance",
                replace_existing=True,
            )
            self._jobs["v2_portfolio_rebalance"] = "Daily at 9:45 AM ET"

        if v2_snapshot_fn:
            # Cron grid is :15/:45 across 9-16 ET; the wrapper clips it to the
            # 09:45-16:15 window (CronTrigger can't express the cross-hour
            # bounds directly).
            def _v2_snapshot_windowed():
                now_et = datetime.now(ET)
                if not (dtime(9, 45) <= now_et.time() <= dtime(16, 15)):
                    return
                v2_snapshot_fn()

            self._scheduler.add_job(
                _v2_snapshot_windowed,
                CronTrigger(minute="15,45", hour="9-16",
                            day_of_week="mon-fri", timezone=ET),
                id="v2_snapshot",
                name="V2 Account Snapshots",
                replace_existing=True,
            )
            self._jobs["v2_snapshot"] = "Every 30m, 9:45 AM - 4:15 PM ET"

        if nightly_scan_fn:
            scan_hour, scan_min = settings.scanner_run_time.split(":")
            self._scheduler.add_job(
                nightly_scan_fn,
                CronTrigger(
                    hour=int(scan_hour),
                    minute=int(scan_min),
                    day_of_week="mon-fri",
                    timezone=ET,
                ),
                id="nightly_scan",
                name="Nightly Data Scan",
                replace_existing=True,
            )
            self._jobs["nightly_scan"] = f"Daily at {settings.scanner_run_time} ET"

        if benchmark_collect_fn:
            self._scheduler.add_job(
                benchmark_collect_fn,
                CronTrigger(hour=16, minute=10, day_of_week="mon-fri", timezone=ET),
                id="benchmark_collect",
                name="Benchmark Collection",
                replace_existing=True,
            )
            self._jobs["benchmark_collect"] = "Daily at 4:10 PM ET"

        if market_snapshot_fn:
            self._scheduler.add_job(
                market_snapshot_fn,
                CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone=ET),
                id="market_snapshot",
                name="Daily Market Snapshot",
                replace_existing=True,
            )
            self._jobs["market_snapshot"] = "Daily at 4:05 PM ET"

        if sector_rotation_fn:
            self._scheduler.add_job(
                sector_rotation_fn,
                CronTrigger(hour=16, minute=15, day_of_week="mon-fri", timezone=ET),
                id="sector_rotation",
                name="Sector Rotation (RRG)",
                replace_existing=True,
            )
            self._jobs["sector_rotation"] = "Daily at 4:15 PM ET"

        if news_accumulate_fn:
            self._scheduler.add_job(
                news_accumulate_fn,
                CronTrigger(minute="30", hour="9-16", day_of_week="mon-fri", timezone=ET),
                id="news_accumulate",
                name="News Accumulation",
                replace_existing=True,
            )
            self._jobs["news_accumulate"] = "Hourly 9:30 AM - 4:30 PM ET"

        if dividend_split_fn:
            self._scheduler.add_job(
                dividend_split_fn,
                CronTrigger(hour=18, minute=30, day_of_week="mon-fri", timezone=ET),
                id="dividend_split",
                name="Dividends & Splits",
                replace_existing=True,
            )
            self._jobs["dividend_split"] = "Daily at 6:30 PM ET"

        if r2_sync_fn:
            # After the nightly scan + PIT snapshot: mirror the day's new
            # daily_bars rows to the R2 Parquet store (incremental).
            self._scheduler.add_job(
                r2_sync_fn,
                CronTrigger(hour=19, minute=0, day_of_week="mon-fri", timezone=ET),
                id="r2_sync",
                name="R2 Bar-Store Sync",
                replace_existing=True,
            )
            self._jobs["r2_sync"] = "Daily at 7:00 PM ET"

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
