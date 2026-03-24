"""
EdgeFinder Scheduler: Arena Multi-Strategy Operation
=====================================================
Runs the arena trading pipeline on a schedule using APScheduler:

    7:00 AM - 6:00 PM ET (Mon-Fri):
        Every 15 min: Arena signal check (all strategies)
        Every  5 min: Monitor open positions (stops, targets, trailing)
    3:50 PM ET: Close all day trades
    4:05 PM ET: Save daily strategy snapshots
    4:30 PM ET: Run nightly fundamental scan + refresh watchlists

Integrates with the FastAPI dashboard via lifespan hook.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import pytz
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from modules.scanner import get_active_watchlist

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# ── MODULE STATE ─────────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None


# ── SCHEDULER LIFECYCLE ──────────────────────────────────────

def create_scheduler() -> BackgroundScheduler:
    """Create and configure the APScheduler with arena trading jobs."""
    executors = {"default": ThreadPoolExecutor(1)}
    scheduler = BackgroundScheduler(timezone=ET, executors=executors)

    from modules.arena.live import (
        arena_signal_check,
        arena_position_monitor,
        arena_close_day_trades,
        arena_snapshot,
        arena_nightly_scan,
    )

    # Signal check: every 15 min, 7AM-6PM ET, Mon-Fri
    scheduler.add_job(
        arena_signal_check,
        CronTrigger(minute="*/15", hour="7-17", day_of_week="mon-fri", timezone=ET),
        id="arena_signal_check",
        replace_existing=True,
    )

    # Position monitor: every 5 min, 7AM-6PM ET, Mon-Fri
    scheduler.add_job(
        arena_position_monitor,
        CronTrigger(
            minute=f"*/{settings.POSITION_MONITOR_INTERVAL_MINUTES}",
            hour="7-17",
            day_of_week="mon-fri",
            timezone=ET,
        ),
        id="arena_position_monitor",
        replace_existing=True,
    )

    # Close day trades: 3:50 PM ET, Mon-Fri
    scheduler.add_job(
        arena_close_day_trades,
        CronTrigger(hour=15, minute=50, day_of_week="mon-fri", timezone=ET),
        id="arena_close_day_trades",
        replace_existing=True,
    )

    # Strategy snapshots: 4:05 PM ET, Mon-Fri
    scheduler.add_job(
        arena_snapshot,
        CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone=ET),
        id="arena_snapshot",
        replace_existing=True,
    )

    # Nightly scan + watchlist refresh: 4:30 PM ET, Mon-Fri
    scheduler.add_job(
        arena_nightly_scan,
        CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone=ET),
        id="arena_nightly_scan",
        replace_existing=True,
    )

    return scheduler


def start_scheduler() -> None:
    """Start the scheduler. Safe to call multiple times (idempotent)."""
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.info("Scheduler already running")
        return

    # Check if watchlist is empty and run initial scan if needed
    watchlist = get_active_watchlist()
    if not watchlist:
        logger.info("No watchlist data — running initial scan in background...")
        thread = threading.Thread(target=_initial_scan, daemon=True)
        thread.start()

    # Initialize arena engine
    from modules.arena.live import init_arena
    init_arena()

    # Create and start scheduler
    _scheduler = create_scheduler()
    _scheduler.start()

    logger.info("=" * 60)
    logger.info("EDGEFINDER ARENA SCHEDULER STARTED")
    logger.info(f"Timezone: US/Eastern")
    logger.info(f"Signal check: every {settings.SIGNAL_CHECK_INTERVAL_MINUTES}m, 7AM-6PM")
    logger.info(f"Position monitor: every {settings.POSITION_MONITOR_INTERVAL_MINUTES}m")
    logger.info(f"Nightly scan: {settings.SCANNER_RUN_TIME} ET")
    for job in _scheduler.get_jobs():
        logger.info(f"  Job: {job.id} → next run: {job.next_run_time}")
    logger.info("=" * 60)


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def get_scheduler_status() -> dict:
    """Return current scheduler state for API."""
    from modules.arena.live import get_arena_status

    jobs = []
    if _scheduler and _scheduler.running:
        for job in _scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })

    arena = get_arena_status()
    arena["scheduler_running"] = _scheduler.running if _scheduler else False
    arena["jobs"] = jobs
    return arena


# ── HELPERS ──────────────────────────────────────────────────

def _initial_scan() -> None:
    """Run initial scan on first boot when watchlist is empty.

    Uses the small default ticker list instead of full universe to avoid
    burning the entire FMP daily budget on startup. The nightly scan at
    4:30 PM will do proper sector rotation with the full universe.
    """
    from modules.scanner import run_scan
    try:
        results = run_scan(tickers=settings.SCANNER_DEFAULT_TICKERS[:40], save_to_db=True)
        logger.info(f"Initial scan complete: {len(results)} stocks on watchlist")

        # Refresh arena watchlists after initial scan
        from modules.arena.live import _refresh_watchlists
        _refresh_watchlists()
    except Exception as e:
        logger.error(f"Initial scan failed: {e}")
