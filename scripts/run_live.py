"""
EdgeFinder Live Runner
=======================
Run the scheduler locally without the dashboard.
Useful for development, testing, and headless operation.

Usage: python scripts/run_live.py
Stop:  Ctrl+C
"""

import logging
import os
import signal
import sys
import threading

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from modules.database import init_db
from modules.scheduler import start_scheduler, stop_scheduler, get_scheduler_status


def main():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
    )
    logger = logging.getLogger("run_live")

    logger.info("=" * 60)
    logger.info("  EDGEFINDER LIVE MODE")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Initialize database
    init_db()

    # Start scheduler
    start_scheduler()

    # Block until interrupted
    stop_event = threading.Event()

    def handle_shutdown(signum, frame):
        logger.info("Shutting down...")
        stop_scheduler()
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    stop_event.wait()
    logger.info("EdgeFinder stopped.")


if __name__ == "__main__":
    main()
