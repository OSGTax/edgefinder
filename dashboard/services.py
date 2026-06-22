"""Runtime services — greenfield rebuild (the old scheduler/jobs are retired).

The autonomous trading agent runs on Claude Code Routines, NOT an in-process
scheduler, so this module no longer starts APScheduler or any of the old
nightly/portfolio jobs. The dashboard is now a thin read surface over the
agent's desk_* tables (+ the kept market-data layer), reached via
``dashboard.dependencies.get_db`` and ``agent.data`` — neither needs a
process-wide provider singleton.

Kept as a no-op lifespan hook so ``dashboard.app`` stays unchanged in shape;
if a future job needs scheduling, the Routine is the home for it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def init_services() -> None:
    """No-op: the agent runs on Routines; no in-process scheduler/jobs."""
    logger.info("Services init — desk-only (no scheduler; agent runs on Routines)")


def shutdown_services() -> None:
    try:
        from edgefinder.db.engine import get_engine
        get_engine().dispose()
    except Exception:  # noqa: BLE001
        logger.debug("Engine dispose at shutdown failed", exc_info=True)
    logger.info("Services shut down")


# Back-compat stubs for any lingering import (none expected post-cutover).
def get_provider():
    return None


def get_scheduler():
    return None


def get_plan_access() -> dict[str, bool]:
    return {}
