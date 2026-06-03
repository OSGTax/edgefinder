"""Admin API — externally-triggered jobs.

The in-process APScheduler can only fire jobs while the web service is awake,
and on Render the service idles after the market-morning traffic dies down —
so scheduled jobs silently never run. External GitHub Actions crons drive them
instead by POSTing protected endpoints; each request both wakes the idle
service and runs the work in-process against the live arena:

  - POST /api/admin/run-eod        post-close jobs (nightly scan, daily-indicator
                                   cycle, EOD snapshot, benchmark/rotation/corp
                                   actions) — fired once after the close.
  - POST /api/admin/run-intraday   one intraday cycle (entries + exits + state
                                   persist + equity snapshot) — fired every
                                   ~5 min during market hours. This is the
                                   single driver of the live trading loop when
                                   settings.intraday_external_driver is on (the
                                   in-process intraday timer is then disabled).

Both are guarded by the same shared secret (settings.eod_trigger_token). Empty
token ⇒ endpoints disabled (503), so they're inert until deliberately set.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Header, HTTPException

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _authorize(authorization: str | None) -> None:
    """Validate the Bearer token against settings.eod_trigger_token.

    The same shared token guards every admin trigger endpoint (EOD +
    intraday) — they're the same trust class (an external cron waking the
    box to run privileged in-process jobs), so one secret covers both.
    """
    token = settings.eod_trigger_token
    if not token:
        raise HTTPException(status_code=503, detail="Admin trigger not configured")
    expected = f"Bearer {token}"
    # Constant-time-ish compare; tokens are short shared secrets, not passwords.
    if not authorization or authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/run-eod", status_code=202)
def run_eod(authorization: str | None = Header(default=None)):
    """Kick off the post-close jobs in a background thread.

    Returns 202 immediately so the caller (and Render's port probe) never
    blocks on the scan, which can take tens of seconds.
    """
    _authorize(authorization)

    # Imported lazily so the module imports cleanly even before services init.
    from dashboard.services import run_eod_jobs

    threading.Thread(
        target=run_eod_jobs, name="eod-jobs", daemon=True
    ).start()
    logger.info("EOD jobs triggered via /api/admin/run-eod")
    return {"status": "started"}


@router.post("/run-intraday", status_code=202)
def run_intraday(authorization: str | None = Header(default=None)):
    """Run one intraday cycle (entries + exits + state persist) in a thread.

    Driven by the intraday-cycle GitHub Actions cron, which both wakes the
    idle Render service and fires exactly one cycle every ~5 min during
    market hours. Returns 202 immediately so the caller never blocks on the
    cycle. A single-flight lock inside run_intraday_jobs guards against an
    overlapping/slow fire double-running on the shared in-memory arena.
    """
    _authorize(authorization)

    # Imported lazily so the module imports cleanly even before services init.
    from dashboard.services import run_intraday_jobs

    threading.Thread(
        target=run_intraday_jobs, name="intraday-cycle", daemon=True
    ).start()
    logger.info("Intraday cycle triggered via /api/admin/run-intraday")
    return {"status": "started"}
