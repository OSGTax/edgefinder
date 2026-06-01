"""Admin API — externally-triggered maintenance jobs.

The in-process APScheduler can only fire jobs while the web service is awake,
and on Render the service idles after the market-morning traffic dies down —
so the post-close jobs (nightly scan, daily-indicator cycle, EOD snapshot,
benchmark/rotation/corporate actions) silently never run. An external
scheduler (GitHub Actions cron) hits POST /api/admin/run-eod after the close;
the request both wakes the idle service and runs the jobs in-process, which
keeps the live arena's in-memory watchlists/fundamentals coherent.

Guarded by a shared secret (settings.eod_trigger_token). Empty token ⇒
endpoint disabled (503), so it's inert until deliberately configured.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Header, HTTPException

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _authorize(authorization: str | None) -> None:
    """Validate the Bearer token against settings.eod_trigger_token."""
    token = settings.eod_trigger_token
    if not token:
        raise HTTPException(status_code=503, detail="EOD trigger not configured")
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
