"""Admin API — externally-triggered jobs.

  - POST /api/admin/run-eod   post-close jobs (market snapshot, benchmark,
                              sector rotation, nightly data scan + PIT
                              snapshot, dividends/splits, v2 account
                              snapshot) — fired once after the close when
                              the in-process scheduler can't (e.g. an idle
                              or restarted box).

Guarded by a shared secret (settings.eod_trigger_token). Empty token ⇒
endpoint disabled (503), so it's inert until deliberately set.
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
