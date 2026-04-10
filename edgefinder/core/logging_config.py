"""EdgeFinder v2 — Centralized logging configuration.

All log records are emitted with timestamps in US/Eastern time,
matching the timezone used by the scheduler and settings.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("US/Eastern")


class ETFormatter(logging.Formatter):
    """Formatter that renders all timestamps in US/Eastern timezone."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=_ET)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install the ET formatter on the root logger and uvicorn loggers.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    formatter = ETFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Uvicorn installs its own handlers on startup — redirect them to
    # the root logger so access logs also use the ET formatter.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True

    _CONFIGURED = True
