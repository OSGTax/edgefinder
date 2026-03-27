"""EdgeFinder v2 — Filesystem-based cache for market data.

Bars stored as Parquet (fast, compact). Fundamentals and universe as JSON.
TTL is checked via file modification time — no metadata DB needed.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import settings
from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)


class DataCache:
    """Filesystem-based cache with configurable TTLs."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = Path(cache_dir or settings.cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Bars ─────────────────────────────────────────

    def get_bars(
        self, ticker: str, timeframe: str, start: date, end: date | None = None
    ) -> pd.DataFrame | None:
        path = self._bars_path(ticker, timeframe)
        if not path.exists():
            return None
        if self._is_expired(path, self._bars_ttl(timeframe)):
            return None
        try:
            df = pd.read_parquet(path)
            if df.index.name == "timestamp" and start:
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()
                df = df.loc[start_ts:end_ts]
            return df
        except Exception:
            logger.warning("Failed to read cached bars for %s/%s", ticker, timeframe)
            return None

    def store_bars(self, ticker: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._bars_path(ticker, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path)
        except Exception:
            logger.warning("Failed to cache bars for %s/%s", ticker, timeframe)

    # ── Fundamentals ─────────────────────────────────

    def get_fundamentals(self, ticker: str) -> TickerFundamentals | None:
        path = self._dir / "fundamentals" / f"{ticker}.json"
        if not path.exists():
            return None
        if self._is_expired(path, timedelta(hours=settings.cache_fundamentals_ttl_hours)):
            return None
        try:
            data = json.loads(path.read_text())
            return TickerFundamentals(**data)
        except Exception:
            logger.warning("Failed to read cached fundamentals for %s", ticker)
            return None

    def store_fundamentals(self, ticker: str, data: TickerFundamentals) -> None:
        path = self._dir / "fundamentals" / f"{ticker}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(data.model_dump_json(indent=2))
        except Exception:
            logger.warning("Failed to cache fundamentals for %s", ticker)

    # ── Universe ─────────────────────────────────────

    def get_universe(self) -> list[str] | None:
        path = self._dir / "universe" / "tickers.json"
        if not path.exists():
            return None
        if self._is_expired(path, timedelta(hours=12)):
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.warning("Failed to read cached universe")
            return None

    def store_universe(self, tickers: list[str]) -> None:
        path = self._dir / "universe" / "tickers.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(tickers))
        except Exception:
            logger.warning("Failed to cache universe")

    # ── Utilities ────────────────────────────────────

    def clear(self) -> None:
        """Remove all cached files."""
        if self._dir.exists():
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)

    # ── Private ──────────────────────────────────────

    def _bars_path(self, ticker: str, timeframe: str) -> Path:
        return self._dir / "bars" / ticker / f"{timeframe}.parquet"

    def _bars_ttl(self, timeframe: str) -> timedelta:
        minutes = settings.cache_bars_ttl_minutes.get(timeframe, 60)
        return timedelta(minutes=minutes)

    @staticmethod
    def _is_expired(path: Path, ttl: timedelta) -> bool:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime > ttl
