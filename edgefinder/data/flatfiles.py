"""Massive (Polygon) flat-files reader — bulk historical aggregates over S3.

Reads the gzipped CSV dumps from the flat-files bucket
(``files.massive.com`` / ``flatfiles``), e.g.::

    us_stocks_sip/day_aggs_v1/2026/05/2026-05-26.csv.gz
    us_stocks_sip/minute_aggs_v1/2026/05/2026-05-26.csv.gz

Each aggregate CSV shares the columns
``ticker,volume,open,close,high,low,window_start,transactions`` where
``window_start`` is a nanosecond UTC epoch.

Used for daily-bar backfill and minute-bar backtesting. NOT for live
intraday data — these files are published T+1.
"""

from __future__ import annotations

import gzip
import io
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# dataset name -> path segment under the stocks-SIP root
DATASETS = {
    "day_aggs": "day_aggs_v1",
    "minute_aggs": "minute_aggs_v1",
    "trades": "trades_v1",
    "quotes": "quotes_v1",
}


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """Add tz-aware ``timestamp`` and python ``date`` from ns ``window_start``."""
    if df.empty:
        return df
    ts = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    df = df.copy()
    df["ticker"] = df["ticker"].astype("string")
    df["timestamp"] = ts
    df["date"] = ts.dt.date
    return df


def parse_aggs_csv(raw: bytes | str) -> pd.DataFrame:
    """Parse aggregate CSV bytes (gzipped or plain) into a typed DataFrame.

    Pure function — no network — so it is unit-testable in isolation.
    """
    if isinstance(raw, (bytes, bytearray)):
        head = bytes(raw[:2])
        data = gzip.decompress(raw) if head == b"\x1f\x8b" else bytes(raw)
        buf: io.IOBase = io.BytesIO(data)
    else:
        buf = io.StringIO(raw)
    df = pd.read_csv(buf)
    return _augment(df)


def day_aggs_to_rows(df: pd.DataFrame, source: str = "flatfiles") -> list[dict]:
    """Convert a parsed day_aggs DataFrame into daily_bars row dicts."""
    rows: list[dict] = []
    for r in df.itertuples(index=False):
        tx = getattr(r, "transactions", None)
        rows.append(
            {
                "symbol": str(r.ticker),
                "date": r.date,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
                "transactions": int(tx) if pd.notna(tx) else None,
                "source": source,
            }
        )
    return rows


class FlatFilesClient:
    """Thin S3 client for the Massive/Polygon flat-files bucket.

    boto3 is imported lazily so the dependency is only required when flat
    files are actually accessed. Downloads are cached under
    ``settings.cache_dir/flatfiles`` keyed by their S3 path.
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
        stocks_prefix: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        endpoint = (endpoint_url or settings.polygon_s3_endpoint_url or "").strip()
        if endpoint and not endpoint.startswith("http"):
            endpoint = "https://" + endpoint
        self._endpoint = endpoint
        self._access_key_id = access_key_id or settings.polygon_s3_access_key_id
        self._secret_access_key = secret_access_key or settings.polygon_s3_secret_access_key
        self._bucket = bucket or settings.polygon_s3_bucket
        self._stocks_prefix = stocks_prefix or settings.flatfiles_stocks_prefix
        self._cache_dir = Path(cache_dir) if cache_dir else (settings.cache_dir / "flatfiles")
        self._client = None

    def _s3(self):
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as e:  # pragma: no cover - environment guard
                raise RuntimeError(
                    "boto3 is required for flat-files access: pip install boto3"
                ) from e
            if not (self._access_key_id and self._secret_access_key):
                raise RuntimeError(
                    "Flat-files credentials missing: set "
                    "EDGEFINDER_POLYGON_S3_ACCESS_KEY_ID and "
                    "EDGEFINDER_POLYGON_S3_SECRET_ACCESS_KEY"
                )
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3},
                ),
            )
        return self._client

    def key_for(self, dataset: str, day: date) -> str:
        segment = DATASETS[dataset]
        return f"{self._stocks_prefix}/{segment}/{day:%Y/%m/%Y-%m-%d}.csv.gz"

    def list_days(self, dataset: str, year: int, month: int) -> list[date]:
        """List the trading days available for a dataset in a given month."""
        segment = DATASETS[dataset]
        prefix = f"{self._stocks_prefix}/{segment}/{year:04d}/{month:02d}/"
        days: list[date] = []
        token = None
        while True:
            kwargs = {"Bucket": self._bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._s3().list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.endswith(".csv.gz"):
                    try:
                        days.append(datetime.strptime(name[:-7], "%Y-%m-%d").date())
                    except ValueError:
                        continue
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return sorted(days)

    def available_days(self, dataset: str, start: date, end: date) -> list[date]:
        """List available trading days for a dataset within [start, end]."""
        days: list[date] = []
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            for d in self.list_days(dataset, year, month):
                if start <= d <= end:
                    days.append(d)
            month += 1
            if month > 12:
                year, month = year + 1, 1
        return sorted(days)

    def download(self, dataset: str, day: date, *, use_cache: bool = True) -> bytes:
        """Return the raw gzipped bytes for a dataset/day, caching to disk."""
        key = self.key_for(dataset, day)
        cache_path = self._cache_dir / key
        if use_cache and cache_path.exists():
            return cache_path.read_bytes()
        obj = self._s3().get_object(Bucket=self._bucket, Key=key)
        raw = obj["Body"].read()
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw)
        return raw

    def read_aggs(
        self,
        dataset: str,
        day: date,
        *,
        symbols: list[str] | None = None,
        use_cache: bool = True,
        chunksize: int = 500_000,
    ) -> pd.DataFrame:
        """Read an aggregate file into a DataFrame, optionally symbol-filtered.

        When ``symbols`` is given the CSV is streamed in chunks and filtered
        before concatenation, so a few tickers can be pulled from a
        full-market minute file without materializing the whole thing.
        """
        raw = self.download(dataset, day, use_cache=use_cache)
        data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        if symbols is None:
            return _augment(pd.read_csv(io.BytesIO(data)))

        wanted = {s.upper() for s in symbols}
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(io.BytesIO(data), chunksize=chunksize):
            parts.append(chunk[chunk["ticker"].str.upper().isin(wanted)])
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return _augment(df)

    def read_day_aggs(
        self, day: date, *, symbols: list[str] | None = None, use_cache: bool = True
    ) -> pd.DataFrame:
        return self.read_aggs("day_aggs", day, symbols=symbols, use_cache=use_cache)

    def read_minute_aggs(
        self, day: date, *, symbols: list[str] | None = None, use_cache: bool = True
    ) -> pd.DataFrame:
        return self.read_aggs("minute_aggs", day, symbols=symbols, use_cache=use_cache)
