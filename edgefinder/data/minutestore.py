"""Minute-bar storage for the INTRADAY pilot: Parquet per symbol-MONTH on R2.

The daily store (edgefinder/data/barstore.py) holds the permanent daily
asset; this module is its minute-resolution sibling for the FROZEN pilot
menu (intraday/menu.json — ~60 liquid names). Same R2 bucket, same auth,
same grow-only discipline:

Layout:
    minute/{SYMBOL}/{YYYY-MM}.parquet
        columns: ts (UTC epoch seconds, int64), open, high, low, close,
        volume — REGULAR-TRADING-HOURS bars only (bar-starts 09:30–15:59
        ET inclusive), sorted and deduped on ts.
    minute/_manifest.json
        {symbol: {"months": {"YYYY-MM": {"rows": n, "min_ts": i,
        "max_ts": j, "complete": bool}}}}
        ``complete`` is set by the backfill when its fetch window covered
        the WHOLE calendar month — only complete months are skipped on
        resume, so a partial nightly top-up can never mask a hole.

Sync semantics are merge-only, exactly like the daily store: a month
object is the UNION of what R2 already holds and the new frame; new rows
win on conflicting ts; the store NEVER shrinks.

Size budget: ~60 symbols × ~390 RTH bars/day × ~1,260 trading days
≈ 30M rows ≈ ~2 GB of Parquet — comfortably inside the R2 free tier.
A later FULL-MARKET expansion must NOT go through ``load_minute_bars``
(which materializes whole symbols in memory); that phase gets a
streaming reader.

Env (same secrets as the daily store, values .strip()ed):
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET.

CLI:
    python -m edgefinder.data.minutestore verify [--symbols A,B]
    python -m edgefinder.data.minutestore stats
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_COLS = ["ts", "open", "high", "low", "close", "volume"]
MINUTE_PREFIX = "minute/"
MANIFEST_KEY = "minute/_manifest.json"

# RTH bar-starts, minutes-past-midnight ET: 09:30 (570) .. 15:59 (959)
_RTH_OPEN_MIN = 9 * 60 + 30
_RTH_LAST_MIN = 15 * 60 + 59


def _month_key(symbol: str, month: str) -> str:
    return f"{MINUTE_PREFIX}{symbol}/{month}.parquet"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="int64" if c == "ts" else "float64")
                         for c in _COLS})


def to_et(ts: pd.Series) -> pd.Series:
    """UTC-epoch-seconds Series -> tz-aware America/New_York datetimes.

    The conversion helper for callers: minute frames carry only the int64
    ``ts`` column; anything session-aware (time-of-day features, day
    grouping) should derive from this, never from naive timestamps.
    """
    return pd.to_datetime(ts.astype("int64"), unit="s", utc=True).dt.tz_convert(ET)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-trading-hours bars (09:30–15:59 ET bar-starts).

    Polygon minute aggs include pre/post-market prints; the pilot store is
    RTH-only by contract so every reader sees the same 390-bar session.
    """
    if df is None or not len(df):
        return _empty_frame()
    et = to_et(df["ts"])
    minutes = et.dt.hour * 60 + et.dt.minute
    keep = (minutes >= _RTH_OPEN_MIN) & (minutes <= _RTH_LAST_MIN)
    return df.loc[keep].reset_index(drop=True)


def month_of(ts: pd.Series) -> pd.Series:
    """'YYYY-MM' bucket per bar, derived from the ET trading session."""
    return to_et(ts).dt.strftime("%Y-%m")


def months_between(start: date, end: date) -> list[str]:
    """All 'YYYY-MM' months touching [start, end], in order."""
    if end < start:
        return []
    out: list[str] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def month_bounds(month: str) -> tuple[date, date]:
    """('YYYY-MM') -> (first day, last day) of the calendar month."""
    from datetime import timedelta

    y, m = int(month[:4]), int(month[5:7])
    first = date(y, m, 1)
    next_first = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return first, next_first - timedelta(days=1)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, _COLS].copy()
    out["ts"] = out["ts"].astype("int64")
    for c in _COLS[1:]:
        out[c] = out[c].astype("float64")
    return (out.sort_values("ts")
               .drop_duplicates(subset="ts", keep="last")
               .reset_index(drop=True))


class MinuteStore:
    """Parquet-per-symbol-MONTH minute bars on any S3-compatible endpoint."""

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        try:
            self.bucket = os.environ["R2_BUCKET"].strip()
            self._s3 = boto3.client(
                "s3",
                endpoint_url=os.environ["R2_ENDPOINT"].strip(),
                aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"].strip(),
                aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"].strip(),
                config=Config(signature_version="s3v4",
                              retries={"max_attempts": 3},
                              max_pool_connections=16),
                region_name="auto",
            )
        except KeyError as e:
            raise RuntimeError(f"missing R2 secret: {e}") from None

    # ── primitives (mirror barstore) ─────────────────────

    def _put_frame(self, key: str, df: pd.DataFrame) -> int:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        body = buf.getvalue()
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=body)
        return len(body)

    def _get_frame(self, key: str) -> pd.DataFrame | None:
        from botocore.exceptions import ClientError

        try:
            body = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
        df = pd.read_parquet(io.BytesIO(body))
        df["ts"] = df["ts"].astype("int64")
        return df

    def read_manifest(self) -> dict:
        from botocore.exceptions import ClientError

        try:
            body = self._s3.get_object(
                Bucket=self.bucket, Key=MANIFEST_KEY)["Body"].read()
            return json.loads(body)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return {}
            raise

    def write_manifest(self, manifest: dict) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=MANIFEST_KEY,
                            Body=json.dumps(manifest).encode())

    def months(self, symbol: str, manifest: dict | None = None) -> dict:
        """{month: entry} for one symbol from the manifest."""
        manifest = self.read_manifest() if manifest is None else manifest
        return dict((manifest.get(symbol) or {}).get("months") or {})

    # ── public API ───────────────────────────────────────

    def sync_symbol_month(self, client_df: pd.DataFrame, symbol: str,
                          month: str, *, complete: bool | None = None,
                          manifest: dict | None = None) -> dict:
        """MERGE new bars into one symbol-month object. GROW-ONLY.

        ``client_df``: frame with the ``ts/open/high/low/close/volume``
        columns. Defensive contract enforcement happens here: non-RTH bars
        are dropped and bars outside ``month`` (ET) are dropped, so a
        careless caller cannot poison an object. The stored frame is the
        union of the existing object and the new rows; NEW rows win on ts
        conflicts (corrections land at the source); rows the new frame
        doesn't carry are PRESERVED — a sync can never shrink the store.

        ``complete=True`` records that the caller's fetch window covered the
        whole calendar month (the resumability latch — it never downgrades).
        An empty ``client_df`` never deletes anything: with no existing
        object it records a rows=0 manifest entry (no object uploaded) so
        a symbol's pre-listing months don't refetch forever.

        Pass ``manifest`` to batch updates across calls; the caller then
        owns the final :meth:`write_manifest`. Returns the manifest entry.
        """
        symbol = symbol.strip().upper()
        new = filter_rth(client_df)
        if len(new):
            new = new.loc[month_of(new["ts"]) == month]
        new = _normalize(new) if len(new) else _empty_frame()

        own_manifest = manifest is None
        manifest = self.read_manifest() if own_manifest else manifest
        sym_entry = manifest.setdefault(symbol, {}).setdefault("months", {})
        prev = sym_entry.get(month) or {}

        key = _month_key(symbol, month)
        if len(new):
            existing = self._get_frame(key)
            if existing is not None and len(existing):
                keep = existing[~existing["ts"].isin(set(new["ts"]))]
                merged = _normalize(pd.concat([keep, new], ignore_index=True))
            else:
                merged = new
            self._put_frame(key, merged)
            entry = {"rows": int(len(merged)),
                     "min_ts": int(merged["ts"].iloc[0]),
                     "max_ts": int(merged["ts"].iloc[-1])}
        elif prev:
            entry = dict(prev)        # empty input never shrinks anything
        else:
            entry = {"rows": 0, "min_ts": None, "max_ts": None}
        entry["complete"] = bool(prev.get("complete")) or bool(complete)

        sym_entry[month] = entry
        if own_manifest:
            self.write_manifest(manifest)
        return entry

    def load_minute_bars(self, symbols: list[str], start_date: date,
                         end_date: date,
                         max_workers: int = 8) -> dict[str, pd.DataFrame]:
        """Read RTH minute bars for ``symbols`` over [start_date, end_date].

        Reads every month object the range touches, concatenates, clips to
        the ET date range, and returns ``{symbol: DataFrame[ts, o, h, l, c,
        v]}`` sorted on ts. Symbols with no bars in range are omitted.
        Sized for the ~60-name pilot menu ONLY — see the module docstring.
        """
        from concurrent.futures import ThreadPoolExecutor

        months = months_between(start_date, end_date)

        def _one(sym: str) -> tuple[str, pd.DataFrame | None]:
            frames = [f for m in months
                      if (f := self._get_frame(_month_key(sym, m))) is not None
                      and len(f)]
            if not frames:
                return sym, None
            df = _normalize(pd.concat(frames, ignore_index=True))
            d = to_et(df["ts"]).dt.date
            df = df.loc[(d >= start_date) & (d <= end_date)]
            return sym, df.reset_index(drop=True)

        out: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for sym, df in pool.map(_one, [s.strip().upper() for s in symbols]):
                if df is not None and len(df):
                    out[sym] = df
        return out

    def verify(self, symbols: list[str] | None = None) -> dict:
        """Prove manifest == objects for the given (default: all) symbols.

        Checks both directions: every manifest month with rows must exist as
        an object with matching row count and min/max ts; every object under
        a symbol's prefix must be in the manifest. Returns
        ``{"ok", "checked", "mismatches": ["SYM 2024-01: reason", ...]}``.
        """
        manifest = self.read_manifest()
        if symbols is None:
            symbols = sorted(manifest)
        mismatches: list[str] = []
        checked = 0
        for sym in (s.strip().upper() for s in symbols):
            months = self.months(sym, manifest)
            object_months = self._list_object_months(sym)
            for month in sorted(object_months - set(months)):
                mismatches.append(f"{sym} {month}: object not in manifest")
            for month, entry in sorted(months.items()):
                checked += 1
                df = self._get_frame(_month_key(sym, month))
                if entry.get("rows", 0) == 0:
                    if df is not None and len(df):
                        mismatches.append(
                            f"{sym} {month}: manifest says empty, object has "
                            f"{len(df)} rows")
                    continue
                if df is None or not len(df):
                    mismatches.append(f"{sym} {month}: object missing")
                    continue
                got = (len(df), int(df['ts'].min()), int(df['ts'].max()))
                want = (entry.get("rows"), entry.get("min_ts"),
                        entry.get("max_ts"))
                if got != want:
                    mismatches.append(
                        f"{sym} {month}: rows/min/max {got} != manifest {want}")
        return {"ok": not mismatches, "checked": checked,
                "mismatches": mismatches}

    def _list_object_months(self, symbol: str) -> set[str]:
        months: set[str] = set()
        token = None
        prefix = f"{MINUTE_PREFIX}{symbol}/"
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.endswith(".parquet"):
                    months.add(name[:-len(".parquet")])
            if not resp.get("IsTruncated"):
                break
            token = resp["NextContinuationToken"]
        return months

    def stats(self) -> dict:
        n = total = 0
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": MINUTE_PREFIX}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                n += 1
                total += obj["Size"]
            if not resp.get("IsTruncated"):
                break
            token = resp["NextContinuationToken"]
        return {"objects": n, "bytes": total, "mb": round(total / 1e6, 1)}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["verify", "stats"])
    p.add_argument("--symbols", default=None,
                   help="comma-separated subset (default: all in manifest)")
    args = p.parse_args(argv)

    store = MinuteStore()
    if args.command == "stats":
        print(json.dumps(store.stats(), indent=2))
        return
    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else None)
    out = store.verify(symbols)
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
