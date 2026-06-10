"""Two-tier storage: the daily-bars history as Parquet on Cloudflare R2.

History (daily_bars: millions of write-once, read-heavy rows) lives as one
Parquet object per symbol in R2; the live/operational data stays in Postgres.
The DB is NEVER deleted from — R2 is a growing, verified mirror that backtest
loaders can read instead of hammering the pooler, and the durable home for
history beyond the DB's size budget.

Layout:
    bars/{SYMBOL}.parquet       columns: date, open, high, low, close, volume
    manifest.json               {symbol: {"rows": n, "max_date": "YYYY-MM-DD"}}

Sync is incremental and idempotent: a symbol is re-exported only when its DB
row count or max date differs from the manifest (a per-symbol rewrite is at
most ~5,400 rows — trivial). ``verify`` proves R2 == DB on a sample before
anything is allowed to read from the store.

CLI:
    python -m edgefinder.data.barstore sync             # export new/changed
    python -m edgefinder.data.barstore verify [-n 25]   # R2 == DB sample check
    python -m edgefinder.data.barstore stats            # object count / bytes

Env (Codespaces/Actions secrets; values are .strip()ed — pasted secrets have
carried trailing newlines before): R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
R2_ENDPOINT, R2_BUCKET.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date

import pandas as pd
from sqlalchemy import func

from edgefinder.db.models import DailyBar

logger = logging.getLogger(__name__)

_COLS = ["date", "open", "high", "low", "close", "volume"]
_BATCH = 200          # symbols per DB pull
MANIFEST_KEY = "manifest.json"


def _bar_key(symbol: str) -> str:
    return f"bars/{symbol}.parquet"


class BarStore:
    """Parquet-per-symbol bar storage on any S3-compatible endpoint."""

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
                              max_pool_connections=16),   # match upload workers
                region_name="auto",
            )
        except KeyError as e:
            raise RuntimeError(f"missing R2 secret: {e}") from None

    # ── primitives ───────────────────────────────────────

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
        # Parquet round-trips dates as datetime64; the engine wants date objects
        df["date"] = pd.to_datetime(df["date"]).dt.date
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

    # ── public API ───────────────────────────────────────

    def load(self, symbols: list[str],
             max_workers: int = 8) -> dict[str, pd.DataFrame]:
        """Read bars for ``symbols`` from R2 (missing symbols are omitted).

        GETs run in parallel (botocore clients are thread-safe): a
        full-market load for PIT universe resolution is ~8.6k objects —
        minutes sequential, tens of seconds at 8-16 workers.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _one(sym: str):
            return sym, self._get_frame(_bar_key(sym))

        out: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for sym, df in pool.map(_one, symbols):
                if df is not None and len(df):
                    out[sym] = df
        return out

    def sync(self, session, symbols: list[str] | None = None) -> dict:
        """Export new/changed symbols from daily_bars to R2. Idempotent."""
        manifest = self.read_manifest()

        db_state = {
            sym: {"rows": n, "max_date": str(mx)}
            for sym, n, mx in (
                session.query(DailyBar.symbol,
                              func.count(DailyBar.id),
                              func.max(DailyBar.date))
                .group_by(DailyBar.symbol).all())
        }
        targets = sorted(
            sym for sym, state in db_state.items()
            if (symbols is None or sym in symbols)
            and manifest.get(sym) != state)
        logger.info("sync: %d of %d symbols changed", len(targets), len(db_state))

        uploaded = bytes_up = 0
        for i in range(0, len(targets), _BATCH):
            batch = targets[i:i + _BATCH]
            rows = (session.query(
                        DailyBar.symbol, DailyBar.date, DailyBar.open,
                        DailyBar.high, DailyBar.low, DailyBar.close,
                        DailyBar.volume)
                    .filter(DailyBar.symbol.in_(batch))
                    .order_by(DailyBar.symbol, DailyBar.date).all())
            by_sym: dict[str, list] = {}
            for sym, *vals in rows:
                by_sym.setdefault(sym, []).append(vals)

            # uploads parallelize cleanly (sequential PUTs would take hours
            # for a full export); the DB pull above stays batched/sequential
            from concurrent.futures import ThreadPoolExecutor

            def _upload(sym: str) -> tuple[str, int]:
                df = pd.DataFrame(by_sym[sym], columns=_COLS)
                return sym, self._put_frame(_bar_key(sym), df)

            with ThreadPoolExecutor(max_workers=16) as pool:
                for sym, nbytes in pool.map(
                        _upload, [s for s in batch if s in by_sym]):
                    bytes_up += nbytes
                    manifest[sym] = db_state[sym]
                    uploaded += 1
            logger.info("sync: %d/%d uploaded (%.1f MB)",
                        uploaded, len(targets), bytes_up / 1e6)
            self.write_manifest(manifest)   # checkpoint per batch
        self.write_manifest(manifest)
        return {"changed": len(targets), "uploaded": uploaded,
                "bytes": bytes_up, "symbols_total": len(db_state)}

    def verify(self, session, sample: int = 25) -> dict:
        """Prove R2 == DB for a deterministic sample of synced symbols."""
        manifest = self.read_manifest()
        if not manifest:
            return {"ok": False, "reason": "empty manifest"}
        symbols = sorted(manifest)
        step = max(1, len(symbols) // sample)
        checked, mismatches = [], []
        for sym in symbols[::step][:sample]:
            r2 = self._get_frame(_bar_key(sym))
            rows = (session.query(
                        DailyBar.date, DailyBar.open, DailyBar.high,
                        DailyBar.low, DailyBar.close, DailyBar.volume)
                    .filter(DailyBar.symbol == sym)
                    .order_by(DailyBar.date).all())
            db = pd.DataFrame(rows, columns=_COLS)
            ok = (r2 is not None and len(r2) == len(db)
                  and list(r2["date"]) == list(db["date"])
                  and all(
                      (r2[c].astype(float).round(6)
                       == db[c].astype(float).round(6)).all()
                      for c in _COLS[1:]))
            checked.append(sym)
            if not ok:
                mismatches.append(sym)
        return {"ok": not mismatches, "checked": len(checked),
                "mismatches": mismatches}

    def stats(self) -> dict:
        n = total = 0
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": "bars/"}
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

    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["sync", "verify", "stats"])
    p.add_argument("-n", "--sample", type=int, default=25,
                   help="verify sample size")
    p.add_argument("--symbols", default=None,
                   help="comma-separated subset (default: all)")
    args = p.parse_args(argv)

    store = BarStore()
    if args.command == "stats":
        print(json.dumps(store.stats(), indent=2))
        return

    session = get_session_factory(get_engine())()
    try:
        if args.command == "sync":
            symbols = ([s.strip().upper() for s in args.symbols.split(",")]
                       if args.symbols else None)
            print(json.dumps(store.sync(session, symbols), indent=2))
        else:
            print(json.dumps(store.verify(session, args.sample), indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
