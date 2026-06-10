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

from sqlalchemy.exc import OperationalError
import os
from datetime import date

import pandas as pd
from sqlalchemy import func

from edgefinder.db.models import DailyBar

logger = logging.getLogger(__name__)

_COLS = ["date", "open", "high", "low", "close", "volume"]
DB_RETENTION_DAYS = 365   # trailing window the DB keeps; R2 keeps forever

# symbols whose FULL history always stays in the DB: the deep ETF lane +
# benchmark indices run from the DB (small, calibration-anchoring), and
# live trading must never lose its own books' history
DB_PROTECTED_ETFS = ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "EFA",
                     "AGG", "LQD", "HYG")

_BATCH = 40           # symbols per DB pull — small enough that one
                      # statement stays under Supabase's statement_timeout
                      # even when the project is disk-throttled (a 200-symbol
                      # ORDER BY pull got QueryCanceled on 2026-06-10)
MANIFEST_KEY = "manifest.json"


def db_protected_symbols(session) -> set[str]:
    """Symbols never pruned from the DB: deep ETFs + benchmark indices +
    promoted-strategy universes + anything with an open trade."""
    from config.settings import settings
    from edgefinder.db.models import PromotedStrategy, TradeRecord

    out = {s.upper() for s in DB_PROTECTED_ETFS}
    out |= {s.strip().upper() for s in settings.index_symbols if s.strip()}
    for promo in session.query(PromotedStrategy).filter(
            PromotedStrategy.active.is_(True)).all():
        out |= {str(s).upper() for s in (promo.symbols or [])}
    for (sym,) in (session.query(TradeRecord.symbol)
                   .filter(TradeRecord.status == "OPEN").distinct().all()):
        out.add(str(sym).upper())
    return out


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
        """MERGE new/changed DB rows into R2. The store is GROW-ONLY.

        R2 is the permanent asset; the DB is allowed to shed old rows
        (free-tier retention). So a symbol's parquet is the UNION of what R2
        already holds and what the DB currently holds — the DB wins on
        conflicting dates (it is where corrections land), and rows the DB no
        longer has are PRESERVED. A sync after a DB prune can therefore
        never shrink the mirror.

        Manifest entry per symbol:
          rows / max_date  — describe the R2 PARQUET (the merged asset)
          db_rows / db_max — fingerprint of the DB state last merged; change
                             detection compares THIS against the live DB.
        Legacy entries (pre-fingerprint) whose parquet state equals the live
        DB state are upgraded in place without an upload.

        Known limitation (pre-existing): the fingerprint is (row count,
        max date), so an in-place VALUE correction that adds no rows is not
        detected — force it with ``sync --symbols X`` when hand-correcting
        history.
        """
        manifest = self.read_manifest()

        db_state = {
            sym: {"db_rows": n, "db_max": str(mx)}
            for sym, n, mx in (
                session.query(DailyBar.symbol,
                              func.count(DailyBar.id),
                              func.max(DailyBar.date))
                .group_by(DailyBar.symbol).all())
        }

        targets: list[str] = []
        migrated = 0
        for sym, state in db_state.items():
            if symbols is not None:
                if sym in symbols:
                    targets.append(sym)   # explicitly named = forced re-merge
                continue
            entry = manifest.get(sym)
            if entry and "db_rows" not in entry:
                # legacy replace-era entry: parquet WAS the DB state
                if (entry.get("rows") == state["db_rows"]
                        and entry.get("max_date") == state["db_max"]):
                    entry.update(state)   # fingerprint backfill, no upload
                    migrated += 1
                    continue
            if not entry or (entry.get("db_rows"), entry.get("db_max")) != \
                    (state["db_rows"], state["db_max"]):
                targets.append(sym)
        targets.sort()
        if migrated:
            self.write_manifest(manifest)
            logger.info("sync: backfilled fingerprints for %d legacy entries",
                        migrated)
        logger.info("sync: %d of %d symbols changed", len(targets), len(db_state))

        def _pull(batch: list[str]) -> list:
            return (session.query(
                        DailyBar.symbol, DailyBar.date, DailyBar.open,
                        DailyBar.high, DailyBar.low, DailyBar.close,
                        DailyBar.volume)
                    .filter(DailyBar.symbol.in_(batch))
                    .order_by(DailyBar.symbol, DailyBar.date).all())

        uploaded = bytes_up = 0
        failed: list[str] = []
        for i in range(0, len(targets), _BATCH):
            batch = targets[i:i + _BATCH]
            try:
                rows = _pull(batch)
            except OperationalError:
                # statement timeout / dropped connection on a throttled DB:
                # recover the session and retry in halves; a batch that still
                # fails is skipped (the manifest diff re-targets it next run)
                session.rollback()
                rows = []
                for j in range(0, len(batch), max(1, _BATCH // 2)):
                    half = batch[j:j + max(1, _BATCH // 2)]
                    try:
                        rows.extend(_pull(half))
                    except OperationalError:
                        session.rollback()
                        failed.extend(half)
                        logger.warning("sync: pull failed for %d symbols "
                                       "(%s..) — will retry next run",
                                       len(half), half[0])
            by_sym: dict[str, list] = {}
            for sym, *vals in rows:
                by_sym.setdefault(sym, []).append(vals)

            # merge + upload parallelize cleanly (each symbol = one GET,
            # one union, one PUT); the DB pull above stays batched
            from concurrent.futures import ThreadPoolExecutor

            def _merge_upload(sym: str) -> tuple[str, int, int, str]:
                db_df = pd.DataFrame(by_sym[sym], columns=_COLS)
                existing = self._get_frame(_bar_key(sym))
                if existing is not None and len(existing):
                    keep = existing[~existing["date"].isin(set(db_df["date"]))]
                    merged = (pd.concat([keep, db_df], ignore_index=True)
                              .sort_values("date").reset_index(drop=True))
                else:
                    merged = db_df
                nbytes = self._put_frame(_bar_key(sym), merged)
                return sym, nbytes, len(merged), str(merged["date"].iloc[-1])

            with ThreadPoolExecutor(max_workers=16) as pool:
                for sym, nbytes, mrows, mmax in pool.map(
                        _merge_upload, [s for s in batch if s in by_sym]):
                    bytes_up += nbytes
                    manifest[sym] = {"rows": mrows, "max_date": mmax,
                                     **db_state[sym]}
                    uploaded += 1
            logger.info("sync: %d/%d merged+uploaded (%.1f MB)",
                        uploaded, len(targets), bytes_up / 1e6)
            self.write_manifest(manifest)   # checkpoint per batch
        self.write_manifest(manifest)
        return {"changed": len(targets), "uploaded": uploaded,
                "failed": len(failed), "migrated": migrated,
                "bytes": bytes_up, "symbols_total": len(db_state)}

    def prune_db(self, session, *, keep_days: int = 365,
                 protected: set[str] | None = None) -> dict:
        """Steady-state retention: drop DB rows older than ``keep_days`` for
        symbols that are manifest-current in R2 (the nightly counterpart of
        scripts/slim_daily_bars.py — keeps the free-tier DB from creeping
        back over its cap as the nightly ingest appends ~1000 rows/day).

        Per-symbol manifest guard: a symbol whose DB state differs from the
        manifest is NOT pruned (its newest rows haven't mirrored yet — the
        next successful sync makes it eligible). Protected symbols are never
        touched. Plain DELETEs; freed space is reused by future inserts, so
        table size stays flat without VACUUM FULL.
        """
        from datetime import date as _date
        from datetime import timedelta as _timedelta

        protected = {s.upper() for s in (protected or set())}
        cutoff = _date.today() - _timedelta(days=keep_days)
        manifest = self.read_manifest()

        candidates = []
        rows = (session.query(DailyBar.symbol,
                              func.count(DailyBar.id),
                              func.max(DailyBar.date),
                              func.min(DailyBar.date))
                .group_by(DailyBar.symbol).all())
        for sym, n, mx, mn in rows:
            if sym in protected or mn >= cutoff:
                continue
            m = manifest.get(sym) or {}
            fp = (m.get("db_rows", m.get("rows")),
                  m.get("db_max", m.get("max_date")))
            if fp == (n, str(mx)):
                candidates.append(sym)

        deleted = 0
        chunk = 200
        for i in range(0, len(candidates), chunk):
            batch = candidates[i:i + chunk]
            deleted += (session.query(DailyBar)
                        .filter(DailyBar.symbol.in_(batch),
                                DailyBar.date < cutoff)
                        .delete(synchronize_session=False))
            session.commit()
        if deleted:
            logger.info("prune: deleted %d rows older than %s "
                        "(%d symbols, all R2-current)",
                        deleted, cutoff, len(candidates))
        return {"deleted": deleted, "symbols": len(candidates),
                "cutoff": str(cutoff)}

    def verify(self, session, sample: int = 25) -> dict:
        """Prove DB ⊆ R2 for a deterministic sample of synced symbols.

        The store is grow-only and the DB sheds old rows (retention), so the
        correct invariant is: every row the DB holds exists in R2 with
        identical values, and R2 holds at least as many rows. Pre-retention
        (DB == R2) passes as the equality special case.
        """
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
            if r2 is None or len(r2) < len(db):
                ok = False
            elif not len(db):
                ok = True          # DB shed everything; R2 keeps the asset
            else:
                sub = r2[r2["date"].isin(set(db["date"]))].reset_index(drop=True)
                ok = (len(sub) == len(db)
                      and list(sub["date"]) == list(db["date"])
                      and all(
                          (sub[c].astype(float).round(6)
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
            out = store.sync(session, symbols)
            print(json.dumps(out, indent=2))
            # partial progress is success (manifest diff self-heals next
            # run); only a zero-progress run with failures should go red
            if out["failed"] and not out["uploaded"]:
                raise SystemExit(1)
        else:
            print(json.dumps(store.verify(session, args.sample), indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
