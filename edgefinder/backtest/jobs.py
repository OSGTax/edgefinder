"""In-memory background runner for universe-scale backtests.

A full-universe replay is CPU-bound and can run for minutes — too long for a
synchronous HTTP request. Jobs run on a single background worker (one at a
time, since each backtest saturates a core), report progress as they go, and
are polled via the API. Job state is in-memory and lost on restart: these are
on-demand research runs, not persisted artifacts.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from edgefinder.backtest.daily_backtest import run_daily_backtest
from edgefinder.db.models import DailyBar, IndexDaily

logger = logging.getLogger(__name__)

MAX_UNIVERSE = 4000          # hard ceiling on symbols resolved per job
MAX_RESULT_TRADES = 500      # cap trades echoed back for display
_MAX_JOBS = 50               # retained job history (in-memory)
_IN_CHUNK = 500              # bar-load IN()-clause chunk size


# ── benchmark (shared with the sync endpoint) ─────────────────────────


def spy_benchmark(db: Session, start, end) -> dict | None:
    """SPY buy-hold return over [start, end], preferring full-range daily_bars
    and falling back to index_daily (shorter history). Aligned to whatever SPY
    data overlaps the backtest window."""
    rows = (
        db.query(DailyBar.date, DailyBar.close)
        .filter(DailyBar.symbol == "SPY", DailyBar.date >= start, DailyBar.date <= end)
        .order_by(DailyBar.date).all()
    )
    if not rows:
        rows = (
            db.query(IndexDaily.date, IndexDaily.close)
            .filter(IndexDaily.symbol == "SPY", IndexDaily.date >= start, IndexDaily.date <= end)
            .order_by(IndexDaily.date).all()
        )
    rows = [(d, c) for d, c in rows if c]
    if len(rows) < 2:
        return None
    first_close, last_close = rows[0][1], rows[-1][1]
    if first_close <= 0:
        return None

    def _d(x):
        return x.date().isoformat() if hasattr(x, "date") else str(x)[:10]

    return {
        "symbol": "SPY",
        "return_pct": (last_close - first_close) / first_close * 100,
        "period": f"{_d(rows[0][0])}..{_d(rows[-1][0])}",
    }


# ── universe resolution + bar loading ─────────────────────────────────


def resolve_universe(db: Session, mode: str, symbols: list[str], top_n: int) -> list[str]:
    """Turn a universe spec into a concrete symbol list from daily_bars."""
    if mode == "symbols":
        return sorted({s.strip().upper() for s in symbols if s.strip()})
    if mode == "full":
        rows = db.query(DailyBar.symbol).distinct().all()
        return sorted(r[0] for r in rows)
    if mode == "top":
        rows = (
            db.query(DailyBar.symbol)
            .group_by(DailyBar.symbol)
            .order_by(func.avg(DailyBar.close * DailyBar.volume).desc())
            .limit(top_n).all()
        )
        return [r[0] for r in rows]
    raise ValueError(f"unknown universe mode {mode!r}")


def _load_bars(db: Session, symbols: list[str], start, end):
    """Load OHLCV as lightweight column tuples (not ORM rows) and group into
    per-symbol DataFrames. Chunks the IN() clause to respect driver limits and
    keep memory bounded for large universes."""
    by_symbol: dict[str, list] = {}
    bt_start = bt_end = None
    for i in range(0, len(symbols), _IN_CHUNK):
        chunk = symbols[i:i + _IN_CHUNK]
        q = db.query(
            DailyBar.symbol, DailyBar.date, DailyBar.open, DailyBar.high,
            DailyBar.low, DailyBar.close, DailyBar.volume,
        ).filter(DailyBar.symbol.in_(chunk))
        if start:
            q = q.filter(DailyBar.date >= start)
        if end:
            q = q.filter(DailyBar.date <= end)
        for sym, d, o, h, lo, c, v in q.all():
            by_symbol.setdefault(sym, []).append((d, o, h, lo, c, v))
            if bt_start is None or d < bt_start:
                bt_start = d
            if bt_end is None or d > bt_end:
                bt_end = d
    bars = {
        sym: pd.DataFrame(recs, columns=["date", "open", "high", "low", "close", "volume"])
        for sym, recs in by_symbol.items()
    }
    return bars, bt_start, bt_end


# ── job model + manager ───────────────────────────────────────────────


@dataclass
class BacktestJob:
    id: str
    params: dict
    status: str = "queued"          # queued | running | done | error
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    num_symbols: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self, include_result: bool = True) -> dict:
        d = {
            "id": self.id, "status": self.status, "progress": self.progress,
            "num_symbols": self.num_symbols, "error": self.error,
            "params": self.params, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }
        if include_result:
            d["result"] = self.result
        return d


def _execute_backtest(job: "BacktestJob", db: Session) -> dict:
    p = job.params
    symbols = resolve_universe(
        db, p.get("mode", "symbols"), p.get("symbols", []), int(p.get("top_n", 100))
    )
    if not symbols:
        raise ValueError("no symbols resolved for this universe")
    if len(symbols) > MAX_UNIVERSE:
        symbols = symbols[:MAX_UNIVERSE]
    job.num_symbols = len(symbols)
    job.progress = {"phase": "loading", "done": 0, "total": len(symbols)}

    bars, bt_start, bt_end = _load_bars(db, symbols, p.get("start"), p.get("end"))
    if not bars:
        raise ValueError("no daily_bars for that universe/range — run the backfill first")
    job.num_symbols = len(bars)

    benchmark = spy_benchmark(db, bt_start, bt_end)

    def progress_cb(info: dict) -> None:
        job.progress = info

    result = run_daily_backtest(
        p["strategy"], bars,
        starting_cash=float(p.get("starting_cash", 10_000.0)),
        benchmark=benchmark, progress_cb=progress_cb,
    )
    result["num_symbols"] = len(bars)
    result["universe_mode"] = p.get("mode", "symbols")

    # Bound payload: keep the most impactful trades; stats already cover all.
    trades = result.get("trades", [])
    if len(trades) > MAX_RESULT_TRADES:
        result["trades_total"] = len(trades)
        result["trades"] = sorted(
            trades, key=lambda t: abs(t.get("pnl_dollars") or 0), reverse=True
        )[:MAX_RESULT_TRADES]
    return result


class _JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, BacktestJob] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest-job")

    def submit(self, params: dict, session_factory) -> BacktestJob:
        job = BacktestJob(id=uuid.uuid4().hex[:12], params=params)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > _MAX_JOBS:
                self._jobs.pop(self._order.pop(0), None)
        self._pool.submit(self._run, job, session_factory)
        return job

    def get(self, job_id: str) -> BacktestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 20) -> list[BacktestJob]:
        with self._lock:
            return [self._jobs[i] for i in self._order[-limit:][::-1]]

    def _run(self, job: BacktestJob, session_factory) -> None:
        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        db = session_factory()
        try:
            job.result = _execute_backtest(job, db)
            job.status = "done"
        except Exception as e:  # noqa: BLE001 — surface any failure to the poller
            logger.exception("backtest job %s failed", job.id)
            job.error = str(e)
            job.status = "error"
        finally:
            db.close()
            job.finished_at = datetime.now(timezone.utc).isoformat()


job_manager = _JobManager()
