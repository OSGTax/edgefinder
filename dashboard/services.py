"""EdgeFinder v2 — Runtime service initialization.

Slim post-arena module: a data provider, a DB session factory, and an
APScheduler running the v2 portfolio engine + nightly data-collection jobs.
The old per-ticker arena (intraday loop, event bus, watchlists, cooldowns)
was retired — the v2 engine (edgefinder/engine/live.py) is the only live
trading path.

Called once from app.py lifespan. Module-level singletons for router access.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from config.settings import settings
from edgefinder.data.cache import DataCache
from edgefinder.data.polygon import PolygonDataProvider
from edgefinder.data.provider import CachedDataProvider
from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.db import models as db_models  # noqa: F401 — registers ORM tables
from edgefinder.db.models import (
    PromotedStrategy,
    StrategyAccount,
    StrategySnapshot,
    SystemHeartbeat,
)
from edgefinder.market.benchmarks import BenchmarkService
from edgefinder.market.snapshot import MarketSnapshotService
from edgefinder.scheduler.scheduler import EdgeFinderScheduler

logger = logging.getLogger(__name__)

# ── Module-level singletons ─────────────────────────────

_provider: CachedDataProvider | None = None
_scheduler: EdgeFinderScheduler | None = None
_session_factory = None
_plan_access: dict[str, bool] = {}


def get_plan_access() -> dict[str, bool]:
    """Get plan access probe results for API display."""
    return _plan_access


def get_provider():
    return _provider


def get_scheduler() -> EdgeFinderScheduler | None:
    return _scheduler


# ── Initialization ──────────────────────────────────────


def init_services() -> None:
    """Initialize the data provider, DB session factory, and scheduler."""
    global _provider, _scheduler, _session_factory

    # Data provider — DataHub wraps Polygon + optional supplements
    try:
        polygon = PolygonDataProvider()
    except ValueError:
        logger.warning(
            "No Polygon API key — scheduled jobs disabled. "
            "Set EDGEFINDER_POLYGON_API_KEY in .env"
        )
        return

    # Probe plan access in background — don't block server startup
    import threading

    def _probe():
        global _plan_access
        logger.info("Probing Massive API plan access...")
        _plan_access = polygon.probe_plan_access()
    threading.Thread(target=_probe, daemon=True, name="plan-probe").start()

    from edgefinder.core.interfaces import DataHub

    hub = DataHub(CachedDataProvider(polygon, DataCache()))

    # Future: register supplemental providers here via hub.register_supplement()

    _provider = hub

    # DB session factory
    engine = get_engine()
    _session_factory = get_session_factory(engine)

    # Scheduler — v2 engine + data-collection jobs only.
    if os.getenv("EDGEFINDER_SCHEDULER_ENABLED", "true").lower() == "false":
        logger.info("Scheduler disabled via EDGEFINDER_SCHEDULER_ENABLED=false")
    else:
        _scheduler = EdgeFinderScheduler()
        # NOTE: the AI analyst's DAILY research runs on GitHub Actions
        # (.github/workflows/analyst.yml), not here — the rationale step needs
        # the `claude` CLI, which the Render web process doesn't have. The
        # on-demand run_analyst_job (Picks "Run now") still works here, with
        # deterministic text. Scheduling it here too would clobber the richer
        # Actions decision, so analyst_fn is intentionally not wired.
        _scheduler.setup(
            portfolio_rebalance_fn=_v2_portfolio_job,
            v2_snapshot_fn=_v2_snapshot_job,
            nightly_scan_fn=_nightly_scan_job,
            benchmark_collect_fn=_benchmark_job,
            market_snapshot_fn=_market_snapshot_job,
            sector_rotation_fn=_sector_rotation_job,
            news_accumulate_fn=_news_accumulate_job,
            dividend_split_fn=_dividend_split_job,
            r2_sync_fn=_r2_sync_job,
        )
        _scheduler.start()

    logger.info("Services initialized — v2 pipeline running")


def shutdown_services() -> None:
    """Gracefully shut down all services."""
    if _scheduler:
        _scheduler.stop()
    try:
        get_engine().dispose()
    except Exception:
        logger.debug("Engine dispose at shutdown failed", exc_info=True)
    logger.info("Services shut down")


# ── Heartbeat ───────────────────────────────────────────


def _record_heartbeat(component: str, ok: bool, detail: dict) -> None:
    """Upsert the single per-component liveness heartbeat row.

    Best-effort and isolated on its own session: heartbeat bookkeeping
    must never roll back or break a job. One row per component, so a
    plain SELECT-then-insert/update is correct on both SQLite (dev) and
    Postgres (prod) without dialect-specific upserts. A controlled skip
    is written fresh with ok=True so a legitimate no-op never reads as a
    stall to the watchdog.
    """
    if not _session_factory:
        return
    session = _session_factory()
    try:
        hb = session.query(SystemHeartbeat).filter(
            SystemHeartbeat.component == component
        ).one_or_none()
        now = datetime.now(timezone.utc)
        if hb is None:
            session.add(SystemHeartbeat(
                component=component, last_run_at=now, ok=ok, detail=detail
            ))
        else:
            hb.last_run_at = now
            hb.ok = ok
            hb.detail = detail
        session.commit()
    except Exception:
        logger.exception("Heartbeat upsert failed for %s", component)
        session.rollback()
    finally:
        session.close()


# ── Scheduler job callbacks ─────────────────────────────


def _v2_portfolio_job() -> None:
    """Called at 9:45 AM ET — run the engine-v2 portfolio paper-trading cycle.

    Trades every active row in promoted_strategies in its own isolated paper
    account (see edgefinder/engine/live.py). No-ops cleanly when nothing is
    promoted; writes the v2_portfolio_cycle heartbeat either way.
    """
    if not _session_factory or not _provider:
        return
    try:
        from edgefinder.engine.live import run_portfolio_cycle

        summary = run_portfolio_cycle(_session_factory, provider=_provider)
        logger.info("V2 portfolio cycle: %s", summary)
    except Exception:
        logger.exception("V2 portfolio cycle failed")


def run_analyst_job(strategy_name: str = "ai_analyst",
                    universe_spec: str | None = None) -> int | None:
    """Run the AI analyst's daily research and persist its decision.

    Shared by the 9:15 ET scheduled job and the on-demand /api/picks/run
    endpoint. Resolves the account's universe from its promoted row (default
    top:200 so the page can preview picks even before the account is
    activated). Narrates with the LLM when CLAUDE_CODE_OAUTH_TOKEN is set,
    deterministic otherwise. Writes the ``analyst`` heartbeat either way.
    """
    if not _session_factory:
        return None
    try:
        from edgefinder.agents.analyst import run_analyst

        spec = universe_spec
        if spec is None:
            session = _session_factory()
            try:
                promo = (session.query(PromotedStrategy)
                         .filter(PromotedStrategy.strategy_name == strategy_name)
                         .one_or_none())
                spec = (promo.universe_spec if promo else None) or "top:200"
            finally:
                session.close()
        rid = run_analyst(_session_factory, strategy_name=strategy_name,
                          universe_spec=spec,
                          use_llm=bool(os.getenv("CLAUDE_CODE_OAUTH_TOKEN")))
        _record_heartbeat("analyst", ok=rid is not None,
                          detail={"strategy": strategy_name,
                                  "decision_id": rid, "universe": spec})
        logger.info("Analyst run for %s: decision_id=%s", strategy_name, rid)
        return rid
    except Exception as exc:
        logger.exception("Analyst job failed")
        _record_heartbeat("analyst", ok=False,
                          detail={"error": f"{type(exc).__name__}: {exc}"})
        return None


def _v2_snapshot_job(now: datetime | None = None) -> None:
    """Every 30 min during market hours — mark v2 accounts + append snapshots.

    For each ACTIVE promoted strategy with an account row: recompute cash from
    the trades + dividend_credits tables (the CLAUDE.md integrity formula via
    engine/live._recalc_cash), mark open lots to the latest price, update the
    strategy_accounts row (engine/live._mark_account), and append one
    StrategySnapshot row — the equity-curve time series the dashboard charts.
    """
    if not _session_factory or not _provider:
        return
    # _mark_account internally recomputes cash via engine/live._recalc_cash
    # (the CLAUDE.md integrity formula + dividend credits) — one shared
    # implementation between the daily cycle and this intraday mark.
    from edgefinder.engine.live import STARTING_CAPITAL, _mark_account, _memoized
    from edgefinder.trading.journal import TradeJournal

    now = now or datetime.now(timezone.utc)
    session = _session_factory()
    try:
        active = {p.strategy_name for p in session.query(PromotedStrategy)
                  .filter(PromotedStrategy.active.is_(True)).all()}
        accounts = {a.strategy_name: a for a in session.query(StrategyAccount).all()}
        names = sorted(active & set(accounts))
        if not names:
            _record_heartbeat("v2_snapshot", ok=True,
                              detail={"skip": "no active promoted accounts"})
            return

        price_fn = _memoized(_provider.get_latest_price)
        journal = TradeJournal(session)
        snapped = 0
        for name in names:
            # _mark_account recomputes cash + positions_value from the trades
            # table and live prices, and upserts the strategy_accounts row.
            _mark_account(session, name, journal, price_fn, dry_run=False)
            acct = accounts[name]
            session.refresh(acct)
            session.add(StrategySnapshot(
                strategy_name=name,
                timestamp=now,
                cash=acct.cash_balance,
                positions_value=acct.open_positions_value,
                total_equity=acct.total_equity,
                drawdown_pct=acct.drawdown_pct,
                total_return_pct=round(
                    (acct.total_equity - STARTING_CAPITAL)
                    / STARTING_CAPITAL * 100, 4),
            ))
            snapped += 1
        session.commit()
        logger.info("V2 snapshot: %d strategies marked", snapped)
        _record_heartbeat("v2_snapshot", ok=True, detail={"strategies": snapped})
    except Exception as exc:
        session.rollback()
        logger.exception("V2 snapshot job failed")
        _record_heartbeat("v2_snapshot", ok=False,
                          detail={"error": f"{type(exc).__name__}: {exc}"})
    finally:
        session.close()


def _resolve_scan_universe() -> list[str]:
    """Resolve the universe to scan, applying the dollar-volume pre-filter.

    Tries get_top_dollar_volume_tickers first (top N most-liquid by
    yesterday's volume * close, 1 API call). Falls back to the full
    common-stock universe if the grouped daily aggs call fails.
    """
    try:
        top = _provider.get_top_dollar_volume_tickers(
            top_n=settings.scanner_max_universe_size,
            min_price=settings.scanner_min_price,
            max_price=settings.scanner_max_price,
        )
    except Exception:
        logger.exception("Dollar-volume pre-filter failed, falling back to full universe")
        top = []

    if top:
        return top

    logger.warning(
        "Dollar-volume universe was empty — falling back to get_ticker_universe"
    )
    return sorted(_provider.get_ticker_universe())


def _nightly_scan_job() -> None:
    """Called at 6:15 PM ET on weekdays — the nightly DATA collector.

    Runs the slimmed unified scanner (fundamentals fetch + persist, no
    strategy qualification) over the top-N dollar-volume universe, then
    snapshots the fundamentals table into the PIT store — that is how the
    honest fundamental history accumulates.
    """
    if not _provider or not _session_factory:
        return

    from edgefinder.scanner.unified_scanner import UnifiedScanner

    weekday = datetime.now().weekday()
    if weekday >= 5:
        logger.info("Weekend — skipping nightly scan")
        _record_heartbeat("nightly_scan", ok=True, detail={"skip": "weekend"})
        return

    tickers = _resolve_scan_universe()
    logger.info("Nightly scan: %d tickers (top dollar-volume universe)", len(tickers))

    scanner = UnifiedScanner(_provider, _session_factory)
    try:
        summary = scanner.run(tickers)
        logger.info("Nightly scan results: %s", summary)
    except Exception as exc:
        logger.exception("Nightly scan failed")
        _record_heartbeat("nightly_scan", ok=False,
                          detail={"error": f"{type(exc).__name__}: {exc}"})
        return

    # PIT fundamentals: snapshot what tonight's scan wrote, dated today —
    # this is how the honest fundamental-strategy history accumulates.
    pit_rows = None
    try:
        from edgefinder.data.pit_fundamentals import snapshot_fundamentals

        session = _session_factory()
        try:
            pit_rows = snapshot_fundamentals(session)
        finally:
            session.close()
    except Exception:
        logger.exception("PIT fundamentals snapshot failed")

    _record_heartbeat("nightly_scan", ok=True,
                      detail={**summary, "pit_rows": pit_rows})


def _benchmark_job() -> None:
    """Called at 4:10 PM ET on weekdays."""
    if not _provider or not _session_factory:
        return
    session = _session_factory()
    try:
        svc = BenchmarkService(_provider, session)
        count = svc.collect_daily()
        logger.info("Benchmark collection: %d records", count)
    except Exception:
        logger.exception("Benchmark collection failed")
    finally:
        session.close()


def _market_snapshot_job() -> None:
    """Called at 4:05 PM ET on weekdays — persist a market-wide snapshot.

    Keeps /api/market/regime fresh now that snapshots are no longer captured
    at (old-arena) trade time.
    """
    if not _provider or not _session_factory:
        return
    session = _session_factory()
    try:
        svc = MarketSnapshotService(_provider, session)
        snapshot_id = svc.capture_and_persist()
        logger.info("Market snapshot persisted (#%s)", snapshot_id)
        _record_heartbeat("market_snapshot", ok=True, detail={"id": snapshot_id})
    except Exception as exc:
        logger.exception("Market snapshot job failed")
        _record_heartbeat("market_snapshot", ok=False,
                          detail={"error": f"{type(exc).__name__}: {exc}"})
    finally:
        session.close()


# Module-level cache for latest sector rotation data (for API access)
_sector_rotation_data: list[dict] = []


def get_sector_rotation() -> list[dict]:
    """Get cached sector rotation data for API endpoints."""
    return _sector_rotation_data


def _sector_rotation_job() -> None:
    """Called at 4:15 PM ET on weekdays. Computes Bloomberg-style RRG."""
    global _sector_rotation_data
    if not _provider:
        return
    try:
        from edgefinder.market.sector_rotation import SectorRotationService
        svc = SectorRotationService(_provider)
        rotation = svc.compute_rotation()
        _sector_rotation_data = [r.to_dict() for r in rotation]
        logger.info(
            "Sector rotation updated: %d sectors, %d leading",
            len(rotation),
            sum(1 for r in rotation if r.quadrant == "leading"),
        )
    except Exception:
        logger.exception("Sector rotation job failed")


def _news_accumulate_job() -> None:
    """Called hourly during market hours. Accumulates news into DB."""
    if not _provider or not _session_factory:
        return
    try:
        from edgefinder.data.accumulator import DataAccumulator
        acc = DataAccumulator(_provider, _session_factory)
        acc.accumulate_news()
    except Exception:
        logger.exception("News accumulation job failed")


def _dividend_split_job() -> None:
    """Called at 6:30 PM ET on weekdays. Accumulates dividends and splits."""
    if not _provider or not _session_factory:
        return
    try:
        from edgefinder.data.accumulator import DataAccumulator
        acc = DataAccumulator(_provider, _session_factory)
        acc.accumulate_dividends()
        acc.accumulate_splits()
    except Exception:
        logger.exception("Dividend/split accumulation job failed")


def _r2_sync_job() -> None:
    """Called at 7:00 PM ET — incremental daily_bars -> R2 merge + DB prune.

    Self-enabling: runs only when the four R2_* env vars are present (add
    them to Render to turn the nightly mirror on; absent, it logs and skips).
    Two steps, in safety order:
      1. sync — MERGE today's new DB rows into the grow-only R2 store.
      2. prune — shed DB rows older than the retention window for symbols
         whose DB state is fingerprint-current in R2 (free-tier cap
         maintenance; a symbol that failed to sync is never pruned).
    """
    if not _session_factory:
        return
    if not all(os.getenv(k) for k in (
            "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")):
        logger.info("R2 sync skipped — R2_* env vars not set")
        return
    try:
        from edgefinder.data.barstore import (
            DB_RETENTION_DAYS,
            BarStore,
            db_protected_symbols,
        )

        store = BarStore()
        session = _session_factory()
        try:
            result = store.sync(session)
            logger.info("R2 sync: %s", result)
            pruned = store.prune_db(
                session, keep_days=DB_RETENTION_DAYS,
                protected=db_protected_symbols(session))
            logger.info("DB prune: %s", pruned)
        finally:
            session.close()
    except Exception:
        logger.exception("R2 sync/prune failed")


# ── On-demand EOD runner (admin endpoint) ───────────────


def run_eod_jobs(jobs: list[str] | None = None) -> dict[str, str]:
    """Run the post-close jobs in-process, on demand.

    Triggered by the /api/admin/run-eod endpoint. Each job is isolated so
    one failure doesn't abort the rest. Returns a
    {job_name: "ok"|"error"|"skipped"} report.
    """
    pipeline = [
        ("market_snapshot", _market_snapshot_job),
        ("benchmark", _benchmark_job),
        ("sector_rotation", _sector_rotation_job),
        ("nightly_scan", _nightly_scan_job),
        ("dividend_split", _dividend_split_job),
        ("v2_snapshot", _v2_snapshot_job),
    ]
    report: dict[str, str] = {}
    for name, fn in pipeline:
        if jobs is not None and name not in jobs:
            report[name] = "skipped"
            continue
        try:
            fn()
            report[name] = "ok"
        except Exception:
            logger.exception("EOD job '%s' failed", name)
            report[name] = "error"
    logger.info("EOD jobs complete: %s", report)
    return report
