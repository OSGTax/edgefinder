"""EdgeFinder v2 — Runtime service initialization.

Creates and wires all trading pipeline components:
Arena, Scheduler, TradeJournal, MarketSnapshot, EventBus subscribers.

Called once from app.py lifespan. Module-level singletons for router access.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import settings
from edgefinder.core.events import event_bus
from edgefinder.data.cache import DataCache
from edgefinder.data.polygon import PolygonDataProvider
from edgefinder.data.provider import CachedDataProvider
from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.db import models as db_models  # noqa: F401 — registers ORM tables
from edgefinder.db.models import (
    StrategyAccount,
    StrategySnapshot,
    Ticker,
)
from edgefinder.market.benchmarks import BenchmarkService
from edgefinder.market.snapshot import MarketSnapshotService
from edgefinder.scanner.scanner import FundamentalScanner
from edgefinder.scheduler.scheduler import EdgeFinderScheduler
from edgefinder.trading.arena import ArenaEngine
from edgefinder.trading.journal import TradeJournal

logger = logging.getLogger(__name__)

# ── Module-level singletons ─────────────────────────────

_provider: CachedDataProvider | None = None
_arena: ArenaEngine | None = None
_scheduler: EdgeFinderScheduler | None = None
_session_factory = None


def get_arena() -> ArenaEngine | None:
    return _arena


def get_scheduler() -> EdgeFinderScheduler | None:
    return _scheduler


# ── Initialization ──────────────────────────────────────


def init_services() -> None:
    """Initialize all trading pipeline services. Called once at startup."""
    global _provider, _arena, _scheduler, _session_factory

    # Data provider
    try:
        polygon = PolygonDataProvider()
    except ValueError:
        logger.warning(
            "No Polygon API key — trading pipeline disabled. "
            "Set EDGEFINDER_POLYGON_API_KEY in .env"
        )
        return

    _provider = CachedDataProvider(polygon, DataCache())

    # DB session factory
    engine = get_engine()
    _session_factory = get_session_factory(engine)

    # Arena — load strategies and set watchlist
    _arena = ArenaEngine(_provider)
    _arena.load_strategies()

    watchlist = _load_watchlist()
    if watchlist:
        _arena.set_global_watchlist(watchlist)
        logger.info("Watchlist set: %s", ", ".join(watchlist))
    else:
        logger.warning("No active tickers in DB — run the scanner first")

    # Event bus — persist trades and capture market snapshots
    _wire_event_bus()

    # Scheduler — wire all jobs and start
    _scheduler = EdgeFinderScheduler()
    _scheduler.setup(
        signal_check_fn=_signal_check_job,
        position_monitor_fn=_position_monitor_job,
        nightly_scan_fn=_nightly_scan_job,
        benchmark_collect_fn=_benchmark_job,
        snapshot_fn=_snapshot_job,
    )
    _scheduler.start()

    logger.info("Trading pipeline initialized — all services running")


def shutdown_services() -> None:
    """Gracefully shut down all services."""
    if _scheduler:
        _scheduler.stop()
    event_bus.clear()
    logger.info("Trading pipeline shut down")


# ── Watchlist ───────────────────────────────────────────


def _load_watchlist() -> list[str]:
    """Load active tickers from DB (populated by scanner)."""
    session = _session_factory()
    try:
        active = (
            session.query(Ticker.symbol)
            .filter(Ticker.is_active == True)
            .all()
        )
        return [row[0] for row in active]
    finally:
        session.close()


# ── Event Bus Wiring ────────────────────────────────────


def _wire_event_bus() -> None:
    """Subscribe to trade events for persistence."""

    def on_trade_opened(trade):
        session = _session_factory()
        try:
            # Capture market snapshot and link to trade
            snapshot_svc = MarketSnapshotService(_provider, session)
            snapshot_id = snapshot_svc.capture_and_persist()
            trade.market_snapshot_id = snapshot_id

            journal = TradeJournal(session)
            journal.log_trade(trade)
            logger.info(
                "[%s] Trade opened: %s %s @ $%.2f (snapshot #%d)",
                trade.strategy_name, trade.direction.value,
                trade.symbol, trade.entry_price, snapshot_id,
            )
        except Exception:
            logger.exception("Failed to persist opened trade %s", trade.trade_id)
        finally:
            session.close()

    def on_trade_closed(trade):
        session = _session_factory()
        try:
            journal = TradeJournal(session)
            journal.log_trade(trade)
            logger.info(
                "[%s] Trade closed: %s %s @ $%.2f — P&L $%.2f (%s)",
                trade.strategy_name, trade.direction.value,
                trade.symbol, trade.exit_price or 0,
                trade.pnl_dollars or 0, trade.exit_reason or "manual",
            )
        except Exception:
            logger.exception("Failed to persist closed trade %s", trade.trade_id)
        finally:
            session.close()

    event_bus.subscribe("trade.opened", on_trade_opened)
    event_bus.subscribe("trade.closed", on_trade_closed)


# ── Scheduler Job Callbacks ─────────────────────────────


def _signal_check_job() -> None:
    """Called every 15 min during market hours."""
    if not _arena:
        return
    try:
        trades = _arena.run_signal_check()
        if trades:
            logger.info("Signal check: %d trades opened", len(trades))
        else:
            logger.debug("Signal check: no new trades")
    except Exception:
        logger.exception("Signal check failed")


def _position_monitor_job() -> None:
    """Called every 5 min during market hours."""
    if not _arena:
        return
    try:
        closed = _arena.check_positions()
        if closed:
            logger.info("Position monitor: %d trades closed", len(closed))

        # Sync account state to DB
        _persist_account_state()
    except Exception:
        logger.exception("Position monitor failed")


def _nightly_scan_job() -> None:
    """Called at 6:15 PM ET on weekdays."""
    if not _provider:
        return
    session = _session_factory()
    try:
        scanner = FundamentalScanner(_provider, session)
        results = scanner.run()
        active = [s.symbol for s in results if s.qualifying_strategies]
        if active and _arena:
            _arena.set_global_watchlist(active)
            logger.info("Nightly scan: %d stocks qualify, watchlist updated", len(active))
        else:
            logger.info("Nightly scan: %d stocks scored, 0 qualified", len(results))
    except Exception:
        logger.exception("Nightly scan failed")
    finally:
        session.close()


def _benchmark_job() -> None:
    """Called at 4:10 PM ET on weekdays."""
    if not _provider:
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


def _snapshot_job() -> None:
    """Called at 4:05 PM ET — persist strategy account snapshots for equity curves."""
    if not _arena:
        return
    session = _session_factory()
    try:
        now = datetime.now(timezone.utc)
        accounts = _arena.get_all_accounts()
        for name, acct in accounts.items():
            snapshot = StrategySnapshot(
                strategy_name=name,
                timestamp=now,
                cash=acct["cash"],
                positions_value=acct["open_positions_value"],
                total_equity=acct["total_equity"],
                drawdown_pct=acct["drawdown_pct"],
                total_return_pct=(
                    (acct["total_equity"] - settings.starting_capital)
                    / settings.starting_capital * 100
                ),
            )
            session.add(snapshot)
        session.commit()
        logger.info("Daily snapshots: %d strategies recorded", len(accounts))
    except Exception:
        logger.exception("Snapshot job failed")
    finally:
        session.close()

    _persist_account_state()


def _persist_account_state() -> None:
    """Sync in-memory arena accounts to the strategy_accounts DB table."""
    if not _arena:
        return
    session = _session_factory()
    try:
        accounts = _arena.get_all_accounts()
        for name, acct in accounts.items():
            existing = (
                session.query(StrategyAccount)
                .filter_by(strategy_name=name)
                .first()
            )
            if existing:
                existing.cash_balance = acct["cash"]
                existing.open_positions_value = acct["open_positions_value"]
                existing.total_equity = acct["total_equity"]
                existing.peak_equity = acct["peak_equity"]
                existing.drawdown_pct = acct["drawdown_pct"]
                existing.is_paused = acct["is_paused"]
            else:
                session.add(StrategyAccount(
                    strategy_name=name,
                    starting_capital=settings.starting_capital,
                    cash_balance=acct["cash"],
                    open_positions_value=acct["open_positions_value"],
                    total_equity=acct["total_equity"],
                    peak_equity=acct["peak_equity"],
                    drawdown_pct=acct["drawdown_pct"],
                    is_paused=acct["is_paused"],
                ))
        session.commit()
    except Exception:
        logger.exception("Failed to persist account state")
    finally:
        session.close()
