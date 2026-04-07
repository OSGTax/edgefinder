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
    TickerStrategyQualification,
    TradeRecord,
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


def _deferred_initial_scan() -> None:
    """Run initial scan in background thread so the web server can start immediately."""
    import threading

    def _scan():
        try:
            logger.info("Background initial scan starting...")
            _run_initial_scan()
            watchlists = _load_watchlists()
            if watchlists and _arena:
                _arena.set_watchlists(watchlists)
                for name, tickers in watchlists.items():
                    logger.info("Background scan: %s watchlist = %d tickers", name, len(tickers))
            else:
                logger.warning("No tickers qualified after background scan")
        except Exception:
            logger.exception("Background initial scan failed")

    thread = threading.Thread(target=_scan, daemon=True, name="initial-scan")
    thread.start()


def init_services() -> None:
    """Initialize all trading pipeline services. Called once at startup."""
    global _provider, _arena, _scheduler, _session_factory

    # Data provider — DataHub wraps Polygon + optional supplements
    try:
        polygon = PolygonDataProvider()
    except ValueError:
        logger.warning(
            "No Polygon API key — trading pipeline disabled. "
            "Set EDGEFINDER_POLYGON_API_KEY in .env"
        )
        return

    from edgefinder.core.interfaces import DataHub

    hub = DataHub(CachedDataProvider(polygon, DataCache()))

    # Future: register supplemental providers here via hub.register_supplement()

    _provider = hub

    # DB session factory
    engine = get_engine()
    _session_factory = get_session_factory(engine)

    # Arena — load strategies and set watchlist
    _arena = ArenaEngine(_provider)
    _arena.load_strategies()

    watchlists = _load_watchlists()
    if not watchlists or not _has_fundamentals():
        # First run or no scored data — scan in background so server starts fast
        logger.info("No scored tickers — launching background initial scan")
        _deferred_initial_scan()
    else:
        _arena.set_watchlists(watchlists)
        for name, tickers in watchlists.items():
            logger.info("Watchlist '%s': %d tickers", name, len(tickers))

    # Restore account state, open positions, then recalculate from trades (source of truth)
    _restore_account_state()
    _restore_open_positions()
    _recalculate_account_balances()

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
    _persist_account_state()  # Save account state before shutdown
    if _scheduler:
        _scheduler.stop()
    event_bus.clear()
    logger.info("Trading pipeline shut down")


# ── Watchlist ───────────────────────────────────────────


def _restore_account_state() -> None:
    """Restore account cash, peak equity, and pause state from DB.

    Must run BEFORE _restore_open_positions() so that the cash balance
    reflects historical P&L, not a fresh $5,000.
    """
    if not _arena or not _session_factory:
        return
    session = _session_factory()
    try:
        rows = session.query(StrategyAccount).all()
        restored = 0
        for row in rows:
            account = _arena.get_account(row.strategy_name)
            if not account:
                continue
            account.cash = row.cash_balance
            account.peak_equity = row.peak_equity
            account.is_paused = row.is_paused
            account.realized_pnl = row.realized_pnl or 0.0
            restored += 1
            logger.info(
                "Restored account state for '%s': cash=$%.2f, peak=$%.2f, paused=%s",
                row.strategy_name, row.cash_balance, row.peak_equity, row.is_paused,
            )
        if restored:
            logger.info("Restored account state for %d strategies", restored)
    except Exception:
        logger.exception("Failed to restore account state")
    finally:
        session.close()


def _restore_open_positions() -> None:
    """Restore open positions from DB into in-memory arena accounts.

    Without this, every restart resets position counts to 0, bypassing
    max_open_positions and allowing duplicate trades.

    NOTE: Does NOT deduct cash — the persisted cash balance from
    _restore_account_state() already reflects open position costs.
    """
    if not _arena or not _session_factory:
        return
    session = _session_factory()
    try:
        from edgefinder.trading.account import Position

        open_trades = (
            session.query(TradeRecord)
            .filter(TradeRecord.status == "OPEN")
            .all()
        )
        restored = 0
        for tr in open_trades:
            account = _arena.get_account(tr.strategy_name)
            if not account:
                continue
            # Skip if already have this position (shouldn't happen, but be safe)
            if account.get_position(tr.symbol):
                continue
            position = Position(
                symbol=tr.symbol,
                shares=tr.shares,
                entry_price=tr.entry_price,
                stop_loss=tr.stop_loss,
                target=tr.target,
                direction=tr.direction,
                trade_type=tr.trade_type,
                entry_time=tr.entry_time,
                trade_id=tr.trade_id,
            )
            # Add position only — cash already reflects this from _restore_account_state()
            account.positions.append(position)
            restored += 1
        if restored:
            logger.info("Restored %d open positions from DB", restored)
    except Exception:
        logger.exception("Failed to restore open positions")
    finally:
        session.close()


def _recalculate_account_balances() -> None:
    """Recalculate cash and realized P&L from the trades table (source of truth).

    Formula: correct_cash = starting_capital
                          + sum(pnl_dollars for CLOSED trades)
                          - sum(entry_price * shares for OPEN trades)

    This self-heals any corruption from restarts where the DB had stale
    cash values (e.g., positions opened after last persist but before shutdown).
    """
    if not _arena or not _session_factory:
        return
    from sqlalchemy import func

    session = _session_factory()
    try:
        for name in _arena.get_strategy_names():
            account = _arena.get_account(name)
            if not account:
                continue

            # Realized P&L from closed trades
            realized = (
                session.query(func.coalesce(func.sum(TradeRecord.pnl_dollars), 0.0))
                .filter(
                    TradeRecord.strategy_name == name,
                    TradeRecord.status == "CLOSED",
                )
                .scalar()
            )

            # Cost basis of open positions from trades table
            open_cost = (
                session.query(
                    func.coalesce(
                        func.sum(TradeRecord.entry_price * TradeRecord.shares), 0.0
                    )
                )
                .filter(
                    TradeRecord.strategy_name == name,
                    TradeRecord.status == "OPEN",
                )
                .scalar()
            )

            correct_cash = round(account.starting_capital + realized - open_cost, 2)
            account.realized_pnl = round(realized, 2)

            if abs(account.cash - correct_cash) > 0.01:
                logger.warning(
                    "Cash correction for '%s': DB had $%.2f, correct is $%.2f (diff $%.2f)",
                    name, account.cash, correct_cash, account.cash - correct_cash,
                )
                account.cash = correct_cash

            # Update peak equity based on corrected values
            equity = account.total_equity
            if equity > account.peak_equity:
                account.peak_equity = equity

            logger.info(
                "Account '%s': cash=$%.2f, realized=$%.2f, open_cost=$%.2f",
                name, correct_cash, realized, open_cost,
            )
    except Exception:
        logger.exception("Failed to recalculate account balances")
    finally:
        session.close()


def _run_initial_scan() -> None:
    """Run a real batched scan on first startup to populate the research tab.

    Uses the same logic as the nightly scan — fetches the full Polygon universe,
    picks today's batch, and scans ~1,000 tickers.
    """
    session = _session_factory()
    try:
        batch_count = settings.scanner_batch_count
        batch_index = datetime.now().weekday()
        if batch_index >= batch_count:
            batch_index = 0  # Weekend fallback to Monday's batch

        universe = _provider.get_ticker_universe()
        sorted_universe = sorted(universe)
        batch = sorted_universe[batch_index::batch_count]

        logger.info(
            "Initial scan batch %d/%d: %d tickers (of %d total)",
            batch_index + 1, batch_count, len(batch), len(universe),
        )

        scanner = FundamentalScanner(_provider, session)
        results = scanner.run(tickers=batch, batch_index=batch_index)
        qualified = sum(1 for s in results if s.qualifying_strategies)
        logger.info(
            "Initial scan complete: %d scored, %d qualified",
            len(results), qualified,
        )
    except Exception:
        logger.exception("Initial scan failed")
    finally:
        session.close()


def _load_watchlist() -> list[str]:
    """Load active tickers from DB (populated by scanner).

    Legacy function — used by /api/research/active endpoint.
    For strategy-specific watchlists, use _load_watchlists().
    """
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


def _load_watchlists() -> dict[str, list[str]]:
    """Load per-strategy qualified tickers from DB.

    Returns dict mapping strategy_name -> list of qualified ticker symbols.
    Each strategy only sees stocks that passed its own qualifies_stock() check.
    """
    session = _session_factory()
    try:
        rows = (
            session.query(
                TickerStrategyQualification.strategy_name,
                TickerStrategyQualification.symbol,
            )
            .filter(TickerStrategyQualification.qualified == True)
            .order_by(
                TickerStrategyQualification.strategy_name,
                TickerStrategyQualification.score.desc().nullslast(),
            )
            .all()
        )
        result: dict[str, list[str]] = {}
        for strategy_name, symbol in rows:
            result.setdefault(strategy_name, []).append(symbol)
        return result
    finally:
        session.close()


def _has_fundamentals() -> bool:
    """Check if active tickers have fundamental data."""
    from edgefinder.db.models import Fundamental

    session = _session_factory()
    try:
        count = (
            session.query(Ticker)
            .join(Fundamental, Ticker.id == Fundamental.ticker_id)
            .filter(Ticker.is_active == True)
            .count()
        )
        return count > 0
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
        _persist_account_state()  # Save immediately so DB reflects new cash

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
        _persist_account_state()  # Save immediately so DB reflects new cash

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
    """Called at 6:15 PM ET on weekdays.

    Scans 1/5 of the stock universe each night (batched by day of week).
    Qualified tickers accumulate across the week so the watchlist grows
    to cover the full market over 5 trading days.
    """
    if not _provider:
        return

    batch_count = settings.scanner_batch_count
    batch_index = datetime.now().weekday()  # Mon=0 ... Fri=4
    if batch_index >= batch_count:
        logger.info("Weekend — skipping nightly scan")
        return

    session = _session_factory()
    try:
        # Get full universe and slice for today's batch
        universe = _provider.get_ticker_universe()
        sorted_universe = sorted(universe)
        batch = sorted_universe[batch_index::batch_count]

        logger.info(
            "Nightly scan batch %d/%d: %d tickers (of %d total)",
            batch_index + 1, batch_count, len(batch), len(universe),
        )

        scanner = FundamentalScanner(_provider, session)
        results = scanner.run(tickers=batch, batch_index=batch_index)

        # Reload per-strategy watchlists (all batches accumulated)
        watchlists = _load_watchlists()
        if watchlists and _arena:
            _arena.set_watchlists(watchlists)
            total = sum(len(t) for t in watchlists.values())
            logger.info(
                "Nightly scan: %d qualified this batch, %d total across %d strategies",
                sum(1 for s in results if s.qualifying_strategies),
                total, len(watchlists),
            )
            for name, tickers in watchlists.items():
                logger.info("  %s: %d tickers", name, len(tickers))
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
                existing.realized_pnl = acct["realized_pnl"]
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
                    realized_pnl=acct["realized_pnl"],
                    is_paused=acct["is_paused"],
                ))
        session.commit()
    except Exception:
        logger.exception("Failed to persist account state")
    finally:
        session.close()
