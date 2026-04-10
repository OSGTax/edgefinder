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
from edgefinder.core.models import TickerFundamentals
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
from edgefinder.scanner.strategy_scanner import StrategyScanner
from edgefinder.scheduler.scheduler import EdgeFinderScheduler
from edgefinder.trading.arena import ArenaEngine
from edgefinder.trading.journal import TradeJournal

logger = logging.getLogger(__name__)

# ── Module-level singletons ─────────────────────────────

_provider: CachedDataProvider | None = None
_arena: ArenaEngine | None = None
_scheduler: EdgeFinderScheduler | None = None
_session_factory = None
_plan_access: dict[str, bool] = {}


def get_plan_access() -> dict[str, bool]:
    """Get plan access probe results for API display."""
    return _plan_access


def get_arena() -> ArenaEngine | None:
    return _arena


def get_scheduler() -> EdgeFinderScheduler | None:
    return _scheduler


# ── Initialization ──────────────────────────────────────


QUICK_START_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "UNH", "DIS", "BAC",
    "XOM", "ABBV", "PFE", "COST", "MRK", "AVGO", "KO", "PEP", "TMO",
    "CSCO", "ABT", "ADBE", "CRM", "NKE", "INTC", "AMD", "QCOM", "TXN",
    "LOW", "NFLX", "AMGN", "GS", "CAT", "BLK", "SYK", "ISRG", "DE",
    "LRCX", "AMAT", "MU", "MRVL", "NOW",
]


def _deferred_initial_scan() -> None:
    """Run initial scan in background: quick-start batch first, then full universe."""
    import threading

    def _scan():
        try:
            # Phase 1: Quick-start with top 50 liquid stocks so trading begins fast
            logger.info("Quick-start scan: %d tickers...", len(QUICK_START_TICKERS))
            _run_scan_batch(QUICK_START_TICKERS)
            watchlists = _load_watchlists()
            if watchlists and _arena:
                _arena.set_watchlists(watchlists)
                _populate_fundamentals_cache()
                for name, tickers in watchlists.items():
                    logger.info("Quick-start: %s = %d tickers", name, len(tickers))
            logger.info("Quick-start complete — strategies can now trade")

            # Phase 2: Full universe scan (takes much longer)
            logger.info("Starting full universe scan...")
            _run_initial_scan()
            watchlists = _load_watchlists()
            if watchlists and _arena:
                _arena.set_watchlists(watchlists)
                _populate_fundamentals_cache()
                for name, tickers in watchlists.items():
                    logger.info("Full scan: %s = %d tickers", name, len(tickers))
        except Exception:
            logger.exception("Background initial scan failed")

    thread = threading.Thread(target=_scan, daemon=True, name="initial-scan")
    thread.start()


def _run_scan_batch(tickers: list[str]) -> None:
    """Run per-strategy scan on a specific list of tickers."""
    from edgefinder.strategies.base import StrategyRegistry

    for strategy in StrategyRegistry.get_instances():
        session = _session_factory()
        try:
            scanner = StrategyScanner(strategy, _provider, session)
            results = scanner.run(tickers=tickers)
            qualified = sum(1 for r in results if r.qualified)
            logger.info("[%s] batch scan: %d qualified of %d", strategy.name, qualified, len(results))
        except Exception:
            logger.exception("Batch scan failed for '%s'", strategy.name)
        finally:
            session.close()


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
        _populate_fundamentals_cache()
        for name, tickers in watchlists.items():
            logger.info("Watchlist '%s': %d tickers", name, len(tickers))

    # Restore account state, open positions, then recalculate from trades (source of truth)
    _restore_account_state()
    _restore_open_positions()
    _restore_close_cooldowns()
    _recalculate_account_balances()

    # Persist any corrections from _recalculate_account_balances back to DB
    # (cash adjustments, auto-unpause, etc.) so they survive even if no trades
    # fire before the next restart.
    _persist_account_state()

    # Diagnostic: log final account state for each strategy after restore + recalc
    if _arena:
        for name in _arena.get_strategy_names():
            account = _arena.get_account(name)
            if account:
                logger.info(
                    "Startup state [%s]: cash=$%.2f, positions=%d, paused=%s, "
                    "drawdown=%.1f%%, peak_equity=$%.2f",
                    name, account.cash, len(account.positions), account.is_paused,
                    account.drawdown_pct * 100, account.peak_equity,
                )

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
        sector_rotation_fn=_sector_rotation_job,
        news_accumulate_fn=_news_accumulate_job,
        dividend_split_fn=_dividend_split_job,
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


def _restore_close_cooldowns() -> None:
    """Restore per-ticker re-entry cooldowns from the trades table.

    The in-memory _last_close_per_ticker map is wiped on every restart,
    which previously allowed the signal check to immediately reopen a
    ticker after a deploy interrupted an active cooldown. This rebuilds
    the map from the most recent CLOSED trade per (strategy, symbol)
    so cooldowns survive restarts.

    Only loads closes within the cooldown window — older closes are
    irrelevant since their cooldown has already expired.
    """
    if not _arena or not _session_factory:
        return
    from datetime import timedelta
    from sqlalchemy import func

    session = _session_factory()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=settings.ticker_reentry_cooldown_minutes
        )
        # Get the most recent close per (strategy_name, symbol) since cutoff
        rows = (
            session.query(
                TradeRecord.strategy_name,
                TradeRecord.symbol,
                func.max(TradeRecord.exit_time).label("last_exit"),
            )
            .filter(
                TradeRecord.status == "CLOSED",
                TradeRecord.exit_time.is_not(None),
                TradeRecord.exit_time >= cutoff,
            )
            .group_by(TradeRecord.strategy_name, TradeRecord.symbol)
            .all()
        )
        restored = 0
        for strategy_name, symbol, last_exit in rows:
            account = _arena.get_account(strategy_name)
            if not account:
                continue
            # Ensure tz-aware UTC for comparison with datetime.now(timezone.utc)
            if last_exit.tzinfo is None:
                last_exit = last_exit.replace(tzinfo=timezone.utc)
            account._last_close_per_ticker[symbol] = last_exit
            restored += 1
        if restored:
            logger.info(
                "Restored %d re-entry cooldowns from trades table (cutoff=%dm)",
                restored, settings.ticker_reentry_cooldown_minutes,
            )
    except Exception:
        logger.exception("Failed to restore close cooldowns")
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

            # CRITICAL: if cash is negative, account is over-leveraged
            # Auto-reset: close all open trades in DB and restart fresh
            if account.cash < 0:
                logger.error(
                    "CRITICAL: '%s' has negative cash $%.2f — auto-resetting account. "
                    "Marking all open trades as CANCELLED and resetting to $%.2f.",
                    name, account.cash, account.starting_capital,
                )
                # Cancel all open trades for this strategy
                open_trades = (
                    session.query(TradeRecord)
                    .filter(TradeRecord.strategy_name == name, TradeRecord.status == "OPEN")
                    .all()
                )
                for trade in open_trades:
                    trade.status = "CANCELLED"
                    trade.exit_reason = "ACCOUNT_RESET"
                session.commit()
                # Reset account
                account.cash = account.starting_capital
                account.positions.clear()
                account.realized_pnl = round(realized, 2)
                account.is_paused = False
                account.peak_equity = account.starting_capital
                logger.info("Account '%s' reset to $%.2f", name, account.starting_capital)

            # Update peak equity based on corrected values
            equity = account.total_equity
            if equity > account.peak_equity:
                account.peak_equity = equity

            # Auto-unpause heuristic: if an account is paused but its drawdown
            # is well below the circuit breaker (< half the threshold), the
            # pause is stale (recovered, or set by an old bug that no longer
            # applies). Clear it so the strategy can resume trading. Always
            # log a WARNING so this is visible — never silent.
            unpause_threshold = settings.drawdown_circuit_breaker_pct / 2
            if account.is_paused and account.drawdown_pct < unpause_threshold:
                logger.warning(
                    "Auto-unpausing '%s': drawdown %.1f%% is below threshold %.1f%% "
                    "(half of %.1f%% circuit breaker). Account had been paused "
                    "but has since recovered.",
                    name, account.drawdown_pct * 100, unpause_threshold * 100,
                    settings.drawdown_circuit_breaker_pct * 100,
                )
                account.is_paused = False

            logger.info(
                "Account '%s': cash=$%.2f, realized=$%.2f, open_cost=$%.2f",
                name, correct_cash, realized, open_cost,
            )
    except Exception:
        logger.exception("Failed to recalculate account balances")
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


def _run_initial_scan() -> None:
    """Run unified scan on first startup to populate watchlists.

    Uses UnifiedScanner: fetches each ticker's fundamentals ONCE and
    qualifies against all 4 strategies in parallel. Pre-filtered to the
    top N most-liquid stocks by dollar volume to keep total scan time
    in the seconds-to-minutes range instead of hours.
    """
    from edgefinder.scanner.unified_scanner import UnifiedScanner
    from edgefinder.strategies.base import StrategyRegistry

    tickers = _resolve_scan_universe()
    logger.info("Initial scan: %d tickers (top dollar-volume universe)", len(tickers))

    strategies = list(StrategyRegistry.get_instances())
    scanner = UnifiedScanner(strategies, _provider, _session_factory)
    try:
        summary = scanner.run(tickers)
        logger.info("Initial scan results: %s", summary)
    except Exception:
        logger.exception("Initial scan failed")


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
    """Load per-strategy qualified tickers from DB, ranked by score.

    Returns dict mapping strategy_name -> list of qualified ticker symbols.
    Each strategy only sees stocks that passed its own qualifies_stock() check.
    Lists are ordered by score (highest first) and capped at top N per strategy.
    """
    max_per_strategy = settings.scanner_max_watchlist_per_strategy
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
            tickers = result.setdefault(strategy_name, [])
            if len(tickers) < max_per_strategy:
                tickers.append(symbol)
        return result
    finally:
        session.close()


def _populate_fundamentals_cache() -> None:
    """Build fundamentals cache from DB and push to arena for re-qualification."""
    if not _arena or not _session_factory:
        return
    from edgefinder.db.models import Fundamental
    session = _session_factory()
    try:
        rows = (
            session.query(Ticker, Fundamental)
            .join(Fundamental, Ticker.id == Fundamental.ticker_id)
            .filter(Ticker.is_active == True)
            .all()
        )
        cache = {}
        for ticker, fund in rows:
            cache[ticker.symbol] = TickerFundamentals(
                symbol=ticker.symbol,
                company_name=ticker.company_name,
                sector=ticker.sector,
                market_cap=ticker.market_cap,
                price=ticker.last_price,
                earnings_growth=fund.earnings_growth,
                revenue_growth=fund.revenue_growth,
                peg_ratio=fund.peg_ratio,
                fcf_yield=fund.fcf_yield,
                current_ratio=fund.current_ratio,
                debt_to_equity=fund.debt_to_equity,
                price_to_tangible_book=fund.price_to_tangible_book,
                ev_to_ebitda=fund.ev_to_ebitda,
                short_interest=fund.short_interest,
            )
        _arena.set_fundamentals_cache(cache)
    except Exception:
        logger.exception("Failed to populate fundamentals cache")
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

            # Capture rich trade context from DB (no extra API calls)
            _capture_trade_context(session, trade)

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


def _capture_trade_context(session, trade) -> None:
    """Capture rich market context at trade time for later AI analysis.

    Pulls from DB (pre-accumulated data) — no extra API calls.
    """
    from edgefinder.db.models import TickerNews, TickerDividend, TradeContext

    try:
        # Recent news for this ticker (from accumulated DB)
        news_rows = (
            session.query(TickerNews)
            .filter_by(symbol=trade.symbol)
            .order_by(TickerNews.id.desc())
            .limit(5)
            .all()
        )
        recent_news = [
            {"title": n.title, "published": n.published_utc, "publisher": n.publisher_name}
            for n in news_rows
        ]

        # Upcoming dividends
        divs = (
            session.query(TickerDividend)
            .filter_by(symbol=trade.symbol)
            .order_by(TickerDividend.id.desc())
            .limit(3)
            .all()
        )
        dividends = [
            {"ex_date": d.ex_dividend_date, "amount": d.cash_amount}
            for d in divs
        ]

        # Sector rotation data (if available)
        sector_prices = {}
        if _sector_rotation_data:
            sector_prices = {r["symbol"]: r.get("quadrant") for r in _sector_rotation_data}

        # Related tickers (from cached fundamentals if available)
        related = None
        if hasattr(trade, "technical_signals") and trade.technical_signals:
            related = trade.technical_signals.get("related_tickers")

        context = TradeContext(
            trade_id=trade.trade_id,
            recent_news=recent_news,
            sector_prices=sector_prices,
            related_tickers=related,
            short_interest=None,  # populated from fundamentals if available
            dividends=dividends,
            indicators=trade.technical_signals,
        )
        session.add(context)
        session.commit()
    except Exception:
        logger.debug("Trade context capture failed for %s", trade.trade_id)


# ── Market Holiday Check ───────────────────────────────

_holidays_cache: list[str] = []  # cached list of holiday date strings
_holidays_last_refresh: float = 0


def _is_market_holiday() -> bool:
    """Check if today is a market holiday. Refreshes cache weekly."""
    import time
    global _holidays_cache, _holidays_last_refresh

    now = time.time()
    if not _holidays_cache or (now - _holidays_last_refresh) > 604800:  # 7 days
        if _provider:
            try:
                holidays = _provider.get_market_holidays()
                _holidays_cache = [
                    h["date"] for h in holidays
                    if h.get("status") == "closed"
                ]
                _holidays_last_refresh = now
            except Exception:
                pass

    today_str = datetime.now().strftime("%Y-%m-%d")
    return today_str in _holidays_cache


# ── Scheduler Job Callbacks ─────────────────────────────


def _signal_check_job() -> None:
    """Called every 5 min during market hours. Skips market holidays."""
    if not _arena:
        logger.warning("Signal check skipped: arena not initialized")
        return
    if _is_market_holiday():
        logger.info("Signal check skipped — market holiday")
        return
    try:
        trades = _arena.run_signal_check()
        if trades:
            logger.info("Signal check: %d trades opened", len(trades))
        else:
            logger.info("Signal check: no new trades")
    except Exception:
        logger.exception("Signal check failed")


def _position_monitor_job() -> None:
    """Called every 5 min during market hours. Skips market holidays."""
    if not _arena:
        return
    if _is_market_holiday():
        logger.debug("Position monitor skipped — market holiday")
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

    Runs the unified scanner: fetches each ticker's fundamentals once and
    qualifies against all 4 strategies in parallel. Pre-filtered to top
    N most-liquid by dollar volume so the full nightly scan completes
    in seconds instead of hours.
    """
    if not _provider:
        return

    from edgefinder.scanner.unified_scanner import UnifiedScanner
    from edgefinder.strategies.base import StrategyRegistry

    weekday = datetime.now().weekday()
    if weekday >= 5:
        logger.info("Weekend — skipping nightly scan")
        return

    tickers = _resolve_scan_universe()
    logger.info("Nightly scan: %d tickers (top dollar-volume universe)", len(tickers))

    strategies = list(StrategyRegistry.get_instances())
    scanner = UnifiedScanner(strategies, _provider, _session_factory)
    try:
        summary = scanner.run(tickers)
        logger.info("Nightly scan results: %s", summary)
    except Exception:
        logger.exception("Nightly scan failed")
        return

    # Reload per-strategy watchlists into the live arena
    watchlists = _load_watchlists()
    if watchlists and _arena:
        _arena.set_watchlists(watchlists)
        _populate_fundamentals_cache()
        total = sum(len(t) for t in watchlists.values())
        logger.info(
            "Nightly scan complete: %d total tickers across %d strategies",
            total, len(watchlists),
        )
        for name, tickers_for_strategy in watchlists.items():
            logger.info("  %s: %d tickers", name, len(tickers_for_strategy))
    else:
        logger.info("Nightly scan: 0 qualified across all strategies")


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
