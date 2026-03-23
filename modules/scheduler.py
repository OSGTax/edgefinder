"""
EdgeFinder Scheduler: Automated Market-Hours Operation
=======================================================
Runs the full trading pipeline on a schedule using APScheduler:

    7:00 AM - 6:00 PM ET (Mon-Fri):
        Every 15 min: Signal check → sentiment gate → paper trade
        Every  5 min: Monitor open positions (stops, targets, trailing)
    3:50 PM ET: Close all day trades
    4:05 PM ET: Save daily account snapshot
    4:30 PM ET: Run nightly fundamental scan

Integrates with the FastAPI dashboard via lifespan hook.
Recovers state from DB on cold starts (Render process restarts).
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytz
import yfinance as yf
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from modules.database import (
    Trade as TradeRecord,
    AccountSnapshot,
    get_session,
    init_db,
)
from modules.journal import TradeJournal
from modules.scanner import get_active_watchlist, run_scan
from modules.sentiment import gate_trade
from modules.signals import scan_watchlist
from modules.trader import AccountState, PaperTrader, Position

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")

# ── MODULE STATE ─────────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None
_trader: Optional[PaperTrader] = None
_journal: Optional[TradeJournal] = None
_status = {
    "running": False,
    "jobs_run": 0,
    "last_signal_check": None,
    "last_position_monitor": None,
    "last_scan": None,
    "open_positions": 0,
    "errors": [],
}


# ── SCHEDULER LIFECYCLE ──────────────────────────────────────

def create_scheduler() -> BackgroundScheduler:
    """Create and configure the APScheduler with all trading jobs."""
    executors = {"default": ThreadPoolExecutor(1)}
    scheduler = BackgroundScheduler(timezone=ET, executors=executors)

    # Signal check: every 15 min, 7AM-6PM ET, Mon-Fri
    scheduler.add_job(
        job_signal_check,
        CronTrigger(minute="*/15", hour="7-17", day_of_week="mon-fri", timezone=ET),
        id="signal_check",
        replace_existing=True,
    )

    # Position monitor: every 5 min, 7AM-6PM ET, Mon-Fri
    scheduler.add_job(
        job_position_monitor,
        CronTrigger(
            minute=f"*/{settings.POSITION_MONITOR_INTERVAL_MINUTES}",
            hour="7-17",
            day_of_week="mon-fri",
            timezone=ET,
        ),
        id="position_monitor",
        replace_existing=True,
    )

    # Close day trades: 3:50 PM ET, Mon-Fri
    scheduler.add_job(
        job_close_day_trades,
        CronTrigger(hour=15, minute=50, day_of_week="mon-fri", timezone=ET),
        id="close_day_trades",
        replace_existing=True,
    )

    # Account snapshot: 4:05 PM ET, Mon-Fri
    scheduler.add_job(
        job_account_snapshot,
        CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone=ET),
        id="account_snapshot",
        replace_existing=True,
    )

    # Nightly scan: 4:30 PM ET, Mon-Fri
    scheduler.add_job(
        job_nightly_scan,
        CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone=ET),
        id="nightly_scan",
        replace_existing=True,
    )

    return scheduler


def start_scheduler() -> None:
    """Start the scheduler. Safe to call multiple times (idempotent)."""
    global _scheduler, _trader, _journal

    if _scheduler and _scheduler.running:
        logger.info("Scheduler already running")
        return

    # Initialize journal
    _journal = TradeJournal()

    # Restore trader state from DB (cold-start recovery)
    _trader = restore_trader_state()
    logger.info(
        f"Trader restored: ${_trader.account.cash:.2f} cash, "
        f"{_trader.account.open_position_count} open positions"
    )

    # Check if watchlist is empty and run initial scan if needed
    watchlist = get_active_watchlist()
    if not watchlist:
        logger.info("No watchlist data — running initial scan in background...")
        thread = threading.Thread(target=_initial_scan, daemon=True)
        thread.start()

    # Create and start scheduler
    _scheduler = create_scheduler()
    _scheduler.start()
    _status["running"] = True

    logger.info("=" * 60)
    logger.info("EDGEFINDER SCHEDULER STARTED")
    logger.info(f"Timezone: US/Eastern")
    logger.info(f"Signal check: every {settings.SIGNAL_CHECK_INTERVAL_MINUTES}m, 7AM-6PM")
    logger.info(f"Position monitor: every {settings.POSITION_MONITOR_INTERVAL_MINUTES}m")
    logger.info(f"Nightly scan: {settings.SCANNER_RUN_TIME} ET")
    for job in _scheduler.get_jobs():
        logger.info(f"  Job: {job.id} → next run: {job.next_run_time}")
    logger.info("=" * 60)


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler and save state."""
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _status["running"] = False
        logger.info("Scheduler stopped")

    # Save any open positions
    if _trader:
        for position in _trader.account.positions.values():
            save_open_position(position)
        _trader.save_account_snapshot()


def get_scheduler_status() -> dict:
    """Return current scheduler state for API."""
    jobs = []
    if _scheduler and _scheduler.running:
        for job in _scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })

    positions = []
    if _trader:
        for p in _trader.account.positions.values():
            positions.append({
                "ticker": p.ticker,
                "entry_price": p.entry_price,
                "shares": p.shares,
                "stop_loss": p.stop_loss,
                "target": p.target,
                "trade_type": p.trade_type,
                "trailing_stop": p.trailing_stop,
            })

    return {
        "running": _status["running"],
        "jobs": jobs,
        "jobs_run": _status["jobs_run"],
        "last_signal_check": _status["last_signal_check"],
        "last_position_monitor": _status["last_position_monitor"],
        "last_scan": _status["last_scan"],
        "open_positions": positions,
        "account": {
            "cash": round(_trader.account.cash, 2) if _trader else settings.STARTING_CAPITAL,
            "total_value": round(_trader.account.total_value, 2) if _trader else settings.STARTING_CAPITAL,
            "drawdown_pct": round(_trader.account.drawdown_pct, 4) if _trader else 0.0,
        },
        "recent_errors": _status["errors"][-5:],
    }


# ── SCHEDULED JOBS ───────────────────────────────────────────

def job_signal_check() -> None:
    """
    Scan watchlist for signals → sentiment gate → execute trades.
    Runs every 15 min during extended market hours.
    """
    try:
        logger.info("JOB: Signal check starting...")

        watchlist = get_active_watchlist()
        if not watchlist:
            logger.warning("Signal check: No active watchlist")
            return

        tickers = [w["ticker"] for w in watchlist]
        ticker_info = {w["ticker"]: w for w in watchlist}

        # Scan for signals
        signals = scan_watchlist(tickers, save_to_db=True)
        logger.info(f"Signal check: {len(signals)} tradeable signals from {len(tickers)} tickers")

        for signal in signals:
            ticker = signal.ticker
            info = ticker_info.get(ticker, {})

            # Sentiment gate
            action, adjusted_confidence, sentiment_result = gate_trade(
                ticker, signal.confidence
            )

            if action == "BLOCK":
                _journal.log_skipped_signal(
                    ticker=ticker,
                    signal_type=signal.signal_type,
                    trade_type=signal.trade_type,
                    confidence=signal.confidence,
                    reason=f"Sentiment BLOCK: {sentiment_result.reason}",
                    indicators=signal.indicators,
                )
                continue

            # Execute trade
            position = _trader.execute_signal(
                ticker=ticker,
                signal_type=signal.signal_type,
                trade_type=signal.trade_type,
                entry_price=signal.price,
                confidence=adjusted_confidence,
                sector=info.get("sector", ""),
                fundamental_score=info.get("composite_score", 0),
                technical_signals=signal.indicators,
                news_sentiment=sentiment_result.avg_compound,
                sentiment_action=action,
            )

            if position:
                save_open_position(position)
                logger.info(f"TRADE OPENED: {ticker} {signal.trade_type} @ ${signal.price:.2f}")
            else:
                can, reason = _trader.can_trade(
                    ticker, signal.trade_type, info.get("sector", "")
                )
                _journal.log_skipped_signal(
                    ticker=ticker,
                    signal_type=signal.signal_type,
                    trade_type=signal.trade_type,
                    confidence=adjusted_confidence,
                    reason=f"Trader rejected: {reason}",
                    indicators=signal.indicators,
                )

        _status["last_signal_check"] = datetime.now(timezone.utc).isoformat()
        _status["jobs_run"] += 1
        _status["open_positions"] = _trader.account.open_position_count

    except Exception as e:
        logger.error(f"Signal check failed: {e}")
        _status["errors"].append(f"signal_check: {e}")


def job_position_monitor() -> None:
    """
    Check current prices for open positions. Trigger stops/targets.
    Runs every 5 min during extended market hours.
    """
    try:
        if not _trader or not _trader.account.positions:
            return

        positions = list(_trader.account.positions.values())
        logger.info(f"JOB: Monitoring {len(positions)} open positions...")

        for position in positions:
            try:
                price = _fetch_current_price(position.ticker)
                if price is None:
                    continue

                action = _trader.update_price(position.trade_id, price)
                if action:
                    # Close the position
                    result = _trader.close_position(
                        position.trade_id, price, exit_reason=action
                    )
                    if result:
                        _journal.log_trade(result, position)
                        _trader.save_trade(result, position)
                        _remove_open_position(position.trade_id)
                        logger.info(
                            f"POSITION CLOSED: {position.ticker} | {action} | "
                            f"P&L: ${result.pnl_dollars:+.2f}"
                        )
            except Exception as e:
                logger.warning(f"Error monitoring {position.ticker}: {e}")

        _status["last_position_monitor"] = datetime.now(timezone.utc).isoformat()
        _status["open_positions"] = _trader.account.open_position_count

    except Exception as e:
        logger.error(f"Position monitor failed: {e}")
        _status["errors"].append(f"position_monitor: {e}")


def job_close_day_trades() -> None:
    """
    Close all DAY-type positions at end of day.
    Runs at 3:50 PM ET.
    """
    try:
        if not _trader:
            return

        day_positions = [
            p for p in _trader.account.positions.values()
            if p.trade_type == "DAY"
        ]

        if not day_positions:
            logger.info("JOB: No day trades to close")
            return

        logger.info(f"JOB: Closing {len(day_positions)} day trades...")

        for position in day_positions:
            try:
                price = _fetch_current_price(position.ticker)
                if price is None:
                    price = position.entry_price  # Fallback to entry if price unavailable

                result = _trader.close_position(
                    position.trade_id, price, exit_reason="END_OF_DAY"
                )
                if result:
                    _journal.log_trade(result, position)
                    _trader.save_trade(result, position)
                    _remove_open_position(position.trade_id)
                    logger.info(
                        f"DAY TRADE CLOSED: {position.ticker} | "
                        f"P&L: ${result.pnl_dollars:+.2f}"
                    )
            except Exception as e:
                logger.warning(f"Error closing day trade {position.ticker}: {e}")

        _status["jobs_run"] += 1

    except Exception as e:
        logger.error(f"Close day trades failed: {e}")
        _status["errors"].append(f"close_day_trades: {e}")


def job_account_snapshot() -> None:
    """
    Save daily account snapshot for equity curve.
    Runs at 4:05 PM ET.
    """
    try:
        if not _trader:
            return

        _trader.save_account_snapshot()
        _status["jobs_run"] += 1
        logger.info(
            f"JOB: Account snapshot saved — "
            f"${_trader.account.total_value:.2f} total"
        )

    except Exception as e:
        logger.error(f"Account snapshot failed: {e}")
        _status["errors"].append(f"account_snapshot: {e}")


def job_nightly_scan() -> None:
    """
    Run fundamental scan with curated ticker list.
    Runs at 4:30 PM ET.
    """
    try:
        logger.info("JOB: Nightly scan starting...")
        tickers = sorted(set(settings.SCANNER_DEFAULT_TICKERS))
        results = run_scan(tickers=tickers, save_to_db=True)
        _status["last_scan"] = datetime.now(timezone.utc).isoformat()
        _status["jobs_run"] += 1
        logger.info(f"JOB: Nightly scan complete — {len(results)} stocks on watchlist")

    except Exception as e:
        logger.error(f"Nightly scan failed: {e}")
        _status["errors"].append(f"nightly_scan: {e}")


# ── STATE PERSISTENCE (cold-start recovery) ──────────────────

def restore_trader_state() -> PaperTrader:
    """
    Rebuild PaperTrader from database after a process restart.

    Queries OPEN trades for positions and latest AccountSnapshot
    for cash/peak values. Returns a fresh trader if no state found.
    """
    try:
        session = get_session()

        # Restore open positions
        open_trades = session.query(TradeRecord).filter(
            TradeRecord.status == "OPEN"
        ).all()

        # Restore account state from latest snapshot
        snapshot = session.query(AccountSnapshot).order_by(
            AccountSnapshot.date.desc()
        ).first()

        session.close()

        if not open_trades and not snapshot:
            logger.info("No saved state — starting fresh trader")
            return PaperTrader()

        # Build account state
        cash = snapshot.cash if snapshot else settings.STARTING_CAPITAL
        peak = snapshot.peak_value if snapshot else settings.STARTING_CAPITAL

        account = AccountState(cash=cash, peak_value=peak)

        # Rebuild positions
        for trade in open_trades:
            position = Position(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                direction=trade.direction or "LONG",
                trade_type=trade.trade_type or "DAY",
                entry_price=trade.entry_price or 0,
                shares=trade.shares or 0,
                stop_loss=trade.stop_loss or 0,
                target=trade.target or 0,
                entry_time=trade.entry_time or datetime.now(timezone.utc),
                fundamental_score=trade.fundamental_score or 0,
                technical_signals=trade.technical_signals or {},
                news_sentiment=trade.news_sentiment or 0,
                confidence_score=trade.confidence_score or 0,
                high_water_mark=trade.entry_price or 0,
            )
            account.positions[trade.trade_id] = position
            # Deduct cost from cash (positions are funded)
            account.cash -= position.cost_basis

        trader = PaperTrader(account=account)
        logger.info(
            f"Restored {len(open_trades)} open positions, "
            f"${account.cash:.2f} cash"
        )
        return trader

    except Exception as e:
        logger.error(f"Failed to restore trader state: {e}")
        return PaperTrader()


def save_open_position(position: Position) -> None:
    """Save an open position to the database for cold-start recovery."""
    try:
        session = get_session()

        # Check if already exists
        existing = session.query(TradeRecord).filter(
            TradeRecord.trade_id == position.trade_id
        ).first()

        if existing:
            session.close()
            return

        record = TradeRecord(
            trade_id=position.trade_id,
            ticker=position.ticker,
            direction=position.direction,
            trade_type=position.trade_type,
            entry_price=position.entry_price,
            shares=position.shares,
            stop_loss=position.stop_loss,
            target=position.target,
            entry_time=position.entry_time,
            status="OPEN",
            fundamental_score=position.fundamental_score,
            technical_signals=position.technical_signals,
            news_sentiment=position.news_sentiment,
            confidence_score=position.confidence_score,
        )
        session.add(record)
        session.commit()
        logger.debug(f"Saved open position: {position.ticker}")
    except Exception as e:
        logger.error(f"Failed to save open position: {e}")
        session.rollback()
    finally:
        session.close()


def _remove_open_position(trade_id: str) -> None:
    """Update a trade record from OPEN to CLOSED in the database."""
    try:
        session = get_session()
        session.query(TradeRecord).filter(
            TradeRecord.trade_id == trade_id
        ).update({"status": "CLOSED"})
        session.commit()
    except Exception as e:
        logger.error(f"Failed to update position status: {e}")
        session.rollback()
    finally:
        session.close()


# ── HELPERS ──────────────────────────────────────────────────

def _fetch_current_price(ticker: str) -> Optional[float]:
    """Fetch current price for a ticker. Lightweight call."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        price = getattr(info, "last_price", None)
        if price is None:
            # Fallback
            info_dict = stock.info
            price = info_dict.get(
                "regularMarketPrice",
                info_dict.get("currentPrice"),
            )
        return float(price) if price else None
    except Exception as e:
        logger.debug(f"Failed to fetch price for {ticker}: {e}")
        return None


def _initial_scan() -> None:
    """Run initial scan on first boot when watchlist is empty."""
    try:
        tickers = sorted(set(settings.SCANNER_DEFAULT_TICKERS))
        results = run_scan(tickers=tickers, save_to_db=True)
        _status["last_scan"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Initial scan complete: {len(results)} stocks on watchlist")
    except Exception as e:
        logger.error(f"Initial scan failed: {e}")


# ============================================================
# HUMAN_ACTION_REQUIRED
# What: Upgrade Render to Starter plan ($7/mo) for always-on operation
# Why: Free tier sleeps after 15 min idle, which stops the scheduler
# How: Render dashboard → your service → Settings → Plan → Starter
# ============================================================
