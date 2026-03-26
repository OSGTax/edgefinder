"""
EdgeFinder Arena Live Integration
===================================
Hooks the arena engine into the scheduler for live market operation.
Fetches data via DataService (Alpaca → FMP → yfinance fallback).

Called by the scheduler jobs — does not own the scheduler lifecycle.
"""

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

from config import settings
from modules.arena.engine import ArenaEngine
from modules.arena.virtual_account import Position
from modules.strategies.base import StrategyRegistry, MarketRegime
from modules.scanner import get_active_watchlist, run_scan
from modules.database import (
    ArenaTradeLog,
    ArenaSnapshot,
    get_session,
)
from modules.utils import to_eastern, compute_trade_hash

logger = logging.getLogger(__name__)


def _job_timeout(seconds: int = 120, status_key: str | None = None):
    """Decorator to prevent scheduler jobs from hanging indefinitely.

    Runs the job in a daemon thread with a timeout. If the job doesn't
    complete within `seconds`, logs an error and returns an empty list
    so the scheduler thread is freed for the next run.

    If *status_key* is provided, the corresponding entry in
    ``_arena_status`` is updated at the START of the call (so the
    dashboard always shows when the job last ran, even on timeout).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Always record that the job was attempted
            if status_key:
                _arena_status[status_key] = to_eastern(datetime.now(timezone.utc))

            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                msg = (
                    f"Job {func.__name__} timed out after {seconds}s — "
                    f"freeing scheduler thread"
                )
                logger.error(msg)
                _arena_status["errors"].append(msg)
                _arena_status["last_signal_result"] = (
                    f"Timed out after {seconds}s"
                )
                return []
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator


# ── MODULE STATE ─────────────────────────────────────────────

_engine: Optional[ArenaEngine] = None
_data_service = None  # services.data_service.DataService instance
_arena_status = {
    "last_signal_check": None,
    "last_signal_result": None,
    "last_position_monitor": None,
    "last_scan": None,
    "errors": deque(maxlen=50),
}


def get_arena_engine() -> Optional[ArenaEngine]:
    """Get the current arena engine instance."""
    return _engine


def get_arena_status() -> dict:
    """Get arena status for API."""
    if _engine is None:
        status = {"running": False, "strategies": 0}
    else:
        status = _engine.get_status()

    status.update({
        "last_signal_check": _arena_status["last_signal_check"],
        "last_signal_result": _arena_status["last_signal_result"],
        "last_position_monitor": _arena_status["last_position_monitor"],
        "last_scan": _arena_status["last_scan"],
        "recent_errors": list(_arena_status["errors"])[-5:],
        "data_source": "DataService (Alpaca/FMP/yfinance)",
    })
    return status


# ── INITIALIZATION ───────────────────────────────────────────

def init_arena() -> ArenaEngine:
    """Initialize the arena engine with Lynch and Burry strategies.

    Called once at startup. Loads strategies from registry, sets up
    watchlists from existing scan data, and initializes DataService.
    """
    global _engine, _data_service

    # Initialize DataService for market data
    try:
        from services.data_service import DataService
        _data_service = DataService()
        alpaca_ok = _data_service.alpaca is not None
        fmp_ok = _data_service.fmp is not None
        logger.info(
            f"DataService initialized: Alpaca={'yes' if alpaca_ok else 'no'}, "
            f"FMP={'yes' if fmp_ok else 'no'}, yfinance=fallback"
        )
    except Exception as e:
        logger.warning(f"DataService init failed, will use yfinance fallback: {e}")
        _data_service = None

    # Import strategy modules to trigger registration
    _strategy_modules = [
        "modules.strategies.lynch",
        "modules.strategies.burry",
        "modules.strategies.alpha",
        "modules.strategies.bravo",
        "modules.strategies.charlie",
        "modules.strategies.delta",
        "modules.strategies.echo",
        "modules.strategies.foxtrot",
        "modules.strategies.golf",
        "modules.strategies.hotel",
        "modules.strategies.india",
        "modules.strategies.juliet",
        "modules.strategies.kilo",
        "modules.strategies.lima",
        "modules.strategies.mike",
        "modules.strategies.november",
        "modules.strategies.oscar",
        "modules.strategies.papa",
    ]
    import importlib
    for mod_name in _strategy_modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            logger.error(f"Failed to import strategy {mod_name}: {e}")

    _engine = ArenaEngine(
        starting_capital=settings.ARENA_STARTING_CAPITAL_PER_STRATEGY,
        max_positions_per_strategy=settings.ARENA_MAX_POSITIONS_PER_STRATEGY,
        drawdown_pause_pct=settings.ARENA_DRAWDOWN_PAUSE_PCT,
    )

    _engine.load_strategies()

    # Populate watchlists from existing scan data
    try:
        _refresh_watchlists()
    except Exception as e:
        logger.error(f"Failed to refresh watchlists: {e}")

    # Restore open positions and account state from database
    _restore_state()

    logger.info(
        f"Arena initialized: {len(_engine.strategies)} strategies, "
        f"${settings.ARENA_STARTING_CAPITAL_PER_STRATEGY} each"
    )
    return _engine


def _refresh_watchlists() -> None:
    """Update strategy watchlists from the current database watchlist."""
    if _engine is None:
        return

    watchlist = get_active_watchlist()
    if not watchlist:
        logger.info("Arena: No watchlist data yet — strategies will trade all tickers")
        return

    scored = []
    seen_tickers: set[str] = set()
    for stock in watchlist:
        ticker = stock.get("ticker", "")
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        scored.append({
            "ticker": ticker,
            "lynch_score": stock.get("lynch_score", 0),
            "burry_score": stock.get("burry_score", 0),
            "composite_score": stock.get("composite_score", 0),
            "lynch_category": stock.get("lynch_category", ""),
            "fcf_yield": stock.get("fcf_yield"),
            "price_to_tangible_book": stock.get("price_to_tangible_book"),
        })

    for name, strategy in _engine.strategies.items():
        if hasattr(strategy, "set_watchlist"):
            strategy.set_watchlist(scored)


def reset_arena() -> dict:
    """Cancel all OPEN arena trades and reset all virtual accounts to starting capital."""
    global _engine

    if _engine is None:
        return {"error": "Arena not initialized", "reset": False}

    now = datetime.now(timezone.utc)
    session = get_session()
    try:
        # DB: mark all OPEN trades as CANCELLED
        open_trades = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.status == "OPEN"
        ).all()

        cancelled_count = len(open_trades)
        for trade in open_trades:
            trade.status = "CANCELLED"
            trade.exit_reason = "ARENA_RESET"
            trade.exit_timestamp = now

        session.commit()

        # In-memory: reset every VirtualAccount
        strategies_reset = 0
        for name, account in _engine.accounts.items():
            account.reset_account()
            strategies_reset += 1

        logger.info(
            f"ARENA RESET: {cancelled_count} trades cancelled, "
            f"{strategies_reset} accounts reset to "
            f"${settings.ARENA_STARTING_CAPITAL_PER_STRATEGY}"
        )

        return {
            "reset": True,
            "trades_cancelled": cancelled_count,
            "strategies_reset": strategies_reset,
            "timestamp": to_eastern(now),
        }

    except Exception as e:
        session.rollback()
        logger.error(f"Arena reset failed: {e}")
        return {"error": str(e), "reset": False}
    finally:
        session.close()


def _restore_state() -> None:
    """Restore full account state from database after restart.

    On every Render deploy, VirtualAccount instances are created fresh with
    starting capital. This function restores:
    1. Realized P&L from CLOSED trades (replenish cash before deducting opens)
    2. Closed trades list (so realized_pnl, win_rate properties work)
    3. PDT day trade counter (from recent DAY-type closed trades)
    4. OPEN positions (deduct cost from cash, rebuild in-memory positions)
    5. Current prices for open positions (so unrealized P&L is visible)
    6. Peak equity (from latest ArenaSnapshot per strategy)

    This ensures account balances survive restarts, deploys, and code changes.
    """
    if _engine is None:
        return

    from datetime import timedelta

    session = get_session()
    try:
        # ── Phase 1: Restore realized P&L + closed trades list ──
        # This MUST run before open positions so cash includes profits
        # from closed trades that may have funded later opens.
        total_realized = 0.0
        total_closed_count = 0
        pdt_cutoff = datetime.now(timezone.utc) - timedelta(
            days=settings.PDT_WINDOW_DAYS
        )

        for strategy_name, account in _engine.accounts.items():
            closed_trades = session.query(ArenaTradeLog).filter(
                ArenaTradeLog.strategy_name == strategy_name,
                ArenaTradeLog.status == "CLOSED",
            ).order_by(ArenaTradeLog.exit_timestamp.asc()).all()

            realized_pnl = 0.0
            for trade in closed_trades:
                pnl = trade.pnl_dollars or 0.0
                realized_pnl += pnl

                # Populate in-memory closed_trades list so
                # realized_pnl property and win_rate work correctly
                account.closed_trades.append({
                    "trade_id": trade.trade_id,
                    "strategy_name": trade.strategy_name,
                    "ticker": trade.ticker,
                    "direction": trade.direction or "LONG",
                    "trade_type": trade.trade_type or "SWING",
                    "entry_price": trade.execution_price,
                    "exit_price": trade.exit_price,
                    "shares": trade.shares,
                    "stop_loss": trade.stop_loss,
                    "target": trade.target,
                    "entry_time": trade.execution_timestamp,
                    "exit_time": trade.exit_timestamp,
                    "pnl_dollars": pnl,
                    "pnl_percent": trade.pnl_percent or 0.0,
                    "r_multiple": trade.r_multiple or 0.0,
                    "exit_reason": trade.exit_reason or "",
                    "confidence": trade.confidence or 0.0,
                    "slippage_applied": trade.slippage or 0.0,
                    "metadata": trade.extra_data or {},
                })

                # Restore PDT day trade counter from recent DAY closes
                # Normalize both sides to naive UTC to avoid
                # offset-naive vs offset-aware comparison errors
                # (PostgreSQL may return naive, SQLite may return aware)
                if (
                    trade.trade_type == "DAY"
                    and trade.exit_timestamp
                    and trade.exit_timestamp.replace(tzinfo=None)
                        >= pdt_cutoff.replace(tzinfo=None)
                ):
                    account._day_trades.append(trade.exit_timestamp)

            # Add realized P&L to cash before open positions are deducted
            account.cash += realized_pnl
            total_realized += realized_pnl
            total_closed_count += len(closed_trades)

            if realized_pnl != 0:
                logger.info(
                    f"Restored realized P&L for {strategy_name}: "
                    f"${realized_pnl:+.2f} from {len(closed_trades)} "
                    f"closed trades"
                )

        # ── Phase 2: Restore OPEN positions ─────────────────────
        open_trades = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.status == "OPEN"
        ).all()

        restored_positions = 0
        for trade in open_trades:
            strategy_name = trade.strategy_name
            if strategy_name not in _engine.accounts:
                logger.warning(
                    f"Restore: strategy '{strategy_name}' not loaded, "
                    f"skipping trade {trade.trade_id}"
                )
                continue

            account = _engine.accounts[strategy_name]
            position = Position(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                direction=trade.direction or "LONG",
                trade_type=trade.trade_type or "SWING",
                entry_price=trade.execution_price,
                shares=trade.shares,
                stop_loss=trade.stop_loss,
                target=trade.target,
                entry_time=trade.execution_timestamp,
                confidence=trade.confidence or 0.5,
                slippage_applied=trade.slippage or 0.0,
                metadata=trade.extra_data or {},
            )

            # Deduct cost from cash and add position — always restore
            # since these were validly opened during live execution
            cost = position.entry_price * position.shares
            if cost > account.cash:
                logger.warning(
                    f"Restore: cash deficit for {trade.ticker} "
                    f"in {strategy_name} (need ${cost:.2f}, "
                    f"have ${account.cash:.2f}) — restoring anyway"
                )
            account.cash -= cost
            position.high_water_mark = position.entry_price
            position.last_known_price = position.entry_price
            account.positions[position.trade_id] = position
            restored_positions += 1
            logger.info(
                f"Restored position: {trade.ticker} ({strategy_name}) "
                f"{trade.shares}x @ ${trade.execution_price:.2f}"
            )

        # ── Phase 2b: Update restored positions with current prices ──
        if restored_positions > 0:
            tickers_needed: set[str] = set()
            for account in _engine.accounts.values():
                for pos in account.positions.values():
                    tickers_needed.add(pos.ticker)

            updated_count = 0
            for ticker in tickers_needed:
                price, src = _get_price(ticker)
                if price:
                    for account in _engine.accounts.values():
                        for pos in account.positions.values():
                            if pos.ticker == ticker:
                                account.update_position_price(pos.trade_id, price)
                                updated_count += 1

            logger.info(
                f"Updated {updated_count} positions across "
                f"{len(tickers_needed)} tickers with current prices"
            )

        # ── Phase 3: Restore peak equity from snapshots ─────────
        for strategy_name, account in _engine.accounts.items():
            latest_snap = session.query(ArenaSnapshot).filter(
                ArenaSnapshot.strategy_name == strategy_name
            ).order_by(ArenaSnapshot.timestamp.desc()).first()

            if latest_snap and latest_snap.peak_equity:
                account.peak_equity = latest_snap.peak_equity

        logger.info(
            f"State restored: {restored_positions} open positions, "
            f"{total_closed_count} closed trades "
            f"(${total_realized:+.2f} realized P&L) "
            f"across {len(_engine.accounts)} strategies"
        )

    except Exception as e:
        logger.error(f"State restoration failed: {e}")
    finally:
        session.close()


# ── DATA FETCHING ────────────────────────────────────────────

def _get_yf_session():
    """Create a yfinance-compatible session with curl_cffi for server compatibility."""
    try:
        from curl_cffi.requests import Session as CffiSession
        return CffiSession(impersonate="chrome")
    except ImportError:
        return None


def _get_bars(tickers: list[str], days_back: int = 365) -> dict:
    """Fetch OHLCV bars via DataService with title-case column mapping.

    signals.py compute_indicators() expects: Open, High, Low, Close, Volume.
    DataService returns lowercase. We rename here.
    """
    bars = {}
    volumes = {}

    if _data_service:
        # Use DataService (Alpaca → yfinance fallback, with caching)
        raw_bars = _data_service.get_multi_bars(
            tickers, timeframe="1Day", days_back=days_back
        )
        for ticker, df in raw_bars.items():
            if df is not None and not df.empty:
                # Rename to title case for signals.py compatibility
                col_map = {}
                for col in df.columns:
                    if col.lower() in ("open", "high", "low", "close", "volume"):
                        col_map[col] = col.capitalize()
                if col_map:
                    df = df.rename(columns=col_map)
                bars[ticker] = df
                if "Volume" in df.columns:
                    vol = df["Volume"].tail(20).mean()
                    volumes[ticker] = float(vol) if vol and vol > 0 else 1_000_000
                else:
                    volumes[ticker] = 1_000_000
    else:
        # Fallback: direct yfinance
        import yfinance as yf
        sess = _get_yf_session()
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker, session=sess)
                df = stock.history(period="1y", interval="1d")
                if df is not None and not df.empty:
                    bars[ticker] = df
                    vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 1_000_000
                    volumes[ticker] = float(vol) if vol and vol > 0 else 1_000_000
            except Exception as e:
                logger.debug(f"yfinance fallback failed for {ticker}: {e}")

    return bars, volumes


def _get_price(ticker: str) -> tuple[Optional[float], str]:
    """Fetch latest price via DataService. Returns (price, source)."""
    if _data_service:
        price = _data_service.get_latest_price(ticker)
        if price:
            source = "alpaca" if _data_service.alpaca else "cache"
            return price, source

    # Fallback
    try:
        import yfinance as yf
        t = yf.Ticker(ticker, session=_get_yf_session())
        price = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        if price and price > 0:
            return round(float(price), 4), "yfinance"
    except Exception as e:
        logger.debug(f"Price fetch failed for {ticker}: {e}")

    return None, "none"


# ── LIVE JOBS ────────────────────────────────────────────────

@_job_timeout(seconds=120, status_key="last_signal_check")
def arena_signal_check() -> list[dict]:
    """Fetch bars for watchlist, feed to all strategies, execute signals.

    Called every 15 min during market hours by the scheduler.
    """
    now = to_eastern(datetime.now(timezone.utc))

    if _engine is None:
        _arena_status["last_signal_check"] = now
        _arena_status["last_signal_result"] = "Engine not initialized"
        logger.warning("Arena not initialized")
        return []

    try:
        logger.info("ARENA: Signal check starting...")

        # Gather all tickers across all strategies
        all_tickers = set()
        for name, strategy in _engine.strategies.items():
            wl = strategy.get_watchlist()
            if wl:
                all_tickers.update(wl)

        strategy_count = len(_engine.strategies)

        if not all_tickers:
            logger.warning(
                "ARENA: No watchlist tickers from strategies — "
                "falling back to default tickers. Run a scanner first."
            )
            all_tickers = set(settings.SCANNER_DEFAULT_TICKERS[:30])

        ticker_count = len(all_tickers)

        # Fetch price data via DataService
        bars, volumes = _get_bars(list(all_tickers))

        if not bars:
            _arena_status["last_signal_check"] = now
            _arena_status["last_signal_result"] = (
                f"No price data fetched (0/{ticker_count} tickers)"
            )
            logger.warning("ARENA: No price data fetched")
            return []

        source = "alpaca" if (_data_service and _data_service.alpaca) else "yfinance"
        logger.info(
            f"ARENA: Fetched bars for {len(bars)}/{ticker_count} tickers "
            f"via {source}"
        )

        # Run signal check across all strategies
        executed = _engine.run_signal_check(
            bars=bars,
            volumes=volumes,
            price_source=source,
        )

        # Persist executed trades to DB
        for trade in executed:
            _save_arena_trade(trade)

        _arena_status["last_signal_result"] = (
            f"{len(executed)} trades from {strategy_count} strategies "
            f"({len(bars)}/{ticker_count} tickers)"
        )

        if executed:
            logger.info(f"ARENA: {len(executed)} trades executed")
        else:
            logger.info("ARENA: No trades this cycle")

        return executed

    except Exception as e:
        _arena_status["last_signal_check"] = now
        _arena_status["last_signal_result"] = f"Error: {e}"
        logger.error(f"Arena signal check failed: {e}")
        _arena_status["errors"].append(f"signal_check: {e}")
        return []


@_job_timeout(seconds=60, status_key="last_position_monitor")
def arena_position_monitor() -> list[dict]:
    """Fetch current prices for open positions, check stops/targets.

    Called every 5 min during market hours by the scheduler.
    """
    if _engine is None:
        return []

    try:
        tickers_needed = set()
        for account in _engine.accounts.values():
            for pos in account.positions.values():
                tickers_needed.add(pos.ticker)

        if not tickers_needed:
            return []

        # Fetch current prices via DataService
        prices = {}
        source = "unknown"
        for ticker in tickers_needed:
            price, src = _get_price(ticker)
            if price:
                prices[ticker] = price
                source = src

        if not prices:
            return []

        closed = _engine.monitor_positions(prices, price_source=source)

        for trade in closed:
            _update_arena_trade_closed(trade)

        if closed:
            logger.info(f"ARENA: {len(closed)} positions closed")

        return closed

    except Exception as e:
        logger.error(f"Arena position monitor failed: {e}")
        _arena_status["errors"].append(f"position_monitor: {e}")
        return []


def arena_close_day_trades() -> list[dict]:
    """Close all DAY-type positions across all strategies at EOD."""
    if _engine is None:
        return []

    try:
        closed = []
        for name, account in _engine.accounts.items():
            day_positions = [
                (tid, p) for tid, p in account.positions.items()
                if p.trade_type == "DAY"
            ]

            for trade_id, position in day_positions:
                price, _ = _get_price(position.ticker)
                if price is None:
                    price = position.last_known_price or position.entry_price

                result = account.close_position(
                    trade_id=trade_id,
                    exit_price=price,
                    exit_reason="END_OF_DAY",
                )
                if result:
                    closed.append(result)
                    _update_arena_trade_closed(result)

        if closed:
            logger.info(f"ARENA: Closed {len(closed)} day trades at EOD")
        return closed

    except Exception as e:
        logger.error(f"Arena close day trades failed: {e}")
        _arena_status["errors"].append(f"close_day_trades: {e}")
        return []


def arena_snapshot() -> list[dict]:
    """Take equity snapshots for all strategies and persist to DB."""
    if _engine is None:
        return []

    try:
        snapshots = _engine.take_snapshots()
        for snap in snapshots:
            _save_arena_snapshot(snap)
        logger.info(f"ARENA: Saved {len(snapshots)} strategy snapshots")
        return snapshots

    except Exception as e:
        logger.error(f"Arena snapshot failed: {e}")
        _arena_status["errors"].append(f"snapshot: {e}")
        return []


def arena_nightly_scan() -> None:
    """Run nightly fundamental scan using sector rotation and refresh watchlists."""
    if _engine is None:
        return

    try:
        from modules.scanner import get_todays_sectors
        sectors = get_todays_sectors()
        if sectors:
            logger.info(f"ARENA: Nightly scan — today's sectors: {sectors}")
            run_scan(sectors=sectors, save_to_db=True)
        else:
            logger.info("ARENA: Weekend — running full universe scan")
            run_scan(tickers=None, save_to_db=True)

        _refresh_watchlists()
        _arena_status["last_scan"] = to_eastern(datetime.now(timezone.utc))
        logger.info("ARENA: Nightly scan complete, watchlists refreshed")

    except Exception as e:
        logger.error(f"Arena nightly scan failed: {e}")
        _arena_status["errors"].append(f"nightly_scan: {e}")


# ── DB PERSISTENCE ───────────────────────────────────────────

def _save_arena_trade(trade: dict) -> None:
    """Persist an arena trade execution to the database with integrity hash."""
    try:
        session = get_session()

        # Get the previous trade's hash for chain linking
        prev_trade = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.integrity_hash != None  # noqa: E711
        ).order_by(ArenaTradeLog.sequence_num.desc()).first()

        prev_hash = prev_trade.integrity_hash if prev_trade else ""
        next_seq = (prev_trade.sequence_num + 1) if prev_trade and prev_trade.sequence_num else 1

        # Compute integrity hash
        integrity_hash = compute_trade_hash(trade, prev_hash)

        record = ArenaTradeLog(
            trade_id=trade.get("trade_id"),
            strategy_name=trade.get("strategy_name"),
            strategy_version=trade.get("strategy_version"),
            ticker=trade.get("ticker"),
            action=trade.get("action"),
            direction="LONG",
            trade_type=trade.get("trade_type"),
            signal_price=trade.get("signal_price"),
            execution_price=trade.get("execution_price"),
            slippage=trade.get("slippage"),
            shares=trade.get("shares"),
            stop_loss=trade.get("stop_loss"),
            target=trade.get("target"),
            confidence=trade.get("confidence"),
            signal_timestamp=trade.get("signal_timestamp"),
            execution_timestamp=trade.get("execution_timestamp"),
            price_source=trade.get("price_source"),
            bar_data_at_decision=trade.get("bar_data_at_decision"),
            market_regime=trade.get("market_regime"),
            signal_overlap=trade.get("signal_overlap", 0),
            position_overlap=trade.get("position_overlap", 0),
            status="OPEN",
            extra_data=trade.get("metadata"),
            sequence_num=next_seq,
            integrity_hash=integrity_hash,
        )
        session.add(record)
        session.commit()
        logger.info(
            f"Trade saved: {trade.get('ticker')} @ "
            f"{to_eastern(trade.get('execution_timestamp'))} ET "
            f"(chain #{next_seq}, hash={integrity_hash[:12]}...)"
        )
    except Exception as e:
        logger.error(f"Failed to save arena trade: {e}")
    finally:
        session.close()


def _update_arena_trade_closed(result: dict) -> None:
    """Update an arena trade record when position is closed."""
    try:
        session = get_session()
        record = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.trade_id == result.get("trade_id")
        ).first()
        if record:
            record.exit_price = result.get("exit_price")
            record.exit_timestamp = result.get("exit_time")
            record.pnl_dollars = result.get("pnl_dollars")
            record.pnl_percent = result.get("pnl_percent")
            record.r_multiple = result.get("r_multiple")
            record.exit_reason = result.get("exit_reason")
            record.status = "CLOSED"
            session.commit()
    except Exception as e:
        logger.error(f"Failed to update arena trade: {e}")
    finally:
        session.close()


def _save_arena_snapshot(snap: dict) -> None:
    """Persist an arena equity snapshot."""
    try:
        session = get_session()
        record = ArenaSnapshot(
            strategy_name=snap.get("strategy_name"),
            timestamp=snap.get("timestamp", datetime.now(timezone.utc)),
            cash=snap.get("cash"),
            positions_value=snap.get("positions_value"),
            total_equity=snap.get("total_equity"),
            peak_equity=snap.get("peak_equity"),
            drawdown_pct=snap.get("drawdown_pct"),
            open_positions=snap.get("open_positions"),
            realized_pnl=snap.get("realized_pnl"),
            unrealized_pnl=snap.get("unrealized_pnl"),
            total_return_pct=snap.get("total_return_pct"),
            is_paused=snap.get("is_paused", False),
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.error(f"Failed to save arena snapshot: {e}")
    finally:
        session.close()
