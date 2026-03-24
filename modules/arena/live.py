"""
EdgeFinder Arena Live Integration
===================================
Hooks the arena engine into the scheduler for live market operation.
Fetches data via DataService, feeds strategies, executes trades.

Called by the scheduler jobs — does not own the scheduler lifecycle.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from config import settings
from modules.arena.engine import ArenaEngine
from modules.arena.virtual_account import Position
from modules.strategies.base import StrategyRegistry, MarketRegime
from modules.scanner import get_active_watchlist, run_scan, score_lynch, score_burry
from modules.signals import fetch_price_history
from modules.database import (
    ArenaTradeLog,
    ArenaSnapshot,
    get_session,
)

logger = logging.getLogger(__name__)

# ── MODULE STATE ─────────────────────────────────────────────

_engine: Optional[ArenaEngine] = None
_arena_status = {
    "running": False,
    "last_signal_check": None,
    "last_position_monitor": None,
    "last_scan": None,
    "errors": [],
}


def get_arena_engine() -> Optional[ArenaEngine]:
    """Get the current arena engine instance."""
    return _engine


def get_arena_status() -> dict:
    """Get arena status for API."""
    if _engine is None:
        return {"running": False, "strategies": 0}

    status = _engine.get_status()
    status.update({
        "last_signal_check": _arena_status["last_signal_check"],
        "last_position_monitor": _arena_status["last_position_monitor"],
        "last_scan": _arena_status["last_scan"],
        "recent_errors": _arena_status["errors"][-5:],
    })
    return status


# ── INITIALIZATION ───────────────────────────────────────────

def init_arena() -> ArenaEngine:
    """Initialize the arena engine with Lynch and Burry strategies.

    Called once at startup. Loads strategies from registry, sets up
    watchlists from existing scan data.
    """
    global _engine

    # Import strategy modules to trigger registration
    import modules.strategies.lynch  # noqa: F401
    import modules.strategies.burry  # noqa: F401

    _engine = ArenaEngine(
        starting_capital=settings.ARENA_STARTING_CAPITAL_PER_STRATEGY,
        max_positions_per_strategy=settings.ARENA_MAX_POSITIONS_PER_STRATEGY,
        drawdown_pause_pct=settings.ARENA_DRAWDOWN_PAUSE_PCT,
    )

    _engine.load_strategies()
    _arena_status["running"] = True

    # Populate watchlists from existing scan data
    _refresh_watchlists()

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

    # Build scored stock dicts for each strategy
    scored = []
    for stock in watchlist:
        scored.append({
            "ticker": stock.get("ticker", ""),
            "lynch_score": stock.get("lynch_score", 0),
            "burry_score": stock.get("burry_score", 0),
            "composite_score": stock.get("composite_score", 0),
            "lynch_category": stock.get("lynch_category", ""),
            "fcf_yield": stock.get("fcf_yield"),
            "price_to_tangible_book": stock.get("price_to_tangible_book"),
        })

    # Set watchlists on each strategy
    for name, strategy in _engine.strategies.items():
        if hasattr(strategy, "set_watchlist"):
            strategy.set_watchlist(scored)


# ── LIVE JOBS ────────────────────────────────────────────────

def arena_signal_check() -> list[dict]:
    """Fetch bars for watchlist, feed to all strategies, execute signals.

    Called every 15 min during market hours by the scheduler.
    """
    if _engine is None:
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

        # Fallback: if no watchlists set, use default tickers
        if not all_tickers:
            all_tickers = set(settings.SCANNER_DEFAULT_TICKERS[:30])

        # Fetch price data
        bars = {}
        volumes = {}
        for ticker in all_tickers:
            try:
                df = fetch_price_history(ticker, period="1y", interval="1d")
                if df is not None and not df.empty:
                    bars[ticker] = df
                    vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 1_000_000
                    volumes[ticker] = float(vol) if vol and vol > 0 else 1_000_000
            except Exception as e:
                logger.debug(f"Arena: Failed to fetch {ticker}: {e}")

        if not bars:
            logger.warning("Arena: No price data fetched")
            return []

        logger.info(f"Arena: Fetched bars for {len(bars)}/{len(all_tickers)} tickers")

        # Run signal check across all strategies
        executed = _engine.run_signal_check(
            bars=bars,
            volumes=volumes,
            price_source="yfinance",
        )

        # Persist executed trades to DB
        for trade in executed:
            _save_arena_trade(trade)

        _arena_status["last_signal_check"] = datetime.now(timezone.utc).isoformat()

        if executed:
            logger.info(f"ARENA: {len(executed)} trades executed")
        else:
            logger.info("ARENA: No trades this cycle")

        return executed

    except Exception as e:
        logger.error(f"Arena signal check failed: {e}")
        _arena_status["errors"].append(f"signal_check: {e}")
        return []


def arena_position_monitor() -> list[dict]:
    """Fetch current prices for open positions, check stops/targets.

    Called every 5 min during market hours by the scheduler.
    """
    if _engine is None:
        return []

    try:
        # Gather all tickers with open positions
        tickers_needed = set()
        for account in _engine.accounts.values():
            for pos in account.positions.values():
                tickers_needed.add(pos.ticker)

        if not tickers_needed:
            return []

        # Fetch current prices
        prices = {}
        for ticker in tickers_needed:
            price = _fetch_current_price(ticker)
            if price:
                prices[ticker] = price

        if not prices:
            return []

        # Monitor positions
        closed = _engine.monitor_positions(prices, price_source="yfinance")

        # Persist closed trades
        for trade in closed:
            _update_arena_trade_closed(trade)

        _arena_status["last_position_monitor"] = datetime.now(timezone.utc).isoformat()

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
                price = _fetch_current_price(position.ticker)
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
    """Run nightly fundamental scan and refresh strategy watchlists."""
    if _engine is None:
        return

    try:
        logger.info("ARENA: Nightly scan starting...")
        tickers = sorted(set(settings.SCANNER_DEFAULT_TICKERS))
        run_scan(tickers=tickers, save_to_db=True)
        _refresh_watchlists()
        _arena_status["last_scan"] = datetime.now(timezone.utc).isoformat()
        logger.info("ARENA: Nightly scan complete, watchlists refreshed")

    except Exception as e:
        logger.error(f"Arena nightly scan failed: {e}")
        _arena_status["errors"].append(f"nightly_scan: {e}")


# ── HELPERS ──────────────────────────────────────────────────

def _fetch_current_price(ticker: str) -> Optional[float]:
    """Fetch latest price for a ticker."""
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        if price and price > 0:
            return round(float(price), 4)
    except Exception as e:
        logger.debug(f"Price fetch failed for {ticker}: {e}")
    return None


def _save_arena_trade(trade: dict) -> None:
    """Persist an arena trade execution to the database."""
    try:
        session = get_session()
        record = ArenaTradeLog(
            trade_id=trade.get("trade_id"),
            strategy_name=trade.get("strategy_name"),
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
        )
        session.add(record)
        session.commit()
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
