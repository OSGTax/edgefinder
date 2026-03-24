"""
EdgeFinder Arena Live Integration
===================================
Hooks the arena engine into the scheduler for live market operation.
Fetches data via DataService (Alpaca → FMP → yfinance fallback).

Called by the scheduler jobs — does not own the scheduler lifecycle.
"""

import logging
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)

# ── MODULE STATE ─────────────────────────────────────────────

_engine: Optional[ArenaEngine] = None
_data_service = None  # services.data_service.DataService instance
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

    for name, strategy in _engine.strategies.items():
        if hasattr(strategy, "set_watchlist"):
            strategy.set_watchlist(scored)


def _restore_state() -> None:
    """Restore open positions and account state from database after restart.

    On every Render deploy, VirtualAccount instances are created fresh with
    starting capital. This function loads OPEN trades from ArenaTradeLog back
    into the in-memory accounts so balances and positions survive restarts.
    """
    if _engine is None:
        return

    session = get_session()
    try:
        open_trades = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.status == "OPEN"
        ).all()

        restored = 0
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

            # Deduct cost from cash and add position
            cost = position.entry_price * position.shares
            if cost <= account.cash:
                account.cash -= cost
                position.high_water_mark = position.entry_price
                position.last_known_price = position.entry_price
                account.positions[position.trade_id] = position
                restored += 1
                logger.info(
                    f"Restored position: {trade.ticker} ({strategy_name}) "
                    f"{trade.shares}x @ ${trade.execution_price:.2f}"
                )
            else:
                logger.warning(
                    f"Restore: insufficient cash for {trade.ticker} "
                    f"in {strategy_name} (need ${cost:.2f}, "
                    f"have ${account.cash:.2f})"
                )

        # Restore peak equity from most recent snapshot per strategy
        for strategy_name, account in _engine.accounts.items():
            latest_snap = session.query(ArenaSnapshot).filter(
                ArenaSnapshot.strategy_name == strategy_name
            ).order_by(ArenaSnapshot.timestamp.desc()).first()

            if latest_snap and latest_snap.peak_equity:
                account.peak_equity = latest_snap.peak_equity

        logger.info(
            f"State restored: {restored} open positions "
            f"across {len(_engine.accounts)} strategies"
        )

    except Exception as e:
        logger.error(f"State restoration failed: {e}")
    finally:
        session.close()


# ── DATA FETCHING ────────────────────────────────────────────

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
        # Fallback: direct yfinance with curl_cffi for server compatibility
        import yfinance as yf
        try:
            from curl_cffi.requests import Session as CffiSession
            _yf_sess = CffiSession(impersonate="chrome")
        except ImportError:
            _yf_sess = None
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker, session=_yf_sess)
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
        try:
            from curl_cffi.requests import Session as CffiSession
            _sess = CffiSession(impersonate="chrome")
        except ImportError:
            _sess = None
        t = yf.Ticker(ticker, session=_sess)
        price = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        if price and price > 0:
            return round(float(price), 4), "yfinance"
    except Exception as e:
        logger.debug(f"Price fetch failed for {ticker}: {e}")

    return None, "none"


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

        if not all_tickers:
            logger.warning(
                "ARENA: No watchlist tickers from strategies — "
                "falling back to default tickers. Run a scanner first."
            )
            all_tickers = set(settings.SCANNER_DEFAULT_TICKERS[:30])

        # Fetch price data via DataService
        bars, volumes = _get_bars(list(all_tickers))

        if not bars:
            logger.warning("ARENA: No price data fetched")
            return []

        source = "alpaca" if (_data_service and _data_service.alpaca) else "yfinance"
        logger.info(
            f"ARENA: Fetched bars for {len(bars)}/{len(all_tickers)} tickers "
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
        _arena_status["last_scan"] = datetime.now(timezone.utc).isoformat()
        logger.info("ARENA: Nightly scan complete, watchlists refreshed")

    except Exception as e:
        logger.error(f"Arena nightly scan failed: {e}")
        _arena_status["errors"].append(f"nightly_scan: {e}")


# ── DB PERSISTENCE ───────────────────────────────────────────

def _save_arena_trade(trade: dict) -> None:
    """Persist an arena trade execution to the database."""
    try:
        session = get_session()
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
