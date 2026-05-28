"""EdgeFinder v2 — Arena engine: multi-strategy orchestration.

Manages multiple strategies, each with its own virtual account, executor,
and risk manager. Computes indicators once per ticker, builds MarketData
objects, and lets strategies evaluate entries/exits against shared data.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from config.settings import settings
from edgefinder.core.events import event_bus
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import (
    Direction,
    ExitIntent,
    Trade,
    TradeIntent,
    TradeStatus,
    TradeType,
)
from edgefinder.data.indicator_engine import compute_indicators_from_bars
from edgefinder.data.market_data import (
    IndicatorHistory,
    IndicatorSnapshot,
    MarketContext,
    MarketData,
)
from edgefinder.strategies.base import StrategyRegistry, TradeNotification
from edgefinder.strategies.strategy_interface import SwingStrategy
from edgefinder.trading.account import Position, VirtualAccount
from edgefinder.trading.executor import Executor
from edgefinder.trading.risk import RiskManager

logger = logging.getLogger(__name__)


class StrategySlot:
    """Bundles a strategy instance with its account, executor, and risk manager."""

    def __init__(self, strategy: SwingStrategy, pdt_enabled: bool = False) -> None:
        self.strategy = strategy
        self.account = VirtualAccount(
            strategy_name=strategy.name,
            starting_capital=settings.starting_capital,
            pdt_enabled=pdt_enabled,
        )
        self.executor = Executor(self.account)
        self.risk_manager = RiskManager(
            risk_pct=strategy.risk_pct,
            stop_pct=strategy.stop_pct,
            target_pct=strategy.target_pct,
        )
        self.watchlist: list[str] = []


class ArenaEngine:
    """Orchestrates multiple strategies competing in isolated accounts."""

    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider
        self._slots: dict[str, StrategySlot] = {}
        self._fundamentals_cache: dict[str, object] = {}  # ticker -> TickerFundamentals
        self._indicator_histories: dict[str, IndicatorHistory] = {}  # per-ticker
        self._daily_bars_cache: dict[str, pd.DataFrame] = {}  # per-ticker
        self._sequence_num = 0
        self._prev_hash = ""

    # ── Strategy Loading ──────────────────────────────

    def load_strategies(self, pdt_config: dict[str, bool] | None = None) -> None:
        """Load all registered strategies into the arena.

        Args:
            pdt_config: Optional dict of {strategy_name: pdt_enabled}.
        """
        pdt_config = pdt_config or {}
        for strategy in StrategyRegistry.get_instances():
            pdt = pdt_config.get(strategy.name, False)
            slot = StrategySlot(strategy, pdt_enabled=pdt)
            self._slots[strategy.name] = slot
            logger.info(
                "Loaded strategy '%s' (risk=%.0f%%, target=%.0f%%, stop=%.0f%%, PDT=%s)",
                strategy.name,
                strategy.risk_pct * 100,
                strategy.target_pct * 100,
                strategy.stop_pct * 100,
                pdt,
            )

    # ── Watchlist Management ──────────────────────────

    def set_watchlists(self, watchlists: dict[str, list[str]]) -> None:
        """Set the watchlist for each strategy."""
        for name, tickers in watchlists.items():
            if name in self._slots:
                self._slots[name].watchlist = tickers

    def set_global_watchlist(self, tickers: list[str]) -> None:
        """Set the same watchlist for all strategies."""
        for slot in self._slots.values():
            slot.watchlist = list(tickers)

    def set_fundamentals_cache(self, cache: dict) -> None:
        """Set cached fundamentals for MarketData objects.

        Called by services.py after each scan. Maps ticker -> TickerFundamentals.
        """
        self._fundamentals_cache = cache
        logger.info("Fundamentals cache updated: %d tickers", len(cache))

    # ── Daily Cycle (after market close) ──────────────

    def run_daily_cycle(self) -> None:
        """Compute daily indicators for all watchlisted tickers.

        Called once after market close (6:15 PM ET). Fetches daily bars,
        computes indicators, and stores snapshots in per-ticker history buffers.
        """
        # Collect all unique tickers across all strategies
        all_tickers: set[str] = set()
        for slot in self._slots.values():
            all_tickers.update(slot.watchlist)

        # Prune stale cache entries for tickers no longer on any watchlist
        stale_bars = set(self._daily_bars_cache.keys()) - all_tickers
        for k in stale_bars:
            del self._daily_bars_cache[k]
        stale_hist = set(self._indicator_histories.keys()) - all_tickers
        for k in stale_hist:
            del self._indicator_histories[k]
        if stale_bars or stale_hist:
            logger.info(
                "Cache pruned: %d bars, %d histories removed",
                len(stale_bars), len(stale_hist),
            )

        computed = 0
        failed = 0

        for ticker in all_tickers:
            bars = self._fetch_daily_bars(ticker)
            if bars is None or bars.empty:
                failed += 1
                continue

            # Cache the bars for intraday use
            self._daily_bars_cache[ticker] = bars

            snapshot = compute_indicators_from_bars(bars)
            if snapshot is None:
                failed += 1
                continue

            # Store in per-ticker history buffer
            if ticker not in self._indicator_histories:
                self._indicator_histories[ticker] = IndicatorHistory(max_days=30)
            self._indicator_histories[ticker].add(snapshot)
            computed += 1

        logger.info(
            "Daily cycle complete: %d tickers computed, %d failed, %d total",
            computed, failed, len(all_tickers),
        )

    # ── Intraday Cycle (every 15 min during market hours) ──

    def run_intraday_cycle(
        self,
        snapshot_data: dict[str, dict],
        market_context: MarketContext,
    ) -> tuple[list[Trade], list[Trade]]:
        """Run entry and exit checks for all strategies.

        Args:
            snapshot_data: Output of get_enriched_snapshots() —
                {ticker: {"price": float, "volume": float, ...}}
            market_context: Current broad market state.

        Returns:
            (opened_trades, closed_trades)
        """
        # Step 0: Dynamic watchlist expansion for Degenerate (3x+ volume anomalies)
        self._expand_degenerate_watchlist(snapshot_data)

        # Step 1: Build MarketData objects for all tickers that have data
        market_data_map = self._build_market_data(snapshot_data, market_context)

        all_opened: list[Trade] = []
        all_closed: list[Trade] = []

        for name, slot in self._slots.items():
            if slot.account.is_paused:
                logger.info(
                    "[%s] slot paused — skipping (cash=$%.2f, drawdown=%.1f%%)",
                    name, slot.account.cash, slot.account.drawdown_pct * 100,
                )
                continue

            # Step 2: Exit checks (always run first — protect capital)
            closed = self._check_exits(slot, market_data_map, snapshot_data)
            all_closed.extend(closed)

            # Step 3: Entry checks
            opened = self._check_entries(slot, market_data_map, snapshot_data)
            all_opened.extend(opened)

        logger.info(
            "Intraday cycle complete: %d opened, %d closed across %d strategies",
            len(all_opened), len(all_closed), len(self._slots),
        )
        return all_opened, all_closed

    # ── Legacy Signal/Position Methods (backward compat) ──

    def run_signal_check(self) -> list[Trade]:
        """Legacy compatibility — runs intraday entry logic with empty context.

        For services.py callers that haven't migrated yet.
        """
        # Build minimal snapshot data from provider
        snapshot_data: dict[str, dict] = {}
        all_tickers: set[str] = set()
        for slot in self._slots.values():
            all_tickers.update(slot.watchlist)

        for ticker in all_tickers:
            try:
                price = self._provider.get_latest_price(ticker)
                if price is not None:
                    snapshot_data[ticker] = {"price": price, "volume": 0.0}
            except Exception:
                pass

        context = MarketContext()
        opened, _ = self.run_intraday_cycle(snapshot_data, context)
        return opened

    def check_positions(self) -> list[Trade]:
        """Legacy compatibility — runs exit logic with current prices.

        For services.py callers that haven't migrated yet.
        """
        snapshot_data: dict[str, dict] = {}
        for slot in self._slots.values():
            for pos in slot.account.positions:
                if pos.symbol not in snapshot_data:
                    try:
                        price = self._provider.get_latest_price(pos.symbol)
                        if price is not None:
                            snapshot_data[pos.symbol] = {"price": price, "volume": 0.0}
                    except Exception:
                        pass

        context = MarketContext()
        market_data_map = self._build_market_data(snapshot_data, context)

        all_closed: list[Trade] = []
        for name, slot in self._slots.items():
            closed = self._check_exits(slot, market_data_map, snapshot_data)
            all_closed.extend(closed)

        return all_closed

    def broadcast_market_snapshot(self, snapshot) -> None:
        """Forward a market snapshot to all strategies (no-op for SwingStrategy).

        Kept for backward compatibility with services.py.
        """
        pass

    # ── Dashboard API (MUST PRESERVE) ─────────────────

    def get_strategy(self, name: str) -> SwingStrategy | None:
        """Get a strategy instance by name."""
        slot = self._slots.get(name)
        return slot.strategy if slot else None

    def get_strategy_names(self) -> list[str]:
        return list(self._slots.keys())

    def get_account(self, strategy_name: str) -> VirtualAccount | None:
        slot = self._slots.get(strategy_name)
        return slot.account if slot else None

    def get_all_accounts(self) -> dict[str, dict]:
        return {name: slot.account.to_dict() for name, slot in self._slots.items()}

    def get_all_watched_tickers(self) -> set[str]:
        """Union of all strategies' watchlists + every open position's symbol.

        Used by the intraday cycle to seed per-ticker snapshot fallback when
        the bulk Polygon snapshot endpoint isn't available.
        """
        tickers: set[str] = set()
        for slot in self._slots.values():
            tickers.update(slot.watchlist)
            for pos in slot.account.positions:
                tickers.add(pos.symbol)
        return tickers

    def get_all_open_positions(self) -> dict[str, list[dict]]:
        result = {}
        for name, slot in self._slots.items():
            result[name] = [
                {
                    "symbol": p.symbol,
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "direction": p.direction,
                    "trade_type": p.trade_type,
                    "trade_id": p.trade_id,
                }
                for p in slot.account.positions
            ]
        return result

    # ── Private: Dynamic Watchlist ───────────────────

    def _expand_degenerate_watchlist(self, snapshot_data: dict[str, dict]) -> None:
        """Temporarily add high-volume tickers to Degenerate's watchlist.

        Scans snapshot data for stocks showing 3x+ time-normalized volume
        that aren't already on any strategy's watchlist.
        """
        degen_slot = self._slots.get("degenerate")
        if degen_slot is None:
            return

        # Collect all tickers already on any watchlist
        all_watched: set[str] = set()
        for slot in self._slots.values():
            all_watched.update(slot.watchlist)

        # Compute time-normalized volume for each ticker in snapshot
        threshold = settings.volume_anomaly_threshold  # 3.0
        added = 0
        for ticker, snap in snapshot_data.items():
            if ticker in all_watched:
                continue
            vol = snap.get("volume", 0)
            if vol <= 0:
                continue
            # We don't have avg volume for unknown tickers, so skip the ratio
            # check — the strategy's evaluate() will still check volume_ratio.
            # For dynamic additions, a raw volume > 0 with high absolute
            # volume (top of snapshot) qualifies as anomalous.
            # Full time-normalized ratio requires avg_daily_volume from indicator
            # engine, which we only have for watched tickers.
            # Instead, just add tickers with very high raw volume (top percentile).
            pass

        # Simpler approach: use snapshot_data volume compared to any available avg
        # Since we lack avg for unwatched tickers, check if they have
        # cached bars with volume data
        for ticker, snap in snapshot_data.items():
            if ticker in all_watched:
                continue
            today_vol = snap.get("volume", 0)
            if today_vol <= 0:
                continue

            # Check cached bars for avg volume
            bars = self._daily_bars_cache.get(ticker)
            if bars is not None and not bars.empty and "volume" in bars.columns:
                avg_vol = bars["volume"].mean()
                if avg_vol > 0:
                    raw_ratio = today_vol / avg_vol
                    if raw_ratio >= threshold:
                        degen_slot.watchlist.append(ticker)
                        all_watched.add(ticker)
                        added += 1

        if added:
            logger.info("Dynamic Degenerate watchlist: added %d volume anomaly tickers", added)

    # ── Private: Data Building ────────────────────────

    def _build_market_data(
        self,
        snapshot_data: dict[str, dict],
        market_context: MarketContext,
    ) -> dict[str, MarketData]:
        """Build MarketData objects for all tickers with snapshot data.

        For each ticker:
        1. Get cached daily bars
        2. If snapshot has current price, append as provisional bar
        3. Compute indicators with provisional bar
        4. Build MarketData with current snapshot, history, fundamentals, context
        """
        result: dict[str, MarketData] = {}

        # Collect all tickers we need data for
        all_tickers: set[str] = set()
        for slot in self._slots.values():
            all_tickers.update(slot.watchlist)
            for pos in slot.account.positions:
                all_tickers.add(pos.symbol)

        for ticker in all_tickers:
            snap = snapshot_data.get(ticker)
            if not snap:
                continue

            current_price = snap.get("price", 0.0)
            if not current_price:
                continue

            # Get daily bars (cached or fresh)
            bars = self._daily_bars_cache.get(ticker)
            if bars is None or bars.empty:
                bars = self._fetch_daily_bars(ticker)
                if bars is not None and not bars.empty:
                    self._daily_bars_cache[ticker] = bars

            # Compute indicators — append provisional bar if we have live data
            current_snapshot = None
            if bars is not None and not bars.empty:
                prov_bars = self._append_provisional_bar(bars, snap)
                current_snapshot = compute_indicators_from_bars(prov_bars)

            if current_snapshot is None:
                # Fall back to a minimal snapshot from raw price data
                current_snapshot = IndicatorSnapshot(
                    close=current_price,
                    open=snap.get("open", current_price),
                    high=snap.get("high", current_price),
                    low=snap.get("low", current_price),
                    volume=snap.get("volume", 0.0),
                )

            # Get history buffer
            history = self._indicator_histories.get(ticker, IndicatorHistory(max_days=30))

            # Compute time-normalized volume ratio:
            # (today_volume / avg_daily_volume) / (minutes_since_open / 390)
            avg_vol = current_snapshot.volume_avg or 0.0
            today_vol = snap.get("volume", 0.0)
            raw_ratio = today_vol / avg_vol if avg_vol > 0 else 0.0
            time_factor = self._minutes_since_market_open() / 390.0
            if time_factor > 0:
                volume_ratio = raw_ratio / time_factor
            else:
                volume_ratio = raw_ratio

            fundamentals = self._fundamentals_cache.get(ticker)

            result[ticker] = MarketData(
                ticker=ticker,
                current=current_snapshot,
                history=history,
                fundamentals=fundamentals,
                context=market_context,
                current_price=current_price,
                today_volume=today_vol,
                avg_daily_volume=avg_vol,
                volume_ratio=volume_ratio,
            )

        return result

    def _append_provisional_bar(
        self, bars: pd.DataFrame, snap: dict
    ) -> pd.DataFrame:
        """Append today's live data as a provisional bar to historical bars."""
        provisional = pd.DataFrame(
            {
                "open": [snap.get("open", snap["price"])],
                "high": [snap.get("high", snap["price"])],
                "low": [snap.get("low", snap["price"])],
                "close": [snap["price"]],
                "volume": [snap.get("volume", 0.0)],
            },
            index=pd.DatetimeIndex(
                [datetime.now(timezone.utc)], name="timestamp"
            ),
        )
        return pd.concat([bars, provisional])

    # ── Private: Entry Logic ──────────────────────────

    def _check_entries(
        self,
        slot: StrategySlot,
        market_data_map: dict[str, MarketData],
        snapshot_data: dict[str, dict],
    ) -> list[Trade]:
        """Check for new trade entries across a strategy's watchlist."""
        opened: list[Trade] = []
        evaluated = 0
        requalif_failed = 0

        for ticker in slot.watchlist:
            # Skip tickers with existing positions
            if slot.account.get_position(ticker):
                continue

            mdata = market_data_map.get(ticker)
            if mdata is None:
                continue

            evaluated += 1

            # Re-qualification gate
            cached_fund = self._fundamentals_cache.get(ticker)
            if cached_fund is not None:
                try:
                    if not slot.strategy.qualifies_stock(cached_fund):
                        requalif_failed += 1
                        continue
                except Exception:
                    logger.exception(
                        "[%s] Re-qualification raised for %s — proceeding anyway",
                        slot.strategy.name, ticker,
                    )

            # Ask strategy to evaluate
            try:
                intent = slot.strategy.evaluate(ticker, mdata)
            except Exception:
                logger.exception(
                    "[%s] evaluate() failed for %s", slot.strategy.name, ticker
                )
                continue

            if intent is None:
                continue

            # Execute the intent
            trade = self._execute_intent(slot, intent, mdata.current_price)
            if trade:
                opened.append(trade)

        if evaluated > 0:
            logger.info(
                "[%s] entry check: evaluated=%d, requalif_failed=%d, opened=%d",
                slot.strategy.name, evaluated, requalif_failed, len(opened),
            )

        return opened

    # ── Private: Exit Logic ───────────────────────────

    def _check_exits(
        self,
        slot: StrategySlot,
        market_data_map: dict[str, MarketData],
        snapshot_data: dict[str, dict],
    ) -> list[Trade]:
        """Check all open positions for exit conditions.

        Three-pass check (in priority order):
        1. RiskManager stop loss (20% — non-negotiable)
        2. RiskManager profit target
        3. Strategy-defined exit (should_exit)
        """
        closed: list[Trade] = []

        for pos in list(slot.account.positions):
            snap = snapshot_data.get(pos.symbol)
            if not snap:
                continue

            current_price = snap.get("price")
            if current_price is None:
                continue

            # Update market price on position for mark-to-market
            pos.market_price = current_price

            rm = slot.risk_manager

            mdata = market_data_map.get(pos.symbol)

            # Pass 1: Non-negotiable stop loss
            if rm.should_stop_out(pos.entry_price, current_price):
                trade = self._close_position(slot, pos, current_price, "STOP_LOSS", mdata)
                if trade:
                    closed.append(trade)
                continue

            # Pass 2: Profit target
            if rm.should_take_profit(pos.entry_price, current_price):
                trade = self._close_position(slot, pos, current_price, "TARGET_HIT", mdata)
                if trade:
                    closed.append(trade)
                continue

            # Pass 3: Strategy-defined exit
            if mdata is not None:
                try:
                    exit_intent = slot.strategy.should_exit(
                        pos.symbol, mdata, pos.entry_price
                    )
                except Exception:
                    logger.exception(
                        "[%s] should_exit() failed for %s",
                        slot.strategy.name, pos.symbol,
                    )
                    exit_intent = None

                if exit_intent is not None:
                    reason = f"STRATEGY_EXIT:{exit_intent.reasoning[:50]}"
                    trade = self._close_position(slot, pos, current_price, reason, mdata)
                    if trade:
                        closed.append(trade)

        return closed

    # ── Private: Trade Execution ──────────────────────

    def _execute_intent(
        self,
        slot: StrategySlot,
        intent: TradeIntent,
        current_price: float,
    ) -> Trade | None:
        """Convert a TradeIntent into an open position and Trade.

        Uses RiskManager for sizing, stops, and targets. Applies slippage.
        """
        rm = slot.risk_manager

        # Apply slippage
        execution_price = Executor._apply_slippage(current_price, "BUY")

        # Compute position parameters
        shares = rm.compute_shares(
            execution_price,
            slot.account.total_equity,
            available_cash=slot.account.buying_power,
        )
        if shares <= 0:
            logger.debug(
                "[%s] Intent rejected: position size is 0 for %s",
                slot.strategy.name, intent.ticker,
            )
            return None

        stop = rm.compute_stop(execution_price)
        target = rm.compute_target(execution_price)
        cost = shares * execution_price

        # Minimum position cost — reject micro-positions
        min_cost = slot.account.starting_capital * 0.01
        if cost < min_cost:
            logger.debug(
                "[%s] Position too small: %d shares @ $%.2f = $%.2f (min $%.2f)",
                slot.strategy.name, shares, execution_price, cost, min_cost,
            )
            return None

        # Check account rules
        allowed, reason = slot.account.can_open_position(
            cost, "SWING", symbol=intent.ticker
        )
        if not allowed:
            logger.info(
                "[%s] Intent rejected for %s: %s",
                slot.strategy.name, intent.ticker, reason,
            )
            return None

        # Create position
        trade_id = str(uuid.uuid4())
        position = Position(
            symbol=intent.ticker,
            shares=shares,
            entry_price=execution_price,
            stop_loss=stop,
            target=target,
            direction=intent.direction,
            trade_type="SWING",
            entry_time=datetime.now(timezone.utc),
            trade_id=trade_id,
            market_price=current_price,
        )

        slot.account.open_position(position)
        self._sequence_num += 1

        trade = Trade(
            trade_id=trade_id,
            strategy_name=intent.strategy_name,
            symbol=intent.ticker,
            direction=Direction(intent.direction),
            trade_type=TradeType.SWING,
            entry_price=execution_price,
            shares=shares,
            stop_loss=stop,
            target=target,
            confidence=0,
            status=TradeStatus.OPEN,
            entry_time=datetime.now(timezone.utc),
            sequence_num=self._sequence_num,
            integrity_hash=self._compute_hash(trade_id),
            entry_reasoning=intent.reasoning,
            indicators_at_entry=intent.indicators_snapshot,
            fundamentals_at_entry=intent.fundamentals_snapshot,
            market_context_at_entry=intent.market_context_snapshot,
        )

        event_bus.publish("trade.opened", trade)
        slot.strategy_notification(trade, "opened")

        logger.info(
            "[%s] Opened %s: %d shares @ $%.2f (stop=$%.2f, target=$%.2f) — %s",
            slot.strategy.name, intent.ticker, shares, execution_price,
            stop, target, intent.reasoning[:60],
        )
        return trade

    def _close_position(
        self,
        slot: StrategySlot,
        position: Position,
        exit_price: float,
        reason: str,
        market_data: MarketData | None = None,
    ) -> Trade | None:
        """Close a position and create a Trade object."""
        result = slot.account.close_position(position, exit_price, reason)
        self._sequence_num += 1

        now = datetime.now(timezone.utc)
        hold_hours = (now - position.entry_time).total_seconds() / 3600
        is_pdt = position.entry_time.date() == now.date()

        trade = Trade(
            trade_id=result["trade_id"],
            strategy_name=slot.account.strategy_name,
            symbol=result["symbol"],
            direction=Direction(result["direction"]),
            trade_type=TradeType(result["trade_type"]),
            entry_price=result["entry_price"],
            exit_price=result["exit_price"],
            shares=result["shares"],
            stop_loss=position.stop_loss,
            target=position.target,
            confidence=0,
            status=TradeStatus.CLOSED,
            pnl_dollars=result["pnl_dollars"],
            pnl_percent=result["pnl_percent"],
            r_multiple=result["r_multiple"],
            exit_reason=reason,
            exit_reasoning=reason,
            exit_time=now,
            sequence_num=self._sequence_num,
            integrity_hash=self._compute_hash(result["trade_id"]),
            indicators_at_exit=market_data.current.to_dict() if market_data else None,
            pdt_flag=is_pdt,
            hold_duration_hours=round(hold_hours, 2),
        )

        event_bus.publish("trade.closed", trade)
        slot.strategy_notification(trade, "closed")

        logger.info(
            "[%s] Closed %s: %s (P&L: $%.2f, %.1fR)",
            slot.strategy.name, result["symbol"], reason,
            result["pnl_dollars"], result["r_multiple"],
        )
        return trade

    # ── Private: Data Fetching / Helpers ─────────────

    @staticmethod
    def _minutes_since_market_open() -> float:
        """Minutes elapsed since 9:30 AM ET. Returns 0 if before open, caps at 390."""
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("US/Eastern")
            now_et = datetime.now(et)
            open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            elapsed = (now_et - open_time).total_seconds() / 60.0
            return max(0.0, min(elapsed, 390.0))
        except Exception:
            return 390.0  # default to full day if timezone fails

    def _fetch_daily_bars(self, ticker: str) -> pd.DataFrame | None:
        """Fetch daily bars for indicator computation (~90 trading days)."""
        end = date.today()
        start = end - timedelta(days=130)  # ~90 trading days
        try:
            return self._provider.get_bars(ticker, "1", start, end, timespan="day")
        except TypeError:
            # Provider may not support timespan kwarg — try positional
            try:
                return self._provider.get_bars(ticker, "day", start, end)
            except Exception:
                logger.debug("Failed to fetch daily bars for %s", ticker)
                return None
        except Exception:
            logger.debug("Failed to fetch daily bars for %s", ticker)
            return None

    def _compute_hash(self, trade_id: str) -> str:
        """SHA-256 hash chaining for audit trail."""
        data = f"{trade_id}:{self._sequence_num}:{self._prev_hash}"
        h = hashlib.sha256(data.encode()).hexdigest()
        self._prev_hash = h
        return h


# ── Monkey-patch StrategySlot with notification helper ──

def _strategy_notification(self: StrategySlot, trade: Trade, event: str) -> None:
    """Send trade notification to strategy (if it supports it)."""
    try:
        if hasattr(self.strategy, "on_trade_executed"):
            self.strategy.on_trade_executed(
                TradeNotification(trade=trade, event=event)
            )
    except Exception:
        logger.exception(
            "Strategy '%s' failed on notification", self.strategy.name
        )


StrategySlot.strategy_notification = _strategy_notification
