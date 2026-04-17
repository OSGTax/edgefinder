"""EdgeFinder v2 — Arena engine: multi-strategy orchestration.

Manages multiple strategies, each with its own virtual account and executor.
Feeds market data to strategies, routes signals through executors, monitors positions.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import Trade
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification
from edgefinder.trading.account import VirtualAccount
from edgefinder.trading.executor import Executor

logger = logging.getLogger(__name__)


class StrategySlot:
    """Bundles a strategy instance with its account and executor."""

    def __init__(self, strategy: BaseStrategy, pdt_enabled: bool = False) -> None:
        self.strategy = strategy
        # Check for per-strategy risk config
        risk_config = getattr(strategy, "risk_config", None) or {}
        self.account = VirtualAccount(
            strategy_name=strategy.name,
            starting_capital=settings.starting_capital,
            pdt_enabled=pdt_enabled,
            max_risk_pct=risk_config.get("max_risk_pct"),
            max_concentration_pct=risk_config.get("max_concentration_pct"),
        )
        self.executor = Executor(self.account)
        self.watchlist: list[str] = []


class ArenaEngine:
    """Orchestrates multiple strategies competing in isolated accounts."""

    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider
        self._slots: dict[str, StrategySlot] = {}
        self._fundamentals_cache: dict[str, object] = {}  # ticker -> TickerFundamentals

    def load_strategies(self, pdt_config: dict[str, bool] | None = None) -> None:
        """Load all registered strategies into the arena.

        Args:
            pdt_config: Optional dict of {strategy_name: pdt_enabled}.
        """
        pdt_config = pdt_config or {}
        for strategy in StrategyRegistry.get_instances():
            pdt = pdt_config.get(strategy.name, False)
            slot = StrategySlot(strategy, pdt_enabled=pdt)
            strategy.init()
            self._slots[strategy.name] = slot
            logger.info(
                "Loaded strategy '%s' v%s (PDT: %s)",
                strategy.name, strategy.version, pdt,
            )

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
        """Set cached fundamentals for re-qualification checks.

        Called by services.py after each scan. The cache maps
        ticker symbol -> TickerFundamentals. Used in run_signal_check()
        to verify a stock still qualifies before generating signals.
        """
        self._fundamentals_cache = cache
        logger.info("Fundamentals cache updated: %d tickers", len(cache))

    def run_signal_check(self) -> list[Trade]:
        """Run signal generation for all strategies on their watchlists.

        Returns all opened trades.
        """
        all_trades: list[Trade] = []

        logger.info(
            "Signal check starting: %d slots, fundamentals_cache=%d",
            len(self._slots), len(self._fundamentals_cache),
        )

        for name, slot in self._slots.items():
            if slot.account.is_paused:
                logger.info(
                    "[%s] slot paused — skipping (cash=$%.2f, drawdown=%.1f%%)",
                    name, slot.account.cash, slot.account.drawdown_pct * 100,
                )
                continue

            logger.info(
                "[%s] watchlist=%d tickers, open_positions=%d",
                name, len(slot.watchlist), len(slot.account.positions),
            )

            evaluated = 0
            bars_missing = 0
            signals_generated = 0
            opened_here = 0
            requalif_failed = 0

            for ticker in slot.watchlist:
                # Skip tickers that already have an open position
                if slot.account.get_position(ticker):
                    logger.debug(
                        "Strategy '%s' already has open position in %s — skipping",
                        name, ticker,
                    )
                    continue

                evaluated += 1

                # Re-qualification gate: verify fundamentals still qualify
                cached_fund = self._fundamentals_cache.get(ticker)
                if cached_fund is not None:
                    try:
                        if not slot.strategy.qualifies_stock(cached_fund):
                            requalif_failed += 1
                            logger.debug(
                                "[%s] Re-qualification failed for %s — skipping",
                                name, ticker,
                            )
                            continue
                    except Exception:
                        logger.exception(
                            "[%s] Re-qualification raised for %s — proceeding anyway",
                            name, ticker,
                        )

                bars = self._fetch_bars(ticker)
                if bars is None or bars.empty:
                    bars_missing += 1
                    continue

                try:
                    signals = slot.strategy.generate_signals(ticker, bars)
                except Exception:
                    logger.exception(
                        "Strategy '%s' failed generate_signals for %s", name, ticker
                    )
                    continue

                signals_generated += len(signals)

                # Fetch fresh market price ONCE per ticker (cached for the
                # next ~30s in the provider's in-memory price cache). This
                # anchors execution to the current price source rather than
                # the historical pattern bar's close, which keeps signal
                # entry and position-monitor exit on the same data source.
                fresh_price = None
                if signals:
                    try:
                        fresh_price = self._provider.get_latest_price(ticker)
                    except Exception:
                        logger.exception("get_latest_price failed for %s", ticker)

                for signal in signals:
                    trade = slot.executor.execute_signal(signal, fresh_price=fresh_price)
                    if trade:
                        all_trades.append(trade)
                        opened_here += 1
                        slot.strategy.on_trade_executed(
                            TradeNotification(trade=trade, event="opened")
                        )
                        break  # one position per ticker per signal check

            logger.info(
                "[%s] signal check done: evaluated=%d, requalif_failed=%d, "
                "bars_missing=%d, signals_generated=%d, trades_opened=%d",
                name, evaluated, requalif_failed,
                bars_missing, signals_generated, opened_here,
            )

        logger.info(
            "Signal check complete: %d trades opened across %d strategies",
            len(all_trades), len(self._slots),
        )
        return all_trades

    def check_positions(self) -> list[Trade]:
        """Check all open positions against current prices and exit signals.

        Two-pass check:
        1. Stop loss / target hit (price-based)
        2. Bearish signal exit (technical-based)

        Returns all closed trades.
        """
        from edgefinder.signals.engine import compute_indicators, detect_signals

        all_closed: list[Trade] = []

        for name, slot in self._slots.items():
            if not slot.account.positions:
                continue

            # Get current prices for all symbols with open positions
            prices: dict[str, float] = {}
            for pos in slot.account.positions:
                price = self._provider.get_latest_price(pos.symbol)
                if price is not None:
                    prices[pos.symbol] = price

            # Pass 1: stop loss / target hit
            closed = slot.executor.check_positions(prices)
            for trade in closed:
                slot.strategy.on_trade_executed(
                    TradeNotification(trade=trade, event="closed")
                )
            all_closed.extend(closed)

            # Pass 2: bearish signal exits
            exit_patterns = slot.strategy.exit_signals
            if not exit_patterns:
                continue

            for pos in list(slot.account.positions):
                price = prices.get(pos.symbol)
                if price is None:
                    continue
                try:
                    bars = self._fetch_bars(pos.symbol)
                    if bars is None or bars.empty:
                        continue
                    indicators = compute_indicators(bars)
                    if indicators is None:
                        continue
                    signals = detect_signals(indicators, pos.symbol)
                    for sig in signals:
                        pattern = sig.metadata.get("pattern", "")
                        if pattern in exit_patterns and sig.action.value == "SELL":
                            trade = slot.executor.close_on_signal(pos, price, pattern)
                            slot.strategy.on_trade_executed(
                                TradeNotification(trade=trade, event="closed")
                            )
                            all_closed.append(trade)
                            logger.info(
                                "[%s] Signal exit: %s closed on %s",
                                name, pos.symbol, pattern,
                            )
                            break  # position is closed, move to next
                except Exception:
                    logger.exception(
                        "Signal exit check failed for %s/%s", name, pos.symbol
                    )

        return all_closed

    def broadcast_market_snapshot(self, snapshot) -> None:
        """Forward a market snapshot to all strategies that implement on_market_snapshot."""
        for name, slot in self._slots.items():
            try:
                slot.strategy.on_market_snapshot(snapshot)
            except Exception:
                logger.exception(
                    "Strategy '%s' failed on_market_snapshot", name
                )

    def get_strategy(self, name: str) -> BaseStrategy | None:
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

    # ── Private ──────────────────────────────────────

    def _fetch_bars(self, ticker: str) -> pd.DataFrame | None:
        """Fetch 5-minute intraday bars (~7 trading days)."""
        from datetime import timedelta
        end = date.today()
        start = end - timedelta(days=10)
        return self._provider.get_bars(ticker, "5", start, end)
