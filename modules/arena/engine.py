"""
EdgeFinder Arena Engine — Multi-Strategy Orchestration
=======================================================
Feeds market data to all registered strategies, collects signals,
routes them through the honest executor, and monitors positions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import settings
from modules.strategies.base import (
    BaseStrategy,
    StrategyRegistry,
    Signal,
    TradeNotification,
    MarketRegime,
)
from modules.arena.virtual_account import VirtualAccount
from modules.arena.executor import Executor

logger = logging.getLogger(__name__)


class ArenaEngine:
    """Orchestrates multiple strategies competing in the arena.

    Responsibilities:
    - Manage strategy lifecycle (init, pause, resume)
    - Feed market data to each strategy
    - Route signals to executor
    - Monitor positions across all accounts
    - Detect signal/position overlap
    - Track market regime
    """

    def __init__(
        self,
        starting_capital: float | None = None,
        max_positions_per_strategy: int | None = None,
        drawdown_pause_pct: float = -15.0,
    ):
        self.starting_capital = starting_capital or settings.STARTING_CAPITAL
        self.max_positions = max_positions_per_strategy or settings.MAX_OPEN_POSITIONS
        self.drawdown_pause_pct = drawdown_pause_pct

        self.strategies: dict[str, BaseStrategy] = {}
        self.accounts: dict[str, VirtualAccount] = {}
        self.executor = Executor()
        self.market_regime = MarketRegime()
        self._enabled: dict[str, bool] = {}

    # ── LIFECYCLE ────────────────────────────────────────────

    def load_strategies(
        self, names: list[str] | None = None
    ) -> dict[str, BaseStrategy]:
        """Load and initialize strategies from the registry.

        Args:
            names: Specific strategy names to load. None = load all.

        Returns:
            Dict of name -> initialized strategy.
        """
        available = StrategyRegistry.list_strategies()

        if names:
            to_load = {n: available[n] for n in names if n in available}
            missing = set(names) - set(available)
            if missing:
                logger.warning(f"Strategies not found: {missing}")
        else:
            to_load = available

        for name, cls in to_load.items():
            try:
                strategy = cls()
                strategy.init()
                self.strategies[name] = strategy
                self.accounts[name] = VirtualAccount(
                    strategy_name=name,
                    starting_capital=self.starting_capital,
                    max_positions=self.max_positions,
                )
                self._enabled[name] = True
                logger.info(
                    f"Loaded strategy: {name} v{strategy.version}"
                )
            except Exception as e:
                logger.error(f"Failed to load strategy '{name}': {e}")

        return self.strategies

    def add_strategy(
        self,
        name: str,
        strategy: BaseStrategy,
        starting_capital: float | None = None,
    ) -> None:
        """Manually add an already-instantiated strategy.

        Useful for testing or dynamic strategy injection.
        """
        self.strategies[name] = strategy
        self.accounts[name] = VirtualAccount(
            strategy_name=name,
            starting_capital=starting_capital or self.starting_capital,
            max_positions=self.max_positions,
        )
        self._enabled[name] = True
        logger.info(f"Added strategy: {name}")

    def enable_strategy(self, name: str) -> None:
        if name in self.strategies:
            self._enabled[name] = True
            self.accounts[name].unpause()
            logger.info(f"Enabled strategy: {name}")

    def disable_strategy(self, name: str) -> None:
        if name in self.strategies:
            self._enabled[name] = False
            logger.info(f"Disabled strategy: {name}")

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, False)

    # ── DATA FEED ────────────────────────────────────────────

    def run_signal_check(
        self,
        bars: dict[str, pd.DataFrame],
        volumes: Optional[dict[str, float]] = None,
        price_source: str = "data_service",
    ) -> list[dict]:
        """Feed bars to all strategies, collect and execute signals.

        Args:
            bars: Dict of ticker -> DataFrame with OHLCV data.
            volumes: Dict of ticker -> avg daily volume (for slippage).
            price_source: Source of price data for audit trail.

        Returns:
            List of executed trade audit entries.
        """
        volumes = volumes or {}
        all_signals: dict[str, list[Signal]] = {}
        executed = []

        # 1. Collect signals from all enabled strategies
        for name, strategy in self.strategies.items():
            if not self.is_enabled(name):
                continue
            if self.accounts[name].is_paused:
                continue

            # Filter bars to strategy's watchlist if specified
            watchlist = strategy.get_watchlist()
            strategy_bars = (
                {t: b for t, b in bars.items() if t in watchlist}
                if watchlist
                else bars
            )

            try:
                signals = strategy.generate_signals(strategy_bars)
                all_signals[name] = signals
                if signals:
                    logger.info(
                        f"[{name}] Generated {len(signals)} signals: "
                        f"{[s.ticker for s in signals]}"
                    )
            except Exception as e:
                logger.error(f"[{name}] Signal generation failed: {e}")
                all_signals[name] = []

        # 2. Calculate overlap
        signal_tickers: dict[str, int] = {}  # ticker -> count of strategies signaling
        position_tickers: dict[str, int] = {}  # ticker -> count of strategies holding
        for name, signals in all_signals.items():
            for s in signals:
                signal_tickers[s.ticker] = signal_tickers.get(s.ticker, 0) + 1
        for name, account in self.accounts.items():
            for pos in account.positions.values():
                position_tickers[pos.ticker] = position_tickers.get(pos.ticker, 0) + 1

        # 3. Execute signals
        for name, signals in all_signals.items():
            account = self.accounts[name]
            strategy = self.strategies[name]

            for signal in signals:
                # Get latest bar data for audit
                bar_data = {}
                if signal.ticker in bars and not bars[signal.ticker].empty:
                    last_bar = bars[signal.ticker].iloc[-1]
                    bar_data = {
                        "open": float(last_bar.get("open", 0)),
                        "high": float(last_bar.get("high", 0)),
                        "low": float(last_bar.get("low", 0)),
                        "close": float(last_bar.get("close", 0)),
                        "volume": int(last_bar.get("volume", 0)),
                    }

                avg_vol = volumes.get(signal.ticker, 1_000_000)

                audit = self.executor.execute_signal(
                    signal=signal,
                    account=account,
                    avg_daily_volume=avg_vol,
                    price_source=price_source,
                    bar_data=bar_data,
                    market_regime=self.market_regime,
                    signal_overlap=signal_tickers.get(signal.ticker, 1) - 1,
                    position_overlap=position_tickers.get(signal.ticker, 0),
                )

                if audit:
                    executed.append(audit.to_dict())

                    # Notify strategy
                    try:
                        strategy.on_trade_executed(TradeNotification(
                            trade_id=audit.trade_id,
                            ticker=signal.ticker,
                            action=signal.action,
                            entry_price=audit.execution_price,
                            shares=audit.shares,
                        ))
                    except Exception as e:
                        logger.error(
                            f"[{name}] on_trade_executed callback failed: {e}"
                        )

        return executed

    # ── POSITION MONITORING ──────────────────────────────────

    def monitor_positions(
        self,
        prices: dict[str, float],
        price_source: str = "data_service",
    ) -> list[dict]:
        """Update all positions with current prices, check stops/targets.

        Args:
            prices: Dict of ticker -> current price.
            price_source: Source of price data.

        Returns:
            List of trade results for positions that were closed.
        """
        closed = []

        for name, account in self.accounts.items():
            if not self.is_enabled(name):
                continue

            # Copy trade_ids to avoid dict mutation during iteration
            trade_ids = list(account.positions.keys())

            for trade_id in trade_ids:
                position = account.positions.get(trade_id)
                if not position:
                    continue

                price = prices.get(position.ticker)
                if price is None:
                    continue

                result = self.executor.close_on_stop_or_target(
                    account=account,
                    trade_id=trade_id,
                    current_price=price,
                    price_source=price_source,
                )

                if result:
                    closed.append(result)

                    # Notify strategy
                    strategy = self.strategies.get(name)
                    if strategy:
                        try:
                            strategy.on_trade_executed(TradeNotification(
                                trade_id=result["trade_id"],
                                ticker=result["ticker"],
                                action="SELL",
                                entry_price=result["entry_price"],
                                exit_price=result["exit_price"],
                                shares=result["shares"],
                                pnl_dollars=result["pnl_dollars"],
                                pnl_percent=result["pnl_percent"],
                                r_multiple=result["r_multiple"],
                                exit_reason=result["exit_reason"],
                            ))
                        except Exception as e:
                            logger.error(
                                f"[{name}] on_trade_executed callback failed: {e}"
                            )

            # Check drawdown circuit breaker
            account.check_drawdown_breaker(self.drawdown_pause_pct)
            if account.is_paused:
                strategy = self.strategies.get(name)
                if strategy:
                    try:
                        strategy.on_strategy_pause(account.pause_reason)
                    except Exception as e:
                        logger.error(
                            f"[{name}] on_strategy_pause callback failed: {e}"
                        )

        return closed

    # ── SNAPSHOTS ────────────────────────────────────────────

    def take_snapshots(self) -> list[dict]:
        """Take equity snapshots for all accounts."""
        snapshots = []
        for name, account in self.accounts.items():
            snapshots.append(account.take_snapshot())
        return snapshots

    # ── MARKET REGIME ────────────────────────────────────────

    def update_market_regime(self, regime: MarketRegime) -> None:
        """Update market regime and notify all strategies."""
        old_trend = self.market_regime.trend
        self.market_regime = regime

        if old_trend != regime.trend:
            logger.info(
                f"Market regime change: {old_trend} → {regime.trend} "
                f"(vol={regime.volatility})"
            )
            for name, strategy in self.strategies.items():
                if self.is_enabled(name):
                    try:
                        strategy.on_market_regime_change(regime)
                    except Exception as e:
                        logger.error(
                            f"[{name}] on_market_regime_change failed: {e}"
                        )

    # ── COMPARISON / STATUS ──────────────────────────────────

    def get_leaderboard(self) -> list[dict]:
        """Get strategy performance comparison, sorted by return."""
        board = []
        for name, account in self.accounts.items():
            info = account.to_dict()
            info["enabled"] = self.is_enabled(name)
            info["version"] = self.strategies[name].version
            board.append(info)

        board.sort(key=lambda x: x["total_return_pct"], reverse=True)
        return board

    def get_overlap_report(self) -> dict:
        """Report signal and position overlap across strategies."""
        position_tickers: dict[str, list[str]] = {}  # ticker -> [strategy_names]
        for name, account in self.accounts.items():
            for pos in account.positions.values():
                position_tickers.setdefault(pos.ticker, []).append(name)

        overlapping = {
            ticker: strategies
            for ticker, strategies in position_tickers.items()
            if len(strategies) > 1
        }

        return {
            "total_strategies": len(self.strategies),
            "enabled_strategies": sum(1 for v in self._enabled.values() if v),
            "overlapping_positions": overlapping,
            "overlap_count": len(overlapping),
        }

    def get_status(self) -> dict:
        """Full arena status for dashboard/API."""
        return {
            "strategies": len(self.strategies),
            "enabled": sum(1 for v in self._enabled.values() if v),
            "paused": sum(1 for a in self.accounts.values() if a.is_paused),
            "total_open_positions": sum(
                a.open_position_count for a in self.accounts.values()
            ),
            "total_closed_trades": sum(
                len(a.closed_trades) for a in self.accounts.values()
            ),
            "market_regime": self.market_regime.trend,
            "market_volatility": self.market_regime.volatility,
            "leaderboard": self.get_leaderboard(),
            "overlap": self.get_overlap_report(),
            "audit_entries": len(self.executor.audit_log),
        }
