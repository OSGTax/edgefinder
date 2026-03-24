"""
EdgeFinder Strategy: Peter Lynch
=================================
Growth-at-a-reasonable-price strategy based on Peter Lynch's criteria.

Fundamental screening: PEG ratio, earnings growth, debt-to-equity,
revenue growth, institutional ownership, and Lynch stock classification.

Technical entry: EMA crossovers, RSI, MACD, volume confirmation.
Sentiment gate: blocks trades on strongly negative news.

Reuses existing modules:
    - modules.scanner: score_lynch(), classify_lynch_category()
    - modules.signals: compute_indicators(), generate_signals()
    - modules.sentiment: gate_trade()
"""

import logging
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
from modules.signals import compute_indicators, generate_signals as detect_signals

logger = logging.getLogger(__name__)


@StrategyRegistry.register("lynch")
class LynchStrategy(BaseStrategy):
    """Peter Lynch growth-at-a-reasonable-price strategy.

    Watchlist: Stocks scoring above threshold on Lynch criteria
    (PEG < 1.5, strong earnings growth, low debt, good revenue growth).

    Entry: Technical signals (EMA crossovers, RSI oversold, MACD) with
    sentiment gate.

    Exit: Managed by arena executor (stop loss, target, trailing stop).
    """

    @property
    def name(self) -> str:
        return "lynch"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_day", "ema_crossover_swing", "macd_crossover"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}  # ticker -> scoring info
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._min_lynch_score: float = 50.0  # Minimum Lynch sub-score
        logger.info(f"Lynch strategy initialized (min_score={self._min_lynch_score})")

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        """Set watchlist from pre-scored stocks.

        Args:
            scored_stocks: List of dicts with keys:
                ticker, lynch_score, composite_score, lynch_category, etc.
        """
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            lynch_score = stock.get("lynch_score", 0)
            if lynch_score >= self._min_lynch_score:
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(
            f"Lynch watchlist: {len(self._watchlist)} stocks "
            f"(min Lynch score: {self._min_lynch_score})"
        )

    def qualifies_stock(self, stock_data: dict) -> bool:
        """A stock qualifies for Lynch if its lynch_score meets the minimum."""
        return (stock_data.get("lynch_score") or 0) >= self._min_lynch_score

    def get_watchlist(self) -> list[str]:
        return list(self._watchlist)

    def generate_signals(
        self, bars: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        """Generate buy signals for Lynch-qualifying stocks.

        For each ticker in bars:
        1. Compute technical indicators
        2. Detect signals (EMA crossover, RSI, MACD, volume)
        3. Apply sentiment gate (optional)
        4. Return qualifying signals
        """
        signals = []

        for ticker, df in bars.items():
            if df is None or df.empty:
                continue

            # Compute technical indicators
            snapshot = compute_indicators(df, ticker=ticker)
            if snapshot is None:
                continue

            # Detect technical signals
            trade_signals = detect_signals(snapshot)
            if not trade_signals:
                continue

            for ts in trade_signals:
                if ts.signal_type != "BUY":
                    continue

                # Only act on momentum-based signals matching this strategy
                signal_names = (
                    set(ts.indicators.keys())
                    if isinstance(ts.indicators, dict)
                    else {ind.get("name") for ind in ts.indicators
                          if isinstance(ind, dict)}
                )
                if not signal_names & self.preferred_signals:
                    continue

                confidence = ts.confidence
                price = ts.price or (
                    float(df.iloc[-1]["Close"])
                    if "Close" in df.columns
                    else float(df.iloc[-1].get("close", 0))
                )

                if price <= 0:
                    continue

                # Apply sentiment gate
                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(
                            ticker, confidence
                        )
                        if action == "BLOCK":
                            logger.info(
                                f"[lynch] {ticker}: blocked by sentiment"
                            )
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[lynch] Sentiment gate error: {e}")

                # Skip if below minimum confidence
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue

                # Calculate stop and target
                risk_pct = 0.05  # 5% stop loss
                stop_loss = round(price * (1 - risk_pct), 2)
                target = round(
                    price + (price - stop_loss) * settings.MIN_REWARD_TO_RISK_RATIO,
                    2,
                )

                # Build metadata with Lynch scoring context
                meta = {
                    "strategy": "lynch",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["lynch_score"] = score_info.get("lynch_score")
                    meta["lynch_category"] = score_info.get("lynch_category")

                signals.append(Signal(
                    ticker=ticker,
                    action="BUY",
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=confidence,
                    trade_type=ts.trade_type,
                    metadata=meta,
                ))

        return signals

    def on_trade_executed(self, notification: TradeNotification) -> None:
        self._trades_log.append(notification)
        logger.info(
            f"[lynch] Trade notification: {notification.action} "
            f"{notification.ticker} @ ${notification.entry_price:.2f}"
        )

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        logger.info(f"[lynch] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[lynch] Strategy paused: {reason}")
