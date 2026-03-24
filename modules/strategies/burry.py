"""
EdgeFinder Strategy: Michael Burry
====================================
Deep-value contrarian strategy based on Michael Burry's criteria.

Fundamental screening: Free cash flow yield, price-to-tangible-book,
EV/EBITDA, current ratio, and short interest (contrarian signal).

Technical entry: EMA crossovers, RSI, MACD, volume confirmation.
Sentiment gate: blocks trades on strongly negative news.

Reuses existing modules:
    - modules.scanner: score_burry()
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


@StrategyRegistry.register("burry")
class BurryStrategy(BaseStrategy):
    """Michael Burry deep-value contrarian strategy.

    Watchlist: Stocks scoring above threshold on Burry criteria
    (high FCF yield, low P/TB, low EV/EBITDA, strong current ratio,
    interesting short interest).

    Entry: Technical signals (EMA crossovers, RSI oversold, MACD) with
    sentiment gate. Burry is more aggressive on mean-reversion entries
    (RSI oversold is weighted higher).

    Exit: Managed by arena executor (stop loss, target, trailing stop).
    """

    @property
    def name(self) -> str:
        return "burry"

    @property
    def version(self) -> str:
        return "1.0.0"

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}  # ticker -> scoring info
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._min_burry_score: float = 50.0   # Minimum Burry sub-score
        self._rsi_confidence_boost: float = 10.0  # Extra confidence for RSI oversold
        logger.info(
            f"Burry strategy initialized (min_score={self._min_burry_score})"
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        """Set watchlist from pre-scored stocks.

        Args:
            scored_stocks: List of dicts with keys:
                ticker, burry_score, composite_score, etc.
        """
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            burry_score = stock.get("burry_score", 0)
            if burry_score >= self._min_burry_score:
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(
            f"Burry watchlist: {len(self._watchlist)} stocks "
            f"(min Burry score: {self._min_burry_score})"
        )

    def qualifies_stock(self, stock_data: dict) -> bool:
        """A stock qualifies for Burry if its burry_score meets the minimum."""
        return (stock_data.get("burry_score") or 0) >= self._min_burry_score

    def get_watchlist(self) -> list[str]:
        return list(self._watchlist)

    def generate_signals(
        self, bars: dict[str, pd.DataFrame]
    ) -> list[Signal]:
        """Generate buy signals for Burry-qualifying stocks.

        For each ticker in bars:
        1. Compute technical indicators
        2. Detect signals (EMA crossover, RSI, MACD, volume)
        3. Apply Burry-specific adjustments (RSI oversold boost)
        4. Apply sentiment gate (optional)
        5. Return qualifying signals
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

                confidence = ts.confidence
                price = ts.price or (
                    float(df.iloc[-1]["Close"])
                    if "Close" in df.columns
                    else float(df.iloc[-1].get("close", 0))
                )

                if price <= 0:
                    continue

                # Burry-specific: boost confidence for RSI oversold
                # (deep value stocks that are also technically oversold
                # are the ideal Burry entry)
                has_rsi_oversold = any(
                    ind.get("name") == "rsi_oversold"
                    for ind in (ts.indicators if isinstance(ts.indicators, list)
                               else [ts.indicators] if ts.indicators else [])
                )
                if has_rsi_oversold:
                    confidence = min(100.0, confidence + self._rsi_confidence_boost)

                # Apply sentiment gate
                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(
                            ticker, confidence
                        )
                        if action == "BLOCK":
                            logger.info(
                                f"[burry] {ticker}: blocked by sentiment"
                            )
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[burry] Sentiment gate error: {e}")

                # Skip if below minimum confidence
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue

                # Burry uses wider stops (value takes longer to realize)
                risk_pct = 0.07  # 7% stop loss (wider than Lynch's 5%)
                stop_loss = round(price * (1 - risk_pct), 2)
                # Burry targets deeper recoveries
                target = round(
                    price + (price - stop_loss) * 2.0,  # 2:1 R:R
                    2,
                )

                # Build metadata with Burry scoring context
                meta = {
                    "strategy": "burry",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "rsi_boost_applied": has_rsi_oversold,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["burry_score"] = score_info.get("burry_score")
                    meta["fcf_yield"] = score_info.get("fcf_yield")
                    meta["price_to_tangible_book"] = score_info.get(
                        "price_to_tangible_book"
                    )

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
            f"[burry] Trade notification: {notification.action} "
            f"{notification.ticker} @ ${notification.entry_price:.2f}"
        )

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        # Burry is contrarian — bear markets are his hunting ground
        if regime.trend == "bear":
            logger.info("[burry] Bear market detected — prime hunting conditions")
        elif regime.trend == "bull":
            logger.info("[burry] Bull market — value harder to find, staying selective")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[burry] Strategy paused: {reason}")
