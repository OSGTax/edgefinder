"""
EdgeFinder Strategy: QUEBEC — Momentum Swing
==============================================
Pure technical momentum/trend strategy for swing trading.
Requires 3+ confirmations before entry: EMA golden cross, MACD crossover,
ADX trend strength, volume spike, and RSI in the 40-70 sweet spot.

No fundamental filtering — trades any stock with sufficient volume and price.
Uses wider ATR-based stops (2.5x) and 3:1 reward-to-risk for swing holds.
Market regime aware: tightens in bear, widens in bull.
"""

import logging

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


def _has_indicator(ts_indicators, name: str) -> bool:
    """Check if a named indicator fired in a signal's indicator data."""
    if isinstance(ts_indicators, dict):
        return name in ts_indicators
    elif isinstance(ts_indicators, list):
        return any(ind.get("name") == name for ind in ts_indicators if isinstance(ind, dict))
    return False


def _count_confirmations(ts_indicators, preferred: set[str]) -> int:
    """Count how many preferred signals fired in a trade signal."""
    if isinstance(ts_indicators, dict):
        return len(set(ts_indicators.keys()) & preferred)
    elif isinstance(ts_indicators, list):
        names = {ind.get("name") for ind in ts_indicators if isinstance(ind, dict)}
        return len(names & preferred)
    return 0


@StrategyRegistry.register("quebec")
class QuebecStrategy(BaseStrategy):
    """Momentum Swing — pure technical, 3+ confirmations, swing timeframe."""

    @property
    def name(self) -> str:
        return "quebec"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_swing", "macd_crossover", "adx_trend", "volume_spike"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._atr_multiplier: float = 2.5
        self._fallback_risk_pct: float = 0.07
        self._rr_ratio: float = 3.0
        self._min_confirmations: int = 3
        self._adx_strong_boost: float = 5.0
        self._volume_exceptional_boost: float = 5.0
        logger.info("Quebec (Momentum Swing) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Pure technical — accept any stock with sufficient volume and price."""
        avg_volume = stock_data.get("avg_volume") or 0
        price = stock_data.get("price") or 0
        return avg_volume > 100_000 and price > 1.0

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Quebec watchlist: {len(self._watchlist)} stocks")

    def get_watchlist(self) -> list[str]:
        return list(self._watchlist)

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for ticker, df in bars.items():
            if df is None or df.empty:
                continue
            snapshot = compute_indicators(df, ticker=ticker)
            if snapshot is None:
                continue
            trade_signals = detect_signals(snapshot)
            if not trade_signals:
                continue
            for ts in trade_signals:
                if ts.signal_type != "BUY":
                    continue
                signal_names = (
                    set(ts.indicators.keys())
                    if isinstance(ts.indicators, dict)
                    else {ind.get("name") for ind in ts.indicators if isinstance(ind, dict)}
                )
                if not signal_names & self.preferred_signals:
                    continue

                # Count confirmations from preferred signals
                confirmations = _count_confirmations(ts.indicators, self.preferred_signals)

                # RSI sweet spot check (40-70) counts as an additional confirmation
                try:
                    rsi_in_range = (
                        snapshot.rsi is not None
                        and isinstance(snapshot.rsi, (int, float))
                        and 40 <= snapshot.rsi <= 70
                    )
                except (TypeError, ValueError):
                    rsi_in_range = False
                if rsi_in_range:
                    confirmations += 1

                # Require 3+ confirmations for conservative entry
                if confirmations < self._min_confirmations:
                    continue

                confidence = ts.confidence
                price = ts.price or (
                    float(df.iloc[-1]["Close"])
                    if "Close" in df.columns
                    else float(df.iloc[-1].get("close", 0))
                )
                if price <= 0:
                    continue

                # Sentiment gate
                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(ticker, confidence)
                        if action == "BLOCK":
                            logger.info(f"[quebec] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[quebec] Sentiment gate error: {e}")

                # Confidence boost: ADX > 30 = very strong trend
                if snapshot.adx is not None and snapshot.adx > 30:
                    confidence = min(100.0, confidence + self._adx_strong_boost)

                # Confidence boost: volume > 2x average = exceptional participation
                if (
                    snapshot.current_volume is not None
                    and snapshot.avg_volume is not None
                    and snapshot.avg_volume > 0
                    and snapshot.current_volume > 2.0 * snapshot.avg_volume
                ):
                    confidence = min(100.0, confidence + self._volume_exceptional_boost)

                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue

                # ATR-based dynamic stop-loss (wider for swing)
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)

                # Target based on R:R ratio
                target = round(price + (price - stop_loss) * self._rr_ratio, 2)

                meta = {
                    "strategy": "quebec",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "confirmations": confirmations,
                    "rsi_in_range": rsi_in_range,
                    "atr_multiplier": self._atr_multiplier,
                    "rr_ratio": self._rr_ratio,
                }

                signals.append(Signal(
                    ticker=ticker,
                    action="BUY",
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    confidence=confidence,
                    trade_type="SWING",
                    metadata=meta,
                ))
        return signals

    def on_trade_executed(self, notification: TradeNotification) -> None:
        self._trades_log.append(notification)
        logger.info(f"[quebec] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bull":
            self._atr_multiplier = 2.5
            self._rr_ratio = 4.0
            logger.info("[quebec] Bull market — widening targets to 4:1 R:R")
        elif regime.trend == "bear":
            self._atr_multiplier = 2.0
            self._rr_ratio = 2.0
            logger.info("[quebec] Bear market — tightening stops and targets")
        else:
            self._atr_multiplier = 2.5
            self._rr_ratio = 3.0
            logger.info(f"[quebec] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[quebec] Strategy paused: {reason}")
