"""
EdgeFinder Strategy: CHARLIE — Burry Deep Value
=================================================
Classic Burry — trading below tangible book, market has left them for dead.
Buying dollars for 50 cents.

Fundamental screening: price_to_tangible_book < 1.0, high FCF yield,
low EV/EBITDA, strong current ratio.

Technical entry: RSI oversold, EMA crossover (swing), volume spike on reversal.
Sentiment gate: blocks trades on strongly negative news.
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


@StrategyRegistry.register("charlie")
class CharlieStrategy(BaseStrategy):
    """Burry Deep Value — below tangible book, high FCF."""

    @property
    def name(self) -> str:
        return "charlie"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"rsi_oversold", "ema_crossover_swing", "stochastic_oversold", "obv_divergence"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._rsi_confidence_boost: float = 10.0
        self._stoch_confidence_boost: float = 10.0
        self._obv_confidence_boost: float = 5.0
        self._atr_multiplier: float = 2.5
        self._fallback_risk_pct: float = 0.07
        logger.info("Charlie (Burry Deep Value) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Below tangible book, strong FCF, cheap EV/EBITDA, liquid."""
        return (
            (stock_data.get("price_to_tangible_book") or 999) < 1.0
            and (stock_data.get("fcf_yield") or 0) >= 0.08
            and (stock_data.get("ev_to_ebitda") or 999) < 8
            and (stock_data.get("current_ratio") or 0) >= 1.5
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Charlie watchlist: {len(self._watchlist)} stocks")

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
                confidence = ts.confidence
                price = ts.price or (
                    float(df.iloc[-1]["Close"])
                    if "Close" in df.columns
                    else float(df.iloc[-1].get("close", 0))
                )
                if price <= 0:
                    continue
                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(ticker, confidence)
                        if action == "BLOCK":
                            logger.info(f"[charlie] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[charlie] Sentiment gate error: {e}")

                # Confidence boosts: deep value indicator confirmations
                has_rsi = _has_indicator(ts.indicators, "rsi_oversold")
                has_stoch = _has_indicator(ts.indicators, "stochastic_oversold")
                has_obv = _has_indicator(ts.indicators, "obv_divergence")
                if has_rsi:
                    confidence = min(100.0, confidence + self._rsi_confidence_boost)
                if has_stoch:
                    confidence = min(100.0, confidence + self._stoch_confidence_boost)
                if has_obv:
                    confidence = min(100.0, confidence + self._obv_confidence_boost)

                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue

                # ATR-based dynamic stop-loss (wider for deep value)
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                target = round(price + (price - stop_loss) * 2.0, 2)
                meta = {
                    "strategy": "charlie",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "rsi_boost_applied": has_rsi,
                    "stoch_boost_applied": has_stoch,
                    "obv_boost_applied": has_obv,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["burry_score"] = score_info.get("burry_score")
                    meta["fcf_yield"] = score_info.get("fcf_yield")
                    meta["price_to_tangible_book"] = score_info.get("price_to_tangible_book")
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
        logger.info(f"[charlie] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bear":
            logger.info("[charlie] Bear market — deep value hunting season")
        elif regime.trend == "bull":
            logger.info("[charlie] Bull market — staying selective on deep value")
        else:
            logger.info(f"[charlie] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[charlie] Strategy paused: {reason}")
