"""
EdgeFinder Strategy: JULIET — Deep Value Contrarian
=====================================================
Extremely cheap on tangible assets, high FCF, market hates them.
More extreme than Charlie — targets the most despised, cheapest stocks.

Fundamental screening: price_to_tangible_book < 1.0, FCF yield >= 6%,
strong current ratio, notable short interest.

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


@StrategyRegistry.register("juliet")
class JulietStrategy(BaseStrategy):
    """Deep Value Contrarian — most despised, cheapest stocks."""

    @property
    def name(self) -> str:
        return "juliet"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"rsi_oversold", "ema_crossover_swing", "stochastic_oversold", "near_52w_low", "obv_divergence"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._rsi_confidence_boost: float = 10.0
        self._stoch_confidence_boost: float = 10.0
        self._52w_low_confidence_boost: float = 5.0
        self._obv_confidence_boost: float = 5.0
        self._atr_multiplier: float = 3.0
        self._fallback_risk_pct: float = 0.08
        logger.info("Juliet (Deep Value Contrarian) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Extremely cheap, generating cash, hated by the market."""
        return (
            (stock_data.get("price_to_tangible_book") or 999) < 1.0
            and (stock_data.get("fcf_yield") or 0) >= 0.06
            and (stock_data.get("current_ratio") or 0) >= 2.0
            and (stock_data.get("short_interest") or 0) >= 0.10
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Juliet watchlist: {len(self._watchlist)} stocks")

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
                # Confidence boosts — max contrarian stack
                _has = lambda name: (name in ts.indicators) if isinstance(ts.indicators, dict) else any(ind.get("name") == name for ind in ts.indicators if isinstance(ind, dict))
                if _has("rsi_oversold"):
                    confidence = min(100.0, confidence + self._rsi_confidence_boost)
                if _has("stochastic_oversold"):
                    confidence = min(100.0, confidence + self._stoch_confidence_boost)
                if _has("near_52w_low"):
                    confidence = min(100.0, confidence + self._52w_low_confidence_boost)
                if _has("obv_divergence"):
                    confidence = min(100.0, confidence + self._obv_confidence_boost)

                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(ticker, confidence)
                        if action == "BLOCK":
                            logger.info(f"[juliet] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[juliet] Sentiment gate error: {e}")
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                # ATR-based dynamic stop-loss
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                target = round(price + (price - stop_loss) * 2.5, 2)
                meta = {
                    "strategy": "juliet",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "atr_stop_used": snapshot.atr is not None,
                    "rsi_boost_applied": _has("rsi_oversold"),
                    "stoch_boost_applied": _has("stochastic_oversold"),
                    "52w_low_boost_applied": _has("near_52w_low"),
                    "obv_boost_applied": _has("obv_divergence"),
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["price_to_tangible_book"] = score_info.get("price_to_tangible_book")
                    meta["fcf_yield"] = score_info.get("fcf_yield")
                    meta["short_interest"] = score_info.get("short_interest")
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
        logger.info(f"[juliet] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bear":
            logger.info("[juliet] Bear market — extreme value hunting ground, prime conditions")
        elif regime.trend == "bull":
            logger.info("[juliet] Bull market — harder to find value, staying patient")
        else:
            logger.info(f"[juliet] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[juliet] Strategy paused: {reason}")
