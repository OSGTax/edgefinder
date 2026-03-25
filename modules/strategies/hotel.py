"""
EdgeFinder Strategy: HOTEL — Lynch-Burry Hybrid
=================================================
Stocks that score well on BOTH Lynch and Burry criteria — the overlap zone.
When two completely different investment philosophies agree, conviction is highest.

Fundamental screening: lynch_score >= 60, burry_score >= 60,
composite_score >= 65, low debt.

Technical entry: EMA crossover (day or swing), MACD crossover, volume confirmation.
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


@StrategyRegistry.register("hotel")
class HotelStrategy(BaseStrategy):
    """Lynch-Burry Hybrid — both systems agree, highest conviction."""

    @property
    def name(self) -> str:
        return "hotel"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_day", "ema_crossover_swing", "macd_crossover", "adx_trend", "bollinger_squeeze"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._adx_confidence_boost: float = 5.0
        self._squeeze_confidence_boost: float = 5.0
        self._atr_multiplier: float = 2.0
        self._fallback_risk_pct: float = 0.05
        logger.info("Hotel (Lynch-Burry Hybrid) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Both Lynch and Burry must agree this stock is good."""
        return (
            (stock_data.get("lynch_score") or 0) >= 60
            and (stock_data.get("burry_score") or 0) >= 60
            and (stock_data.get("composite_score") or 0) >= 65
            and (stock_data.get("debt_to_equity") or 999) < 0.7
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Hotel watchlist: {len(self._watchlist)} stocks")

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
                # Confidence boosts
                _has = lambda name: (name in ts.indicators) if isinstance(ts.indicators, dict) else any(ind.get("name") == name for ind in ts.indicators if isinstance(ind, dict))
                if _has("adx_trend"):
                    confidence = min(100.0, confidence + self._adx_confidence_boost)
                if _has("bollinger_squeeze"):
                    confidence = min(100.0, confidence + self._squeeze_confidence_boost)

                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(ticker, confidence)
                        if action == "BLOCK":
                            logger.info(f"[hotel] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[hotel] Sentiment gate error: {e}")
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                # ATR-based dynamic stop-loss
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                target = round(price + (price - stop_loss) * 1.8, 2)
                meta = {
                    "strategy": "hotel",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "atr_stop_used": snapshot.atr is not None,
                    "adx_boost_applied": _has("adx_trend"),
                    "squeeze_boost_applied": _has("bollinger_squeeze"),
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["lynch_score"] = score_info.get("lynch_score")
                    meta["burry_score"] = score_info.get("burry_score")
                    meta["composite_score"] = score_info.get("composite_score")
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
        logger.info(f"[hotel] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bull":
            logger.info("[hotel] Bull market — consensus signals aligning well")
        elif regime.trend == "bear":
            logger.info("[hotel] Bear market — requiring stronger agreement from both Lynch+Burry criteria")
        else:
            logger.info(f"[hotel] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[hotel] Strategy paused: {reason}")
