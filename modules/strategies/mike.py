"""
EdgeFinder Strategy: MIKE — Cash Flow Compounder
==================================================
High FCF + low debt = self-funding growth machines.
Companies that don't need Wall Street's money to grow.

Fundamental screening: FCF yield >= 7%, very low debt,
cheap EV/EBITDA, positive earnings growth.

Technical entry: EMA crossover (swing), MACD crossover, Bollinger breakout,
Stochastic oversold.
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


@StrategyRegistry.register("mike")
class MikeStrategy(BaseStrategy):
    """Cash Flow Compounder — self-funding growers, no dilution."""

    @property
    def name(self) -> str:
        return "mike"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_swing", "macd_crossover", "bollinger_breakout", "stochastic_oversold"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._atr_multiplier: float = 2.0
        self._fallback_risk_pct: float = 0.05
        self._hybrid_confidence_boost: float = 5.0
        logger.info("Mike (Cash Flow Compounder) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """High FCF, low debt, cheap, growing."""
        return (
            (stock_data.get("fcf_yield") or 0) >= 0.07
            and (stock_data.get("debt_to_equity") or 999) < 0.5
            and (stock_data.get("ev_to_ebitda") or 999) < 12
            and (stock_data.get("earnings_growth") or 0) >= 0.10
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Mike watchlist: {len(self._watchlist)} stocks")

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
                            logger.info(f"[mike] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[mike] Sentiment gate error: {e}")
                # Confidence boost: hybrid signal (bollinger_breakout OR stochastic_oversold)
                _has = lambda name: (name in ts.indicators) if isinstance(ts.indicators, dict) else any(ind.get("name") == name for ind in ts.indicators if isinstance(ind, dict))
                hybrid_boost_applied = False
                if _has("bollinger_breakout") or _has("stochastic_oversold"):
                    confidence = min(100.0, confidence + self._hybrid_confidence_boost)
                    hybrid_boost_applied = True
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                # ATR-based stop-loss
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                    atr_stop_used = True
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                    atr_stop_used = False
                target = round(price + (price - stop_loss) * 1.8, 2)
                meta = {
                    "strategy": "mike",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "atr_stop_used": atr_stop_used,
                    "hybrid_boost_applied": hybrid_boost_applied,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["fcf_yield"] = score_info.get("fcf_yield")
                    meta["debt_to_equity"] = score_info.get("debt_to_equity")
                    meta["earnings_growth"] = score_info.get("earnings_growth")
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
        logger.info(f"[mike] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "sideways":
            logger.info("[mike] Sideways market — range-bound suits FCF+growth hybrid")
        elif regime.trend == "bull":
            logger.info("[mike] Bull market — FCF compounders should outperform")
        elif regime.trend == "bear":
            logger.info("[mike] Bear market — FCF strength provides downside resilience")
        else:
            logger.info(f"[mike] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[mike] Strategy paused: {reason}")
