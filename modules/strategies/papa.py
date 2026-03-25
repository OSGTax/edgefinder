"""
EdgeFinder Strategy: PAPA — Short Squeeze Candidate
=====================================================
Heavily shorted + improving fundamentals = squeeze setup.
Unlike Golf (bets bears are wrong), Papa actively hunts the squeeze
mechanics via volume explosion.

Fundamental screening: short_interest >= 20%, positive revenue growth,
adequate liquidity, reasonable EV/EBITDA.

Technical entry: RSI oversold, EMA crossover (day), Bollinger squeeze,
OBV divergence, near 52-week low.
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


@StrategyRegistry.register("papa")
class PapaStrategy(BaseStrategy):
    """Short Squeeze Candidate — volume-triggered squeeze play."""

    @property
    def name(self) -> str:
        return "papa"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"rsi_oversold", "ema_crossover_day", "bollinger_squeeze", "obv_divergence", "near_52w_low"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._rsi_confidence_boost: float = 10.0
        self._squeeze_obv_boost: float = 10.0
        self._52w_low_confidence_boost: float = 5.0
        self._atr_multiplier: float = 3.0
        self._fallback_risk_pct: float = 0.08
        logger.info("Papa (Short Squeeze Candidate) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Very heavily shorted with improving fundamentals."""
        return (
            (stock_data.get("short_interest") or 0) >= 0.20
            and (stock_data.get("revenue_growth") or 0) >= 0.05
            and (stock_data.get("current_ratio") or 0) >= 1.2
            and (stock_data.get("ev_to_ebitda") or 999) < 15
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Papa watchlist: {len(self._watchlist)} stocks")

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
                # Confidence boosts for squeeze signals
                _has = lambda name: (name in ts.indicators) if isinstance(ts.indicators, dict) else any(ind.get("name") == name for ind in ts.indicators if isinstance(ind, dict))
                rsi_boost_applied = False
                squeeze_obv_boost_applied = False
                near_52w_low_boost_applied = False
                if _has("rsi_oversold"):
                    confidence = min(100.0, confidence + self._rsi_confidence_boost)
                    rsi_boost_applied = True
                if _has("bollinger_squeeze") and _has("obv_divergence"):
                    confidence = min(100.0, confidence + self._squeeze_obv_boost)
                    squeeze_obv_boost_applied = True
                if _has("near_52w_low"):
                    confidence = min(100.0, confidence + self._52w_low_confidence_boost)
                    near_52w_low_boost_applied = True
                if self._use_sentiment:
                    try:
                        from modules.sentiment import gate_trade
                        action, adjusted_confidence, _ = gate_trade(ticker, confidence)
                        if action == "BLOCK":
                            logger.info(f"[papa] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[papa] Sentiment gate error: {e}")
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                # ATR-based stop-loss
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                    atr_stop_used = True
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                    atr_stop_used = False
                target = round(price + (price - stop_loss) * 3.0, 2)  # Highest R:R target
                meta = {
                    "strategy": "papa",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "atr_stop_used": atr_stop_used,
                    "rsi_boost_applied": rsi_boost_applied,
                    "squeeze_obv_boost_applied": squeeze_obv_boost_applied,
                    "near_52w_low_boost_applied": near_52w_low_boost_applied,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["short_interest"] = score_info.get("short_interest")
                    meta["revenue_growth"] = score_info.get("revenue_growth")
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
        logger.info(f"[papa] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.volatility == "high":
            logger.info("[papa] High volatility — prime squeeze territory")
        elif regime.trend == "bear":
            logger.info("[papa] Bear market — shorts pile on, more squeeze candidates")
        else:
            logger.info(f"[papa] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[papa] Strategy paused: {reason}")
