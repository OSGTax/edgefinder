"""
EdgeFinder Strategy: OSCAR — GARP Conservative
================================================
Growth at a reasonable price — steady, no fireworks, capital preservation.
The boring strategy that quietly wins.

Fundamental screening: PEG between 0.5-1.5, moderate earnings growth,
low debt, balanced institutional ownership.

Technical entry: EMA crossover (swing), MACD confirmation, near 52-week high,
ADX trend strength.
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


@StrategyRegistry.register("oscar")
class OscarStrategy(BaseStrategy):
    """GARP Conservative — steady growth, low volatility, consistent."""

    @property
    def name(self) -> str:
        return "oscar"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_swing", "macd_crossover", "near_52w_high", "adx_trend"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._adx_confidence_boost: float = 5.0
        self._atr_multiplier: float = 1.5
        self._fallback_risk_pct: float = 0.04
        logger.info("Oscar (GARP Conservative) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Reasonable PEG, moderate growth, low debt, not over-owned."""
        peg = stock_data.get("peg_ratio") or 999
        eg = stock_data.get("earnings_growth") or 0
        return (
            0.5 <= peg <= 1.5
            and 0.10 <= eg <= 0.30
            and (stock_data.get("debt_to_equity") or 999) < 0.7
            and (stock_data.get("institutional_pct") or 1.0) < 0.65
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Oscar watchlist: {len(self._watchlist)} stocks")

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
                            logger.info(f"[oscar] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[oscar] Sentiment gate error: {e}")
                # Confidence boost: ADX trend strength for conservative play
                _has = lambda name: (name in ts.indicators) if isinstance(ts.indicators, dict) else any(ind.get("name") == name for ind in ts.indicators if isinstance(ind, dict))
                adx_boost_applied = False
                if _has("adx_trend"):
                    confidence = min(100.0, confidence + self._adx_confidence_boost)
                    adx_boost_applied = True
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                # ATR-based stop-loss
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                    atr_stop_used = True
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                    atr_stop_used = False
                target = round(price + (price - stop_loss) * 1.5, 2)
                meta = {
                    "strategy": "oscar",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "adx_boost_applied": adx_boost_applied,
                    "atr_stop_used": atr_stop_used,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["peg_ratio"] = score_info.get("peg_ratio")
                    meta["earnings_growth"] = score_info.get("earnings_growth")
                    meta["debt_to_equity"] = score_info.get("debt_to_equity")
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
        logger.info(f"[oscar] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bull":
            logger.info("[oscar] Bull market — conservative plays deliver steady returns")
        elif regime.trend == "bear":
            logger.info("[oscar] Bear market — capital preservation priority")
        else:
            logger.info(f"[oscar] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[oscar] Strategy paused: {reason}")
