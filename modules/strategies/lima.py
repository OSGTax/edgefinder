"""
EdgeFinder Strategy: LIMA — Small Cap Quality
===============================================
Smaller companies with strong fundamentals, institutions just starting
to accumulate. Catches stocks before the crowd.

Fundamental screening: market_cap < $2B, low institutional ownership
(10-40%), revenue_growth >= 15%, low debt.

Technical entry: Volume spike (institutional buying footprint), EMA crossover (day).
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


@StrategyRegistry.register("lima")
class LimaStrategy(BaseStrategy):
    """Small Cap Quality — pre-institutional discovery phase."""

    @property
    def name(self) -> str:
        return "lima"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"volume_spike", "ema_crossover_day"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        logger.info("Lima (Small Cap Quality) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Small cap, low institutional ownership, strong revenue growth."""
        inst = stock_data.get("institutional_pct") or 0
        return (
            (stock_data.get("market_cap") or 0) < 2_000_000_000
            and 0.10 <= inst <= 0.40
            and (stock_data.get("revenue_growth") or 0) >= 0.15
            and (stock_data.get("debt_to_equity") or 999) < 0.8
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Lima watchlist: {len(self._watchlist)} stocks")

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
                            logger.info(f"[lima] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[lima] Sentiment gate error: {e}")
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                risk_pct = 0.06
                stop_loss = round(price * (1 - risk_pct), 2)
                target = round(price + (price - stop_loss) * 2.0, 2)
                meta = {
                    "strategy": "lima",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["market_cap"] = score_info.get("market_cap")
                    meta["institutional_pct"] = score_info.get("institutional_pct")
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
        logger.info(f"[lima] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        logger.info(f"[lima] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[lima] Strategy paused: {reason}")
