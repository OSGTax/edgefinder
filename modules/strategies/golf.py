"""
EdgeFinder Strategy: GOLF — Burry Contrarian
==============================================
Heavily shorted stocks where Burry's fundamentals say the bears are wrong.
When short sellers are wrong AND fundamentals are sound, the unwind creates
outsized moves.

Fundamental screening: high short interest, positive FCF, cheap EV/EBITDA,
adequate liquidity, decent Burry score.

Technical entry: RSI oversold, volume spike, EMA crossover (day).
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


@StrategyRegistry.register("golf")
class GolfStrategy(BaseStrategy):
    """Burry Contrarian — high short interest + sound fundamentals."""

    @property
    def name(self) -> str:
        return "golf"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"rsi_oversold", "volume_spike", "ema_crossover_day"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        logger.info("Golf (Burry Contrarian) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Heavily shorted but fundamentally sound."""
        return (
            (stock_data.get("short_interest") or 0) >= 0.15
            and (stock_data.get("fcf_yield") or 0) >= 0.05
            and (stock_data.get("ev_to_ebitda") or 999) < 12
            and (stock_data.get("current_ratio") or 0) >= 1.3
            and (stock_data.get("burry_score") or 0) >= 50
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Golf watchlist: {len(self._watchlist)} stocks")

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
                            logger.info(f"[golf] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[golf] Sentiment gate error: {e}")
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                risk_pct = 0.07  # Wider stop for contrarian plays
                stop_loss = round(price * (1 - risk_pct), 2)
                target = round(price + (price - stop_loss) * 2.5, 2)
                meta = {
                    "strategy": "golf",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["burry_score"] = score_info.get("burry_score")
                    meta["short_interest"] = score_info.get("short_interest")
                    meta["fcf_yield"] = score_info.get("fcf_yield")
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
        logger.info(f"[golf] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bear":
            logger.info("[golf] Bear market — contrarian setups multiplying")
        else:
            logger.info(f"[golf] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[golf] Strategy paused: {reason}")
