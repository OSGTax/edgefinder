"""
EdgeFinder Strategy: BRAVO — Lynch Stalwart
=============================================
Large, reliable growers — Lynch's "hold through anything" picks.
Capital preservation with steady returns, portfolio anchor.

Fundamental screening: lynch_category == stalwart, large cap,
moderate earnings growth, low debt, balanced institutional ownership.

Technical entry: EMA crossovers (swing), MACD crossover.
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


@StrategyRegistry.register("bravo")
class BravoStrategy(BaseStrategy):
    """Lynch Stalwart — large cap steady growers."""

    @property
    def name(self) -> str:
        return "bravo"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_swing", "macd_crossover", "adx_trend", "near_52w_high"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._atr_multiplier: float = 1.5
        self._fallback_risk_pct: float = 0.04
        self._adx_confidence_boost: float = 5.0
        logger.info("Bravo (Lynch Stalwart) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Stalwart: large cap, moderate growth, low debt, balanced ownership."""
        eg = stock_data.get("earnings_growth") or 0
        inst = stock_data.get("institutional_pct") or 0
        return (
            stock_data.get("lynch_category") == "stalwart"
            and (stock_data.get("market_cap") or 0) >= 10_000_000_000
            and 0.10 <= eg <= 0.20
            and (stock_data.get("debt_to_equity") or 999) < 0.6
            and 0.30 <= inst <= 0.70
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(f"Bravo watchlist: {len(self._watchlist)} stocks")

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
                            logger.info(f"[bravo] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[bravo] Sentiment gate error: {e}")

                # Confidence boost: ADX trend confirms steady trend for stalwarts
                has_adx = _has_indicator(ts.indicators, "adx_trend")
                if has_adx:
                    confidence = min(100.0, confidence + self._adx_confidence_boost)

                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue

                # ATR-based dynamic stop-loss
                if snapshot.atr and snapshot.atr > 0:
                    stop_loss = round(price - (snapshot.atr * self._atr_multiplier), 2)
                else:
                    stop_loss = round(price * (1 - self._fallback_risk_pct), 2)
                target = round(price + (price - stop_loss) * 1.5, 2)
                meta = {
                    "strategy": "bravo",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "adx_boost_applied": has_adx,
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["lynch_score"] = score_info.get("lynch_score")
                    meta["lynch_category"] = score_info.get("lynch_category")
                    meta["market_cap"] = score_info.get("market_cap")
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
        logger.info(f"[bravo] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        if regime.trend == "bull":
            logger.info("[bravo] Bull market — steady gains expected from stalwarts")
        elif regime.trend == "bear":
            logger.info("[bravo] Bear market — stalwarts hold up best")
        else:
            logger.info(f"[bravo] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[bravo] Strategy paused: {reason}")
