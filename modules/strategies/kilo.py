"""
EdgeFinder Strategy: KILO — Sector Momentum Rotation
======================================================
Strongest fundamentals within the best-performing sectors.
Rides institutional capital rotation waves.

Fundamental screening: composite_score >= 55, earnings_growth >= 10%,
stock is in a top-performing sector by average composite score.

Technical entry: EMA crossover (swing), MACD crossover.
Sentiment gate: blocks trades on strongly negative news.
"""

import logging
from collections import defaultdict

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


@StrategyRegistry.register("kilo")
class KiloStrategy(BaseStrategy):
    """Sector Momentum Rotation — best stocks in best sectors."""

    @property
    def name(self) -> str:
        return "kilo"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def preferred_signals(self) -> set[str]:
        return {"ema_crossover_swing", "macd_crossover"}

    def init(self) -> None:
        self._watchlist: list[str] = []
        self._scores: dict[str, dict] = {}
        self._trades_log: list[TradeNotification] = []
        self._use_sentiment: bool = True
        self._top_sectors: set[str] = set()
        self._num_top_sectors: int = 2
        logger.info("Kilo (Sector Momentum Rotation) strategy initialized")

    def qualifies_stock(self, stock_data: dict) -> bool:
        """Good composite score, growing earnings, in a top sector."""
        if not self._top_sectors:
            # Before set_watchlist is called, use basic filter
            return (
                (stock_data.get("composite_score") or 0) >= 55
                and (stock_data.get("earnings_growth") or 0) >= 0.10
            )
        return (
            (stock_data.get("composite_score") or 0) >= 55
            and (stock_data.get("earnings_growth") or 0) >= 0.10
            and stock_data.get("sector", "") in self._top_sectors
        )

    def set_watchlist(self, scored_stocks: list[dict]) -> None:
        # First, compute sector averages to find top sectors
        sector_scores: dict[str, list[float]] = defaultdict(list)
        for stock in scored_stocks:
            sector = stock.get("sector", "")
            score = stock.get("composite_score") or 0
            if sector and score > 0:
                sector_scores[sector].append(score)

        sector_avgs = {
            sector: sum(scores) / len(scores)
            for sector, scores in sector_scores.items()
            if scores
        }
        sorted_sectors = sorted(sector_avgs, key=sector_avgs.get, reverse=True)
        self._top_sectors = set(sorted_sectors[:self._num_top_sectors])

        # Now filter stocks in top sectors
        self._watchlist = []
        self._scores = {}
        for stock in scored_stocks:
            if self.qualifies_stock(stock):
                ticker = stock["ticker"]
                self._watchlist.append(ticker)
                self._scores[ticker] = stock
        logger.info(
            f"Kilo watchlist: {len(self._watchlist)} stocks "
            f"(top sectors: {self._top_sectors})"
        )

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
                            logger.info(f"[kilo] {ticker}: blocked by sentiment")
                            continue
                        confidence = adjusted_confidence
                    except Exception as e:
                        logger.debug(f"[kilo] Sentiment gate error: {e}")
                if confidence < settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE:
                    continue
                risk_pct = 0.06
                stop_loss = round(price * (1 - risk_pct), 2)
                target = round(price + (price - stop_loss) * 1.5, 2)
                meta = {
                    "strategy": "kilo",
                    "indicators": ts.indicators,
                    "trade_reason": ts.reason,
                    "top_sectors": list(self._top_sectors),
                }
                score_info = self._scores.get(ticker, {})
                if score_info:
                    meta["composite_score"] = score_info.get("composite_score")
                    meta["sector"] = score_info.get("sector")
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
        logger.info(f"[kilo] Trade: {notification.action} {notification.ticker} @ ${notification.entry_price:.2f}")

    def on_market_regime_change(self, regime: MarketRegime) -> None:
        logger.info(f"[kilo] Market regime: {regime.trend}/{regime.volatility}")

    def on_strategy_pause(self, reason: str) -> None:
        logger.warning(f"[kilo] Strategy paused: {reason}")
