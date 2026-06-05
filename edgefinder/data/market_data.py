"""Shared market data objects for strategy consumption.

MarketData bundles current indicators, 30-day history, fundamentals,
and market context into a single object that all strategies receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from edgefinder.core.models import TickerFundamentals


@dataclass
class IndicatorSnapshot:
    """All indicator values for a single daily bar (or provisional bar)."""

    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    ema_9: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    rsi: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    atr: float | None = None
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    stoch_rsi_k: float | None = None
    stoch_rsi_d: float | None = None
    williams_r: float | None = None
    volume_avg: float | None = None
    volume_ratio: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class IndicatorHistory:
    """Rolling buffer of daily IndicatorSnapshots (max N days)."""

    def __init__(self, max_days: int = 30) -> None:
        self._max_days = max_days
        self._snapshots: list[IndicatorSnapshot] = []

    def add(self, snapshot: IndicatorSnapshot) -> None:
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_days:
            self._snapshots = self._snapshots[-self._max_days:]

    @property
    def latest(self) -> IndicatorSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def previous(self) -> IndicatorSnapshot | None:
        return self._snapshots[-2] if len(self._snapshots) >= 2 else None

    @property
    def oldest(self) -> IndicatorSnapshot | None:
        return self._snapshots[0] if self._snapshots else None

    def get_field_series(self, field_name: str) -> list[float | None]:
        """Get a list of values for a specific field across all snapshots."""
        return [getattr(s, field_name, None) for s in self._snapshots]

    def __len__(self) -> int:
        return len(self._snapshots)


@dataclass
class MarketContext:
    """Broad market state shared across all tickers."""

    spy_price: float = 0.0
    spy_change_pct: float = 0.0
    qqq_price: float = 0.0
    qqq_change_pct: float = 0.0
    iwm_price: float = 0.0
    iwm_change_pct: float = 0.0
    dia_price: float = 0.0
    dia_change_pct: float = 0.0
    vix_level: float = 0.0
    market_regime: str = "sideways"
    sector_performance: dict = field(default_factory=dict)
    # SPY trend state (0.0/None = unknown — strategies must treat absence as
    # "no information", NOT as bearish). Live fills these from daily_bars;
    # the backtester from its SPY series, so regime-gated entries are
    # testable with the same semantics in both.
    spy_sma_200: float = 0.0

    @property
    def spy_uptrend(self) -> bool | None:
        """True/False when both SPY price and its 200dma are known; else None."""
        if self.spy_price > 0 and self.spy_sma_200 > 0:
            return self.spy_price > self.spy_sma_200
        return None


@dataclass
class MarketData:
    """Everything a strategy needs to make a decision for one ticker."""

    ticker: str
    current: IndicatorSnapshot
    history: IndicatorHistory
    fundamentals: TickerFundamentals | None
    context: MarketContext
    current_price: float = 0.0
    today_volume: float = 0.0
    avg_daily_volume: float = 0.0
    volume_ratio: float = 0.0  # time-normalized
