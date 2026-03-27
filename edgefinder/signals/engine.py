"""EdgeFinder v2 — Technical signal engine.

Pure pandas indicator calculations + signal detection.
No pandas-ta dependency — everything computed from scratch.
All thresholds come from config/settings.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import settings
from edgefinder.core.models import Signal, SignalAction, TradeType

logger = logging.getLogger(__name__)


# ── Indicator Snapshot ───────────────────────────────────


@dataclass
class IndicatorSnapshot:
    """All computed indicator values for the latest bar."""

    close: float
    prev_close: float | None = None
    ema_fast_day: float | None = None
    ema_slow_day: float | None = None
    prev_ema_fast_day: float | None = None
    prev_ema_slow_day: float | None = None
    ema_fast_swing: float | None = None
    ema_slow_swing: float | None = None
    rsi: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    prev_macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    atr: float | None = None
    volume_avg: float | None = None
    volume_ratio: float | None = None
    recent_low: float | None = None
    recent_high: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── Pure pandas indicator functions ──────────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # When avg_loss is 0 (all gains), RSI = 100. When avg_gain is 0, RSI = 0.
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)  # all gains → RSI 100
    rsi = rsi.where(avg_gain > 0, 0)  # all losses → RSI 0
    return rsi


def _macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    close: pd.Series, period: int, std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle, upper, lower."""
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _ema(tr, period)


# ── Main compute function ───────────────────────────────


def compute_indicators(df: pd.DataFrame) -> IndicatorSnapshot | None:
    """Compute all technical indicators on OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
            Index is DatetimeIndex. Needs ~200+ rows for swing EMAs.

    Returns:
        IndicatorSnapshot with latest-bar values, or None if insufficient data.
    """
    min_rows = max(settings.signal_ema_slow_swing, 200)
    if len(df) < min_rows:
        logger.debug("Insufficient data: %d rows (need %d)", len(df), min_rows)
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMAs
    ema_fast_day = _ema(close, settings.signal_ema_fast_day)
    ema_slow_day = _ema(close, settings.signal_ema_slow_day)
    ema_fast_swing = _ema(close, settings.signal_ema_fast_swing)
    ema_slow_swing = _ema(close, settings.signal_ema_slow_swing)

    # RSI
    rsi = _rsi(close, settings.signal_rsi_period)

    # MACD
    macd_line, macd_signal, macd_hist = _macd(
        close, settings.signal_macd_fast, settings.signal_macd_slow, settings.signal_macd_signal
    )

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = _bollinger_bands(
        close, settings.signal_bb_period, settings.signal_bb_std
    )
    bb_width_series = (bb_upper - bb_lower) / bb_middle.replace(0, float("nan"))

    # ATR
    atr = _atr(high, low, close, settings.signal_atr_period)

    # Volume
    vol_avg = volume.rolling(window=settings.signal_volume_avg_period).mean()
    vol_ratio = volume / vol_avg.replace(0, float("nan"))

    # Recent swing points (last 10 bars)
    recent_low = low.iloc[-10:].min()
    recent_high = high.iloc[-10:].max()

    return IndicatorSnapshot(
        close=float(close.iloc[-1]),
        prev_close=float(close.iloc[-2]) if len(close) >= 2 else None,
        ema_fast_day=float(ema_fast_day.iloc[-1]),
        ema_slow_day=float(ema_slow_day.iloc[-1]),
        prev_ema_fast_day=float(ema_fast_day.iloc[-2]),
        prev_ema_slow_day=float(ema_slow_day.iloc[-2]),
        ema_fast_swing=float(ema_fast_swing.iloc[-1]),
        ema_slow_swing=float(ema_slow_swing.iloc[-1]),
        rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        macd_line=float(macd_line.iloc[-1]),
        macd_signal=float(macd_signal.iloc[-1]),
        macd_histogram=float(macd_hist.iloc[-1]),
        prev_macd_histogram=float(macd_hist.iloc[-2]),
        bb_upper=float(bb_upper.iloc[-1]),
        bb_middle=float(bb_middle.iloc[-1]),
        bb_lower=float(bb_lower.iloc[-1]),
        bb_width=float(bb_width_series.iloc[-1]) if not pd.isna(bb_width_series.iloc[-1]) else None,
        atr=float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
        volume_avg=float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else None,
        volume_ratio=float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else None,
        recent_low=float(recent_low),
        recent_high=float(recent_high),
    )


# ── Signal detection ─────────────────────────────────────


def detect_signals(indicators: IndicatorSnapshot, ticker: str) -> list[Signal]:
    """Detect actionable signals from indicator state.

    Returns 0-N Signal objects with confidence, entry/stop/target.
    """
    signals: list[Signal] = []
    now = datetime.utcnow()

    is_volume_spike = (
        indicators.volume_ratio is not None
        and indicators.volume_ratio > settings.signal_volume_spike_multiplier
    )
    is_bb_squeeze = (
        indicators.bb_width is not None
        and indicators.bb_width < settings.signal_bb_squeeze_threshold
    )

    # ── EMA Crossover Bullish ──
    sig = _detect_ema_crossover_bullish(indicators, ticker, now, is_volume_spike)
    if sig:
        signals.append(sig)

    # ── EMA Crossover Bearish ──
    sig = _detect_ema_crossover_bearish(indicators, ticker, now, is_volume_spike)
    if sig:
        signals.append(sig)

    # ── RSI Oversold ──
    sig = _detect_rsi_oversold(indicators, ticker, now, is_volume_spike, is_bb_squeeze)
    if sig:
        signals.append(sig)

    # ── RSI Overbought ──
    sig = _detect_rsi_overbought(indicators, ticker, now, is_volume_spike)
    if sig:
        signals.append(sig)

    # ── MACD Bullish Cross ──
    sig = _detect_macd_bullish_cross(indicators, ticker, now, is_volume_spike)
    if sig:
        signals.append(sig)

    # ── MACD Bearish Cross ──
    sig = _detect_macd_bearish_cross(indicators, ticker, now, is_volume_spike)
    if sig:
        signals.append(sig)

    # ── BB Lower Touch ──
    sig = _detect_bb_lower_touch(indicators, ticker, now, is_volume_spike)
    if sig:
        signals.append(sig)

    # ── Volume Spike (standalone) ──
    sig = _detect_volume_spike(indicators, ticker, now)
    if sig:
        signals.append(sig)

    return signals


# ── Individual pattern detectors ─────────────────────────


def _make_buy_signal(
    ticker: str,
    timestamp: datetime,
    close: float,
    atr: float | None,
    confidence: float,
    trade_type: TradeType,
    pattern: str,
    indicators: IndicatorSnapshot,
    recent_low: float | None = None,
) -> Signal:
    """Helper to build a BUY signal with entry/stop/target."""
    atr_val = atr or close * 0.02  # fallback 2%
    stop = (recent_low or close) - 0.5 * atr_val
    risk = close - stop
    target = close + max(2.0 * risk, atr_val)
    return Signal(
        ticker=ticker,
        action=SignalAction.BUY,
        entry_price=close,
        stop_loss=round(stop, 2),
        target=round(target, 2),
        confidence=min(confidence, 100),
        trade_type=trade_type,
        indicators=indicators.to_dict(),
        metadata={"pattern": pattern},
        timestamp=timestamp,
    )


def _make_sell_signal(
    ticker: str,
    timestamp: datetime,
    close: float,
    atr: float | None,
    confidence: float,
    trade_type: TradeType,
    pattern: str,
    indicators: IndicatorSnapshot,
    recent_high: float | None = None,
) -> Signal:
    """Helper to build a SELL signal with entry/stop/target."""
    atr_val = atr or close * 0.02
    stop = (recent_high or close) + 0.5 * atr_val
    risk = stop - close
    target = close - max(2.0 * risk, atr_val)
    return Signal(
        ticker=ticker,
        action=SignalAction.SELL,
        entry_price=close,
        stop_loss=round(stop, 2),
        target=round(target, 2),
        confidence=min(confidence, 100),
        trade_type=trade_type,
        indicators=indicators.to_dict(),
        metadata={"pattern": pattern},
        timestamp=timestamp,
    )


def _detect_ema_crossover_bullish(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool
) -> Signal | None:
    if (
        ind.prev_ema_fast_day is not None
        and ind.prev_ema_slow_day is not None
        and ind.prev_ema_fast_day <= ind.prev_ema_slow_day
        and ind.ema_fast_day is not None
        and ind.ema_slow_day is not None
        and ind.ema_fast_day > ind.ema_slow_day
    ):
        conf = settings.signal_confidence_moderate
        if vol_spike:
            conf += 10
        if ind.rsi is not None and ind.rsi < 60:
            conf += 10
        return _make_buy_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.DAY,
            "ema_crossover_bullish", ind, ind.recent_low,
        )
    return None


def _detect_ema_crossover_bearish(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool
) -> Signal | None:
    if (
        ind.prev_ema_fast_day is not None
        and ind.prev_ema_slow_day is not None
        and ind.prev_ema_fast_day >= ind.prev_ema_slow_day
        and ind.ema_fast_day is not None
        and ind.ema_slow_day is not None
        and ind.ema_fast_day < ind.ema_slow_day
    ):
        conf = settings.signal_confidence_moderate
        if vol_spike:
            conf += 10
        if ind.rsi is not None and ind.rsi > 40:
            conf += 10
        return _make_sell_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.DAY,
            "ema_crossover_bearish", ind, ind.recent_high,
        )
    return None


def _detect_rsi_oversold(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool, bb_squeeze: bool
) -> Signal | None:
    if ind.rsi is not None and ind.rsi < settings.signal_rsi_oversold:
        conf = settings.signal_confidence_moderate
        if vol_spike:
            conf += 10
        if ind.bb_lower is not None and ind.close <= ind.bb_lower:
            conf += 10
        if bb_squeeze:
            conf += 10
        target = ind.bb_middle if ind.bb_middle else ind.close * 1.05
        return _make_buy_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.SWING,
            "rsi_oversold", ind, ind.recent_low,
        )
    return None


def _detect_rsi_overbought(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool
) -> Signal | None:
    if ind.rsi is not None and ind.rsi > settings.signal_rsi_overbought:
        conf = settings.signal_confidence_moderate
        if vol_spike:
            conf += 10
        if ind.bb_upper is not None and ind.close >= ind.bb_upper:
            conf += 10
        return _make_sell_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.SWING,
            "rsi_overbought", ind, ind.recent_high,
        )
    return None


def _detect_macd_bullish_cross(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool
) -> Signal | None:
    if (
        ind.prev_macd_histogram is not None
        and ind.macd_histogram is not None
        and ind.prev_macd_histogram < 0
        and ind.macd_histogram >= 0
    ):
        conf = settings.signal_confidence_low
        if ind.macd_line is not None and ind.macd_line > 0:
            conf += 20
        if vol_spike:
            conf += 10
        return _make_buy_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.DAY,
            "macd_bullish_cross", ind, ind.recent_low,
        )
    return None


def _detect_macd_bearish_cross(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool
) -> Signal | None:
    if (
        ind.prev_macd_histogram is not None
        and ind.macd_histogram is not None
        and ind.prev_macd_histogram > 0
        and ind.macd_histogram <= 0
    ):
        conf = settings.signal_confidence_low
        if ind.macd_line is not None and ind.macd_line < 0:
            conf += 20
        if vol_spike:
            conf += 10
        return _make_sell_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.DAY,
            "macd_bearish_cross", ind, ind.recent_high,
        )
    return None


def _detect_bb_lower_touch(
    ind: IndicatorSnapshot, ticker: str, now: datetime, vol_spike: bool
) -> Signal | None:
    if ind.bb_lower is not None and ind.close <= ind.bb_lower:
        conf = settings.signal_confidence_moderate
        if ind.rsi is not None and ind.rsi < 40:
            conf += 10
        if vol_spike:
            conf += 10
        return _make_buy_signal(
            ticker, now, ind.close, ind.atr, conf, TradeType.SWING,
            "bb_lower_touch", ind, ind.recent_low,
        )
    return None


def _detect_volume_spike(
    ind: IndicatorSnapshot, ticker: str, now: datetime
) -> Signal | None:
    if (
        ind.volume_ratio is not None
        and ind.volume_ratio > settings.signal_volume_spike_multiplier
        and ind.prev_close is not None
    ):
        conf = settings.signal_confidence_low
        if ind.close > ind.prev_close:
            return _make_buy_signal(
                ticker, now, ind.close, ind.atr, conf, TradeType.DAY,
                "volume_spike_bullish", ind, ind.recent_low,
            )
        elif ind.close < ind.prev_close:
            return _make_sell_signal(
                ticker, now, ind.close, ind.atr, conf, TradeType.DAY,
                "volume_spike_bearish", ind, ind.recent_high,
            )
    return None
