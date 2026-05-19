"""Shared indicator computation — pure functions, no signal detection.

Reuses the math from edgefinder/signals/engine.py but returns an
IndicatorSnapshot instead of detecting patterns. Called once per ticker
per cycle by the arena.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import settings
from edgefinder.data.market_data import IndicatorSnapshot

logger = logging.getLogger(__name__)


# ── Pure pandas indicator functions (from signals/engine.py) ──


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    rsi = rsi.where(avg_gain > 0, 0)
    return rsi


def _macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    close: pd.Series, period: int, std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _ema(tr, period)


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, float("nan")))
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, float("nan")))
    di_sum = plus_di + minus_di
    dx = 100 * ((plus_di - minus_di).abs() / di_sum.replace(0, float("nan")))
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _stochastic_rsi(
    close: pd.Series, rsi_period: int, k_period: int, d_period: int
) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, rsi_period)
    rsi_min = rsi.rolling(window=rsi_period).min()
    rsi_max = rsi.rolling(window=rsi_period).max()
    rsi_range = rsi_max - rsi_min
    stoch_rsi = ((rsi - rsi_min) / rsi_range.replace(0, float("nan"))) * 100
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    wr = -100 * ((highest - close) / (highest - lowest).replace(0, float("nan")))
    return wr


# ── Main computation ──────────────────────────────────


MIN_BARS = 30  # need at least 30 bars for meaningful indicators


def compute_indicators_from_bars(df: pd.DataFrame) -> IndicatorSnapshot | None:
    """Compute all technical indicators on daily OHLCV bars.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
            Needs 30+ rows for meaningful results.

    Returns:
        IndicatorSnapshot with latest-bar values, or None if insufficient data.
    """
    if len(df) < MIN_BARS:
        logger.debug("Insufficient data: %d rows (need %d)", len(df), MIN_BARS)
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMAs
    ema_9 = _ema(close, 9)
    ema_21 = _ema(close, 21)
    ema_50 = _ema(close, 50) if len(df) >= 50 else pd.Series([None] * len(df))
    ema_200 = _ema(close, 200) if len(df) >= 200 else pd.Series([None] * len(df))

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

    # ADX
    adx_series, plus_di_series, minus_di_series = _adx(
        high, low, close, settings.signal_adx_period
    )

    # Stochastic RSI
    stoch_k, stoch_d = _stochastic_rsi(
        close, settings.signal_stoch_rsi_period,
        settings.signal_stoch_rsi_k, settings.signal_stoch_rsi_d,
    )

    # Williams %R
    williams = _williams_r(high, low, close, settings.signal_williams_r_period)

    # Volume
    vol_avg = volume.rolling(window=settings.signal_volume_avg_period).mean()
    vol_ratio = volume / vol_avg.replace(0, float("nan"))

    def _safe_float(series, idx=-1):
        try:
            val = series.iloc[idx]
            return float(val) if not pd.isna(val) else None
        except (IndexError, TypeError):
            return None

    return IndicatorSnapshot(
        close=float(close.iloc[-1]),
        open=float(df["open"].iloc[-1]),
        high=float(high.iloc[-1]),
        low=float(low.iloc[-1]),
        volume=float(volume.iloc[-1]),
        ema_9=_safe_float(ema_9),
        ema_21=_safe_float(ema_21),
        ema_50=_safe_float(ema_50),
        ema_200=_safe_float(ema_200),
        rsi=_safe_float(rsi),
        macd_line=_safe_float(macd_line),
        macd_signal=_safe_float(macd_signal),
        macd_histogram=_safe_float(macd_hist),
        bb_upper=_safe_float(bb_upper),
        bb_middle=_safe_float(bb_middle),
        bb_lower=_safe_float(bb_lower),
        bb_width=_safe_float(bb_width_series),
        atr=_safe_float(atr),
        adx=_safe_float(adx_series),
        plus_di=_safe_float(plus_di_series),
        minus_di=_safe_float(minus_di_series),
        stoch_rsi_k=_safe_float(stoch_k),
        stoch_rsi_d=_safe_float(stoch_d),
        williams_r=_safe_float(williams),
        volume_avg=_safe_float(vol_avg),
        volume_ratio=_safe_float(vol_ratio),
    )
