"""
EdgeFinder Module 2: Technical Signal Engine
=============================================
Monitors watchlist candidates for entry signals using technical indicators:
- EMA crossovers (9/21 for day trades, 50/200 for swing trades)
- RSI oversold/overbought
- MACD crossovers
- Volume spikes

Runs every 15-30 minutes during market hours.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import pandas_ta as ta
    # Verify pandas_ta is real (not a MagicMock from test fixtures)
    _TA_AVAILABLE = callable(getattr(ta, "ema", None)) and not hasattr(ta.ema, "_mock_name")
except ImportError:
    _TA_AVAILABLE = False

import numpy as np
import yfinance as yf

from config import settings
from modules.database import Signal as SignalRecord, get_session, init_db

logger = logging.getLogger(__name__)


# ── PURE PANDAS/NUMPY FALLBACKS (when pandas_ta unavailable) ─────

def _fallback_ema(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Compute EMA using pandas ewm (no pandas_ta required)."""
    if series is None or len(series) < length:
        return None
    return series.ewm(span=length, adjust=False).mean()


def _fallback_rsi(series: pd.Series, length: int = 14) -> Optional[pd.Series]:
    """Compute RSI using pure pandas (no pandas_ta required)."""
    if series is None or len(series) < length + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _fallback_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Optional[pd.DataFrame]:
    """Compute MACD using pure pandas (no pandas_ta required)."""
    if series is None or len(series) < slow + signal:
        return None
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDs_{fast}_{slow}_{signal}": macd_signal,
        f"MACDh_{fast}_{slow}_{signal}": macd_hist,
    })


def _fallback_sma(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Compute SMA using pandas rolling (no pandas_ta required)."""
    if series is None or len(series) < length:
        return None
    return series.rolling(window=length).mean()


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    """Computed technical indicators for a single ticker at a point in time."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # EMAs
    ema_fast_day: Optional[float] = None       # 9-day
    ema_slow_day: Optional[float] = None       # 21-day
    ema_fast_swing: Optional[float] = None     # 50-day
    ema_slow_swing: Optional[float] = None     # 200-day

    # RSI
    rsi: Optional[float] = None                # 14-period

    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Volume
    current_volume: Optional[float] = None
    avg_volume: Optional[float] = None         # 20-day average

    # Price context
    current_price: Optional[float] = None
    prev_close: Optional[float] = None

    # Previous values for crossover detection
    prev_ema_fast_day: Optional[float] = None
    prev_ema_slow_day: Optional[float] = None
    prev_ema_fast_swing: Optional[float] = None
    prev_ema_slow_swing: Optional[float] = None
    prev_macd_line: Optional[float] = None
    prev_macd_signal: Optional[float] = None


@dataclass
class TradeSignal:
    """A detected trading signal with confidence scoring."""
    ticker: str
    signal_type: str            # "BUY" or "SELL"
    trade_type: str             # "DAY" or "SWING"
    confidence: float           # 0-100
    indicators: dict            # Which indicators fired and their values
    timestamp: datetime = field(default_factory=datetime.utcnow)
    price: float = 0.0
    reason: str = ""            # Human-readable summary


# ── INDICATOR CALCULATION ────────────────────────────────────

def compute_indicators(df: pd.DataFrame, ticker: str = "") -> Optional[IndicatorSnapshot]:
    """
    Compute all technical indicators from OHLCV price history.

    Args:
        df: DataFrame with columns: Open, High, Low, Close, Volume.
            Must have at least 200 rows for swing EMAs.
        ticker: Stock ticker symbol for logging.

    Returns:
        IndicatorSnapshot with all computed values, or None if insufficient data.
    """
    if df is None or df.empty:
        logger.debug(f"{ticker}: No price data available")
        return None

    min_rows = settings.SIGNAL_EMA_SLOW_SWING + 1  # Need 201 rows for 200-day EMA
    if len(df) < min_rows:
        logger.debug(f"{ticker}: Insufficient data ({len(df)} rows, need {min_rows})")
        return None

    snap = IndicatorSnapshot(ticker=ticker)

    try:
        close = df["Close"]
        volume = df["Volume"]

        # EMAs
        _ema = ta.ema if _TA_AVAILABLE else _fallback_ema
        ema_fast_day = _ema(close, length=settings.SIGNAL_EMA_FAST_DAY)
        ema_slow_day = _ema(close, length=settings.SIGNAL_EMA_SLOW_DAY)
        ema_fast_swing = _ema(close, length=settings.SIGNAL_EMA_FAST_SWING)
        ema_slow_swing = _ema(close, length=settings.SIGNAL_EMA_SLOW_SWING)

        if ema_fast_day is not None and len(ema_fast_day) >= 2:
            snap.ema_fast_day = _safe_float(ema_fast_day.iloc[-1])
            snap.prev_ema_fast_day = _safe_float(ema_fast_day.iloc[-2])
        if ema_slow_day is not None and len(ema_slow_day) >= 2:
            snap.ema_slow_day = _safe_float(ema_slow_day.iloc[-1])
            snap.prev_ema_slow_day = _safe_float(ema_slow_day.iloc[-2])
        if ema_fast_swing is not None and len(ema_fast_swing) >= 2:
            snap.ema_fast_swing = _safe_float(ema_fast_swing.iloc[-1])
            snap.prev_ema_fast_swing = _safe_float(ema_fast_swing.iloc[-2])
        if ema_slow_swing is not None and len(ema_slow_swing) >= 2:
            snap.ema_slow_swing = _safe_float(ema_slow_swing.iloc[-1])
            snap.prev_ema_slow_swing = _safe_float(ema_slow_swing.iloc[-2])

        # RSI
        _rsi = ta.rsi if _TA_AVAILABLE else _fallback_rsi
        rsi = _rsi(close, length=settings.SIGNAL_RSI_PERIOD)
        if rsi is not None and len(rsi) >= 1:
            snap.rsi = _safe_float(rsi.iloc[-1])

        # MACD
        if _TA_AVAILABLE:
            macd_df = ta.macd(
                close,
                fast=settings.SIGNAL_MACD_FAST,
                slow=settings.SIGNAL_MACD_SLOW,
                signal=settings.SIGNAL_MACD_SIGNAL,
            )
        else:
            macd_df = _fallback_macd(
                close,
                fast=settings.SIGNAL_MACD_FAST,
                slow=settings.SIGNAL_MACD_SLOW,
                signal=settings.SIGNAL_MACD_SIGNAL,
            )
        if macd_df is not None and len(macd_df) >= 2:
            # Dynamically find MACD columns by prefix to handle pandas_ta
            # naming variations (e.g., MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9)
            macd_col = next(
                (c for c in macd_df.columns if c.startswith("MACD_")), None
            )
            signal_col = next(
                (c for c in macd_df.columns if c.startswith("MACDs_")), None
            )
            hist_col = next(
                (c for c in macd_df.columns if c.startswith("MACDh_")), None
            )

            if macd_col:
                snap.macd_line = _safe_float(macd_df[macd_col].iloc[-1])
                snap.prev_macd_line = _safe_float(macd_df[macd_col].iloc[-2])
            if signal_col:
                snap.macd_signal = _safe_float(macd_df[signal_col].iloc[-1])
                snap.prev_macd_signal = _safe_float(macd_df[signal_col].iloc[-2])
            if hist_col:
                snap.macd_histogram = _safe_float(macd_df[hist_col].iloc[-1])

        # Volume
        _sma = ta.sma if _TA_AVAILABLE else _fallback_sma
        vol_avg = _sma(volume.astype(float), length=settings.SIGNAL_VOLUME_AVG_PERIOD)
        if vol_avg is not None and len(vol_avg) >= 1:
            snap.avg_volume = _safe_float(vol_avg.iloc[-1])
        snap.current_volume = _safe_float(volume.iloc[-1])

        # Price
        snap.current_price = _safe_float(close.iloc[-1])
        if len(close) >= 2:
            snap.prev_close = _safe_float(close.iloc[-2])

    except Exception as e:
        logger.warning(f"{ticker}: Error computing indicators — {e}")
        return None

    return snap


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV price history from yfinance.

    Args:
        ticker: Stock ticker symbol.
        period: How far back to fetch (e.g., "1y", "6mo").
        interval: Bar size (e.g., "1d", "1h").

    Returns:
        DataFrame with OHLCV data, or None on failure.
    """
    try:
        # Use curl_cffi session for server compatibility (Yahoo blocks data center IPs)
        try:
            from curl_cffi.requests import Session as CffiSession
            _sess = CffiSession(impersonate="chrome")
        except ImportError:
            _sess = None
        stock = yf.Ticker(ticker, session=_sess)
        df = stock.history(period=period, interval=interval)
        if df is None or df.empty:
            logger.debug(f"{ticker}: No price history returned")
            return None
        return df
    except Exception as e:
        logger.debug(f"{ticker}: Failed to fetch price history — {e}")
        return None


# ── SIGNAL DETECTION ─────────────────────────────────────────

def detect_ema_crossover_day(snap: IndicatorSnapshot) -> Optional[dict]:
    """
    Detect 9/21 EMA crossover for day trades.

    Returns dict with signal info if crossover detected, None otherwise.
    """
    if (snap.ema_fast_day is None or snap.ema_slow_day is None or
            snap.prev_ema_fast_day is None or snap.prev_ema_slow_day is None):
        return None

    # Bullish crossover: fast crosses above slow
    if (snap.prev_ema_fast_day <= snap.prev_ema_slow_day and
            snap.ema_fast_day > snap.ema_slow_day):
        return {
            "name": "ema_crossover_day",
            "direction": "BUY",
            "ema_fast": round(snap.ema_fast_day, 4),
            "ema_slow": round(snap.ema_slow_day, 4),
        }

    # Bearish crossover: fast crosses below slow
    if (snap.prev_ema_fast_day >= snap.prev_ema_slow_day and
            snap.ema_fast_day < snap.ema_slow_day):
        return {
            "name": "ema_crossover_day",
            "direction": "SELL",
            "ema_fast": round(snap.ema_fast_day, 4),
            "ema_slow": round(snap.ema_slow_day, 4),
        }

    return None


def detect_ema_crossover_swing(snap: IndicatorSnapshot) -> Optional[dict]:
    """
    Detect 50/200 EMA crossover (golden/death cross) for swing trades.

    Returns dict with signal info if crossover detected, None otherwise.
    """
    if (snap.ema_fast_swing is None or snap.ema_slow_swing is None or
            snap.prev_ema_fast_swing is None or snap.prev_ema_slow_swing is None):
        return None

    # Golden cross: 50 crosses above 200
    if (snap.prev_ema_fast_swing <= snap.prev_ema_slow_swing and
            snap.ema_fast_swing > snap.ema_slow_swing):
        return {
            "name": "ema_crossover_swing",
            "direction": "BUY",
            "ema_fast": round(snap.ema_fast_swing, 4),
            "ema_slow": round(snap.ema_slow_swing, 4),
        }

    # Death cross: 50 crosses below 200
    if (snap.prev_ema_fast_swing >= snap.prev_ema_slow_swing and
            snap.ema_fast_swing < snap.ema_slow_swing):
        return {
            "name": "ema_crossover_swing",
            "direction": "SELL",
            "ema_fast": round(snap.ema_fast_swing, 4),
            "ema_slow": round(snap.ema_slow_swing, 4),
        }

    return None


def detect_rsi_signal(snap: IndicatorSnapshot) -> Optional[dict]:
    """
    Detect RSI oversold/overbought conditions.

    Returns dict with signal info if RSI is in extreme zone, None otherwise.
    """
    if snap.rsi is None:
        return None

    if snap.rsi <= settings.SIGNAL_RSI_OVERSOLD:
        return {
            "name": "rsi_oversold",
            "direction": "BUY",
            "rsi": round(snap.rsi, 2),
        }

    if snap.rsi >= settings.SIGNAL_RSI_OVERBOUGHT:
        return {
            "name": "rsi_overbought",
            "direction": "SELL",
            "rsi": round(snap.rsi, 2),
        }

    return None


def detect_macd_crossover(snap: IndicatorSnapshot) -> Optional[dict]:
    """
    Detect MACD line crossing the signal line.

    Returns dict with signal info if crossover detected, None otherwise.
    """
    if (snap.macd_line is None or snap.macd_signal is None or
            snap.prev_macd_line is None or snap.prev_macd_signal is None):
        return None

    # Bullish: MACD crosses above signal
    if (snap.prev_macd_line <= snap.prev_macd_signal and
            snap.macd_line > snap.macd_signal):
        return {
            "name": "macd_crossover",
            "direction": "BUY",
            "macd_line": round(snap.macd_line, 4),
            "macd_signal": round(snap.macd_signal, 4),
            "histogram": round(snap.macd_histogram, 4) if snap.macd_histogram else None,
        }

    # Bearish: MACD crosses below signal
    if (snap.prev_macd_line >= snap.prev_macd_signal and
            snap.macd_line < snap.macd_signal):
        return {
            "name": "macd_crossover",
            "direction": "SELL",
            "macd_line": round(snap.macd_line, 4),
            "macd_signal": round(snap.macd_signal, 4),
            "histogram": round(snap.macd_histogram, 4) if snap.macd_histogram else None,
        }

    return None


def detect_volume_spike(snap: IndicatorSnapshot) -> Optional[dict]:
    """
    Detect unusual volume (above threshold multiplier of average).

    Returns dict with volume info if spike detected, None otherwise.
    Volume spikes confirm other signals but don't have a direction on their own.
    """
    if snap.current_volume is None or snap.avg_volume is None:
        return None

    if snap.avg_volume <= 0:
        return None

    ratio = snap.current_volume / snap.avg_volume
    if ratio >= settings.SIGNAL_VOLUME_SPIKE_MULTIPLIER:
        return {
            "name": "volume_spike",
            "direction": "NEUTRAL",
            "volume_ratio": round(ratio, 2),
            "current_volume": snap.current_volume,
            "avg_volume": round(snap.avg_volume, 0),
        }

    return None


# ── SIGNAL AGGREGATION & CONFIDENCE ─────────────────────────

def classify_trade_type(indicators: list[dict]) -> str:
    """
    Classify whether a signal set suggests a day trade or swing trade.

    Swing trade if a swing EMA crossover is present; otherwise day trade.
    """
    for ind in indicators:
        if ind.get("name") == "ema_crossover_swing":
            return "SWING"
    return "DAY"


def compute_confidence(indicators: list[dict], has_volume_spike: bool) -> float:
    """
    Compute a confidence score (0-100) based on how many directional
    indicators agree and whether volume confirms.

    Scoring:
    - Each directional indicator contributing to the signal: base points
    - Volume spike confirmation: bonus points
    - More agreeing indicators = higher confidence

    Returns:
        Confidence score 0-100.
    """
    if not indicators:
        return 0.0

    # Count directional indicators (exclude volume which is NEUTRAL)
    directional = [i for i in indicators if i.get("direction") not in ("NEUTRAL", None)]
    count = len(directional)

    if count == 0:
        return 0.0

    # Base confidence from indicator count
    if count >= 3:
        base = settings.SIGNAL_CONFIDENCE_HIGH  # 80
    elif count >= 2:
        base = settings.SIGNAL_CONFIDENCE_MODERATE  # 60
    else:
        base = settings.SIGNAL_CONFIDENCE_LOW  # 40

    # Volume confirmation bonus: +10 if volume spike present
    volume_bonus = 10.0 if has_volume_spike else 0.0

    # Indicator-specific bonuses for strong readings
    bonus = 0.0
    for ind in directional:
        if ind.get("name") == "rsi_oversold" and ind.get("rsi", 50) <= 20:
            bonus += 5.0  # Deeply oversold
        elif ind.get("name") == "rsi_overbought" and ind.get("rsi", 50) >= 80:
            bonus += 5.0  # Deeply overbought

    return min(100.0, base + volume_bonus + bonus)


def determine_direction(indicators: list[dict]) -> Optional[str]:
    """
    Determine the consensus direction from a list of directional indicators.

    Returns "BUY", "SELL", or None if no consensus.
    """
    directional = [i for i in indicators if i.get("direction") in ("BUY", "SELL")]
    if not directional:
        return None

    buy_count = sum(1 for i in directional if i["direction"] == "BUY")
    sell_count = sum(1 for i in directional if i["direction"] == "SELL")

    if buy_count > sell_count:
        return "BUY"
    elif sell_count > buy_count:
        return "SELL"
    else:
        return None  # Conflicting signals — no trade


def _build_signal(
    snap: IndicatorSnapshot,
    direction: str,
    dir_indicators: list[dict],
    volume_spike: Optional[dict],
) -> TradeSignal:
    """Build a TradeSignal from grouped directional indicators.

    Args:
        snap: The source indicator snapshot.
        direction: "BUY" or "SELL".
        dir_indicators: Indicators matching this direction.
        volume_spike: Volume spike indicator, if detected.

    Returns:
        A TradeSignal with computed confidence and reason.
    """
    has_volume = volume_spike is not None
    inds_with_volume = dir_indicators + ([volume_spike] if volume_spike else [])
    confidence = compute_confidence(inds_with_volume, has_volume)
    trade_type = classify_trade_type(dir_indicators)

    indicator_names = [i["name"] for i in dir_indicators]
    reason = f"{direction} signal: {', '.join(indicator_names)}"
    if has_volume:
        reason += f" (volume {volume_spike['volume_ratio']}x avg)"

    return TradeSignal(
        ticker=snap.ticker,
        signal_type=direction,
        trade_type=trade_type,
        confidence=round(confidence, 1),
        indicators={i["name"]: i for i in inds_with_volume},
        price=snap.current_price or 0.0,
        reason=reason,
    )


def generate_signals(snap: IndicatorSnapshot) -> list[TradeSignal]:
    """
    Analyze an indicator snapshot and generate trade signals.

    Runs all detectors, groups by direction, computes confidence,
    and returns signals that meet minimum confidence threshold.

    Returns:
        List of TradeSignal objects (usually 0 or 1, rarely 2).
    """
    if snap is None:
        return []

    # Run all detectors with error isolation
    all_indicators = []
    for detector in [
        detect_ema_crossover_day,
        detect_ema_crossover_swing,
        detect_rsi_signal,
        detect_macd_crossover,
        detect_volume_spike,
    ]:
        try:
            result = detector(snap)
            if result is not None:
                all_indicators.append(result)
        except Exception as e:
            logger.warning(
                f"{snap.ticker}: Detector {detector.__name__} failed — {e}"
            )

    if not all_indicators:
        return []

    # Separate volume from directional
    volume_spike = None
    directional = []
    for ind in all_indicators:
        if ind.get("direction") == "NEUTRAL":
            volume_spike = ind
        else:
            directional.append(ind)

    if not directional:
        return []

    # Group by direction
    buy_indicators = [i for i in directional if i["direction"] == "BUY"]
    sell_indicators = [i for i in directional if i["direction"] == "SELL"]

    signals = []

    if buy_indicators:
        signals.append(_build_signal(snap, "BUY", buy_indicators, volume_spike))

    if sell_indicators:
        signals.append(_build_signal(snap, "SELL", sell_indicators, volume_spike))

    return signals


# ── PIPELINE ─────────────────────────────────────────────────

def scan_ticker(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    df: Optional[pd.DataFrame] = None,
) -> list[TradeSignal]:
    """
    Full signal scan for a single ticker.

    1. Fetch price history (or use provided DataFrame)
    2. Compute indicators
    3. Detect and score signals

    Args:
        ticker: Stock ticker symbol.
        period: yfinance period for price history.
        interval: yfinance interval for price bars.
        df: Pre-fetched DataFrame (skips yfinance call if provided).

    Returns:
        List of TradeSignal objects.
    """
    if df is None:
        df = fetch_price_history(ticker, period=period, interval=interval)

    snap = compute_indicators(df, ticker=ticker)
    if snap is None:
        return []

    signals = generate_signals(snap)

    for sig in signals:
        logger.info(
            f"{sig.ticker} | {sig.signal_type} {sig.trade_type} | "
            f"Confidence: {sig.confidence} | {sig.reason}"
        )

    return signals


def scan_watchlist(
    tickers: list[str],
    save_to_db: bool = True,
    min_confidence: Optional[float] = None,
) -> list[TradeSignal]:
    """
    Scan all watchlist tickers for technical signals.

    Args:
        tickers: List of ticker symbols to scan.
        save_to_db: Whether to persist signals to database.
        min_confidence: Override minimum confidence (defaults to settings).

    Returns:
        List of TradeSignal objects that meet minimum confidence.
    """
    if min_confidence is None:
        min_confidence = settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE

    logger.info("=" * 60)
    logger.info("EDGEFINDER SIGNAL SCAN STARTING")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Scanning {len(tickers)} tickers")
    logger.info("=" * 60)

    all_signals: list[TradeSignal] = []

    for ticker in tickers:
        try:
            signals = scan_ticker(ticker)
            all_signals.extend(signals)
        except Exception as e:
            logger.warning(f"{ticker}: Signal scan failed — {e}")

    # Filter by minimum confidence
    tradeable = [s for s in all_signals if s.confidence >= min_confidence]
    below_threshold = [s for s in all_signals if s.confidence < min_confidence]

    logger.info(f"Total signals detected: {len(all_signals)}")
    logger.info(f"Tradeable (confidence >= {min_confidence}): {len(tradeable)}")
    logger.info(f"Below threshold: {len(below_threshold)}")

    # Save to database
    if save_to_db:
        _save_signals(tradeable, was_traded=False)  # Marked False until trader acts
        _save_signals(
            below_threshold,
            was_traded=False,
            reason_skipped="Below confidence threshold",
        )

    # Summary
    if tradeable:
        logger.info("-" * 60)
        logger.info("TRADEABLE SIGNALS:")
        for sig in sorted(tradeable, key=lambda s: s.confidence, reverse=True):
            logger.info(
                f"  {sig.ticker:<6} | {sig.signal_type:<4} {sig.trade_type:<5} | "
                f"Confidence: {sig.confidence:>5.1f} | "
                f"${sig.price:>8.2f} | {sig.reason}"
            )
        logger.info("-" * 60)

    logger.info("SIGNAL SCAN COMPLETE")
    return tradeable


# ── DATABASE PERSISTENCE ─────────────────────────────────────

def _save_signals(
    signals: list[TradeSignal],
    was_traded: bool = False,
    reason_skipped: Optional[str] = None,
) -> None:
    """Save signals to the database."""
    if not signals:
        return

    session = None
    try:
        session = get_session()
        for sig in signals:
            record = SignalRecord(
                ticker=sig.ticker,
                signal_type=sig.signal_type,
                trade_type=sig.trade_type,
                confidence=sig.confidence,
                indicators=sig.indicators,
                was_traded=was_traded,
                reason_skipped=reason_skipped,
                timestamp=sig.timestamp,
            )
            session.add(record)
        session.commit()
        logger.info(f"Saved {len(signals)} signals to database")
    except Exception as e:
        logger.error(f"Failed to save signals: {e}")
        if session:
            session.rollback()
    finally:
        if session:
            session.close()


# ── UTILITIES ────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float, handling None/NaN/Inf."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
