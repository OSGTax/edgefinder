"""EdgeFinder v2 — Relative Rotation Graph (RRG) sector analysis.

Bloomberg-style RRG implementation using JdK RS-Ratio and RS-Momentum.
Classifies sectors into 4 quadrants based on relative performance vs SPY:

  IMPROVING (top-left)  │  LEADING (top-right)
  RS-Ratio < 100        │  RS-Ratio > 100
  RS-Momentum > 100     │  RS-Momentum > 100
  ──────────────────────┼──────────────────────
  LAGGING (bottom-left) │  WEAKENING (bottom-right)
  RS-Ratio < 100        │  RS-Ratio > 100
  RS-Momentum < 100     │  RS-Momentum < 100

Sectors rotate clockwise: Improving → Leading → Weakening → Lagging → Improving

Trading applications:
- Momentum strategies: favor stocks in LEADING/IMPROVING sectors
- Value strategies: look for opportunities in LAGGING sectors (contrarian)
- Risk management: reduce exposure to WEAKENING sectors

Reference: Julius de Kempenaer's RRG methodology (Bloomberg RRG command since 2011)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from edgefinder.core.interfaces import DataProvider

logger = logging.getLogger(__name__)

# Standard sector ETF universe
SECTOR_ETFS = [
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLE", "Energy"),
    ("XLV", "Healthcare"),
    ("XLI", "Industrials"),
    ("XLP", "Consumer Staples"),
    ("XLY", "Consumer Discretionary"),
    ("XLU", "Utilities"),
    ("XLRE", "Real Estate"),
    ("XLC", "Communication Services"),
    ("XLB", "Materials"),
]


@dataclass
class SectorRRG:
    """RRG data point for a single sector."""

    etf: str
    sector_name: str
    rs_ratio: float = 100.0      # >100 = outperforming, <100 = underperforming
    rs_momentum: float = 100.0   # >100 = improving, <100 = deteriorating
    quadrant: str = "unknown"    # leading, weakening, lagging, improving
    return_5d: float | None = None
    return_20d: float | None = None
    return_60d: float | None = None
    # Previous values for trail visualization
    prev_rs_ratio: float | None = None
    prev_rs_momentum: float | None = None

    def to_dict(self) -> dict:
        return {
            "etf": self.etf,
            "sector_name": self.sector_name,
            "rs_ratio": round(self.rs_ratio, 2),
            "rs_momentum": round(self.rs_momentum, 2),
            "quadrant": self.quadrant,
            "return_5d": self.return_5d,
            "return_20d": self.return_20d,
            "return_60d": self.return_60d,
            "prev_rs_ratio": round(self.prev_rs_ratio, 2) if self.prev_rs_ratio else None,
            "prev_rs_momentum": round(self.prev_rs_momentum, 2) if self.prev_rs_momentum else None,
        }


def _compute_jdk_rs_ratio(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    lookback: int = 10,
    smoothing: int = 10,
) -> pd.Series:
    """Compute JdK RS-Ratio: smoothed relative strength trend.

    1. Compute raw relative strength: stock / benchmark * 100
    2. Normalize to percentage of its own moving average
    3. Smooth with exponential moving average
    4. Center at 100 (>100 = outperforming, <100 = underperforming)
    """
    # Raw relative strength
    rs_raw = (stock_close / benchmark_close) * 100

    # Normalize: RS as % of its own SMA
    rs_sma = rs_raw.rolling(window=lookback).mean()
    rs_normalized = (rs_raw / rs_sma) * 100

    # Smooth
    rs_ratio = rs_normalized.ewm(span=smoothing, adjust=False).mean()

    return rs_ratio


def _compute_jdk_rs_momentum(
    rs_ratio: pd.Series,
    lookback: int = 10,
    smoothing: int = 10,
) -> pd.Series:
    """Compute JdK RS-Momentum: rate of change of RS-Ratio.

    1. Compute ROC of RS-Ratio over lookback period
    2. Normalize and center at 100
    3. Smooth with EMA

    RS-Momentum > 100 = RS-Ratio is accelerating (improving)
    RS-Momentum < 100 = RS-Ratio is decelerating (weakening)
    """
    # Rate of change of RS-Ratio
    roc = (rs_ratio / rs_ratio.shift(lookback)) * 100

    # Smooth
    rs_momentum = roc.ewm(span=smoothing, adjust=False).mean()

    return rs_momentum


def _classify_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """Classify into RRG quadrant based on RS-Ratio and RS-Momentum."""
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "leading"
    elif rs_ratio >= 100 and rs_momentum < 100:
        return "weakening"
    elif rs_ratio < 100 and rs_momentum < 100:
        return "lagging"
    else:  # rs_ratio < 100 and rs_momentum >= 100
        return "improving"


class SectorRotationService:
    """Computes Bloomberg-style RRG sector rotation analysis.

    Uses JdK RS-Ratio and RS-Momentum methodology to classify
    each sector into one of four quadrants relative to SPY.
    """

    def __init__(self, provider: DataProvider) -> None:
        self._provider = provider

    def compute_rotation(self) -> list[SectorRRG]:
        """Compute RRG data for all sector ETFs.

        Fetches daily bars for SPY (benchmark) and each sector ETF,
        computes RS-Ratio and RS-Momentum, classifies quadrants.
        """
        end = date.today()
        start = end - timedelta(days=120)  # need ~60 trading days of data

        # Fetch benchmark (SPY) bars
        spy_bars = self._provider.get_bars("SPY", "day", start, end)
        if spy_bars is None or len(spy_bars) < 40:
            logger.warning("Insufficient SPY data for sector rotation")
            return []

        spy_close = spy_bars["close"]

        results: list[SectorRRG] = []
        for etf, name in SECTOR_ETFS:
            bars = self._provider.get_bars(etf, "day", start, end)
            if bars is None or len(bars) < 40:
                results.append(SectorRRG(etf=etf, sector_name=name))
                continue

            sector_close = bars["close"]

            # Align indices (in case of different trading days)
            aligned = pd.DataFrame({
                "sector": sector_close,
                "spy": spy_close,
            }).dropna()

            if len(aligned) < 30:
                results.append(SectorRRG(etf=etf, sector_name=name))
                continue

            # Compute JdK RS-Ratio and RS-Momentum
            rs_ratio = _compute_jdk_rs_ratio(aligned["sector"], aligned["spy"])
            rs_momentum = _compute_jdk_rs_momentum(rs_ratio)

            # Get current and previous values
            current_ratio = float(rs_ratio.iloc[-1]) if not pd.isna(rs_ratio.iloc[-1]) else 100.0
            current_momentum = float(rs_momentum.iloc[-1]) if not pd.isna(rs_momentum.iloc[-1]) else 100.0
            prev_ratio = float(rs_ratio.iloc[-2]) if len(rs_ratio) >= 2 and not pd.isna(rs_ratio.iloc[-2]) else None
            prev_momentum = float(rs_momentum.iloc[-2]) if len(rs_momentum) >= 2 and not pd.isna(rs_momentum.iloc[-2]) else None

            # Simple returns for display
            close = aligned["sector"]
            return_5d = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 2) if len(close) >= 5 else None
            return_20d = round((close.iloc[-1] / close.iloc[-20] - 1) * 100, 2) if len(close) >= 20 else None
            return_60d = round((close.iloc[-1] / close.iloc[-60] - 1) * 100, 2) if len(close) >= 60 else None

            rrg = SectorRRG(
                etf=etf,
                sector_name=name,
                rs_ratio=current_ratio,
                rs_momentum=current_momentum,
                quadrant=_classify_quadrant(current_ratio, current_momentum),
                return_5d=return_5d,
                return_20d=return_20d,
                return_60d=return_60d,
                prev_rs_ratio=prev_ratio,
                prev_rs_momentum=prev_momentum,
            )
            results.append(rrg)

        # Sort by quadrant priority: leading > improving > weakening > lagging
        quadrant_order = {"leading": 0, "improving": 1, "weakening": 2, "lagging": 3, "unknown": 4}
        results.sort(key=lambda x: (quadrant_order.get(x.quadrant, 4), -x.rs_ratio))

        logger.info(
            "Sector rotation: %s",
            ", ".join(f"{r.etf}={r.quadrant}" for r in results),
        )
        return results

    def get_sectors_by_quadrant(self) -> dict[str, list[str]]:
        """Group sector ETFs by quadrant.

        Returns: {"leading": ["XLK", "XLV"], "lagging": ["XLE"], ...}
        """
        rotation = self.compute_rotation()
        result: dict[str, list[str]] = {}
        for rrg in rotation:
            result.setdefault(rrg.quadrant, []).append(rrg.etf)
        return result
