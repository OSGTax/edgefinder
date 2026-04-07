"""EdgeFinder v2 — Relative strength computation.

Computes a stock's performance relative to a benchmark (SPY or sector ETF).
RS > 1.0 means outperforming the benchmark over the lookback period.
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


def compute_relative_strength(
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    lookback_days: int | None = None,
) -> float | None:
    """Compute relative strength ratio: stock return / benchmark return.

    RS > 1.0 = stock outperforming benchmark over lookback period.
    RS < 1.0 = stock underperforming.

    Args:
        stock_bars: OHLCV DataFrame for the stock (needs 'close' column)
        benchmark_bars: OHLCV DataFrame for benchmark (SPY, sector ETF)
        lookback_days: Number of trading days to look back (default from settings)

    Returns:
        RS ratio or None if insufficient data.
    """
    lookback = lookback_days or getattr(settings, "rs_lookback_days", 20)

    if stock_bars is None or benchmark_bars is None:
        return None
    if len(stock_bars) < lookback or len(benchmark_bars) < lookback:
        return None

    try:
        stock_close = stock_bars["close"]
        bench_close = benchmark_bars["close"]

        stock_return = stock_close.iloc[-1] / stock_close.iloc[-lookback]
        bench_return = bench_close.iloc[-1] / bench_close.iloc[-lookback]

        if bench_return == 0:
            return None

        return round(stock_return / bench_return, 4)
    except (IndexError, KeyError, ZeroDivisionError):
        return None


# Sector-to-ETF mapping for sector relative strength
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Electronic Technology": "XLK",
    "Information Technology Services": "XLK",
    "Financial Services": "XLF",
    "Finance": "XLF",
    "Energy": "XLE",
    "Energy Minerals": "XLE",
    "Healthcare": "XLV",
    "Health Technology": "XLV",
    "Health Services": "XLV",
    "Industrials": "XLI",
    "Industrial Services": "XLI",
    "Producer Manufacturing": "XLI",
    "Consumer Defensive": "XLP",
    "Consumer Non-Durables": "XLP",
    "Consumer Cyclical": "XLY",
    "Consumer Durables": "XLY",
    "Retail Trade": "XLY",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Communications": "XLC",
    "Basic Materials": "XLB",
    "Non-Energy Minerals": "XLB",
    "Process Industries": "XLB",
    "Distribution Services": "XLI",
    "Transportation": "XLI",
    "Commercial Services": "XLI",
}


def get_sector_etf(sector: str | None) -> str | None:
    """Map a Polygon/Massive sector name to a sector ETF symbol."""
    if not sector:
        return None
    return SECTOR_ETF_MAP.get(sector)
