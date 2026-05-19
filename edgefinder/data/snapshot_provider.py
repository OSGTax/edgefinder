"""Enriched snapshot provider — price + volume for all tickers in one call.

Uses Polygon's get_snapshot_all which returns today's OHLCV for every stock.
This replaces individual get_latest_price calls in the intraday cycle.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_enriched_snapshots(provider) -> dict[str, dict]:
    """Fetch price + volume for all tickers in one API call.

    Returns: {ticker: {"price": float, "volume": float, "open": float,
                        "high": float, "low": float}}
    """
    try:
        snapshots = provider._client.get_snapshot_all("stocks")
    except Exception:
        logger.exception("get_enriched_snapshots failed")
        return {}

    if not snapshots:
        return {}

    result: dict[str, dict] = {}
    for s in snapshots:
        ticker = getattr(s, "ticker", None)
        if not ticker:
            continue

        price = None
        volume = 0.0
        open_price = 0.0
        high = 0.0
        low = 0.0

        if s.day:
            price = float(s.day.close) if s.day.close else None
            volume = float(s.day.volume) if s.day.volume else 0.0
            open_price = float(s.day.open) if getattr(s.day, "open", None) else 0.0
            high = float(s.day.high) if getattr(s.day, "high", None) else 0.0
            low = float(s.day.low) if getattr(s.day, "low", None) else 0.0

        if price is None and s.prev_day and s.prev_day.close:
            price = float(s.prev_day.close)

        if price:
            result[ticker] = {
                "price": price,
                "volume": volume,
                "open": open_price,
                "high": high,
                "low": low,
            }

    logger.info("Enriched snapshots: %d tickers", len(result))
    return result
