"""EdgeFinder v2 — Unified StockProfile.

Merges all data sources for a single ticker into one queryable object:
- Fundamentals (from Massive: ratios, earnings, analyst, short interest, news)
- Technical indicators (from compute_indicators: RSI, MACD, ADX, etc.)
- Relative strength (vs SPY, vs sector ETF)

Used by: scoring engine, strategy qualification, signal generation.
Any field from any source can be accessed via profile.get(field_name).
"""

from __future__ import annotations

from dataclasses import dataclass

from edgefinder.core.models import TickerFundamentals

# Sentinel for distinguishing None (field exists, value is None) from missing
_SENTINEL = object()


@dataclass
class StockProfile:
    """Complete data profile for a single ticker — all sources unified."""

    symbol: str
    fundamentals: TickerFundamentals | None = None
    indicators: object | None = None  # IndicatorSnapshot (avoid circular import)
    rs_vs_spy: float | None = None
    rs_vs_sector: float | None = None
    sector_etf: str | None = None

    def get(self, field_name: str) -> float | None:
        """Get any numeric field by name — checks all data sources.

        Priority: direct attributes → fundamentals → indicators.
        Returns None if the field doesn't exist or isn't numeric.
        """
        # Check direct attributes first (rs_vs_spy, rs_vs_sector)
        val = getattr(self, field_name, _SENTINEL)
        if val is not _SENTINEL and val is not None and isinstance(val, (int, float)):
            return float(val)

        # Then fundamentals
        if self.fundamentals:
            val = getattr(self.fundamentals, field_name, None)
            if val is not None and isinstance(val, (int, float)):
                return float(val)

        # Then indicators
        if self.indicators:
            val = getattr(self.indicators, field_name, None)
            if val is not None and isinstance(val, (int, float)):
                return float(val)

        return None

    def get_str(self, field_name: str) -> str | None:
        """Get any string field by name."""
        if self.fundamentals:
            val = getattr(self.fundamentals, field_name, None)
            if val is not None and isinstance(val, str):
                return val
        return None

    def get_bool(self, field_name: str) -> bool | None:
        """Get any boolean field by name."""
        if self.fundamentals:
            val = getattr(self.fundamentals, field_name, None)
            if isinstance(val, bool):
                return val
        return None

    def to_dict(self) -> dict:
        """Flatten all data into a single dict for API responses."""
        result = {"symbol": self.symbol}

        if self.fundamentals:
            for key, val in self.fundamentals.__dict__.items():
                if key != "raw_data" and val is not None:
                    result[key] = val

        if self.indicators and hasattr(self.indicators, "to_dict"):
            for key, val in self.indicators.to_dict().items():
                result[f"ind_{key}"] = val

        if self.rs_vs_spy is not None:
            result["rs_vs_spy"] = self.rs_vs_spy
        if self.rs_vs_sector is not None:
            result["rs_vs_sector"] = self.rs_vs_sector
        if self.sector_etf is not None:
            result["sector_etf"] = self.sector_etf

        return result
