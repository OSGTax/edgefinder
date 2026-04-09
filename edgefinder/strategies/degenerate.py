"""Degenerate Strategy — Calculated high-risk momentum plays.

Targets volatile stocks with high short interest, extreme volume, or
strong momentum. Takes bigger positions with wider stops for larger
potential payoffs. Accepts ALL signal types including low-confidence ones.

Risk profile:
- 10% of equity per trade (vs 2% default)
- 50% max concentration (vs 20% default)
- No confidence floor
- Accepts all signal patterns (bullish AND bearish for shorting)
- Targets squeeze candidates and momentum breakouts

NOTE: This strategy will have larger drawdowns and more volatile returns.
It's designed for learning what high-risk looks like in practice.
"""

from __future__ import annotations

import pandas as pd

from edgefinder.core.models import Signal, TickerFundamentals
from edgefinder.signals.engine import compute_indicators, detect_signals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification

try:
    from edgefinder.scanner.scoring import ScoringFactor, ScoringProfile
except ImportError:
    ScoringProfile = None
    ScoringFactor = None


@StrategyRegistry.register("degenerate")
class DegenerateStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "degenerate"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def risk_config(self) -> dict:
        """Override default risk limits for aggressive trading."""
        return {
            "max_risk_pct": 0.20,          # 20% equity per trade (10x normal)
            "max_concentration_pct": 1.00,  # 100% — can go all-in on one position
        }

    @property
    def preferred_signals(self) -> list[str]:
        """Accept ALL bullish signals — cast the widest net."""
        return [
            "ema_crossover_bullish",
            "macd_bullish_cross",
            "rsi_oversold",
            "bb_lower_touch",
            "volume_spike_bullish",
        ]

    @property
    def exit_signals(self) -> list[str]:
        return [
            "ema_crossover_bearish",
            "rsi_overbought",
            "macd_bearish_cross",
            "volume_spike_bearish",
        ]

    @property
    def scoring_profile(self):
        """Favors volatility, short squeeze potential, and momentum."""
        if ScoringProfile is None:
            return None
        return ScoringProfile(
            factors=[
                ScoringFactor("short_interest", 0.30, "high"),   # squeeze candidates
                ScoringFactor("earnings_growth", 0.20, "high"),  # momentum
                ScoringFactor("revenue_growth", 0.20, "high"),   # growth
                ScoringFactor("ev_to_ebitda", 0.15, "low"),      # cheap relative to earnings
                ScoringFactor("fcf_yield", 0.15, "high"),        # cash generation
            ],
            top_n=50,
        )

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        """Cast a wide net — qualify anything with some sign of life.

        Unlike other strategies which filter tightly, degenerate only
        rejects stocks with zero data. If there's any fundamental data
        at all, it qualifies. The scoring profile handles ranking.
        """
        # Must have at least SOME data to avoid penny stocks with no info
        if fundamentals.market_cap is not None and fundamentals.market_cap < 100_000_000:
            return False  # Skip micro-caps under $100M (too illiquid)

        # Qualify if we have ANY of these: growth, cash flow, or short interest
        has_growth = (
            (fundamentals.earnings_growth is not None and fundamentals.earnings_growth != 0)
            or (fundamentals.revenue_growth is not None and fundamentals.revenue_growth != 0)
        )
        has_value = fundamentals.fcf_yield is not None and fundamentals.fcf_yield > 0
        has_squeeze = fundamentals.short_interest is not None and fundamentals.short_interest > 0.05

        return has_growth or has_value or has_squeeze

    def generate_signals(self, ticker: str, bars: pd.DataFrame) -> list[Signal]:
        """Generate signals with NO confidence filter — take everything."""
        indicators = compute_indicators(bars)
        if indicators is None:
            return []
        all_signals = detect_signals(indicators, ticker)
        result = []
        for sig in all_signals:
            pattern = sig.metadata.get("pattern", "")
            if pattern in self.preferred_signals and sig.action.value == "BUY":
                sig.strategy_name = self.name
                result.append(sig)
        return result

    def on_trade_executed(self, notification: TradeNotification) -> None:
        pass
