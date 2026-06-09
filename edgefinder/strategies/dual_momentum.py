"""Dual Momentum — rotate across asset classes, hold only what's trending.

Goal class: RISK-ADJUSTED outperformance of SPY (beat its Sharpe / cut its
drawdown), NOT raw-return alpha. This is the most robust risk-adjusted edge in
the literature (Antonacci, "Dual Momentum Investing"; Faber GTAA): hold the
strongest few of a set of LOW-CORRELATION assets (US large/tech/small/dow,
gold, long bonds, international), and only while each is in its own uptrend —
otherwise sit in cash. When stocks fall, bonds/gold usually rise, so rotating
to whatever is trending keeps return while sidestepping the worst drawdowns —
exactly what single-asset trend timing (trend_timer) could not do in a sharp
V-recovery.

Two momentum filters, both required:
- ABSOLUTE: an asset is eligible only while above its own 200-EMA (else its
  slot goes to cash — the crash protection),
- RELATIVE: among eligible assets, hold the top ``top_k`` by momentum
  (close / 200-EMA − 1), equal-weight; rotate as the ranking changes.

No look-ahead: each day records every watched asset's score into a per-day
buffer (the engine sweeps the whole watchlist daily); entry/exit rank against
YESTERDAY's COMPLETED buffer.

Sizing: ``risk_pct = 0.20 / top_k`` against the fixed 20% stop ⇒ each of the
top_k positions ≈ equity/top_k (equal-weight, ~fully invested when top_k
qualify, partial/cash otherwise). No target / trailing / max-hold — the
ranking and the 200-EMA filter are the only exits.

Knobs: top_k, lookback_ema (momentum/trend EMA). Pre-registered 2026-06-09
(risk-adjusted round 1) before any test.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy

# The default tradable set — low-correlation, liquid, full-history ETFs.
ASSETS = ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "EFA")
_BUFFER_DAYS = 5


@StrategyRegistry.register("dual_momentum")
class DualMomentumStrategy(SwingStrategy):

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self._scores: dict = {}  # day -> {ticker: momentum score}

    @property
    def name(self) -> str:
        return "dual_momentum"

    @property
    def top_k(self) -> int:
        return self._p("top_k", 3)

    @property
    def risk_pct(self) -> float:
        # ~equal-weight: each of top_k positions ≈ equity/top_k vs the 20% stop.
        return 0.20 / float(self.top_k)

    @property
    def max_concentration_pct(self) -> float:
        # Allow a touch above the equal weight so rounding never blocks a fill.
        return min(1.0, 1.4 / float(self.top_k))

    @property
    def target_pct(self) -> float:
        return 100.0  # no profit target — ride the trend

    @property
    def max_hold_days(self) -> int:
        return 0  # the ranking / 200-EMA filter is the only exit

    @property
    def trailing_stop_pct(self):
        return None

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    # ── per-day cross-asset momentum buffer ──

    def _score(self, ind) -> float | None:
        # Momentum proxy = distance above the long EMA (the data window we have).
        if ind.ema_200 and ind.close:
            return ind.close / ind.ema_200 - 1.0
        return None

    def _record(self, ticker: str, data: MarketData) -> None:
        d = data.context.as_of
        if d is None:
            return
        score = self._score(data.current)
        if score is None:
            return
        self._scores.setdefault(d, {})[ticker] = score
        if len(self._scores) > _BUFFER_DAYS:
            for old in sorted(self._scores)[:-_BUFFER_DAYS]:
                del self._scores[old]

    def _yesterday(self, today):
        prior = [d for d in self._scores if d < today]
        return self._scores[max(prior)] if prior else None

    def _rank(self, snap: dict, ticker: str) -> int | None:
        if ticker not in snap:
            return None
        s = snap[ticker]
        return 1 + sum(1 for v in snap.values() if v > s)

    # ── decisions ──

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        today = data.context.as_of
        self._record(ticker, data)
        if today is None:
            return None
        snap = self._yesterday(today)
        if snap is None:
            return None
        rank = self._rank(snap, ticker)
        # RELATIVE: must be in the top_k yesterday. ABSOLUTE: must be trending
        # (score > 0 == above its own 200-EMA).
        if rank is None or rank > self.top_k or snap[ticker] <= 0:
            return None
        return self.make_intent(
            ticker, data,
            f"Dual-momentum: {ticker} rank #{rank}/{len(snap)} "
            f"(+{snap[ticker] * 100:.1f}% vs 200-EMA), trending — hold",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        today = data.context.as_of
        self._record(ticker, data)
        if today is None:
            return None
        snap = self._yesterday(today)
        if snap is None:
            return None
        rank = self._rank(snap, ticker)
        # Exit if it left the top_k OR lost its uptrend (absolute momentum off).
        if rank is None or rank > self.top_k or snap.get(ticker, 0) <= 0:
            shown = rank if rank is not None else "n/a"
            return self.make_exit(
                ticker, data,
                f"Dual-momentum: {ticker} rank {shown} / trend lost — rotate out",
            )
        return None
