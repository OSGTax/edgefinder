"""Cross-Sectional Momentum — hold the K strongest trends in the universe.

Effect class: cross-sectional momentum (Jegadeesh-Titman family) — the most
evidence-backed equity anomaly not yet tried in this lab. Names that lead the
universe keep leading over weeks-to-months horizons.

Score: distance above the 200dma (close/ema_200 − 1). With only 30 days of
rolling history available to strategies, this is the long-horizon momentum
proxy the engine exposes; it doubles as a trend gate (score > 0 required).

Mechanics (no look-ahead by construction):
- every evaluate() call RECORDS the ticker's score into a per-day buffer
  (the engine evaluates the whole watchlist daily, so the buffer holds the
  full cross-section);
- entry/exit decisions on day T rank against day T−1's COMPLETED buffer —
  one-day-stale ranks, never same-day;
- enter only when the name ranked in the top K yesterday (at most K names
  can qualify per day, so the engine's first-come slot filling introduces
  no selection bias);
- exit when the name decays out of the top ``exit_rank`` (= 3×K, derived,
  not searched) or on max_hold; the engine's fixed 20% stop backstops.

Knobs (2 searched + derived exit): top_k, max_hold_days. risk_pct fixed,
target fixed wide (0.50) so momentum is never clipped by a profit target.

Pre-registered kill criteria (same family as prior rounds): screen avg net
per trade < $5 or PF < 1.10 → dead.

Defaults below are PRE-REGISTERED (2026-06-05, round 3 ws2) before any
screen.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy

_MIN_UNIVERSE = 50   # need a real cross-section before ranks mean anything
_BUFFER_DAYS = 5     # keep only a few days of score history


@StrategyRegistry.register("xsec_mom")
class XsecMomStrategy(SwingStrategy):

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        # day -> {ticker: score}; populated by evaluate() as the engine
        # sweeps the watchlist each simulated/live day.
        self._scores: dict = {}

    @property
    def name(self) -> str:
        return "xsec_mom"

    @property
    def risk_pct(self) -> float:
        return 0.03  # fixed

    @property
    def target_pct(self) -> float:
        # Fixed wide: rank decay / max_hold are the real exits — a profit
        # target would clip exactly the tail momentum lives in.
        return 0.50

    @property
    def max_hold_days(self) -> int:
        return self._p("max_hold_days", 42)

    @property
    def top_k(self) -> int:
        return self._p("top_k", 5)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    # ── cross-sectional buffer ────────────────────────

    @staticmethod
    def _score(ind) -> float | None:
        if ind.ema_200 and ind.close:
            return ind.close / ind.ema_200 - 1
        return None

    def _record(self, ticker: str, data: MarketData) -> None:
        d = data.context.as_of
        if d is None:
            return
        score = self._score(data.current)
        if score is None:
            return
        day = self._scores.setdefault(d, {})
        day[ticker] = score
        if len(self._scores) > _BUFFER_DAYS:
            for old in sorted(self._scores)[:-_BUFFER_DAYS]:
                del self._scores[old]

    def _yesterday(self, today) -> dict | None:
        """The most recent COMPLETED cross-section before ``today``."""
        prior = [d for d in self._scores if d < today]
        if not prior:
            return None
        snap = self._scores[max(prior)]
        return snap if len(snap) >= _MIN_UNIVERSE else None

    @staticmethod
    def _rank(snap: dict, ticker: str) -> int | None:
        """1-based rank of ``ticker`` in ``snap`` (1 = strongest score)."""
        if ticker not in snap:
            return None
        s = snap[ticker]
        return 1 + sum(1 for v in snap.values() if v > s)

    # ── decisions ─────────────────────────────────────

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        today = data.context.as_of
        self._record(ticker, data)  # always record, even when not entering
        if today is None:
            return None
        snap = self._yesterday(today)
        if snap is None:
            return None
        rank = self._rank(snap, ticker)
        if rank is None or rank > self.top_k:
            return None
        if snap[ticker] <= 0:  # must actually be above its 200dma
            return None
        return self.make_intent(
            ticker, data,
            f"Momentum rank #{rank}/{len(snap)} "
            f"({snap[ticker] * 100:+.1f}% above 200dma)",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        today = data.context.as_of
        self._record(ticker, data)  # held names must stay in the buffer
        if today is None:
            return None
        snap = self._yesterday(today)
        if snap is None:
            return None
        rank = self._rank(snap, ticker)
        exit_rank = self.top_k * 3  # derived, not searched
        if rank is None or rank > exit_rank:
            shown = rank if rank is not None else "n/a"
            return self.make_exit(
                ticker, data,
                f"Momentum decayed: rank {shown} fell out of top {exit_rank}",
            )
        return None
