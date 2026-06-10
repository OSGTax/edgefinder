"""Research strategies on the portfolio interface.

Ports of the pre-registered old-engine strategies, re-expressed as pure
``rebalance(ctx) -> weights`` functions so the clean engine can re-test them
honestly — the Phase-5 experiment: which old "failures" were the strategy,
and which were the old engine's trading-sequence bugs?

Parameters are carried over VERBATIM from the original pre-registrations
(dual_momentum aa1d351, trend_timer b872a44, both 2026-06-09) — zero new
degrees of freedom, so those pre-registrations still stand.
"""

from __future__ import annotations

from edgefinder.engine.strategy import BuyAndHold, EqualWeight, RebalanceContext

# The pre-registered tradable set — low-correlation, liquid, full-history ETFs.
ASSETS = ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "EFA")


def make_strategy_factory(spec: str):
    """Strategy spec -> a fresh-instance factory (shared by the validation
    CLI, the promotion CLI, and the live portfolio runner).

    Specs: ``equal_weight`` | ``buy_and_hold:SYM`` | ``trend_timer:SYM`` |
    ``dual_momentum`` (pre-registered 7-ETF menu, top_k=3).
    """
    if spec == "equal_weight":
        return EqualWeight
    if spec == "dual_momentum":
        return DualMomentum
    if spec.startswith("buy_and_hold:"):
        sym = spec.split(":", 1)[1].upper()
        return lambda: BuyAndHold(sym)
    if spec.startswith("trend_timer:"):
        sym = spec.split(":", 1)[1].upper()
        return lambda: TrendTimer(sym)
    raise ValueError(
        f"unknown strategy spec {spec!r} (use equal_weight, dual_momentum, "
        "buy_and_hold:SYM, or trend_timer:SYM)")


class TrendTimer:
    """Faber trend timing: hold the index above its 200-EMA, else cash.

    Knobs: NONE (pure pre-registered trend rule).
    """

    def __init__(self, symbol: str = "SPY") -> None:
        self.symbol = symbol

    @property
    def name(self) -> str:
        return f"trend_timer_{self.symbol.lower()}_v2"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        a = ctx.get(self.symbol)
        if a and a.indicators.ema_200 and a.price > a.indicators.ema_200:
            return {self.symbol: 1.0}
        return {}


class DualMomentum:
    """Antonacci/Faber dual momentum over low-correlation asset classes.

    ABSOLUTE filter: an asset is eligible only above its own 200-EMA (else its
    slot sits in cash — the crash protection). RELATIVE filter: hold the top
    ``top_k`` eligible assets by momentum (close / 200-EMA - 1), each at a
    fixed 1/top_k weight so unfilled slots stay in cash.
    """

    def __init__(self, symbols: tuple[str, ...] = ASSETS, top_k: int = 3) -> None:
        self.symbols = symbols
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "dual_momentum_v2"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored: list[tuple[float, str]] = []
        for s in self.symbols:
            a = ctx.get(s)
            if a and a.indicators.ema_200 and a.price > a.indicators.ema_200:
                scored.append((a.price / a.indicators.ema_200 - 1.0, s))
        scored.sort(reverse=True)
        return {s: 1.0 / self.top_k for _, s in scored[: self.top_k]}
