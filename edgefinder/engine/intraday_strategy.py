"""The intraday (minute-bar) strategy interface — the daily interface's sibling.

Same shape as engine/strategy.py: a strategy sees the universe AS OF a decision
bar (point-in-time — nothing past the decision index exists in the context) and
returns the portfolio it wants:

    decide(ctx) -> {symbol: target_weight}

Weights are fractions of equity in [0, 1]; their sum is the invested fraction
(rest cash). The intraday engine (engine/intraday_backtest.py) turns those into
trades at the NEXT bar's open with realistic costs, so a strategy can never have
a trading-sequence bug — that lives in one tested place, exactly like daily.

WHY backed by numpy arrays + a current index rather than DataFrame slices: an
intraday session is ~390 bars and the engine steps EVERY bar. Slicing the full
history per bar (``df.iloc[:i+1]``) copies O(i) rows each step, which is
O(n^2) per session-day — death by a thousand copies. Here every view is a cheap
numpy slice of a pre-built array, and scalars are O(1) index reads. NOTHING in
this module may read past ``i`` (the decision index): that would be look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np

# RTH session bounds in minutes-past-midnight ET (mirrors data/minutestore).
_RTH_OPEN_MIN = 9 * 60 + 30      # 09:30 -> 570
_RTH_CLOSE_MIN = 16 * 60         # 16:00 -> 960 (last RTH bar STARTS at 15:59)


@dataclass(frozen=True)
class IntradayAssetView:
    """Everything known about ONE asset at the decision bar (point-in-time).

    Backed by per-symbol numpy arrays for the WHOLE loaded window plus this
    session's start offset and the decision index ``i``. Every accessor reads
    only indices ``<= i`` (look-ahead-free by construction) and returns cheap
    numpy views — never a copy of the full history.
    """

    symbol: str
    _o: np.ndarray            # open  (whole loaded window for this symbol)
    _h: np.ndarray            # high
    _l: np.ndarray            # low
    _c: np.ndarray            # close
    _v: np.ndarray            # volume
    _ts: np.ndarray           # ts (UTC epoch seconds, int64)
    i: int                    # decision-bar index into the arrays
    session_start: int        # index of this session's FIRST bar (this symbol)

    # ── scalars ──
    @property
    def price(self) -> float:
        """Decision-bar close (the price a decision is made on)."""
        return float(self._c[self.i])

    @property
    def ts(self) -> int:
        return int(self._ts[self.i])

    @property
    def bar(self) -> dict:
        """The decision bar's OHLCV."""
        i = self.i
        return {"open": float(self._o[i]), "high": float(self._h[i]),
                "low": float(self._l[i]), "close": float(self._c[i]),
                "volume": float(self._v[i])}

    def ret(self, n: int) -> float | None:
        """Return over the last ``n`` bars (close[i] / close[i-n] - 1)."""
        j = self.i - n
        if n <= 0 or j < 0 or self._c[j] <= 0:
            return None
        return float(self._c[self.i] / self._c[j] - 1.0)

    # ── last-n views (cheap numpy slices, never copies of full history) ──
    def closes(self, n: int) -> np.ndarray:
        return self._c[max(0, self.i - n + 1): self.i + 1]

    def highs(self, n: int) -> np.ndarray:
        return self._h[max(0, self.i - n + 1): self.i + 1]

    def lows(self, n: int) -> np.ndarray:
        return self._l[max(0, self.i - n + 1): self.i + 1]

    def volumes(self, n: int) -> np.ndarray:
        return self._v[max(0, self.i - n + 1): self.i + 1]

    # ── session-so-far (this session through the decision bar) ──
    @property
    def session_open(self) -> float:
        return float(self._o[self.session_start])

    @property
    def session_high(self) -> float:
        return float(self._h[self.session_start: self.i + 1].max())

    @property
    def session_low(self) -> float:
        return float(self._l[self.session_start: self.i + 1].min())

    @property
    def session_vwap(self) -> float | None:
        """Volume-weighted avg price this session through the decision bar.

        Uses the bar typical price (h+l+c)/3 weighted by volume. None if the
        session has had no volume so far (degenerate)."""
        s, e = self.session_start, self.i + 1
        vol = self._v[s:e]
        tot = float(vol.sum())
        if tot <= 0:
            return None
        typ = (self._h[s:e] + self._l[s:e] + self._c[s:e]) / 3.0
        return float((typ * vol).sum() / tot)

    @property
    def prev_close(self) -> float | None:
        """Prior session's last close (the overnight-gap reference). None if
        this is the first loaded session for this symbol."""
        j = self.session_start - 1
        if j < 0:
            return None
        return float(self._c[j])

    def opening_range(self, m: int) -> tuple[float, float] | None:
        """(high, low) over the FIRST ``m`` bars of THIS session.

        Look-ahead-free by construction: it reads the slice
        ``[session_start : min(i+1, session_start+m)]`` — never past the
        decision index ``i``. Before the range has fully formed
        (``bars_since_open < m``) it returns the PARTIAL high/low over the
        bars seen so far. Returns None only if the session has no bar yet
        (degenerate; can't happen at a decision bar)."""
        s = self.session_start
        e = min(self.i + 1, s + m)
        if e <= s:
            return None
        hi = float(self._h[s:e].max())
        lo = float(self._l[s:e].min())
        return (hi, lo)


@dataclass(frozen=True)
class IntradayContext:
    """The universe at one decision bar, handed to a strategy each step."""

    ts: int                       # decision-bar ts (UTC epoch seconds)
    session_date: date            # ET session date of the decision bar
    minute_of_day: int            # ET minutes past midnight (from to_et)
    bars_since_open: int          # bars this session through the decision bar
    bars_until_close: int         # from the ET CLOCK (live-replicable, not look-ahead)
    assets: dict[str, IntradayAssetView]
    is_last_decision_bar: bool    # final decision bar before this session's close

    def symbols(self) -> list[str]:
        return list(self.assets)

    def get(self, symbol: str) -> IntradayAssetView | None:
        return self.assets.get(symbol)

    def price(self, symbol: str) -> float | None:
        a = self.assets.get(symbol)
        return a.price if a else None


@runtime_checkable
class IntradayStrategy(Protocol):
    """Anything with a name and a decide() returning intraday target weights."""

    @property
    def name(self) -> str: ...

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        """Return desired {symbol: weight}; weights in [0,1], sum <= 1 (rest
        cash). Symbols not in ctx.assets, or with non-positive weight, are
        ignored by the engine."""
        ...


# ── reference strategies (engine self-tests + a real-data smoke) ──


class IntradayFlat:
    """Always all-cash — the trivial null (its curve must equal start_cash
    every session)."""

    @property
    def name(self) -> str:
        return "intraday_flat"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        return {}


class BuyHoldFromOpen:
    """Buy one symbol at the first decision bar of each scored session, hold to
    close — the per-session correctness anchor (with flatten_at_close and no
    costs, each session's return must equal that symbol's open->close move)."""

    def __init__(self, symbol: str = "SPY") -> None:
        self.symbol = symbol

    @property
    def name(self) -> str:
        return f"buy_hold_open_{self.symbol.lower()}"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        return {self.symbol: 1.0} if self.symbol in ctx.assets else {}


class IntradayMeanReversion:
    """One simple REAL example: long when price is ``z`` stds below its trailing
    ``lookback``-bar mean, flat otherwise; flattened at the close by the engine.

    A churning strategy (for the toll-bleed test) and the CLI's real-data
    smoke. Knobs are FIXED at construction — never optimized in a backtest.
    """

    def __init__(self, symbol: str = "SPY", lookback: int = 20, z: float = 1.0) -> None:
        self.symbol = symbol
        self.lookback = int(lookback)
        self.z = float(z)

    @property
    def name(self) -> str:
        return f"intraday_mean_rev_{self.symbol.lower()}_{self.lookback}_{self.z:g}"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        a = ctx.get(self.symbol)
        if a is None or ctx.is_last_decision_bar:
            return {}                                   # let the engine flatten
        c = a.closes(self.lookback)
        if len(c) < self.lookback:
            return {}
        mean = float(c.mean())
        sd = float(c.std())
        if sd <= 0:
            return {}
        zscore = (a.price - mean) / sd
        return {self.symbol: 1.0} if zscore <= -self.z else {}
