"""edgefinder.data.indicator_engine — RSI/ADX bounded-window regression.

[C-183]: _rsi/_adx apply Wilder EWM smoothing over the WHOLE input series
with no burn-in reset. Fed the full ~560-day live-cycle window, a single
large one-day gap (avg_loss decaying toward zero faster than avg_gain when
nothing comparably volatile follows it, so the RS ratio blows up) kept
"today's" RSI pinned near 100 for weeks on an otherwise-flat name (APGE:
a 06-22 +47% gap still read RSI 78-90 on 08-10/11, seven weeks later).
compute_indicators_from_bars now bounds the RSI/ADX/Stoch-RSI inputs to a
trailing RSI_ADX_WARMUP_BARS window so any one shock ages out completely
after a known, bounded number of sessions.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _flat_with_one_gap(n_before: int, gap_pct: float, n_after: int,
                        base: float = 100.0) -> pd.DataFrame:
    """A choppy-but-directionless series, one huge one-day gap, then
    choppy-but-directionless again — the APGE shape: price does nothing
    before or after (small alternating up/down noise, no net drift), only
    the gap moves it. Genuinely zero movement pins RSI's own gain=0 special
    case (rsi=0 by construction, unrelated to the bug this guards); real
    tickers always have SOME tick noise, so the fixture needs it too."""
    import numpy as np

    dates = pd.bdate_range("2024-01-02", periods=n_before + 1 + n_after)
    noise_before = [base * (1 + 0.003 * (1 if i % 2 == 0 else -1))
                    for i in range(n_before)]
    gapped = base * (1 + gap_pct)
    noise_after = [gapped * (1 + 0.003 * (1 if i % 2 == 0 else -1))
                   for i in range(n_after + 1)]
    closes = (noise_before + noise_after)[: len(dates)]
    opens = closes
    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "volume": np.full(len(dates), 1_000_000.0),
    })
    return df


def test_rsi_recovers_to_neutral_long_after_a_single_gap():
    from edgefinder.data.indicator_engine import (
        compute_indicators_from_bars, RSI_ADX_WARMUP_BARS,
    )

    # A gap old enough to have fully aged out of the bounded window, on an
    # otherwise dead-flat name — nothing in the last RSI_ADX_WARMUP_BARS
    # sessions should look extended.
    n_after = RSI_ADX_WARMUP_BARS + 20
    df = _flat_with_one_gap(n_before=40, gap_pct=0.47, n_after=n_after)
    snap = compute_indicators_from_bars(df[["open", "high", "low", "close", "volume"]])
    assert snap is not None
    assert 30 <= snap.rsi <= 70, f"RSI still extended {n_after} sessions after the gap: {snap.rsi}"
    assert snap.adx < 30, f"ADX still reads trending {n_after} sessions after a flat stretch: {snap.adx}"


def test_rsi_stays_extended_soon_after_the_gap():
    # Sanity check the fixture itself: right after the gap, RSI SHOULD be
    # pinned extended — the bounded window doesn't erase real, recent moves.
    from edgefinder.data.indicator_engine import compute_indicators_from_bars

    df = _flat_with_one_gap(n_before=40, gap_pct=0.47, n_after=5)
    snap = compute_indicators_from_bars(df[["open", "high", "low", "close", "volume"]])
    assert snap is not None
    assert snap.rsi > 70, f"expected a fresh gap to still read extended: {snap.rsi}"


def test_normal_trend_rsi_unaffected_by_bounding():
    # A clean, steady uptrend well inside the bounded window should give a
    # normal (not artificially flattened) RSI — the bound shouldn't distort
    # ordinary readings, only cap how long an old shock can linger.
    import numpy as np
    from edgefinder.data.indicator_engine import compute_indicators_from_bars

    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    closes = [100.0 * (1.01 ** i) for i in range(n)]  # steady +1%/day
    df = pd.DataFrame({
        "date": dates, "open": closes,
        "high": [c * 1.001 for c in closes], "low": [c * 0.999 for c in closes],
        "close": closes, "volume": np.full(n, 1_000_000.0),
    })
    snap = compute_indicators_from_bars(df[["open", "high", "low", "close", "volume"]])
    assert snap is not None
    assert snap.rsi > 70  # a real, sustained uptrend should still read hot
    assert snap.adx > 20  # and trending


def test_insufficient_bars_returns_none():
    import numpy as np
    from edgefinder.data.indicator_engine import compute_indicators_from_bars, MIN_BARS

    dates = pd.bdate_range("2024-01-02", periods=MIN_BARS - 1)
    df = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": np.full(MIN_BARS - 1, 1_000_000.0),
    })
    assert compute_indicators_from_bars(df[["open", "high", "low", "close", "volume"]]) is None
