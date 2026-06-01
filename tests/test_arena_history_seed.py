"""Arena seeds per-ticker IndicatorHistory from the daily_bars loader.

Regression guard for the gambler stall: its entry/exit need
``history.previous`` (yesterday's MACD histogram), but the in-memory daily
cycle that populated history was lost on every restart and rarely ran. The
arena now seeds history from the persisted daily_bars table via a loader, so
``history.previous`` is available immediately.
"""

from datetime import date, timedelta

import pandas as pd

from edgefinder.data.market_data import MarketContext
from edgefinder.trading.arena import ArenaEngine


def _series(prices: list[float], start: date = date(2024, 1, 1)) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(prices):
        rows.append({
            "date": start + timedelta(days=i),
            "open": p, "high": p * 1.01, "low": p * 0.99,
            "close": p, "volume": 1_000_000.0,
        })
    return pd.DataFrame(rows)


class _DummyProvider:
    def get_bars(self, *a, **k):
        return None

    def get_latest_price(self, *a, **k):
        return None


def test_ensure_history_seeds_from_loader():
    bars = _series([100.0 + i * 0.5 for i in range(80)])
    arena = ArenaEngine(_DummyProvider(), bars_loader=lambda sym: bars)

    hist = arena._ensure_history("ABC")

    assert len(hist) >= 2
    assert hist.previous is not None
    assert hist.latest is not None
    # MACD histogram is what gambler reads off history.previous.
    assert hist.previous.macd_histogram is not None


def test_ensure_history_is_capped_and_cached():
    bars = _series([100.0 + i * 0.5 for i in range(120)])
    calls = {"n": 0}

    def loader(sym):
        calls["n"] += 1
        return bars

    arena = ArenaEngine(_DummyProvider(), bars_loader=loader)
    h1 = arena._ensure_history("ABC")
    h2 = arena._ensure_history("ABC")  # same ET day -> cached, no reload

    assert len(h1) == 30  # IndicatorHistory(max_days=30)
    assert h2 is h1
    assert calls["n"] == 1


def test_ensure_history_without_loader_is_empty():
    arena = ArenaEngine(_DummyProvider())  # no loader -> legacy behavior
    assert len(arena._ensure_history("ABC")) == 0


def test_ensure_history_handles_missing_bars_gracefully():
    arena = ArenaEngine(_DummyProvider(), bars_loader=lambda sym: None)
    assert len(arena._ensure_history("ABC")) == 0


def test_build_market_data_exposes_history_previous():
    bars = _series([100.0 + i * 0.3 for i in range(80)])
    arena = ArenaEngine(_DummyProvider(), bars_loader=lambda sym: bars)
    arena.load_strategies()
    arena.set_watchlists({"gambler": ["ABC"]})
    # Pre-seed the bar cache so current-bar indicators compute without a provider.
    arena._daily_bars_cache["ABC"] = bars

    mdm = arena._build_market_data(
        {"ABC": {"price": 124.0, "volume": 1_000_000.0}}, MarketContext()
    )

    assert "ABC" in mdm
    assert mdm["ABC"].history.previous is not None
