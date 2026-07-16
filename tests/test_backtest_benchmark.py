"""The SPY benchmark must match the strategy bars' return basis (TR-vs-TR).

THE regression test for the 2026-07-16 scoreboard fix: strategy bars load
dividend-adjusted (total-return) while the benchmark was price-only, handing
every strategy the benchmark's dividend yield as phantom "excess" (~+50pp
compounded over the 2006-2018 in-sample half). The anchor: a backtest that
effectively buys-and-holds SPY must show ~zero excess against its own
benchmark — with the old price-only benchmark the same run "beat SPY" by
exactly the dividends.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

START = date(2024, 1, 2)
TRADE_START = START + timedelta(days=300)   # 300 warmup days, 100 scored


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'bench.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def seed_spy_bars(store, *, n_days: int = 400) -> None:
    """A steadily rising synthetic SPY — momentum's trailing return stays
    positive at every rebalance, so momentum:1 over [SPY] holds SPY from the
    first scored bar on (effectively buy-and-hold)."""
    rows, px = [], 100.0
    for i in range(n_days):
        px *= 1.0005
        rows.append({"symbol": "SPY", "date": START + timedelta(days=i),
                     "open": round(px, 4), "high": round(px * 1.001, 4),
                     "low": round(px * 0.999, 4), "close": round(px, 4),
                     "volume": 1_000_000.0, "source": "test"})
    store.insert("daily_bars", rows, returning=False)


def seed_spy_dividends(store) -> None:
    # One ex-date before the scored window (its back-adjustment must not leak
    # into window excess) and three inside it (~0.9% of price each — the
    # phantom excess a price-only benchmark hands the strategy).
    for off in (100, 320, 350, 380):
        store.insert("dividends", {"symbol": "SPY",
                                   "ex_date": START + timedelta(days=off),
                                   "cash_amount": 1.0}, returning=False)


def test_spy_buy_and_hold_shows_zero_excess_tr_vs_tr(store):
    from agent import backtest_tool

    seed_spy_bars(store)
    seed_spy_dividends(store)
    out = backtest_tool.run(["SPY"], "momentum:1", schedule="monthly",
                            start=TRADE_START, costed=False)
    assert "error" not in out
    assert out["num_trades"] >= 1  # it actually entered SPY
    # Holding the benchmark itself is ~zero excess. Tolerance covers integer-
    # share flooring, the residual-cash drag, and the flat-bps entry fill.
    assert out["excess_return_pct"] == pytest.approx(0.0, abs=0.5)


def test_price_only_benchmark_would_have_shown_phantom_excess(store):
    """The OLD pairing (TR strategy bars vs price-only SPY) on the SAME data
    reports the dividends as excess — the bug the TR benchmark removes."""
    from agent import backtest_tool
    from agent.data import load_bars, spy_series_df

    seed_spy_bars(store)
    seed_spy_dividends(store)
    bars = load_bars(["SPY"], div_adjust=True)

    old = backtest_tool.run_prepared(bars, spy_series_df(), "momentum:1",
                                     schedule="monthly", start=TRADE_START,
                                     costed=False)
    fixed = backtest_tool.run_prepared(bars, spy_series_df(total_return=True),
                                       "momentum:1", schedule="monthly",
                                       start=TRADE_START, costed=False)
    # ~2.6% of in-window dividends read as "alpha" against the price-only SPY
    assert old["excess_return_pct"] > 1.5
    assert fixed["excess_return_pct"] == pytest.approx(0.0, abs=0.5)
    assert old["excess_return_pct"] - fixed["excess_return_pct"] > 1.5


def test_total_return_degrades_to_price_when_no_dividend_rows(store):
    """No SPY dividend rows: the adjustment is a no-op (TR ≈ PR), never a
    crash — the documented graceful-degradation contract."""
    from agent.data import spy_series_df

    seed_spy_bars(store)
    tr = spy_series_df(total_return=True)
    pr = spy_series_df()
    assert len(tr) == len(pr) == 400
    assert tr["close"].iloc[0] == pytest.approx(float(pr["close"].iloc[0]))
    assert tr["close"].iloc[-1] == pytest.approx(float(pr["close"].iloc[-1]))
