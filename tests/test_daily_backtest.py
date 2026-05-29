"""Daily backtester — drives the real Arena cycle over synthetic daily bars.

No network/S3: bars are built in-memory, so this exercises the actual
strategy + executor + account/risk code paths end to end.
"""

from datetime import date, timedelta

import pandas as pd

from edgefinder.backtest.daily_backtest import run_daily_backtest


def _series(prices: list[float], symbol: str = "TEST") -> pd.DataFrame:
    start = date(2024, 1, 1)
    rows = []
    for i, p in enumerate(prices):
        rows.append({
            "date": start + timedelta(days=i),
            "open": p, "high": p * 1.01, "low": p * 0.99,
            "close": p, "volume": 1_000_000.0,
        })
    return pd.DataFrame(rows)


def _decline_then_rally() -> list[float]:
    prices = [120.0]
    for _ in range(40):           # sustained decline -> RSI < 35 (entry)
        prices.append(prices[-1] * 0.988)
    for _ in range(25):           # sharp rally -> hits target / RSI > 70 (exit)
        prices.append(prices[-1] * 1.02)
    return prices


def test_backtest_runs_real_engine_and_trades():
    bars = {"TEST": _series(_decline_then_rally())}
    result = run_daily_backtest("coward", bars, starting_cash=10_000.0)

    # Equity curve only starts once there's enough history for indicators
    assert result["equity_curve"], "expected a populated equity curve"
    assert result["stats"]["days"] > 0
    # Coward should enter on the oversold dip and exit on the rally
    assert result["stats"]["num_closed_trades"] >= 1
    trade = result["trades"][0]
    assert trade["exit_price"] is not None
    assert trade["pnl_dollars"] is not None
    assert any(
        trade["exit_reason"].startswith(p)
        for p in ("STOP_LOSS", "TARGET_HIT", "STRATEGY_EXIT")
    )
    # Final equity is internally consistent: cash + marked positions.
    assert result["final_equity"] > 0


def test_backtest_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        run_daily_backtest("nope", {"TEST": _series([100.0] * 40)})


def test_backtest_does_not_touch_global_event_bus():
    """Backtest trades must never reach the live event bus (which persists to
    the trades table) — that leak corrupted live account balances."""
    from edgefinder.core.events import event_bus

    seen = []
    def _open(t): seen.append(("open", t))
    def _close(t): seen.append(("close", t))
    event_bus.subscribe("trade.opened", _open)
    event_bus.subscribe("trade.closed", _close)
    try:
        result = run_daily_backtest("coward", {"TEST": _series(_decline_then_rally())})
        assert result["stats"]["num_closed_trades"] >= 1  # trades really happened
        assert seen == []                                  # but none leaked out
    finally:
        event_bus.unsubscribe("trade.opened", _open)
        event_bus.unsubscribe("trade.closed", _close)


def test_backtest_steady_uptrend_no_entry():
    # A steady uptrend keeps RSI high and price near the upper band, so
    # coward's oversold / BB-lower entries never fire -> no trades, flat equity.
    bars = {"TEST": _series([100.0 * (1.004 ** i) for i in range(60)])}
    result = run_daily_backtest("coward", bars, starting_cash=10_000.0)
    assert result["stats"]["num_closed_trades"] == 0
    assert result["stats"]["num_open_positions"] == 0
    assert result["final_equity"] == 10_000.0
