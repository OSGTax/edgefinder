"""Correctness tests for the new portfolio backtest engine.

The anchor test: buy-and-hold THROUGH the engine must equal the asset's own
buy-and-hold return (net of one entry cost). If that holds, the trading
sequence — point-in-time decision, next-open fill, mark-to-market — is right.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.strategy import BuyAndHold, EqualWeight, RebalanceContext


def _bars(closes, symbol="SPY", start=date(2024, 1, 1)):
    rows = []
    for i, c in enumerate(closes):
        rows.append({"date": start + timedelta(days=i), "open": float(c),
                     "high": float(c) * 1.001, "low": float(c) * 0.999,
                     "close": float(c), "volume": 1_000_000.0})
    return pd.DataFrame(rows)


def test_buy_and_hold_equals_asset_return_no_cost():
    # Enter at day-2 open (100), final close 121 → exactly +21%, zero cost.
    closes = [100, 100, 100, 110, 110, 110, 110, 110, 110, 121]
    res = run_backtest({"SPY": _bars(closes)}, BuyAndHold("SPY"),
                       start_cash=10_000.0, schedule="daily", cost_bps=0.0,
                       warmup_days=2)
    assert res.stats["return_pct"] == pytest.approx(21.0, abs=0.01)
    assert res.stats["open_positions"] == 1
    assert res.stats["num_trades"] == 1  # one entry, never sells


def test_cost_reduces_return():
    closes = [100, 100, 100, 110, 110, 110, 110, 110, 110, 121]
    free = run_backtest({"SPY": _bars(closes)}, BuyAndHold("SPY"),
                        schedule="daily", cost_bps=0.0, warmup_days=2)
    costed = run_backtest({"SPY": _bars(closes)}, BuyAndHold("SPY"),
                          schedule="daily", cost_bps=50.0, warmup_days=2)  # 50bps
    assert costed.stats["return_pct"] < free.stats["return_pct"]


def test_no_lookahead_fills_at_next_open_not_decision_close():
    # Decision is made on yesterday's data; the fill is TODAY's open. A gap-up
    # the morning of entry must be paid (entry > the decision-day close).
    closes = [100, 100, 100]
    df = _bars(closes)
    # day-2 opens at 130 (gap up) though its close is 100
    df.loc[2, ["open", "high"]] = [130.0, 131.0]
    res = run_backtest({"SPY": df}, BuyAndHold("SPY"), schedule="daily",
                       cost_bps=0.0, warmup_days=2)
    assert res.trades[0]["price"] == pytest.approx(130.0)  # next open, not 100


def test_equal_weight_splits_capital():
    a = _bars([100] * 10, "AAA")
    b = _bars([50] * 10, "BBB")
    res = run_backtest({"AAA": a, "BBB": b}, EqualWeight(), schedule="daily",
                       cost_bps=0.0, warmup_days=2)
    # ~50/50 of 10k: ~50 AAA @100, ~100 BBB @50
    sides = {t["symbol"]: t for t in res.trades if t["side"] == "BUY"}
    assert sides["AAA"]["shares"] == 50
    assert sides["BBB"]["shares"] == 100


def test_delisted_holding_is_closed_at_last_price():
    alive = _bars([100] * 12, "ALIVE")             # runs the whole calendar
    dead = _bars([100, 100, 100, 100, 100], "DEAD")  # data ends at day 5
    res = run_backtest({"ALIVE": alive, "DEAD": dead}, EqualWeight(),
                       schedule="daily", cost_bps=0.0, warmup_days=2)
    delist = [t for t in res.trades if t.get("reason") == "DELISTED"]
    assert len(delist) == 1 and delist[0]["symbol"] == "DEAD"
    assert res.stats["open_positions"] >= 1  # ALIVE still held at the end


def test_bad_print_does_not_distort():
    # A dropped-digit low (10 instead of 100) is sanitized in the precompute;
    # buy-and-hold return is unaffected.
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    df = _bars(closes)
    df.loc[6, "low"] = 10.0  # corrupt low
    res = run_backtest({"SPY": df}, BuyAndHold("SPY"), schedule="daily",
                       cost_bps=0.0, warmup_days=2)
    assert res.stats["return_pct"] == pytest.approx(0.0, abs=0.01)


def test_benchmark_excess_and_drawdown_metrics():
    closes = [100, 100, 100, 110, 105, 120, 118, 130, 125, 140]
    bench = _bars(closes)[["date", "close"]]
    res = run_backtest({"SPY": _bars(closes)}, BuyAndHold("SPY"),
                       schedule="daily", cost_bps=0.0, warmup_days=2,
                       benchmark=bench)
    s = res.stats
    assert "benchmark_return_pct" in s and "excess_return_pct" in s
    assert "drawdown_reduction_pct" in s


def test_strategy_protocol_shape():
    # A trivial inline strategy proves how general/easy the interface is.
    class FirstAlphabetical:
        name = "first_alphabetical"
        def rebalance(self, ctx: RebalanceContext) -> dict:
            syms = sorted(ctx.symbols())
            return {syms[0]: 1.0} if syms else {}

    a = _bars([100] * 8, "AAA")
    z = _bars([100] * 8, "ZZZ")
    res = run_backtest({"AAA": a, "ZZZ": z}, FirstAlphabetical(),
                       schedule="daily", cost_bps=0.0, warmup_days=2)
    assert any(t["symbol"] == "AAA" for t in res.trades)
    assert not any(t["symbol"] == "ZZZ" and t["side"] == "BUY" for t in res.trades)


def test_rebalance_band_kills_retrue_churn_but_not_entries():
    # Two names drifting slightly apart: an exact daily re-true trades dust
    # every day; the live-equivalent 1% band should trade only the entries.
    a = _bars([100 + 0.1 * i for i in range(30)], "AAA")
    b = _bars([100 - 0.1 * i for i in range(30)], "BBB")
    exact = run_backtest({"AAA": a, "BBB": b}, EqualWeight(),
                         schedule="daily", cost_bps=0.0, warmup_days=2)
    banded = run_backtest({"AAA": a, "BBB": b}, EqualWeight(),
                          schedule="daily", cost_bps=0.0, warmup_days=2,
                          rebalance_band=0.01)
    assert banded.stats["num_trades"] < exact.stats["num_trades"]
    buys = [t for t in banded.trades if t["side"] == "BUY"]
    assert {t["symbol"] for t in buys} >= {"AAA", "BBB"}   # entries unaffected
    # both end fully invested in the same names — the band only skips dust
    assert banded.stats["open_positions"] == exact.stats["open_positions"]
    assert banded.stats["return_pct"] == pytest.approx(
        exact.stats["return_pct"], abs=0.5)


def test_rebalance_band_never_blocks_a_full_close():
    # Strategy drops AAA entirely mid-run; even a tiny position must be SOLD
    # (the band exempts deltas that open or fully close a position).
    class DropsAAA:
        name = "drops_aaa"

        def __init__(self):
            self.calls = 0

        def rebalance(self, ctx):
            self.calls += 1
            if self.calls <= 2:
                return {"AAA": 0.01, "BBB": 0.99}    # AAA is a dust position
            return {"BBB": 0.99}                      # full exit of AAA

    a = _bars([100] * 12, "AAA")
    b = _bars([100] * 12, "BBB")
    res = run_backtest({"AAA": a, "BBB": b}, DropsAAA(), schedule="daily",
                       cost_bps=0.0, warmup_days=2, rebalance_band=0.05)
    aaa_sells = [t for t in res.trades
                 if t["symbol"] == "AAA" and t["side"] == "SELL"]
    assert aaa_sells, "full close must bypass the no-trade band"


def test_rebalance_band_zero_is_default_exact_retrue():
    a = _bars([100 + 0.1 * i for i in range(30)], "AAA")
    b = _bars([100 - 0.1 * i for i in range(30)], "BBB")
    default = run_backtest({"AAA": a, "BBB": b}, EqualWeight(),
                           schedule="daily", cost_bps=0.0, warmup_days=2)
    explicit = run_backtest({"AAA": a, "BBB": b}, EqualWeight(),
                            schedule="daily", cost_bps=0.0, warmup_days=2,
                            rebalance_band=0.0)
    assert default.trades == explicit.trades
    assert default.stats == explicit.stats
