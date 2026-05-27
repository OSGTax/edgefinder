"""Tests for the minute-bar backtester (synthetic bars, no network)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from edgefinder.backtest.engine import BacktestEngine, Order
from edgefinder.backtest.examples import SmaCrossStrategy

START = datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc)


def _bars(symbol, closes):
    ts = [START + timedelta(minutes=i) for i in range(len(closes))]
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": ts,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * len(closes),
        }
    )


class _BuyOnce:
    def __init__(self):
        self.bought = False

    def on_bar(self, bar, ctx):
        if not self.bought:
            self.bought = True
            return [Order(bar.symbol, "BUY", 10)]
        return None


def test_buy_and_mark_to_market():
    engine = BacktestEngine(starting_cash=10_000.0, slippage=0.0)
    result = engine.run(_bars("NVDA", [100, 110, 120]), _BuyOnce())
    assert result.num_fills == 1
    # 10 sh @ 100 -> cash 9000; marked at last close 120 -> 9000 + 1200
    assert result.final_equity == pytest.approx(10_200.0)
    assert result.return_pct == pytest.approx(2.0)
    assert len(result.equity_curve) == 3


def test_realized_pnl_on_sell():
    class Strat:
        def on_bar(self, bar, ctx):
            if bar.close == 100.0 and ctx.position(bar.symbol) is None:
                return [Order(bar.symbol, "BUY", 10)]
            if bar.close == 120.0 and ctx.position(bar.symbol) is not None:
                return [Order(bar.symbol, "SELL", 10)]
            return None

    engine = BacktestEngine(starting_cash=10_000.0, slippage=0.0)
    result = engine.run(_bars("NVDA", [100, 110, 120]), Strat())
    assert result.num_fills == 2
    assert result.realized_pnl == pytest.approx(200.0)
    assert result.final_equity == pytest.approx(10_200.0)


def test_insufficient_cash_clips_quantity():
    engine = BacktestEngine(starting_cash=150.0, slippage=0.0)
    # wants 10 @ 100 = 1000 but only 150 cash -> buys 1
    result = engine.run(_bars("NVDA", [100, 100]), _BuyOnce())
    assert result.num_fills == 1
    assert result.fills[0].quantity == 1


def test_run_requires_columns():
    engine = BacktestEngine()
    with pytest.raises(ValueError):
        engine.run(pd.DataFrame({"symbol": ["X"]}), _BuyOnce())


def test_sma_cross_executes_trades():
    # Down, then up (golden cross -> BUY), then down (death cross -> SELL).
    closes = [140 - i for i in range(40)] + [100 + i for i in range(40)] + [139 - i for i in range(20)]
    engine = BacktestEngine(starting_cash=10_000.0)
    result = engine.run(_bars("NVDA", closes), SmaCrossStrategy(fast=5, slow=20, target_dollars=2_000))
    assert result.num_fills >= 1
    assert {f.side for f in result.fills} <= {"BUY", "SELL"}
