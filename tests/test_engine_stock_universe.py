"""Tests for the stock-universe machine: PIT universes, realistic costs,
dividend adjustment."""

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.backtest.costs import CostModel
from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.data import adjust_for_dividends
from edgefinder.engine.strategy import BuyAndHold, EqualWeight
from edgefinder.engine.walkforward import plan_folds, run_walkforward


def _bars(n: int, price: float = 100.0, drift: float = 0.1,
          start: date = date(2020, 1, 1), volume: float = 1e6,
          spread: float = 0.0) -> pd.DataFrame:
    dates = [start + timedelta(days=i) for i in range(n)]
    px = [price + i * drift for i in range(n)]
    return pd.DataFrame({
        "date": dates, "open": px,
        "high": [p * (1 + spread) for p in px],
        "low": [p * (1 - spread) for p in px],
        "close": px, "volume": [volume] * n,
    })


class TestPlanFolds:
    def test_geometry_matches_run_walkforward(self):
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(420)]
        folds, holdout = plan_folds(days, is_days=100, oos_days=50,
                                    step_days=50, holdout_days=50)
        assert len(folds) == 5
        assert folds[0] == (days[100], days[149])
        assert folds[1] == (days[150], days[199])
        assert holdout == (days[370], days[419])

    def test_holdout_start_pins(self):
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(420)]
        folds, holdout = plan_folds(days, is_days=100, oos_days=50,
                                    step_days=50, holdout_start=days[300])
        assert holdout == (days[300], days[419])
        assert all(end < days[300] for _, end in folds)


class TestPITUniverse:
    def test_each_fold_trades_only_its_universe(self):
        bars = {"AAA": _bars(420), "BBB": _bars(420, price=50.0)}
        days = sorted(bars["AAA"]["date"])
        folds, _ = plan_folds(days, is_days=100, oos_days=50,
                              step_days=50, holdout_days=50)
        # BBB enters the universe only from the third fold on
        per_window = {start: (["AAA"] if i < 2 else ["AAA", "BBB"])
                      for i, (start, _) in enumerate(folds)}
        card = run_walkforward(
            bars, EqualWeight, schedule="daily", warmup_days=10,
            is_days=100, oos_days=50, step_days=50,
            holdout_days=50, holdout_eval=False,
            universe_fn=per_window.get)
        assert card["config"]["pit_universe"] is True
        # folds 0-1: single-asset equal weight -> ~100% AAA; folds 2+:
        # two assets. Distinguish via fold return profiles: with BBB's
        # different price the fill pattern differs; assert via trades count
        # instead: fold 0 has 1 buy (AAA), fold 2 has 2 (AAA+BBB).
        assert card["folds"][0]["trades"] < card["folds"][2]["trades"]


class TestCostModel:
    def test_flat_path_unchanged_when_no_cost_model(self):
        bars = {"AAA": _bars(100)}
        a = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                         cost_bps=2.0, warmup_days=2)
        b = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                         cost_bps=2.0, warmup_days=2, cost_model=None)
        assert a.stats == b.stats

    def test_costed_fills_pay_liquidity_tiered_floor(self):
        # mega-liquid name (ADV ~$100B) pays the 2bp-tier floor, not the
        # microcap 0.5% — but still pays SOMETHING vs frictionless
        bars = {"AAA": _bars(100, volume=1e9)}
        flat = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                            cost_bps=0.0, warmup_days=30,
                            start_cash=1_000_000.0)
        costed = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                              cost_bps=0.0, warmup_days=30,
                              start_cash=1_000_000.0, cost_model=CostModel())
        assert costed.stats["return_pct"] < flat.stats["return_pct"]
        fill_flat = flat.trades[0]["price"]
        fill_costed = costed.trades[0]["price"]
        assert fill_costed >= fill_flat * (1 + 0.0002 / 2)      # tiered floor
        assert fill_costed < fill_flat * (1 + 0.005 / 2)        # NOT microcap

    def test_thin_name_pays_microcap_floor(self):
        # ADV ~$300k: tradeable but thin -> full 0.5% floor applies
        bars = {"THIN": _bars(100, price=3.0, volume=1e5)}
        costed = run_backtest(bars, BuyAndHold("THIN"), schedule="daily",
                              cost_bps=0.0, warmup_days=30,
                              start_cash=10_000.0, cost_model=CostModel())
        first = next(t for t in costed.trades if t["side"] == "BUY")
        assert first["price"] >= 3.0 * (1 + 0.005 / 2)

    def test_illiquid_name_is_untradeable(self):
        # ADV ~ $500/day << min_adv_dollars -> cap_shares refuses the trade
        bars = {"THIN": _bars(100, price=5.0, volume=100.0)}
        res = run_backtest(bars, BuyAndHold("THIN"), schedule="daily",
                           cost_bps=0.0, warmup_days=30, cost_model=CostModel())
        assert res.trades == []
        assert res.stats["return_pct"] == 0.0

    def test_participation_cap_limits_size(self):
        # ADV $100k/day, 5% cap -> at most ~$5k of fills despite $1M equity
        bars = {"SMALL": _bars(100, price=10.0, volume=10_000.0)}
        res = run_backtest(bars, BuyAndHold("SMALL"), schedule="daily",
                           cost_bps=0.0, warmup_days=30,
                           start_cash=1_000_000.0, cost_model=CostModel())
        first_buy = next(t for t in res.trades if t["side"] == "BUY")
        assert first_buy["shares"] * 10.0 <= 0.05 * 100_000 * 1.1


class TestReviewFixes:
    def test_costed_delist_pays_spread_floor_not_flat_bps(self):
        # a holding whose data ends mid-run must liquidate THROUGH the model
        long = _bars(80, volume=1e6)
        short = _bars(60, price=50.0, volume=1e6)      # dies 20 days early
        res = run_backtest({"AAA": long, "DEAD": short}, EqualWeight(),
                           schedule="daily", cost_bps=0.0, warmup_days=30,
                           start_cash=1_000_000.0, cost_model=CostModel())
        delist = next(t for t in res.trades if t.get("reason") == "DELISTED")
        last_close = float(short["close"].iloc[-1])
        # ADV ~$50M -> 0.0005-tier floor; proceeds at most last_close minus
        # half the tiered spread (and impact on a full-position dump)
        assert delist["price"] <= last_close * (1 - 0.0005 / 2) + 1e-6

    def test_spread_floor_tiers_by_liquidity(self):
        cm = CostModel()
        assert cm.spread_floor_for(5e9) == 0.0002      # mega-cap
        assert cm.spread_floor_for(100e6) == 0.0005
        assert cm.spread_floor_for(10e6) == 0.002
        assert cm.spread_floor_for(1e6) == 0.005       # microcap floor
        assert cm.spread_floor_for(None) == 0.005      # legacy callers

    def test_cost_fraction_clamped_no_negative_fills(self):
        cm = CostModel()
        # absurd participation: order 100x ADV at high vol
        fill = cm.fill_price(10.0, "SELL", order_dollars=1e7,
                             adv_dollars=1e5, volatility=0.5, spread_frac=0.01)
        assert fill >= 10.0 * (1 - cm.MAX_COST_FRACTION) > 0

    def test_collapsed_liquidity_freezes_holding_not_dump(self):
        # liquid long enough to buy, then volume collapses below the
        # untradeable floor while the strategy still wants the name
        df = _bars(120, price=10.0, volume=1e6)
        df.loc[60:, "volume"] = 10.0                   # ADV collapses
        res = run_backtest({"AAA": df}, BuyAndHold("AAA"), schedule="daily",
                           cost_bps=0.0, warmup_days=30, cost_model=CostModel())
        sells = [t for t in res.trades if t["side"] == "SELL"]
        assert sells == []                             # frozen, never dumped
        assert res.stats["open_positions"] == 1

    def test_future_declared_dividend_leaves_series_untouched(self):
        df = _bars(10, price=100.0, drift=0.0)
        future_ex = df["date"].iloc[-1] + timedelta(days=10)
        out = adjust_for_dividends({"AAA": df}, {"AAA": [(future_ex, 1.0)]})["AAA"]
        assert out["close"].iloc[0] == 100.0           # nothing rescaled
        assert out["close"].iloc[-1] == 100.0

    def test_adjusted_frames_carry_raw_close(self):
        df = _bars(10, price=100.0, drift=0.0)
        out = adjust_for_dividends(
            {"AAA": df}, {"AAA": [(df["date"].iloc[5], 1.0)]})["AAA"]
        assert "close_raw" in out.columns
        assert out["close_raw"].iloc[0] == 100.0       # raw preserved
        assert out["close"].iloc[0] == pytest.approx(99.0)

    def test_calendar_mismatch_fails_loud(self):
        bars = {"AAA": _bars(420)}
        with pytest.raises(ValueError, match="different calendar"):
            run_walkforward(
                bars, EqualWeight, schedule="daily", warmup_days=10,
                is_days=100, oos_days=50, step_days=50,
                universe_fn=lambda d: None)            # resolver misses


class TestDividendAdjustment:
    def test_back_adjustment_math(self):
        df = _bars(10, price=100.0, drift=0.0)        # flat $100
        dates = list(df["date"])
        adjusted = adjust_for_dividends(
            {"AAA": df}, {"AAA": [(dates[5], 1.0)]})["AAA"]
        # rows before the ex-date scaled by 1 - 1/100; later rows unchanged
        assert adjusted["close"].iloc[4] == pytest.approx(99.0)
        assert adjusted["close"].iloc[5] == 100.0
        assert adjusted["close"].iloc[-1] == 100.0
        assert adjusted["open"].iloc[0] == pytest.approx(99.0)

    def test_total_return_exceeds_price_return(self):
        # flat price + dividends -> adjusted buy-and-hold shows a positive
        # return (the reinvested dividends), raw shows zero
        df = _bars(300, price=100.0, drift=0.0)
        dates = list(df["date"])
        divs = [(dates[i], 0.5) for i in (60, 120, 180, 240)]
        adjusted = adjust_for_dividends({"AAA": df}, {"AAA": divs})["AAA"]
        tr = (adjusted["close"].iloc[-1] / adjusted["close"].iloc[0] - 1) * 100
        assert tr == pytest.approx(4 * 0.5, abs=0.1)   # ~2% over the window

    def test_no_dividends_is_identity(self):
        df = _bars(10)
        out = adjust_for_dividends({"AAA": df}, {})
        assert out["AAA"] is df

    def test_bogus_dividend_ignored(self):
        df = _bars(10, price=100.0, drift=0.0)
        dates = list(df["date"])
        out = adjust_for_dividends(
            {"AAA": df}, {"AAA": [(dates[5], 150.0)]})["AAA"]   # div > price
        assert out["close"].iloc[0] == 100.0
