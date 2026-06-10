"""Tests for the engine-v2 walk-forward harness + the trade_start affordance."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.record import record_validation_run
from edgefinder.engine.strategy import BuyAndHold, EqualWeight
from edgefinder.engine.walkforward import Fold, _aggregate, _slice, run_walkforward


def _bars(n: int, start_price: float = 100.0, drift: float = 0.1,
          start: date = date(2020, 1, 1)) -> pd.DataFrame:
    """n synthetic daily bars with a linear price path (open == close)."""
    dates = [start + timedelta(days=i) for i in range(n)]
    px = [start_price + i * drift for i in range(n)]
    return pd.DataFrame({
        "date": dates, "open": px, "high": px, "low": px, "close": px,
        "volume": [1e6] * n,
    })


class TestTradeStart:
    def test_equity_marks_start_at_trade_start(self):
        bars = {"AAA": _bars(100)}
        ts = bars["AAA"]["date"][40]
        res = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                           cost_bps=0.0, trade_start=ts)
        assert res.equity_curve[0][0] == ts
        assert len(res.equity_curve) == 60
        assert all(t["date"] >= ts for t in res.trades)

    def test_scored_region_matches_buy_and_hold_anchor(self):
        bars = {"AAA": _bars(100)}
        df = bars["AAA"]
        ts = df["date"][40]
        res = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                           cost_bps=0.0, start_cash=1_000_000.0, trade_start=ts)
        # first fill is at the trade_start bar's open; flooring noise is
        # negligible at this start_cash
        expected = (df["close"].iloc[-1] / df["open"][40] - 1) * 100
        assert res.stats["return_pct"] == pytest.approx(expected, abs=0.05)

    def test_none_preserves_index_warmup_behavior(self):
        bars = {"AAA": _bars(100)}
        res = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                           cost_bps=0.0, warmup_days=2)
        assert len(res.equity_curve) == 100   # warmup days still marked

    def test_first_scored_bar_rebalances_on_slow_schedules(self):
        # trade_start mid-month must fill immediately, not wait for the next
        # month boundary — the fold-start dead-cash artifact (review finding)
        bars = {"AAA": _bars(100)}
        ts = bars["AAA"]["date"][40]            # mid-month by construction
        assert ts.day not in (1, 2)
        res = run_backtest(bars, BuyAndHold("AAA"), schedule="monthly",
                           cost_bps=0.0, trade_start=ts)
        assert res.trades and res.trades[0]["date"] == ts

    def test_self_benchmark_measures_zero_excess(self):
        # buy-and-hold scored against its own bars must read ~0 excess —
        # the open-anchored benchmark kills the day-one half-day mismatch
        bars = {"AAA": _bars(100)}
        ts = bars["AAA"]["date"][40]
        res = run_backtest(bars, BuyAndHold("AAA"), schedule="daily",
                           cost_bps=0.0, start_cash=1_000_000.0,
                           trade_start=ts,
                           benchmark=_slice(bars["AAA"], ts,
                                            bars["AAA"]["date"][99]))
        assert res.stats["excess_return_pct"] == pytest.approx(0.0, abs=0.05)


class TestFoldGeometry:
    def test_fold_count_windows_and_sealed_holdout(self):
        bars = {"AAA": _bars(420), "BBB": _bars(420, start_price=50.0)}
        days = sorted(bars["AAA"]["date"])
        card = run_walkforward(
            bars, EqualWeight, schedule="daily", warmup_days=10,
            is_days=100, oos_days=50, step_days=50,
            holdout_days=50, holdout_eval=False)
        assert card["config"]["num_folds"] == 5
        assert card["holdout"] is None
        assert card["config"]["holdout_evaluated"] is False
        # the sealed boundary is pinned in the record even though unevaluated
        assert card["config"]["holdout_window"] == f"{days[370]}..{days[419]}"
        # no fold may touch the final 50 days (the sealed region)
        last_oos_end = max(
            date.fromisoformat(f["window"].split("..")[1]) for f in card["folds"])
        assert last_oos_end < days[370]
        # folds step contiguously
        assert card["folds"][0]["window"].startswith(str(days[100]))
        assert card["folds"][1]["window"].startswith(str(days[150]))

    def test_holdout_evaluated_when_requested(self):
        bars = {"AAA": _bars(420)}
        days = sorted(bars["AAA"]["date"])
        card = run_walkforward(
            bars, lambda: BuyAndHold("AAA"), schedule="daily", warmup_days=10,
            is_days=100, oos_days=50, step_days=50,
            holdout_days=50, holdout_eval=True)
        assert card["holdout"] is not None
        assert card["holdout"]["window"] == f"{days[370]}..{days[419]}"
        assert isinstance(card["holdout"]["passes"], bool)
        assert card["config"]["holdout_evaluated"] is True

    def test_insufficient_history_raises(self):
        bars = {"AAA": _bars(120)}
        with pytest.raises(ValueError, match="not enough history"):
            run_walkforward(bars, EqualWeight, is_days=100, oos_days=50)

    def test_holdout_start_pins_the_sealed_boundary(self):
        bars = {"AAA": _bars(420)}
        days = sorted(bars["AAA"]["date"])
        pin = days[370]
        card = run_walkforward(
            bars, EqualWeight, schedule="daily", warmup_days=10,
            is_days=100, oos_days=50, step_days=50,
            holdout_start=pin, holdout_eval=False)
        assert card["config"]["holdout_window"] == f"{days[370]}..{days[419]}"
        # new bars accrue -> the sealed START must NOT move (the whole point)
        bars2 = {"AAA": _bars(460)}
        card2 = run_walkforward(
            bars2, EqualWeight, schedule="daily", warmup_days=10,
            is_days=100, oos_days=50, step_days=50,
            holdout_start=pin, holdout_eval=False)
        assert card2["config"]["holdout_window"].startswith(str(pin))
        # and folds still never touch the sealed region
        last_oos_end = max(
            date.fromisoformat(f["window"].split("..")[1]) for f in card2["folds"])
        assert last_oos_end < pin


def _fold(i, excess_sharpe, dd_reduction, trades=10, excess_ret=1.0):
    return Fold(
        index=i, oos_start=date(2021, 1, 1), oos_end=date(2021, 6, 30),
        regime="bull_calm",
        stats={"return_pct": 5.0, "sharpe": 1.0,
               "excess_return_pct": excess_ret, "excess_sharpe": excess_sharpe,
               "drawdown_reduction_pct": dd_reduction,
               "max_drawdown_pct": 10.0, "num_trades": trades})


_AGG_KW = dict(is_days=100, oos_days=50, step_days=50, warmup_days=10,
               schedule="daily", cost_bps=2.0, start_cash=10_000.0,
               holdout=None, holdout_days=0, holdout_eval=True,
               pass_min_trades=30)


class TestCriteria:
    def test_risk_adjusted_all_met(self):
        folds = [_fold(0, 0.5, 2.0), _fold(1, 0.3, 1.0), _fold(2, -0.1, 3.0)]
        card = _aggregate("s", folds, risk_adjusted=True, **_AGG_KW)
        c = card["criteria"]
        assert c["mode"] == "risk_adjusted"
        assert c["sharpe_beats_spy"] is True          # mean 0.23 > 0
        assert c["majority_folds_higher_sharpe"] is True   # 2/3
        assert c["lower_drawdown_than_spy"] is True
        assert c["traded"] is True
        assert c["all_met"] is True
        assert card["verdict"] == "PASS"

    def test_risk_adjusted_tie_fails_majority(self):
        # 2/4 folds higher Sharpe is NOT a strict majority
        folds = [_fold(0, 0.5, 2.0), _fold(1, 0.5, 2.0),
                 _fold(2, -0.1, 2.0), _fold(3, -0.1, 2.0)]
        card = _aggregate("s", folds, risk_adjusted=True, **_AGG_KW)
        assert card["criteria"]["majority_folds_higher_sharpe"] is False
        assert card["criteria"]["all_met"] is False
        assert card["verdict"] == "FAIL"

    def test_total_return_min_trades_floor(self):
        folds = [_fold(0, 0.5, 2.0, trades=5), _fold(1, 0.5, 2.0, trades=5)]
        card = _aggregate("s", folds, risk_adjusted=False, **_AGG_KW)
        assert card["criteria"]["mode"] == "total_return"
        assert card["criteria"]["min_trades_met"] is False  # 10 fills < 30
        assert card["criteria"]["all_met"] is False

    def test_negative_drawdown_reduction_fails(self):
        folds = [_fold(0, 0.5, -1.0), _fold(1, 0.5, -2.0)]
        card = _aggregate("s", folds, risk_adjusted=True, **_AGG_KW)
        assert card["criteria"]["lower_drawdown_than_spy"] is False
        assert card["criteria"]["all_met"] is False

    def test_all_cash_folds_count_in_sharpe_denominator(self):
        # a flat fold (sharpe None) with a known benchmark Sharpe is imputed
        # excess_sharpe = -benchmark_sharpe, so a bull fold a cash-timer sat
        # out counts AGAINST it (review finding: the silent-drop hole)
        flat_bull = Fold(
            index=2, oos_start=date(2021, 1, 1), oos_end=date(2021, 6, 30),
            regime="bull_calm",
            stats={"return_pct": 0.0, "sharpe": None,
                   "excess_return_pct": -10.0, "benchmark_sharpe": 1.5,
                   "drawdown_reduction_pct": 8.0,
                   "max_drawdown_pct": 0.0, "num_trades": 0})
        folds = [_fold(0, 0.5, 2.0), _fold(1, 0.4, 1.0), flat_bull]
        card = _aggregate("s", folds, risk_adjusted=True, **_AGG_KW)
        assert card["oos"]["folds_higher_sharpe"] == "2/3"   # not 2/2
        assert card["folds"][2]["excess_sharpe"] == -1.5
        # and a flat fold in a BEAR counts positively (symmetric)
        flat_bear = Fold(
            index=2, oos_start=date(2022, 1, 1), oos_end=date(2022, 6, 30),
            regime="bear_volatile",
            stats={"return_pct": 0.0, "sharpe": None,
                   "excess_return_pct": 15.0, "benchmark_sharpe": -1.2,
                   "drawdown_reduction_pct": 20.0,
                   "max_drawdown_pct": 0.0, "num_trades": 0})
        card2 = _aggregate("s", [flat_bear], risk_adjusted=True, **_AGG_KW)
        assert card2["folds"][0]["excess_sharpe"] == 1.2

    def test_compounded_return_omitted_when_folds_overlap(self):
        folds = [_fold(0, 0.5, 2.0), _fold(1, 0.4, 1.0)]
        kw = dict(_AGG_KW, step_days=25)        # step < oos: windows overlap
        card = _aggregate("s", folds, risk_adjusted=True, **kw)
        assert card["oos"]["total_return_pct"] is None


class TestScorecardPlumbing:
    def test_scorecard_is_json_serializable(self):
        bars = {"AAA": _bars(420)}
        card = run_walkforward(
            bars, lambda: BuyAndHold("AAA"), schedule="daily", warmup_days=10,
            is_days=100, oos_days=50, step_days=50,
            holdout_days=50, holdout_eval=True)
        json.dumps(card)   # raises TypeError on any stray date object

    def test_record_and_validated_semantics(self):
        engine = get_engine(url="sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = get_session_factory(engine)()
        card = {
            "strategy": "ew_test", "config": {"engine": "v2"},
            "oos": {"mean_excess_sharpe": 0.2},
            "criteria": {"mode": "risk_adjusted", "all_met": True},
            "holdout": {"window": "2026-01-01..2026-06-01", "passes": True},
            "verdict": "PASS",
        }
        run_id = record_validation_run(session, card, universe="etf7+v2",
                                       git_sha="abc1234")
        from edgefinder.db.models import ValidationRun
        row = session.get(ValidationRun, run_id)
        assert row.strategy_name == "ew_test"
        assert row.universe == "etf7+v2"
        # the dashboard's validated rule
        validated = bool(row.criteria.get("all_met")
                         and row.holdout is not None
                         and row.holdout.get("passes"))
        assert validated is True
        # sealed holdout -> not validated, even with criteria met
        card2 = dict(card, strategy="ew_sealed", holdout=None)
        run_id2 = record_validation_run(session, card2, universe="etf7+v2")
        row2 = session.get(ValidationRun, run_id2)
        assert row2.holdout is None
        assert not (row2.criteria.get("all_met") and row2.holdout)
        session.close()
