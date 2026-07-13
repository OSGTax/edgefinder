"""Strategy Lab: grid, split-sample scoring, leaderboard, and the new rules.

The lab's honesty machinery is the test target: qualification requires
beating SPY in BOTH halves, ranking uses the WORST half, and the leaderboard
always carries the tested count next to the winners.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from agent.lab import build_grid, score_combo


# ── grid + scoring (pure) ──


def test_grid_is_deterministic_and_complete():
    g1, g2 = build_grid(), build_grid()
    assert g1 == g2
    assert len(g1) == len({(c["rule"], c["schedule"], c["universe"])
                           for c in g1})  # no duplicate combos
    assert len(g1) > 100  # a real sweep space, not a toy


def test_score_combo_requires_both_halves_positive():
    good = {"excess_return_pct": 12.0, "sharpe": 1.1, "max_drawdown_pct": -20}
    weak = {"excess_return_pct": 3.0, "sharpe": 0.6, "max_drawdown_pct": -25}
    bad = {"excess_return_pct": -4.0}

    v = score_combo(good, weak)
    assert v["qualifies"] is True
    assert v["score"] == 3.0  # the WORST half is the number a skeptic quotes

    assert score_combo(good, bad)["qualifies"] is False
    assert score_combo(bad, good)["qualifies"] is False
    assert score_combo({"excess_return_pct": None}, good)["qualifies"] is False


# ── the new rule families (point-in-time behavior on synthetic views) ──


def _view(sym, price, *, ema_200=None, rsi=None, closes=None):
    from edgefinder.data.market_data import IndicatorSnapshot
    from edgefinder.engine.strategy import AssetView

    hist = pd.DataFrame({"close": closes if closes is not None else [price] * 300})
    return AssetView(symbol=sym, price=price,
                     indicators=IndicatorSnapshot(close=price, ema_200=ema_200,
                                                  rsi=rsi),
                     history=hist)


def _ctx(views):
    from edgefinder.engine.strategy import RebalanceContext

    return RebalanceContext(date=date(2026, 7, 13),
                            assets={v.symbol: v for v in views})


def test_meanrev_buys_oversold_in_uptrends_only():
    from agent.backtest_tool import build_strategy

    s = build_strategy("meanrev:2")
    w = s.rebalance(_ctx([
        _view("UP_OVERSOLD", 100, ema_200=90, rsi=25),
        _view("UP_NEUTRAL", 100, ema_200=90, rsi=55),
        _view("DOWN_OVERSOLD", 100, ema_200=110, rsi=15),  # knife: below EMA
    ]))
    assert set(w) == {"UP_OVERSOLD", "UP_NEUTRAL"}  # knife excluded
    assert w["UP_OVERSOLD"] == pytest.approx(0.5)


def test_breakout_prefers_names_at_their_highs():
    from agent.backtest_tool import build_strategy

    s = build_strategy("breakout:1")
    # NEAR: rising into a fresh 252-bar high (positive 3m return, ratio 1.0).
    near = [50.0] * 189 + [float(x) for x in range(50, 113)]
    # FAR: 40% off its old high and flat lately (fails the 3m-return gate).
    far = [200.0] * 252 + [120.0] * 48
    w = s.rebalance(_ctx([
        _view("NEAR", 112.0, closes=near),
        _view("FAR", 120.0, closes=far),
    ]))
    assert list(w) == ["NEAR"]


def test_regime_momentum_goes_to_cash_below_spy_ema():
    from agent.backtest_tool import build_strategy

    s = build_strategy("regime_momentum:2")
    riser = [float(50 + i * 0.5) for i in range(300)]
    risk_off = _ctx([_view("SPY", 100, ema_200=110),
                     _view("AAA", 100, closes=riser)])
    assert s.rebalance(risk_off) == {}  # SPY below trend → all cash

    risk_on = _ctx([_view("SPY", 100, ema_200=90),
                    _view("AAA", 100, closes=riser)])
    w = s.rebalance(risk_on)
    assert list(w) == ["AAA"] and "SPY" not in w  # gauge never held


def test_momo_trend_requires_uptrend_membership():
    from agent.backtest_tool import build_strategy

    s = build_strategy("momo_trend:2")
    riser = [float(50 + i * 0.5) for i in range(300)]
    w = s.rebalance(_ctx([
        _view("TRENDING", 200.0, ema_200=150, closes=riser),
        _view("FALLEN", 200.0, ema_200=250, closes=riser),  # below its EMA
    ]))
    assert list(w) == ["TRENDING"]


# ── leaderboard (store round-trip) ──


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'lab.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def seed_lab_row(store, rule, uni, sched, *, qualifies, score, ts=None):
    store.insert("desk_backtests", {
        "account": "agent", "run_id": "lab-test",
        "ts": ts or datetime.utcnow(),
        "label": f"lab:{rule}@{uni}/{sched}",
        "spec": {"rule": rule, "universe": uni, "schedule": sched},
        "result": {"qualifies": qualifies, "score": score,
                   "in_sample_excess_pct": score,
                   "out_sample_excess_pct": score + 1}}, returning=False)


def test_leaderboard_ranks_by_worst_half_and_reports_tested(store):
    from agent.lab import leaderboard

    seed_lab_row(store, "momo_trend:5", "top40", "monthly",
                 qualifies=True, score=9.0)
    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=True, score=4.0)
    seed_lab_row(store, "breakout:5", "top20", "weekly",
                 qualifies=False, score=-2.0)
    # A non-lab backtest row must never leak into the board.
    store.insert("desk_backtests", {
        "account": "agent", "label": "momo_trend:5 on shortlist",
        "spec": {}, "result": {"qualifies": True, "score": 99}},
        returning=False)

    b = leaderboard(top=5)
    assert b["combos_tested"] == 3
    assert b["qualified"] == 2
    assert [e["rule"] for e in b["top"]] == ["momo_trend:5", "momentum:5"]
    assert "expect live shrinkage" in b["honesty"]


def test_leaderboard_dedupes_to_newest_result(store):
    from agent.lab import leaderboard

    old = datetime.utcnow() - timedelta(days=3)
    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=True, score=8.0, ts=old)
    # The same combo re-tested tonight FAILED its out-sample half.
    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=False, score=-1.0)

    b = leaderboard(top=5)
    assert b["combos_tested"] == 1
    assert b["qualified"] == 0  # tonight's verdict wins; stale wins don't linger
