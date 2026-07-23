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
    # The mid-tier slice is in the grid: rules must survive beyond megacaps.
    assert {c["universe"] for c in g1} == {"top20", "top40", "top60", "mid200"}


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
    import edgefinder.db.models  # noqa: F401 — daily_bars/dividends tables

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def seed_lab_row(store, rule, uni, sched, *, qualifies, score, ts=None,
                 basis=None, extra=None):
    result = {"qualifies": qualifies, "score": score,
              "in_sample_excess_pct": score,
              "out_sample_excess_pct": score + 1}
    if basis is not None:
        result["universe_basis"] = basis
    if extra:
        result.update(extra)
    store.insert("desk_backtests", {
        "account": "agent", "run_id": "lab-test",
        "ts": ts or datetime.utcnow(),
        "label": f"lab:{rule}@{uni}/{sched}",
        "spec": {"rule": rule, "universe": uni, "schedule": sched},
        "result": result}, returning=False)


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


def test_lab_endpoint_serves_leaderboard(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None
    seed_lab_row(store, "regime_momentum:3", "top60", "monthly",
                 qualifies=True, score=12.0)

    from dashboard.app import app

    with TestClient(app) as c:
        r = c.get("/api/desk/lab").json()
        assert r["combos_tested"] == 1 and r["qualified"] == 1
        assert r["top"][0]["rule"] == "regime_momentum:3"
        assert "shrinkage" in r["honesty"]
        # The desk page carries the lab card + loader.
        page = c.get("/desk").text
        assert 'id="desk-lab"' in page


def test_leaderboard_carries_universe_basis_and_flags_survivorship(store):
    from agent.lab import SPLIT_DATE, leaderboard

    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=True, score=6.0, basis=f"as_of_{SPLIT_DATE}")
    # A row persisted before the basis stamp existed — those sweeps all
    # ranked their universe present-day, so it must be labeled as such.
    seed_lab_row(store, "breakout:5", "mid200", "weekly",
                 qualifies=True, score=4.0)

    b = leaderboard(top=5)
    by_rule = {e["rule"]: e for e in b["top"]}
    assert by_rule["momentum:5"]["universe_basis"] == f"as_of_{SPLIT_DATE}"
    assert (by_rule["breakout:5"]["universe_basis"]
            == "present_day_survivorship_inflated")
    assert "survivorship" in b["honesty"]


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


def test_leaderboard_tracks_qualification_streak_and_flags_it(store):
    from agent.lab import STREAK_CLAIM_THRESHOLD, leaderboard

    now = datetime.utcnow()
    for i in range(STREAK_CLAIM_THRESHOLD):
        seed_lab_row(store, "momentum:5", "top40", "monthly",
                     qualifies=True, score=5.0,
                     ts=now - timedelta(days=STREAK_CLAIM_THRESHOLD - i))
    b = leaderboard(top=5)
    entry = next(e for e in b["top"] if e["rule"] == "momentum:5")
    assert entry["qualified_streak"] == STREAK_CLAIM_THRESHOLD
    assert any(f["rule"] == "momentum:5" for f in b["flagged_for_claim"])


def test_leaderboard_streak_stops_at_a_failed_night(store):
    from agent.lab import leaderboard

    now = datetime.utcnow()
    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=True, score=5.0, ts=now - timedelta(days=3))
    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=False, score=-1.0, ts=now - timedelta(days=2))
    seed_lab_row(store, "momentum:5", "top40", "monthly",
                 qualifies=True, score=5.0, ts=now - timedelta(days=1))
    b = leaderboard(top=5)
    entry = next(e for e in b["top"] if e["rule"] == "momentum:5")
    assert entry["qualified_streak"] == 1  # only the newest night counts
    assert not any(f["rule"] == "momentum:5" for f in b["flagged_for_claim"])


# ── value_momentum: PIT fundamentals gate (validation-passed 2026-07-14) ──


class _FakeFunds:
    """raw_asof stand-in: {sym: (first_filed, raw_dict)} — None before
    coverage, the dict after (mirrors PITFundamentals semantics)."""

    def __init__(self, rows):
        self.rows = rows

    def raw_asof(self, sym, d):
        ent = self.rows.get(sym)
        if not ent:
            return None
        filed, raw = ent
        return raw if d >= filed else None


def _rising(price, gain=0.5, n=300):
    return [price * (1 + gain * i / (n - 1)) / (1 + gain) for i in range(n)]


def _funds(sym_specs):
    return _FakeFunds({sym: (date(2010, 1, 1),
                             {"_net_income_ttm": ni, "_shares": sh})
                       for sym, (ni, sh) in sym_specs.items()})


def test_value_momentum_excludes_losses_and_expensive_half():
    from agent.backtest_tool import _ValueMomentum

    s = _ValueMomentum(k=2, fundamentals=_funds({
        "CHEAP_FAST": (100.0, 10.0),   # P/E 10 at price 100
        "CHEAP_SLOW": (50.0, 10.0),    # P/E 20
        "PRICY": (10.0, 10.0),         # P/E 100 — above the median cutoff
        "LOSS": (-5.0, 10.0),          # loss-maker — never qualifies
    }), splits={})
    w = s.rebalance(_ctx([
        _view("CHEAP_FAST", 100, closes=_rising(100, gain=0.8)),
        _view("CHEAP_SLOW", 100, closes=_rising(100, gain=0.2)),
        _view("PRICY", 100, closes=_rising(100, gain=0.9)),
        _view("LOSS", 100, closes=_rising(100, gain=0.9)),
    ]))
    assert set(w) == {"CHEAP_FAST", "CHEAP_SLOW"}


def test_value_momentum_split_basis_correction():
    from agent.backtest_tool import _ValueMomentum

    funds = {"SPLITCO": (100.0, 10.0), "PLAIN": (100.0, 10.0)}
    views = [_view("SPLITCO", 100, closes=_rising(100)),
             _view("PLAIN", 100, closes=_rising(100))]

    # A 4:1 split AFTER the decision date (2026-07-13): filed shares must be
    # multiplied ×4 to match the back-adjusted price basis → P/E 40 vs 10,
    # so SPLITCO lands above the median cutoff and is excluded.
    s = _ValueMomentum(k=2, fundamentals=_funds(funds),
                       splits={"SPLITCO": [(date(2026, 8, 1), 4.0)]})
    assert set(s.rebalance(_ctx(views))) == {"PLAIN"}

    # The same split BEFORE the decision date changes nothing — filed shares
    # already reflect it. Both names read P/E 10 and both are held.
    s2 = _ValueMomentum(k=2, fundamentals=_funds(funds),
                        splits={"SPLITCO": [(date(2026, 1, 2), 4.0)]})
    assert set(s2.rebalance(_ctx(views))) == {"SPLITCO", "PLAIN"}


def test_value_momentum_holds_cash_before_coverage():
    from agent.backtest_tool import _ValueMomentum

    # First filing arrives AFTER the decision date → nothing is knowable,
    # nothing is held. Pre-2009 honesty: no borrowed future fundamentals.
    late = _FakeFunds({"AAA": (date(2027, 1, 1),
                               {"_net_income_ttm": 100.0, "_shares": 10.0})})
    s = _ValueMomentum(k=2, fundamentals=late, splits={})
    assert s.rebalance(_ctx([_view("AAA", 100, closes=_rising(100))])) == {}


def test_value_momentum_in_lab_grid():
    grid = build_grid()
    rules = {g["rule"] for g in grid}
    assert {"value_momentum:5", "value_momentum:8"} <= rules


# ── universe basis: PIT-as-of-SPLIT_DATE with an honest fallback label ──


def test_resolve_universe_prefers_split_date_ranking(monkeypatch):
    import agent.data as agent_data
    from agent import lab

    calls = []

    def fake_universe(n, *, as_of=None):
        calls.append((n, as_of))
        return [f"S{i}" for i in range(n)]

    monkeypatch.setattr(lab, "_pit_breadth", lambda as_of, *, need: need)
    monkeypatch.setattr(agent_data, "universe", fake_universe)
    syms, basis = lab._resolve_universe("top20")
    assert basis == f"as_of_{lab.SPLIT_DATE}"
    assert calls == [(20, lab.SPLIT_DATE)]  # ranked at the boundary, once
    assert "SPY" in syms


def test_resolve_universe_falls_back_and_labels_survivorship(monkeypatch):
    import agent.data as agent_data
    from agent import lab

    monkeypatch.setattr(lab, "_pit_breadth", lambda as_of, *, need: need)

    def thin_pit(n, *, as_of=None):
        if as_of is not None:
            return ["LONE"]  # hot set holds almost nothing near 2018
        return [f"S{i}" for i in range(n)]

    monkeypatch.setattr(agent_data, "universe", thin_pit)
    syms, basis = lab._resolve_universe("top40")
    assert basis == "present_day_survivorship_inflated"
    assert len([s for s in syms if s != "SPY"]) == 40  # present-day ranking

    # mid200 slices ranks 41-240: a thin PIT slice trips its floor too
    def thin_mid(n, *, as_of=None):
        return [f"S{i}" for i in range(60 if as_of else n)]

    monkeypatch.setattr(agent_data, "universe", thin_mid)
    _, basis = lab._resolve_universe("mid200")
    assert basis == "present_day_survivorship_inflated"


def test_resolve_universe_survives_pit_ranking_errors(monkeypatch):
    import agent.data as agent_data
    from agent import lab

    monkeypatch.setattr(lab, "_pit_breadth", lambda as_of, *, need: need)

    def flaky(n, *, as_of=None):
        if as_of is not None:
            raise RuntimeError("no bars near as_of")
        return [f"S{i}" for i in range(n)]

    monkeypatch.setattr(agent_data, "universe", flaky)
    syms, basis = lab._resolve_universe("top20")
    assert basis == "present_day_survivorship_inflated"
    assert len(syms) >= 20


def test_sweep_stamps_universe_basis_on_results(store, monkeypatch):
    import pandas as pd

    import agent.backtest_tool as backtest_tool
    import agent.data as agent_data
    from agent import lab

    # SQLite store has no bars near 2018 → the breadth probe finds nothing →
    # present-day fallback, and the label must say so end to end.
    monkeypatch.setattr(
        agent_data, "universe",
        lambda n, *, as_of=None: [] if as_of else [f"S{i}" for i in range(n)])
    frame = pd.DataFrame({
        "date": [date(2006, 1, 2) + timedelta(days=i) for i in range(220)],
        "open": [100.0] * 220, "high": [100.0] * 220, "low": [100.0] * 220,
        "close": [100.0] * 220, "volume": [1e6] * 220})
    monkeypatch.setattr(agent_data, "load_bars",
                        lambda syms, **kw: {s: frame.copy() for s in syms})
    monkeypatch.setattr(agent_data, "spy_series_df",
                        lambda **kw: frame.copy())
    monkeypatch.setattr(backtest_tool, "run_prepared",
                        lambda *a, **kw: {"excess_return_pct": 5.0,
                                          "sharpe": 1.0,
                                          "max_drawdown_pct": -10.0})

    res = lab.sweep(max_combos=1, offset=0)
    assert res["tested"] == 1
    assert res["top"][0]["universe_basis"] == "present_day_survivorship_inflated"
    rows = [r for r in store.select("desk_backtests")
            if str(r.get("label") or "").startswith("lab:")]
    assert rows
    assert (rows[0]["result"]["universe_basis"]
            == "present_day_survivorship_inflated")


# ── H1: the protected-ETF collision must never mislabel a PIT universe ──


def seed_bars(store, symbol, dates, *, volume=1_000_000.0):
    store.insert("daily_bars", [
        {"symbol": symbol, "date": d, "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0, "volume": volume, "source": "test"}
        for d in dates], returning=False)


def test_pit_breadth_counts_only_nonprotected_inside_window(store):
    from agent import lab

    d = lab.SPLIT_DATE - timedelta(days=2)
    seed_bars(store, "SPY", [d])
    seed_bars(store, "QQQ", [d])
    assert lab._pit_breadth(lab.SPLIT_DATE, need=5) == 0  # keeps ≠ breadth

    seed_bars(store, "AAPL", [d])
    seed_bars(store, "MSFT", [lab.SPLIT_DATE])                 # inclusive edge
    seed_bars(store, "OLD", [lab.SPLIT_DATE - timedelta(days=30)])  # too old
    seed_bars(store, "NEW", [lab.SPLIT_DATE + timedelta(days=1)])   # the future
    assert lab._pit_breadth(lab.SPLIT_DATE, need=5) == 2


def test_prod_shape_resolves_present_day_not_as_of_2018(store):
    """THE H1 regression: production's data shape is ~10 protected ETFs with
    deep history (2018 bars included) plus a ~400-day hot set for everything
    else. The old SPY-anchored probe passed on that shape every night, the
    PIT "ranking" returned exactly those 10 deep ETFs, and 10 >= max(20//2,
    5) labeled an ETF-only top20 sweep with the trusted as_of basis — rows
    the leaderboard dedup then let replace legitimate present-day results.
    On this shape every universe must resolve present-day and say so."""
    from edgefinder.data.barstore import DB_PROTECTED_ETFS

    from agent import lab

    near_split = [lab.SPLIT_DATE - timedelta(days=k) for k in (1, 4, 7)]
    recent = [date.today() - timedelta(days=k) for k in (1, 2, 3)]
    for sym in DB_PROTECTED_ETFS:          # deep keeps: 2018 AND recent bars
        seed_bars(store, sym, near_split + recent, volume=5e6)
    for i in range(30):                    # the hot set: recent bars ONLY
        seed_bars(store, f"HOT{i:02d}", recent, volume=1e8 - i * 1e5)

    syms, basis = lab._resolve_universe("top20")
    assert basis == "present_day_survivorship_inflated"
    assert basis != f"as_of_{lab.SPLIT_DATE}"
    assert sum(1 for s in syms if s.startswith("HOT")) == 20  # present-day rank


def test_resolve_universe_rejects_protected_only_pit_set(monkeypatch):
    """Belt two: even with the breadth probe forced green, a resolved set
    that is (mostly) protected ETFs is the hot set echoing its keep-list,
    not a 2018 universe — exactly 10 ETFs >= the old 10-name top20 floor WAS
    the collision. It must fall back and carry the survivorship label."""
    import agent.data as agent_data
    from edgefinder.data.barstore import DB_PROTECTED_ETFS

    from agent import lab

    monkeypatch.setattr(lab, "_pit_breadth", lambda as_of, *, need: need)

    def etf_only(n, *, as_of=None):
        if as_of is not None:
            return list(DB_PROTECTED_ETFS)   # the collision's exact output
        return [f"S{i}" for i in range(n)]

    monkeypatch.setattr(agent_data, "universe", etf_only)
    _, basis = lab._resolve_universe("top20")
    assert basis == "present_day_survivorship_inflated"


# ── M2: the TR benchmark's dividend coverage must be visible, not silent ──


def test_sweep_stamps_benchmark_div_coverage(store, monkeypatch):
    import agent.backtest_tool as backtest_tool
    import agent.data as agent_data
    from agent import lab

    store.insert("dividends", {"symbol": "SPY", "ex_date": date(2012, 3, 16),
                               "cash_amount": 0.65}, returning=False)
    monkeypatch.setattr(
        agent_data, "universe",
        lambda n, *, as_of=None: [] if as_of else [f"S{i}" for i in range(n)])
    frame = pd.DataFrame({
        "date": [date(2006, 1, 2) + timedelta(days=i) for i in range(220)],
        "open": [100.0] * 220, "high": [100.0] * 220, "low": [100.0] * 220,
        "close": [100.0] * 220, "volume": [1e6] * 220})
    monkeypatch.setattr(agent_data, "load_bars",
                        lambda syms, **kw: {s: frame.copy() for s in syms})
    monkeypatch.setattr(agent_data, "spy_series_df",
                        lambda **kw: frame.copy())
    monkeypatch.setattr(backtest_tool, "run_prepared",
                        lambda *a, **kw: {"excess_return_pct": 5.0,
                                          "sharpe": 1.0,
                                          "max_drawdown_pct": -10.0})

    res = lab.sweep(max_combos=1, offset=0)
    assert res["benchmark_div_from"] == "2012-03-16"  # queried once per sweep
    assert res["top"][0]["benchmark_div_from"] == "2012-03-16"
    rows = [r for r in store.select("desk_backtests")
            if str(r.get("label") or "").startswith("lab:")]
    assert rows[0]["result"]["benchmark_div_from"] == "2012-03-16"


def test_leaderboard_flags_thin_benchmark_dividend_coverage(store):
    from agent.lab import IN_SAMPLE_START, leaderboard

    # SPY dividend rows starting AFTER the in-sample start mean the in-sample
    # "TR" benchmark was effectively price-only — the honesty string says so.
    assert str(IN_SAMPLE_START) < "2012-03-16"
    seed_lab_row(store, "momentum:5", "top40", "monthly", qualifies=True,
                 score=6.0, extra={"benchmark_div_from": "2012-03-16"})
    b = leaderboard(top=5)
    assert b["top"][0]["benchmark_div_from"] == "2012-03-16"
    assert "dividend coverage" in b["honesty"]


def test_leaderboard_flags_null_benchmark_coverage(store):
    from agent.lab import leaderboard

    # stamped None = the dividends table held NO SPY rows at all: fully thin
    seed_lab_row(store, "momentum:5", "top40", "monthly", qualifies=True,
                 score=6.0, extra={"benchmark_div_from": None})
    assert "dividend coverage" in leaderboard(top=5)["honesty"]


def test_leaderboard_quiet_on_deep_coverage_and_legacy_rows(store):
    from agent.lab import leaderboard

    seed_lab_row(store, "momentum:5", "top40", "monthly", qualifies=True,
                 score=6.0, extra={"benchmark_div_from": "2004-01-05"})
    seed_lab_row(store, "breakout:5", "top20", "weekly", qualifies=True,
                 score=3.0)   # legacy row: coverage unknown, never flagged
    b = leaderboard(top=5)
    assert "dividend coverage" not in b["honesty"]
    by_rule = {e["rule"]: e for e in b["top"]}
    assert "benchmark_div_from" not in by_rule["breakout:5"]  # honest absence
