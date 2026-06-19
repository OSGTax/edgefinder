"""Tests for the research-agent account: scorers, selection, decision, seam."""

from datetime import date, timedelta

import pandas as pd
import pytest

from edgefinder.agents.analyst import (
    Candidate,
    build_decision,
    run_analyst,
    score_breakout,
    score_pullback,
    score_rs_momentum,
    score_trend,
    screen,
    select_with_hysteresis,
)
from edgefinder.data.market_data import IndicatorSnapshot
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import AgentDecision, DailyBar
from edgefinder.engine.analyst_strategy import AnalystStrategy
from edgefinder.engine.strategy import AssetView, RebalanceContext


def _asset(symbol, closes, *, ema_200=None, ema_50=None, rsi=None, price=None):
    hist = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1e6] * len(closes),
    })
    ind = IndicatorSnapshot(ema_50=ema_50, ema_200=ema_200, rsi=rsi)
    return AssetView(symbol=symbol, price=price if price is not None else closes[-1],
                     indicators=ind, history=hist)


# ── entry rules ──────────────────────────────────────────────────────────


class TestEntryRules:
    def test_trend_fires_above_rising_200ema(self):
        a = _asset("X", [100.0] * 260, ema_200=90.0, ema_50=95.0, price=110.0)
        hit = score_trend(a)
        assert hit and hit.rule == "trend" and 0 < hit.score <= 1

    def test_trend_silent_below_200ema(self):
        a = _asset("X", [100.0] * 260, ema_200=120.0, price=110.0)
        assert score_trend(a) is None

    def test_trend_silent_when_structure_broken(self):
        # price above 200-EMA but 50-EMA below 200-EMA = not a clean uptrend
        a = _asset("X", [100.0] * 260, ema_200=100.0, ema_50=95.0, price=110.0)
        assert score_trend(a) is None

    def test_breakout_fires_near_52w_high(self):
        closes = [50.0] * 259 + [100.0]      # at the high
        a = _asset("X", closes, price=100.0)
        hit = score_breakout(a)
        assert hit and hit.rule == "breakout"

    def test_breakout_silent_far_from_high(self):
        closes = [100.0] * 259 + [70.0]      # 30% below the high
        a = _asset("X", closes, price=70.0)
        assert score_breakout(a) is None

    def test_breakout_needs_enough_history(self):
        a = _asset("X", [100.0] * 50, price=100.0)   # < 200 bars
        assert score_breakout(a) is None

    def test_pullback_fires_on_dip_in_uptrend(self):
        a = _asset("X", [100.0] * 260, ema_200=90.0, rsi=40.0, price=100.0)
        hit = score_pullback(a)
        assert hit and hit.rule == "pullback"

    def test_pullback_silent_when_not_oversold(self):
        a = _asset("X", [100.0] * 260, ema_200=90.0, rsi=65.0, price=100.0)
        assert score_pullback(a) is None

    def test_rs_momentum_scales_with_return(self):
        rising = [50.0 + i * 0.5 for i in range(260)]   # strong uptrend
        a = _asset("X", rising)
        hit = score_rs_momentum(a)
        assert hit and hit.score > 0

    def test_rs_momentum_silent_when_negative(self):
        falling = [200.0 - i * 0.5 for i in range(260)]
        a = _asset("X", falling)
        assert score_rs_momentum(a) is None


class TestScreen:
    def test_ranks_by_composite_and_drops_non_firing(self):
        strong = _asset("STRONG", [50.0 + i * 0.5 for i in range(260)],
                        ema_200=90.0, ema_50=120.0, rsi=45.0)
        # clear downtrend: below 200-EMA, far from its high, negative momentum
        weak = _asset("WEAK", [200.0 - i * 0.3 for i in range(260)], ema_200=180.0)
        ctx = RebalanceContext(date=date(2026, 6, 12),
                               assets={"STRONG": strong, "WEAK": weak})
        cands = screen(ctx)
        assert [c.symbol for c in cands] == ["STRONG"]
        assert cands[0].composite > 0 and cands[0].hits


# ── hysteresis selection ──────────────────────────────────────────────────


class TestHysteresis:
    def test_new_book_fills_to_cap_from_enter_band(self):
        ranked = [f"S{i}" for i in range(30)]
        book = select_with_hysteresis(ranked, [], cap=5, enter_rank=10, exit_rank=20)
        assert book == ["S0", "S1", "S2", "S3", "S4"]

    def test_held_name_in_buffer_is_kept_not_churned(self):
        ranked = [f"S{i}" for i in range(30)]
        # S15 is held and sits in the buffer (enter=10 <= 15 < exit=20) → kept
        book = select_with_hysteresis(ranked, ["S15"], cap=5,
                                      enter_rank=10, exit_rank=20)
        assert "S15" in book

    def test_held_name_past_exit_band_is_dropped(self):
        ranked = [f"S{i}" for i in range(30)]
        book = select_with_hysteresis(ranked, ["S25"], cap=5,
                                      enter_rank=10, exit_rank=20)
        assert "S25" not in book

    def test_held_name_no_longer_ranked_is_dropped(self):
        book = select_with_hysteresis(["A", "B"], ["GONE"], cap=5,
                                      enter_rank=10, exit_rank=20)
        assert "GONE" not in book and book == ["A", "B"]


# ── decision assembly ──────────────────────────────────────────────────────


class TestBuildDecision:
    def _cands(self, syms):
        from edgefinder.agents.analyst import RuleHit
        return [Candidate(s, 1.0, [RuleHit("trend", 1.0, "up")], {}) for s in syms]

    def test_equal_weight_and_actions(self):
        target, picks = build_decision(
            ["A", "B"], self._cands(["A", "B"]),
            holdings={"A": 0.5}, news={})
        assert target == {"A": 0.5, "B": 0.5}
        by = {p["symbol"]: p for p in picks}
        assert by["A"]["action"] == "hold"   # already ~at target
        assert by["B"]["action"] == "buy"    # new position

    def test_exited_holding_listed_as_sell(self):
        target, picks = build_decision(
            ["A"], self._cands(["A"]), holdings={"A": 1.0, "OLD": 0.4}, news={})
        sells = [p for p in picks if p["action"] == "sell"]
        assert [p["symbol"] for p in sells] == ["OLD"]
        assert "OLD" not in target


# ── the strategy seam (AnalystStrategy) ───────────────────────────────────


@pytest.fixture()
def factory():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return get_session_factory(engine)


def _ctx(d, symbols, holdings=None):
    assets = {s: _asset(s, [100.0] * 5) for s in symbols}
    return RebalanceContext(date=d, assets=assets, holdings=holdings or {})


class TestAnalystStrategy:
    def test_reads_persisted_target(self, factory):
        s = factory()
        s.add(AgentDecision(strategy_name="ai_analyst", decision_date=date(2026, 6, 12),
                            target_weights={"AAA": 0.5, "BBB": 0.5}, picks=[]))
        s.commit()
        s.close()
        strat = AnalystStrategy(session_factory=factory)
        out = strat.rebalance(_ctx(date(2026, 6, 12), ["AAA", "BBB"]))
        assert out == {"AAA": 0.5, "BBB": 0.5}

    def test_drops_untradable_names(self, factory):
        s = factory()
        s.add(AgentDecision(strategy_name="ai_analyst", decision_date=date(2026, 6, 12),
                            target_weights={"AAA": 0.5, "GONE": 0.5}, picks=[]))
        s.commit()
        s.close()
        strat = AnalystStrategy(session_factory=factory)
        out = strat.rebalance(_ctx(date(2026, 6, 12), ["AAA"]))  # GONE not in ctx
        assert out == {"AAA": 0.5}

    def test_holds_current_book_when_no_decision(self, factory):
        strat = AnalystStrategy(session_factory=factory)
        out = strat.rebalance(_ctx(date(2026, 6, 12), ["AAA"], holdings={"AAA": 1.0}))
        assert out == {"AAA": 1.0}

    def test_stale_decision_holds_current_book(self, factory):
        s = factory()
        s.add(AgentDecision(strategy_name="ai_analyst", decision_date=date(2026, 1, 1),
                            target_weights={"AAA": 1.0}, picks=[]))
        s.commit()
        s.close()
        strat = AnalystStrategy(session_factory=factory)
        # ctx.date is months after the only decision → stale → hold
        out = strat.rebalance(_ctx(date(2026, 6, 12), ["AAA"], holdings={"BBB": 1.0}))
        assert out == {"BBB": 1.0}


# ── end-to-end orchestration ──────────────────────────────────────────────


def _seed_bars(session, symbol, start, days, price_fn):
    d, added = start, 0
    while added < days:
        if d.weekday() < 5:
            px = price_fn(added)
            session.add(DailyBar(symbol=symbol, date=d, open=px, high=px,
                                 low=px, close=px, volume=1e6, source="test"))
            added += 1
        d += timedelta(days=1)
    session.commit()


class TestRunAnalystEndToEnd:
    def test_produces_and_persists_a_decision(self, factory):
        s = factory()
        # two strong uptrends + one clear downtrend
        _seed_bars(s, "UP1", date(2025, 1, 1), 260, lambda i: 50.0 + i * 0.4)
        _seed_bars(s, "UP2", date(2025, 1, 1), 260, lambda i: 40.0 + i * 0.5)
        _seed_bars(s, "DOWN", date(2025, 1, 1), 260, lambda i: 200.0 - i * 0.3)
        s.close()

        sc = factory()
        last = sc.query(DailyBar.date).order_by(DailyBar.date.desc()).first()[0]
        sc.close()
        rid = run_analyst(factory, symbols=["UP1", "UP2", "DOWN"],
                          today=last + timedelta(days=1), cap=5)
        assert rid is not None

        s2 = factory()
        row = s2.query(AgentDecision).filter_by(strategy_name="ai_analyst").one()
        # the two uptrends are picked; the downtrend is not
        assert set(row.target_weights) == {"UP1", "UP2"}
        assert abs(sum(row.target_weights.values()) - 1.0) < 1e-6
        assert row.picks and all(p["symbol"] in {"UP1", "UP2"} for p in row.picks)
        assert row.summary
        s2.close()

    def test_no_bars_returns_none(self, factory):
        assert run_analyst(factory, symbols=["NOPE"],
                           today=date(2026, 6, 12)) is None

    def test_picks_carry_backtest_proof(self, factory):
        s = factory()
        _seed_bars(s, "UP1", date(2025, 1, 1), 260, lambda i: 50.0 + i * 0.4)
        _seed_bars(s, "UP2", date(2025, 1, 1), 260, lambda i: 40.0 + i * 0.5)
        # SPY benchmark so the rule backtest can compute excess vs SPY
        _seed_bars(s, "SPY", date(2025, 1, 1), 260, lambda i: 400.0 + i * 0.1)
        s.close()
        sc = factory()
        last = sc.query(DailyBar.date).order_by(DailyBar.date.desc()).first()[0]
        sc.close()
        rid = run_analyst(factory, symbols=["UP1", "UP2", "SPY"],
                          today=last + timedelta(days=1), cap=5, with_proof=True)
        assert rid is not None
        s2 = factory()
        row = s2.query(AgentDecision).filter_by(strategy_name="ai_analyst").one()
        # every pick carries a proof dict for each firing rule with stats
        assert row.picks
        pick = next(p for p in row.picks if p["symbol"] in ("UP1", "UP2"))
        assert pick["proof"], "pick should carry rule track records"
        any_rule = next(iter(pick["proof"].values()))
        assert "return_pct" in any_rule and "sharpe" in any_rule
        s2.close()
