"""Tests for the real-money order-ticket + reconciliation engine."""

from datetime import date, datetime, timedelta, timezone

import pytest

from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import (
    DailyBar,
    PromotedStrategy,
    StrategyAccount,
    SystemHeartbeat,
)
from edgefinder.engine.live_ticket import (
    MIN_NOTIONAL,
    StaleDataError,
    assert_data_fresh,
    dry_run_weights,
    propose_orders,
    reconcile,
)


# ── propose_orders (pure) ───────────────────────────────────


class TestProposeOrders:
    def test_flat_account_buys_the_whole_basket(self):
        # fresh $2,000 account, fractional: buy both names to target weight
        ticket = propose_orders(
            "book", {"SPY": 0.6, "QQQ": 0.4}, {}, 2000.0,
            {"SPY": 500.0, "QQQ": 400.0})
        assert ticket.equity == 2000.0
        sides = {l.symbol: l for l in ticket.lines}
        assert sides["SPY"].side == "BUY" and sides["QQQ"].side == "BUY"
        assert sides["SPY"].notional == pytest.approx(1200.0)
        assert sides["QQQ"].notional == pytest.approx(800.0)
        # fractional shares: a $2k book holds a $500 ETF
        assert sides["SPY"].shares == pytest.approx(1200.0 / 500.0)
        assert ticket.cash_after == pytest.approx(0.0)

    def test_fractional_lets_small_account_hold_expensive_etf(self):
        # whole-share mode CANNOT buy a $739 ETF with $700; fractional can
        whole = propose_orders("b", {"SPY": 1.0}, {}, 700.0, {"SPY": 739.0},
                               fractional=False)
        assert whole.is_noop or all(l.shares < 1 for l in whole.lines) or \
            sum(l.notional for l in whole.lines) == 0
        frac = propose_orders("b", {"SPY": 1.0}, {}, 700.0, {"SPY": 739.0})
        assert frac.lines and frac.lines[0].notional == pytest.approx(700.0)

    def test_noop_when_already_at_target(self):
        # holding exactly 60/40 of a $1000 book → nothing to do
        ticket = propose_orders(
            "book", {"SPY": 0.6, "QQQ": 0.4},
            {"SPY": 6.0, "QQQ": 8.0}, 0.0, {"SPY": 100.0, "QQQ": 50.0})
        assert ticket.is_noop

    def test_sells_before_buys_and_funds_the_buys(self):
        # holding all SPY, target all QQQ → sell SPY, then buy QQQ from proceeds
        ticket = propose_orders(
            "book", {"QQQ": 1.0}, {"SPY": 10.0}, 0.0,
            {"SPY": 100.0, "QQQ": 50.0})
        assert [l.side for l in ticket.lines] == ["SELL", "BUY"]
        assert ticket.lines[0].symbol == "SPY"
        assert ticket.lines[1].symbol == "QQQ"
        # $1000 of SPY sold funds $1000 of QQQ
        assert ticket.lines[1].notional == pytest.approx(1000.0)

    def test_full_exit_sells_entire_position(self):
        ticket = propose_orders(
            "book", {"SPY": 1.0}, {"SPY": 5.0, "QQQ": 4.0}, 0.0,
            {"SPY": 100.0, "QQQ": 50.0})
        sell = [l for l in ticket.lines if l.symbol == "QQQ"][0]
        assert sell.side == "SELL"
        assert sell.shares == pytest.approx(4.0)
        assert sell.target_shares == pytest.approx(0.0)

    def test_rebalance_band_skips_dust(self):
        # 0.5% drift on a $10k book is under the 1% band → skipped
        ticket = propose_orders(
            "book", {"SPY": 0.5, "QQQ": 0.5},
            {"SPY": 49.5, "QQQ": 50.0}, 50.0, {"SPY": 100.0, "QQQ": 100.0})
        assert ticket.is_noop

    def test_weights_over_one_are_normalized(self):
        ticket = propose_orders(
            "book", {"SPY": 0.8, "QQQ": 0.8}, {}, 1000.0,
            {"SPY": 100.0, "QQQ": 100.0})
        # normalized to 0.5/0.5 → never levered past cash
        assert sum(l.notional for l in ticket.lines) == pytest.approx(1000.0)

    def test_missing_price_is_warned_not_traded(self):
        ticket = propose_orders(
            "book", {"SPY": 0.5, "QQQ": 0.5}, {}, 1000.0, {"SPY": 100.0})
        assert any("QQQ" in w for w in ticket.warnings)
        assert {l.symbol for l in ticket.lines} == {"SPY"}

    def test_min_notional_floor(self):
        # a $0.50 re-true is below the $1 fractional floor → no order
        ticket = propose_orders(
            "book", {"SPY": 1.0}, {"SPY": 9.995}, 0.5, {"SPY": 100.0})
        assert all(l.notional >= MIN_NOTIONAL for l in ticket.lines)


# ── reconcile (pure) ────────────────────────────────────────


class TestReconcile:
    def test_clean_when_matching(self):
        r = reconcile({"SPY": 5.0, "QQQ": 3.0}, {"SPY": 5.0, "QQQ": 3.0},
                      db_cash=100.0, broker_cash=100.0)
        assert r.clean
        assert r.position_diffs == []

    def test_position_mismatch_detected(self):
        r = reconcile({"SPY": 5.0}, {"SPY": 4.0})
        assert not r.clean
        assert r.position_diffs[0].symbol == "SPY"
        assert r.position_diffs[0].delta == pytest.approx(-1.0)

    def test_missing_position_on_either_side(self):
        r = reconcile({"SPY": 5.0}, {"SPY": 5.0, "QQQ": 2.0})
        assert not r.clean
        assert {d.symbol for d in r.position_diffs} == {"QQQ"}

    def test_cash_mismatch_breaks_clean(self):
        r = reconcile({"SPY": 5.0}, {"SPY": 5.0},
                      db_cash=100.0, broker_cash=150.0)
        assert not r.clean
        assert r.cash_diff == pytest.approx(50.0)

    def test_tolerance_absorbs_float_noise(self):
        r = reconcile({"SPY": 5.00000001}, {"SPY": 5.0},
                      db_cash=100.001, broker_cash=100.0)
        assert r.clean


# ── DB-backed: freshness gate, pause, dry-run weights ───────


@pytest.fixture()
def factory():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return get_session_factory(engine)


def _seed_bars(session, symbol, start, days, price=100.0):
    d, added, last = start, 0, start
    while added < days:
        if d.weekday() < 5:
            session.add(DailyBar(symbol=symbol, date=d, open=price, high=price,
                                 low=price, close=price, volume=1e6, source="test"))
            added += 1
            last = d
        d += timedelta(days=1)
    session.commit()
    return last


class TestFreshnessGate:
    def test_fresh_data_passes(self, factory):
        session = factory()
        last = _seed_bars(session, "SPY", date(2026, 1, 1), 60)
        assert assert_data_fresh(session, today=last + timedelta(days=1)) <= 4

    def test_stale_data_refused(self, factory):
        session = factory()
        last = _seed_bars(session, "SPY", date(2026, 1, 1), 60)
        with pytest.raises(StaleDataError):
            assert_data_fresh(session, today=last + timedelta(days=30))

    def test_no_data_fails_closed(self, factory):
        session = factory()
        with pytest.raises(StaleDataError):
            assert_data_fresh(session, today=date(2026, 6, 15))


class TestPauseAndWeights:
    def _setup(self, factory, paused=False):
        session = factory()
        _seed_bars(session, "AAA", date(2025, 1, 1), 260, price=100.0)
        _seed_bars(session, "BBB", date(2025, 1, 1), 260, price=50.0)
        session.add(PromotedStrategy(
            strategy_name="equal_weight", spec="equal_weight",
            symbols=["AAA", "BBB"], schedule="monthly", active=True))
        if paused:
            session.add(StrategyAccount(
                strategy_name="equal_weight", starting_capital=100000.0,
                cash_balance=100000.0, total_equity=100000.0,
                peak_equity=100000.0, is_paused=True))
        session.commit()
        last = (session.query(DailyBar.date)
                .order_by(DailyBar.date.desc()).first())[0]
        session.close()
        return last

    def _price_fn(self, factory):
        def fn(sym):
            s = factory()
            try:
                row = (s.query(DailyBar.close).filter(DailyBar.symbol == sym)
                       .order_by(DailyBar.date.desc()).first())
                return float(row[0]) if row else None
            finally:
                s.close()
        return fn

    def test_dry_run_weights_returns_targets(self, factory):
        last = self._setup(factory)
        weights = dry_run_weights(
            factory, "equal_weight", price_fn=self._price_fn(factory),
            today=last + timedelta(days=1))
        assert set(weights) == {"AAA", "BBB"}
        assert weights["AAA"] == pytest.approx(0.5)

    def test_paused_book_yields_no_ticket(self, factory):
        last = self._setup(factory, paused=True)
        with pytest.raises(ValueError, match="no target weights"):
            dry_run_weights(
                factory, "equal_weight", price_fn=self._price_fn(factory),
                today=last + timedelta(days=1))
