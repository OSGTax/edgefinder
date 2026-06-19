"""Tests for the v2 promotion pipeline + live portfolio paper-trading runner."""

from datetime import date, timedelta

import pytest

from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import (
    DailyBar,
    DividendCredit,
    DividendRecord,
    PromotedStrategy,
    StrategyAccount,
    SystemHeartbeat,
    TradeRecord,
    ValidationRun,
)
from edgefinder.engine.live import STARTING_CAPITAL, run_portfolio_cycle
from edgefinder.engine.promote import demote, is_validated, promote


@pytest.fixture()
def factory():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return get_session_factory(engine)


def _seed_bars(session, symbol: str, start: date, days: int,
               price: float = 100.0, drift: float = 0.0) -> date:
    """Insert ``days`` daily bars; returns the last bar date."""
    d = start
    added = 0
    while added < days:
        if d.weekday() < 5:               # weekdays only, like real bars
            px = price + added * drift
            session.add(DailyBar(symbol=symbol, date=d, open=px, high=px,
                                 low=px, close=px, volume=1e6, source="test"))
            added += 1
            last = d
        d += timedelta(days=1)
    session.commit()
    return last


def _prices(session_factory):
    """price_fn stub: a symbol's most recent close in daily_bars."""
    def price_fn(symbol: str):
        s = session_factory()
        try:
            row = (s.query(DailyBar.close).filter(DailyBar.symbol == symbol)
                   .order_by(DailyBar.date.desc()).first())
            return float(row[0]) if row else None
        finally:
            s.close()
    return price_fn


class TestPromotion:
    def test_validated_tier_requires_passing_validation(self, factory):
        session = factory()
        with pytest.raises(ValueError, match="not validated"):
            promote(session, spec="equal_weight", symbols=["AAA"],
                    schedule="monthly", tier="validated")

    def test_validated_tier_accepts_passing_run(self, factory):
        session = factory()
        session.add(ValidationRun(
            strategy_name="equal_weight", universe="test",
            criteria={"all_met": True}, holdout={"passes": True},
            verdict="PASS"))
        session.commit()
        row = promote(session, spec="equal_weight", symbols=["AAA"],
                      schedule="monthly", tier="validated")
        assert row.tier == "validated"
        assert row.validation_run_id is not None

    def test_experimental_tier_always_allowed_and_demotable(self, factory):
        session = factory()
        row = promote(session, spec="equal_weight", symbols=["AAA", "BBB"],
                      schedule="monthly", tier="experimental")
        assert row.active is True
        assert demote(session, "equal_weight") is True
        assert (session.query(PromotedStrategy).one().active is False)

    def test_is_validated_rule(self):
        assert not is_validated(None)
        run = ValidationRun(strategy_name="x", criteria={"all_met": True},
                            holdout=None, verdict="PASS")
        assert not is_validated(run)          # sealed holdout != validated
        run.holdout = {"passes": True}
        assert is_validated(run)


class TestLiveCycle:
    def _setup(self, factory, schedule="monthly"):
        session = factory()
        # bars end Wed 2024-01-31; first cycle runs Thu 2024-02-01
        _seed_bars(session, "AAA", date(2023, 1, 2), 260, price=100.0)
        _seed_bars(session, "BBB", date(2023, 1, 2), 260, price=50.0)
        promote(session, spec="equal_weight", symbols=["AAA", "BBB"],
                schedule=schedule, tier="experimental")
        session.close()
        return _prices(factory)

    def test_implausible_quote_is_refused(self, factory):
        # A stale/garbage quote (>2x the last close) must NOT be filled — the
        # name is skipped and a fill_price observation is recorded (the
        # 2026-06-11 stale-fill class).
        from edgefinder.db.models import AgentObservation
        session = factory()
        _seed_bars(session, "AAA", date(2023, 1, 2), 260, price=100.0)
        _seed_bars(session, "BBB", date(2023, 1, 2), 260, price=50.0)
        promote(session, spec="equal_weight", symbols=["AAA", "BBB"],
                schedule="monthly", tier="experimental")
        session.close()

        def bad_price(sym):
            return 100.0 if sym == "AAA" else 500.0   # BBB = 10x its ~50 close

        run_portfolio_cycle(factory, today=date(2024, 2, 1), price_fn=bad_price)

        session = factory()
        syms = {t.symbol for t in session.query(TradeRecord)
                .filter_by(strategy_name="equal_weight", status="OPEN").all()}
        assert "AAA" in syms and "BBB" not in syms     # BBB refused
        obs = session.query(AgentObservation).filter_by(category="fill_price").all()
        assert len(obs) >= 1
        session.close()

    def test_first_cycle_opens_positions_and_account(self, factory):
        price_fn = self._setup(factory)
        summary = run_portfolio_cycle(
            factory, today=date(2024, 2, 1), price_fn=price_fn)
        res = summary["strategies"]["equal_weight"]
        assert res["action"] == "rebalance"
        assert {t["symbol"] for t in res["trades"]} == {"AAA", "BBB"}

        session = factory()
        lots = (session.query(TradeRecord)
                .filter_by(strategy_name="equal_weight", status="OPEN").all())
        assert len(lots) == 2
        # ~50/50 split, sized off starting capital (small slippage tolerance)
        shares = {t.symbol: t.shares for t in lots}
        half = STARTING_CAPITAL / 2
        assert shares["AAA"] == pytest.approx(half / 100.0, rel=0.02)
        assert shares["BBB"] == pytest.approx(half / 50.0, rel=0.02)

        acct = session.query(StrategyAccount).filter_by(
            strategy_name="equal_weight").one()
        # integrity: cash = start - open cost basis (no closed trades yet)
        open_cost = sum(t.entry_price * t.shares for t in lots)
        assert acct.cash_balance == pytest.approx(
            STARTING_CAPITAL - open_cost, abs=0.01)
        assert acct.total_equity == pytest.approx(
            acct.cash_balance + acct.open_positions_value, abs=0.01)

        hb = session.query(SystemHeartbeat).filter_by(
            component="v2_portfolio_cycle").one()
        assert hb.ok is True
        session.close()

    def test_non_boundary_day_holds(self, factory):
        price_fn = self._setup(factory)
        run_portfolio_cycle(factory, today=date(2024, 2, 1), price_fn=price_fn)
        # next trading day arrives with a fresh bar — same month -> hold
        session = factory()
        _seed_bars(session, "AAA", date(2024, 2, 1), 1, price=100.0)
        _seed_bars(session, "BBB", date(2024, 2, 1), 1, price=50.0)
        session.close()
        summary = run_portfolio_cycle(
            factory, today=date(2024, 2, 2), price_fn=price_fn)
        assert summary["strategies"]["equal_weight"]["action"] == "hold"

    def test_rebalance_sells_split_lots_and_keep_integrity(self, factory):
        price_fn = self._setup(factory)
        run_portfolio_cycle(factory, today=date(2024, 2, 1), price_fn=price_fn)
        # AAA doubles by the March boundary -> equal-weight must sell ~half
        session = factory()
        _seed_bars(session, "AAA", date(2024, 2, 1), 20, price=200.0)
        _seed_bars(session, "BBB", date(2024, 2, 1), 20, price=50.0)
        session.close()
        summary = run_portfolio_cycle(
            factory, today=date(2024, 3, 1), price_fn=price_fn)
        res = summary["strategies"]["equal_weight"]
        sides = {t["side"] for t in res["trades"]}
        assert "SELL" in sides and "BUY" in sides

        session = factory()
        closed = (session.query(TradeRecord)
                  .filter_by(strategy_name="equal_weight", status="CLOSED").all())
        assert closed and all(t.exit_reason == "REBALANCE" for t in closed)
        assert all(t.pnl_dollars > 0 for t in closed)     # sold AAA at 2x
        # the split remainder reopened at the ORIGINAL entry price
        opens = (session.query(TradeRecord)
                 .filter_by(strategy_name="equal_weight", status="OPEN",
                            symbol="AAA").all())
        assert len(opens) == 1
        assert opens[0].entry_price == closed[0].entry_price
        assert "rebalance split" in (opens[0].entry_reasoning or "")
        # integrity formula across the whole history
        open_cost = sum(t.entry_price * t.shares
                        for t in session.query(TradeRecord)
                        .filter_by(strategy_name="equal_weight", status="OPEN"))
        realized = sum(t.pnl_dollars or 0 for t in closed)
        acct = session.query(StrategyAccount).filter_by(
            strategy_name="equal_weight").one()
        assert acct.cash_balance == pytest.approx(
            STARTING_CAPITAL + realized - open_cost, abs=0.01)
        session.close()

    def test_dry_run_persists_nothing(self, factory):
        price_fn = self._setup(factory)
        summary = run_portfolio_cycle(
            factory, today=date(2024, 2, 1), price_fn=price_fn, dry_run=True)
        assert summary["strategies"]["equal_weight"]["trades"]
        session = factory()
        assert session.query(TradeRecord).count() == 0
        assert session.query(StrategyAccount).count() == 0
        session.close()

    def test_no_promotions_is_clean_noop(self, factory):
        summary = run_portfolio_cycle(
            factory, today=date(2024, 2, 1), price_fn=lambda s: 100.0)
        assert summary["note"] == "no active promoted strategies"


class TestStatefulHoldings:
    """Live parity for the stateful interface (v5.57): the runner threads the
    current book into ctx.holdings, so a stateful strategy that decides to HOLD
    its existing positions produces NO churn across cycles."""

    seen_holdings: list = []

    def _setup(self, factory, monkeypatch):
        session = factory()
        _seed_bars(session, "AAA", date(2023, 1, 2), 260, price=100.0)
        _seed_bars(session, "BBB", date(2023, 1, 2), 260, price=50.0)
        promote(session, spec="equal_weight", symbols=["AAA", "BBB"],
                schedule="daily", tier="experimental")
        session.close()

        TestStatefulHoldings.seen_holdings = []

        class HoldWhatYouHave:
            """Stateful: on a flat book, equal-weight; once holding, KEEP the
            exact current weights (reads ctx.holdings -> zero churn)."""

            name = "equal_weight"

            def rebalance(self, ctx):
                TestStatefulHoldings.seen_holdings.append(dict(ctx.holdings))
                if ctx.holdings:
                    return dict(ctx.holdings)        # hold the existing book
                syms = ctx.symbols()
                return {s: 1.0 / len(syms) for s in syms} if syms else {}

        import edgefinder.engine.live as live_mod
        monkeypatch.setattr(live_mod, "make_strategy_factory",
                            lambda spec: HoldWhatYouHave)
        return _prices(factory)

    def test_stateful_strategy_holds_across_cycles_no_churn(self, factory,
                                                            monkeypatch):
        price_fn = self._setup(factory, monkeypatch)
        # first cycle opens the book (flat -> equal-weight)
        first = run_portfolio_cycle(
            factory, today=date(2024, 2, 1), price_fn=price_fn)
        assert first["strategies"]["equal_weight"]["action"] == "rebalance"
        assert TestStatefulHoldings.seen_holdings[0] == {}     # opened flat

        # next trading day, daily schedule, SAME prices -> strategy sees its
        # book in ctx.holdings and holds: no SELL/BUY churn
        session = factory()
        _seed_bars(session, "AAA", date(2024, 2, 1), 1, price=100.0)
        _seed_bars(session, "BBB", date(2024, 2, 1), 1, price=50.0)
        session.close()
        second = run_portfolio_cycle(
            factory, today=date(2024, 2, 2), price_fn=price_fn)
        res = second["strategies"]["equal_weight"]
        # the second decision SAW a populated book (the live runner threaded it)
        assert any(h for h in TestStatefulHoldings.seen_holdings[1:]), \
            "live runner never threaded holdings into ctx"
        # ...and the strategy held its existing book -> NO churn (no trades)
        assert res["trades"] == []


class TestDividendCredits:
    """Live TR parity: ex-dates crossed while holding credit cash.

    Setup mirrors TestLiveCycle: equal_weight on AAA/BBB, first cycle (and
    lot entries) on Thu 2024-02-01.
    """

    def _setup(self, factory):
        session = factory()
        _seed_bars(session, "AAA", date(2023, 1, 2), 260, price=100.0)
        _seed_bars(session, "BBB", date(2023, 1, 2), 260, price=50.0)
        promote(session, spec="equal_weight", symbols=["AAA", "BBB"],
                schedule="monthly", tier="experimental")
        session.close()
        price_fn = _prices(factory)
        run_portfolio_cycle(factory, today=date(2024, 2, 1), price_fn=price_fn)
        return price_fn

    def _add_dividend(self, factory, symbol, ex_date, amount):
        session = factory()
        session.add(DividendRecord(symbol=symbol, ex_date=ex_date,
                                   cash_amount=amount))
        session.commit()
        session.close()

    def _advance_bars(self, factory, last_seed: date, days: int):
        session = factory()
        _seed_bars(session, "AAA", last_seed, days, price=100.0)
        _seed_bars(session, "BBB", last_seed, days, price=50.0)
        session.close()

    def test_ex_date_while_held_credits_cash(self, factory):
        price_fn = self._setup(factory)
        self._add_dividend(factory, "AAA", date(2024, 2, 2), 1.0)
        self._advance_bars(factory, date(2024, 2, 1), 1)
        # Fri 2024-02-02 = ex-date, a hold day (same month)
        run_portfolio_cycle(factory, today=date(2024, 2, 2), price_fn=price_fn)

        session = factory()
        credit = session.query(DividendCredit).one()
        shares = (session.query(TradeRecord)
                  .filter_by(strategy_name="equal_weight", symbol="AAA",
                             status="OPEN").one().shares)
        assert credit.symbol == "AAA"
        assert credit.shares == shares
        assert credit.amount == pytest.approx(shares * 1.0, abs=0.01)
        # cash includes the credit (extended integrity formula)
        open_cost = sum(t.entry_price * t.shares
                        for t in session.query(TradeRecord)
                        .filter_by(strategy_name="equal_weight", status="OPEN"))
        acct = session.query(StrategyAccount).filter_by(
            strategy_name="equal_weight").one()
        assert acct.cash_balance == pytest.approx(
            STARTING_CAPITAL + credit.amount - open_cost, abs=0.01)
        session.close()

    def test_no_credit_for_entry_on_or_after_ex_date(self, factory):
        price_fn = self._setup(factory)
        # ex-date == entry date: bought ex-dividend, not entitled
        self._add_dividend(factory, "AAA", date(2024, 2, 1), 1.0)
        self._advance_bars(factory, date(2024, 2, 1), 1)
        run_portfolio_cycle(factory, today=date(2024, 2, 2), price_fn=price_fn)
        session = factory()
        assert session.query(DividendCredit).count() == 0
        session.close()

    def test_credit_is_idempotent_across_cycles(self, factory):
        price_fn = self._setup(factory)
        self._add_dividend(factory, "AAA", date(2024, 2, 2), 1.0)
        self._advance_bars(factory, date(2024, 2, 1), 2)
        run_portfolio_cycle(factory, today=date(2024, 2, 2), price_fn=price_fn)
        run_portfolio_cycle(factory, today=date(2024, 2, 5), price_fn=price_fn)
        session = factory()
        assert session.query(DividendCredit).count() == 1
        session.close()

    def test_missed_cycle_ex_date_self_heals(self, factory):
        price_fn = self._setup(factory)
        # ex-date Mon 2024-02-05; no cycle ran that day (outage). The next
        # cycle (Tue) still credits it: shares unchanged since no trades ran.
        self._add_dividend(factory, "BBB", date(2024, 2, 5), 0.5)
        self._advance_bars(factory, date(2024, 2, 1), 3)
        run_portfolio_cycle(factory, today=date(2024, 2, 6), price_fn=price_fn)
        session = factory()
        credit = session.query(DividendCredit).one()
        assert credit.symbol == "BBB"
        assert credit.ex_date == date(2024, 2, 5)
        session.close()

    def test_dry_run_writes_no_credits(self, factory):
        price_fn = self._setup(factory)
        self._add_dividend(factory, "AAA", date(2024, 2, 2), 1.0)
        self._advance_bars(factory, date(2024, 2, 1), 1)
        run_portfolio_cycle(factory, today=date(2024, 2, 2),
                            price_fn=price_fn, dry_run=True)
        session = factory()
        assert session.query(DividendCredit).count() == 0
        session.close()
