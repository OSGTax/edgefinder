"""Cross-sectional universe promotions + the live universe trading path.

The live runner must mirror engine/validate's --universe semantics, with
the RANKING run in SQL over daily_bars (resolve_universe — the hot set
holds the nightly top-1000, a superset of any top-500; the full-manifest
store load OOM'd production on 2026-06-11) and frames loaded from R2 only
for the resolved set + held names (with DB recency top-up), split
adjustment on the traded slice, PIT fundamentals in the context — plus the
live-only pieces: the <90% shrink guard with last-good fallback + CRITICAL
observation, persistence of resolved_symbols, and a cheap hold-day path
that never touches the store.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

import edgefinder.engine.live as live_mod
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import (
    AgentObservation,
    DailyBar,
    FundamentalsSnapshot,
    PromotedStrategy,
    TradeRecord,
    ValidationRun,
)
from edgefinder.engine.data import trailing_rank_start
from edgefinder.engine.live import run_portfolio_cycle
from edgefinder.engine.promote import is_finalist_confirmed, promote


@pytest.fixture()
def factory():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return get_session_factory(engine)


def _seed_db_bars(session, symbol: str, start: date, end: date,
                  price: float = 100.0, volume: float = 1e6) -> None:
    d = start
    while d <= end:
        if d.weekday() < 5:
            session.add(DailyBar(symbol=symbol, date=d, open=price, high=price,
                                 low=price, close=price, volume=volume,
                                 source="test"))
        d += timedelta(days=1)
    session.commit()


def _frame(start: date, end: date, price: float = 100.0,
           volume: float = 1e6) -> pd.DataFrame:
    rows = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            rows.append({"date": d, "open": price, "high": price, "low": price,
                         "close": price, "volume": volume})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


START = date(2023, 1, 2)
JAN_END = date(2024, 1, 31)          # Wed; first cycle runs Thu 2024-02-01

# volumes give a deterministic dollar-volume order: AAA > BBB > ... > FFF;
# SPY is kept tiny so it never crowds into a top:5 resolution
VOLUMES = {"AAA": 10e6, "BBB": 9e6, "CCC": 8e6, "DDD": 7e6,
           "EEE": 6e6, "FFF": 5e6, "SPY": 1e3}


def _market(end: date = JAN_END) -> dict[str, pd.DataFrame]:
    return {s: _frame(START, end, volume=v) for s, v in VOLUMES.items()}


def _stub_store(monkeypatch, frames: dict) -> None:
    # the live loader is TARGETED now — serve only the requested symbols,
    # like the real store reader
    monkeypatch.setattr(
        live_mod, "load_bars_from_store",
        lambda symbols, start=None, end=None: {
            s: frames[s] for s in (symbols or frames) if s in frames})


class TestPromoteUniverse:
    def test_universe_storage_round_trip(self, factory):
        session = factory()
        row = promote(session, spec="equal_weight", universe="top:500",
                      rank_window=90, schedule="monthly", tier="experimental")
        assert row.universe_spec == "top:500"
        assert row.rank_window == 90
        assert row.symbols is None
        assert row.resolved_symbols is None and row.resolved_at is None
        # re-read from a fresh session: JSON/typed columns persist
        session.close()
        session = factory()
        row = session.query(PromotedStrategy).one()
        assert (row.universe_spec, row.rank_window) == ("top:500", 90)

    def test_symbols_and_universe_are_mutually_exclusive(self, factory):
        session = factory()
        with pytest.raises(ValueError, match="exactly one"):
            promote(session, spec="equal_weight", symbols=["SPY"],
                    universe="top:500", schedule="monthly", tier="experimental")
        with pytest.raises(ValueError, match="exactly one"):
            promote(session, spec="equal_weight",
                    schedule="monthly", tier="experimental")

    def test_bad_universe_spec_rejected(self, factory):
        session = factory()
        with pytest.raises(ValueError, match="bad universe spec"):
            promote(session, spec="equal_weight", universe="bottom:500",
                    schedule="monthly", tier="experimental")

    def test_changing_universe_spec_clears_stale_resolution(self, factory):
        session = factory()
        row = promote(session, spec="equal_weight", universe="top:500",
                      schedule="monthly", tier="experimental")
        row.resolved_symbols = ["AAA"]
        row.resolved_at = date(2026, 1, 5)
        session.commit()
        # same spec: the last-good fallback survives a re-promotion
        row = promote(session, spec="equal_weight", universe="top:500",
                      schedule="monthly", tier="experimental")
        assert row.resolved_symbols == ["AAA"]
        # different spec: the old resolution would be a wrong fallback
        row = promote(session, spec="equal_weight", universe="top:200",
                      schedule="monthly", tier="experimental")
        assert row.resolved_symbols is None and row.resolved_at is None


class TestFinalistGate:
    def _run(self, session, holdout, criteria=None) -> None:
        session.add(ValidationRun(
            strategy_name="equal_weight", universe="top:500@pit",
            criteria=criteria or {"all_met": False}, holdout=holdout,
            verdict="FAIL"))
        session.commit()

    def test_refuses_without_any_run(self, factory):
        session = factory()
        with pytest.raises(ValueError, match="finalist standard"):
            promote(session, spec="equal_weight", universe="top:500",
                    schedule="monthly", tier="validated", finalist=True)

    def test_refuses_unevaluated_holdout(self, factory):
        session = factory()
        self._run(session, holdout=None)
        with pytest.raises(ValueError, match="holdout unevaluated"):
            promote(session, spec="equal_weight", universe="top:500",
                    schedule="monthly", tier="validated", finalist=True)

    def test_refuses_negative_or_missing_excess(self, factory):
        session = factory()
        self._run(session, holdout={"excess_vs_spy_pct": -3.1, "passes": False})
        with pytest.raises(ValueError, match="finalist standard"):
            promote(session, spec="equal_weight", universe="top:500",
                    schedule="monthly", tier="validated", finalist=True)
        assert not is_finalist_confirmed(
            ValidationRun(strategy_name="x", holdout={}, verdict="FAIL"))

    def test_passes_positive_excess_even_when_criteria_fail(self, factory):
        """The finalist bar is the pre-registered total-return standard —
        criteria.all_met (the risk-adjusted bar) is allowed to be False."""
        session = factory()
        self._run(session, holdout={"excess_vs_spy_pct": 25.9, "passes": False},
                  criteria={"all_met": False})
        row = promote(session, spec="equal_weight", universe="top:500",
                      schedule="monthly", tier="validated", finalist=True)
        assert row.tier == "validated"
        assert row.validation_run_id is not None
        # the strict default gate still refuses the same run
        with pytest.raises(ValueError, match="not.*validated|validated bar"):
            promote(session, spec="equal_weight", universe="top:500",
                    schedule="monthly", tier="validated", finalist=False)


class TestLiveUniverseCycle:
    def _promote(self, factory, top_n=5, candidates=None):
        """Seed the DB ranking source (daily_bars carries the hot set the
        SQL resolver ranks) and promote. ``candidates`` limits which
        ranked names exist in the DB (the shrink tests' lever)."""
        session = factory()
        for sym, vol in VOLUMES.items():
            if candidates is not None and sym not in candidates and sym != "SPY":
                continue
            _seed_db_bars(session, sym, START, JAN_END, volume=vol)
        promote(session, spec="equal_weight", universe=f"top:{top_n}",
                rank_window=126, schedule="monthly", tier="experimental")
        session.close()

    def test_first_cycle_resolves_trades_and_persists(self, factory, monkeypatch):
        self._promote(factory)
        _stub_store(monkeypatch, _market())
        summary = run_portfolio_cycle(
            factory, today=date(2024, 2, 1), price_fn=lambda s: 100.0)
        res = summary["strategies"]["equal_weight"]
        assert res["action"] == "rebalance"
        assert res["universe"] == {"spec": "top:5", "size": 5}
        bought = {t["symbol"] for t in res["trades"]}
        assert bought == {"AAA", "BBB", "CCC", "DDD", "EEE"}

        session = factory()
        promo = session.query(PromotedStrategy).one()
        assert sorted(promo.resolved_symbols) == sorted(bought)
        assert promo.resolved_at == date(2024, 2, 1)
        lots = (session.query(TradeRecord)
                .filter_by(strategy_name="equal_weight", status="OPEN").all())
        assert len(lots) == 5
        session.close()

    def test_hold_day_never_touches_the_store(self, factory, monkeypatch):
        self._promote(factory)
        _stub_store(monkeypatch, _market())
        run_portfolio_cycle(factory, today=date(2024, 2, 1),
                            price_fn=lambda s: 100.0)

        def _bomb(symbols, start=None, end=None):
            raise AssertionError("store loaded on a hold day")
        monkeypatch.setattr(live_mod, "load_bars_from_store", _bomb)
        refreshed: list = []
        real_refresh = live_mod.ensure_recent_bars

        def _track_refresh(session, provider, symbols, today):
            refreshed.append(list(symbols))
            return real_refresh(session, provider, symbols, today)
        monkeypatch.setattr(live_mod, "ensure_recent_bars", _track_refresh)

        session = factory()
        _seed_db_bars(session, "SPY", date(2024, 2, 1), date(2024, 2, 1))
        session.close()
        summary = run_portfolio_cycle(factory, today=date(2024, 2, 2),
                                      price_fn=lambda s: 100.0)
        res = summary["strategies"]["equal_weight"]
        assert res["action"] == "hold"          # would be "error" if loaded
        # per-symbol bar refresh covered ONLY the held names, not the universe
        assert refreshed == [["AAA", "BBB", "CCC", "DDD", "EEE"]]

    def test_held_name_dropped_from_universe_is_sold(self, factory, monkeypatch):
        self._promote(factory)
        frames = _market()
        _stub_store(monkeypatch, frames)
        run_portfolio_cycle(factory, today=date(2024, 2, 1),
                            price_fn=lambda s: 100.0)

        # EEE's liquidity collapses; FFF takes its top-5 slot at the next
        # boundary. The DB is one day fresher than the store (top-up path).
        feb_end = date(2024, 2, 29)
        frames = _market(end=date(2024, 2, 28))
        frames["EEE"] = _frame(START, date(2024, 2, 28), volume=1.0)
        _stub_store(monkeypatch, frames)
        session = factory()
        for sym, vol in VOLUMES.items():
            _seed_db_bars(session, sym, date(2024, 2, 1), feb_end, volume=vol)
        # the ranking is SQL-side now: EEE collapses in daily_bars too
        session.query(DailyBar).filter(DailyBar.symbol == "EEE").update(
            {DailyBar.volume: 1.0})
        session.commit()
        session.close()

        summary = run_portfolio_cycle(factory, today=date(2024, 3, 1),
                                      price_fn=lambda s: 100.0)
        res = summary["strategies"]["equal_weight"]
        sells = {t["symbol"] for t in res["trades"] if t["side"] == "SELL"}
        buys = {t["symbol"] for t in res["trades"] if t["side"] == "BUY"}
        assert "EEE" in sells and "FFF" in buys
        session = factory()
        assert (session.query(TradeRecord)
                .filter_by(strategy_name="equal_weight", symbol="EEE",
                           status="OPEN").count()) == 0
        promo = session.query(PromotedStrategy).one()
        assert "FFF" in promo.resolved_symbols
        assert "EEE" not in promo.resolved_symbols
        session.close()

    def test_shrink_guard_falls_back_and_alerts(self, factory, monkeypatch):
        self._promote(factory, candidates=("AAA", "BBB"))
        session = factory()
        promo = session.query(PromotedStrategy).one()
        promo.resolved_symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        promo.resolved_at = date(2024, 1, 2)
        session.commit()
        session.close()

        # only 3 names rankable in the DB incl. tiny SPY (< 90% of 5)
        # -> guard fires;
        # frames exist only for AAA/BBB of the fallback list
        shrunk = {s: f for s, f in _market().items()
                  if s in ("SPY", "AAA", "BBB")}
        _stub_store(monkeypatch, shrunk)
        summary = run_portfolio_cycle(factory, today=date(2024, 2, 1),
                                      price_fn=lambda s: 100.0)
        res = summary["strategies"]["equal_weight"]
        assert res["action"] == "rebalance"     # traded the FALLBACK list
        assert "fell back" in res["universe"]["note"]
        # only fallback names with frames are in the context -> AAA/BBB
        assert {t["symbol"] for t in res["trades"]} == {"AAA", "BBB"}

        session = factory()
        obs = session.query(AgentObservation).one()
        assert obs.severity == "CRITICAL"
        assert obs.category == "live_universe"
        assert obs.agent_name == "engine.live"
        assert "resolved only 3/5" in obs.message
        # the bad resolution must NOT overwrite the last good one
        promo = session.query(PromotedStrategy).one()
        assert promo.resolved_symbols == ["AAA", "BBB", "CCC", "DDD", "EEE"]
        assert promo.resolved_at == date(2024, 1, 2)
        session.close()

    def test_shrink_guard_without_fallback_errors(self, factory, monkeypatch):
        self._promote(factory, candidates=("AAA", "BBB"))
        shrunk = {s: f for s, f in _market().items()
                  if s in ("SPY", "AAA", "BBB")}
        _stub_store(monkeypatch, shrunk)
        summary = run_portfolio_cycle(factory, today=date(2024, 2, 1),
                                      price_fn=lambda s: 100.0)
        res = summary["strategies"]["equal_weight"]
        assert res["action"] == "error"
        assert "refusing to trade" in res["reason"]
        session = factory()
        assert session.query(AgentObservation).count() == 1
        assert session.query(TradeRecord).count() == 0
        session.close()

    def test_fundamentals_reach_universe_strategies(self, factory, monkeypatch):
        """A strategy that only buys names WITH fundamentals must trade live —
        the PIT reader is wired into the context exactly as in the lab."""

        class FundamentalsBuyer:
            name = "funda_buyer"

            def rebalance(self, ctx):
                syms = sorted(s for s, a in ctx.assets.items()
                              if a.fundamentals is not None
                              and (a.fundamentals.earnings_growth or 0) > 0)
                return {s: 1.0 / len(syms) for s in syms} if syms else {}

        self._promote(factory)
        monkeypatch.setattr(live_mod, "make_strategy_factory",
                            lambda spec: FundamentalsBuyer)
        session = factory()
        for sym in ("AAA", "BBB"):
            session.add(FundamentalsSnapshot(
                symbol=sym, as_of=date(2024, 1, 15),
                data={"symbol": sym, "earnings_growth": 12.5}))
        session.commit()
        session.close()

        _stub_store(monkeypatch, _market())
        summary = run_portfolio_cycle(factory, today=date(2024, 2, 1),
                                      price_fn=lambda s: 100.0)
        res = summary["strategies"]["equal_weight"]
        assert res["action"] == "rebalance"
        assert {t["symbol"] for t in res["trades"]} == {"AAA", "BBB"}
        assert res["weights"] == {"AAA": 0.5, "BBB": 0.5}


def test_trailing_rank_start_mirrors_validator_arithmetic():
    """The shared helper must equal the validator's inline computation
    (days[max(0, i - rank_window)] with as_of = days[max(0, i - 1)])."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(60)]
    for i in (0, 1, 5, 30, 59):
        as_of = days[max(0, i - 1)]
        for rw in (0, 1, 5, 126):
            expected = days[max(0, i - rw)] if rw else None
            assert trailing_rank_start(days, as_of, rw) == expected
