"""Tests for the point-in-time fundamentals store."""

from datetime import date

import pytest

from edgefinder.data.pit_fundamentals import PITFundamentals, snapshot_fundamentals
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import Fundamental, FundamentalsSnapshot, Ticker


@pytest.fixture()
def session():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = get_session_factory(engine)()
    t = Ticker(symbol="AAPL")
    sess.add(t)
    sess.flush()
    sess.add(Fundamental(ticker_id=t.id, symbol="AAPL",
                         peg_ratio=1.5, earnings_growth=0.20))
    sess.commit()
    return sess


class TestSnapshot:
    def test_snapshot_copies_current_fundamentals(self, session):
        n = snapshot_fundamentals(session, as_of=date(2026, 6, 1))
        assert n == 1
        row = session.query(FundamentalsSnapshot).one()
        assert row.symbol == "AAPL"
        assert row.as_of == date(2026, 6, 1)
        assert row.data["peg_ratio"] == 1.5
        assert row.data["symbol"] == "AAPL"

    def test_snapshot_idempotent_per_date(self, session):
        snapshot_fundamentals(session, as_of=date(2026, 6, 1))
        assert snapshot_fundamentals(session, as_of=date(2026, 6, 1)) == 0
        assert session.query(FundamentalsSnapshot).count() == 1

    def test_history_accumulates_across_dates(self, session):
        snapshot_fundamentals(session, as_of=date(2026, 6, 1))
        # the live table mutates (a new scan), then a new snapshot
        session.query(Fundamental).one().peg_ratio = 2.5
        session.commit()
        snapshot_fundamentals(session, as_of=date(2026, 6, 8))
        assert session.query(FundamentalsSnapshot).count() == 2


class TestAsOfReader:
    def test_asof_returns_latest_at_or_before(self, session):
        snapshot_fundamentals(session, as_of=date(2026, 6, 1))
        session.query(Fundamental).one().peg_ratio = 2.5
        session.commit()
        snapshot_fundamentals(session, as_of=date(2026, 6, 8))

        pit = PITFundamentals(session)
        assert pit.asof("AAPL", date(2026, 5, 31)) is None      # before coverage
        assert pit.asof("AAPL", date(2026, 6, 1)).peg_ratio == 1.5
        assert pit.asof("AAPL", date(2026, 6, 5)).peg_ratio == 1.5  # holds between
        assert pit.asof("AAPL", date(2026, 6, 8)).peg_ratio == 2.5
        assert pit.asof("AAPL", date(2027, 1, 1)).peg_ratio == 2.5
        assert pit.asof("UNKNOWN", date(2026, 6, 8)) is None

    def test_engine_context_uses_pit_lookup(self, session):
        from datetime import timedelta

        import pandas as pd

        from edgefinder.engine.backtest import _build_context, prepare_bars

        snapshot_fundamentals(session, as_of=date(2026, 6, 1))
        dates = [date(2026, 5, 28) + timedelta(days=i) for i in range(10)]
        px = [100.0 + i for i in range(10)]
        bars = {"AAPL": pd.DataFrame({
            "date": dates, "open": px, "high": px, "low": px,
            "close": px, "volume": [1e6] * 10})}
        prep, _ = prepare_bars(bars)
        pit = PITFundamentals(session)

        before = _build_context(prep, date(2026, 5, 30), pit)
        assert before.get("AAPL").fundamentals is None           # honest: no data yet
        after = _build_context(prep, date(2026, 6, 3), pit)
        assert after.get("AAPL").fundamentals.peg_ratio == 1.5
        # static-dict path still works (the disclosed-look-ahead mode)
        static = _build_context(prep, date(2026, 5, 30),
                                {"AAPL": after.get("AAPL").fundamentals})
        assert static.get("AAPL").fundamentals.peg_ratio == 1.5