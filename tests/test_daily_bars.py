"""Tests for the daily_bars model + backfill upsert (in-memory SQLite)."""

from datetime import date

from sqlalchemy.orm import sessionmaker

from edgefinder.db.engine import Base, get_engine
from edgefinder.db.models import DailyBar
from scripts.backfill_daily_bars import upsert_daily_bars

ROWS = [
    {
        "symbol": "NVDA", "date": date(2026, 5, 26),
        "open": 216.54, "high": 218.18, "low": 212.0, "close": 214.86,
        "volume": 1000.0, "transactions": 10, "source": "flatfiles",
    },
    {
        "symbol": "AAPL", "date": date(2026, 5, 26),
        "open": 150.0, "high": 152.0, "low": 149.5, "close": 151.0,
        "volume": 2000.0, "transactions": 5, "source": "flatfiles",
    },
]


def _engine():
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_upsert_inserts_rows():
    engine = _engine()
    assert upsert_daily_bars(engine, ROWS) == 2
    with sessionmaker(bind=engine)() as s:
        assert s.query(DailyBar).count() == 2


def test_upsert_idempotent_and_updates():
    engine = _engine()
    upsert_daily_bars(engine, ROWS)
    upsert_daily_bars(engine, [dict(ROWS[0], close=999.0, volume=5.0)])
    with sessionmaker(bind=engine)() as s:
        assert s.query(DailyBar).count() == 2  # no duplicate row
        nvda = s.query(DailyBar).filter_by(symbol="NVDA").one()
        assert nvda.close == 999.0
        assert nvda.volume == 5.0


def test_upsert_empty_is_noop():
    engine = _engine()
    assert upsert_daily_bars(engine, []) == 0


def test_load_universe_reads_tickers(tmp_path):
    """--universe pulls the symbol list from the tickers table (sorted, upper, deduped)."""
    from edgefinder.db.models import Ticker
    from scripts.backfill_daily_bars import _load_universe

    url = f"sqlite:///{tmp_path}/universe.db"
    engine = get_engine(url=url)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        s.add_all([Ticker(symbol="AAPL"), Ticker(symbol="nvda"), Ticker(symbol="MSFT")])
        s.commit()

    assert _load_universe(url) == ["AAPL", "MSFT", "NVDA"]
