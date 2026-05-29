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


def test_with_benchmarks_folds_in_index_etfs():
    """A scoped backfill auto-includes the benchmark ETFs so the backtest's
    full-range SPY/QQQ/IWM/DIA benchmark stays populated."""
    from config.settings import settings
    from scripts.backfill_daily_bars import _with_benchmarks

    out = _with_benchmarks(["AAPL", "NVDA"])
    for idx in settings.index_symbols:
        assert idx.upper() in out
    assert "AAPL" in out and "NVDA" in out
    assert out == sorted(out) and len(out) == len(set(out))  # sorted + deduped


def test_with_benchmarks_noops_when_unscoped_or_disabled():
    from scripts.backfill_daily_bars import _with_benchmarks

    assert _with_benchmarks(None) is None                       # full-market run
    assert _with_benchmarks(["AAPL"], include=False) == ["AAPL"]  # opted out
    # Already-present ETF isn't duplicated.
    assert _with_benchmarks(["SPY"]).count("SPY") == 1


def test_run_backfill_skips_failing_day():
    """One bad day is logged + skipped; the rest of the range still loads."""
    import pandas as pd
    from scripts.backfill_daily_bars import run_backfill

    engine = _engine()

    def _df(symbol, d):
        return pd.DataFrame([{
            "ticker": symbol, "open": 1.0, "high": 2.0, "low": 0.5,
            "close": 1.5, "volume": 100.0, "transactions": 10, "date": d,
        }])

    class FakeClient:
        def read_day_aggs(self, d, symbols=None, use_cache=True):
            if d == date(2025, 1, 2):
                raise RuntimeError("boom: corrupt file")
            return _df("AAPL", d)

    days = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    total, failed = run_backfill(FakeClient(), engine, days)

    assert total == 2                                  # two good days written
    assert [d for d, _ in failed] == [date(2025, 1, 2)]  # bad day captured
    with sessionmaker(bind=engine)() as s:
        assert s.query(DailyBar).count() == 2
