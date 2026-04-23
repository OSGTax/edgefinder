"""Tests for edgefinder/market/ — snapshot and benchmarks."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from edgefinder.core.models import MarketRegime
from edgefinder.db.models import IndexDaily, MarketSnapshotRecord
from edgefinder.market.benchmarks import BenchmarkService
from edgefinder.market.snapshot import MarketSnapshotService


# ── Snapshot Tests ───────────────────────────────────────


class TestMarketSnapshotService:
    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        prices = {
            "SPY": 450.0, "QQQ": 380.0, "IWM": 200.0, "DIA": 350.0,
            "VIXY": 18.0,
            "XLK": 180.0, "XLF": 38.0, "XLE": 85.0, "XLV": 140.0,
            "XLI": 105.0, "XLP": 75.0, "XLY": 170.0, "XLU": 65.0,
            "XLRE": 38.0, "XLC": 72.0, "XLB": 82.0,
        }
        provider.get_latest_price.side_effect = lambda s: prices.get(s)
        # Mock get_bars to return 2 daily bars for change % computation
        def mock_bars(ticker, timeframe, start, end=None):
            price = prices.get(ticker)
            if price is None:
                return None
            prev_price = price * 0.99  # 1% up from previous close
            dates = pd.date_range("2026-04-15", periods=2, freq="D")
            return pd.DataFrame({
                "open": [prev_price, price],
                "high": [prev_price + 1, price + 1],
                "low": [prev_price - 1, price - 1],
                "close": [prev_price, price],
                "volume": [1000000, 1000000],
            }, index=dates)
        provider.get_bars.side_effect = mock_bars
        return provider

    @pytest.fixture
    def service(self, mock_provider, db_session):
        return MarketSnapshotService(mock_provider, db_session)

    def test_capture_returns_snapshot(self, service):
        snap = service.capture()
        assert snap.spy_price == 450.0
        assert snap.qqq_price == 380.0
        assert snap.vix_level == 18.0
        assert snap.spy_change_pct != 0.0  # should be ~1%
        assert "XLK" in snap.sector_performance

    def test_capture_and_persist(self, service, db_session):
        snap_id = service.capture_and_persist()
        assert snap_id is not None
        record = db_session.get(MarketSnapshotRecord, snap_id)
        assert record.spy_price == 450.0

    def test_capture_and_persist_no_commit_flushes_only(self, service, db_session):
        """With commit=False the id is available but txn stays open.

        Needed for atomic snapshot + trade writes in on_trade_opened.
        """
        snap_id = service.capture_and_persist(commit=False)
        assert snap_id is not None
        record = db_session.get(MarketSnapshotRecord, snap_id)
        assert record.spy_price == 450.0
        # Rolling back should wipe it — proves commit didn't happen.
        db_session.rollback()
        assert db_session.get(MarketSnapshotRecord, snap_id) is None

    def test_get_latest(self, service, db_session):
        service.capture_and_persist()
        latest = service.get_latest()
        assert latest is not None
        assert latest.spy_price == 450.0

    def test_regime_bull_low_vix(self, db_session):
        provider = MagicMock()
        prices = {"SPY": 450.0, "QQQ": 380.0, "IWM": 200.0, "DIA": 350.0, "VIXY": 12.0}
        provider.get_latest_price.side_effect = lambda s: prices.get(s)
        # 1% up day triggers bull with low VIX
        def mock_bars(ticker, timeframe, start, end=None):
            price = prices.get(ticker)
            if price is None:
                return None
            prev = price * 0.99
            dates = pd.date_range("2026-04-15", periods=2, freq="D")
            return pd.DataFrame({"open": [prev, price], "high": [prev+1, price+1],
                "low": [prev-1, price-1], "close": [prev, price], "volume": [1e6, 1e6]}, index=dates)
        provider.get_bars.side_effect = mock_bars
        service = MarketSnapshotService(provider, db_session)
        snap = service.capture()
        assert snap.market_regime == MarketRegime.BULL

    def test_regime_bear_high_vix(self, db_session):
        provider = MagicMock()
        prices = {"SPY": 380.0, "QQQ": 300.0, "IWM": 160.0, "DIA": 290.0, "VIXY": 35.0}
        provider.get_latest_price.side_effect = lambda s: prices.get(s)
        # -2% down day + high VIX = bear
        def mock_bars(ticker, timeframe, start, end=None):
            price = prices.get(ticker)
            if price is None:
                return None
            prev = price * 1.02  # price fell 2% from prev
            dates = pd.date_range("2026-04-15", periods=2, freq="D")
            return pd.DataFrame({"open": [prev, price], "high": [prev+1, price+1],
                "low": [prev-1, price-1], "close": [prev, price], "volume": [1e6, 1e6]}, index=dates)
        provider.get_bars.side_effect = mock_bars
        service = MarketSnapshotService(provider, db_session)
        snap = service.capture()
        assert snap.market_regime == MarketRegime.BEAR

    def test_multiple_snapshots(self, service, db_session):
        service.capture_and_persist()
        service.capture_and_persist()
        count = db_session.query(MarketSnapshotRecord).count()
        assert count == 2


# ── Benchmark Tests ──────────────────────────────────────


class TestBenchmarkService:
    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.get_latest_price.side_effect = lambda s: {
            "SPY": 450.0, "QQQ": 380.0, "IWM": 200.0, "DIA": 350.0,
        }.get(s)

        # Mock bars for backfill
        n = 30
        idx = pd.date_range(end=date.today(), periods=n, freq="B", name="timestamp")
        df = pd.DataFrame({
            "open": np.linspace(430, 450, n),
            "high": np.linspace(432, 452, n),
            "low": np.linspace(428, 448, n),
            "close": np.linspace(430, 450, n),
            "volume": np.full(n, 1e6),
        }, index=idx)
        provider.get_bars.return_value = df
        return provider

    @pytest.fixture
    def service(self, mock_provider, db_session):
        return BenchmarkService(mock_provider, db_session)

    def test_collect_daily(self, service, db_session):
        stored = service.collect_daily()
        assert stored == 4
        records = db_session.query(IndexDaily).all()
        assert len(records) == 4

    def test_collect_daily_upsert(self, service, db_session):
        service.collect_daily()
        service.collect_daily()
        count = db_session.query(IndexDaily).filter_by(symbol="SPY").count()
        assert count == 1

    def test_collect_daily_change_pct(self, service, db_session):
        # First day — no previous, so change_pct = 0
        service.collect_daily(as_of=date(2024, 1, 15))
        # Check SPY record
        record = db_session.query(IndexDaily).filter_by(symbol="SPY").first()
        assert record is not None
        assert record.change_pct == 0.0

    def test_backfill(self, service, db_session):
        stored = service.backfill(days=30)
        assert stored > 0
        records = db_session.query(IndexDaily).filter_by(symbol="SPY").all()
        assert len(records) > 0

    def test_get_comparison_data_empty(self, service):
        result = service.get_comparison_data()
        assert result["dates"] == []
        assert result["indices"] == {}

    def test_get_comparison_data_with_data(self, service, db_session):
        # Seed some data
        base = datetime(2024, 1, 2)
        for i in range(5):
            db_session.add(IndexDaily(
                symbol="SPY",
                date=base + timedelta(days=i),
                close=450.0 + i * 2,
                change_pct=round(2 / 450 * 100, 4),
            ))
        db_session.commit()

        result = service.get_comparison_data(start_date=date(2024, 1, 2))
        assert "SPY" in result["indices"]
        assert len(result["indices"]["SPY"]) == 5
        assert result["indices"]["SPY"][0] == 0.0  # first day is 0%

    def test_previous_close(self, service, db_session):
        db_session.add(IndexDaily(
            symbol="SPY", date=datetime(2024, 1, 15), close=448.0, change_pct=0.5,
        ))
        db_session.commit()
        prev = service._get_previous_close("SPY")
        assert prev == 448.0

    def test_previous_close_none(self, service):
        assert service._get_previous_close("SPY") is None
