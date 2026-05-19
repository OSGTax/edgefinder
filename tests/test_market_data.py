"""Tests for the shared MarketData layer."""

from edgefinder.data.market_data import (
    IndicatorSnapshot,
    IndicatorHistory,
    MarketContext,
    MarketData,
)


class TestIndicatorSnapshot:
    def test_to_dict_excludes_none(self):
        snap = IndicatorSnapshot(close=100.0, rsi=45.0)
        d = snap.to_dict()
        assert d["close"] == 100.0
        assert d["rsi"] == 45.0
        assert "macd_line" not in d  # None fields excluded


class TestIndicatorHistory:
    def test_add_and_retrieve(self):
        hist = IndicatorHistory(max_days=30)
        snap1 = IndicatorSnapshot(close=100.0, rsi=40.0)
        snap2 = IndicatorSnapshot(close=105.0, rsi=55.0)
        hist.add(snap1)
        hist.add(snap2)
        assert len(hist) == 2
        assert hist.latest.close == 105.0

    def test_max_days_enforced(self):
        hist = IndicatorHistory(max_days=3)
        for i in range(5):
            hist.add(IndicatorSnapshot(close=float(100 + i)))
        assert len(hist) == 3
        assert hist.latest.close == 104.0  # most recent
        assert hist.oldest.close == 102.0  # oldest kept

    def test_get_field_series(self):
        hist = IndicatorHistory(max_days=30)
        hist.add(IndicatorSnapshot(close=100.0, rsi=30.0))
        hist.add(IndicatorSnapshot(close=105.0, rsi=45.0))
        hist.add(IndicatorSnapshot(close=110.0, rsi=60.0))
        rsi_series = hist.get_field_series("rsi")
        assert rsi_series == [30.0, 45.0, 60.0]

    def test_previous_returns_day_before_latest(self):
        hist = IndicatorHistory(max_days=30)
        hist.add(IndicatorSnapshot(close=100.0, macd_histogram=-0.5))
        hist.add(IndicatorSnapshot(close=105.0, macd_histogram=0.3))
        assert hist.previous.macd_histogram == -0.5


class TestMarketData:
    def test_construction(self):
        snap = IndicatorSnapshot(close=100.0, rsi=45.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext(
            spy_price=450.0, spy_change_pct=0.5,
            qqq_price=380.0, qqq_change_pct=0.3,
            iwm_price=200.0, iwm_change_pct=-0.1,
            dia_price=350.0, dia_change_pct=0.2,
            vix_level=18.0, market_regime="bull",
        )
        md = MarketData(
            ticker="AAPL",
            current=snap,
            history=hist,
            fundamentals=None,
            context=ctx,
            current_price=150.0,
            today_volume=5_000_000,
            avg_daily_volume=3_000_000,
            volume_ratio=2.5,
        )
        assert md.ticker == "AAPL"
        assert md.current.rsi == 45.0
        assert md.volume_ratio == 2.5
