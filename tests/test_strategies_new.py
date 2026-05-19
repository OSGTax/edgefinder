"""Tests for the three new strategies: Coward, Gambler, Degenerate."""

import pytest
from edgefinder.core.models import TickerFundamentals
from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.strategies.coward import CowardStrategy
from edgefinder.strategies.gambler import GamblerStrategy
from edgefinder.strategies.degenerate_v2 import DegenerateStrategy


def _make_data(ticker="AAPL", rsi=50.0, bb_lower=95.0, close=100.0, **kwargs):
    snap = IndicatorSnapshot(close=close, rsi=rsi, bb_lower=bb_lower, **kwargs)
    hist = IndicatorHistory(max_days=30)
    hist.add(snap)
    ctx = MarketContext()
    return MarketData(
        ticker=ticker, current=snap, history=hist,
        fundamentals=None, context=ctx, current_price=close,
    )


class TestCowardStrategy:
    def test_name_and_risk(self):
        s = CowardStrategy()
        assert s.name == "coward"
        assert s.risk_pct == 0.05
        assert s.target_pct == 0.15
        assert s.stop_pct == 0.20
        assert s.watchlist_size == 50

    def test_qualifies_stock_passes(self):
        s = CowardStrategy()
        fund = TickerFundamentals(
            symbol="AAPL", earnings_growth=0.15, current_ratio=2.0,
        )
        assert s.qualifies_stock(fund) is True

    def test_qualifies_stock_fails_no_earnings(self):
        s = CowardStrategy()
        fund = TickerFundamentals(
            symbol="AAPL", earnings_growth=-0.05, current_ratio=2.0,
        )
        assert s.qualifies_stock(fund) is False

    def test_qualifies_stock_fails_low_current_ratio(self):
        s = CowardStrategy()
        fund = TickerFundamentals(
            symbol="AAPL", earnings_growth=0.10, current_ratio=1.2,
        )
        assert s.qualifies_stock(fund) is False

    def test_entry_on_rsi_oversold(self):
        s = CowardStrategy()
        data = _make_data(rsi=30.0)
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "RSI" in intent.reasoning

    def test_entry_on_bb_lower_touch(self):
        s = CowardStrategy()
        # Price within 1% of BB lower: close=100, bb_lower=99.5
        data = _make_data(rsi=50.0, bb_lower=99.5, close=100.0)
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "BB" in intent.reasoning or "Bollinger" in intent.reasoning

    def test_no_entry_when_no_conditions_met(self):
        s = CowardStrategy()
        data = _make_data(rsi=50.0, bb_lower=80.0, close=100.0)
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_exit_on_rsi_overbought(self):
        s = CowardStrategy()
        data = _make_data(rsi=72.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is not None
        assert "RSI" in exit_intent.reasoning

    def test_no_exit_when_rsi_normal(self):
        s = CowardStrategy()
        data = _make_data(rsi=55.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is None


class TestGamblerStrategy:
    def test_name_and_risk(self):
        s = GamblerStrategy()
        assert s.name == "gambler"
        assert s.risk_pct == 0.10
        assert s.target_pct == 0.25
        assert s.watchlist_size == 100

    def test_qualifies_with_earnings_growth(self):
        s = GamblerStrategy()
        fund = TickerFundamentals(symbol="AAPL", earnings_growth=0.10)
        assert s.qualifies_stock(fund) is True

    def test_qualifies_with_revenue_growth_only(self):
        s = GamblerStrategy()
        fund = TickerFundamentals(symbol="AAPL", revenue_growth=0.05)
        assert s.qualifies_stock(fund) is True

    def test_rejects_no_growth(self):
        s = GamblerStrategy()
        fund = TickerFundamentals(symbol="AAPL")
        assert s.qualifies_stock(fund) is False

    def test_entry_on_macd_cross_and_rsi_midrange(self):
        s = GamblerStrategy()
        snap_prev = IndicatorSnapshot(close=98.0, macd_histogram=-0.5, rsi=50.0)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=0.3, rsi=50.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "MACD" in intent.reasoning

    def test_no_entry_without_macd_cross(self):
        s = GamblerStrategy()
        snap_prev = IndicatorSnapshot(close=98.0, macd_histogram=0.2, rsi=50.0)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=0.5, rsi=50.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_no_entry_rsi_out_of_range(self):
        s = GamblerStrategy()
        snap_prev = IndicatorSnapshot(close=98.0, macd_histogram=-0.5, rsi=65.0)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=0.3, rsi=65.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_exit_on_macd_negative_cross(self):
        s = GamblerStrategy()
        snap_prev = IndicatorSnapshot(close=102.0, macd_histogram=0.5)
        snap_curr = IndicatorSnapshot(close=100.0, macd_histogram=-0.2)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap_prev)
        hist.add(snap_curr)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap_curr, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )
        exit_intent = s.should_exit("AAPL", data, entry_price=95.0)
        assert exit_intent is not None
        assert "MACD" in exit_intent.reasoning


class TestDegenerateStrategy:
    def test_name_and_risk(self):
        s = DegenerateStrategy()
        assert s.name == "degenerate"
        assert s.risk_pct == 0.20
        assert s.target_pct == 0.50
        assert s.watchlist_size == 200

    def test_qualifies_anything_with_data(self):
        s = DegenerateStrategy()
        fund = TickerFundamentals(symbol="AAPL", market_cap=500_000_000)
        assert s.qualifies_stock(fund) is True

    def test_entry_on_volume_spike_and_momentum(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=100.0, rsi=55.0, ema_21=98.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=100.0, volume_ratio=2.5,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert "volume" in intent.reasoning.lower()

    def test_no_entry_without_volume_spike(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=100.0, rsi=55.0, ema_21=98.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=100.0, volume_ratio=1.2,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_no_entry_without_momentum(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=100.0, rsi=40.0, ema_21=102.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=100.0, volume_ratio=2.5,
        )
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_exit_on_volume_fade_and_overbought(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=150.0, rsi=82.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=150.0, volume_ratio=0.8,
        )
        exit_intent = s.should_exit("AAPL", data, entry_price=100.0)
        assert exit_intent is not None

    def test_no_exit_if_volume_still_high(self):
        s = DegenerateStrategy()
        snap = IndicatorSnapshot(close=150.0, rsi=82.0)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        data = MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx,
            current_price=150.0, volume_ratio=1.5,
        )
        exit_intent = s.should_exit("AAPL", data, entry_price=100.0)
        assert exit_intent is None
