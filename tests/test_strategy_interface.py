"""Tests for the new strategy interface."""

from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.strategies.strategy_interface import SwingStrategy


class FakeStrategy(SwingStrategy):
    name = "fake"
    risk_pct = 0.10
    target_pct = 0.25
    watchlist_size = 50

    def qualifies_stock(self, fundamentals):
        return True

    def evaluate(self, ticker, data):
        if data.current.rsi and data.current.rsi < 30:
            return self.make_intent(ticker, data, "RSI oversold")
        return None

    def should_exit(self, ticker, data, entry_price):
        if data.current.rsi and data.current.rsi > 70:
            return self.make_exit(ticker, data, "RSI overbought")
        return None


class TestSwingStrategy:
    def _make_data(self, rsi=50.0):
        snap = IndicatorSnapshot(close=100.0, rsi=rsi)
        hist = IndicatorHistory(max_days=30)
        hist.add(snap)
        ctx = MarketContext()
        return MarketData(
            ticker="AAPL", current=snap, history=hist,
            fundamentals=None, context=ctx, current_price=100.0,
        )

    def test_evaluate_returns_intent_when_conditions_met(self):
        s = FakeStrategy()
        data = self._make_data(rsi=25.0)
        intent = s.evaluate("AAPL", data)
        assert intent is not None
        assert intent.ticker == "AAPL"
        assert intent.direction == "LONG"
        assert "RSI oversold" in intent.reasoning

    def test_evaluate_returns_none_when_no_signal(self):
        s = FakeStrategy()
        data = self._make_data(rsi=50.0)
        intent = s.evaluate("AAPL", data)
        assert intent is None

    def test_should_exit_returns_intent(self):
        s = FakeStrategy()
        data = self._make_data(rsi=75.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is not None
        assert "RSI overbought" in exit_intent.reasoning

    def test_should_exit_returns_none(self):
        s = FakeStrategy()
        data = self._make_data(rsi=50.0)
        exit_intent = s.should_exit("AAPL", data, entry_price=90.0)
        assert exit_intent is None

    def test_stop_pct_is_always_20(self):
        s = FakeStrategy()
        assert s.stop_pct == 0.20
