"""Tests for the three new strategies: Coward, Gambler, Degenerate."""

import pytest
from edgefinder.core.models import TickerFundamentals
from edgefinder.data.market_data import (
    IndicatorHistory, IndicatorSnapshot, MarketContext, MarketData,
)
from edgefinder.strategies.coward import CowardStrategy


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
