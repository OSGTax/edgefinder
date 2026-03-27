"""Tests for edgefinder/core/models.py."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from edgefinder.core.models import (
    AggregatedSentiment,
    BarData,
    Direction,
    MarketRegime,
    MarketSnapshot,
    Signal,
    SignalAction,
    SentimentSource,
    StrategyAccountState,
    TickerFundamentals,
    TickerSentiment,
    Trade,
    TradeStatus,
    TradeType,
)


class TestSignal:
    def test_construction(self):
        s = Signal(
            ticker="AAPL",
            action=SignalAction.BUY,
            entry_price=150.0,
            stop_loss=145.0,
            target=160.0,
            confidence=75.0,
            strategy_name="alpha",
        )
        assert s.ticker == "AAPL"
        assert s.action == SignalAction.BUY
        assert s.trade_type == TradeType.DAY

    def test_risk_per_share(self):
        s = Signal(
            ticker="AAPL",
            action=SignalAction.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            target=110.0,
        )
        assert s.risk_per_share == 5.0

    def test_reward_to_risk(self):
        s = Signal(
            ticker="AAPL",
            action=SignalAction.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            target=110.0,
        )
        assert s.reward_to_risk == 2.0

    def test_reward_to_risk_zero_risk(self):
        s = Signal(
            ticker="AAPL",
            action=SignalAction.BUY,
            entry_price=100.0,
            stop_loss=100.0,
            target=110.0,
        )
        assert s.reward_to_risk == 0.0

    def test_default_shares_zero(self):
        s = Signal(
            ticker="AAPL",
            action=SignalAction.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            target=110.0,
        )
        assert s.shares == 0


class TestTrade:
    def test_open_trade(self):
        t = Trade(
            trade_id="abc-123",
            strategy_name="alpha",
            symbol="MSFT",
            direction=Direction.LONG,
            trade_type=TradeType.SWING,
            entry_price=300.0,
            shares=10,
            stop_loss=290.0,
            target=320.0,
            confidence=80.0,
        )
        assert t.status == TradeStatus.OPEN
        assert t.exit_price is None
        assert t.pnl_dollars is None

    def test_closed_trade(self):
        t = Trade(
            trade_id="abc-456",
            strategy_name="bravo",
            symbol="GOOG",
            direction=Direction.LONG,
            trade_type=TradeType.DAY,
            entry_price=100.0,
            exit_price=110.0,
            shares=50,
            stop_loss=95.0,
            target=110.0,
            confidence=70.0,
            status=TradeStatus.CLOSED,
            pnl_dollars=500.0,
            pnl_percent=10.0,
            r_multiple=2.0,
            exit_reason="TARGET_HIT",
        )
        assert t.status == TradeStatus.CLOSED
        assert t.pnl_dollars == 500.0


class TestMarketSnapshot:
    def test_construction(self):
        ms = MarketSnapshot(
            spy_price=450.0,
            spy_change_pct=0.5,
            qqq_price=380.0,
            qqq_change_pct=0.8,
            iwm_price=200.0,
            iwm_change_pct=-0.3,
            dia_price=350.0,
            dia_change_pct=0.2,
            vix_level=15.5,
            market_regime=MarketRegime.BULL,
            sector_performance={"XLK": 1.2, "XLF": -0.5},
        )
        assert ms.market_regime == MarketRegime.BULL
        assert ms.sector_performance["XLK"] == 1.2

    def test_defaults(self):
        ms = MarketSnapshot()
        assert ms.spy_price == 0.0
        assert ms.market_regime == MarketRegime.SIDEWAYS


class TestTickerSentiment:
    def test_score_bounds(self):
        ts = TickerSentiment(
            symbol="AAPL",
            source=SentimentSource.REDDIT,
            score=0.5,
            mention_count=100,
        )
        assert ts.score == 0.5

    def test_score_too_high(self):
        with pytest.raises(ValidationError):
            TickerSentiment(
                symbol="AAPL",
                source=SentimentSource.REDDIT,
                score=1.5,
            )

    def test_score_too_low(self):
        with pytest.raises(ValidationError):
            TickerSentiment(
                symbol="AAPL",
                source=SentimentSource.TWITTER,
                score=-1.5,
            )


class TestAggregatedSentiment:
    def test_construction(self):
        agg = AggregatedSentiment(
            symbol="TSLA",
            composite_score=0.3,
            source_scores={"reddit": 0.5, "twitter": 0.1},
            total_mentions=250,
            is_trending=True,
        )
        assert agg.composite_score == 0.3
        assert agg.total_mentions == 250


class TestTickerFundamentals:
    def test_optional_fields(self):
        tf = TickerFundamentals(symbol="AAPL")
        assert tf.company_name is None
        assert tf.peg_ratio is None
        assert tf.raw_data is None

    def test_full_fields(self):
        tf = TickerFundamentals(
            symbol="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            market_cap=3_000_000_000_000,
            peg_ratio=1.2,
            earnings_growth=0.25,
            debt_to_equity=0.5,
        )
        assert tf.company_name == "Apple Inc."
        assert tf.peg_ratio == 1.2


class TestStrategyAccountState:
    def test_defaults(self):
        sa = StrategyAccountState(strategy_name="alpha")
        assert sa.starting_capital == 5_000.00
        assert sa.cash_balance == 5_000.00
        assert sa.buying_power == 5_000.00
        assert sa.pdt_enabled is False
        assert sa.open_positions == []

    def test_buying_power_equals_cash(self):
        sa = StrategyAccountState(
            strategy_name="alpha",
            cash_balance=3_200.00,
            open_positions_value=1_800.00,
        )
        assert sa.buying_power == 3_200.00


class TestBarData:
    def test_construction(self):
        bar = BarData(
            timestamp=datetime(2024, 1, 15, 10, 30),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.5,
            volume=1_000_000,
        )
        assert bar.close == 151.5
        assert bar.vwap is None

    def test_with_optional_fields(self):
        bar = BarData(
            timestamp=datetime(2024, 1, 15, 10, 30),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.5,
            volume=1_000_000,
            vwap=150.8,
            trade_count=5000,
        )
        assert bar.vwap == 150.8
        assert bar.trade_count == 5000
