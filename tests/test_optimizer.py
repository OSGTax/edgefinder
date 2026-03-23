"""
EdgeFinder Module 5 Tests: Strategy Optimizer
==============================================
Tests cover: indicator analysis, trade type analysis, confidence bands,
exit reason analysis, recommendations, parameter logging, and edge cases.

Run: python -m pytest tests/test_optimizer.py -v
"""

import pytest
from datetime import datetime, timezone

from modules.optimizer import (
    StrategyOptimizer,
    OptimizationReport,
    IndicatorPerformance,
    AnalysisBucket,
    Recommendation,
    MIN_TRADES_FOR_ANALYSIS,
    MIN_TRADES_FOR_RECOMMENDATIONS,
)
from modules.database import (
    Trade as TradeRecord,
    StrategyParameter,
    get_session,
)


# ── FIXTURES ─────────────────────────────────────────────────

@pytest.fixture
def optimizer() -> StrategyOptimizer:
    return StrategyOptimizer()


def _make_trade(
    trade_id: str,
    ticker: str = "AAPL",
    trade_type: str = "DAY",
    pnl: float = 10.0,
    r_multiple: float = 1.0,
    exit_reason: str = "TARGET_HIT",
    confidence: float = 70.0,
    signals: dict = None,
) -> None:
    """Insert a trade record into the database."""
    session = get_session()
    session.add(TradeRecord(
        trade_id=trade_id,
        ticker=ticker,
        direction="LONG",
        trade_type=trade_type,
        entry_price=100.0,
        exit_price=100.0 + (pnl / 10),
        shares=10,
        stop_loss=98.0,
        target=103.0,
        entry_time=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
        status="CLOSED",
        pnl_dollars=pnl,
        pnl_percent=pnl / 1000,
        r_multiple=r_multiple,
        exit_reason=exit_reason,
        confidence_score=confidence,
        technical_signals=signals or {},
    ))
    session.commit()
    session.close()


def _seed_many_trades(n: int = 60, win_rate: float = 0.5):
    """Seed N trades with a given approximate win rate."""
    for i in range(n):
        is_win = (i % int(1 / win_rate if win_rate > 0 else 999)) == 0 if win_rate < 1 else True
        pnl = 30.0 if is_win else -20.0
        r = 1.5 if is_win else -1.0
        signals = {"rsi_oversold": {"rsi": 25}} if i % 2 == 0 else {"ema_crossover_day": {}}
        _make_trade(
            trade_id=f"bulk-{i:03d}",
            ticker=f"T{i % 10}",
            trade_type="DAY" if i % 3 != 0 else "SWING",
            pnl=pnl,
            r_multiple=r,
            exit_reason="TARGET_HIT" if is_win else "STOP_HIT",
            confidence=50 + (i % 4) * 15,
            signals=signals,
        )


# ════════════════════════════════════════════════════════════
# INSUFFICIENT DATA
# ════════════════════════════════════════════════════════════

class TestInsufficientData:
    """Test behavior with few trades."""

    def test_no_trades(self, in_memory_db, optimizer):
        report = optimizer.analyze()
        assert report.total_trades_analyzed == 0
        assert report.sufficient_data is False
        assert report.recommendations == []

    def test_few_trades(self, in_memory_db, optimizer):
        for i in range(5):
            _make_trade(f"t-{i}", pnl=10.0)
        report = optimizer.analyze()
        assert report.total_trades_analyzed == 5
        assert report.sufficient_data is False

    def test_enough_for_analysis(self, in_memory_db, optimizer):
        for i in range(MIN_TRADES_FOR_ANALYSIS):
            _make_trade(f"t-{i}", pnl=10.0)
        report = optimizer.analyze()
        assert report.total_trades_analyzed == MIN_TRADES_FOR_ANALYSIS
        assert len(report.trade_type_performance) > 0


# ════════════════════════════════════════════════════════════
# INDICATOR ANALYSIS
# ════════════════════════════════════════════════════════════

class TestIndicatorAnalysis:
    """Test win rate analysis by indicator."""

    def test_indicator_performance(self, in_memory_db, optimizer):
        # 5 wins with RSI, 5 losses with EMA
        for i in range(5):
            _make_trade(f"w-{i}", pnl=20.0, signals={"rsi_oversold": {"rsi": 25}})
        for i in range(5):
            _make_trade(f"l-{i}", pnl=-15.0, signals={"ema_crossover_day": {}})
        report = optimizer.analyze()
        rsi = next((p for p in report.indicator_performance if p.name == "rsi_oversold"), None)
        ema = next((p for p in report.indicator_performance if p.name == "ema_crossover_day"), None)
        assert rsi is not None
        assert rsi.win_rate == 1.0
        assert rsi.appearances == 5
        assert ema is not None
        assert ema.win_rate == 0.0

    def test_multi_indicator_trades(self, in_memory_db, optimizer):
        """Trade with multiple indicators should count for each."""
        for i in range(10):
            _make_trade(f"m-{i}", pnl=10.0, signals={
                "rsi_oversold": {"rsi": 25},
                "macd_crossover": {"macd": 0.5},
            })
        report = optimizer.analyze()
        rsi = next(p for p in report.indicator_performance if p.name == "rsi_oversold")
        macd = next(p for p in report.indicator_performance if p.name == "macd_crossover")
        assert rsi.appearances == 10
        assert macd.appearances == 10

    def test_no_signals_field(self, in_memory_db, optimizer):
        """Trades without technical_signals should not crash."""
        for i in range(10):
            _make_trade(f"ns-{i}", pnl=10.0, signals=None)
        report = optimizer.analyze()
        assert report.indicator_performance == []


# ════════════════════════════════════════════════════════════
# TRADE TYPE ANALYSIS
# ════════════════════════════════════════════════════════════

class TestTradeTypeAnalysis:
    """Test DAY vs SWING performance analysis."""

    def test_trade_type_breakdown(self, in_memory_db, optimizer):
        for i in range(6):
            _make_trade(f"d-{i}", trade_type="DAY", pnl=10.0)
        for i in range(6):
            _make_trade(f"s-{i}", trade_type="SWING", pnl=-5.0)
        report = optimizer.analyze()
        day = next(b for b in report.trade_type_performance if b.label == "DAY")
        swing = next(b for b in report.trade_type_performance if b.label == "SWING")
        assert day.count == 6
        assert day.win_rate == 1.0
        assert swing.count == 6
        assert swing.win_rate == 0.0


# ════════════════════════════════════════════════════════════
# CONFIDENCE BAND ANALYSIS
# ════════════════════════════════════════════════════════════

class TestConfidenceBands:
    """Test performance by confidence score range."""

    def test_confidence_bands(self, in_memory_db, optimizer):
        for i in range(5):
            _make_trade(f"lo-{i}", confidence=50.0, pnl=-10.0)
        for i in range(5):
            _make_trade(f"hi-{i}", confidence=85.0, pnl=20.0)
        report = optimizer.analyze()
        lo = next((b for b in report.confidence_band_performance if "Low" in b.label), None)
        hi = next((b for b in report.confidence_band_performance if "High" in b.label), None)
        assert lo is not None
        assert lo.count == 5
        assert lo.win_rate == 0.0
        assert hi is not None
        assert hi.count == 5
        assert hi.win_rate == 1.0

    def test_confidence_outside_bands(self, in_memory_db, optimizer):
        """Trades with confidence < 40 should not appear in any band."""
        for i in range(10):
            _make_trade(f"vc-{i}", confidence=20.0, pnl=5.0)
        report = optimizer.analyze()
        total_in_bands = sum(b.count for b in report.confidence_band_performance)
        assert total_in_bands == 0


# ════════════════════════════════════════════════════════════
# EXIT REASON ANALYSIS
# ════════════════════════════════════════════════════════════

class TestExitReasonAnalysis:
    def test_exit_reason_breakdown(self, in_memory_db, optimizer):
        for i in range(5):
            _make_trade(f"tgt-{i}", exit_reason="TARGET_HIT", pnl=20.0)
        for i in range(5):
            _make_trade(f"stp-{i}", exit_reason="STOP_HIT", pnl=-15.0)
        report = optimizer.analyze()
        tgt = next(b for b in report.exit_reason_performance if b.label == "TARGET_HIT")
        stp = next(b for b in report.exit_reason_performance if b.label == "STOP_HIT")
        assert tgt.win_rate == 1.0
        assert stp.win_rate == 0.0


# ════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ════════════════════════════════════════════════════════════

class TestRecommendations:
    """Test parameter adjustment recommendations."""

    def test_no_recs_below_threshold(self, in_memory_db, optimizer):
        """Should not generate recommendations with < 50 trades."""
        for i in range(20):
            _make_trade(f"t-{i}", pnl=10.0)
        report = optimizer.analyze()
        assert report.recommendations == []

    def test_recs_generated_with_enough_data(self, in_memory_db, optimizer):
        """With 60+ trades and poor low-confidence performance, should recommend."""
        # Low confidence trades that lose
        for i in range(20):
            _make_trade(f"lo-{i}", confidence=50.0, pnl=-10.0, exit_reason="STOP_HIT",
                        signals={"rsi_oversold": {}})
        # High confidence trades that win
        for i in range(40):
            _make_trade(f"hi-{i}", confidence=85.0, pnl=20.0, exit_reason="TARGET_HIT",
                        signals={"ema_crossover_day": {}})
        report = optimizer.analyze()
        assert report.sufficient_data is True
        # Should have at least one recommendation (confidence threshold or stop-out rate)
        assert len(report.recommendations) >= 1

    def test_high_stop_rate_recommendation(self, in_memory_db, optimizer):
        """If >50% of trades are stop-outs, recommend wider stops."""
        for i in range(35):
            _make_trade(f"stp-{i}", exit_reason="STOP_HIT", pnl=-10.0, confidence=70.0)
        for i in range(25):
            _make_trade(f"tgt-{i}", exit_reason="TARGET_HIT", pnl=15.0, confidence=70.0)
        report = optimizer.analyze()
        stop_recs = [r for r in report.recommendations if "RISK" in r.parameter]
        assert len(stop_recs) >= 1

    def test_recommendation_dataclass(self):
        rec = Recommendation(
            parameter="TEST_PARAM",
            current_value="10",
            suggested_value="15",
            reason="Testing",
            confidence="medium",
        )
        assert rec.parameter == "TEST_PARAM"
        assert rec.confidence == "medium"


# ════════════════════════════════════════════════════════════
# PARAMETER LOGGING
# ════════════════════════════════════════════════════════════

class TestParameterLogging:
    """Test persisting recommendations to database."""

    def test_apply_recommendation(self, in_memory_db, optimizer):
        rec = Recommendation(
            parameter="SIGNAL_RSI_OVERSOLD",
            current_value="30",
            suggested_value="25",
            reason="Tighter RSI improves win rate",
        )
        optimizer.apply_recommendation(rec)

        session = get_session()
        records = session.query(StrategyParameter).all()
        assert len(records) == 1
        assert records[0].parameter_name == "SIGNAL_RSI_OVERSOLD"
        assert records[0].old_value == "30"
        assert records[0].new_value == "25"
        session.close()

    def test_get_parameter_history(self, in_memory_db, optimizer):
        for i in range(3):
            rec = Recommendation(
                parameter=f"PARAM_{i}",
                current_value=str(i),
                suggested_value=str(i + 10),
                reason=f"Reason {i}",
            )
            optimizer.apply_recommendation(rec)

        history = optimizer.get_parameter_history()
        assert len(history) == 3
        # Most recent first
        assert history[0]["parameter_name"] == "PARAM_2"

    def test_parameter_history_empty(self, in_memory_db, optimizer):
        history = optimizer.get_parameter_history()
        assert history == []

    def test_parameter_history_limit(self, in_memory_db, optimizer):
        for i in range(10):
            optimizer.apply_recommendation(Recommendation(
                parameter=f"P{i}", current_value="0", suggested_value="1", reason="test"
            ))
        history = optimizer.get_parameter_history(limit=5)
        assert len(history) == 5


# ════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions."""

    def test_all_winning_trades(self, in_memory_db, optimizer):
        for i in range(15):
            _make_trade(f"w-{i}", pnl=10.0)
        report = optimizer.analyze()
        assert report.total_trades_analyzed == 15

    def test_all_losing_trades(self, in_memory_db, optimizer):
        for i in range(15):
            _make_trade(f"l-{i}", pnl=-10.0)
        report = optimizer.analyze()
        assert report.total_trades_analyzed == 15

    def test_zero_pnl_trades(self, in_memory_db, optimizer):
        for i in range(10):
            _make_trade(f"z-{i}", pnl=0.0, r_multiple=0.0)
        report = optimizer.analyze()
        assert report.total_trades_analyzed == 10

    def test_report_dataclass_defaults(self):
        report = OptimizationReport()
        assert report.total_trades_analyzed == 0
        assert report.recommendations == []
        assert report.sufficient_data is False

    def test_analyze_with_days_filter(self, in_memory_db, optimizer):
        """Trades with old exit_time should be excluded by days filter."""
        for i in range(10):
            _make_trade(f"t-{i}", pnl=10.0)
        # exit_time in _make_trade is Jan 2025, so days=7 should exclude all
        report = optimizer.analyze(days=7)
        assert report.total_trades_analyzed == 0
        # Without filter, all should appear
        report_all = optimizer.analyze()
        assert report_all.total_trades_analyzed == 10


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_optimizer.py -v
#
# Expected results:
#   TestInsufficientData:       3 tests
#   TestIndicatorAnalysis:      3 tests
#   TestTradeTypeAnalysis:      1 test
#   TestConfidenceBands:        2 tests
#   TestExitReasonAnalysis:     1 test
#   TestRecommendations:        4 tests
#   TestParameterLogging:       4 tests
#   TestEdgeCases:              5 tests
#
# TOTAL: 23 tests
# ════════════════════════════════════════════════════════════
