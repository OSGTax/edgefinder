"""
EdgeFinder Module 4 Tests: Trade Journal
=========================================
Tests cover: trade logging, skipped signal logging, querying,
statistics computation, equity curve, and summary reporting.

Run: python -m pytest tests/test_journal.py -v
"""

import pytest
from datetime import datetime, timezone

from modules.journal import TradeJournal, TradeStats, JournalEntry
from modules.trader import TradeResult, Position
from modules.database import Trade as TradeRecord, Signal as SignalRecord, AccountSnapshot, get_session


# ── FIXTURES ─────────────────────────────────────────────────

@pytest.fixture
def journal() -> TradeJournal:
    """Fresh trade journal."""
    return TradeJournal()


@pytest.fixture
def sample_trade_result() -> TradeResult:
    """A sample winning trade result."""
    return TradeResult(
        trade_id="test-001",
        ticker="AAPL",
        direction="LONG",
        trade_type="DAY",
        entry_price=100.0,
        exit_price=105.0,
        shares=10,
        stop_loss=98.0,
        target=103.0,
        pnl_dollars=50.0,
        pnl_percent=0.05,
        r_multiple=2.5,
        exit_reason="TARGET_HIT",
        entry_time=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_losing_result() -> TradeResult:
    """A sample losing trade result."""
    return TradeResult(
        trade_id="test-002",
        ticker="MSFT",
        direction="LONG",
        trade_type="SWING",
        entry_price=200.0,
        exit_price=195.0,
        shares=5,
        stop_loss=196.0,
        target=206.0,
        pnl_dollars=-25.0,
        pnl_percent=-0.025,
        r_multiple=-1.25,
        exit_reason="STOP_HIT",
        entry_time=datetime(2025, 1, 14, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 16, 10, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_position() -> Position:
    """A sample position for context."""
    return Position(
        trade_id="test-001",
        ticker="AAPL",
        direction="LONG",
        trade_type="DAY",
        entry_price=100.0,
        shares=10,
        stop_loss=98.0,
        target=103.0,
        fundamental_score=75.0,
        confidence_score=80.0,
        news_sentiment=0.3,
        technical_signals={"rsi_oversold": {"rsi": 25}},
    )


def _seed_trades(journal, in_memory_db, results):
    """Helper to seed multiple trades into the DB."""
    for r in results:
        journal.log_trade(r)


# ════════════════════════════════════════════════════════════
# TRADE LOGGING
# ════════════════════════════════════════════════════════════

class TestTradeLogging:
    """Test logging trades to the database."""

    def test_log_trade(self, in_memory_db, journal, sample_trade_result, sample_position):
        journal.log_trade(sample_trade_result, sample_position)

        session = get_session()
        records = session.query(TradeRecord).all()
        assert len(records) == 1
        assert records[0].ticker == "AAPL"
        assert records[0].pnl_dollars == 50.0
        assert records[0].fundamental_score == 75.0
        assert records[0].confidence_score == 80.0
        session.close()

    def test_log_trade_without_position(self, in_memory_db, journal, sample_trade_result):
        """Should work even without position context."""
        journal.log_trade(sample_trade_result)

        session = get_session()
        records = session.query(TradeRecord).all()
        assert len(records) == 1
        assert records[0].fundamental_score is None
        session.close()

    def test_log_multiple_trades(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)

        session = get_session()
        assert session.query(TradeRecord).count() == 2
        session.close()


# ════════════════════════════════════════════════════════════
# SKIPPED SIGNAL LOGGING
# ════════════════════════════════════════════════════════════

class TestSkippedSignalLogging:
    """Test logging skipped signals."""

    def test_log_skipped_signal(self, in_memory_db, journal):
        journal.log_skipped_signal(
            ticker="GOOGL",
            signal_type="BUY",
            trade_type="DAY",
            confidence=45.0,
            reason="Below confidence threshold",
            indicators={"rsi_oversold": {"rsi": 28}},
        )

        session = get_session()
        records = session.query(SignalRecord).all()
        assert len(records) == 1
        assert records[0].ticker == "GOOGL"
        assert records[0].was_traded is False
        assert records[0].reason_skipped == "Below confidence threshold"
        session.close()

    def test_log_skipped_no_indicators(self, in_memory_db, journal):
        journal.log_skipped_signal(
            ticker="X", signal_type="SELL", trade_type="SWING",
            confidence=30.0, reason="Long-only system",
        )
        session = get_session()
        assert session.query(SignalRecord).count() == 1
        session.close()


# ════════════════════════════════════════════════════════════
# QUERYING TRADES
# ════════════════════════════════════════════════════════════

class TestQuerying:
    """Test trade and signal querying."""

    def test_get_trades_empty(self, in_memory_db, journal):
        entries = journal.get_trades()
        assert entries == []

    def test_get_trades_returns_entries(self, in_memory_db, journal, sample_trade_result):
        journal.log_trade(sample_trade_result)
        entries = journal.get_trades()
        assert len(entries) == 1
        assert isinstance(entries[0], JournalEntry)
        assert entries[0].ticker == "AAPL"

    def test_get_trades_filter_by_ticker(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)
        entries = journal.get_trades(ticker="AAPL")
        assert len(entries) == 1
        assert entries[0].ticker == "AAPL"

    def test_get_trades_filter_by_type(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)
        entries = journal.get_trades(trade_type="SWING")
        assert len(entries) == 1
        assert entries[0].trade_type == "SWING"

    def test_get_trades_respects_limit(self, in_memory_db, journal):
        for i in range(10):
            result = TradeResult(
                trade_id=f"test-{i:03d}", ticker="X", direction="LONG",
                trade_type="DAY", entry_price=100.0, exit_price=101.0,
                shares=1, stop_loss=99.0, target=102.0,
                pnl_dollars=1.0, pnl_percent=0.01, r_multiple=1.0,
                exit_reason="TARGET_HIT",
                entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
            journal.log_trade(result)
        entries = journal.get_trades(limit=5)
        assert len(entries) == 5

    def test_get_skipped_signals(self, in_memory_db, journal):
        journal.log_skipped_signal(
            ticker="X", signal_type="BUY", trade_type="DAY",
            confidence=35.0, reason="Low confidence",
        )
        signals = journal.get_skipped_signals()
        assert len(signals) == 1
        assert signals[0]["reason_skipped"] == "Low confidence"

    def test_get_skipped_signals_filter_ticker(self, in_memory_db, journal):
        journal.log_skipped_signal(
            ticker="AAPL", signal_type="BUY", trade_type="DAY",
            confidence=35.0, reason="Reason A",
        )
        journal.log_skipped_signal(
            ticker="MSFT", signal_type="BUY", trade_type="DAY",
            confidence=40.0, reason="Reason B",
        )
        signals = journal.get_skipped_signals(ticker="AAPL")
        assert len(signals) == 1


# ════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════

class TestStatistics:
    """Test trade statistics computation."""

    def test_stats_empty(self, in_memory_db, journal):
        stats = journal.compute_stats()
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0

    def test_stats_single_winner(self, in_memory_db, journal, sample_trade_result):
        journal.log_trade(sample_trade_result)
        stats = journal.compute_stats()
        assert stats.total_trades == 1
        assert stats.winning_trades == 1
        assert stats.win_rate == 1.0
        assert stats.total_pnl == 50.0

    def test_stats_single_loser(self, in_memory_db, journal, sample_losing_result):
        journal.log_trade(sample_losing_result)
        stats = journal.compute_stats()
        assert stats.total_trades == 1
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.0
        assert stats.total_pnl == -25.0

    def test_stats_mixed(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)
        stats = journal.compute_stats()
        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.5
        assert stats.total_pnl == 25.0  # 50 - 25
        assert stats.avg_winner == 50.0
        assert stats.avg_loser == -25.0
        assert stats.largest_winner == 50.0
        assert stats.largest_loser == -25.0

    def test_profit_factor(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)
        stats = journal.compute_stats()
        assert stats.profit_factor == 2.0  # 50 / 25

    def test_profit_factor_no_losers(self, in_memory_db, journal, sample_trade_result):
        journal.log_trade(sample_trade_result)
        stats = journal.compute_stats()
        assert stats.profit_factor == float("inf")

    def test_avg_r_multiple(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)
        stats = journal.compute_stats()
        expected_avg_r = (2.5 + (-1.25)) / 2
        assert stats.avg_r_multiple == round(expected_avg_r, 2)

    def test_trade_type_breakdown(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)   # DAY
        journal.log_trade(sample_losing_result)  # SWING
        stats = journal.compute_stats()
        assert stats.day_trades == 1
        assert stats.swing_trades == 1

    def test_signal_stats(self, in_memory_db, journal):
        journal.log_skipped_signal(
            ticker="X", signal_type="BUY", trade_type="DAY",
            confidence=35.0, reason="Low confidence",
        )
        stats = journal.compute_stats()
        assert stats.total_signals == 1
        assert stats.skipped_signals == 1


# ════════════════════════════════════════════════════════════
# EQUITY CURVE
# ════════════════════════════════════════════════════════════

class TestEquityCurve:
    """Test equity curve from account snapshots."""

    def test_empty_curve(self, in_memory_db, journal):
        curve = journal.get_equity_curve()
        assert curve == []

    def test_curve_with_snapshots(self, in_memory_db, journal):
        session = get_session()
        for i in range(3):
            snap = AccountSnapshot(
                date=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                cash=2500.0 + (i * 50),
                positions_value=0.0,
                total_value=2500.0 + (i * 50),
                open_positions=0,
                peak_value=2500.0 + (i * 50),
                drawdown_pct=0.0,
            )
            session.add(snap)
        session.commit()
        session.close()

        curve = journal.get_equity_curve()
        assert len(curve) == 3
        # Should be in chronological order
        assert curve[0]["total_value"] <= curve[-1]["total_value"]

    def test_curve_respects_limit(self, in_memory_db, journal):
        session = get_session()
        for i in range(10):
            snap = AccountSnapshot(
                date=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
                cash=2500.0, positions_value=0.0, total_value=2500.0,
                open_positions=0, peak_value=2500.0, drawdown_pct=0.0,
            )
            session.add(snap)
        session.commit()
        session.close()

        curve = journal.get_equity_curve(limit=5)
        assert len(curve) == 5


# ════════════════════════════════════════════════════════════
# SUMMARY REPORTING
# ════════════════════════════════════════════════════════════

class TestSummaryReporting:
    """Test the human-readable summary generation."""

    def test_summary_empty(self, in_memory_db, journal):
        summary = journal.print_summary()
        assert "TRADE JOURNAL" in summary
        assert "Total trades" in summary

    def test_summary_with_trades(self, in_memory_db, journal, sample_trade_result, sample_losing_result):
        journal.log_trade(sample_trade_result)
        journal.log_trade(sample_losing_result)
        summary = journal.print_summary()
        assert "$25.00" in summary or "25.00" in summary  # Total P&L
        assert "50.0%" in summary  # Win rate

    def test_summary_with_days_filter(self, in_memory_db, journal):
        summary = journal.print_summary(days=7)
        assert "Last 7 days" in summary


# ════════════════════════════════════════════════════════════
# EDGE CASES
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions."""

    def test_breakeven_trade_stats(self, in_memory_db, journal):
        result = TradeResult(
            trade_id="be-001", ticker="X", direction="LONG",
            trade_type="DAY", entry_price=100.0, exit_price=100.0,
            shares=10, stop_loss=98.0, target=103.0,
            pnl_dollars=0.0, pnl_percent=0.0, r_multiple=0.0,
            exit_reason="MANUAL",
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        journal.log_trade(result)
        stats = journal.compute_stats()
        assert stats.breakeven_trades == 1
        assert stats.total_pnl == 0.0

    def test_profit_factor_all_losers(self, in_memory_db, journal, sample_losing_result):
        journal.log_trade(sample_losing_result)
        stats = journal.compute_stats()
        assert stats.profit_factor == 0.0

    def test_trade_stats_dataclass_defaults(self):
        stats = TradeStats()
        assert stats.total_trades == 0
        assert stats.profit_factor == 0.0
        assert stats.avg_r_multiple == 0.0


# ════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration: full pipeline from trader → journal."""

    def test_trader_to_journal_pipeline(self, in_memory_db):
        """Open trade via trader, close it, log to journal, query stats."""
        from modules.trader import PaperTrader

        trader = PaperTrader()
        journal = TradeJournal()

        # Open position
        pos = trader.open_position(
            ticker="AAPL", entry_price=100.0, stop_loss=98.0,
            target=103.0, shares=10, trade_type="DAY",
            sector="Technology",
        )
        assert pos is not None

        # Close with profit
        result = trader.close_position(pos.trade_id, exit_price=103.0, exit_reason="TARGET_HIT")
        assert result is not None

        # Log to journal
        journal.log_trade(result, pos)

        # Query
        entries = journal.get_trades()
        assert len(entries) == 1
        assert entries[0].ticker == "AAPL"
        assert entries[0].pnl_dollars == 30.0

        # Stats
        stats = journal.compute_stats()
        assert stats.total_trades == 1
        assert stats.win_rate == 1.0
        assert stats.total_pnl == 30.0

        # Summary
        summary = journal.print_summary()
        assert "AAPL" not in summary  # Summary is aggregate, not per-ticker
        assert "1" in summary  # Total trades


# ════════════════════════════════════════════════════════════
# TEST RESULTS SUMMARY
# ════════════════════════════════════════════════════════════
#
# Run: python -m pytest tests/test_journal.py -v
#
# Expected results:
#   TestTradeLogging:           3 tests  — all should PASS
#   TestSkippedSignalLogging:   2 tests  — all should PASS
#   TestQuerying:               7 tests  — all should PASS
#   TestStatistics:             9 tests  — all should PASS
#   TestEquityCurve:            3 tests  — all should PASS
#   TestSummaryReporting:       3 tests  — all should PASS
#   TestEdgeCases:              3 tests  — all should PASS
#   TestIntegration:            1 test   — should PASS
#
# TOTAL: 31 tests
# ════════════════════════════════════════════════════════════
