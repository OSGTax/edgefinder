"""Tests for the analytics module: regime tagger, trade analytics, conditional stats."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from edgefinder.analytics.conditional_stats import (
    ConditionStats,
    StrategyProfile,
    build_strategy_profiles,
    compute_condition_stats,
    get_best_strategy_for_regime,
    get_strategy_scores_for_regime,
)
from edgefinder.analytics.regime import (
    MarketCondition,
    classify_regime,
)
from edgefinder.analytics.trade_analytics import (
    TradeFeatures,
    _extract_signal_patterns,
    build_trade_features,
)
from edgefinder.db.models import MarketSnapshotRecord, TradeRecord


# ── Regime tagger tests ─────────────────────────────────


class TestClassifyRegime:
    def test_bull_calm(self):
        result = classify_regime(vix_level=15.0, spy_change_pct=0.5)
        assert result == MarketCondition.BULL_CALM

    def test_bull_volatile(self):
        result = classify_regime(vix_level=25.0, spy_change_pct=0.5)
        assert result == MarketCondition.BULL_VOLATILE

    def test_bear_calm(self):
        result = classify_regime(vix_level=15.0, spy_change_pct=-0.5)
        assert result == MarketCondition.BEAR_CALM

    def test_bear_volatile(self):
        result = classify_regime(vix_level=30.0, spy_change_pct=-1.0)
        assert result == MarketCondition.BEAR_VOLATILE

    def test_sideways_calm(self):
        result = classify_regime(vix_level=15.0, spy_change_pct=0.1)
        assert result == MarketCondition.SIDEWAYS_CALM

    def test_sideways_volatile(self):
        result = classify_regime(vix_level=25.0, spy_change_pct=0.0)
        assert result == MarketCondition.SIDEWAYS_VOLATILE

    def test_regime_string_fallback_bull(self):
        result = classify_regime(vix_level=15.0, spy_change_pct=0.0, market_regime="bull")
        assert result == MarketCondition.BULL_CALM

    def test_regime_string_fallback_bear(self):
        result = classify_regime(vix_level=25.0, spy_change_pct=0.0, market_regime="bear")
        assert result == MarketCondition.BEAR_VOLATILE

    def test_vix_boundary_20_is_volatile(self):
        result = classify_regime(vix_level=20.0, spy_change_pct=0.5)
        assert result == MarketCondition.BULL_VOLATILE

    def test_vix_just_below_20_is_calm(self):
        result = classify_regime(vix_level=19.9, spy_change_pct=0.5)
        assert result == MarketCondition.BULL_CALM


# ── Signal pattern extraction tests ─────────────────────


class TestExtractSignalPatterns:
    def test_none_returns_empty(self):
        assert _extract_signal_patterns(None) == []

    def test_empty_dict_returns_empty(self):
        assert _extract_signal_patterns({}) == []

    def test_patterns_key(self):
        result = _extract_signal_patterns({"patterns": ["ema_crossover_bullish", "rsi_oversold"]})
        assert result == ["ema_crossover_bullish", "rsi_oversold"]

    def test_boolean_flags(self):
        result = _extract_signal_patterns({
            "ema_crossover_bullish": True,
            "rsi_oversold": False,
            "macd_bullish_cross": True,
        })
        assert "ema_crossover_bullish" in result
        assert "macd_bullish_cross" in result
        assert "rsi_oversold" not in result


# ── Trade features from DB tests ────────────────────────


class TestBuildTradeFeatures:
    def _make_snapshot(self, db_session, vix=18.0, spy_chg=0.3, regime="bull"):
        snap = MarketSnapshotRecord(
            timestamp=datetime.now(timezone.utc),
            spy_price=450.0, spy_change_pct=spy_chg,
            qqq_price=380.0, qqq_change_pct=0.2,
            iwm_price=200.0, iwm_change_pct=0.1,
            dia_price=350.0, dia_change_pct=0.15,
            vix_level=vix, market_regime=regime,
        )
        db_session.add(snap)
        db_session.flush()
        return snap.id

    def _make_trade(self, db_session, strategy="alpha", pnl=50.0, snap_id=None):
        now = datetime.now(timezone.utc)
        trade = TradeRecord(
            trade_id=f"test-{strategy}-{pnl}-{id(pnl)}",
            strategy_name=strategy,
            symbol="AAPL",
            direction="LONG",
            trade_type="DAY",
            entry_price=150.0,
            shares=10,
            stop_loss=145.0,
            target=160.0,
            confidence=75.0,
            entry_time=now - timedelta(hours=2),
            exit_time=now,
            status="CLOSED",
            pnl_dollars=pnl,
            pnl_percent=pnl / (150.0 * 10) * 100,
            r_multiple=pnl / (5.0 * 10),
            exit_reason="TARGET_HIT" if pnl > 0 else "STOP_HIT",
            market_snapshot_id=snap_id,
        )
        db_session.add(trade)
        db_session.flush()
        return trade

    def test_empty_db_returns_empty(self, db_session):
        features = build_trade_features(db_session)
        assert features == []

    def test_builds_features_with_snapshot(self, db_session):
        snap_id = self._make_snapshot(db_session)
        self._make_trade(db_session, pnl=50.0, snap_id=snap_id)
        db_session.commit()

        features = build_trade_features(db_session)
        assert len(features) == 1
        f = features[0]
        assert f.won is True
        assert f.pnl_dollars == 50.0
        assert f.strategy_name == "alpha"
        assert f.regime == MarketCondition.BULL_CALM  # vix=18 < 20, spy_chg=0.3

    def test_trade_without_snapshot_gets_default_regime(self, db_session):
        self._make_trade(db_session, pnl=-30.0, snap_id=None)
        db_session.commit()

        features = build_trade_features(db_session)
        assert len(features) == 1
        assert features[0].regime == MarketCondition.SIDEWAYS_CALM

    def test_hold_duration_computed(self, db_session):
        self._make_trade(db_session, pnl=20.0)
        db_session.commit()

        features = build_trade_features(db_session)
        assert features[0].hold_minutes == pytest.approx(120.0, abs=1.0)

    def test_only_closed_trades_included(self, db_session):
        now = datetime.now(timezone.utc)
        open_trade = TradeRecord(
            trade_id="open-trade",
            strategy_name="alpha",
            symbol="MSFT",
            direction="LONG",
            trade_type="DAY",
            entry_price=300.0,
            shares=5,
            stop_loss=295.0,
            target=310.0,
            confidence=60.0,
            entry_time=now,
            status="OPEN",
        )
        db_session.add(open_trade)
        db_session.commit()

        features = build_trade_features(db_session)
        assert len(features) == 0


# ── Conditional stats tests ──────────────────────────────


def _make_features(strategy, regime, wins, losses):
    """Helper to create TradeFeatures for testing."""
    features = []
    for i in range(wins):
        features.append(TradeFeatures(
            trade_id=f"{strategy}-win-{i}",
            strategy_name=strategy,
            symbol="AAPL",
            direction="LONG",
            trade_type="DAY",
            won=True,
            pnl_dollars=100.0,
            pnl_percent=2.0,
            r_multiple=2.0,
            exit_reason="TARGET_HIT",
            hold_minutes=60.0,
            regime=regime,
            vix_level=15.0,
            spy_change_pct=0.5,
            confidence=75.0,
        ))
    for i in range(losses):
        features.append(TradeFeatures(
            trade_id=f"{strategy}-loss-{i}",
            strategy_name=strategy,
            symbol="AAPL",
            direction="LONG",
            trade_type="DAY",
            won=False,
            pnl_dollars=-50.0,
            pnl_percent=-1.0,
            r_multiple=-1.0,
            exit_reason="STOP_HIT",
            hold_minutes=30.0,
            regime=regime,
            vix_level=15.0,
            spy_change_pct=0.5,
            confidence=60.0,
        ))
    return features


class TestComputeConditionStats:
    def test_by_regime(self):
        trades = _make_features("alpha", MarketCondition.BULL_CALM, 7, 3)
        stats = compute_condition_stats(trades, "alpha", "regime")
        assert "bull_calm" in stats
        s = stats["bull_calm"]
        assert s.total_trades == 10
        assert s.wins == 7
        assert s.win_rate == 0.7
        assert s.is_reliable is True

    def test_unreliable_with_few_trades(self):
        trades = _make_features("alpha", MarketCondition.BULL_CALM, 2, 1)
        stats = compute_condition_stats(trades, "alpha", "regime")
        s = stats["bull_calm"]
        assert s.total_trades == 3
        assert s.is_reliable is False

    def test_expectancy_positive_for_winning_strategy(self):
        trades = _make_features("alpha", MarketCondition.BULL_CALM, 8, 2)
        stats = compute_condition_stats(trades, "alpha", "regime")
        assert stats["bull_calm"].expectancy > 0

    def test_expectancy_negative_for_losing_strategy(self):
        trades = _make_features("bravo", MarketCondition.BEAR_VOLATILE, 1, 9)
        stats = compute_condition_stats(trades, "bravo", "regime")
        assert stats["bear_volatile"].expectancy < 0

    def test_by_exit_reason(self):
        trades = _make_features("alpha", MarketCondition.BULL_CALM, 5, 5)
        stats = compute_condition_stats(trades, "alpha", "exit_reason")
        assert "TARGET_HIT" in stats
        assert "STOP_HIT" in stats
        assert stats["TARGET_HIT"].wins == 5
        assert stats["STOP_HIT"].losses == 5


class TestBuildStrategyProfiles:
    def test_builds_profiles_for_multiple_strategies(self):
        features = (
            _make_features("alpha", MarketCondition.BULL_CALM, 7, 3)
            + _make_features("bravo", MarketCondition.BULL_CALM, 4, 6)
        )
        profiles = build_strategy_profiles(features)
        assert "alpha" in profiles
        assert "bravo" in profiles
        assert profiles["alpha"].overall_win_rate == 0.7
        assert profiles["bravo"].overall_win_rate == 0.4

    def test_profile_has_regime_breakdown(self):
        features = (
            _make_features("alpha", MarketCondition.BULL_CALM, 5, 1)
            + _make_features("alpha", MarketCondition.BEAR_VOLATILE, 2, 4)
        )
        profiles = build_strategy_profiles(features)
        p = profiles["alpha"]
        assert "bull_calm" in p.by_regime
        assert "bear_volatile" in p.by_regime
        assert p.by_regime["bull_calm"].win_rate > p.by_regime["bear_volatile"].win_rate


class TestGetBestStrategyForRegime:
    def test_returns_best_by_expectancy(self):
        features = (
            _make_features("alpha", MarketCondition.BULL_CALM, 8, 2)
            + _make_features("bravo", MarketCondition.BULL_CALM, 3, 7)
        )
        profiles = build_strategy_profiles(features)
        best = get_best_strategy_for_regime(profiles, MarketCondition.BULL_CALM)
        assert best == "alpha"

    def test_returns_none_for_unreliable_data(self):
        features = _make_features("alpha", MarketCondition.BULL_CALM, 2, 1)
        profiles = build_strategy_profiles(features)
        best = get_best_strategy_for_regime(profiles, MarketCondition.BULL_CALM)
        assert best is None  # only 3 trades, below MIN_TRADES_FOR_CONFIDENCE

    def test_returns_none_for_unknown_regime(self):
        features = _make_features("alpha", MarketCondition.BULL_CALM, 8, 2)
        profiles = build_strategy_profiles(features)
        best = get_best_strategy_for_regime(profiles, MarketCondition.BEAR_VOLATILE)
        assert best is None

    def test_excludes_echo_strategy(self):
        features = (
            _make_features("echo", MarketCondition.BULL_CALM, 10, 0)
            + _make_features("alpha", MarketCondition.BULL_CALM, 6, 4)
        )
        profiles = build_strategy_profiles(features)
        best = get_best_strategy_for_regime(profiles, MarketCondition.BULL_CALM)
        assert best == "alpha"  # echo excluded even though it's "better"


class TestGetStrategyScoresForRegime:
    def test_sorted_by_expectancy(self):
        features = (
            _make_features("alpha", MarketCondition.BULL_CALM, 8, 2)
            + _make_features("bravo", MarketCondition.BULL_CALM, 3, 7)
            + _make_features("charlie", MarketCondition.BULL_CALM, 6, 4)
        )
        profiles = build_strategy_profiles(features)
        scores = get_strategy_scores_for_regime(profiles, MarketCondition.BULL_CALM)
        names = [s[0] for s in scores]
        assert names[0] == "alpha"  # highest expectancy

    def test_excludes_echo(self):
        features = (
            _make_features("echo", MarketCondition.BULL_CALM, 10, 0)
            + _make_features("alpha", MarketCondition.BULL_CALM, 6, 4)
        )
        profiles = build_strategy_profiles(features)
        scores = get_strategy_scores_for_regime(profiles, MarketCondition.BULL_CALM)
        names = [s[0] for s in scores]
        assert "echo" not in names
