"""Tests for edgefinder/agents/watchdog.py — invariant checks + reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from edgefinder.agents.watchdog import (
    check_account_paused,
    check_cash_drift,
    check_high_drawdown,
    check_negative_cash,
    persist_checks,
    run_checks,
)
from edgefinder.db.models import (
    AgentObservation,
    StrategyAccount,
    TradeRecord,
)


def _make_account(
    session,
    name: str,
    cash: float = 5000.0,
    starting: float = 5000.0,
    peak: float = 5000.0,
    drawdown: float = 0.0,
    paused: bool = False,
) -> StrategyAccount:
    account = StrategyAccount(
        strategy_name=name,
        starting_capital=starting,
        cash_balance=cash,
        open_positions_value=0.0,
        total_equity=cash,
        peak_equity=peak,
        drawdown_pct=drawdown,
        realized_pnl=0.0,
        is_paused=paused,
    )
    session.add(account)
    session.commit()
    return account


def _make_trade(
    session,
    strategy: str,
    symbol: str,
    status: str,
    entry: float,
    shares: int,
    pnl: float | None = None,
    trade_id: str | None = None,
) -> TradeRecord:
    record = TradeRecord(
        trade_id=trade_id or f"{strategy}-{symbol}-{status}-{entry}",
        strategy_name=strategy,
        symbol=symbol,
        direction="LONG",
        trade_type="SWING",
        entry_price=entry,
        shares=shares,
        stop_loss=entry * 0.95,
        target=entry * 1.10,
        confidence=0.7,
        entry_time=datetime(2026, 4, 10, tzinfo=timezone.utc),
        status=status,
        pnl_dollars=pnl,
    )
    session.add(record)
    session.commit()
    return record


class TestCheckCashDrift:
    def test_matching_cash_has_no_drift(self, db_session):
        _make_account(db_session, "alpha", cash=5000.0)
        specs = check_cash_drift(db_session, drift_threshold_pct=0.01)
        assert specs == []

    def test_drift_above_threshold_is_flagged(self, db_session):
        # Closed trade +$100 profit → correct_cash = 5000 + 100 = 5100.
        # Stored cash = 4000 ⇒ diff = -1100 ⇒ 22% of starting.
        _make_account(db_session, "alpha", cash=4000.0)
        _make_trade(db_session, "alpha", "AAPL", "CLOSED", 150.0, 10, pnl=100.0)
        specs = check_cash_drift(db_session, drift_threshold_pct=0.01)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.severity == "CRITICAL"  # 22% > 10%
        assert spec.category == "cash_drift"
        assert spec.dedup_key == ("cash_drift", "alpha")
        assert spec.metadata["diff"] == pytest.approx(-1100.0, abs=1e-2)

    def test_open_position_reduces_correct_cash(self, db_session):
        # Open position cost = 150*10 = 1500 ⇒ correct_cash = 5000 - 1500 = 3500.
        _make_account(db_session, "alpha", cash=3500.0)
        _make_trade(db_session, "alpha", "AAPL", "OPEN", 150.0, 10)
        specs = check_cash_drift(db_session, drift_threshold_pct=0.01)
        assert specs == []  # in sync

    def test_small_drift_below_threshold_is_ignored(self, db_session):
        # 0.5% drift vs 1% threshold
        _make_account(db_session, "alpha", cash=4975.0)
        specs = check_cash_drift(db_session, drift_threshold_pct=0.01)
        assert specs == []


class TestCheckNegativeCash:
    def test_positive_cash_not_flagged(self, db_session):
        _make_account(db_session, "alpha", cash=1.0)
        assert check_negative_cash(db_session) == []

    def test_negative_cash_is_critical(self, db_session):
        _make_account(db_session, "alpha", cash=-50.0)
        specs = check_negative_cash(db_session)
        assert len(specs) == 1
        assert specs[0].severity == "CRITICAL"
        assert specs[0].dedup_key == ("negative_cash", "alpha")


class TestCheckAccountPaused:
    def test_running_account_not_flagged(self, db_session):
        _make_account(db_session, "alpha", paused=False)
        assert check_account_paused(db_session) == []

    def test_paused_account_flagged(self, db_session):
        _make_account(db_session, "alpha", paused=True, drawdown=0.25)
        specs = check_account_paused(db_session)
        assert len(specs) == 1
        assert specs[0].severity == "WARN"
        assert specs[0].dedup_key == ("account_paused", "alpha")


class TestCheckHighDrawdown:
    def test_below_warn_not_flagged(self, db_session):
        _make_account(db_session, "alpha", drawdown=0.10)
        assert check_high_drawdown(db_session, 0.15, 0.18) == []

    def test_above_warn_flagged(self, db_session):
        _make_account(db_session, "alpha", drawdown=0.16)
        specs = check_high_drawdown(db_session, 0.15, 0.18)
        assert len(specs) == 1
        assert specs[0].severity == "WARN"

    def test_above_critical_flagged(self, db_session):
        _make_account(db_session, "alpha", drawdown=0.19)
        specs = check_high_drawdown(db_session, 0.15, 0.18)
        assert specs[0].severity == "CRITICAL"

    def test_paused_account_skipped(self, db_session):
        # Paused accounts surface via check_account_paused — no double-report.
        _make_account(db_session, "alpha", drawdown=0.22, paused=True)
        assert check_high_drawdown(db_session, 0.15, 0.18) == []


class TestPersistChecks:
    def test_new_observation_inserted(self, db_session):
        # paused-only scenario: cash matches correct_cash and drawdown=0
        # so this triggers exactly one check.
        _make_account(db_session, "alpha", paused=True)
        specs = run_checks(db_session, {})
        assert len(specs) == 1
        summary = persist_checks(db_session, specs, agent_name="watchdog")
        assert summary == {"new": 1, "kept": 0, "resolved": 0}
        assert db_session.query(AgentObservation).count() == 1

    def test_persisting_same_spec_twice_does_not_duplicate(self, db_session):
        _make_account(db_session, "alpha", paused=True)
        specs = run_checks(db_session, {})

        first = persist_checks(db_session, specs, agent_name="watchdog")
        assert first["new"] == 1

        second = persist_checks(db_session, specs, agent_name="watchdog")
        assert second == {"new": 0, "kept": 1, "resolved": 0}
        assert db_session.query(AgentObservation).count() == 1

    def test_cleared_condition_auto_resolves(self, db_session):
        account = _make_account(db_session, "alpha", paused=True)
        persist_checks(db_session, run_checks(db_session, {}), agent_name="watchdog")

        # Fix the condition — unpause.
        account.is_paused = False
        db_session.commit()

        summary = persist_checks(
            db_session, run_checks(db_session, {}), agent_name="watchdog"
        )
        assert summary["resolved"] == 1

        obs = db_session.query(AgentObservation).one()
        assert obs.resolved_at is not None
        assert obs.resolved_by == "watchdog"

    def test_different_agents_do_not_interfere(self, db_session):
        _make_account(db_session, "alpha", paused=True)
        persist_checks(
            db_session, run_checks(db_session, {}), agent_name="watchdog"
        )
        # Another agent records something else — should not affect
        # watchdog's reconciliation.
        other = AgentObservation(
            agent_name="strategist",
            severity="INFO",
            category="proposal",
            message="whatever",
            obs_metadata={"key": "alpha"},
        )
        db_session.add(other)
        db_session.commit()

        summary = persist_checks(
            db_session, run_checks(db_session, {}), agent_name="watchdog"
        )
        assert summary == {"new": 0, "kept": 1, "resolved": 0}

    def test_multiple_conditions_each_get_own_observation(self, db_session):
        # A single account can legitimately trigger multiple checks.
        _make_account(db_session, "alpha", cash=-10.0)
        specs = run_checks(db_session, {})
        # cash_drift + negative_cash both fire on this account.
        categories = {s.category for s in specs}
        assert "cash_drift" in categories
        assert "negative_cash" in categories

        summary = persist_checks(db_session, specs, agent_name="watchdog")
        assert summary["new"] == len(specs)
        assert db_session.query(AgentObservation).count() == len(specs)
