"""Tests for edgefinder/agents/watchdog.py — invariant checks + reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from edgefinder.agents.watchdog import (
    RECONCILE_COMPONENT_PREFIX,
    check_account_paused,
    check_cash_drift,
    check_cycle_liveness,
    check_high_drawdown,
    check_negative_cash,
    check_real_book_reconciliation,
    persist_checks,
    run_checks,
)
from edgefinder.db.models import (
    AgentObservation,
    PromotedStrategy,
    StrategyAccount,
    SystemHeartbeat,
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

    def test_dividend_credits_are_not_drift(self, db_session):
        # v2 extension: cash legitimately includes dividend credits.
        # Without the credits term this would flag 4% drift.
        from datetime import date

        from edgefinder.db.models import DividendCredit

        _make_account(db_session, "alpha", cash=5200.0)
        db_session.add(DividendCredit(
            strategy_name="alpha", symbol="SPY", ex_date=date(2026, 3, 20),
            shares=100, amount=200.0))
        db_session.commit()
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
        specs = run_checks(db_session, {"liveness_enabled": False})
        assert len(specs) == 1
        summary = persist_checks(db_session, specs, agent_name="watchdog")
        assert summary == {"new": 1, "kept": 0, "resolved": 0}
        assert db_session.query(AgentObservation).count() == 1

    def test_persisting_same_spec_twice_does_not_duplicate(self, db_session):
        _make_account(db_session, "alpha", paused=True)
        specs = run_checks(db_session, {"liveness_enabled": False})

        first = persist_checks(db_session, specs, agent_name="watchdog")
        assert first["new"] == 1

        second = persist_checks(db_session, specs, agent_name="watchdog")
        assert second == {"new": 0, "kept": 1, "resolved": 0}
        assert db_session.query(AgentObservation).count() == 1

    def test_cleared_condition_auto_resolves(self, db_session):
        account = _make_account(db_session, "alpha", paused=True)
        persist_checks(db_session, run_checks(db_session, {"liveness_enabled": False}), agent_name="watchdog")

        # Fix the condition — unpause.
        account.is_paused = False
        db_session.commit()

        summary = persist_checks(
            db_session, run_checks(db_session, {"liveness_enabled": False}), agent_name="watchdog"
        )
        assert summary["resolved"] == 1

        obs = db_session.query(AgentObservation).one()
        assert obs.resolved_at is not None
        assert obs.resolved_by == "watchdog"

    def test_different_agents_do_not_interfere(self, db_session):
        _make_account(db_session, "alpha", paused=True)
        persist_checks(
            db_session, run_checks(db_session, {"liveness_enabled": False}), agent_name="watchdog"
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
            db_session, run_checks(db_session, {"liveness_enabled": False}), agent_name="watchdog"
        )
        assert summary == {"new": 0, "kept": 1, "resolved": 0}

    def test_multiple_conditions_each_get_own_observation(self, db_session):
        # A single account can legitimately trigger multiple checks.
        _make_account(db_session, "alpha", cash=-10.0)
        specs = run_checks(db_session, {"liveness_enabled": False})
        # cash_drift + negative_cash both fire on this account.
        categories = {s.category for s in specs}
        assert "cash_drift" in categories
        assert "negative_cash" in categories

        summary = persist_checks(db_session, specs, agent_name="watchdog")
        assert summary["new"] == len(specs)
        assert db_session.query(AgentObservation).count() == len(specs)


# ── cycle liveness ──────────────────────────────────────────

# Wed 2026-06-03 15:00 UTC = 11:00 ET (EDT) — a weekday after the 10:00 ET
# check-window opens (the v2 cycle fires daily at 9:45 ET).
_MID_SESSION = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)


def _make_heartbeat(
    session,
    component: str = "v2_portfolio_cycle",
    age_hours: float = 1.0,
    ok: bool = True,
    detail: dict | None = None,
    now: datetime = _MID_SESSION,
) -> SystemHeartbeat:
    hb = SystemHeartbeat(
        component=component,
        last_run_at=now - timedelta(hours=age_hours),
        ok=ok,
        detail=detail or {},
    )
    session.add(hb)
    session.commit()
    return hb


class TestCheckCycleLiveness:
    def test_fresh_heartbeat_is_healthy(self, db_session):
        _make_heartbeat(db_session, age_hours=1.0)
        assert check_cycle_liveness(db_session, 26, now=_MID_SESSION) == []

    def test_yesterdays_heartbeat_is_still_healthy(self, db_session):
        # ~24h old (yesterday's daily cycle) must NOT alarm — 26h tolerates
        # normal daily jitter.
        _make_heartbeat(db_session, age_hours=24.5)
        assert check_cycle_liveness(db_session, 26, now=_MID_SESSION) == []

    def test_stale_heartbeat_is_critical(self, db_session):
        # older than 26h on a weekday after 10:00 ET = a fully missed day
        _make_heartbeat(db_session, age_hours=30.0)
        specs = check_cycle_liveness(db_session, 26, now=_MID_SESSION)
        assert len(specs) == 1
        assert specs[0].severity == "CRITICAL"
        assert specs[0].category == "cycle_liveness"
        assert specs[0].dedup_key == ("cycle_liveness", "v2_portfolio_cycle")
        assert specs[0].metadata["reason"] == "stale"

    def test_stale_on_weekend_is_silent(self, db_session):
        _make_heartbeat(db_session, age_hours=60.0)
        saturday = datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)
        assert check_cycle_liveness(db_session, 26, now=saturday) == []

    def test_before_check_window_is_silent(self, db_session):
        # 13:30 UTC = 09:30 ET (EDT) — before 10:00 ET, the cycle (9:45) may
        # legitimately not have run yet today.
        early = datetime(2026, 6, 3, 13, 30, tzinfo=timezone.utc)
        _make_heartbeat(db_session, age_hours=30.0, now=early)
        assert check_cycle_liveness(db_session, 26, now=early) == []

    def test_errored_run_is_critical(self, db_session):
        _make_heartbeat(
            db_session, age_hours=1.0, ok=False,
            detail={"error": "ValueError: boom"},
        )
        specs = check_cycle_liveness(db_session, 26, now=_MID_SESSION)
        assert len(specs) == 1
        assert specs[0].metadata["reason"] == "error"
        assert "boom" in specs[0].message

    def test_controlled_skip_reads_healthy(self, db_session):
        # A controlled skip (holiday / nothing promoted) writes a *fresh*
        # ok=True heartbeat → not a stall, so the watchdog needs no holiday
        # calendar of its own.
        _make_heartbeat(
            db_session, age_hours=0.5, ok=True,
            detail={"skip": "none promoted"},
        )
        assert check_cycle_liveness(db_session, 26, now=_MID_SESSION) == []

    def test_missing_heartbeat_is_critical(self, db_session):
        specs = check_cycle_liveness(db_session, 26, now=_MID_SESSION)
        assert len(specs) == 1
        assert specs[0].metadata["reason"] == "missing"

    def test_reconciliation_resolves_when_fresh_again(self, db_session):
        _make_heartbeat(db_session, age_hours=30.0)
        specs = check_cycle_liveness(db_session, 26, now=_MID_SESSION)
        summary = persist_checks(db_session, specs, agent_name="watchdog")
        assert summary["new"] == 1

        hb = db_session.query(SystemHeartbeat).one()
        hb.last_run_at = _MID_SESSION - timedelta(hours=1)
        db_session.commit()
        specs2 = check_cycle_liveness(db_session, 26, now=_MID_SESSION)
        summary2 = persist_checks(db_session, specs2, agent_name="watchdog")
        assert summary2["resolved"] == 1


class TestRealBookReconciliation:
    def _live_book(self, session, name="live_core"):
        session.add(PromotedStrategy(
            strategy_name=name, spec="growth_value_barbell",
            execution_mode="live_manual", active=True))
        session.commit()

    def _reconcile_hb(self, session, name, ok, detail=None):
        session.add(SystemHeartbeat(
            component=RECONCILE_COMPONENT_PREFIX + name,
            last_run_at=datetime.now(timezone.utc), ok=ok, detail=detail or {}))
        session.commit()

    def test_no_live_books_is_noop(self, db_session):
        assert check_real_book_reconciliation(db_session) == []

    def test_clean_reconcile_not_flagged(self, db_session):
        self._live_book(db_session)
        self._reconcile_hb(db_session, "live_core", ok=True)
        assert check_real_book_reconciliation(db_session) == []

    def test_never_reconciled_not_flagged(self, db_session):
        # a freshly created live book may not be funded/reconciled yet
        self._live_book(db_session)
        assert check_real_book_reconciliation(db_session) == []

    def test_recorded_mismatch_is_critical(self, db_session):
        self._live_book(db_session)
        self._reconcile_hb(db_session, "live_core", ok=False,
                           detail={"summary": "SPY: db=5 broker=4"})
        specs = check_real_book_reconciliation(db_session)
        assert len(specs) == 1
        assert specs[0].severity == "CRITICAL"
        assert "SPY" in specs[0].message

    def test_paper_book_ignored(self, db_session):
        db_session.add(PromotedStrategy(
            strategy_name="paper_book", spec="equal_weight",
            execution_mode="paper", active=True))
        self._reconcile_hb(db_session, "paper_book", ok=False)
        db_session.commit()
        assert check_real_book_reconciliation(db_session) == []
