"""Tests for edgefinder/agents/reasoning.py + memory.py + window check."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from edgefinder.agents.memory import DEFAULT_MEMORY, load_memory, save_memory
from edgefinder.agents.reasoning import (
    Decision,
    ReasoningResult,
    _parse_response,
    reason_over_tick,
    run_reasoning,
)
from edgefinder.agents.watchdog import is_in_active_window
from edgefinder.db.models import (
    AgentAction,
    AgentMemory,
    AgentObservation,
    StrategyAccount,
)


# ── Mock Anthropic client ───────────────────────────────────


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.content = [_FakeTextBlock(json.dumps(payload))]


class _FakeMessages:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._payload)


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self.messages = _FakeMessages(payload)


# ── Helpers ─────────────────────────────────────────────────


def _make_obs(db_session, **kwargs) -> AgentObservation:
    obs = AgentObservation(
        agent_name=kwargs.get("agent_name", "watchdog"),
        severity=kwargs.get("severity", "WARN"),
        category=kwargs.get("category", "cash_drift"),
        message=kwargs.get("message", "test"),
        obs_metadata=kwargs.get("metadata", {"key": "alpha"}),
    )
    db_session.add(obs)
    db_session.commit()
    return obs


# ── Memory ──────────────────────────────────────────────────


class TestMemory:
    def test_load_missing_creates_default(self, db_session):
        content = load_memory(db_session, "watchdog")
        assert content == DEFAULT_MEMORY
        # Persisted for future calls
        assert db_session.query(AgentMemory).count() == 1

    def test_save_then_load_roundtrip(self, db_session):
        save_memory(db_session, "watchdog", "# hello\nfoo")
        assert load_memory(db_session, "watchdog") == "# hello\nfoo"

    def test_save_updates_existing(self, db_session):
        save_memory(db_session, "watchdog", "v1")
        save_memory(db_session, "watchdog", "v2")
        assert db_session.query(AgentMemory).count() == 1
        assert load_memory(db_session, "watchdog") == "v2"

    def test_different_agents_independent(self, db_session):
        save_memory(db_session, "watchdog", "wd")
        save_memory(db_session, "strategist", "st")
        assert load_memory(db_session, "watchdog") == "wd"
        assert load_memory(db_session, "strategist") == "st"


# ── Response parsing ────────────────────────────────────────


class TestParseResponse:
    def test_valid_response(self):
        text = json.dumps({
            "summary": "one finding, escalated",
            "decisions": [{
                "observation_id": 42,
                "assessment": "critical",
                "action": "escalate",
                "reasoning": "negative cash on alpha — matches prior race bug",
            }],
            "memory_update": "# Updated memory\nnew pattern.",
        })
        result = _parse_response(text)
        assert isinstance(result, ReasoningResult)
        assert result.summary == "one finding, escalated"
        assert len(result.decisions) == 1
        assert result.decisions[0].observation_id == 42
        assert result.decisions[0].assessment == "critical"
        assert result.memory_update == "# Updated memory\nnew pattern."

    def test_null_memory_update(self):
        text = json.dumps({
            "summary": "all clean",
            "decisions": [],
            "memory_update": None,
        })
        result = _parse_response(text)
        assert result.memory_update is None


# ── LLM call with mocked client ─────────────────────────────


class TestReasonOverTick:
    def test_sends_caching_and_adaptive_thinking(self, db_session):
        obs = _make_obs(db_session, agent_name="watchdog")
        payload = {
            "summary": "test",
            "decisions": [{
                "observation_id": obs.id,
                "assessment": "expected",
                "action": "monitor",
                "reasoning": "routine",
            }],
            "memory_update": None,
        }
        client = _FakeClient(payload)

        reason_over_tick(
            observations=[obs],
            memory="# memory",
            trade_summary={"alpha": {"count": 3, "pnl": 10.0}},
            commits=["abc123 [v4.7.2] foo"],
            client=client,
            model="claude-opus-4-7",
        )

        assert len(client.messages.calls) == 1
        call = client.messages.calls[0]

        assert call["model"] == "claude-opus-4-7"
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"]["format"]["type"] == "json_schema"

        # Two cached system blocks: prompt + memory
        system = call["system"]
        assert len(system) == 2
        assert all(b.get("cache_control") == {"type": "ephemeral"} for b in system)

        # User message is JSON carrying the observation + trade summary
        user_content = call["messages"][0]["content"]
        parsed = json.loads(user_content)
        assert parsed["unresolved_observations"][0]["id"] == obs.id
        assert parsed["recent_trades_24h"]["alpha"]["count"] == 3
        assert parsed["recent_trading_commits"] == ["abc123 [v4.7.2] foo"]


# ── Orchestration ───────────────────────────────────────────


class TestRunReasoning:
    def test_no_observations_short_circuits(self, db_session):
        result = run_reasoning(db_session, client=_FakeClient({}))
        assert result.decisions == []
        assert result.memory_update is None
        # No LLM call made (client.messages would raise if it had been)

    def test_escalate_decision_records_action(self, db_session):
        obs = _make_obs(db_session)
        # Give it a starting balance row so trade summary query succeeds.
        db_session.add(StrategyAccount(
            strategy_name="alpha",
            starting_capital=5000.0,
            cash_balance=5000.0,
            open_positions_value=0.0,
            total_equity=5000.0,
            peak_equity=5000.0,
            drawdown_pct=0.0,
        ))
        db_session.commit()

        payload = {
            "summary": "escalating",
            "decisions": [{
                "observation_id": obs.id,
                "assessment": "critical",
                "action": "escalate",
                "reasoning": "Matches the v4.6.5 race class — flag for review.",
            }],
            "memory_update": None,
        }
        result = run_reasoning(db_session, client=_FakeClient(payload))

        assert len(result.decisions) == 1
        actions = db_session.query(AgentAction).all()
        assert len(actions) == 1
        assert actions[0].action_type == "diagnose"
        assert actions[0].observation_id == obs.id
        assert actions[0].status == "pending"

    def test_monitor_decision_does_not_record_action(self, db_session):
        obs = _make_obs(db_session)
        db_session.add(StrategyAccount(
            strategy_name="alpha", starting_capital=5000.0, cash_balance=5000.0,
            open_positions_value=0.0, total_equity=5000.0, peak_equity=5000.0,
            drawdown_pct=0.0,
        ))
        db_session.commit()

        payload = {
            "summary": "routine",
            "decisions": [{
                "observation_id": obs.id,
                "assessment": "expected",
                "action": "monitor",
                "reasoning": "Normal behavior.",
            }],
            "memory_update": None,
        }
        run_reasoning(db_session, client=_FakeClient(payload))
        assert db_session.query(AgentAction).count() == 0

    def test_memory_update_persisted_when_changed(self, db_session):
        obs = _make_obs(db_session)
        db_session.add(StrategyAccount(
            strategy_name="alpha", starting_capital=5000.0, cash_balance=5000.0,
            open_positions_value=0.0, total_equity=5000.0, peak_equity=5000.0,
            drawdown_pct=0.0,
        ))
        db_session.commit()

        new_memory = "# Updated\n- new pattern learned"
        payload = {
            "summary": "learned something",
            "decisions": [{
                "observation_id": obs.id,
                "assessment": "needs_investigation",
                "action": "investigate",
                "reasoning": "novel",
            }],
            "memory_update": new_memory,
        }
        run_reasoning(db_session, client=_FakeClient(payload))
        assert load_memory(db_session, "watchdog") == new_memory

    def test_null_memory_update_leaves_memory_intact(self, db_session):
        obs = _make_obs(db_session)
        db_session.add(StrategyAccount(
            strategy_name="alpha", starting_capital=5000.0, cash_balance=5000.0,
            open_positions_value=0.0, total_equity=5000.0, peak_equity=5000.0,
            drawdown_pct=0.0,
        ))
        db_session.commit()
        save_memory(db_session, "watchdog", "# original\n- important note")

        payload = {
            "summary": "nothing new",
            "decisions": [{
                "observation_id": obs.id,
                "assessment": "expected",
                "action": "monitor",
                "reasoning": "seen before",
            }],
            "memory_update": None,
        }
        run_reasoning(db_session, client=_FakeClient(payload))
        assert load_memory(db_session, "watchdog") == "# original\n- important note"


# ── Active-window check ─────────────────────────────────────


class TestActiveWindow:
    def test_monday_midday_is_active(self):
        # Monday April 13 2026, 14:00 ET → UTC 18:00
        t = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
        assert is_in_active_window(t) is True

    def test_monday_early_morning_is_inactive(self):
        # Monday 07:00 ET → UTC 11:00 (before 08:30 window start)
        t = datetime(2026, 4, 13, 11, 0, tzinfo=timezone.utc)
        assert is_in_active_window(t) is False

    def test_monday_late_evening_is_inactive(self):
        # Monday 19:00 ET → UTC 23:00 (after 17:00 window end)
        t = datetime(2026, 4, 13, 23, 0, tzinfo=timezone.utc)
        assert is_in_active_window(t) is False

    def test_saturday_is_inactive(self):
        # Saturday April 18 2026, 12:00 ET
        t = datetime(2026, 4, 18, 16, 0, tzinfo=timezone.utc)
        assert is_in_active_window(t) is False

    def test_boundary_open_is_active(self):
        # Exactly 08:30 ET on a Monday
        t = datetime(2026, 4, 13, 12, 30, tzinfo=timezone.utc)
        assert is_in_active_window(t) is True

    def test_boundary_close_is_active(self):
        # Exactly 17:00 ET on a Monday
        t = datetime(2026, 4, 13, 21, 0, tzinfo=timezone.utc)
        assert is_in_active_window(t) is True
