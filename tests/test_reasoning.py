"""Tests for edgefinder/agents/reasoning.py + memory.py + window check."""

from __future__ import annotations

import json
from datetime import datetime, timezone

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


# ── Mock subprocess runner ──────────────────────────────────


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RecordingRunner:
    """Stand-in for subprocess.run — captures args + returns a fixed result."""

    def __init__(self, result: _FakeCompletedProcess):
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        return self.result


def _envelope(payload: dict, wrapping: str | None = None) -> str:
    """Build the `claude -p --output-format json` envelope."""
    inner = json.dumps(payload)
    if wrapping:
        inner = wrapping.replace("{INNER}", inner)
    return json.dumps({"type": "result", "result": inner})


def _runner_for(payload: dict) -> _RecordingRunner:
    return _RecordingRunner(_FakeCompletedProcess(stdout=_envelope(payload)))


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


@pytest.fixture(autouse=True)
def _fake_oauth_token(monkeypatch):
    """All reasoning tests need CLAUDE_CODE_OAUTH_TOKEN to pass the env check."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-token-for-tests")


# ── Memory ──────────────────────────────────────────────────


class TestMemory:
    def test_load_missing_creates_default(self, db_session):
        content = load_memory(db_session, "watchdog")
        assert content == DEFAULT_MEMORY
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


# ── CLI transport ───────────────────────────────────────────


class TestCliTransport:
    def test_builds_claude_cli_command(self, db_session):
        obs = _make_obs(db_session)
        payload = {
            "summary": "via cli",
            "decisions": [{
                "observation_id": obs.id,
                "assessment": "expected",
                "action": "monitor",
                "reasoning": "routine",
            }],
            "memory_update": None,
        }
        runner = _runner_for(payload)

        result = reason_over_tick(
            observations=[obs],
            memory="# mem",
            cli_runner=runner,
            model="claude-opus-4-7",
        )

        assert result.decisions[0].observation_id == obs.id
        assert len(runner.calls) == 1
        call = runner.calls[0]
        # Command is `claude -p --model <M> --output-format json`
        assert call["cmd"][0] == "claude"
        assert "-p" in call["cmd"]
        assert "--output-format" in call["cmd"] and "json" in call["cmd"]
        assert "--model" in call["cmd"] and "claude-opus-4-7" in call["cmd"]
        # Prompt is piped via stdin (argv length limits).
        assert isinstance(call["input"], str) and len(call["input"]) > 100

    def test_cli_failure_raises(self, db_session):
        obs = _make_obs(db_session)
        runner = _RecordingRunner(_FakeCompletedProcess(returncode=1, stderr="auth failed"))
        with pytest.raises(RuntimeError, match="auth failed"):
            reason_over_tick(
                observations=[obs],
                memory="",
                cli_runner=runner,
            )

    def test_cli_extracts_json_from_wrapping_prose(self, db_session):
        obs = _make_obs(db_session)
        payload = {"summary": "ok", "decisions": [], "memory_update": None}
        # Claude sometimes wraps JSON in apology/explanation — the CLI
        # path must extract the first balanced JSON object.
        wrapped = f"Sure, here's the JSON you asked for:\n\n{json.dumps(payload)}\n\nLet me know if you need anything else."
        envelope = json.dumps({"type": "result", "result": wrapped})
        runner = _RecordingRunner(_FakeCompletedProcess(stdout=envelope))

        result = reason_over_tick(
            observations=[obs],
            memory="",
            cli_runner=runner,
        )
        assert result.summary == "ok"

    def test_missing_oauth_token_raises(self, db_session, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        obs = _make_obs(db_session)
        with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN not set"):
            reason_over_tick(
                observations=[obs],
                memory="",
                cli_runner=_RecordingRunner(_FakeCompletedProcess()),
            )


# ── Orchestration ───────────────────────────────────────────


def _seed_account(db_session) -> None:
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


class TestRunReasoning:
    def test_no_observations_short_circuits(self, db_session):
        # Runner must not be invoked when there are no observations —
        # pass a runner that raises if called.
        def _boom(*_a, **_kw):
            raise AssertionError("runner should not be called")
        result = run_reasoning(db_session, cli_runner=_boom)
        assert result.decisions == []
        assert result.memory_update is None

    def test_escalate_decision_records_action(self, db_session):
        obs = _make_obs(db_session)
        _seed_account(db_session)

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
        run_reasoning(db_session, cli_runner=_runner_for(payload))

        actions = db_session.query(AgentAction).all()
        assert len(actions) == 1
        assert actions[0].action_type == "diagnose"
        assert actions[0].observation_id == obs.id
        assert actions[0].status == "pending"

    def test_monitor_decision_does_not_record_action(self, db_session):
        obs = _make_obs(db_session)
        _seed_account(db_session)
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
        run_reasoning(db_session, cli_runner=_runner_for(payload))
        assert db_session.query(AgentAction).count() == 0

    def test_memory_update_persisted_when_changed(self, db_session):
        obs = _make_obs(db_session)
        _seed_account(db_session)

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
        run_reasoning(db_session, cli_runner=_runner_for(payload))
        assert load_memory(db_session, "watchdog") == new_memory

    def test_null_memory_update_leaves_memory_intact(self, db_session):
        obs = _make_obs(db_session)
        _seed_account(db_session)
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
        run_reasoning(db_session, cli_runner=_runner_for(payload))
        assert load_memory(db_session, "watchdog") == "# original\n- important note"


# ── Active-window check ─────────────────────────────────────


class TestActiveWindow:
    def test_monday_midday_is_active(self):
        t = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)  # Mon 14:00 ET
        assert is_in_active_window(t) is True

    def test_monday_early_morning_is_inactive(self):
        t = datetime(2026, 4, 13, 11, 0, tzinfo=timezone.utc)  # Mon 07:00 ET
        assert is_in_active_window(t) is False

    def test_monday_late_evening_is_inactive(self):
        t = datetime(2026, 4, 13, 23, 0, tzinfo=timezone.utc)  # Mon 19:00 ET
        assert is_in_active_window(t) is False

    def test_saturday_is_inactive(self):
        t = datetime(2026, 4, 18, 16, 0, tzinfo=timezone.utc)  # Sat
        assert is_in_active_window(t) is False

    def test_boundary_open_is_active(self):
        t = datetime(2026, 4, 13, 12, 30, tzinfo=timezone.utc)  # Mon 08:30 ET
        assert is_in_active_window(t) is True

    def test_boundary_close_is_active(self):
        t = datetime(2026, 4, 13, 21, 0, tzinfo=timezone.utc)  # Mon 17:00 ET
        assert is_in_active_window(t) is True
