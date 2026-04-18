"""Tests for edgefinder/agents/ — kill-switch config + observation/action journal."""

from __future__ import annotations

import json

import pytest

from edgefinder.agents import (
    AgentConfigError,
    get_agent_config,
    is_agent_enabled,
    record_action,
    record_observation,
    resolve_observation,
)
from edgefinder.db.models import AgentAction, AgentObservation


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    """Point the agent config loader at a tmp file via env override."""
    path = tmp_path / "agent-config.json"
    monkeypatch.setenv("EDGEFINDER_AGENT_CONFIG", str(path))
    return path


def _write(path, data):
    path.write_text(json.dumps(data))


class TestAgentConfig:
    def test_missing_file_is_disabled(self, config_path):
        assert is_agent_enabled("watchdog") is False
        cfg = get_agent_config("watchdog")
        assert cfg.enabled is False
        assert cfg.settings == {}

    def test_global_off_disables_all(self, config_path):
        _write(config_path, {
            "enabled": False,
            "agents": {"watchdog": {"enabled": True}},
        })
        assert is_agent_enabled("watchdog") is False

    def test_per_agent_off_disables_one(self, config_path):
        _write(config_path, {
            "enabled": True,
            "agents": {
                "watchdog": {"enabled": False},
                "strategist": {"enabled": True},
            },
        })
        assert is_agent_enabled("watchdog") is False
        assert is_agent_enabled("strategist") is True

    def test_settings_surface_to_caller(self, config_path):
        _write(config_path, {
            "enabled": True,
            "agents": {"watchdog": {"enabled": True, "interval_minutes": 7}},
        })
        cfg = get_agent_config("watchdog")
        assert cfg.enabled is True
        assert cfg.settings["interval_minutes"] == 7

    def test_unknown_agent_defaults_disabled(self, config_path):
        _write(config_path, {"enabled": True, "agents": {}})
        assert is_agent_enabled("nonexistent") is False

    def test_invalid_json_raises_from_get_but_swallowed_by_is_enabled(
        self, config_path
    ):
        config_path.write_text("{not valid json")
        with pytest.raises(AgentConfigError):
            get_agent_config("watchdog")
        # is_agent_enabled wraps the error and defaults to False (fail-safe).
        assert is_agent_enabled("watchdog") is False

    def test_hot_reload_via_file_rewrite(self, config_path):
        _write(config_path, {"enabled": True, "agents": {"watchdog": {"enabled": False}}})
        assert is_agent_enabled("watchdog") is False
        _write(config_path, {"enabled": True, "agents": {"watchdog": {"enabled": True}}})
        # No explicit reload — the loader reads the file every call.
        assert is_agent_enabled("watchdog") is True


class TestObservationJournal:
    def test_record_and_resolve(self, db_session):
        obs_id = record_observation(
            db_session,
            agent_name="watchdog",
            severity="WARN",
            category="cash_drift",
            message="alpha drift $5.12",
            metadata={"strategy": "alpha", "drift": 5.12},
        )
        row = db_session.get(AgentObservation, obs_id)
        assert row is not None
        assert row.agent_name == "watchdog"
        assert row.severity == "WARN"
        assert row.obs_metadata == {"strategy": "alpha", "drift": 5.12}
        assert row.resolved_at is None

        assert resolve_observation(db_session, obs_id, "human") is True
        db_session.refresh(row)
        assert row.resolved_at is not None
        assert row.resolved_by == "human"

    def test_resolve_unknown_returns_false(self, db_session):
        assert resolve_observation(db_session, 99999, "human") is False

    def test_invalid_severity_rejected(self, db_session):
        with pytest.raises(ValueError):
            record_observation(
                db_session, "watchdog", "SEVERE", "cash_drift", "oops"
            )


class TestActionJournal:
    def test_record_action_with_observation_link(self, db_session):
        obs_id = record_observation(
            db_session, "watchdog", "ERROR", "cash_drift", "negative cash"
        )
        action_id = record_action(
            db_session,
            agent_name="watchdog",
            action_type="open_pr",
            summary="Propose fix for cash drift",
            files_touched=["dashboard/services.py"],
            pr_url="https://github.com/osgtax/edgefinder/pull/42",
            status="submitted",
            observation_id=obs_id,
        )
        row = db_session.get(AgentAction, action_id)
        assert row is not None
        assert row.observation_id == obs_id
        assert row.files_touched == ["dashboard/services.py"]
        assert row.status == "submitted"

    def test_invalid_status_rejected(self, db_session):
        with pytest.raises(ValueError):
            record_action(
                db_session,
                agent_name="watchdog",
                action_type="comment",
                summary="x",
                status="bogus",
            )
