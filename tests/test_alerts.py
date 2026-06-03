"""Tests for edgefinder/agents/alerts.py — CRITICAL → GitHub-issue projection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from edgefinder.agents.alerts import _title_for, sync_alerts
from edgefinder.db.models import AgentObservation


class FakeGh:
    """Stand-in for subprocess.run over the `gh` CLI.

    Records every argv and returns canned stdout for `issue list`. Matches
    the call signature alerts._gh uses: runner(argv, capture_output=, text=,
    check=, **kw) → object with .stdout.
    """

    def __init__(self, list_titles: list[str] | None = None):
        self.calls: list[list[str]] = []
        self._list = [
            {"number": i + 1, "title": t} for i, t in enumerate(list_titles or [])
        ]

    def __call__(self, argv, capture_output=True, text=True, check=True, **kw):
        self.calls.append(argv)
        action = argv[2] if len(argv) > 2 else ""
        stdout = json.dumps(self._list) if action == "list" else ""
        return SimpleNamespace(stdout=stdout, returncode=0)

    def actions(self) -> list[str]:
        return [c[2] for c in self.calls if len(c) > 2]


def _add_obs(
    session,
    category: str = "cycle_liveness",
    key: str | None = "intraday_cycle",
    severity: str = "CRITICAL",
    resolved: bool = False,
) -> AgentObservation:
    obs = AgentObservation(
        agent_name="watchdog",
        severity=severity,
        category=category,
        message=f"{category} fired",
        obs_metadata={"key": key} if key else None,
        resolved_at=datetime.now(timezone.utc) if resolved else None,
    )
    session.add(obs)
    session.commit()
    return obs


def test_opens_issue_for_unresolved_critical(db_session):
    obs = _add_obs(db_session)
    gh = FakeGh(list_titles=[])  # no open alert issues
    summary = sync_alerts(db_session, runner=gh)

    assert summary["opened"] == 1
    assert summary["closed"] == 0
    assert summary["active"] == 1
    create = [c for c in gh.calls if c[2] == "create"]
    assert len(create) == 1
    assert _title_for(obs) in create[0]
    assert "edgefinder-alert" in create[0]


def test_idempotent_when_issue_already_open(db_session):
    obs = _add_obs(db_session)
    gh = FakeGh(list_titles=[_title_for(obs)])  # already filed
    summary = sync_alerts(db_session, runner=gh)

    assert summary["opened"] == 0
    assert "create" not in gh.actions()


def test_auto_closes_issue_when_condition_cleared(db_session):
    # No unresolved CRITICAL, but a stale alert issue is still open.
    gh = FakeGh(list_titles=["[edgefinder] CRITICAL: cycle_liveness/intraday_cycle"])
    summary = sync_alerts(db_session, runner=gh)

    assert summary["closed"] == 1
    assert summary["opened"] == 0
    close = [c for c in gh.calls if c[2] == "close"]
    assert len(close) == 1
    assert "1" in close[0]  # issue number


def test_ignores_non_critical_and_resolved(db_session):
    _add_obs(db_session, severity="WARN")
    _add_obs(db_session, key="other", resolved=True)
    gh = FakeGh(list_titles=[])
    summary = sync_alerts(db_session, runner=gh)

    assert summary == {"opened": 0, "closed": 0, "active": 0}
    assert "create" not in gh.actions()


def test_category_filter(db_session):
    _add_obs(db_session, category="cash_drift", key="alpha")
    gh = FakeGh(list_titles=[])

    # Restricted to cycle_liveness → the cash_drift CRITICAL is skipped.
    s1 = sync_alerts(db_session, runner=gh, categories={"cycle_liveness"})
    assert s1["opened"] == 0

    # Unrestricted → it pages.
    gh2 = FakeGh(list_titles=[])
    s2 = sync_alerts(db_session, runner=gh2)
    assert s2["opened"] == 1
