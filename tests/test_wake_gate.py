"""The workflow gate's decision logic — scripts/wake_gate.py.

The gate runs before Claude in every GitHub Actions trading run. These
tests pin its three behaviors: dispatch events always run, scheduled runs
fire on a due wake OR a quiet chain during desk hours, and any Supabase
failure fails CLOSED (no tokens spent discovering an outage). All REST
calls are monkeypatched — no live HTTP.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import wake_gate  # noqa: E402


# 2026-07-15 is a Wednesday; 15:00 UTC = 11:00 AM EDT (desk hours).
DESK_TIME = datetime(2026, 7, 15, 15, 0, 0)
# 02:00 UTC same day = 10:00 PM EDT the prior evening (off hours).
NIGHT_TIME = datetime(2026, 7, 15, 2, 0, 0)
# 2026-07-18 is a Saturday, mid-desk-hours clock time.
WEEKEND_TIME = datetime(2026, 7, 18, 15, 0, 0)


def test_dispatch_events_always_run(monkeypatch):
    def boom(*a, **k):  # the DB must not even be consulted
        raise AssertionError("dispatch events must not query the DB")
    monkeypatch.setattr(wake_gate, "_rest", boom)
    assert wake_gate.decide("workflow_dispatch", DESK_TIME) is True
    assert wake_gate.decide("repository_dispatch", NIGHT_TIME) is True


def test_schedule_runs_on_due_wake_even_off_hours(monkeypatch):
    monkeypatch.setattr(wake_gate, "due_wakes_exist", lambda now=None: True)
    monkeypatch.setattr(wake_gate, "recent_cycle_exists",
                        lambda now=None: True)
    assert wake_gate.decide("schedule", NIGHT_TIME) is True
    assert wake_gate.decide("schedule", WEEKEND_TIME) is True


def test_schedule_restarts_a_quiet_chain_during_desk_hours(monkeypatch):
    monkeypatch.setattr(wake_gate, "due_wakes_exist", lambda now=None: False)
    monkeypatch.setattr(wake_gate, "recent_cycle_exists",
                        lambda now=None: False)
    assert wake_gate.decide("schedule", DESK_TIME) is True     # chain quiet -> re-seed
    assert wake_gate.decide("schedule", NIGHT_TIME) is False   # not desk hours
    assert wake_gate.decide("schedule", WEEKEND_TIME) is False


def test_schedule_skips_when_chain_is_healthy(monkeypatch):
    monkeypatch.setattr(wake_gate, "due_wakes_exist", lambda now=None: False)
    monkeypatch.setattr(wake_gate, "recent_cycle_exists",
                        lambda now=None: True)
    assert wake_gate.decide("schedule", DESK_TIME) is False


def test_supabase_outage_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("supabase down")
    monkeypatch.setattr(wake_gate, "_rest", boom)
    assert wake_gate.decide("schedule", DESK_TIME) is False


def test_desk_hours_boundaries():
    # 12:59 UTC = 8:59 AM EDT — one minute before the prep window opens.
    assert wake_gate.desk_hours(datetime(2026, 7, 15, 12, 59)) is False
    # 13:00 UTC = 9:00 AM EDT — prep window opens.
    assert wake_gate.desk_hours(datetime(2026, 7, 15, 13, 0)) is True
    # 20:29 UTC = 4:29 PM EDT — wrap window still open.
    assert wake_gate.desk_hours(datetime(2026, 7, 15, 20, 29)) is True
    # 20:30 UTC = 4:30 PM EDT — closed.
    assert wake_gate.desk_hours(datetime(2026, 7, 15, 20, 30)) is False
    # January (EST): 14:00 UTC = 9:00 AM EST opens; 13:59 doesn't.
    assert wake_gate.desk_hours(datetime(2026, 1, 14, 13, 59)) is False
    assert wake_gate.desk_hours(datetime(2026, 1, 14, 14, 0)) is True


def test_rest_queries_shape(monkeypatch):
    """due_wakes_exist / recent_cycle_exists ask PostgREST the right question."""
    calls: list[tuple[str, list]] = []

    def fake_rest(path, params):
        calls.append((path, params))
        return []

    monkeypatch.setattr(wake_gate, "_rest", fake_rest)
    assert wake_gate.due_wakes_exist(DESK_TIME) is False
    assert wake_gate.recent_cycle_exists(DESK_TIME) is False

    wpath, wparams = calls[0]
    assert wpath == "desk_wakes"
    assert ("honored_run_id", "is.null") in wparams
    ats = [v for k, v in wparams if k == "at"]
    assert any(v.startswith("gte.") for v in ats)
    assert any(v.startswith("lte.") for v in ats)

    dpath, dparams = calls[1]
    assert dpath == "desk_decisions"
    assert any(k == "ts" and v.startswith("gte.") for k, v in dparams)
