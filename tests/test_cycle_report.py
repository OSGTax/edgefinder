"""The per-run cycle-report email — scripts/cycle_report.py.

Pins: gather matches this run's -gha suffix (with the time-window
fallback when a cycle forgot it), compose leads with the verdict and
degrades gracefully, and send() is a silent no-op without SMTP config —
the report must never fail a trading run.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cycle_report  # noqa: E402

NOW = datetime(2026, 7, 16, 18, 30, 0)


def _fake_rest(tables):
    def fake(path, params):
        spec = dict()
        for k, v in params:
            spec.setdefault(k, []).append(v)
        return tables(path, spec)
    return fake


def test_gather_prefers_run_id_match(monkeypatch):
    def tables(path, spec):
        if path == "desk_decisions" and any(
                v.startswith("like.*gha123*") for v in spec.get("run_id", [])):
            return [{"ts": "2026-07-16T18:25:00", "run_id": "x-gha123",
                     "summary": "held; studied financials"}]
        if path == "desk_trades":
            return []
        if path == "desk_wakes":
            return [{"at": "2026-07-16T19:00:00", "reason": "chain"}]
        if path == "desk_equity":
            return [{"ts": "2026-07-16T18:26:00", "equity": 97000.0,
                     "return_pct": -3.0}]
        return []
    monkeypatch.setattr(cycle_report, "_rest", _fake_rest(tables))
    facts = cycle_report.gather("123", NOW)
    assert facts["matched_run"] is True
    assert facts["decision"]["summary"] == "held; studied financials"
    assert facts["next_wake"]["reason"] == "chain"


def test_gather_falls_back_to_recent_window(monkeypatch):
    def tables(path, spec):
        if path == "desk_decisions":
            if any(v.startswith("like.") for v in spec.get("run_id", [])):
                return []  # cycle forgot its -gha suffix
            return [{"ts": "2026-07-16T18:20:00", "run_id": "2026-07-16T18:16",
                     "summary": "hold"}]
        return []
    monkeypatch.setattr(cycle_report, "_rest", _fake_rest(tables))
    facts = cycle_report.gather("123", NOW)
    assert facts["matched_run"] is False
    assert facts["decision"]["summary"] == "hold"


def test_gather_surfaces_loop_health_flags(monkeypatch):
    monkeypatch.setattr(cycle_report, "_rest", _fake_rest(lambda path, spec: []))
    facts = cycle_report.gather("123", NOW)
    assert any("no active claims" in f for f in facts["loop_health"])
    assert any("no pick has cited" in f for f in facts["loop_health"])


def test_gather_loop_health_clean_when_claims_cited(monkeypatch):
    def tables(path, spec):
        if path == "desk_claims":
            return [{"id": 1}]
        if path == "desk_decisions" and spec.get("select") == ["picks"]:
            return [{"picks": [{"symbol": "NVDA", "claims": [1]}]}]
        return []
    monkeypatch.setattr(cycle_report, "_rest", _fake_rest(tables))
    facts = cycle_report.gather("123", NOW)
    assert facts["loop_health"] == []


def test_compose_success_with_fills():
    facts = {
        "matched_run": True,
        "decision": {"ts": "2026-07-16T18:25:00", "summary": "Bought AMD dip"},
        "fills": [{"side": "buy", "symbol": "AMD", "shares": 2.5, "qty": None,
                   "price": 495.1, "notional": 1237.75,
                   "ts": "2026-07-16T18:24:00"}],
        "next_wake": {"at": "2026-07-16T19:00:00", "reason": "chain: 30m"},
        "equity": {"ts": "2026-07-16T18:26:00", "equity": 97123.0,
                   "return_pct": -2.88},
    }
    subject, body = cycle_report.compose("123", "success", facts, NOW)
    assert "Bought AMD dip" in subject
    assert "BUY AMD x2.5 @ 495.1" in body
    assert "$97,123" in body
    assert "Next check-in:" in body and "chain: 30m" in body
    assert "actions/runs/123" in body


def test_compose_failure_leads_with_failure():
    facts = {"matched_run": False, "decision": None, "fills": [],
             "next_wake": None, "equity": None}
    subject, body = cycle_report.compose("456", "failure", facts, NOW)
    assert "FAILED" in subject
    assert "No fills this cycle." in body
    assert "none pending" in body


def test_compose_renders_loop_health():
    facts = {"matched_run": False, "decision": None, "fills": [],
             "next_wake": None, "equity": None,
             "loop_health": ["no active claims — the knowledge base is empty"]}
    _, body = cycle_report.compose("789", "success", facts, NOW)
    assert "Knowledge loop: no active claims" in body


def test_send_without_config_is_a_noop(monkeypatch):
    for var in ("CYCLE_REPORT_TO", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    def boom(*a, **k):
        raise AssertionError("must not open SMTP without config")
    monkeypatch.setattr(cycle_report.smtplib, "SMTP_SSL", boom)
    assert cycle_report.send("s", "b") is False
