"""Tests for edgefinder/agents/weekly_summary.py."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from edgefinder.agents import weekly_summary as ws
from edgefinder.db.models import TradeRecord


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RecordingRunner:
    def __init__(self, defaults=None):
        self.defaults = defaults or {}
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        for prefix, response in self.defaults.items():
            if tuple(cmd[: len(prefix)]) == prefix:
                if response.returncode != 0 and kwargs.get("check"):
                    raise __import__("subprocess").CalledProcessError(
                        response.returncode, cmd, response.stdout, response.stderr
                    )
                return response
        return _FakeCompletedProcess()


def _envelope(summary: str) -> str:
    return json.dumps({"type": "result", "result": json.dumps({"summary": summary})})


@pytest.fixture(autouse=True)
def _fake_oauth(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-for-tests")


class TestIsoWeekLabel:
    def test_returns_iso_week(self):
        assert ws._iso_week_label(date(2026, 4, 18)) == "WEEK-2026-W16"


class TestFetchRecentTrades:
    def test_returns_recent_closed_trades(self, db_session):
        for days_ago in (2, 5, 10):  # 10 is outside the 7-day window
            db_session.add(
                TradeRecord(
                    trade_id=f"t-{days_ago}",
                    strategy_name="alpha",
                    symbol="AAPL",
                    direction="LONG",
                    trade_type="SWING",
                    entry_price=150.0,
                    exit_price=152.0,
                    shares=10,
                    stop_loss=140.0,
                    target=170.0,
                    confidence=0.7,
                    entry_time=datetime.utcnow() - timedelta(days=days_ago, hours=1),
                    exit_time=datetime.utcnow() - timedelta(days=days_ago),
                    status="CLOSED",
                    pnl_dollars=20.0,
                )
            )
        db_session.commit()
        rows = ws.fetch_recent_trades(db_session, days=7)
        assert len(rows) == 2  # 2 + 5, not the 10-day-old one


class TestReadWeekReviews:
    def test_returns_recent_review_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ws, "REVIEWS_DIR", tmp_path)
        recent = tmp_path / "2026-04-13-alpha.md"
        recent.write_text("# alpha\nfresh")
        # Older file: backdate mtime 30 days
        old = tmp_path / "2026-03-01-bravo.md"
        old.write_text("# bravo\nstale")
        import os, time
        old_ts = time.time() - 30 * 86400
        os.utime(old, (old_ts, old_ts))

        out = ws.read_week_reviews(days=7)
        assert len(out) == 1
        assert "fresh" in out[0]


class TestRunWeeklySummary:
    def test_dry_run_no_writes(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr(ws, "REVIEWS_DIR", tmp_path / "reviews")
        result = ws.run_weekly_summary(
            db_session,
            today=date(2026, 4, 18),
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert not (tmp_path / "reviews").exists()

    def test_full_run_writes_and_commits(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr(ws, "REVIEWS_DIR", tmp_path / "reviews")
        runner = _RecordingRunner({
            ("claude", "-p"): _FakeCompletedProcess(stdout=_envelope("All five strategies underperformed Tuesday.")),
        })
        result = ws.run_weekly_summary(
            db_session,
            today=date(2026, 4, 18),
            runner=runner,
        )
        assert result["summary_path"].endswith("WEEK-2026-W16.md")
        # Verify file exists with the expected content
        path = tmp_path / "reviews" / "WEEK-2026-W16.md"
        assert "Tuesday" in path.read_text()

        # git add + commit + push
        cmds = [tuple(c["cmd"][:2]) for c in runner.calls]
        assert ("git", "add") in cmds
        assert ("git", "commit") in cmds
        assert ("git", "push") in cmds


class TestCallSummary:
    def test_missing_oauth_raises(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN"):
            ws.call_summary("hi")

    def test_cli_failure_raises(self):
        runner = _RecordingRunner({
            ("claude", "-p"): _FakeCompletedProcess(returncode=1, stderr="boom"),
        })
        with pytest.raises(RuntimeError, match="boom"):
            ws.call_summary("hi", runner=runner)
