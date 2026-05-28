"""Tests for edgefinder/agents/coach.py."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from edgefinder.agents import coach
from edgefinder.db.models import TradeRecord


# ── Mock subprocess ─────────────────────────────────────────


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _RecordingRunner:
    """Stub for subprocess.run — captures calls + returns canned responses."""

    def __init__(self, defaults: dict[tuple[str, ...], _FakeCompletedProcess] | None = None):
        # Map exact argv prefix tuple → response. Falls back to a generic
        # success for git/gh.
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
        # Default success (git/gh mocks)
        return _FakeCompletedProcess(stdout="https://github.com/owner/repo/pull/1\n")


def _envelope_for_coach(review: str, change: dict | None = None) -> str:
    payload = {"review": review, "proposed_change": change}
    return json.dumps({"type": "result", "result": json.dumps(payload)})


@pytest.fixture(autouse=True)
def _fake_oauth(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-for-tests")


# ── Strategy rotation ───────────────────────────────────────


class TestPickStrategy:
    def test_active_strategies_reads_registry(self):
        # Guards against the rotation drifting from the shipped code: the
        # coach must derive its list from the live StrategyRegistry.
        names = coach.active_strategies()
        assert {"coward", "gambler", "degenerate"} <= set(names)
        assert names == sorted(names)  # stable order

    def test_weekday_rotation_is_round_robin(self, monkeypatch):
        monkeypatch.setattr(coach, "active_strategies", lambda: ["aa", "bb", "cc"])
        # Three consecutive weekdays (Mon–Wed Apr 13–15 2026, 18:00 UTC =
        # 14:00 ET) advance day-of-year by one each day, so a 3-strategy
        # list is covered exactly once with no repeats.
        picks = [
            coach.pick_strategy_for_today(datetime(2026, 4, d, 18, 0, tzinfo=timezone.utc))
            for d in (13, 14, 15)
        ]
        assert sorted(picks) == ["aa", "bb", "cc"]
        assert len(set(picks)) == 3

    def test_rotation_is_deterministic(self, monkeypatch):
        monkeypatch.setattr(coach, "active_strategies", lambda: ["aa", "bb", "cc"])
        dt = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
        assert coach.pick_strategy_for_today(dt) == coach.pick_strategy_for_today(dt)

    def test_saturday_returns_none(self):
        assert coach.pick_strategy_for_today(datetime(2026, 4, 18, 18, 0, tzinfo=timezone.utc)) is None

    def test_sunday_returns_none(self):
        assert coach.pick_strategy_for_today(datetime(2026, 4, 19, 18, 0, tzinfo=timezone.utc)) is None

    def test_no_registered_strategies_returns_none(self, monkeypatch):
        monkeypatch.setattr(coach, "active_strategies", lambda: [])
        assert coach.pick_strategy_for_today(datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)) is None


# ── Trade fetching ─────────────────────────────────────────


def _make_closed_trade(session, strategy: str, days_ago: int, pnl: float = 10.0) -> TradeRecord:
    exit_dt = datetime.utcnow() - timezone.utc.utcoffset(datetime.utcnow())  # naive utc
    from datetime import timedelta as _td
    exit_dt = datetime.utcnow() - _td(days=days_ago)
    record = TradeRecord(
        trade_id=f"{strategy}-{days_ago}-{pnl}",
        strategy_name=strategy,
        symbol="AAPL",
        direction="LONG",
        trade_type="SWING",
        entry_price=150.0,
        exit_price=155.0,
        shares=10,
        stop_loss=140.0,
        target=170.0,
        confidence=0.7,
        entry_time=exit_dt - _td(hours=1),
        exit_time=exit_dt,
        status="CLOSED",
        pnl_dollars=pnl,
    )
    session.add(record)
    session.commit()
    return record


class TestFetchRecentTrades:
    def test_returns_closed_trades_in_window(self, db_session):
        _make_closed_trade(db_session, "alpha", days_ago=5)
        _make_closed_trade(db_session, "alpha", days_ago=15)
        # outside lookback window
        _make_closed_trade(db_session, "alpha", days_ago=45)
        # different strategy
        _make_closed_trade(db_session, "bravo", days_ago=5)

        trades = coach.fetch_recent_trades(db_session, "alpha", days=30)
        assert len(trades) == 2
        assert all(t["symbol"] == "AAPL" for t in trades)


# ── apply_change ───────────────────────────────────────────


class TestApplyChange:
    def test_unique_match_replaces(self, tmp_path, monkeypatch):
        target = tmp_path / "settings.py"
        target.write_text("a = 1\nb = 2\n")
        monkeypatch.setattr(coach, "SETTINGS_FILE", target)
        coach.apply_change({"old_text": "b = 2", "new_text": "b = 3", "rationale": "x"})
        assert target.read_text() == "a = 1\nb = 3\n"

    def test_zero_matches_raises(self, tmp_path, monkeypatch):
        target = tmp_path / "settings.py"
        target.write_text("a = 1\n")
        monkeypatch.setattr(coach, "SETTINGS_FILE", target)
        with pytest.raises(ValueError, match="matched 0 times"):
            coach.apply_change({"old_text": "missing", "new_text": "x", "rationale": "x"})

    def test_multiple_matches_raises(self, tmp_path, monkeypatch):
        target = tmp_path / "settings.py"
        target.write_text("x = 1\nx = 1\n")
        monkeypatch.setattr(coach, "SETTINGS_FILE", target)
        with pytest.raises(ValueError, match="matched 2 times"):
            coach.apply_change({"old_text": "x = 1", "new_text": "x = 2", "rationale": "x"})


# ── write_review_file ──────────────────────────────────────


class TestWriteReviewFile:
    def test_creates_dated_markdown(self, tmp_path, monkeypatch):
        monkeypatch.setattr(coach, "REVIEWS_DIR", tmp_path / "reviews")
        path = coach.write_review_file("alpha", "Looks fine.", today=date(2026, 4, 13))
        assert path.name == "2026-04-13-alpha.md"
        body = path.read_text()
        assert body.startswith("# alpha review — 2026-04-13\n\n")
        assert "Looks fine." in body


# ── End-to-end orchestration ───────────────────────────────


class TestRunCoach:
    def test_dry_run_returns_prompt_and_writes_nothing(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr(coach, "REVIEWS_DIR", tmp_path / "reviews")
        monkeypatch.setattr(coach, "SETTINGS_FILE", tmp_path / "settings.py")
        (tmp_path / "settings.py").write_text("# fake settings\n")

        result = coach.run_coach(
            db_session,
            strategy_name="alpha",
            today=date(2026, 4, 13),
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert "alpha" in result["prompt"]
        assert not (tmp_path / "reviews").exists()

    def test_review_only_commits_to_main(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr(coach, "REVIEWS_DIR", tmp_path / "reviews")
        monkeypatch.setattr(coach, "SETTINGS_FILE", tmp_path / "settings.py")
        (tmp_path / "settings.py").write_text("# fake settings\n")

        runner = _RecordingRunner({
            ("claude", "-p"): _FakeCompletedProcess(
                stdout=_envelope_for_coach("Looks healthy. No change needed.", change=None)
            ),
        })

        result = coach.run_coach(
            db_session,
            strategy_name="alpha",
            today=date(2026, 4, 13),
            runner=runner,
        )
        assert result["tune"] is False
        assert result["review_path"].endswith("2026-04-13-alpha.md")

        # Should have done: claude -p, git add, git commit, git push (4 calls)
        cmds = [tuple(c["cmd"][:2]) for c in runner.calls]
        assert ("claude", "-p") in cmds
        assert ("git", "add") in cmds
        assert ("git", "commit") in cmds
        assert ("git", "push") in cmds
        # No PR
        assert not any(c["cmd"][:2] == ["gh", "pr"] for c in runner.calls)

    def test_tune_opens_pr_with_automerge(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setattr(coach, "REVIEWS_DIR", tmp_path / "reviews")
        monkeypatch.setattr(coach, "SETTINGS_FILE", tmp_path / "settings.py")
        (tmp_path / "settings.py").write_text("max_risk_per_trade_pct: float = 0.02\n")

        change = {
            "old_text": "max_risk_per_trade_pct: float = 0.02",
            "new_text": "max_risk_per_trade_pct: float = 0.025",
            "rationale": "R-multiple drifting down; widen risk slightly",
        }
        runner = _RecordingRunner({
            ("claude", "-p"): _FakeCompletedProcess(
                stdout=_envelope_for_coach("Win rate dropped this week.", change=change)
            ),
        })

        result = coach.run_coach(
            db_session,
            strategy_name="alpha",
            today=date(2026, 4, 13),
            runner=runner,
        )
        assert result["tune"] is True
        assert result["pr_url"].startswith("https://github.com/")

        cmds = [tuple(c["cmd"][:3]) for c in runner.calls]
        # Branch + push
        assert ("git", "checkout", "-b") in cmds
        assert ("git", "push", "-u") in cmds
        # PR creation + auto-merge
        assert ("gh", "pr", "create") in cmds
        assert ("gh", "pr", "merge") in cmds

        # Settings was actually edited
        assert "0.025" in (tmp_path / "settings.py").read_text()


# ── Claude-call error path ─────────────────────────────────


class TestCallCoach:
    def test_missing_oauth_raises(self, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN"):
            coach.call_coach("hi")

    def test_cli_failure_raises(self):
        runner = _RecordingRunner({
            ("claude", "-p"): _FakeCompletedProcess(returncode=1, stderr="auth bad"),
        })
        with pytest.raises(RuntimeError, match="auth bad"):
            coach.call_coach("hi", runner=runner)

    def test_extracts_json_from_wrapping_prose(self):
        payload = {"review": "ok", "proposed_change": None}
        wrapped = f"Here is the JSON: {json.dumps(payload)} hope it helps!"
        envelope = json.dumps({"type": "result", "result": wrapped})
        runner = _RecordingRunner({
            ("claude", "-p"): _FakeCompletedProcess(stdout=envelope),
        })
        response = coach.call_coach("hi", runner=runner)
        assert response.review == "ok"
        assert response.proposed_change is None
