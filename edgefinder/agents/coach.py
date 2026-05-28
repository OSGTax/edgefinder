"""Coach agent — daily per-strategy reviewer + tuner.

Once per weekday (5:30 PM ET, after market close), picks today's
strategy by round-robin over the registered strategies (read live from
StrategyRegistry, so the rotation never drifts from the shipped code) and:

  1. Pulls the last 30 days of that strategy's closed trades from
     Supabase.
  2. Reads the strategy's current parameters (the whole config/settings.py
     is small — let Claude find what's relevant).
  3. Hands trades + settings + the last few prior reviews to Claude,
     asking for: a short review paragraph + (optionally) ONE parameter
     tweak.
  4. Writes the review to reviews/YYYY-MM-DD-<strategy>.md.
  5. If a tweak was proposed, edits config/settings.py, commits both
     files on a new branch, opens a PR, and enables auto-merge — the PR
     waits for the test suite to go green before landing.
  6. If no tweak, commits the review file directly to main (no risk —
     it's just a markdown journal).

CLI:
    python -m edgefinder.agents.coach                    # auto-pick by rotation
    python -m edgefinder.agents.coach --strategy coward  # override
    python -m edgefinder.agents.coach --strategy coward --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from edgefinder.agents.config import get_agent_config
from edgefinder.agents.reasoning import _extract_json_object
from edgefinder.db.models import TradeRecord

logger = logging.getLogger(__name__)

AGENT_NAME = "coach"
DEFAULT_MODEL = "claude-opus-4-8"
LOOKBACK_DAYS = 30
PRIOR_REVIEWS_TO_INCLUDE = 3
REVIEWS_DIR = Path("reviews")
SETTINGS_FILE = Path("config/settings.py")

_ET = ZoneInfo("America/New_York")


SYSTEM_PROMPT = """You are EdgeFinder's strategy coach. Each weekday after market close you review one paper-trading strategy and may propose ONE small parameter change to improve it.

Your job for the strategy named in the input:
1. Read the recent closed trades and look for patterns: what's working, what's losing, exit-reason distributions, R-multiple trends, time-of-day effects, anything notable.
2. Write a short review (2-4 paragraphs, plain English). Anchor every claim in the data — cite specific trades or numbers when you can.
3. OPTIONALLY propose ONE parameter change to config/settings.py. The change should be small, defensible from the data, and structured as an exact string replacement (old_text must appear in the file exactly once).

Hard rules:
- Propose at most ONE change per review. If you can't justify a change confidently from the data, set proposed_change to null. A null is the right answer most days.
- Only edit lines inside config/settings.py. Never edit strategy plugin code under edgefinder/strategies/.
- Never propose changes to risk circuit breakers (drawdown_circuit_breaker_pct, pdt limits) or scanner credentials.
- old_text must match the file exactly once and be at most ~5 lines. Keep diffs surgical.
- If the strategy has too few trades to draw conclusions (say <10), say so in the review and propose nothing.

Return ONLY a JSON object matching the schema. No markdown fences, no commentary."""


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["review", "proposed_change"],
    "properties": {
        "review": {
            "type": "string",
            "description": "2-4 paragraphs of plain-English review.",
        },
        "proposed_change": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["old_text", "new_text", "rationale"],
                    "properties": {
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                },
            ]
        },
    },
}


@dataclass
class CoachResponse:
    review: str
    proposed_change: dict[str, str] | None


# ── Strategy rotation ──────────────────────────────────────


def active_strategies() -> list[str]:
    """Registered strategy names, sorted for a stable rotation order.

    Reads the live StrategyRegistry rather than a hardcoded list, so the
    rotation always matches the strategies actually shipped in the code —
    adding or removing a strategy plugin is picked up automatically with
    no change needed here.
    """
    import edgefinder.strategies  # noqa: F401 — import triggers registration
    from edgefinder.strategies.base import StrategyRegistry

    return sorted(StrategyRegistry.list_names())


def pick_strategy_for_today(now: datetime | None = None) -> str | None:
    """Today's strategy by round-robin over the registered strategies.

    Returns None on Sat/Sun (the coach only runs after weekday closes).
    On weekdays it cycles through ``active_strategies()`` by day-of-year,
    so every strategy is reviewed on an even cadence and a newly-added
    strategy joins the rotation automatically.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(_ET)
    if current.weekday() >= 5:  # Sat/Sun
        return None
    strategies = active_strategies()
    if not strategies:
        return None
    return strategies[current.timetuple().tm_yday % len(strategies)]


# ── Context builders ───────────────────────────────────────


def fetch_recent_trades(
    session: Session, strategy_name: str, days: int = LOOKBACK_DAYS
) -> list[dict[str, Any]]:
    """Last N days of CLOSED trades for the strategy."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (
        session.query(TradeRecord)
        .filter(
            TradeRecord.strategy_name == strategy_name,
            TradeRecord.status == "CLOSED",
            TradeRecord.exit_time.is_not(None),
            TradeRecord.exit_time >= cutoff,
        )
        .order_by(TradeRecord.exit_time.desc())
        .all()
    )
    return [
        {
            "trade_id": r.trade_id,
            "symbol": r.symbol,
            "direction": r.direction,
            "trade_type": r.trade_type,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "shares": r.shares,
            "entry_time": r.entry_time.isoformat() if r.entry_time else None,
            "exit_time": r.exit_time.isoformat() if r.exit_time else None,
            "pnl_dollars": r.pnl_dollars,
            "pnl_percent": r.pnl_percent,
            "r_multiple": r.r_multiple,
            "exit_reason": r.exit_reason,
        }
        for r in rows
    ]


def read_settings_text() -> str:
    return SETTINGS_FILE.read_text(encoding="utf-8")


def read_recent_reviews(strategy_name: str, n: int = PRIOR_REVIEWS_TO_INCLUDE) -> list[str]:
    """Return up to N most recent prior reviews for the strategy."""
    if not REVIEWS_DIR.exists():
        return []
    matches = sorted(REVIEWS_DIR.glob(f"*-{strategy_name}.md"), reverse=True)[:n]
    return [p.read_text(encoding="utf-8") for p in matches]


def build_prompt(
    strategy_name: str,
    trades: list[dict[str, Any]],
    settings_text: str,
    prior_reviews: list[str],
) -> str:
    """Assemble the full prompt for `claude -p`."""
    schema_str = json.dumps(RESPONSE_SCHEMA, indent=2)
    trades_str = json.dumps(trades, indent=2, default=str)
    prior_block = (
        "\n\n---\n\n".join(prior_reviews) if prior_reviews else "(no prior reviews)"
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Required output schema (JSON Schema)\n\n```json\n{schema_str}\n```\n\n"
        f"## Strategy: {strategy_name}\n\n"
        f"## Closed trades — last {LOOKBACK_DAYS} days ({len(trades)} trades)\n\n"
        f"```json\n{trades_str}\n```\n\n"
        f"## Current config/settings.py\n\n```python\n{settings_text}\n```\n\n"
        f"## Last {len(prior_reviews)} prior reviews of this strategy\n\n{prior_block}\n\n"
        "Respond with ONE JSON object matching the schema above. "
        "No markdown fences, no commentary, no text outside the JSON."
    )


# ── Claude call ────────────────────────────────────────────


def call_coach(
    prompt: str,
    model: str = DEFAULT_MODEL,
    runner: Any | None = None,
) -> CoachResponse:
    """Invoke `claude -p`, parse response JSON into a CoachResponse."""
    runner = runner or subprocess.run

    if not os.getenv("CLAUDE_CODE_OAUTH_TOKEN"):
        raise RuntimeError(
            "CLAUDE_CODE_OAUTH_TOKEN not set. Generate locally with "
            "`claude setup-token` and add it as a GitHub Actions secret."
        )

    result = runner(
        ["claude", "-p", "--model", model, "--output-format", "json"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p exited {result.returncode}: {result.stderr[:500]}")

    envelope = json.loads(result.stdout)
    response_text = envelope.get("result")
    if not isinstance(response_text, str):
        raise RuntimeError(f"Unexpected claude -p envelope: {result.stdout[:300]}")

    inner = _extract_json_object(response_text)
    data = json.loads(inner)
    return CoachResponse(
        review=data["review"],
        proposed_change=data.get("proposed_change"),
    )


# ── File and git operations ────────────────────────────────


def write_review_file(strategy_name: str, body: str, today: date | None = None) -> Path:
    """Write the review markdown file. Returns the path."""
    today = today or date.today()
    REVIEWS_DIR.mkdir(exist_ok=True)
    path = REVIEWS_DIR / f"{today.isoformat()}-{strategy_name}.md"
    header = f"# {strategy_name} review — {today.isoformat()}\n\n"
    path.write_text(header + body.rstrip() + "\n", encoding="utf-8")
    return path


def apply_change(change: dict[str, str]) -> Path:
    """Apply old_text → new_text in SETTINGS_FILE. Raises if old_text
    doesn't match exactly once.
    """
    text = SETTINGS_FILE.read_text(encoding="utf-8")
    occurrences = text.count(change["old_text"])
    if occurrences != 1:
        raise ValueError(
            f"old_text matched {occurrences} times in {SETTINGS_FILE} "
            f"(must match exactly once)"
        )
    new_text = text.replace(change["old_text"], change["new_text"], 1)
    SETTINGS_FILE.write_text(new_text, encoding="utf-8")
    return SETTINGS_FILE


def _git(args: list[str], runner: Any | None = None, **kwargs) -> subprocess.CompletedProcess:
    runner = runner or subprocess.run
    return runner(["git", *args], capture_output=True, text=True, check=True, **kwargs)


def _gh(args: list[str], runner: Any | None = None, **kwargs) -> subprocess.CompletedProcess:
    runner = runner or subprocess.run
    return runner(["gh", *args], capture_output=True, text=True, check=True, **kwargs)


def commit_review_only_to_main(
    review_path: Path, message: str, runner: Any | None = None
) -> None:
    """Commit the review markdown directly to main and push."""
    _git(["add", str(review_path)], runner=runner)
    _git(["commit", "-m", message], runner=runner)
    _git(["push", "origin", "HEAD:main"], runner=runner)


def open_tune_pr_with_automerge(
    branch: str,
    files: list[Path],
    title: str,
    body: str,
    runner: Any | None = None,
) -> str:
    """Create branch, commit, push, open PR, enable auto-merge.

    Returns the PR URL. The PR waits for required checks (the tests.yml
    workflow) to go green before merging itself.
    """
    _git(["checkout", "-b", branch], runner=runner)
    for f in files:
        _git(["add", str(f)], runner=runner)
    _git(["commit", "-m", title], runner=runner)
    _git(["push", "-u", "origin", branch], runner=runner)
    pr_result = _gh(
        ["pr", "create", "--title", title, "--body", body, "--base", "main", "--head", branch],
        runner=runner,
    )
    pr_url = pr_result.stdout.strip().splitlines()[-1]
    # `--auto` enables auto-merge; merges once required checks pass. If
    # the repo has auto-merge disabled this will fail loudly and the PR
    # stays open for manual review.
    _gh(["pr", "merge", "--auto", "--squash", "--delete-branch", pr_url], runner=runner)
    return pr_url


# ── Orchestration ──────────────────────────────────────────


def run_coach(
    session: Session,
    strategy_name: str,
    today: date | None = None,
    dry_run: bool = False,
    runner: Any | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Full pipeline. Returns a summary dict for the caller to log."""
    today = today or date.today()

    trades = fetch_recent_trades(session, strategy_name)
    settings_text = read_settings_text()
    prior_reviews = read_recent_reviews(strategy_name)
    prompt = build_prompt(strategy_name, trades, settings_text, prior_reviews)

    if dry_run:
        return {
            "strategy": strategy_name,
            "trades": len(trades),
            "prompt_chars": len(prompt),
            "dry_run": True,
            "prompt": prompt,
        }

    response = call_coach(prompt, model=model, runner=runner)
    review_path = write_review_file(strategy_name, response.review, today=today)

    if response.proposed_change:
        apply_change(response.proposed_change)
        rationale = response.proposed_change.get("rationale", "tune")
        title = f"[coach] {strategy_name}: {rationale[:120]}"
        body = (
            f"## Review\n\n{response.review}\n\n"
            f"## Proposed change\n\n**Rationale:** {rationale}\n\n"
            "```diff\n"
            f"- {response.proposed_change['old_text']}\n"
            f"+ {response.proposed_change['new_text']}\n"
            "```\n\n"
            "Auto-merge enabled — this PR will land on `main` once the "
            "test suite goes green."
        )
        branch = f"agent/coach-{strategy_name}-{today.isoformat()}"
        pr_url = open_tune_pr_with_automerge(
            branch=branch,
            files=[review_path, SETTINGS_FILE],
            title=title,
            body=body,
            runner=runner,
        )
        return {
            "strategy": strategy_name,
            "trades": len(trades),
            "review_path": str(review_path),
            "pr_url": pr_url,
            "tune": True,
        }

    commit_review_only_to_main(
        review_path,
        f"[coach] {strategy_name} review {today.isoformat()}",
        runner=runner,
    )
    return {
        "strategy": strategy_name,
        "trades": len(trades),
        "review_path": str(review_path),
        "tune": False,
    }


# ── CLI ────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the coach for one strategy")
    parser.add_argument(
        "--strategy",
        default=None,
        help="Override the rotation pick with any registered strategy name "
        "(e.g. coward, gambler, degenerate)",
    )
    parser.add_argument("--force", action="store_true", help="Run even if disabled in agent-config")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt + plan, no writes")
    parser.add_argument("--model", default=None, help="Override Claude model")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = get_agent_config(AGENT_NAME)
    if not cfg.enabled and not args.force:
        logger.info("[coach] disabled by .claude/agent-config.json — exiting cleanly")
        return 0

    strategy = args.strategy or pick_strategy_for_today()
    if strategy is None:
        logger.info("[coach] weekend, no strategy assigned today — exiting")
        return 0
    known = active_strategies()
    if strategy not in known:
        logger.error("[coach] unknown strategy %r — known: %s", strategy, known)
        return 2

    from edgefinder.db.engine import get_engine, get_session_factory

    session = get_session_factory(get_engine())()
    try:
        result = run_coach(
            session,
            strategy_name=strategy,
            dry_run=args.dry_run,
            model=args.model or os.getenv("COACH_MODEL") or DEFAULT_MODEL,
        )
    finally:
        session.close()

    logger.info("[coach] %s", json.dumps(result, default=str)[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
