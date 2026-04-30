"""Weekly portfolio summary — Saturday morning cross-strategy synthesis.

Reads the past 7 days of `reviews/` markdown + the past 7 days of trades
across ALL strategies, asks Claude to synthesize portfolio-level
patterns ("alpha and bravo both lost on Tuesday — was it the regime?",
"charlie's win rate is climbing while degenerate's drops"), writes the
result to reviews/WEEK-YYYY-WW.md, and pushes directly to main.

No code changes are proposed at the portfolio level — this is a
read-and-report job. Daily coach runs handle per-strategy tuning.

CLI:
    python -m edgefinder.agents.weekly_summary
    python -m edgefinder.agents.weekly_summary --dry-run
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

from sqlalchemy.orm import Session

from edgefinder.agents.config import get_agent_config
from edgefinder.agents.reasoning import _extract_json_object
from edgefinder.db.models import TradeRecord

logger = logging.getLogger(__name__)

AGENT_NAME = "weekly_summary"
DEFAULT_MODEL = "claude-opus-4-7"
LOOKBACK_DAYS = 7
REVIEWS_DIR = Path("reviews")


SYSTEM_PROMPT = """You are EdgeFinder's portfolio-level analyst. Once a week you read the past week's per-strategy reviews and the past week's trades across ALL strategies, then write a portfolio-level synthesis.

Your goals:
1. Identify cross-strategy patterns the daily coach can't see — e.g., "all five strategies took losses on Tuesday afternoon, suggesting a market-regime effect," or "alpha and bravo entered the same names within 30 minutes on three days, indicating signal correlation."
2. Note which strategies are trending up vs. down on win rate, expectancy, R-multiple, or trade frequency.
3. Call out anything anomalous: a strategy that stopped trading, a P&L distribution that suddenly widened, a new exit-reason pattern.

You do NOT propose code or parameter changes at the portfolio level. The daily coach handles tuning per strategy. Your job is observation and synthesis only.

Output a single JSON object with one field, "summary", containing 3-6 paragraphs of plain English. Anchor every claim in the data — cite specific strategies, dates, or numbers when you can.

Return ONLY a JSON object matching the schema. No markdown fences, no commentary outside the JSON."""


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "3-6 paragraphs of plain-English portfolio-level synthesis.",
        },
    },
}


@dataclass
class WeeklySummary:
    summary: str


# ── Context builders ───────────────────────────────────────


def fetch_recent_trades(session: Session, days: int = LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """All CLOSED trades across all strategies in the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (
        session.query(TradeRecord)
        .filter(
            TradeRecord.status == "CLOSED",
            TradeRecord.exit_time.is_not(None),
            TradeRecord.exit_time >= cutoff,
        )
        .order_by(TradeRecord.exit_time.desc())
        .all()
    )
    return [
        {
            "strategy": r.strategy_name,
            "symbol": r.symbol,
            "direction": r.direction,
            "entry_time": r.entry_time.isoformat() if r.entry_time else None,
            "exit_time": r.exit_time.isoformat() if r.exit_time else None,
            "pnl_dollars": r.pnl_dollars,
            "r_multiple": r.r_multiple,
            "exit_reason": r.exit_reason,
        }
        for r in rows
    ]


def read_week_reviews(days: int = LOOKBACK_DAYS) -> list[str]:
    """All review markdown files modified in the last N days."""
    if not REVIEWS_DIR.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    out: list[str] = []
    for path in sorted(REVIEWS_DIR.glob("*.md")):
        if datetime.fromtimestamp(path.stat().st_mtime) >= cutoff:
            out.append(path.read_text(encoding="utf-8"))
    return out


def build_prompt(trades: list[dict[str, Any]], reviews: list[str]) -> str:
    schema_str = json.dumps(RESPONSE_SCHEMA, indent=2)
    trades_str = json.dumps(trades, indent=2, default=str)
    reviews_block = "\n\n---\n\n".join(reviews) if reviews else "(no reviews this week)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Required output schema\n\n```json\n{schema_str}\n```\n\n"
        f"## Closed trades — last {LOOKBACK_DAYS} days, all strategies ({len(trades)} trades)\n\n"
        f"```json\n{trades_str}\n```\n\n"
        f"## Daily reviews from this week ({len(reviews)} reviews)\n\n{reviews_block}\n\n"
        "Respond with ONE JSON object matching the schema. "
        "No markdown fences, no commentary outside the JSON."
    )


# ── Claude call ────────────────────────────────────────────


def call_summary(
    prompt: str,
    model: str = DEFAULT_MODEL,
    runner: Any | None = None,
) -> WeeklySummary:
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
    return WeeklySummary(summary=data["summary"])


# ── File + git ─────────────────────────────────────────────


def _iso_week_label(today: date) -> str:
    """e.g. WEEK-2026-W17 — ISO week, year-aligned."""
    iso_year, iso_week, _ = today.isocalendar()
    return f"WEEK-{iso_year}-W{iso_week:02d}"


def write_summary_file(body: str, today: date | None = None) -> Path:
    today = today or date.today()
    REVIEWS_DIR.mkdir(exist_ok=True)
    label = _iso_week_label(today)
    path = REVIEWS_DIR / f"{label}.md"
    header = f"# Portfolio summary — {label} (generated {today.isoformat()})\n\n"
    path.write_text(header + body.rstrip() + "\n", encoding="utf-8")
    return path


def commit_summary_to_main(path: Path, message: str, runner: Any | None = None) -> None:
    runner = runner or subprocess.run
    runner(["git", "add", str(path)], capture_output=True, text=True, check=True)
    runner(["git", "commit", "-m", message], capture_output=True, text=True, check=True)
    runner(["git", "push", "origin", "HEAD:main"], capture_output=True, text=True, check=True)


# ── Orchestration ──────────────────────────────────────────


def run_weekly_summary(
    session: Session,
    today: date | None = None,
    dry_run: bool = False,
    runner: Any | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    today = today or date.today()
    trades = fetch_recent_trades(session)
    reviews = read_week_reviews()
    prompt = build_prompt(trades, reviews)

    if dry_run:
        return {
            "trades": len(trades),
            "reviews": len(reviews),
            "prompt_chars": len(prompt),
            "dry_run": True,
            "prompt": prompt,
        }

    result = call_summary(prompt, model=model, runner=runner)
    path = write_summary_file(result.summary, today=today)
    commit_summary_to_main(
        path,
        f"[weekly-summary] {_iso_week_label(today)}",
        runner=runner,
    )
    return {
        "trades": len(trades),
        "reviews": len(reviews),
        "summary_path": str(path),
    }


# ── CLI ────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the weekly portfolio summary")
    parser.add_argument("--force", action="store_true", help="Run even if disabled in agent-config")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt + plan, no writes")
    parser.add_argument("--model", default=None, help="Override Claude model")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = get_agent_config(AGENT_NAME)
    if not cfg.enabled and not args.force:
        logger.info("[weekly-summary] disabled by .claude/agent-config.json — exiting cleanly")
        return 0

    from edgefinder.db.engine import get_engine, get_session_factory

    session = get_session_factory(get_engine())()
    try:
        result = run_weekly_summary(
            session,
            dry_run=args.dry_run,
            model=args.model or os.getenv("WEEKLY_SUMMARY_MODEL") or DEFAULT_MODEL,
        )
    finally:
        session.close()

    logger.info("[weekly-summary] %s", json.dumps(result, default=str)[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
