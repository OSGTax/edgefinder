"""Agentic reasoning step for management agents.

The deterministic watchdog tick just writes observations. This module
calls Claude over the current unresolved observations + the agent's
persistent memory + recent context, and returns structured decisions
(assessment + recommended action + reasoning) plus a possibly-updated
memory blob.

This is the *agentic* layer — the piece that actually reasons rather
than just running SQL checks.

CLI:
    python -m edgefinder.agents.reasoning [--agent-name watchdog] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import func
from sqlalchemy.orm import Session

from edgefinder.agents.config import get_agent_config
from edgefinder.agents.journal import record_action
from edgefinder.agents.memory import load_memory, save_memory
from edgefinder.db.models import AgentObservation, TradeRecord

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
MAX_OUTPUT_TOKENS = 4096

# Static system prompt — stable across every tick so we cache it.
SYSTEM_PROMPT = """You are the EdgeFinder watchdog — a management agent for a live paper-trading system. You are one piece of a monitoring pipeline: deterministic SQL checks run first and write findings to agent_observations; you then triage those findings with the full context below.

Your job each tick:
1. For every unresolved observation, assess whether it is actually critical, expected/routine, worth investigating, or a known false positive. Use the agent memory (patterns seen before, known false positives, recent resolutions) to avoid re-escalating things you've already triaged.
2. Recommend an action per observation — escalate, investigate, monitor, or dismiss. Do not execute any action yourself; the orchestrator decides what to do with your recommendation.
3. Update the memory ONLY if you learned something genuinely new this tick — a new pattern, a newly-confirmed false positive, or a resolution that matters for future ticks. If nothing new, return memory_update=null and leave the prior memory intact. Keep the memory under ~2000 words; prune outdated entries when you rewrite.

Hard rules:
- Never hallucinate findings. Every decision must reference an observation_id that's actually in the input.
- Never propose editing trading code, strategies, or the production DB. You are read-mostly.
- Do not recommend escalating every observation — the point of memory is to suppress known noise.
- Severity scale: critical (human attention now) > investigate (worth a look soon) > monitor (note but don't act) > dismiss (known false positive).

Return ONLY JSON matching the required schema. No markdown, no prose outside the JSON."""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "string",
            "description": "One-paragraph human-readable summary of this tick.",
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "observation_id": {"type": "integer"},
                    "assessment": {
                        "type": "string",
                        "enum": ["critical", "expected", "needs_investigation", "false_positive"],
                    },
                    "action": {
                        "type": "string",
                        "enum": ["escalate", "investigate", "monitor", "dismiss"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["observation_id", "assessment", "action", "reasoning"],
            },
        },
        "memory_update": {
            "type": ["string", "null"],
            "description": "Full new memory content, or null to leave unchanged.",
        },
    },
    "required": ["summary", "decisions", "memory_update"],
}


@dataclass
class Decision:
    observation_id: int
    assessment: str  # critical | expected | needs_investigation | false_positive
    action: str      # escalate | investigate | monitor | dismiss
    reasoning: str


@dataclass
class ReasoningResult:
    summary: str
    decisions: list[Decision]
    memory_update: str | None


# ── Context builders ──────────────────────────────────────


def _serialize_observations(observations: Iterable[AgentObservation]) -> list[dict]:
    return [
        {
            "id": obs.id,
            "severity": obs.severity,
            "category": obs.category,
            "message": obs.message,
            "metadata": obs.obs_metadata or {},
            "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
        }
        for obs in observations
    ]


def _recent_trade_summary(session: Session, hours: int = 24) -> dict[str, Any]:
    """Per-strategy trade counts and P&L over the last N hours."""
    rows = (
        session.query(
            TradeRecord.strategy_name,
            func.count(TradeRecord.id).label("count"),
            func.coalesce(func.sum(TradeRecord.pnl_dollars), 0.0).label("pnl"),
        )
        .filter(TradeRecord.created_at >= datetime.utcnow().replace(microsecond=0))
        .group_by(TradeRecord.strategy_name)
        .all()
    )
    return {r.strategy_name: {"count": r.count, "pnl": round(r.pnl or 0.0, 2)} for r in rows}


def _recent_commits(limit: int = 10) -> list[str]:
    """Shell out to git log for the last N commits touching trading paths.
    Returns an empty list if git isn't available or the call fails —
    reasoning should still work without commit context.
    """
    try:
        result = subprocess.run(
            [
                "git", "log",
                f"-{limit}",
                "--pretty=format:%h %s",
                "--",
                "edgefinder/trading/",
                "edgefinder/strategies/",
                "edgefinder/db/",
                "dashboard/services.py",
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        return []


def _build_user_message(
    observations: list[AgentObservation],
    trade_summary: dict[str, Any],
    commits: list[str],
) -> str:
    return json.dumps(
        {
            "unresolved_observations": _serialize_observations(observations),
            "recent_trades_24h": trade_summary,
            "recent_trading_commits": commits,
            "now_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        default=str,
    )


# ── LLM call ──────────────────────────────────────────────


def _parse_response(raw_text: str) -> ReasoningResult:
    """Parse the model's JSON response into a ReasoningResult."""
    data = json.loads(raw_text)
    decisions = [
        Decision(
            observation_id=int(d["observation_id"]),
            assessment=d["assessment"],
            action=d["action"],
            reasoning=d["reasoning"],
        )
        for d in data.get("decisions", [])
    ]
    return ReasoningResult(
        summary=data.get("summary", ""),
        decisions=decisions,
        memory_update=data.get("memory_update"),
    )


def _extract_json_object(text: str) -> str:
    """Return the first top-level JSON object substring in text.

    The Claude Code CLI prompts for JSON but sometimes returns prose
    wrapping ("Sure, here's the JSON: {...}"). Strip to the first
    balanced braces. Fail hard if none found so the caller sees the
    raw output.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def _call_claude_cli(
    system_prompt: str,
    memory: str,
    user_content: str,
    model: str,
    runner: Any | None = None,
) -> str:
    """Invoke `claude -p` and return the inner JSON text.

    Consumes the Claude Pro/Max/Team/Enterprise subscription quota via
    `CLAUDE_CODE_OAUTH_TOKEN` — no Anthropic API key required. The
    prompt is piped via stdin (argv has length limits). `runner`
    defaults to subprocess.run; tests inject a stub.
    """
    runner = runner or subprocess.run

    if not os.getenv("CLAUDE_CODE_OAUTH_TOKEN"):
        raise RuntimeError(
            "CLAUDE_CODE_OAUTH_TOKEN not set. Generate locally with "
            "`claude setup-token` and add it as a GitHub Actions secret."
        )

    schema_str = json.dumps(RESPONSE_SCHEMA, indent=2)
    prompt = (
        f"{system_prompt}\n\n"
        f"## Required output schema (JSON Schema draft 2020-12)\n\n"
        f"```json\n{schema_str}\n```\n\n"
        f"## Current agent memory\n\n{memory}\n\n"
        f"## Input\n\n{user_content}\n\n"
        "Respond with ONE JSON object that validates against the schema above. "
        "No markdown fences, no commentary, no text outside the JSON."
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
        raise RuntimeError(
            f"claude -p exited {result.returncode}: {result.stderr[:500]}"
        )

    envelope = json.loads(result.stdout)
    response_text = envelope.get("result")
    if not isinstance(response_text, str):
        raise RuntimeError(
            f"Unexpected claude -p envelope (no 'result' string): {result.stdout[:300]}"
        )
    return _extract_json_object(response_text)


def reason_over_tick(
    observations: list[AgentObservation],
    memory: str,
    trade_summary: dict[str, Any] | None = None,
    commits: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    cli_runner: Any | None = None,
) -> ReasoningResult:
    """Call Claude over the tick's context and return structured decisions.

    Uses `claude -p` with the user's Claude subscription token. No
    Anthropic API key is used or required.
    """
    user_content = _build_user_message(
        observations,
        trade_summary or {},
        commits or [],
    )
    raw = _call_claude_cli(SYSTEM_PROMPT, memory, user_content, model, cli_runner)
    return _parse_response(raw)


# ── Orchestration ─────────────────────────────────────────


def run_reasoning(
    session: Session,
    agent_name: str = "watchdog",
    model: str | None = None,
    cli_runner: Any | None = None,
) -> ReasoningResult:
    """Full tick: load context, call Claude via `claude -p`, persist."""
    observations = (
        session.query(AgentObservation)
        .filter(
            AgentObservation.agent_name == agent_name,
            AgentObservation.resolved_at.is_(None),
        )
        .all()
    )

    if not observations:
        logger.info("[%s-reasoning] no unresolved observations — skipping LLM call", agent_name)
        return ReasoningResult(summary="No unresolved observations.", decisions=[], memory_update=None)

    memory = load_memory(session, agent_name)
    trade_summary = _recent_trade_summary(session)
    commits = _recent_commits()

    effective_model = model or os.getenv("WATCHDOG_REASONING_MODEL") or DEFAULT_MODEL
    result = reason_over_tick(
        observations=observations,
        memory=memory,
        trade_summary=trade_summary,
        commits=commits,
        model=effective_model,
        cli_runner=cli_runner,
    )

    # Persist the agent's meaningful decisions as AgentActions. "Monitor"
    # and "dismiss" are not worth logging — they produce audit noise.
    for decision in result.decisions:
        if decision.action not in {"escalate", "investigate"}:
            continue
        record_action(
            session,
            agent_name=agent_name,
            action_type="diagnose",
            summary=f"[{decision.assessment}] {decision.reasoning}"[:500],
            status="pending",
            observation_id=decision.observation_id,
            notes=decision.reasoning,
            commit=False,
        )

    if result.memory_update and result.memory_update.strip() != memory.strip():
        save_memory(session, agent_name, result.memory_update)

    session.commit()
    return result


# ── CLI ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one reasoning step over the watchdog's findings")
    parser.add_argument("--agent-name", default="watchdog")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the kill switch disables the agent",
    )
    parser.add_argument(
        "--ignore-window",
        action="store_true",
        help="Run even outside the active market window (Mon-Fri 08:30-17:00 ET)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Claude model (default: WATCHDOG_REASONING_MODEL or claude-opus-4-7)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Mirror the deterministic watchdog's window behavior so the workflow's
    # hourly cron doesn't try to call `claude -p` outside market hours —
    # that was the silent cause of failure emails when the deterministic
    # step cleanly exited but this step fired anyway.
    from edgefinder.agents.watchdog import is_in_active_window

    if not args.ignore_window and not is_in_active_window():
        logger.info(
            "[%s-reasoning] outside active window (Mon-Fri 08:30-17:00 ET) — "
            "exiting cleanly (use --ignore-window to override)",
            args.agent_name,
        )
        return 0

    cfg = get_agent_config(args.agent_name)
    if not cfg.enabled and not args.force:
        logger.info(
            "[%s-reasoning] disabled by agent-config.json — exiting cleanly",
            args.agent_name,
        )
        return 0

    from edgefinder.db.engine import get_engine, get_session_factory

    engine = get_engine()
    session = get_session_factory(engine)()
    try:
        result = run_reasoning(session, agent_name=args.agent_name, model=args.model)
        logger.info("[%s-reasoning] %s", args.agent_name, result.summary)
        for d in result.decisions:
            logger.info(
                "[%s-reasoning] obs=%d assessment=%s action=%s — %s",
                args.agent_name, d.observation_id, d.assessment, d.action, d.reasoning,
            )
        if result.memory_update:
            logger.info("[%s-reasoning] memory updated", args.agent_name)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
