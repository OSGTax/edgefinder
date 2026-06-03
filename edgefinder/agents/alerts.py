"""Alerting — project unresolved CRITICAL observations to GitHub issues.

The watchdog records findings to ``agent_observations`` but does not page
anyone. This module is the alerting layer: it opens a GitHub issue for each
unresolved CRITICAL observation and closes the issue once the condition
clears. The GitHub issue is a pure *projection* of DB state — never the
source of truth — so the loop is fully idempotent and self-healing.

Issues are deduplicated by a deterministic title derived from the
observation's category + metadata key, under a dedicated ``edgefinder-alert``
label so alert issues never collide with human-filed ones.

Designed to run from a GitHub Actions cron where the ``gh`` CLI is
pre-installed and authenticated via the workflow ``GITHUB_TOKEN``. The
``_gh`` shell-out takes an injectable ``runner`` so the reconciliation logic
is unit-testable without GitHub (mirrors ``coach._gh``).

CLI:
    python -m edgefinder.agents.alerts [--dry-run] [--force] [--ignore-window]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from typing import Any, Iterable

from sqlalchemy.orm import Session

from edgefinder.agents.config import get_agent_config
from edgefinder.agents.watchdog import AGENT_NAME, is_in_active_window
from edgefinder.db.models import AgentObservation

logger = logging.getLogger(__name__)

ALERT_LABEL = "edgefinder-alert"
TITLE_PREFIX = "[edgefinder] CRITICAL: "


def _gh(args: list[str], runner: Any | None = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a ``gh`` CLI command. Injectable ``runner`` for tests."""
    runner = runner or subprocess.run
    return runner(["gh", *args], capture_output=True, text=True, check=True, **kwargs)


def _dedup_label(obs: AgentObservation) -> str:
    """Stable identity for an observation across ticks: category[/key]."""
    key = (obs.obs_metadata or {}).get("key")
    return f"{obs.category}/{key}" if key is not None else obs.category


def _title_for(obs: AgentObservation) -> str:
    return TITLE_PREFIX + _dedup_label(obs)


def _body_for(obs: AgentObservation) -> str:
    meta = json.dumps(obs.obs_metadata or {}, indent=2, sort_keys=True)
    return (
        f"**{obs.severity}** · `{obs.category}`\n\n"
        f"{obs.message}\n\n"
        f"- agent: `{obs.agent_name}`\n"
        f"- first seen: {obs.timestamp}\n\n"
        f"```json\n{meta}\n```\n\n"
        "_Filed automatically by edgefinder alerts. Auto-closes when the "
        "watchdog marks the underlying observation resolved._"
    )


def sync_alerts(
    session: Session,
    runner: Any | None = None,
    categories: Iterable[str] | None = None,
) -> dict[str, int]:
    """Reconcile open ``edgefinder-alert`` issues with unresolved CRITICALs.

    - unresolved CRITICAL with no matching open issue → ``gh issue create``
    - open alert issue with no matching unresolved CRITICAL → ``gh issue close``

    ``categories`` optionally restricts which CRITICAL categories page
    (default: all). Returns {opened, closed, active}.
    """
    query = session.query(AgentObservation).filter(
        AgentObservation.severity == "CRITICAL",
        AgentObservation.resolved_at.is_(None),
    )
    if categories is not None:
        query = query.filter(AgentObservation.category.in_(list(categories)))

    # Deterministic title per dedup identity; if two rows share an identity
    # (shouldn't, given watchdog reconciliation) the last simply wins.
    desired: dict[str, AgentObservation] = {_title_for(o): o for o in query.all()}

    listed = _gh(
        ["issue", "list", "--state", "open", "--label", ALERT_LABEL,
         "--json", "number,title", "--limit", "100"],
        runner=runner,
    )
    open_by_title: dict[str, int] = {
        e["title"]: e["number"] for e in json.loads(listed.stdout or "[]")
    }

    opened = closed = 0
    for title, obs in desired.items():
        if title in open_by_title:
            continue
        try:
            _gh(["issue", "create", "--title", title, "--label", ALERT_LABEL,
                 "--body", _body_for(obs)], runner=runner)
            opened += 1
        except Exception:
            logger.exception("Failed to open alert issue: %s", title)

    for title, number in open_by_title.items():
        if title in desired:
            continue
        try:
            _gh(["issue", "close", str(number), "--comment",
                 "Auto-resolved: the underlying condition cleared "
                 "(watchdog reconciliation)."], runner=runner)
            closed += 1
        except Exception:
            logger.exception("Failed to close alert issue #%s", number)

    return {"opened": opened, "closed": closed, "active": len(desired)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile unresolved CRITICAL observations to GitHub issues"
    )
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be opened; make no GitHub writes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run even if agent-config.json disables the agent",
    )
    parser.add_argument(
        "--ignore-window", action="store_true",
        help="Run even outside the active market window",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.ignore_window and not is_in_active_window():
        logger.info("[alerts] outside active window — exiting cleanly")
        return 0

    cfg = get_agent_config(args.agent_name)
    if not cfg.enabled and not args.force:
        logger.info(
            "[alerts] disabled by .claude/agent-config.json — exiting cleanly "
            "(use --force to override)"
        )
        return 0

    from edgefinder.db.engine import get_engine, get_session_factory

    session = get_session_factory(get_engine())()
    try:
        if args.dry_run:
            unresolved = session.query(AgentObservation).filter(
                AgentObservation.severity == "CRITICAL",
                AgentObservation.resolved_at.is_(None),
            ).all()
            for obs in unresolved:
                logger.info("[alerts][DRY] would ensure issue: %s", _title_for(obs))
            logger.info("[alerts][DRY] %d unresolved CRITICAL observation(s)", len(unresolved))
            return 0

        summary = sync_alerts(session)
        logger.info(
            "[alerts] opened=%d closed=%d active=%d",
            summary["opened"], summary["closed"], summary["active"],
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
