#!/usr/bin/env python3
"""The trading-agent workflow's gate — stdlib only, runs before Claude.

Decides whether a workflow run should actually invoke a trading cycle, so
scheduled (cron-floor) runs with nothing due cost seconds of runner time
and zero Claude tokens. Also the failure reporter: --report-failure writes
a desk_journal note so a dead OAuth token / PAT is loud on the desk, not
silent in a red checkmark nobody watches.

Decision (mirrors agent/streamer.py dispatch_reason and brain.wake_due —
ONE definition of "due"):
  - event workflow_dispatch  -> run (the dispatcher or the owner asked)
  - event schedule           -> run iff an unhonored wake-plan is due
    within the same 8h lookback the skill's wake-due uses
  - Supabase unreachable     -> should_run=false (preflight would abort
    the cycle anyway; don't pay Claude to discover an outage)

Output: `should_run=true|false` appended to $GITHUB_OUTPUT (or stdout).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

LOOKBACK_HOURS = 8.0  # keep in lockstep with brain.wake_due / dispatch_reason


def _rest(path: str, params: list[tuple[str, str]]) -> list[dict]:
    # params is a LIST of pairs: PostgREST ranges repeat the same key
    # (at=gte.X&at=lte.Y), which a dict cannot express.
    base = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/"
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    url = base + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "apikey": key, "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def due_wakes_exist(now: datetime | None = None) -> bool:
    now = now or _now()
    lookback = now - timedelta(hours=LOOKBACK_HOURS)
    rows = _rest("desk_wakes", [
        ("select", "id"),
        ("account", "eq.agent"),
        ("honored_run_id", "is.null"),
        ("at", f"gte.{lookback.isoformat()}"),
        ("at", f"lte.{now.isoformat()}"),
        ("limit", "5"),
    ])
    return bool(rows)


def decide(event: str) -> bool:
    if event in ("workflow_dispatch", "repository_dispatch"):
        return True
    try:
        return due_wakes_exist()
    except Exception as exc:  # noqa: BLE001 — outage => don't spend tokens
        print(f"gate: supabase unreachable ({exc}); should_run=false",
              file=sys.stderr)
        return False


def report_failure(run_id: str) -> None:
    base = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/desk_journal"
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    body = json.dumps({
        "account": "agent", "kind": "note",
        "title": "Autonomous trading-cycle run FAILED",
        "body": (f"GitHub Actions run {run_id} failed. Most likely causes: "
                 "expired CLAUDE_CODE_OAUTH_TOKEN (regenerate with `claude "
                 "setup-token`), a missing repo secret, or a mid-cycle "
                 "crash — open the run's log in the repo's Actions tab. "
                 "The claude.ai Routine floor and manual fires still work."),
        "ts": _now().isoformat(),
    }).encode()
    req = urllib.request.Request(base, data=body, method="POST", headers={
        "apikey": key, "Authorization": f"Bearer {key}",
        "Content-Type": "application/json", "Prefer": "return=minimal"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"failure journaled ({resp.status})")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--report-failure":
        report_failure(sys.argv[2] if len(sys.argv) > 2 else "unknown")
        return
    event = os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    should = decide(event)
    line = f"should_run={'true' if should else 'false'}"
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
