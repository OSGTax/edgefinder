#!/usr/bin/env python3
"""Email the owner a brief report when a GitHub-fired trading cycle finishes.

Runs as the LAST step of trading-agent.yml (after the Claude step, success
or failure). Reads what THIS run wrote to the desk tables — the decision,
any fills, the next planned check-in — plus the latest equity mark, and
sends a short plain-text email. Stdlib only, same as wake_gate.py.

Configuration is entirely via repo secrets exposed as env vars:
  CYCLE_REPORT_TO   recipient address           (unset => step no-ops)
  SMTP_USERNAME     sending mailbox login       (unset => step no-ops)
  SMTP_PASSWORD     its app password
  SMTP_HOST         default smtp.gmail.com
  SMTP_PORT         default 465 (SSL)
Missing config is a silent, successful no-op by design — the report is a
convenience and must never fail the trading run.
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _rest(path: str, params: list[tuple[str, str]]) -> list[dict]:
    base = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/"
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    url = base + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "apikey": key, "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def gather(gh_run_id: str, now: datetime | None = None) -> dict:
    """Everything the email needs, matched to this workflow run.

    Cycles append -gha<run_id> to their run ids; when a cycle forgot the
    suffix (it happens), fall back to anything written in the last 40 min
    so the report degrades to 'latest activity' instead of 'nothing'.
    """
    now = now or _now()
    pat = f"like.*gha{gh_run_id}*"
    recent = ("gte", (now - timedelta(minutes=40)).isoformat())

    def rows(table, select, filters, order, limit):
        params = [("select", select), ("account", "eq.agent")]
        params += filters
        params += [("order", order), ("limit", str(limit))]
        return _rest(table, params)

    dec = rows("desk_decisions", "ts,run_id,summary", [("run_id", pat)],
               "ts.desc", 1)
    matched = bool(dec)
    if not dec:
        dec = rows("desk_decisions", "ts,run_id,summary",
                   [("ts", f"gte.{recent[1]}")], "ts.desc", 1)
    fills_filter = [("run_id", pat)] if matched else [("ts", f"gte.{recent[1]}")]
    fills = rows("desk_trades", "ts,symbol,side,shares,qty,price,notional",
                 fills_filter, "ts.asc", 12)
    nxt = rows("desk_wakes", "at,reason", [("honored_run_id", "is.null"),
               ("at", f"gte.{now.isoformat()}")], "at.asc", 1)
    eq = rows("desk_equity", "ts,equity,return_pct", [], "ts.desc", 1)
    return {"decision": dec[0] if dec else None, "matched_run": matched,
            "fills": fills, "next_wake": nxt[0] if nxt else None,
            "equity": eq[0] if eq else None}


def _fmt_et(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))
        return dt.replace(tzinfo=timezone.utc).astimezone(ET).strftime(
            "%-I:%M %p ET")
    except Exception:  # noqa: BLE001 — a bad timestamp must not kill the email
        return ts


def compose(gh_run_id: str, conclusion: str, facts: dict,
            now: datetime | None = None) -> tuple[str, str]:
    """(subject, body) — plain text, brief, verdict first."""
    now = now or _now()
    when = now.replace(tzinfo=timezone.utc).astimezone(ET).strftime(
        "%a %-I:%M %p ET")
    d = facts.get("decision")
    lines: list[str] = []

    if conclusion and conclusion != "success":
        headline = f"cycle FAILED ({conclusion})"
        lines.append(f"The trading run finished with status {conclusion} — "
                     "open the log link below. The desk journal has the "
                     "likely cause; the next wake or floor slot retries.")
    elif d:
        summary = (d.get("summary") or "").strip()
        headline = summary[:70] or "cycle complete"
        lines.append(f"Decision ({_fmt_et(d.get('ts') or '')}): {summary}")
        if not facts.get("matched_run"):
            lines.append("(Note: matched by time, not run id — the cycle "
                         "skipped its -gha run-id suffix.)")
    else:
        headline = "run finished; no cycle content found"
        lines.append("The workflow finished but no decision row was found "
                     "for it — likely a gated no-op or a stand-down.")

    fills = facts.get("fills") or []
    if fills:
        lines.append("Fills:")
        for f in fills:
            qty = f.get("shares") or f.get("qty") or ""
            lines.append(f"  {(f.get('side') or '').upper()} {f.get('symbol')}"
                         f" x{qty} @ {f.get('price')}"
                         f" (${round(float(f.get('notional') or 0)):,})")
    else:
        lines.append("No fills this cycle.")

    eq = facts.get("equity")
    if eq:
        lines.append(f"Equity: ${float(eq.get('equity') or 0):,.0f} "
                     f"({float(eq.get('return_pct') or 0):+.2f}% since "
                     f"inception), marked {_fmt_et(eq.get('ts') or '')}.")

    nxt = facts.get("next_wake")
    if nxt:
        lines.append(f"Next check-in: {_fmt_et(str(nxt.get('at')))} — "
                     f"{(nxt.get('reason') or '')[:120]}")
    else:
        lines.append("Next check-in: none pending (floor slots / tripwires).")

    lines.append(f"Log: https://github.com/OSGTax/edgefinder/actions/runs/"
                 f"{gh_run_id}")
    lines.append("Desk: https://edgefinder-pm8h.onrender.com/desk")
    return f"EdgeFinder cycle {when} — {headline}", "\n".join(lines)


def send(subject: str, body: str) -> bool:
    to = os.environ.get("CYCLE_REPORT_TO", "").strip()
    user = os.environ.get("SMTP_USERNAME", "").strip()
    pwd = os.environ.get("SMTP_PASSWORD", "").strip()
    if not (to and user and pwd):
        print("cycle_report: email not configured (CYCLE_REPORT_TO / "
              "SMTP_USERNAME / SMTP_PASSWORD) — skipping, not an error")
        return False
    host = os.environ.get("SMTP_HOST", "").strip() or "smtp.gmail.com"
    port = int(os.environ.get("SMTP_PORT", "").strip() or "465")
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(),
                          timeout=30) as smtp:
        smtp.login(user, pwd)
        smtp.send_message(msg)
    print(f"cycle_report: sent to {to}")
    return True


def main() -> None:
    gh_run_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    conclusion = sys.argv[2] if len(sys.argv) > 2 else "success"
    try:
        facts = gather(gh_run_id)
    except Exception as exc:  # noqa: BLE001 — report the outage, don't crash
        facts = {"decision": None, "matched_run": False, "fills": [],
                 "next_wake": None, "equity": None}
        print(f"cycle_report: gather failed ({exc}); sending bare notice",
              file=sys.stderr)
    subject, body = compose(gh_run_id, conclusion, facts)
    try:
        send(subject, body)
    except Exception as exc:  # noqa: BLE001 — never fail the trading run
        print(f"cycle_report: send failed ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
