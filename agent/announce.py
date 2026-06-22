"""announce — the app-evolution routine's tool for the "What's New" surface.

When the agent ships a genuinely useful change to what the dashboard shows, it
records it here. Each entry is a short title + a plain-English explanation of
the feature and why it helps. The /desk page lights a "NEW" badge for recent
entries and lists them in the "What's New" panel, with the explanation, so
users can see how the app is growing.

This is the ONLY way features reach the What's New surface — a UI change that
isn't announced is invisible to users, so the routine announces every change it
ships (and only changes that actually merged/deployed).

CLI:
  python -m agent.announce --title "Drawdown band on the equity curve" \
      --kind feature --version 6.1.0 \
      --detail "The equity chart now shades peak-to-trough drawdowns so you can
                see how deep the book's dips ran and how fast it recovered."
  python -m agent.announce --list           # recent entries (JSON)
"""

from __future__ import annotations

import argparse
import json
import sys

VALID_KINDS = ("feature", "improvement", "data", "disclaimer", "fix")


def announce(title: str, detail: str | None = None, *, kind: str = "feature",
             version: str | None = None, run_id: str | None = None) -> int:
    """Insert one What's-New entry. Returns the new row id."""
    title = (title or "").strip()
    if not title:
        raise ValueError("announce: title is required")
    if kind not in VALID_KINDS:
        raise ValueError(f"announce: kind must be one of {VALID_KINDS}, got {kind!r}")

    from datetime import datetime, timezone

    from agent.models import ACCOUNT
    from agent.store import get_store

    rows = get_store().insert("desk_changelog", {
        "account": ACCOUNT, "kind": kind, "title": title[:160],
        "detail": (detail or "").strip() or None, "version": (version or None),
        "run_id": (run_id or None),
        "ts": datetime.now(timezone.utc).replace(tzinfo=None)})
    return int(rows[0]["id"]) if rows else 0


def recent(limit: int = 20) -> list[dict]:
    """Recent entries, newest first (for --list / sanity checks)."""
    from agent.models import ACCOUNT
    from agent.store import get_store

    rows = get_store().select("desk_changelog", filters={"account": ACCOUNT},
                              order=[("ts", "desc")], limit=limit)
    return [{"id": r.get("id"), "ts": str(r["ts"]) if r.get("ts") else None,
             "kind": r.get("kind"), "title": r.get("title"), "detail": r.get("detail"),
             "version": r.get("version"), "run_id": r.get("run_id")} for r in rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--title")
    p.add_argument("--detail", default=None)
    p.add_argument("--kind", default="feature", choices=VALID_KINDS)
    p.add_argument("--version", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--list", action="store_true", help="print recent entries and exit")
    args = p.parse_args(argv)

    if args.list:
        print(json.dumps(recent(), indent=2))
        return 0
    if not args.title:
        p.error("--title is required (or use --list)")
    new_id = announce(args.title, args.detail, kind=args.kind,
                      version=args.version, run_id=args.run_id)
    print(json.dumps({"announced": new_id, "title": args.title.strip(),
                      "kind": args.kind, "version": args.version}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
