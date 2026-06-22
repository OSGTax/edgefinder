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

    from agent.data import session_factory
    from agent.models import ACCOUNT, DeskChangelog

    sess = session_factory()()
    try:
        row = DeskChangelog(account=ACCOUNT, kind=kind, title=title[:160],
                            detail=(detail or "").strip() or None,
                            version=(version or None), run_id=(run_id or None))
        sess.add(row)
        sess.commit()
        return int(row.id)
    finally:
        sess.close()


def recent(limit: int = 20) -> list[dict]:
    """Recent entries, newest first (for --list / sanity checks)."""
    from sqlalchemy import desc

    from agent.data import session_factory
    from agent.models import ACCOUNT, DeskChangelog

    sess = session_factory()()
    try:
        rows = (sess.query(DeskChangelog)
                .filter(DeskChangelog.account == ACCOUNT)
                .order_by(desc(DeskChangelog.ts)).limit(limit).all())
        return [{"id": r.id, "ts": r.ts.isoformat() if r.ts else None,
                 "kind": r.kind, "title": r.title, "detail": r.detail,
                 "version": r.version, "run_id": r.run_id} for r in rows]
    finally:
        sess.close()


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
