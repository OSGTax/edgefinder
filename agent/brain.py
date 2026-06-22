"""The agent's evolving mind — strategy state, journal, thinking, decisions.

These are the write-side tools the charter skill calls each cycle to (a)
read/update its living strategy, (b) journal a pivot with its reasoning, (c)
stream its thinking for the live feed, and (d) save the run's decision +
watchlist. JSON payloads (rules, params, weights, picks, watchlist) are passed
as files to avoid shell-escaping large blobs.

CLI:
  python -m agent.brain state-get
  python -m agent.brain state-set --name "trend+momentum" --thesis "..." \
      --rules-file rules.json --params-file params.json --bump
  python -m agent.brain journal --kind pivot --title "..." --body "..." --to 2
  python -m agent.brain think --run-id R --phase research --text "..."
  python -m agent.brain decision --run-id R --regime risk_on --summary "..." \
      --weights-file w.json --picks-file picks.json --watchlist-file wl.json
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from sqlalchemy import desc
from sqlalchemy.orm import Session

from agent.models import (
    ACCOUNT,
    DeskDecision,
    DeskJournal,
    DeskStrategyState,
    DeskThinking,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: str | None):
    if not path:
        return None
    with open(path) as fh:
        return json.load(fh)


# ── strategy state ──────────────────────────────────────


def get_state(session: Session, account: str = ACCOUNT) -> dict:
    row = (session.query(DeskStrategyState)
           .filter(DeskStrategyState.account == account)
           .order_by(desc(DeskStrategyState.version), desc(DeskStrategyState.id))
           .first())
    if not row:
        return {"exists": False, "version": 0, "name": None, "thesis": None,
                "rules": {}, "params": {}}
    return {"exists": True, "version": row.version, "name": row.name,
            "thesis": row.thesis, "rules": row.rules or {}, "params": row.params or {},
            "updated_at": row.updated_at.isoformat() if row.updated_at else None}


def set_state(session: Session, *, name: str, thesis: str | None = None,
              rules: dict | None = None, params: dict | None = None,
              bump: bool = False, account: str = ACCOUNT) -> dict:
    """Update the living strategy. ``bump`` writes a NEW version row (a pivot);
    otherwise the latest row is updated in place (a tweak)."""
    cur = get_state(session, account)
    if bump or not cur["exists"]:
        version = (cur["version"] or 0) + 1
        row = DeskStrategyState(
            account=account, version=version, name=name, thesis=thesis,
            rules=rules or {}, params=params or {}, updated_at=_utcnow())
        session.add(row)
    else:
        row = (session.query(DeskStrategyState)
               .filter(DeskStrategyState.account == account)
               .order_by(desc(DeskStrategyState.version), desc(DeskStrategyState.id))
               .first())
        row.name = name
        if thesis is not None:
            row.thesis = thesis
        if rules is not None:
            row.rules = rules
        if params is not None:
            row.params = params
        row.updated_at = _utcnow()
        version = row.version
    session.commit()
    return {"ok": True, "version": version, "name": name}


# ── journal ─────────────────────────────────────────────


def add_journal(session: Session, *, kind: str, title: str, body: str | None = None,
                version_from: int | None = None, version_to: int | None = None,
                account: str = ACCOUNT) -> dict:
    row = DeskJournal(account=account, kind=kind, title=title, body=body,
                      version_from=version_from, version_to=version_to, ts=_utcnow())
    session.add(row)
    session.commit()
    return {"ok": True, "id": row.id}


# ── thinking feed ───────────────────────────────────────


def think(session: Session, *, run_id: str, text: str, phase: str | None = None,
          account: str = ACCOUNT) -> dict:
    row = DeskThinking(account=account, run_id=run_id, phase=phase, text=text, ts=_utcnow())
    session.add(row)
    session.commit()
    return {"ok": True, "id": row.id}


# ── decision ────────────────────────────────────────────


def save_decision(session: Session, *, run_id: str, regime: str | None = None,
                  summary: str | None = None, target_weights: dict | None = None,
                  picks: list | None = None, watchlist: list | None = None,
                  strategy_version: int | None = None,
                  decision_date: date | None = None, account: str = ACCOUNT) -> dict:
    existing = (session.query(DeskDecision)
                .filter(DeskDecision.account == account, DeskDecision.run_id == run_id)
                .one_or_none())
    if existing is None:
        existing = DeskDecision(account=account, run_id=run_id, ts=_utcnow())
        session.add(existing)
    existing.decision_date = decision_date or date.today()
    existing.regime = regime
    existing.summary = summary
    existing.target_weights = target_weights
    existing.picks = picks
    existing.watchlist = watchlist
    existing.strategy_version = strategy_version
    session.commit()
    return {"ok": True, "run_id": run_id, "id": existing.id}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from agent.data import session_factory

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state-get")

    ss = sub.add_parser("state-set")
    ss.add_argument("--name", required=True)
    ss.add_argument("--thesis", default=None)
    ss.add_argument("--rules-file", default=None)
    ss.add_argument("--params-file", default=None)
    ss.add_argument("--bump", action="store_true")

    jr = sub.add_parser("journal")
    jr.add_argument("--kind", required=True, choices=["pivot", "tweak", "note"])
    jr.add_argument("--title", required=True)
    jr.add_argument("--body", default=None)
    jr.add_argument("--from", dest="vfrom", type=int, default=None)
    jr.add_argument("--to", dest="vto", type=int, default=None)

    th = sub.add_parser("think")
    th.add_argument("--run-id", required=True)
    th.add_argument("--phase", default=None)
    th.add_argument("--text", required=True)

    de = sub.add_parser("decision")
    de.add_argument("--run-id", required=True)
    de.add_argument("--regime", default=None)
    de.add_argument("--summary", default=None)
    de.add_argument("--weights-file", default=None)
    de.add_argument("--picks-file", default=None)
    de.add_argument("--watchlist-file", default=None)
    de.add_argument("--strategy-version", type=int, default=None)

    args = p.parse_args(argv)
    sess = session_factory()()
    try:
        if args.cmd == "state-get":
            print(json.dumps(get_state(sess), indent=2))
        elif args.cmd == "state-set":
            print(json.dumps(set_state(
                sess, name=args.name, thesis=args.thesis,
                rules=_load_json(args.rules_file), params=_load_json(args.params_file),
                bump=args.bump), indent=2))
        elif args.cmd == "journal":
            print(json.dumps(add_journal(
                sess, kind=args.kind, title=args.title, body=args.body,
                version_from=args.vfrom, version_to=args.vto), indent=2))
        elif args.cmd == "think":
            print(json.dumps(think(sess, run_id=args.run_id, phase=args.phase,
                                   text=args.text), indent=2))
        elif args.cmd == "decision":
            print(json.dumps(save_decision(
                sess, run_id=args.run_id, regime=args.regime, summary=args.summary,
                target_weights=_load_json(args.weights_file),
                picks=_load_json(args.picks_file),
                watchlist=_load_json(args.watchlist_file),
                strategy_version=args.strategy_version), indent=2))
    finally:
        sess.close()


if __name__ == "__main__":
    main()
