"""The agent's evolving mind — strategy state, journal, thinking, decisions.

These are the write-side tools the charter skill calls each cycle to (a)
read/update its living strategy, (b) journal a pivot with its reasoning, (c)
stream its thinking for the live feed, and (d) save the run's decision +
watchlist. Persistence goes through ``agent.store`` (pg or rest transport).

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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store():
    from agent.store import get_store

    return get_store()


def _load_json(path: str | None):
    if not path:
        return None
    with open(path) as fh:
        return json.load(fh)


# ── strategy state ──────────────────────────────────────


def _latest_state_row(store, account: str):
    rows = store.select("desk_strategy_state", filters={"account": account},
                        order=[("version", "desc"), ("id", "desc")], limit=1)
    return rows[0] if rows else None


def get_state(store=None, account: str = "agent") -> dict:
    store = store or _store()
    row = _latest_state_row(store, account)
    if not row:
        return {"exists": False, "version": 0, "name": None, "thesis": None,
                "rules": {}, "params": {}}
    return {"exists": True, "version": row["version"], "name": row["name"],
            "thesis": row.get("thesis"), "rules": row.get("rules") or {},
            "params": row.get("params") or {},
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None}


def set_state(store=None, *, name: str, thesis: str | None = None,
              rules: dict | None = None, params: dict | None = None,
              bump: bool = False, account: str = "agent") -> dict:
    """Update the living strategy. ``bump`` writes a NEW version row (a pivot);
    otherwise the latest row is updated in place (a tweak)."""
    store = store or _store()
    cur = get_state(store, account)
    if bump or not cur["exists"]:
        version = (cur["version"] or 0) + 1
        store.insert("desk_strategy_state", {
            "account": account, "version": version, "name": name, "thesis": thesis,
            "rules": rules or {}, "params": params or {}, "updated_at": _utcnow()},
            returning=False)
    else:
        row = _latest_state_row(store, account)
        values: dict = {"name": name, "updated_at": _utcnow()}
        if thesis is not None:
            values["thesis"] = thesis
        if rules is not None:
            values["rules"] = rules
        if params is not None:
            values["params"] = params
        store.update("desk_strategy_state", {"id": row["id"]}, values, returning=False)
        version = row["version"]
    return {"ok": True, "version": version, "name": name}


# ── journal ─────────────────────────────────────────────


def add_journal(store=None, *, kind: str, title: str, body: str | None = None,
                version_from: int | None = None, version_to: int | None = None,
                account: str = "agent") -> dict:
    store = store or _store()
    rows = store.insert("desk_journal", {
        "account": account, "kind": kind, "title": title, "body": body,
        "version_from": version_from, "version_to": version_to, "ts": _utcnow()})
    return {"ok": True, "id": (rows[0]["id"] if rows else None)}


# ── thinking feed ───────────────────────────────────────


def think(store=None, *, run_id: str, text: str, phase: str | None = None,
          account: str = "agent") -> dict:
    store = store or _store()
    rows = store.insert("desk_thinking", {
        "account": account, "run_id": run_id, "phase": phase, "text": text,
        "ts": _utcnow()})
    return {"ok": True, "id": (rows[0]["id"] if rows else None)}


# ── decision ────────────────────────────────────────────


def save_decision(store=None, *, run_id: str, regime: str | None = None,
                  summary: str | None = None, target_weights: dict | None = None,
                  picks: list | None = None, watchlist: list | None = None,
                  strategy_version: int | None = None,
                  decision_date: date | None = None, account: str = "agent") -> dict:
    store = store or _store()
    values = {
        "decision_date": decision_date or date.today(), "regime": regime,
        "summary": summary, "target_weights": target_weights, "picks": picks,
        "watchlist": watchlist, "strategy_version": strategy_version}
    existing = store.select("desk_decisions",
                            filters={"account": account, "run_id": run_id}, limit=1)
    if existing:
        store.update("desk_decisions", {"id": existing[0]["id"]}, values,
                     returning=False)
        return {"ok": True, "run_id": run_id, "id": existing[0]["id"]}
    rows = store.insert("desk_decisions", {
        "account": account, "run_id": run_id, "ts": _utcnow(), **values})
    return {"ok": True, "run_id": run_id, "id": (rows[0]["id"] if rows else None)}


def main(argv: list[str] | None = None) -> None:
    import argparse

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
    store = _store()
    if args.cmd == "state-get":
        print(json.dumps(get_state(store), indent=2))
    elif args.cmd == "state-set":
        print(json.dumps(set_state(
            store, name=args.name, thesis=args.thesis,
            rules=_load_json(args.rules_file), params=_load_json(args.params_file),
            bump=args.bump), indent=2))
    elif args.cmd == "journal":
        print(json.dumps(add_journal(
            store, kind=args.kind, title=args.title, body=args.body,
            version_from=args.vfrom, version_to=args.vto), indent=2))
    elif args.cmd == "think":
        print(json.dumps(think(store, run_id=args.run_id, phase=args.phase,
                               text=args.text), indent=2))
    elif args.cmd == "decision":
        print(json.dumps(save_decision(
            store, run_id=args.run_id, regime=args.regime, summary=args.summary,
            target_weights=_load_json(args.weights_file),
            picks=_load_json(args.picks_file),
            watchlist=_load_json(args.watchlist_file),
            strategy_version=args.strategy_version), indent=2))


if __name__ == "__main__":
    main()
