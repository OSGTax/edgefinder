"""CLI: promote engine-v2 strategies to self-running paper trading.

Two tiers:
- ``validated`` — requires the strategy's latest validation_runs row to have
  criteria.all_met AND an evaluated, passing sealed holdout. The real bar.
- ``experimental`` — explicit opt-in to paper-trade WITHOUT a passing
  backtest (the "throw it in and watch it run" stage). The tier is stored
  and displayed so the two are never confused.

Examples:

    python -m edgefinder.engine.promote --list
    python -m edgefinder.engine.promote --spec equal_weight \
        --symbols SPY,QQQ,IWM,DIA,GLD,TLT,EFA --schedule monthly \
        --tier experimental
    python -m edgefinder.engine.promote --demote equal_weight
"""

from __future__ import annotations

import argparse

from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.db.models import PromotedStrategy, ValidationRun
from edgefinder.engine.strategies import make_strategy_factory


def _latest_validation(session, strategy_name: str) -> ValidationRun | None:
    return (session.query(ValidationRun)
            .filter(ValidationRun.strategy_name == strategy_name)
            .order_by(ValidationRun.run_at.desc(), ValidationRun.id.desc())
            .first())


def is_validated(run: ValidationRun | None) -> bool:
    """The dashboard's rule: criteria.all_met AND evaluated holdout passes."""
    return bool(run is not None
                and (run.criteria or {}).get("all_met")
                and run.holdout is not None
                and run.holdout.get("passes"))


def promote(session, *, spec: str, symbols: list[str], schedule: str,
            tier: str) -> PromotedStrategy:
    """Create (or reactivate) a promotion row; raises on a failed gate."""
    name = make_strategy_factory(spec)().name
    run = _latest_validation(session, name)
    if tier == "validated" and not is_validated(run):
        if run is None:
            state = "none"
        else:
            holdout = ("unevaluated" if run.holdout is None
                       else f"passes={run.holdout.get('passes')}")
            state = f"{run.verdict}, holdout {holdout}"
        raise ValueError(
            f"{name!r} is not validated (latest run: {state}). "
            "Pass --tier experimental to paper-trade it anyway.")

    row = (session.query(PromotedStrategy)
           .filter(PromotedStrategy.strategy_name == name).one_or_none())
    if row is None:
        row = PromotedStrategy(strategy_name=name)
        session.add(row)
    row.spec = spec
    row.symbols = symbols
    row.schedule = schedule
    row.tier = tier
    row.validation_run_id = run.id if run is not None else None
    row.prices_basis = (run.config or {}).get("prices") if run is not None else None
    row.active = True
    session.commit()
    return row


def demote(session, name: str) -> bool:
    row = (session.query(PromotedStrategy)
           .filter(PromotedStrategy.strategy_name == name).one_or_none())
    if row is None:
        return False
    row.active = False
    session.commit()
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="show all promotions")
    p.add_argument("--spec", help="strategy spec, e.g. equal_weight or trend_timer:SPY")
    p.add_argument("--symbols", help="comma-separated universe it trades")
    p.add_argument("--schedule", default="monthly",
                   choices=["daily", "weekly", "monthly"])
    p.add_argument("--tier", default="validated",
                   choices=["validated", "experimental"])
    p.add_argument("--demote", metavar="NAME", help="deactivate a promotion")
    args = p.parse_args(argv)

    session = get_session_factory(get_engine())()
    try:
        if args.list:
            rows = session.query(PromotedStrategy).all()
            if not rows:
                print("no promotions")
            for r in rows:
                state = "ACTIVE" if r.active else "demoted"
                print(f"{r.strategy_name:<28} {state:<8} tier={r.tier:<12} "
                      f"{r.schedule:<8} {','.join(r.symbols or [])}")
            return
        if args.demote:
            print("demoted" if demote(session, args.demote)
                  else f"no promotion named {args.demote!r}")
            return
        if not args.spec or not args.symbols:
            p.error("--spec and --symbols are required to promote")
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        try:
            row = promote(session, spec=args.spec, symbols=symbols,
                          schedule=args.schedule, tier=args.tier)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        print(f"promoted {row.strategy_name} (tier={row.tier}, "
              f"{row.schedule}, {len(symbols)} symbols)")
    finally:
        session.close()


if __name__ == "__main__":
    main()
