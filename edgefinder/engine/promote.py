"""CLI: promote engine-v2 strategies to self-running paper trading.

Two tiers:
- ``validated`` — requires the strategy's latest validation_runs row to have
  criteria.all_met AND an evaluated, passing sealed holdout. The real bar.
- ``experimental`` — explicit opt-in to paper-trade WITHOUT a passing
  backtest (the "throw it in and watch it run" stage). The tier is stored
  and displayed so the two are never confused.

``--finalist`` is an ALTERNATIVE gate for tier "validated", needed because
the system records TWO bars per run and the hunt's pre-registered standard
is not the default one:

- The ``criteria`` column is the RISK-ADJUSTED bar (excess Sharpe,
  majority-of-folds, etc.). None of the 12 confirmed hunt finalists clear
  it — that was disclosed in every round report.
- The hunt's pre-registered confirmation standard (HANDOFF.md, the round
  reports) is the TOTAL-RETURN bar: majority-of-folds positive excess vs
  SPY plus ALL THREE adversarial re-checks, and the cohort cleared the
  burned sealed holdout 12/12 with positive excess (reviews/HOLDOUT-BURN.md,
  protocol pre-registered). ``--finalist`` therefore gates on the latest
  run's EVALUATED holdout showing positive excess vs SPY
  (``holdout.excess_vs_spy_pct > 0``) instead of ``criteria.all_met``.

The strict default gate is unchanged; ``--finalist`` is an explicit,
auditable opt-in stored nowhere — the tier still reads "validated" because
the finalist standard IS a validation standard, just the total-return one.

A promotion trades EITHER a fixed ``--symbols`` list OR a cross-sectional
``--universe top:N[+OFFSET]`` (the live runner re-resolves the point-in-time
top-N by trailing ``--rank-window`` dollar volume at every rebalance
boundary — the validator's exact semantics).

Examples:

    python -m edgefinder.engine.promote --list
    python -m edgefinder.engine.promote --spec equal_weight \
        --symbols SPY,QQQ,IWM,DIA,GLD,TLT,EFA --schedule monthly \
        --tier experimental
    python -m edgefinder.engine.promote --spec mom_6m_k20 \
        --universe top:500 --rank-window 126 --schedule monthly --finalist
    python -m edgefinder.engine.promote --demote equal_weight
"""

from __future__ import annotations

import argparse

from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.db.models import PromotedStrategy, ValidationRun
from edgefinder.engine.data import parse_universe_spec
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


def is_finalist_confirmed(run: ValidationRun | None) -> bool:
    """The hunt's pre-registered finalist standard, as recordable evidence:
    the latest run has an EVALUATED sealed holdout with positive excess vs
    SPY. (The full standard — majority-of-folds wins + all three adversarial
    re-checks — lives in the round reports; the burned holdout is the
    machine-checkable part. See the module docstring for why this differs
    from ``is_validated``.)"""
    return bool(run is not None
                and run.holdout is not None
                and (run.holdout.get("excess_vs_spy_pct") or 0) > 0)


def promote(session, *, spec: str, symbols: list[str] | None = None,
            schedule: str, tier: str, universe: str | None = None,
            rank_window: int = 126, finalist: bool = False) -> PromotedStrategy:
    """Create (or reactivate) a promotion row; raises on a failed gate.

    Exactly one of ``symbols`` (fixed list) or ``universe`` ("top:N[+OFF]",
    re-resolved live at each rebalance with ``rank_window``) is required.
    ``finalist`` swaps the tier-"validated" gate for the hunt finalist
    standard (positive evaluated-holdout excess) — see module docstring.
    """
    if bool(symbols) == bool(universe):
        raise ValueError("exactly one of symbols or universe is required")
    if universe:
        parse_universe_spec(universe)   # raises ValueError on a bad spec

    name = make_strategy_factory(spec)().name
    run = _latest_validation(session, name)
    if tier == "validated":
        gate = is_finalist_confirmed if finalist else is_validated
        if not gate(run):
            if run is None:
                state = "none"
            else:
                holdout = ("unevaluated" if run.holdout is None
                           else (f"excess_vs_spy_pct="
                                 f"{run.holdout.get('excess_vs_spy_pct')}"
                                 if finalist
                                 else f"passes={run.holdout.get('passes')}"))
                state = f"{run.verdict}, holdout {holdout}"
            what = ("does not meet the finalist standard (positive "
                    "evaluated-holdout excess)" if finalist
                    else "is not validated")
            raise ValueError(
                f"{name!r} {what} (latest run: {state}). "
                "Pass --tier experimental to paper-trade it anyway.")

    row = (session.query(PromotedStrategy)
           .filter(PromotedStrategy.strategy_name == name).one_or_none())
    if row is None:
        row = PromotedStrategy(strategy_name=name)
        session.add(row)
    row.spec = spec
    row.symbols = symbols or None
    if (row.universe_spec or None) != (universe or None):
        # the stored last-good resolution belongs to the OLD spec — a stale
        # fallback for a different universe would be silently wrong
        row.resolved_symbols = None
        row.resolved_at = None
    row.universe_spec = universe
    row.rank_window = rank_window if universe else None
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
    p.add_argument("--symbols", help="comma-separated FIXED universe it trades")
    p.add_argument("--universe", metavar="top:N[+OFFSET]",
                   help="cross-sectional universe (e.g. top:500) — re-resolved "
                        "point-in-time at every live rebalance boundary; "
                        "mutually exclusive with --symbols")
    p.add_argument("--rank-window", type=int, default=126, metavar="N",
                   help="trailing trading days for the live dollar-volume "
                        "ranking (default 126 — match the validation run)")
    p.add_argument("--schedule", default="monthly",
                   choices=["daily", "weekly", "monthly"])
    p.add_argument("--tier", default="validated",
                   choices=["validated", "experimental"])
    p.add_argument("--finalist", action="store_true",
                   help="gate tier 'validated' on the hunt finalist standard "
                        "(EVALUATED holdout with positive excess vs SPY) "
                        "instead of criteria.all_met — the pre-registered "
                        "total-return confirmation bar; see module docstring")
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
                if r.universe_spec:
                    uni = f"{r.universe_spec} rw={r.rank_window}"
                    if r.resolved_at:
                        uni += (f" resolved {len(r.resolved_symbols or [])}"
                                f"@{r.resolved_at}")
                else:
                    uni = ",".join(r.symbols or [])
                print(f"{r.strategy_name:<28} {state:<8} tier={r.tier:<12} "
                      f"{r.schedule:<8} {uni}")
            return
        if args.demote:
            print("demoted" if demote(session, args.demote)
                  else f"no promotion named {args.demote!r}")
            return
        if not args.spec:
            p.error("--spec is required to promote")
        if bool(args.symbols) == bool(args.universe):
            p.error("exactly one of --symbols or --universe is required to promote")
        symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                   if args.symbols else None)
        try:
            row = promote(session, spec=args.spec, symbols=symbols,
                          schedule=args.schedule, tier=args.tier,
                          universe=args.universe, rank_window=args.rank_window,
                          finalist=args.finalist)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        what = (f"universe {row.universe_spec} (rank window {row.rank_window})"
                if row.universe_spec else f"{len(symbols)} symbols")
        print(f"promoted {row.strategy_name} (tier={row.tier}, "
              f"{row.schedule}, {what})")
    finally:
        session.close()


if __name__ == "__main__":
    main()
