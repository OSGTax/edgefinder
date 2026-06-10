"""Persist engine-v2 validation scorecards to validation_runs.

Relocated from edgefinder/backtest/validate.py so the new engine's harness
never imports the old per-ticker strategy registry (validate.py populates it
at module scope). Same table, same shapes — the dashboard's
GET /api/strategies/validation and its ``validated`` semantics
(criteria.all_met AND holdout is not None AND holdout.passes) work unchanged.
"""

from __future__ import annotations

from edgefinder.db.models import ValidationRun


def record_validation_run(
    session, scorecard: dict, *, universe: str, git_sha: str | None = None
) -> int:
    """Persist a walk-forward scorecard to validation_runs (offline evidence).

    Per-fold detail and the regime breakdown ride inside the ``oos`` JSON
    (added for the lab explorer, 2026-06-10) — legacy rows simply lack the
    keys and the dashboard renders "fold detail not recorded".
    """
    oos = dict(scorecard.get("oos") or {})
    if scorecard.get("folds") is not None:
        oos["folds"] = scorecard["folds"]
    if scorecard.get("by_regime") is not None:
        oos["by_regime"] = scorecard["by_regime"]
    row = ValidationRun(
        strategy_name=scorecard["strategy"],
        git_sha=git_sha,
        universe=universe,
        config=scorecard.get("config"),
        oos=oos,
        criteria=scorecard.get("criteria"),
        holdout=scorecard.get("holdout"),
        verdict=scorecard.get("verdict", "FAIL"),
    )
    session.add(row)
    session.commit()
    return row.id


def current_git_sha() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None
