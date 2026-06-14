"""Persist engine-v2 validation scorecards to validation_runs.

Relocated from edgefinder/backtest/validate.py so the new engine's harness
never imports the old per-ticker strategy registry (validate.py populates it
at module scope). Same table, same shapes — the dashboard's
GET /api/strategies/validation and its ``validated`` semantics
(criteria.all_met AND holdout is not None AND holdout.passes) work unchanged.
"""

from __future__ import annotations

from edgefinder.db.models import ValidationRun


def _jsonable(obj):
    """Recursively coerce numpy scalars (np.float64 / np.bool_ / np.int64) to
    native Python types so the JSON column can serialize them.

    The intraday engine produces numpy scalars in its scorecard (np.float64
    from array math, and the criteria booleans become np.bool_ via
    ``np.float64 > 0``); the stdlib JSON serializer raises "Object of type
    bool is not JSON serializable" on np.bool_ (intraday-r1 wave 1, 2026-06-14
    — 9/12 jobs computed fine but failed at the write). Coercing here fixes the
    whole class for every engine, present and future. NaN/Inf floats are
    nulled (Postgres JSON rejects them; a missing metric reads as None)."""
    import math

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # numpy scalars expose .item() -> native python; guard without importing numpy
    item = getattr(obj, "item", None)
    if item is not None and obj.__class__.__module__ == "numpy":
        obj = item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


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
        config=_jsonable(scorecard.get("config")),
        oos=_jsonable(oos),
        criteria=_jsonable(scorecard.get("criteria")),
        holdout=_jsonable(scorecard.get("holdout")),
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
