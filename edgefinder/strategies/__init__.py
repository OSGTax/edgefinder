# Register strategies (import side-effect populates StrategyRegistry).
# Live trading is additionally gated by settings.live_strategies — research
# candidates below the line are lab-only until they pass validation.
from edgefinder.strategies import coward, gambler, degenerate_v2  # noqa: F401

# Research candidates (2026-06-05 round 1) — lab-only, not in live_strategies.
from edgefinder.strategies import gap_drift, pullback_rider, turtle_adx  # noqa: F401
