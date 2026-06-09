# Register strategies (import side-effect populates StrategyRegistry).
# Live trading is additionally gated by settings.live_strategies — research
# candidates below the line are lab-only until they pass validation.
from edgefinder.strategies import coward, gambler, degenerate_v2  # noqa: F401

# Research candidates (2026-06-05 round 1) — lab-only, not in live_strategies.
from edgefinder.strategies import gap_drift, gap_drift_v2, pullback_rider, trend_dip, turtle_adx  # noqa: F401

# Research candidates (2026-06-05 round 2).
from edgefinder.strategies import gap_carry  # noqa: F401

# Research candidates (2026-06-05 round 3).
from edgefinder.strategies import tom_seasonality, xsec_mom  # noqa: F401

# Microcap research candidates (2026-06-09) — small-cap/illiquid band + cost model.
from edgefinder.strategies import micro_reversal  # noqa: F401

# Risk-adjusted candidates (2026-06-09) — beat SPY on Sharpe/drawdown, not return.
from edgefinder.strategies import dual_momentum, trend_timer  # noqa: F401
