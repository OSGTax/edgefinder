"""Shared router constants (kept tiny on purpose)."""

# Live paper fills and lab backtests price costs differently; any view that
# shows the two side-by-side carries this disclosure (values are the live
# tiers' actual constants: engine/live.SLIPPAGE_BPS and the old arena's
# settings.slippage_base_rate, vs engine/backtest's cost_bps default).
COST_DISCLOSURE = {
    "live_slippage_bps_per_side": 5.0,
    "lab_default_cost_bps": 2.0,
    "note": ("live paper fills pay ~5 bps/side slippage; v2 lab runs assume "
             "2 bps flat per fill unless run --costed (realistic spread + "
             "impact model)"),
}
