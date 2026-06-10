"""Trading-cost models for the portfolio engine.

The old per-ticker lab (engine/jobs/walkforward/validate/daily_backtest)
was retired with the arena; the realistic cost model in ``costs.py`` is the
sole survivor — it is shared by edgefinder/engine/backtest.py and
edgefinder/engine/validate.py.
"""
