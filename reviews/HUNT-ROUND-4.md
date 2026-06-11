# HUNT ROUND 4 — 12 pre-registered candidates, honest verdicts (2026-06-11)

**Protocol:** unchanged — roster frozen in `engine/hunt_r4.py` (v5.49)
BEFORE any run; PIT top-500, `--costed --div-adjust --bars-from r2`;
sealed holdout **2025-12-05 — zero looks burned through all four
rounds**; re-checks = fold shifts ±21d + late subperiod, ALL THREE must
pass. Run by the continuous hunt loop.

**Instrument checks:** null −0.02pp (0/38); fresh randoms s47/s53
−6.42/−5.34pp (1/6 each). Clean.

## ✅ SIX NEW CONFIRMED FINALISTS — the goal round

| finalist | main | shift− | shift+ | late |
|---|---|---|---|---|
| **growth_mom_barbell** | +19.6 (5/6) | +26.7 (4/6) | +15.6 (4/5) | **+36.2 (4/4)** |
| **tri_sleeve** | +17.3 (5/6) | +19.6 (5/6) | +15.5 (4/5) | +29.0 (4/4) |
| **growth_value_barbell** | +13.1 (5/6) | +12.2 (5/6) | +12.6 (4/5) | +19.8 (4/4) |
| **mom_3_12** | +12.9 (5/6) | +22.0 (4/6) | +13.3 (4/5) | +27.3 (4/4) |
| **peg_growers** | +7.1 (4/6) | +6.0 (5/6) | +3.2 (4/5) | +14.0 (4/4) |
| **fast_growers_rev** | +4.5 (5/6) | +5.9 (5/6) | +2.4 (3/5) | +10.9 (4/4) |

Highlights: growth_mom_barbell posted the largest re-check numbers in
lab history; growth_value_barbell pairs +13pp excess with the best risk
profile of any finalist (main Sharpe 2.10, worst-dd 21.5%); mom_3_12
(skip-a-quarter formation) re-checked HARDER than its main run.
fast_growers_rev's shift+ (3/5, +2.4) is the thinnest pass among the
twelve — ranked weakest accordingly.

## Killed

| candidate | result | note |
|---|---|---|
| fast_growers_mom | re-check 2/3 (shift− 3/6) | growth×momentum integration fails like value×momentum rank-blend did — interactions need SLEEVES |
| mom_sharpe_rank | +4.9, 3/6 | vol belongs in the weights (r2), not the rank |
| roe_value | +1.8, 3/6 | quality-per-price: market-like |
| value_cr_fortress | +0.4, 3/6 | the cheap-and-liquid screen adds nothing |
| steady_compounders | −6.1, 1/6 | Lynch stalwarts don't beat SPY at monthly cadence |
| min_vol_uptrend | −8.6, 1/6 | low-vol dies with or without a trend gate (dd 7.1% though — best capital preservation in the hunt) |

## Machine integrity

One job (mom_3_12 shift−) failed on raw TCP connection timeouts to the
Supabase pooler at job start, across all 3 workflow attempts — and the
v5.48.1 fix made it fail LOUDLY (run conclusion: failure) instead of
round 3's silent green. The single-run redo recorded first try. Loop
ops note: the session-bound monitor slept through one wave overnight
(~3h gap); a 25-min keepalive tick now backstops it, and HANDOFF carries
the resume protocol either way.

## 🎯 GOAL REACHED: 12 confirmed finalists (target was 10)

| # | finalist | round | main excess | engine family |
|---|---|---|---|---|
| 1 | xsec_mom_12_1 | r1 | +9.2 (4/6) | momentum |
| 2 | value_mom_barbell | r2 | +16.9 (5/6) | value+momentum sleeves |
| 3 | mom_inverse_vol | r2 | +8.1 (4/6) | momentum (1/vol weights) |
| 4 | mom_earnings_tilt | r2 | +5.9 (4/6) | momentum (profitability tilt) |
| 5 | barbell_trend_value | r3 | +18.0 (5/6) | value+momentum sleeves |
| 6 | fast_growers | r3 | +7.5 (5/6) | growth |
| 7 | growth_mom_barbell | r4 | +19.6 (5/6) | growth+momentum sleeves |
| 8 | tri_sleeve | r4 | +17.3 (5/6) | value+momentum+growth |
| 9 | growth_value_barbell | r4 | +13.1 (5/6) | growth+value sleeves |
| 10 | mom_3_12 | r4 | +12.9 (5/6) | momentum (3-12 formation) |
| 11 | peg_growers | r4 | +7.1 (4/6) | growth (PEG-constrained) |
| 12 | fast_growers_rev | r4 | +4.5 (5/6) | growth (revenue rank) |

**Honest rollup, four rounds, 73 pre-registered candidates, 12
confirmed (16% hit rate):**
- The twelve cluster into THREE return engines — 12-1-family momentum,
  trend-gated deep value, fast-grower fundamentals — plus their sleeve
  combinations. Twelve validated strategies, not twelve independent
  edges. Live capital across all twelve would concentrate, not
  diversify.
- Every control stayed clean throughout: nulls ≈ 0, 8 random books
  −5.3 to −8.9pp, dumb sweep 0/6 — the instrument never drifted.
- All finalists clear the TOTAL-RETURN bar; none clear the risk-adjusted
  bar (excess Sharpe −0.11 to −0.99; drawdowns up to ~38% worst-fold).
  These are return edges that ride through pain. growth_value_barbell
  is the closest to comfortable.
- The sleeve principle is the hunt's central discovery: every
  integrated/rank-blended interaction DIED (value_mom_rank_blend,
  fast_growers_mom, mom_sharpe_rank); every separate-sleeve pairing of
  confirmed engines SURVIVED (all five barbells/tri-sleeve).

**What the finalists still face (owner decisions):**
1. **The sealed holdout (2025-12-05→) has never been looked at.** The
   final exam: burn it once, on the twelve, with owner sign-off — the
   pre-registered design from day one.
2. **Live universe mechanics** for cross-sectional strategies
   (resolve-at-promotion vs nightly re-resolve) — blocks all twelve
   from paper trading.
