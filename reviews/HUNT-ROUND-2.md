# HUNT ROUND 2 — 12 pre-registered candidates, honest verdicts (2026-06-10)

**Protocol:** identical to round 1 — parameters frozen in `engine/hunt_r2.py`
(commit 322afcb) BEFORE any run; PIT top-500 universe (trailing 126-day
dollar-volume), realistic costs (`--costed`), total-return prices
(`--div-adjust`), bars read from the R2 store (`--bars-from r2`, post-slim;
equivalence vs DB proven bit-exact), sealed holdout **2025-12-05 — still
zero looks burned**. Wave 1 = ids 78–92; adversarial re-checks = ids
93–110 (fold shifts `--is-days 357/399` ≈ ±21d + late subperiod
`--start 2022-06-01`; ALL THREE must pass — the standard that confirmed
xsec_mom_12_1 and killed deep_value_pe10 in round 1).

**Instrument checks (in-batch):**
- Null control (buy_and_hold:SPY vs SPY): **−0.02pp, 0/38 folds** — calibrated.
- Fresh random-20 seeds (s23/s31): **−8.28 / −8.89pp, 1/6 each** — the
  noise floor on this menu still reads ≈ −7 to −9pp. Zero false positives.

## Confirmed finalists (all-three-re-checks standard)

### ✅ value_mom_barbell — CONFIRMED (total-return bar)
Half deep-value (10 cheapest profitable, P/E<10) + half 12-1 momentum
(10 strongest, trend-gated), each name 5%, monthly, top-500 PIT.
| run | excess/fold | folds>SPY | fold Sharpe |
|---|---|---|---|
| main | **+16.92pp** | **5/6** | 1.56 |
| re-check shift −21d | **+22.72pp** | 5/6 | 1.74 |
| re-check shift +21d | **+17.26pp** | 4/5 | 1.70 |
| re-check late (2022-06→) | **+28.69pp** | 4/4 | 2.02 |
The strongest candidate in lab history — best main-run excess, best
re-check margins, and the only 5/6 main. The value and momentum sleeves
appear to hedge each other's bad folds.

### ✅ mom_inverse_vol — CONFIRMED (total-return bar)
12-1 momentum top-20, trend-gated, weighted 1/vol(60d), monthly.
| run | excess/fold | folds>SPY | fold Sharpe |
|---|---|---|---|
| main | +8.11pp | 4/6 | 1.05 |
| re-check shift −21d | +15.83pp | 4/6 | 1.27 |
| re-check shift +21d | +5.71pp | 4/5 | 1.14 |
| re-check late | +17.94pp | 4/4 | 1.52 |

### ✅ mom_earnings_tilt — CONFIRMED (total-return bar, thin margins)
12-1 momentum top-20 restricted to PIT-profitable names (EPS > 0).
| run | excess/fold | folds>SPY | fold Sharpe |
|---|---|---|---|
| main | +5.87pp | 4/6 | 1.10 |
| re-check shift −21d | +4.89pp | 4/6 | 1.08 |
| re-check shift +21d | +4.45pp | 3/5 | 1.07 |
| re-check late | +13.77pp | 4/4 | 1.60 |
Every check clears the bar, but the shift margins are the thinnest of any
confirmed finalist — treat as the weakest of the four.

**Shared caveat (disclosed, structural):** like xsec_mom_12_1, all three
fail the risk-adjusted bar (mean excess Sharpe −0.14/−0.65/−0.60;
drawdowns up to ~15–27pp deeper than SPY; barbell worst-dd ~30%,
inverse-vol ~36%). These are RETURN edges, not comfort edges.

**Correlation caveat:** three of the four confirmed finalists run on the
same 12-1 momentum return engine (inverse-vol = same book, different
weights; barbell = half that book). The barbell's value sleeve is the only
genuinely distinct confirmed return source so far. For the 10-finalist
goal these count individually, but they are NOT four independent edges.

## Killed by the re-checks (2/3 is not 3/3)

| candidate | main | failed check | failing number |
|---|---|---|---|
| value_pe12 | +8.59pp 4/6, Sharpe 1.95 | shift −21d | 3/6 folds (below majority) |
| mom_6m_k20 | +6.29pp 4/6 | shift +21d | **+0.45pp, 2/5** — edge vanished |
| mom_12_1_k40 | +3.78pp 4/6 | shift −21d | 3/6 folds |

value_pe12 repeats its sibling deep_value_pe10's round-1 pattern exactly:
beautiful main run (came within ONE criterion — drawdown — of the
risk-adjusted bar, like pe10), then a fold-shift knocks it under majority.
The cheapness edge is real but fold-boundary-fragile. mom_6m_k20 is the
textbook case for why the re-checks exist: a 21-day shift collapsed
+6.3pp to +0.45pp.

## Full wave-1 scoreboard (exret = mean excess return/fold)

| candidate | exret | folds | Sharpe | note |
|---|---|---|---|---|
| value_mom_barbell | **+16.92** | 5/6 | 1.56 | → CONFIRMED |
| value_pe12 | +8.59 | 4/6 | 1.95 | re-check kill |
| earnings_yield_top | +8.59 | 4/6 | 1.95 | bit-identical to value_pe12 (see below) |
| mom_inverse_vol | +8.11 | 4/6 | 1.05 | → CONFIRMED |
| mom_6m_k20 | +6.29 | 4/6 | 1.01 | re-check kill |
| value_roe | +6.20 | 3/6 | 1.74 | one fold short (again — the round-1 cousin pattern) |
| mom_earnings_tilt | +5.87 | 4/6 | 1.10 | → CONFIRMED |
| mom_12_1_k40 | +3.78 | 4/6 | 0.90 | re-check kill |
| mom_regime_gated | +2.62 | 3/6 | 0.41 | gate DESTROYS the edge |
| value_momentum | −0.00 | 2/6 | 1.19 | |
| value_low_debt | −1.02 | 2/6 | 1.22 | |
| low_vol_value | −4.88 | 1/6 | 1.31 | smallest drawdowns, as always |
| random_20_s23 | −8.28 | 1/6 | 0.35 | yardstick |
| random_20_s31 | −8.89 | 1/6 | 0.37 | yardstick |

**Registration lesson:** earnings_yield_top produced results
bit-identical to value_pe12 — ranking by earnings yield IS ranking by
inverse P/E among profitable names, and the P/E<12 ceiling never binds in
a top-20 selection from the top-500. A redundant registration; its
re-check slot was deliberately skipped (pe12's covers both).

## Round-2 learnings (families, not params)

1. **The value×momentum INTERACTION is the find of the round.** The
   barbell (+16.9, re-checks +17→+29) beats both of its sleeves run alone
   (pe12 +8.6 fold-fragile; pure 12-1 top-20 +9.17 in round 1) and passed
   re-checks with the widest margins ever recorded here.
2. **The 12-1 formation window is the robust momentum spec.** 6-month
   formation died on a fold shift; doubling the book to k40 diluted the
   edge to +3.8 then died; inverse-vol weighting KEPT the edge and passed.
   Quality tilt passes but thins it.
3. **Regime gates are expensive insurance:** SPY-200EMA gating cost
   −5.5pp vs ungated momentum (whipsaw re-entries) in this window —
   consistent with round 1's defensive-ETF findings.
4. **Pure value keeps missing by one fold.** value_roe 3/6 (+6.2) joins
   round 1's earnings_growers/garp_momentum/garp_quality as "one fold
   short"; both P/E screens that DID clear wave 1 then failed a fold
   shift. Cheapness alone is real but unstable; it needs the momentum
   blend to survive perturbation.
5. **Machine integrity at volume:** 33 runs (15 + 18), zero lost jobs,
   zero retries needed, all controls clean. First full round run entirely
   from the R2 store at top-500 breadth.

## Scoreboard after round 2

**Goal: 10 finalists. Confirmed: 4** — xsec_mom_12_1 (r1),
value_mom_barbell, mom_inverse_vol, mom_earnings_tilt (r2). Holdout still
sealed. Promotion of any cross-sectional finalist remains blocked on the
open owner decision: live universe mechanics (resolve-at-promotion vs
nightly re-resolve).

**Round-3 directions implied by the data (to pre-register, not tune):**
the value×momentum interaction space (different sleeve ratios are NEW
strategies, not tunes — register sparingly), momentum overlays that
target the drawdown caveat (vol-targeting the BOOK rather than the
weights), and non-momentum/non-value families (quality standalone,
profitability growth, seasonality at monthly cadence) to diversify the
finalist pool away from one return engine.
