# HOLDOUT BURN PROTOCOL — pre-registered BEFORE the burn (2026-06-11)

This document is committed BEFORE the sealed holdout is evaluated. It
fixes the mechanics, the pass criteria, and the read restraint in
advance, so nothing about the interpretation can be adjusted after the
results are visible. Owner-approved restraint design (2026-06-11);
the burn itself executes only on explicit owner go.

## What is being burned

- **Window:** 2025-12-05 → the burn date (~6 months). No validation run
  in any of the four hunt rounds has ever evaluated a bar in this
  window (`--holdout-start 2025-12-05` sealed it on every run).
- **Cohort:** the 12 confirmed finalists, exactly as pre-registered —
  xsec_mom_12_1, value_mom_barbell, mom_inverse_vol, mom_earnings_tilt,
  barbell_trend_value, fast_growers, growth_mom_barbell, tri_sleeve,
  growth_value_barbell, mom_3_12, peg_growers, fast_growers_rev.
  No parameter may differ from the registered spec.
- **Mechanics:** one run per finalist on the standard lane
  (top:500 PIT, `--costed --div-adjust --bars-from r2`,
  `--holdout-start 2025-12-05`) plus the explicit **`--burn-holdout`**
  flag, `--record`, label `holdout-burn:*`. A null control
  (buy_and_hold:SPY) runs in the same batch.

## Pass criteria — FIXED IN ADVANCE (cohort-level)

The unit of judgment is the COHORT, not the strategy. Six months is one
regime; per-strategy holdout results are too short to kill or crown
individual strategies and will not be used to.

Primary metric per strategy: excess TOTAL RETURN vs SPY over the
holdout window (one number per strategy).

- **COHORT PASS — ≥ 7 of 12 positive excess:** the hunt's selection
  process is validated out-of-sample. Proceed to promotion with the
  full cohort.
- **COHORT MIXED — 5–6 of 12 positive:** consistent with a real but
  regime-dependent edge set. Proceed to promotion; the live-vs-lab
  scorecard becomes the binding evidence; revisit at 6 live months.
- **COHORT FAIL — ≤ 4 of 12 positive:** treat the hunt's selection as
  unproven. STOP — no promotions until owner review. (For calibration:
  twelve random books would be expected to land ~3–5 positive in a
  typical window; ≤4 means the finalists are indistinguishable from
  luck on fresh data.)

Null-control sanity: the SPY null must read ≈ 0.0 excess, else the burn
batch itself is invalid and is re-run before any reading.

Per-strategy results are recorded (the `--record` audit trail requires
it) and may be shown as flags next to promoted strategies, but DO NOT
individually gate promotion and DO NOT trigger demotions.

## Read restraint — the leakage clause (owner-approved)

The burn is read at cohort level ONLY. Specifically, the burn report
will state: the cohort tally (N of 12 positive), the cohort median
excess, and the single per-strategy excess number. The report and all
future analysis will NOT dissect:

- which MONTHS inside the holdout drove results,
- which SECTORS or individual names drove results,
- which STYLES won or lost beyond the single number per strategy,
- any fold-level, regime-tagged, or attribution breakdown of the window.

Rationale: holdout information leaks into future research through the
researcher, not the database. The less structure we extract from the
window, the less it can steer future roster design. (Acknowledged
residual: even one number per strategy reveals coarse style
information; this is the accepted floor, since promotion bookkeeping
needs per-strategy rows.)

Future rosters must NEVER cite burned-window performance as
out-of-sample evidence, and roster pre-registrations written after the
burn must not reference holdout results as motivation.

## Re-seal clause — effective at the moment of the burn

1. A NEW holdout wall is set at the burn date. Every validation run
   after the burn uses `--holdout-start <burn-date>`.
2. The burned window (2025-12-05 → burn date) becomes ordinary
   in-sample data for all future work.
3. The new holdout accumulates automatically: post-burn data did not
   exist when the burn was read, so no leakage into it is possible —
   for new strategies it is clean by construction.
4. Defense in depth for any leaked design: future candidates must still
   pass the 2021–2025 walk-forward folds (untouched by the leak), all
   three adversarial re-checks, the NEW sealed holdout, and live
   paper trading. Leakage can inflate apparent performance only on the
   burned slice, which is a minority of in-sample and none of the new
   out-of-sample.

## Execution checklist (when the owner says go)

1. Commit this protocol (done — this file) and the burn wave manifest.
2. Run the 12 + null batch via hunt-batch (`--burn-holdout`).
3. Read results at cohort level per the criteria above; write
   `reviews/HOLDOUT-BURN.md` containing ONLY the permitted numbers.
4. Same commit: set the new wall — document `--holdout-start <burn-date>`
   as the standard lane flag in CLAUDE.md and HANDOFF.
5. Proceed per the cohort verdict.
