# HOLDOUT BURN — the one look (2026-06-11)

**Protocol:** `reviews/HOLDOUT-BURN-PROTOCOL.md`, committed before the
burn. This report contains ONLY the permitted reading: batch validity,
the cohort tally, the cohort median, and one excess-return number per
strategy. No month, sector, style, fold, or attribution breakdown has
been produced, and none may be produced from this window in the future.

- **Window:** 2025-12-05 → 2026-06-09 (the sealed holdout; first and
  only evaluation).
- **Batch validity:** SPY null control −0.02pp ✓.

## Verdict: **COHORT PASS — 12 of 12 finalists positive excess vs SPY** (criterion: ≥7)

| finalist | holdout excess total return vs SPY |
|---|---|
| mom_inverse_vol | +58.2pp |
| xsec_mom_12_1 | +55.8pp |
| mom_3_12 | +54.5pp |
| mom_earnings_tilt | +43.6pp |
| growth_mom_barbell | +35.0pp |
| tri_sleeve | +29.2pp |
| fast_growers_rev | +26.7pp |
| growth_value_barbell | +25.1pp |
| barbell_trend_value | +24.1pp |
| value_mom_barbell | +20.7pp |
| peg_growers | +14.8pp |
| fast_growers | +13.5pp |

**Cohort median: +25.9pp.** Cohort-level note: all twelve also posted
positive excess Sharpe in the window — the first time the cohort has
cleared the risk-adjusted bar anywhere.

**Honest qualifiers (cohort-level, pre-committed framing):**
- The window was a single regime (tagged bull_calm) and six months is
  one draw. PASS validates the selection process out-of-sample; it does
  not promise these magnitudes persist. The binding evidence from here
  is live paper trading.
- Per the protocol, these per-strategy numbers do not individually rank,
  gate, or demote anything.

## Re-seal (effective now)

- **New holdout wall: 2026-06-11.** All future validation runs use
  `--holdout-start 2026-06-11`. The new holdout accumulates clean data
  from today forward.
- The burned window (2025-12-05 → 2026-06-09) is now ordinary in-sample
  data. It may never again be cited as out-of-sample evidence, and
  future roster pre-registrations may not reference this report as
  motivation.

## Consequence per the pre-fixed criteria

COHORT PASS → proceed to promotion with the full cohort. Remaining
blocker: the live-universe mechanics decision (resolve-at-promotion vs
re-resolve-at-rebalance), then `engine.promote` per strategy.
