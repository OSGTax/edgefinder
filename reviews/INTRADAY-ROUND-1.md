# INTRADAY HUNT — ROUND 1, honest verdicts (2026-06-14)

**Protocol:** pre-registered roster `engine/intraday_roster.py` (v5.54,
committed BEFORE any run); minute-bar walk-forward engine
(`engine/intraday_backtest.py`), liquid lane (SPY,QQQ,IWM + 10 mega-caps),
2024-01-01 → sealed-at-2026-04-01 holdout wall, **costed** (FIXED spread +
sqrt-impact + caps), flatten-at-close, **decision every 5 minutes**,
23 walk-forward folds vs SPY, total-return bar primary. Holdout NOT burned.

## Verdict: **0 of 10 candidates pass. No candidate beats SPY on a single fold-majority; none clears even the do-nothing floor.**

### The floor (controls — what "no edge" costs)
| control | excess/fold vs SPY | folds>SPY | trades | reads as |
|---|---|---|---|---|
| `intraday_flat` (null) | **−1.29pp** | 7/23 | 0 | do-nothing floor: all-cash gives up SPY's drift |
| `buy_hold_open:SPY` (anchor) | **−2.45pp** | 3/23 | 1,796 | **the honest intraday floor** — buying the open and flattening at close STRUCTURALLY forfeits overnight drift + pays 1 toll/day |
| `ir_random_101` | −6.32pp | 3/23 | 100,541 | active-churn floor: a coin-flip basket re-drawn every 5 min bleeds ~6pp to tolls |
| `ir_random_103` | −5.77pp | 3/23 | 100,567 | same |

The anchor is the key calibration: a flatten-at-close intraday strategy
begins **~2.45pp/fold in the hole vs full SPY** purely from forgoing the
overnight component — so "beats SPY" is a stiff bar, and the fair internal
yardstick is excess-vs-the-anchor. Even on that generous basis, nothing
wins.

### Candidates (all FAIL — sorted best→worst)
| candidate | excess/fold | folds>SPY | trades | vs anchor |
|---|---|---|---|---|
| ir_gap_fade | −2.46pp | 5/23 | 5,959 | ≈ flat (≈ anchor; barely trades) |
| ir_high_break | −3.05pp | 5/23 | 2,025 | −0.6 |
| ir_gap_go | −4.47pp | 2/23 | 9,240 | −2.0 |
| ir_late_mom | −7.31pp | 0/23 | 20,379 | −4.9 |
| ir_orb | −15.85pp | 0/23 | 57,919 | −13.4 |
| ir_vwap_rev | −24.29pp | 0/23 | 71,926 | −21.8 |
| ir_reversal | −60.52pp | 0/23 | 164,159 | −58.1 |
| ir_momentum | −60.59pp | 0/23 | 169,235 | −58.1 |

(Reversal and momentum — deliberate opposites — BOTH lost ~−60pp. When a
thing and its mirror both lose ~the same large amount, the signal isn't
the driver; the turnover is.)

## The finding, in one sentence
**Turnover is the whole story.** Plotting losses against trade count is
nearly monotonic: the candidates that trade most (momentum 169k, reversal
164k, vwap 72k, orb 58k) lose most (−24 to −61pp); the ones that barely
trade (high_break 2k, gap_fade 6k) sit at the do-nothing floor. The
five-minute full-basket reselection cadence pays the bid-ask + impact toll
hundreds of times per day, and no fast signal on these liquid names
out-earns that toll. Candidates that churn MORE than the random control
(momentum/reversal/vwap) do WORSE than random — their signal adds turnover
without adding edge. This is "intraday is where retail goes to die" made
concrete on our own data and toll model.

## Honesty instruments — all clean
- Null (flat) at −1.29, anchor at −2.45, randoms at ~−6: a clean,
  ordered floor. No candidate beat the active-churn floor except by
  trading less (gap_fade/high_break ≈ do-nothing).
- The engine's correctness is not in question (Phase-2 tests + the
  buy_hold_open anchor behaving exactly as the overnight-drift math
  predicts). The tolls are real and the verdict is real.

## No re-checks queued
Re-checks exist to stress-test PASSERS. Nothing passed — nothing reached
even the SPY bar, let alone with margin — so there is nothing to re-check.
Holdout stays sealed (2026-04-01), unburned.

## Round-2 directions (the data dictates them)
The lesson is unambiguous: **a viable intraday strategy must trade rarely.**
Round 2 should test the turnover hypothesis directly, NOT retune these:
1. **Hold-to-close, enter-once families**: signal at the open (gap, ORB),
   take ONE position, hold to the close — no mid-session reselection. The
   gap/ORB *theses* were never really tested here because 5-min reselection
   churned them; an enter-once version pays ~2 tolls/day, not hundreds.
2. **Coarse cadence**: decision-interval 30–60 min, or hold-until-stop
   logic, to cut toll events by 6–12×.
3. **Overnight-hold variants** (a thesis change): forgoing overnight drift
   costs ~2.45pp/fold; strategies that HOLD winners overnight reclaim it —
   but that's no longer "pure intraday," an owner call.
4. Keep the controls; add a low-turnover random to recalibrate the floor
   at the new cadence.
