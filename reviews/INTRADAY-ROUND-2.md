# INTRADAY HUNT — ROUND 2 (low-turnover, both lanes), verdicts (2026-06-14)

**Protocol:** pre-registered roster `engine/intraday_roster_r2.py` (v5.55,
committed before any run). ENTER-ONCE, HOLD-TO-CLOSE families (stable
session-fixed baskets) with the engine's new `--rebalance-band 0.01` so a
held basket costs ~2 tolls/day, not hundreds. Each family run in BOTH a
**flat** lane (`--flatten-at-close`, pure intraday) and an **overnight**
lane (`--hold-overnight`). Liquid lane, 2024-01-01 → sealed-2026-04-01,
costed, walk-forward vs SPY, total-return bar. Holdout NOT burned.

## Verdict: **0 of 8 candidate-lanes pass.** No candidate beats SPY on fold-majority; none beats its lane's PASSIVE SPY anchor. But the round is highly informative — the turnover fix worked, and one signal beats random.

### The floors (controls)
| control | excess/fold | folds>SPY | trades | reads as |
|---|---|---|---|---|
| flat-null | −1.29 | 7/23 | 0 | do-nothing |
| **anchor-flat** (SPY bh, flatten) | **−2.45** | 3/23 | 1,694 | pure-intraday floor (forfeits overnight drift) |
| **anchor-on** (SPY bh, overnight) | **−0.23** | 6/23 | 793 | overnight reclaims the drift → ≈ flat SPY, as predicted |
| random-flat (201/203) | −6.35 / −5.98 | — | ~10k | enter-once random floor, flat lane |
| random-on (201/203) | −3.49 / −2.94 | — | ~9k | enter-once random floor, overnight lane |

Two structural facts the controls nail: (1) the **overnight lane beats the
flat lane by ~+2.5–3pp across the board** — the reclaimed overnight drift,
not an edge (anchor −2.45→−0.23; randoms ~−6→~−3). (2) Even passive SPY,
traded intraday-flat, sits −2.45pp behind buy-hold SPY — the pure-intraday
thesis is structurally handicapped.

### Candidates
| candidate | flat excess | overnight excess | folds>SPY (on) | trades |
|---|---|---|---|---|
| **iro_gap_go** | −4.14 | **−0.70** | **10/23** | ~4.3k |
| iro_gap_fade | −2.66 | −2.56 | 7/23 | ~3.4k |
| iro_morning_trend | −6.58 | −3.79 | 5/23 | ~10.6k |
| iro_orb | −8.24 | −5.41 | 3/23 | ~11.6k |

## What round 2 actually established
1. **The turnover hypothesis was right.** Holding a fixed basket (~2
   tolls/day) instead of 5-min reselection transformed the numbers: gap-go
   went from round-1-style annihilation to −0.70pp. The toll model and the
   round-1 diagnosis are vindicated.
2. **The overnight effect is real and measured (~+2.5–3pp/fold)** — but it
   belongs to SPY's drift, available to a coin flip; it is not alpha.
3. **gap-go (gap-up continuation) is the one signal with genuine value
   OVER RANDOM:** overnight −0.70 vs the overnight random floor ~−3.2 ≈
   **+2.5pp of signal contribution**, and 10/23 folds beat SPY (the most of
   any candidate). gap-fade adds a little over random too. orb and
   morning-trend are at/below random — no signal value (consistent with
   round 1).
4. **But nothing beats PASSIVE SPY.** gap-go-on (−0.70) is still below the
   overnight anchor (−0.23): the signal beats random, yet doesn't out-earn
   simply holding SPY. No candidate clears the bar; **no re-checks** (those
   are for passers). Holdout stays sealed.

## The honest two-round conclusion
On the liquid mega-cap pilot, **no intraday strategy beats SPY net of
realistic tolls.** Low turnover removes the catastrophic bleed but not the
gap to SPY. The only signal showing value-over-random (gap-up
continuation) earns it by HOLDING OVERNIGHT — i.e. it's a low-turnover
*swing* effect, not intraday alpha, and a swing strategy is the daily
fleet's territory, not this initiative's.

## Round-3 fork (a genuine strategic choice — owner call)
1. **Accept the null & park intraday.** Two rounds give a decisive,
   honest answer for liquid names; the engine + minute asset remain for
   future use. Return attention to the 12 live monthly finalists.
2. **Different universe** (higher-vol / less-efficient names): intraday
   edges may live where mega-caps don't — but that's exactly where tolls
   bite hardest (the liquid-only premise was deliberate). A real tension.
3. **Reframe gap-continuation as an overnight SWING strategy** and validate
   it on the DAILY engine/fleet (where overnight holding is native) rather
   than forcing it through the intraday lane. The one live signal points
   here.
