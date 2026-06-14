# DAILY-DECISION HUNT — ROUND 2 (stateful hold/exit), verdicts (2026-06-14)

**Protocol:** pre-registered roster `engine/hunt_daily_r2.py` (v5.57) using the NEW stateful
interface (`RebalanceContext.holdings`) + entry/exit hysteresis. Same lane as daily-r1
(top:500, `--schedule daily --rebalance-band 0.01`, costed, div-adjust, R2, 2021-06-01 →
sealed-2026-04-01, total-return vs SPY). Holdout NOT burned.

## Verdict: **0 confirmed finalists.** One candidate (sh_trend_hold) passed the main criteria but FAILED the all-three adversarial re-checks. The stateful interface itself is a clear success.

### Win #1 — the hysteresis fixed the churn (the round's real point)
Trade counts collapsed vs the stateless daily-r1 twins, exactly as designed:
| strategy | daily-r1 (stateless) | daily-r2 (hysteresis) | reduction |
|---|---|---|---|
| trend hold | 12,687 | 739 | 17× |
| high break | 13,372 | 653 | 20× |
| momentum-20 | 6,856 | 2,057 | 3.3× |
The no-trade band + holdings-aware entry/exit bands turn "decide daily" into "hold until the
signal breaks." The engine change is validated machinery (kept regardless of this round's hunt
verdict); the 12 monthly finalists are byte-identical (golden test).

### Wave-1 scoreboard
| strategy | excess/fold | folds>SPY | Sharpe | trades | crit |
|---|---|---|---|---|---|
| **sh_trend_hold** | **+1.96** | **4/6** | 0.81 | 739 | **PASS** |
| buy_and_hold_spy (null) | −0.06 | 0/6 | 1.70 | 6 | — |
| sh_random_73 (control) | −2.17 | 4/6 | 0.90 | 346 | fail |
| sh_random_71 (control) | −3.39 | 2/6 | 1.12 | 213 | fail |
| sh_high_break | −7.33 | 0/6 | 0.82 | 653 | fail |
| sh_mom_20 | −20.80 | 1/6 | −0.23 | 2,057 | fail |
| sh_reversal_5 | −32.82 | 1/6 | −1.07 | 4,221 | fail |

sh_trend_hold was the FIRST candidate in any fast/daily lane to clear SPY + criteria
(+1.96pp, 4/6). Short-momentum and reversal lose even when held (the system's momentum edge is
the long/trend kind, not 20-day or reversal). Note the controls: with hysteresis a stable
random long basket nearly tracks SPY (−2 to −3pp) and even hits 4/6 folds once — so the bar to
beat is the random floor, and sh_trend_hold's true margin is ~+4pp over it.

### Win #2 inverted — the re-checks killed it (the honest verdict)
| run | excess/fold | folds>SPY |
|---|---|---|
| main | +1.96 | 4/6 ✓ |
| re-check shift −21d | +10.77 | **3/6** ✗ (not majority) |
| re-check shift +21d | **+0.23** | **3/6** ✗ (excess ≈ 0, not majority) |
| re-check late (2022-06→) | +6.15 | 3/4 ✓ |

Positive excess on all three, but **only 1 of 3 keeps a fold-majority**, and shift+ collapses
to +0.23pp. This is the precise non-confirmation pattern that killed value_pe12 and mom_6m_k20
in the monthly hunt: high mean excess driven by a FEW big folds, not consistent fold-wise
outperformance. By the pre-registered all-three-re-checks standard (the same bar that confirmed
the 12), sh_trend_hold **does not confirm**.

### Distinctness caveat (would matter even if it had confirmed)
sh_trend_hold is "hold names above their 200-EMA with positive 12-month momentum, exit on a
200-EMA break" — the SAME trend/momentum edge the monthly fleet already captures (xsec_mom_12_1,
mom_inverse_vol, mom_3_12…), just at daily cadence. It is not a new edge; at best it would have
been a more-responsive expression of an existing one. Given it doesn't clear the bar, there is
no case to promote it over the live monthly finalists.

## The four-round synthesis (intraday r1/r2 + daily r1/r2)
1. **Turnover tolls dominate faster cadences.** Every rotating/high-frequency family died;
   losses were monotonic in trade count across all four rounds.
2. **The stateful hold/exit interface is the right fix for turnover** — it cut churn 17–20×
   and let a candidate reach the main bar for the first time. Kept as permanent machinery.
3. **But no fast/daily strategy CONFIRMS** under the standard that produced the 12 monthly
   finalists. The one that came closest (sh_trend_hold) is the monthly fleet's own
   trend/momentum edge, fragile at daily cadence.
4. **The validated edge in this system lives at MONTHLY cadence** — the 12 live finalists.
   Daily decisioning is now proven to add no confirmed edge on this universe; the honest place
   to stand is the monthly fleet as the proven core.

## No finalist; holdout sealed (2026-04-01, unburned); queue idle.
Remaining open option (owner's earlier interest): the blind LLM judgment agent, whose real
test is LIVE experimental paper (built, v5.56) — it sidesteps the backtest-turnover problem
because live it can simply HOLD. Available whenever the owner wants it promoted.
