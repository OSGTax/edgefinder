# DAILY-DECISION HUNT — ROUND 1, verdicts (2026-06-14)

**Protocol:** pre-registered roster `engine/hunt_daily_r1.py` (v5.56). Daily engine,
`--schedule daily --rebalance-band 0.01` (analyze daily, trade only when the target
changes), top:500 PIT, 2021-06-01 → sealed-2026-04-01, costed, total-return vs SPY,
6 walk-forward folds. Holdout NOT burned.

## Verdict: **0 of 9 candidates pass.** None beats SPY on any fold-majority (all 0/6 or 1/6). But the round exposed a real STRUCTURAL finding about the daily lane.

| strategy | excess/fold | folds>SPY | Sharpe | trades | note |
|---|---|---|---|---|---|
| **buy_and_hold_spy** (null) | **−0.06** | 0/6 | 1.70 | **6** | the band holds a FIXED book → ~0 excess, 1 trade/fold |
| dr_trend_hold (position) | −4.56 | 0/6 | 0.92 | 12,687 | least-bad active; sticky membership but still rotates |
| random_20_s61 (control) | −5.55 | 1/6 | 0.77 | 1,582 | rotates MONTHLY (hash by month) → ~1.6k trades |
| random_20_s67 (control) | −7.66 | 1/6 | 0.48 | 1,606 | "" |
| dr_mom_20 (swing) | −17.15 | 0/6 | −0.27 | 6,856 | |
| dr_high_break (swing) | −31.08 | 0/6 | −3.15 | 13,372 | |
| dr_reversal_5 (flip) | −38.27 | 0/6 | −1.74 | 11,831 | |
| dr_vol_breakout (swing) | −54.32 | 0/6 | −6.14 | 26,036 | |
| dr_gap_cont (flip) | −61.18 | 0/6 | −4.93 | 25,316 | |
| dr_gap_fade (flip) | −65.96 | 0/6 | −5.05 | 25,034 | |
| dr_reversal_1 (flip) | −74.01 | 0/6 | −6.02 | 25,046 | fully rotates daily → max churn → max bleed |

## The structural finding (why this matters more than the null)
The no-trade band works exactly as designed — proven by the null: **buy_and_hold:SPY traded
6 times total** (once per fold) because its book is FIXED. But every *candidate* is a
**cross-sectional daily-reselected top-K basket on 500 names**, and that membership churns by
construction: today's "top-20 worst 1-day returns" (or best momentum, or above-200-EMA set)
is a largely different 20 names than yesterday's. The band suppresses re-trues of names you
KEEP — but a daily reselection keeps almost nothing, so it rotates and pays the toll 20-in /
20-out, every day.

The evidence is unambiguous and monotonic in turnover (the same through-line as both intraday
rounds):
- Fixed book (SPY null): 6 trades, −0.06pp.
- Sticky-but-rotating (trend_hold): 12.7k trades, −4.56pp (Sharpe 0.92 — the least-bad).
- Fully-rotating daily (reversal_1, gap): ~25k trades, −66 to −74pp.
- Longer lookback = stickier basket = fewer trades = smaller loss (reversal_5 −38 vs
  reversal_1 −74; mom_20 −17 vs gap −66).

**The band cannot rescue a strategy whose MEMBERSHIP rotates; it only rescues HELD positions.**
On a large universe, daily cross-sectional reselection is inherently high-turnover regardless
of the band. The monthly fleet works precisely because monthly rebalancing amortizes that
rotation toll over ~21 days; daily rebalancing pays it every day.

## The real lesson for "analyze daily, trade only when warranted"
The owner's intent is sound — but expressing it as **stateless cross-sectional top-K
reselection defeats it**, because top-K membership is a moving set. To honor the intent, a
daily-decision strategy must be able to HOLD NAMED POSITIONS and exit only on a signal — i.e.
it needs STATE (what do I currently hold?), which the current stateless
`rebalance(ctx)->weights` interface does not give the strategy. The only low-turnover thing in
this round was the one fixed-membership book (the SPY null).

Two honest paths to a productive daily-decision lane (next-step options, owner call):
1. **Small / fixed universes** (e.g. the ETF menu, or a fixed watchlist) where membership is
   stable, so daily decisions re-true a stable book — the band then genuinely controls churn.
2. **A stateful hold/exit strategy model**: positions persist with entry+exit rules (buy on
   signal, hold until exit signal), so turnover is signal-driven, not reselection-driven. This
   is a new strategy interface (the strategy sees its current holdings), distinct from the
   stateless cross-sectional weights contract.

This also reshapes the BLIND AGENT (Part B): giving the agent its CURRENT HOLDINGS lets it
choose to HOLD (low turnover) rather than implicitly reselecting every day — both more
realistic and the only way an agent avoids the same rotation-toll death.

## No re-checks; holdout sealed
Nothing beat SPY; re-checks are for passers. Holdout (2026-04-01) unburned.

## Through-line across all three hunt rounds
intraday-r1 (5-min churn), intraday-r2 (enter-once helped but still lost to SPY), daily-r1
(cross-sectional daily reselection churns membership): **higher decision frequency on rotating
baskets is dominated by turnover tolls.** The validated edge in this system lives at MONTHLY
cadence (the 12 live finalists). A faster lane needs fixed/small universes or a stateful
hold/exit model — not more rotating-basket families.
