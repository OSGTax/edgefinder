# HUNT ROUND 1 — 35 pre-registered candidates, honest verdicts (2026-06-10)

**Protocol:** every candidate's parameters frozen in `engine/hunt_r1.py`
(commit f5feb12) BEFORE its first run. Fixed-param walk-forward folds
(no optimizer exists in the engine), PIT top-N universes with trailing
126-day dollar-volume ranking, realistic costs (`--costed`) and total-return
prices (`--div-adjust`) on all stock lanes, sealed holdout pinned at
**2025-12-05 — zero looks burned**. Both bars scored per run: risk-adjusted
(recorded criteria) and total-return (computed from the same recorded fold
stats). All runs in `validation_runs` labeled `hunt-r1:*` (ids 29–68).

**Instrument checks (in-batch):**
- Null control (buy_and_hold:SPY vs SPY): **0.00 excess Sharpe / −0.02pp,
  0/38 folds** — calibrated.
- Menu control (equal-weight buy-hold of the PIT top-500 menu): **−3.3pp,
  0/6** — no survivorship/menu tilt left to harvest.
- Dumb sweep false-positive rate: **0/6 passed either bar**; random-20
  baselines read −7.0/−7.9pp. The noise floor on this menu is ~−7pp;
  passes below stand far above it.

## Finalists

### ✅ xsec_mom_12_1 — CONFIRMED (total-return bar)
12-1 cross-sectional momentum, trend-gated, top-20 EW, monthly, top-500 PIT.
| run | excess/fold | folds>SPY | fold Sharpe | fills |
|---|---|---|---|---|
| main | **+9.17pp** | 4/6 | 0.97 | 985 |
| re-check shift −21d | **+18.55pp** | 4/6 | 1.10 | 1047 |
| re-check shift +21d | **+7.15pp** | 4/5 | 1.08 | 822 |
| re-check late (2022-06→) | **+21.08pp** | 4/4 | 1.45 | 680 |
Caveat (disclosed, structural): drawdowns run 13–20pp DEEPER than SPY —
fails the risk-adjusted bar everywhere. This is a return edge, not a
comfort edge. First full criteria pass + adversarial survival in lab history.

### 🟡 deep_value_pe10 — provisional (re-check in flight)
Profitable names at P/E < 10, top-20 cheapest, monthly, top-500 PIT +
PIT fundamentals. Main run: **+8.71pp/fold, 4/6 folds, fold Sharpe 1.98**
(highest of the round), 853 fills; missed the risk-adjusted bar only on
drawdown (−1.9pp vs SPY). Verdict pending fold-shift ± and late-subperiod
re-checks (wave 4).

## Full scoreboard (FAIL unless noted; exsh = mean excess Sharpe, exret = mean excess return/fold)

**ETF defensive/regime (21yr, 38 folds incl. 2008+2020):** every candidate
cut drawdowns; none beat SPY's Sharpe in a majority of regimes.
| candidate | exsh | folds_sh | exret | dd cut |
|---|---|---|---|---|
| inverse_vol_etf7 | **+0.01** | 19/38 (missed by 1) | −0.67 | +3.4 |
| risk_parity_lite | −0.03 | 17/38 | −1.04 | **+5.1** |
| vol_target_spy_tlt | −0.16 | 12/38 | −1.04 | +3.0 |
| trend_breadth_gate | −0.30 | 17/38 | −0.87 | +1.3 |
| dual_momentum_9 | −0.39 | 12/38 | −1.15 | +2.9 |
| golden_cross_spy_tlt | −0.19 | 4/38 | −1.20 | +0.9 |
| golden_cross_spy | −0.21 | 5/38 | −1.43 | +2.8 |
| vol_target_spy | −0.22 | 7/38 | −1.40 | +3.7 |
| breadth_tilt_spy | −0.29 | 10/38 | −1.57 | +4.1 |
| mom_6m_top2 | −0.37 | 13/38 | −0.93 | +2.0 |
| canary_efa_iwm | −0.50 | 7/38 | −2.28 | +0.9 |

**Stock cross-sectional (top-500 PIT, costed, TR, 6 folds):**
| candidate | exsh | exret | folds_ret | note |
|---|---|---|---|---|
| xsec_mom_12_1 | −0.73 | **+9.17** | 4/6 | → FINALIST (TR bar) |
| inv_vol_top50 | −0.11 | −0.82 | 3/6 | near-miss both bars |
| vol_squeeze | −0.05 | −8.63 | 1/6 | dd cut +4.7 |
| qual_mom_blend | −0.53 | −5.08 | 1/6 | |
| dollar_vol_fade | −0.80 | −2.58 | 2/6 | no liquidity premium |
| low_vol_50 | −0.27 | −6.74 | 0/6 | textbook: dd cut +5.0, return lag |
| near_52wk_high | −1.13 | −7.19 | 1/6 | |
| near_52wk_low | −1.45 | −8.18 | 2/6 | contrarian worse than momentum |
| rsi_oversold_q | −2.13 | −15.14 | 1/6 | churn + costs |
| st_reversal_w | −2.29 | −20.59 | 0/6 | costs annihilate weekly reversal |

**Lynch/fundamental (first fundamental strategies ever tested here):**
| candidate | exsh | exret | folds_ret | note |
|---|---|---|---|---|
| deep_value_pe10 | **+0.29** | **+8.71** | 4/6 | → provisional finalist |
| earnings_growers | −0.30 | +4.23 | 3/6 | one fold short of majority |
| garp_momentum | −0.63 | +2.84 | 3/6 | one fold short |
| garp_quality | −0.34 | +1.65 | 3/6 | one fold short |
| garp_classic | −0.69 | −1.43 | 3/6 | |
| low_debt_value | −0.50 | −1.57 | 1/6 | |
| cash_rich_growth | −0.85 | −3.90 | 1/6 | |
| quality_smallcap (1k-3k band) | −0.90 | −2.62 | 1/6 | band stays dead |

**Dumb sweep (the yardstick):** letter_b −8.6pp 0/6 · early_month −12.6pp ·
random_s7 −7.0pp 0/6 · random_s13 −7.9pp 0/6 · alphabet −5.7pp 0/6 ·
tuesday_hold −19.9pp (139k fills of pure cost, as designed). **0/6 false
positives.**

## Round-1 learnings (families, not params — for round-2 pre-registration)

1. **Momentum's return premium is alive in the top-500** and survives PIT
   menus + realistic costs + TR — at the price of much deeper drawdowns.
   Variants worth registering: vol-managed momentum overlay, momentum with
   crash gate (regime filter on entry only), longer top-k.
2. **Value-with-profitability is the Lynch lane's live vein** (deep_value
   +8.7pp, three more cousins at +1.7..+4.2pp, each ONE fold short of
   majority). Pure GARP/PEG screens underperform their simple-value cousins.
3. **Defensive ETF rotation cuts drawdowns ~3–5pp but cannot beat SPY's
   Sharpe across 38 folds** — fourth independent confirmation (trend_timer,
   dual_momentum old + new, now 11 more). The risk-adjusted bar on liquid
   ETFs looks structurally near-unreachable; inverse-vol missed by ONE fold
   and is the family's best face.
4. **Costs kill churn, exactly as modeled:** weekly reversal −20.6pp, RSI
   dip-buying −15.1pp, daily dumb churn −19.9pp.
5. **Machine integrity at volume:** 40 runs, two transient pooler timeouts
   (one job lost pre-retry-fix, zero after), null/menu/dumb controls all
   clean. The factory works.

## Scoreboard after round 1

**Goal: 10 finalists. Current: 1 confirmed (xsec_mom_12_1) + 1 provisional
(deep_value_pe10).** Holdout sealed; promotion to experimental paper trading
pending the live-universe design decision (a top-500 cross-sectional
strategy needs a concrete live symbol list — resolve-at-promotion vs
nightly re-resolve; flagged for the owner alongside the infra work).
