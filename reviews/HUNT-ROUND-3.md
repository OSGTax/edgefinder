# HUNT ROUND 3 — 12 pre-registered candidates, honest verdicts (2026-06-10)

**Protocol:** identical to rounds 1–2 — parameters frozen in
`engine/hunt_r3.py` (v5.48) BEFORE any run; PIT top-500, `--costed
--div-adjust --bars-from r2`; sealed holdout **2025-12-05 — still zero
looks burned**; re-checks = fold shifts ±21d + late subperiod, ALL THREE
must pass. First round driven by the continuous hunt loop (owner
directive: loop until 10 confirmed finalists).

**Instrument checks:** null −0.02pp (0/38); fresh randoms s41/s43
−8.35/−7.21pp (0/6, 1/6) — noise floor ≈ −7 to −9pp holds. Wave 1
double-recorded (trigger fired on branch AND main — fixed in-round;
verdicts use latest row per candidate).

## Confirmed finalists (all-three-re-checks standard)

### ✅ barbell_trend_value — CONFIRMED (total-return bar)
Round 2's barbell with the value sleeve also trend-gated: 10 cheapest
profitable P/E<10 IN UPTRENDS + 10 strongest 12-1 momentum, 5% each.
| run | excess/fold | folds>SPY | fold Sharpe |
|---|---|---|---|
| main | **+18.04pp** | **5/6** | 1.58 |
| re-check shift −21d | **+22.42pp** | 5/6 | 1.73 |
| re-check shift +21d | **+18.25pp** | 4/5 | 1.76 |
| re-check late (2022-06→) | **+30.14pp** | 4/4 | 2.04 |
Beats its already-confirmed sibling value_mom_barbell on every number
(main +18.0 vs +16.9; re-checks +22/+18/+30 vs +23/+17/+29 — equal or
better) and its mean excess Sharpe is only −0.11 vs SPY (worst-dd 27.8%)
— far closer to the risk-adjusted bar than the pure-momentum finalists.
NOTE: it shares both sleeves with value_mom_barbell — treat the pair as
ONE family occupying two finalist slots.

### ✅ fast_growers — CONFIRMED (total-return bar)
Lynch fast growers, systematized: EPS>0, earnings growth >20%, revenue
growth >10%, uptrend; top-20 by earnings growth, monthly.
| run | excess/fold | folds>SPY | fold Sharpe |
|---|---|---|---|
| main | +7.54pp | **5/6** | 1.59 |
| re-check shift −21d | +7.65pp | 5/6 | 1.67 |
| re-check shift +21d | +6.92pp | 4/5 | 1.67 |
| re-check late | +15.08pp | 4/4 | 2.17 |
The most STABLE re-check set recorded (margins barely move under
perturbation) and the first confirmed finalist with **no 12-1 momentum
engine and no P/E screen** — the diversification the pool needed. Mean
excess Sharpe −0.11; worst-dd 28.3%.

## Killed in wave 1 (no re-check earned)

| candidate | exret | folds | note |
|---|---|---|---|
| mom_soft_gate | +5.87 | 3/6 | half-book brake: keeps return, loses majority |
| mom_spy_vol_brake | +5.85 | 3/6 | vol-scaled book: same pattern |
| value_mom_rank_blend | +1.17 | 2/6 | the interaction does NOT survive integration — it lives in the two-sleeve structure |
| quality_roe_top | +0.16 | 2/6 | quality standalone is market-like |
| ew_top100 | −0.81 | 1/6 | no EW premium after costs in this window |
| seasonal_spy | −6.68 | 0/6 | Halloween effect: absent |
| quality_momentum | −6.98 | 1/6 | the HARD quality cut destroys momentum (r2's EPS>0 tilt was the right dose) |
| mom_52w_high | −7.39 | 1/6 | 52w-high signal: at the noise floor |
| random s41/s43 | −8.35/−7.21 | 0–1/6 | yardstick |

**INVALID (not failed):** fcf_yield_top and dividend_value returned
bit-identical all-cash artifacts (Sharpe null, dd 0) — the PIT snapshot
store doesn't carry those fields. Coverage audit (128,854 snapshots):
eg 80% · rg 74% · roe 89% · cr 98% · de 99% · eps 70% — usable; fcf_yield
1.4% · dividend_yield 0% · price_to_sales 0% · ev_to_ebitda 0.7% ·
price_to_book 0.9% — unusable. Future rosters must screen on covered
fields only.

## Machine integrity (two real bugs caught and fixed in-round)

1. **Double-fired wave:** hunt-batch triggered on both the session branch
   and the main merge → 15 duplicate rows. Fixed: trigger is session-
   branch-only.
2. **Silently lost records:** 3 of 6 re-check jobs hit Supabase
   Session-mode pool checkout timeouts (ECHECKOUTTIMEOUT) at the record
   step; `validate.py` swallowed the exception ("recording must never
   void a completed run") → compute succeeded, row lost, job GREEN.
   Fixed (v5.48.1): record retries 4× in-process with backoff then exits
   3 so the workflow retry engages; wave parallelism 6 → 4. The redo
   wave recorded 3/3 on the first attempt.

## Round-3 learnings (families, not params)

1. **Trend-gating the value sleeve is pure improvement** — the barbell's
   weak folds came from value traps, not the momentum sleeve.
2. **Growth fundamentals are a REAL, independent return engine** —
   fast_growers re-checks barely move under perturbation, unlike every
   momentum variant.
3. **The value×momentum interaction requires SEPARATE sleeves** — rank
   integration (+1.2) destroys what the barbell structure (+18.0) keeps.
4. **Soft risk overlays don't re-check** — both kept the raw edge but
   diluted fold wins below majority. The drawdown caveat stays priced in.
5. **Quality is a tilt, not an engine** — EPS>0 helps momentum (r2);
   ROE>15% standalone is flat; ROE>15% as a momentum filter is fatal.

## Scoreboard after round 3

**Goal: 10 finalists. Confirmed: 6** — xsec_mom_12_1 (r1),
value_mom_barbell, mom_inverse_vol, mom_earnings_tilt (r2),
barbell_trend_value, fast_growers (r3). Holdout sealed; promotion still
blocked on the live-universe owner decision. Family concentration:
4 of 6 lean on 12-1 momentum; barbells count as one family in spirit.
Round 4 directions: structural probes of the GROWTH vein
(rank-by-revenue, growth×momentum), sleeve pairings of the three
confirmed engines (growth+value, growth+momentum, tri-sleeve), momentum
formation cousins (3-12 intermediate, risk-adjusted rank), and
covered-field value cousins.
