# Fundamentals validation report — EDGAR vs the frozen Polygon table

**Verdict: PASS** (2026-07-14, re-confirmed 2026-07-15 on the full
overlap). The lab may consume `fundamentals_pit`; `value_momentum:K`
activated in v9.4.0.

Owner's bar (locked 2026-07-14, before any results were seen): **≥90% of
comparables within ±10% per core metric**, cross-checked against the frozen
128k-row Polygon `fundamentals_snapshots` table (final vendor snapshot
2026-06-10). Minimum 20 pairs per metric.

## Final results (747 companies — every symbol present in both tables)

| Metric | Pairs | Within ±10% | Pass |
|---|---:|---:|---|
| Revenue (fiscal year) | 690 | **97.0%** | ✅ |
| Net income (fiscal year) | 735 | **90.3%** | ✅ |
| Book equity | 738 | **90.7%** | ✅ |
| Total assets | 734 | **98.6%** | ✅ |
| Current ratio | 646 | **96.4%** | ✅ |
| Shares outstanding | 727 | **92.7%** | ✅ |
| Diluted EPS (as filed) | 680 | **96.2%** | ✅ |
| Operating cash flow | 728 | **96.0%** | ✅ |

85 symbols skipped (frozen row carries no raw filing extracts — mostly
ETFs). 12 period-lag swaps (vendor snapshot held the one-year-earlier
10-K; alignment counted and reported, see methodology).

Reproduce: `python -m agent.edgar validate --max-symbols 1000`.

## Methodology — and why the first attempt was invalid

The first run (2026-07-14) compared **blended ratios at snapshot dates**
and failed every metric. The autopsy showed the comparison itself was
broken, in three ways:

1. **Stale-annual reference.** Each frozen snapshot embeds the company's
   latest **annual** filing at snapshot time — AAPL's 2026-06-10 row holds
   FY-2025 (September-end) figures. Our fresh TTM values were being scored
   against numbers up to a year older. No correct dataset passes that.
2. **Definition mismatches.** Polygon `debt_to_equity` = total
   *liabilities* / equity (AAPL 3.87) vs our interest-bearing *debt* /
   equity (0.78) — both arithmetically correct, different concepts.
   Polygon `free_cash_flow` equals operating cash flow (capex never
   subtracted). Polygon `capex` is broader than XBRL PP&E purchases
   (ABBV: $6.6B including intangible/milestone payments vs $1.2B actual
   PP&E) with a mixed sign convention (1,094 negative / 149 positive rows).
3. **Convention differences.** Our EPS is a stated TTM-NI/shares
   convention; the vendor reports the filed diluted EPS. Multi-class
   filers (ABNB) have several legitimate share counts.

**The fix: compare identical quantities from the same filing.** The frozen
rows embed Polygon's raw filing extracts (`raw_data.financials`). The gate
takes our newest 10-K PIT row filed on/before the snapshot (at a 10-K's
filed date our TTM equals that fiscal year exactly — Q4 derives as FY
minus three quarters) and compares our extraction against theirs for the
SAME filed number: revenue, net income, equity, assets, current ratio,
shares (scored against the closest of the vendor's share-count
definitions), filed diluted EPS, and operating cash flow. This tests
exactly what the gate exists to test — does free EDGAR give us the numbers
a paid vendor extracted from the same filings? — with timing and
definitions cancelled out. Where the vendor's concept is incomparable
(their capex/FCF), the metric is excluded and the reason documented here,
never silently redefined. If revenue disagrees >15%, the harness tries the
one-year-earlier 10-K and keeps it only on a <2% revenue match (vendor
snapshot lag — period alignment, not cherry-picking; every swap is counted
in the report).

## What the failed first run fixed in OUR pipeline

The autopsy was not one-sided — it caught three real defects on our side,
all fixed before the passing run (v9.3.1–v9.3.3):

- **YTD cash-flow differencing** (v9.3.1): 10-Q cash-flow statements carry
  only cumulative year-to-date durations; our quarterly filter discarded
  them, leaving FCF null for 97% of symbols. Quarters now derive by
  differencing consecutive cumulative periods sharing a fiscal-year start.
- **Revenue tag priority** (v9.3.2): the ASC-606 contract-revenue tag is
  only a subset of the top line for commodity/financial firms — ADM
  extracted at 31% of its real revenue. Total `Revenues` now outranks it.
- **Legacy tags** (v9.3.1): pre-2014 capex
  (`PaymentsToAcquireProductiveAssets`), pre-2015 D&A
  (`DepreciationAmortizationAndAccretionNet`).

Spot-checks against the public record after the fixes: AAPL TTM FCF
mid-2015 $64.5B ✓, mid-2020 $66.6B ✓; AAPL filed FY-2025 diluted EPS 7.46
= vendor's 7.46 exactly; ABBV FY-2025 operating cash flow $19.03B =
vendor's to the dollar.

## Residual disagreements (the honest tail)

The 3–10% outside tolerance per metric cluster into known causes, none
suggesting systematic extraction error:

- **Vendor errors**: Agilent (A) frozen book equity −$226M — a $6.7B
  company cannot have negative equity; ours matches the filed balance.
- **NCI/consolidation**: ACMR-class names where the vendor's net income
  includes noncontrolling interests (`ProfitLoss`) vs our
  parent-attributable `NetIncomeLoss` — a definition tail, not a wrong
  number.
- **Near-zero denominators**: loss-makers around breakeven (CLRO revenue
  $4.2M vs vendor −$0.3M) where relative difference explodes on tiny bases.
- **Period misalignment survivors**: odd fiscal calendars (DLTR, CORZ)
  where the vendor's "latest annual" and ours differ by one year and the
  revenue-match swap couldn't align them.

## Standing limits

XBRL fundamentals begin ~2009 (fundamentals-gated lab rules hold cash
before then; their effective in-sample half is 2009–2017); listed-today
survivorship (we fetch current-universe symbols only); no analyst
estimates; custom-tag filers degrade to nulls, never guesses. Source
decision record: `fundamentals-sources.md`.
