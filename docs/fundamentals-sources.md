# Fundamentals source decision — SEC EDGAR (+ FINRA short interest)

**Decision (owner, 2026-07-14):** SEC EDGAR is the desk's fundamentals
source. Free, public domain, point-in-time by construction. FINRA's
bi-weekly short-interest file may join later as a second public feed.

## Why the decision was needed

The desk's original fundamentals came from Polygon's fundamentals API,
which was **sunset on 2026-06-22** — the 128k-row `fundamentals_snapshots`
table is frozen at 2026-06-10 and can never update. Any strategy or
display built on fundamentals needed a new source *before* building on
top (owner's revision: "if we don't know the structure of that data then
its pointless to build on top").

## The five candidates evaluated

| Source | PIT? | Public display? | Cost | Verdict |
|---|---|---|---|---|
| **SEC EDGAR** (companyfacts API) | **Yes, by construction** — every fact carries its `filed` date; restatements are later-filed rows | **Yes — public domain** | Free | **CHOSEN** |
| Sharadar SF1 (Nasdaq Data Link) | Yes, native (`datekey`) + ~10k delisted names | No (subscriber-only redistribution terms) | $69/mo | Best paid option if survivorship-free depth is ever needed |
| Financial Modeling Prep | Partial (`fillingDate` present) | No — ToS prohibits public display of data/derived results on self-serve tiers | $19–139/mo | Rejected |
| Finnhub | No true PIT on fundamentals | No on free/starter tiers | Free–$$$ | Rejected |
| Intrinio / others | Varies | Enterprise licensing required for display | $$$+ | Rejected |

Every commercial self-serve tier prohibits displaying the data (or values
derived from it) on a public page — and the `/desk` page is public. EDGAR
data is US-government work: **display anything**.

## What EDGAR gives us

- `https://www.sec.gov/files/company_tickers.json` — symbol → CIK.
- `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json` — every
  XBRL fact the company ever filed, with `filed`, `start`/`end`, `form`.
  One call returns the company's entire history (2009+, when the XBRL
  mandate began).
- Fair-access policy: ≤10 req/s with a declared User-Agent containing a
  contact (ours: `EdgeFinder mike@oshorelinegroup.com`, owner-approved;
  `EDGEFINDER_EDGAR_USER_AGENT` overrides). We throttle to ~8/s.
- The frames API is NOT point-in-time — never use it for backtests.

## Known limits (stated, not hidden)

- **2009+ only.** No structured pre-XBRL fundamentals. Backtests using
  fundamentals have an in-sample floor of 2009.
- **Listed-today survivorship.** Companyfacts is fetched per symbol in our
  current universe; delisted names' filings exist on EDGAR but we don't
  crawl them. Sharadar SF1 is the upgrade path if that ever gates a
  strategy decision.
- **No analyst estimates.** Impossible from filings; not pretended.
- **Custom-tag filers degrade to nulls**, never to guessed numbers
  (visible in `python -m agent.edgar coverage`).

## The pipeline (v9.2.0–v9.3.x)

`agent/edgar.py` → `fundamentals_pit` (one row per FILING, keyed by the
`filed` date; price-free values + `_`-prefixed ratio ingredients).
`edgefinder/data/pit_fundamentals.py` serves the engine's
`asof(symbol, date)` protocol. Price ratios (P/E, P/B, EV/…) are computed
at decision/display time against a price the caller vouches for
(`agent.edgar.price_ratios`). Nightly top-up rides `agent.refresh`'s
market refresh. Validation gate and results: `fundamentals-validation.md`.

## FINRA short interest (deferred)

Bi-weekly public short-interest file; joins our shares-outstanding to give
short % of float. Deferred to the activation commit — URL/format to be
verified when built; parse will sit behind a setting.
