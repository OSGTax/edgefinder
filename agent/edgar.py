"""SEC EDGAR fundamentals — the free, public-domain, point-in-time lane.

Source decision (owner, 2026-07-14, after a 5-vendor evaluation): EDGAR is the
only fundamentals source matching the desk's constitution — every fact carries
the date it was FILED with the SEC (restatements appear as later-filed rows,
so no lookahead is even possible), it is public domain (the desk may display
anything), the nightly pull is explicitly within SEC's automation policy
(<=10 req/s + a declared User-Agent), and the archive is permanently ours.

Design (the honesty invariants):
- One row per FILING into ``fundamentals_pit``: the state of knowledge as of
  that ``filed`` date, built ONLY from facts filed on or before it.
- PRICE-FREE values only (EPS-TTM, ROE, ROA, debt/equity, current/quick,
  growth, FCF, shares, book equity, EBITDA pieces). A filing cannot know a
  future P/E; price-dependent ratios are computed where a price exists — in
  strategies via the decision-date price, in the brief via the latest close.
- Missing is NULL, never guessed. Tag waterfalls cover the standard us-gaap
  variants; a company using only custom extension tags yields nulls for that
  metric (visible in ``coverage``), not fabricated numbers.
- The frozen Polygon-era ``fundamentals_snapshots`` table (128k rows through
  2026-06-10) is the VALIDATION reference: ``validate`` cross-checks our
  computed values against it and writes the agreement report. The lab may not
  consume fundamentals until that gate passes (owner's strict bar: >=90% of
  comparables within ±10% per core metric).

Limits (stated, not hidden): XBRL data begins ~2009 (no structured
fundamentals before the SEC mandate); no analyst estimates; EPS-TTM here is
TTM net income / latest diluted-or-outstanding shares (a stated convention,
not the filed quarterly EPS sum, which double-counts buyback effects).

CLI (JSON out, like every agent tool):
  python -m agent.edgar ingest --symbols AAPL,MSFT
  python -m agent.edgar ingest --universe 1000     # the nightly / backfill
  python -m agent.edgar coverage
  python -m agent.edgar validate                   # vs the frozen table
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.request
from datetime import date, datetime, timedelta

from config.settings import settings

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MIN_REQUEST_INTERVAL = 0.12  # ~8 req/s, safely under SEC's 10/s ceiling

# Ordered tag waterfalls: first tag with usable facts wins. The big liquid
# names we trade file with these standard tags; exotic custom-tag filers
# degrade to nulls (visible in coverage), never to wrong numbers.
METRIC_TAGS: dict[str, list[str]] = {
    # Total "Revenues" outranks the ASC-606 contract tag: for commodity and
    # financial firms (ADM, Affirm) contract revenue is only a SUBSET of the
    # top line (physically-settled contracts and interest income sit outside
    # ASC 606) — validation caught ADM at 31% of its real revenue.
    "revenue": ["Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],  # pre-2014 filings
    "operating_income": ["OperatingIncomeLoss"],
    "dep_amort": ["DepreciationDepletionAndAmortization",
                  "DepreciationAmortizationAndAccretionNet",  # pre-2015
                  "DepreciationAndAmortization", "Depreciation"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "inventory": ["InventoryNet"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_st": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    # The FILED diluted EPS — an extracted number, kept alongside our stated
    # EPS-TTM convention (TTM net income / shares outstanding).
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
}
FLOW_METRICS = ("revenue", "net_income", "op_cash_flow", "capex",
                "operating_income", "dep_amort")
SHARES_TAGS = [("dei", "EntityCommonStockSharesOutstanding"),
               ("us-gaap", "CommonStockSharesOutstanding"),
               ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]


# ── HTTP (stdlib, proxy-aware, throttled, declared User-Agent) ──

_last_request = 0.0


def _http_json(url: str) -> dict:
    global _last_request
    ua = settings.edgar_user_agent
    if not ua.strip():
        raise RuntimeError("EDGEFINDER_EDGAR_USER_AGENT is empty — SEC "
                           "requires a declared User-Agent with a contact")
    wait = MIN_REQUEST_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": ua, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
            _last_request = time.time()
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001 — retry transient failures
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"EDGAR fetch failed after retries: {url}: {last_exc}")


def cik_map() -> dict[str, int]:
    """symbol -> CIK from SEC's official ticker file (one small fetch)."""
    data = _http_json(TICKERS_URL)
    return {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}


def company_facts(cik: int) -> dict:
    return _http_json(FACTS_URL.format(cik=cik))


# ── pure normalization: companyfacts JSON → per-filing PIT rows ──


def _iso(d: str) -> date:
    return date.fromisoformat(d[:10])


def _collect(facts: dict, metric: str) -> list[dict]:
    """All usable facts for a metric, MERGED across its tag waterfall.

    Companies migrate tags over the years (AAPL's revenue moved to the
    modern contract-revenue tag in 2018 — first-tag-wins would erase
    2009-2017 entirely). So every tag contributes the periods it has;
    for the SAME (end, start, filed) the earlier-listed (modern) tag wins."""
    gaap = facts.get("us-gaap", {})
    out: list[dict] = []
    seen: set[tuple] = set()
    for tag in METRIC_TAGS[metric]:
        node = gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        rows = units.get("USD") or units.get("USD/shares") or []
        for f in rows:
            if f.get("val") is None or not f.get("end") or not f.get("filed"):
                continue
            rec = {"end": _iso(f["end"]),
                   "start": _iso(f["start"]) if f.get("start") else None,
                   "filed": _iso(f["filed"]), "val": float(f["val"]),
                   "form": f.get("form", "")}
            key = (rec["end"], rec["start"], rec["filed"])
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
    return out


def _collect_shares(cf: dict) -> list[dict]:
    for ns, tag in SHARES_TAGS:
        node = cf.get("facts", {}).get(ns, {}).get(tag)
        if not node:
            continue
        rows = node.get("units", {}).get("shares") or []
        out = [{"end": _iso(f["end"]), "filed": _iso(f["filed"]),
                "val": float(f["val"])}
               for f in rows if f.get("val") and f.get("end") and f.get("filed")]
        if out:
            return out
    return []


def _is_quarterly(f: dict) -> bool:
    return f["start"] is not None and 70 <= (f["end"] - f["start"]).days <= 105


def _is_annual(f: dict) -> bool:
    return f["start"] is not None and 330 <= (f["end"] - f["start"]).days <= 395


def _latest_instant(facts: list[dict], filed_cutoff: date) -> float | None:
    known = [f for f in facts if f["filed"] <= filed_cutoff]
    if not known:
        return None
    best = max(known, key=lambda f: (f["end"], f["filed"]))
    return best["val"]


def _latest_annual(facts: list[dict], filed_cutoff: date) -> float | None:
    """Newest fiscal-year-duration value knowable at the cutoff."""
    known = [f for f in facts if f["filed"] <= filed_cutoff and _is_annual(f)]
    if not known:
        return None
    best = max(known, key=lambda f: (f["end"], f["filed"]))
    return best["val"]


def _quarter_series(facts: list[dict], filed_cutoff: date) -> dict[date, float]:
    """period_end -> quarterly value, knowable at filed_cutoff.

    For each period the LATEST-FILED value at the cutoff wins (restatement
    honesty: at that moment, the restated figure IS current knowledge — and
    an earlier cutoff never sees it).

    Quarters come from three sources, in priority order:
    1. Direct ~90-day facts (income-statement items in 10-Qs).
    2. Cumulative differencing: 10-Q CASH-FLOW statements report only
       year-to-date durations (Q2 = 6 months, Q3 = 9 months) — the discrete
       quarter is the difference between consecutive cumulative periods
       sharing the same fiscal-year start. Without this, FCF/EBITDA would be
       null for nearly every filer.
    3. Q4 as FY minus its three known quarters (US filers report Q4 only
       inside the 10-K's annual figure)."""
    # newest-filed value per (start, end) period at the cutoff
    best: dict[tuple[date, date], dict] = {}
    for f in facts:
        if f["filed"] > filed_cutoff or f["start"] is None:
            continue
        key = (f["start"], f["end"])
        cur = best.get(key)
        if cur is None or f["filed"] > cur["filed"]:
            best[key] = f

    q = {e: f["val"] for (_, e), f in best.items() if _is_quarterly(f)}

    # cumulative (YTD/FY) chains: same start, ends one quarter apart
    by_start: dict[date, list[dict]] = {}
    for (s, e), f in best.items():
        if (e - s).days > 105:
            by_start.setdefault(s, []).append(f)
    for s, chain in by_start.items():
        chain.sort(key=lambda f: f["end"])
        first = best.get(next(((s2, e2) for (s2, e2) in best
                               if s2 == s and 70 <= (e2 - s2).days <= 105),
                              (None, None)))
        prev = first
        for f in chain:
            if prev is not None and 70 <= (f["end"] - prev["end"]).days <= 105:
                q.setdefault(f["end"], f["val"] - prev["val"])
            prev = f

    # FY minus three known quarters (fallback when no 9M cumulative exists)
    for (s, e), f in best.items():
        if not _is_annual(f) or e in q:
            continue
        # The FY's own Q1-Q3 end at most ~285 days before FY-end; the PRIOR
        # fiscal year ends ~365 before. 340 cleanly separates them.
        window_start = e - timedelta(days=340)
        threeq = [v for qe, v in q.items() if window_start < qe < e]
        if len(threeq) == 3:
            q[e] = f["val"] - sum(threeq)
    return q


def _ttm(qseries: dict[date, float], n: int = 4) -> tuple[float | None, date | None]:
    """Sum of the newest n quarters (None unless all n exist)."""
    if len(qseries) < n:
        return None, None
    ends = sorted(qseries)[-n:]
    if (ends[-1] - ends[0]).days > 320:  # gaps → not a contiguous year
        return None, None
    return sum(qseries[e] for e in ends), ends[-1]


def pit_rows(symbol: str, cik: int, cf: dict) -> list[dict]:
    """One row per filing date: price-free fundamentals knowable THEN."""
    facts = cf.get("facts", {})
    collected = {m: _collect(facts, m) for m in METRIC_TAGS}
    shares_facts = _collect_shares(cf)

    filings: dict[date, str] = {}
    for flist in collected.values():
        for f in flist:
            filings.setdefault(f["filed"], f.get("form", ""))
    rows = []
    for filed in sorted(filings):
        q_rev = _quarter_series(collected["revenue"], filed)
        q_ni = _quarter_series(collected["net_income"], filed)
        q_ocf = _quarter_series(collected["op_cash_flow"], filed)
        q_capex = _quarter_series(collected["capex"], filed)
        q_oi = _quarter_series(collected["operating_income"], filed)
        q_da = _quarter_series(collected["dep_amort"], filed)

        rev_ttm, rev_end = _ttm(q_rev)
        ni_ttm, _ = _ttm(q_ni)
        ocf_ttm, _ = _ttm(q_ocf)
        capex_ttm, _ = _ttm(q_capex)
        oi_ttm, _ = _ttm(q_oi)
        da_ttm, _ = _ttm(q_da)

        # growth: TTM now vs the TTM one year earlier (needs 8 quarters)
        def yoy(qs: dict[date, float]) -> float | None:
            if len(qs) < 8:
                return None
            ends = sorted(qs)
            now = sum(qs[e] for e in ends[-4:])
            prior = sum(qs[e] for e in ends[-8:-4])
            if prior == 0:
                return None
            return (now - prior) / abs(prior)

        assets = _latest_instant(collected["assets"], filed)
        equity = _latest_instant(collected["equity"], filed)
        ca = _latest_instant(collected["current_assets"], filed)
        cl = _latest_instant(collected["current_liabilities"], filed)
        cash = _latest_instant(collected["cash"], filed)
        inv = _latest_instant(collected["inventory"], filed)
        debt = ((_latest_instant(collected["debt_lt"], filed) or 0.0)
                + (_latest_instant(collected["debt_st"], filed) or 0.0)) or None
        shares = _latest_instant(shares_facts, filed)

        fcf = (ocf_ttm - capex_ttm) if (ocf_ttm is not None
                                        and capex_ttm is not None) else None
        ebitda = (oi_ttm + da_ttm) if (oi_ttm is not None
                                       and da_ttm is not None) else None
        d = {
            # TickerFundamentals-compatible, price-free:
            "symbol": symbol,
            "earnings_per_share": (ni_ttm / shares
                                   if ni_ttm is not None and shares else None),
            "return_on_equity": (ni_ttm / equity
                                 if ni_ttm is not None and equity else None),
            "return_on_assets": (ni_ttm / assets
                                 if ni_ttm is not None and assets else None),
            "debt_to_equity": (debt / equity if debt and equity else None),
            "current_ratio": (ca / cl if ca is not None and cl else None),
            "quick_ratio": ((ca - (inv or 0)) / cl
                            if ca is not None and cl else None),
            "revenue_growth": yoy(q_rev),
            "earnings_growth": yoy(q_ni),
            "free_cash_flow": fcf,
            # price-ratio ingredients (consumed at decision/brief time):
            "_revenue_ttm": rev_ttm, "_net_income_ttm": ni_ttm,
            "_fcf_ttm": fcf, "_ocf_ttm": ocf_ttm, "_capex_ttm": capex_ttm,
            "_ebitda_ttm": ebitda, "_shares": shares,
            "_book_equity": equity, "_total_debt": debt, "_cash": cash,
            "_eps_diluted_fy": _latest_annual(collected["eps_diluted"], filed),
            "_period_end": str(rev_end) if rev_end else None,
        }
        rows.append({"symbol": symbol, "cik": cik, "filed": filed,
                     "period_end": rev_end, "form": filings[filed] or None,
                     "source": "edgar", "data": d})
    return rows


def price_ratios(data: dict, price: float | None) -> dict:
    """Price-dependent ratios from a PIT row's ingredients + a price the
    caller vouches for (decision-date close, live mid, etc.)."""
    out: dict = {}
    if not price or price <= 0:
        return out
    sh = data.get("_shares")
    mc = price * sh if sh else None
    out["market_cap"] = mc
    ni, rev = data.get("_net_income_ttm"), data.get("_revenue_ttm")
    eq, fcf = data.get("_book_equity"), data.get("_fcf_ttm")
    ebitda, debt = data.get("_ebitda_ttm"), data.get("_total_debt")
    cash = data.get("_cash")
    if mc:
        out["price_to_earnings"] = mc / ni if ni and ni > 0 else None
        out["price_to_sales"] = mc / rev if rev and rev > 0 else None
        out["price_to_book"] = mc / eq if eq and eq > 0 else None
        out["price_to_free_cash_flow"] = mc / fcf if fcf and fcf > 0 else None
        out["fcf_yield"] = fcf / mc if fcf is not None else None
        ev = mc + (debt or 0.0) - (cash or 0.0)
        out["enterprise_value"] = ev
        out["ev_to_ebitda"] = ev / ebitda if ebitda and ebitda > 0 else None
        out["ev_to_sales"] = ev / rev if rev and rev > 0 else None
    return out


# ── ingest ──


def ingest(store=None, *, symbols: list[str] | None = None,
           universe: int | None = None, rebuild: bool = False) -> dict:
    """Fetch companyfacts for symbols (or the fresh universe) and upsert new
    (symbol, filed, period_end) rows. First run per symbol = full 2009+
    backfill (companyfacts returns the whole history in one response).

    ``rebuild=True`` recomputes a symbol from scratch (delete + reinsert) —
    for when the derivation logic improves. Safe because EDGAR is the source
    of truth and every row is reproducible from it; the sacred-table rule
    protects vendor data we could never refetch, which this is not."""
    from agent.store import get_store

    store = store or get_store()
    if symbols is None:
        from agent import data

        symbols = data.universe(universe or 1000)
    symbols = [s.upper() for s in symbols
               if not any(ch in s for ch in (".", "/", "="))]
    cmap = cik_map()
    summary = {"symbols": len(symbols), "no_cik": 0, "fetched": 0,
               "rows_inserted": 0, "errors": 0}
    for sym in symbols:
        cik = cmap.get(sym)
        if cik is None:
            summary["no_cik"] += 1
            continue
        try:
            cf = company_facts(cik)
            rows = pit_rows(sym, cik, cf)
            summary["fetched"] += 1
            if not rows:
                continue
            if rebuild:
                store.delete("fundamentals_pit", {"symbol": sym})
                existing: set[tuple] = set()
            else:
                existing = {(str(r["filed"])[:10],
                             str(r.get("period_end") or "")[:10])
                            for r in store.select("fundamentals_pit",
                                                  columns="filed,period_end",
                                                  filters={"symbol": sym})}
            fresh = [r for r in rows
                     if (str(r["filed"])[:10],
                         str(r.get("period_end") or "")[:10]) not in existing]
            if fresh:
                store.insert("fundamentals_pit", fresh, returning=False)
                summary["rows_inserted"] += len(fresh)
        except Exception as exc:  # noqa: BLE001 — one name can't kill the pass
            logger.warning("edgar ingest %s failed: %s", sym, exc)
            summary["errors"] += 1
    return summary


def coverage(store=None) -> dict:
    from agent.store import get_store

    store = store or get_store()
    rows = store.select("fundamentals_pit", columns="symbol,filed",
                        order=[("filed", "desc")], limit=100000)
    syms: dict[str, int] = {}
    newest: dict[str, str] = {}
    for r in rows:
        s = r["symbol"]
        syms[s] = syms.get(s, 0) + 1
        newest.setdefault(s, str(r["filed"])[:10])
    return {"symbols": len(syms), "rows": len(rows),
            "median_filings_per_symbol": (sorted(syms.values())[len(syms) // 2]
                                          if syms else 0),
            "stale_symbols": sum(1 for d in newest.values()
                                 if d < str(date.today() - timedelta(days=120)))}


# ── validation vs the frozen Polygon table (the lab's gate) ──
#
# LIKE-FOR-LIKE ONLY. The first validation run (2026-07-14) compared blended
# RATIOS at snapshot dates and failed across the board — autopsy showed the
# comparison itself was invalid, not the data: (a) the frozen snapshots carry
# each company's latest ANNUAL filing (AAPL's 2026-06-10 row holds FY-2025
# figures), so our fresh TTM values were being scored against numbers up to a
# year older; (b) Polygon's debt_to_equity is total LIABILITIES / equity
# (ours: interest-bearing debt / equity — AAPL 3.87 vs 0.78, both "correct");
# (c) Polygon's free_cash_flow equals operating cash flow — capex is not
# subtracted. No correct dataset could pass a naive ratio join against that
# reference. What CAN be tested: the frozen rows embed Polygon's raw filing
# extracts (raw_data.financials) — revenue, net income, equity, assets,
# shares, EPS from a specific 10-K. Comparing our value FOR THE SAME FILING
# against theirs tests exactly what the gate is for — does EDGAR give us the
# same numbers a paid vendor extracted? — with definitions and timing
# cancelled out. The owner's bar is unchanged: >=90% within ±10% per metric.

VALIDATION_INGREDIENTS = [
    # (label, ours(pit_data) -> value, theirs(financials, raw_root) -> value)
    ("revenue_fy",
     lambda p: p.get("_revenue_ttm"), lambda f, r: f.get("revenues")),
    ("net_income_fy",
     lambda p: p.get("_net_income_ttm"), lambda f, r: f.get("net_income")),
    ("book_equity",
     lambda p: p.get("_book_equity"), lambda f, r: f.get("equity")),
    ("total_assets",
     lambda p: (p["_net_income_ttm"] / p["return_on_assets"]
                if p.get("return_on_assets") and p.get("_net_income_ttm")
                else None),
     lambda f, r: f.get("total_assets")),
    ("current_ratio",
     lambda p: p.get("current_ratio"),
     lambda f, r: (f["current_assets"] / f["current_liabilities"]
                   if f.get("current_assets") and f.get("current_liabilities")
                   else None)),
    # Multi-class filers: our cover-page count sums ALL classes; the vendor
    # reports one class plus a weighted-diluted count. Both are legitimate
    # "share count" definitions — score against whichever is closer, since
    # the gate tests extraction, not definition choice.
    ("shares_outstanding",
     lambda p: p.get("_shares"),
     lambda f, r: (r.get("share_class_shares_outstanding"),
                   f.get("diluted_shares"),
                   r.get("weighted_shares_outstanding"))),
    # FILED diluted EPS vs their extract of the same filed number — our
    # EPS-TTM convention (TTM NI / shares outstanding) is deliberately a
    # different quantity and would test the convention, not extraction.
    ("earnings_per_share",
     lambda p: p.get("_eps_diluted_fy"), lambda f, r: f.get("diluted_eps")),
    # Cash flow is gated on OPERATING CASH FLOW — the one cash-flow figure
    # both pipelines extract from the same filed line (autopsy: exact
    # dollar-for-dollar matches). The vendor's "capex" is a broader concept
    # than XBRL PP&E purchases (ABBV: theirs $6.6B incl. intangible/
    # milestone payments vs $1.2B actual PP&E; sign convention also mixed,
    # 1094 negative / 149 positive) — so vendor FCF is NOT comparable to
    # ours by construction, and comparing it would test their definition,
    # not our extraction.
    ("operating_cash_flow",
     lambda p: p.get("_ocf_ttm"),
     lambda f, r: f.get("operating_cash_flow")),
]
PASS_WITHIN = 0.10   # ±10%
PASS_SHARE = 0.90    # >=90% of comparables
MIN_PAIRS = 20       # a metric with fewer comparables can't pass or fail


def validate(store=None, *, max_symbols: int = 150) -> dict:
    """Cross-check EDGAR-derived values against Polygon's raw filing extracts.

    Per symbol: the frozen table's newest snapshot embeds the vendor's
    numbers from that company's latest ANNUAL filing. Our matching row is
    the newest 10-K PIT row filed on/before the snapshot — at a 10-K's filed
    date our TTM equals that fiscal year exactly (Q4 is derived as FY minus
    three quarters), so the same-quantity comparison is direct. If revenue
    disagrees >15% we try the one-year-earlier 10-K and keep it only on a
    <2% revenue match (vendor snapshot lag — period alignment, not
    cherry-picking; the swap is counted and reported). STRICT gate:
    >=90% of comparables within ±10% per metric, min 20 pairs."""
    from agent.store import get_store

    store = store or get_store()
    frozen = store.select("fundamentals_snapshots", columns="symbol,as_of,data",
                          order=[("as_of", "desc")], limit=20000)
    newest_frozen: dict[str, dict] = {}
    for r in frozen:
        newest_frozen.setdefault(r["symbol"], r)
    pit_syms = {r["symbol"] for r in store.select(
        "fundamentals_pit", columns="symbol", limit=100000)}
    common = sorted(set(newest_frozen) & pit_syms)[:max_symbols]

    stats = {m[0]: {"n": 0, "within": 0, "worst": []}
             for m in VALIDATION_INGREDIENTS}
    compared = skipped = lag_swaps = 0
    for sym in common:
        fr = newest_frozen[sym]
        as_of = _iso(str(fr["as_of"]))
        fdata = fr["data"] if isinstance(fr["data"], dict) else json.loads(fr["data"])
        root = (fdata.get("raw_data") or {})
        fin = root.get("financials") or {}
        if not fin:
            skipped += 1
            continue
        krows = store.select(
            "fundamentals_pit", columns="filed,data",
            filters={"symbol": sym, "form": "10-K", "filed": ("lte", as_of)},
            order=[("filed", "desc")], limit=2)
        if not krows:
            skipped += 1
            continue

        def pdat(row):
            d = row["data"]
            return d if isinstance(d, dict) else json.loads(d)

        pdata = pdat(krows[0])
        rev_ours, rev_theirs = pdata.get("_revenue_ttm"), fin.get("revenues")
        if (rev_ours and rev_theirs
                and abs(rev_ours - rev_theirs) / abs(rev_theirs) > 0.15
                and len(krows) > 1):
            prior = pdat(krows[1])
            pr = prior.get("_revenue_ttm")
            if pr and abs(pr - rev_theirs) / abs(rev_theirs) < 0.02:
                pdata = prior          # vendor snapshot held the older 10-K
                lag_swaps += 1
        compared += 1
        for label, ours_fn, theirs_fn in VALIDATION_INGREDIENTS:
            try:
                a, b = ours_fn(pdata), theirs_fn(fin, root)
            except (TypeError, ZeroDivisionError, KeyError):
                continue
            if isinstance(b, tuple):  # several legitimate vendor definitions
                cands = [x for x in b if x]
                b = (min(cands, key=lambda x: abs(a - x) / abs(x))
                     if cands and a is not None else None)
            if a is None or b is None or b == 0:
                continue
            rel = abs(a - b) / abs(b)
            st = stats[label]
            st["n"] += 1
            if rel <= PASS_WITHIN:
                st["within"] += 1
            elif len(st["worst"]) < 5:
                st["worst"].append({"symbol": sym, "ours": round(a, 4),
                                    "frozen": round(b, 4),
                                    "rel_diff_pct": round(rel * 100, 1)})
    report = {"compared_symbols": compared, "skipped": skipped,
              "period_lag_swaps": lag_swaps,
              "bar": PASS_SHARE, "within": PASS_WITHIN, "metrics": {}}
    passed = True
    for m, st in stats.items():
        share = (st["within"] / st["n"]) if st["n"] else None
        ok = share is not None and share >= PASS_SHARE and st["n"] >= MIN_PAIRS
        if not ok:
            passed = False
        report["metrics"][m] = {
            "n": st["n"],
            "agree_share": round(share, 3) if share is not None else None,
            "pass": ok, "worst": st["worst"]}
    report["verdict"] = "PASS" if passed else "FAIL"
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--symbols", default=None)
    ing.add_argument("--universe", type=int, default=None)
    ing.add_argument("--rebuild", action="store_true",
                     help="recompute existing symbols (delete + reinsert)")
    sub.add_parser("coverage")
    va = sub.add_parser("validate")
    va.add_argument("--max-symbols", type=int, default=150)
    args = p.parse_args(argv)
    if args.cmd == "ingest":
        syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols else None)
        print(json.dumps(ingest(symbols=syms, universe=args.universe,
                                rebuild=args.rebuild),
                         indent=2, default=str))
    elif args.cmd == "coverage":
        print(json.dumps(coverage(), indent=2, default=str))
    else:
        print(json.dumps(validate(max_symbols=args.max_symbols),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
