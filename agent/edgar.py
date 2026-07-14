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
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "operating_income": ["OperatingIncomeLoss"],
    "dep_amort": ["DepreciationDepletionAndAmortization",
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


def _quarter_series(facts: list[dict], filed_cutoff: date) -> dict[date, float]:
    """period_end -> quarterly value, knowable at filed_cutoff.

    For each period the LATEST-FILED value at the cutoff wins (restatement
    honesty: at that moment, the restated figure IS current knowledge — and
    an earlier cutoff never sees it). Q4 is derived as FY minus its three
    known quarters when no explicit Q4 fact exists (US filers report Q4
    only inside the 10-K's annual figure)."""
    def newest_per_end(pred) -> dict[date, dict]:
        best: dict[date, dict] = {}
        for f in facts:
            if f["filed"] <= filed_cutoff and pred(f):
                cur = best.get(f["end"])
                if cur is None or f["filed"] > cur["filed"]:
                    best[f["end"]] = f
        return best

    q = {e: f["val"] for e, f in newest_per_end(_is_quarterly).items()}
    for e, f in newest_per_end(_is_annual).items():
        if e in q:
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
            "_fcf_ttm": fcf, "_ebitda_ttm": ebitda, "_shares": shares,
            "_book_equity": equity, "_total_debt": debt, "_cash": cash,
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
           universe: int | None = None) -> dict:
    """Fetch companyfacts for symbols (or the fresh universe) and upsert new
    (symbol, filed, period_end) rows. First run per symbol = full 2009+
    backfill (companyfacts returns the whole history in one response)."""
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
            existing = {(str(r["filed"])[:10], str(r.get("period_end") or "")[:10])
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

VALIDATION_METRICS = [
    # (our field, frozen field, price_based)
    ("earnings_per_share", "earnings_per_share", False),
    ("return_on_equity", "return_on_equity", False),
    ("return_on_assets", "return_on_assets", False),
    ("debt_to_equity", "debt_to_equity", False),
    ("current_ratio", "current_ratio", False),
    ("price_to_earnings", "price_to_earnings", True),
    ("price_to_sales", "price_to_sales", True),
    ("price_to_book", "price_to_book", True),
    ("market_cap", "market_cap", True),
]
PASS_WITHIN = 0.10   # ±10%
PASS_SHARE = 0.90    # >=90% of comparables


def validate(store=None, *, max_symbols: int = 150) -> dict:
    """Cross-check EDGAR-derived values against the frozen vendor table.

    For each symbol present in both: take the frozen row at its newest as_of,
    the newest PIT row filed <= that as_of, and the daily_bars close on/before
    it for price-based ratios. Per-metric agreement stats; the STRICT gate is
    >=90% of comparables within ±10%. Fundamentals stay quarantined from the
    lab until this passes (owner decision 2026-07-14)."""
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

    stats = {m[0]: {"n": 0, "within": 0, "worst": []} for m in VALIDATION_METRICS}
    for sym in common:
        fr = newest_frozen[sym]
        as_of = _iso(str(fr["as_of"]))
        fdata = fr["data"] if isinstance(fr["data"], dict) else json.loads(fr["data"])
        prows = store.select("fundamentals_pit", columns="filed,data",
                             filters={"symbol": sym, "filed": ("lte", as_of)},
                             order=[("filed", "desc")], limit=1)
        if not prows:
            continue
        pdata = prows[0]["data"]
        if not isinstance(pdata, dict):
            pdata = json.loads(pdata)
        bars = store.select("daily_bars", columns="date,close",
                            filters={"symbol": sym, "date": ("lte", as_of)},
                            order=[("date", "desc")], limit=1)
        close = float(bars[0]["close"]) if bars else None
        ours = {**pdata, **price_ratios(pdata, close)}
        for our_f, frozen_f, _price_based in VALIDATION_METRICS:
            a, b = ours.get(our_f), fdata.get(frozen_f)
            if a is None or b is None or b == 0:
                continue
            rel = abs(a - b) / abs(b)
            st = stats[our_f]
            st["n"] += 1
            if rel <= PASS_WITHIN:
                st["within"] += 1
            elif len(st["worst"]) < 5:
                st["worst"].append({"symbol": sym, "ours": round(a, 4),
                                    "frozen": round(b, 4),
                                    "rel_diff_pct": round(rel * 100, 1)})
    report = {"compared_symbols": len(common), "bar": PASS_SHARE,
              "within": PASS_WITHIN, "metrics": {}}
    passed = True
    for m, st in stats.items():
        share = (st["within"] / st["n"]) if st["n"] else None
        ok = share is not None and share >= PASS_SHARE and st["n"] >= 20
        if share is not None and not ok:
            passed = False
        report["metrics"][m] = {"n": st["n"],
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
    sub.add_parser("coverage")
    va = sub.add_parser("validate")
    va.add_argument("--max-symbols", type=int, default=150)
    args = p.parse_args(argv)
    if args.cmd == "ingest":
        syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols else None)
        print(json.dumps(ingest(symbols=syms, universe=args.universe),
                         indent=2, default=str))
    elif args.cmd == "coverage":
        print(json.dumps(coverage(), indent=2, default=str))
    else:
        print(json.dumps(validate(max_symbols=args.max_symbols),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
