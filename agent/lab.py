"""The Strategy Lab — compute-driven strategy search over 21 years of bars.

The trading brain's cadence is bounded by how often it should TRADE; learning
is not. This tool is where mass experimentation belongs: sweep a grid of
rule x parameter x universe x schedule combos through the kept backtest
engine, score every combo on SPLIT-SAMPLE CONSISTENCY, and publish an honest
leaderboard the nightly brief carries to every trading cycle.

The honesty rules (multiple-comparisons discipline):
- Every combo is judged on TWO halves of history (in-sample / out-sample,
  split at SPLIT_DATE). A strategy "qualifies" only when it beats SPY net of
  costs in BOTH halves — a lucky fit rarely survives the half it wasn't
  fitted to.
- The SPY benchmark is TOTAL-RETURN (dividend-adjusted, same basis as the
  strategy bars, as of 2026-07-16): a price-only SPY handed every combo
  ~+50pp of phantom in-sample excess (the compounded dividend yield).
- Universe membership is ranked as of SPLIT_DATE when the local data allows;
  otherwise the result is labeled survivorship-inflated (``universe_basis``
  on every result row and leaderboard entry).
- Ranking is by the WORST half's excess return (maximin), not the best —
  the number a skeptic would quote.
- The leaderboard always reports how many combos were tested alongside the
  winners. Picking 5 of 300 tested means expect shrinkage live; saying so
  is the difference between research and marketing.
- Bars are loaded ONCE per universe and reused across the whole grid
  (loading dominates backtest cost).

CLI (JSON out, like every agent tool):
  python -m agent.lab sweep --max-combos 80 --time-budget-secs 2400
  python -m agent.lab leaderboard --top 10
"""

from __future__ import annotations

import json
import time
from datetime import date

# Where in-sample ends and out-of-sample begins. Chosen so both halves hold
# multiple regimes (GFC + 2010s bull in-sample; covid, 2022 bear, the AI bull
# out-of-sample). Fixed, not data-mined.
SPLIT_DATE = date(2018, 1, 1)
IN_SAMPLE_START = date(2006, 1, 1)

RULES = (
    ["equal_weight"]
    + [f"momentum:{k}" for k in (3, 5, 8, 12)]
    + [f"momo_trend:{k}" for k in (3, 5, 8, 12)]
    + [f"meanrev:{k}" for k in (3, 5, 8)]
    + [f"breakout:{k}" for k in (3, 5, 8)]
    + [f"regime_momentum:{k}" for k in (3, 5, 8, 12)]
    # PIT SEC fundamentals (validation gate PASSED 2026-07-14 — see
    # docs/fundamentals-validation.md). XBRL floor: no filings before ~2009,
    # so these rules hold cash pre-2009 and their effective in-sample window
    # is 2009-2017 — judge their split-sample consistency accordingly.
    + [f"value_momentum:{k}" for k in (5, 8)]
)
SCHEDULES = ("weekly", "monthly")
# Most-liquid slices of the hot set, plus mid200 (dollar-volume ranks 41-240):
# a rule that only works on megacaps is riding fame, not edge. Membership is
# ranked as of SPLIT_DATE when the hot set still holds bars there (see
# _resolve_universe); when it doesn't, the sweep falls back to present-day
# ranking and stamps the result survivorship-inflated — deep-history tests on
# today's top names back-test tomorrow's survivors, which is free excess.
UNIVERSES = ("top20", "top40", "top60", "mid200")


def build_grid() -> list[dict]:
    """The full combo grid, deterministic order."""
    return [{"rule": r, "schedule": s, "universe": u}
            for u in UNIVERSES for r in RULES for s in SCHEDULES]


def score_combo(half_a: dict, half_b: dict) -> dict:
    """Split-sample verdict for one combo (pure logic).

    qualifies = positive excess vs SPY in BOTH halves.
    score     = the WORST half's excess (maximin) — what a skeptic quotes.
    """
    ex_a = half_a.get("excess_return_pct")
    ex_b = half_b.get("excess_return_pct")
    if ex_a is None or ex_b is None:
        return {"qualifies": False, "score": None,
                "note": "missing excess in one half"}
    qualifies = ex_a > 0 and ex_b > 0
    return {"qualifies": qualifies,
            "score": round(min(ex_a, ex_b), 2),
            "in_sample_excess_pct": round(ex_a, 2),
            "out_sample_excess_pct": round(ex_b, 2),
            "sharpe_out": half_b.get("sharpe"),
            "max_dd_out": half_b.get("max_drawdown_pct")}


def _universe_symbols(name: str, as_of: date | None = None) -> list[str]:
    from agent import data

    if name == "mid200":
        syms = data.universe(240, as_of=as_of)[40:]
    else:
        n = {"top20": 20, "top40": 40, "top60": 60}[name]
        syms = data.universe(n, as_of=as_of)
    if "SPY" not in syms:
        syms = [*syms, "SPY"]  # regime gauge for regime_momentum; harmless else
    return syms


def _pit_breadth(as_of: date, *, need: int) -> int:
    """Bounded count of DISTINCT NON-PROTECTED symbols with a bar in the 10
    calendar days ending at ``as_of`` (inclusive both sides).

    The old probe anchored on SPY — but SPY is one of the ~10
    ``DB_PROTECTED_ETFS`` whose FULL history always stays in ``daily_bars``,
    so it passed every night while the hot set held nothing else near
    ``as_of``; the PIT "ranking" then returned exactly those deep-history
    ETFs, and an ETF-only sweep got mislabeled with the trusted as_of basis.
    Real breadth means names beyond the protected keeps, so only those count.
    Bounded: a 10-calendar-day window holds ≤8 sessions, so ordering by
    symbol guarantees the first ``cap`` rows span at least
    ``need + len(protected)`` distinct symbols when they exist — never a
    full-table scan, on either transport."""
    from datetime import timedelta

    from agent.store import get_store
    from edgefinder.data.barstore import DB_PROTECTED_ETFS

    protected = {s.upper() for s in DB_PROTECTED_ETFS}
    cap = (need + len(protected) + 5) * 10
    rows = get_store().select(
        "daily_bars", columns="symbol",
        filters={"date": [("gte", as_of - timedelta(days=10)),
                          ("lte", as_of)]},
        order=[("symbol", "asc")], limit=cap)
    return len({str(r["symbol"]).upper() for r in rows} - protected)


def _resolve_universe(name: str) -> tuple[list[str], str]:
    """Universe symbols plus the honest ``universe_basis`` label.

    Ranking membership as of TODAY back-tests tomorrow's survivors — free
    excess for any long rule. So try a point-in-time ranking as of SPLIT_DATE
    first (the out-sample half's information boundary) — but only when the
    hot set holds REAL breadth there. Two belts, both counting only
    NON-protected symbols (the protected ETFs keep deep history forever, so
    they prove nothing about coverage): the ``_pit_breadth`` probe must find
    at least ``floor`` names with bars near SPLIT_DATE BEFORE the ranked
    universe call, and the RESOLVED set must itself hold ``floor``
    non-protected names. In today's production shape (10 deep ETFs + a
    ~400-day hot set) both gates fail, so every universe falls back to
    present-day ranking and SAYS SO in the label every result row carries.
    TODO: true SPLIT_DATE-era (and 2006-era in-sample) PIT ranking needs
    R2-depth frames — rank via ``edgefinder.engine.data.rank_top_universe``
    over the archive.
    """
    from edgefinder.data.barstore import DB_PROTECTED_ETFS

    requested = 200 if name == "mid200" else int(name.removeprefix("top"))
    floor = max(requested // 2, 5)
    protected = {s.upper() for s in DB_PROTECTED_ETFS}
    try:
        if _pit_breadth(SPLIT_DATE, need=floor) >= floor:
            syms = _universe_symbols(name, as_of=SPLIT_DATE)
            # Belt two: a resolved set that is mostly protected ETFs is the
            # hot set echoing its keep-list, not a 2018 universe.
            if sum(1 for s in syms if s.upper() not in protected) >= floor:
                return syms, f"as_of_{SPLIT_DATE}"
    except Exception:  # noqa: BLE001 — PIT ranking is best-effort
        pass
    return _universe_symbols(name), "present_day_survivorship_inflated"


def sweep(*, max_combos: int = 80, time_budget_secs: int = 2400,
          offset: int | None = None, run_id: str | None = None) -> dict:
    """Run up to ``max_combos`` grid combos within the time budget.

    ``offset`` rotates the starting point through the grid so successive
    nights cover different regions (defaults to day-of-year so the rotation
    is deterministic per night, no scheduling state needed). Each combo runs
    twice (both halves) and is persisted to desk_backtests with a lab: label.
    """
    from agent import backtest_tool, data
    from agent.store import get_store

    t0 = time.time()
    grid = build_grid()
    if offset is None:
        offset = (date.today().timetuple().tm_yday * 7) % len(grid)
    order = grid[offset:] + grid[:offset]

    store = get_store()
    # Total-return SPY: the strategy bars are dividend-adjusted, so the
    # benchmark must be too — excess is TR-vs-TR, never TR-vs-price.
    bench = data.spy_series_df(total_return=True)
    # TR is only as deep as the dividends table: if SPY's first stored
    # ex-date lands AFTER the in-sample start, the in-sample half's "TR"
    # benchmark is effectively price-only there. Stamp the coverage on every
    # verdict so the data-bounded efficacy is visible, never silent.
    div_min = store.select("dividends", columns="ex_date",
                           filters={"symbol": "SPY"},
                           order=[("ex_date", "asc")], limit=1)
    benchmark_div_from = str(div_min[0]["ex_date"])[:10] if div_min else None
    bars_cache: dict[str, dict] = {}
    basis_cache: dict[str, str] = {}
    tested = qualified = errors = 0
    results: list[dict] = []

    for combo in order:
        if tested >= max_combos or (time.time() - t0) > time_budget_secs:
            break
        uni = combo["universe"]
        try:
            if uni not in bars_cache:
                syms, basis_cache[uni] = _resolve_universe(uni)
                bars = data.load_bars(syms, div_adjust=True, source="auto")
                bars_cache[uni] = {s: df for s, df in bars.items()
                                   if df is not None and len(df) > 210}
            bars = bars_cache[uni]
            if len(bars) < 5:
                errors += 1
                continue
            # Two honest halves: in-sample is bounded by TRIMMING bars at the
            # split (no peeking), out-sample simply starts trading there.
            half_a = _bounded_half(bars, bench, combo, backtest_tool)
            half_b = backtest_tool.run_prepared(
                bars, bench, combo["rule"], schedule=combo["schedule"],
                start=SPLIT_DATE)
            verdict = score_combo(half_a, half_b)
            # Stamp the membership basis on every result payload so the
            # leaderboard/brief carry the survivorship caveat with the number.
            verdict["universe_basis"] = basis_cache[uni]
            verdict["benchmark_div_from"] = benchmark_div_from
            tested += 1
            qualified += 1 if verdict["qualifies"] else 0
            row = {**combo, **verdict}
            results.append(row)
            store.insert("desk_backtests", {
                "account": "agent", "run_id": run_id,
                "label": f"lab:{combo['rule']}@{uni}/{combo['schedule']}",
                "spec": combo,
                "result": {**verdict,
                           "out_sample": half_b,
                           "split": str(SPLIT_DATE)}}, returning=False)
        except Exception as exc:  # noqa: BLE001 — one combo can't kill the sweep
            errors += 1
            results.append({**combo, "error": f"{type(exc).__name__}: {exc}"[:120]})

    ranked = sorted((r for r in results if r.get("qualifies")),
                    key=lambda r: -(r.get("score") or -999))
    return {"tested": tested, "qualified": qualified, "errors": errors,
            "elapsed_secs": round(time.time() - t0, 1),
            "grid_size": len(grid), "offset": offset,
            "split": str(SPLIT_DATE),
            "benchmark_div_from": benchmark_div_from,
            "honesty": f"{qualified} of {tested} tested combos qualified "
                       "(positive excess in BOTH halves); expect live "
                       "results to shrink toward the worst half.",
            "top": ranked[:10]}


def _before_split(df):
    """Rows strictly before SPLIT_DATE, robust to the date column's dtype."""
    import pandas as pd

    mask = pd.to_datetime(df["date"]) < pd.Timestamp(SPLIT_DATE)
    return df[mask]


def _bounded_half(bars: dict, bench, combo: dict, backtest_tool) -> dict:
    """In-sample half: start→SPLIT_DATE, honestly bounded by trimming bars —
    the engine literally cannot see a post-split row."""
    trimmed = {}
    for s, df in bars.items():
        t = _before_split(df)
        if len(t) > 210:
            trimmed[s] = t
    if len(trimmed) < 5:
        return {"excess_return_pct": None}
    b = bench
    if b is not None and len(b):
        b = _before_split(b)
    return backtest_tool.run_prepared(trimmed, b, combo["rule"],
                                      schedule=combo["schedule"],
                                      start=IN_SAMPLE_START)


def leaderboard(*, top: int = 10, days: int = 14) -> dict:
    """The current lab leaderboard from recent persisted sweeps.

    Dedupes by (rule, universe, schedule) keeping the newest result, keeps
    qualifiers only, ranks by worst-half excess. Always reports tested count."""
    from datetime import datetime, timedelta, timezone

    from agent.store import get_store

    store = get_store()
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None)
              - timedelta(days=days))
    rows = store.select("desk_backtests", filters={"account": "agent"},
                        order=[("ts", "desc")], limit=500)
    seen: set[tuple] = set()
    entries, tested = [], 0
    for r in rows:
        label = str(r.get("label") or "")
        if not label.startswith("lab:"):
            continue
        if str(r.get("ts") or "") < str(cutoff):
            continue
        spec = r.get("spec") or {}
        key = (spec.get("rule"), spec.get("universe"), spec.get("schedule"))
        if key in seen:
            continue
        seen.add(key)
        tested += 1
        res = r.get("result") or {}
        if res.get("qualifies"):
            entry = {"rule": spec.get("rule"),
                     "universe": spec.get("universe"),
                     "schedule": spec.get("schedule"),
                     "score": res.get("score"),
                     "in_sample_excess_pct": res.get("in_sample_excess_pct"),
                     "out_sample_excess_pct": res.get("out_sample_excess_pct"),
                     "sharpe_out": res.get("sharpe_out"),
                     "max_dd_out": res.get("max_dd_out"),
                     # rows persisted before the basis stamp were all
                     # ranked present-day — label them honestly
                     "universe_basis": res.get("universe_basis")
                     or "present_day_survivorship_inflated"}
            # carried only when the sweep stamped it — an absent key is a
            # legacy row (unknown coverage), not "no dividends"
            if "benchmark_div_from" in res:
                entry["benchmark_div_from"] = res["benchmark_div_from"]
            entries.append(entry)
    entries.sort(key=lambda e: -(e.get("score") or -999))
    survivorship = sum(1 for e in entries
                       if e["universe_basis"] != f"as_of_{SPLIT_DATE}")
    # Benchmark dividend coverage starting AFTER the in-sample start means
    # the in-sample "TR" benchmark was effectively price-only — the TR fix's
    # efficacy is data-bounded, and that must ride with the numbers.
    thin_bench = sum(
        1 for e in entries
        if "benchmark_div_from" in e
        and (e["benchmark_div_from"] is None
             or str(e["benchmark_div_from"])[:10] > str(IN_SAMPLE_START)))
    honesty = (f"top picks are {min(top, len(entries))} of "
               f"{tested} tested — expect live shrinkage toward "
               "the worst-half number")
    if survivorship:
        honesty += (f"; {survivorship} of {len(entries)} qualifiers ranked "
                    "their universe present-day (survivorship-inflated)")
    if thin_bench:
        honesty += (f"; {thin_bench} of {len(entries)} qualifiers scored vs "
                    "a benchmark whose dividend coverage starts after the "
                    "in-sample start (in-sample TR is effectively price-only)")
    return {"window_days": days, "combos_tested": tested,
            "qualified": len(entries),
            "honesty": honesty,
            "top": entries[:top]}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--max-combos", type=int, default=80)
    sw.add_argument("--time-budget-secs", type=int, default=2400)
    sw.add_argument("--offset", type=int, default=None)
    sw.add_argument("--run-id", default=None)
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--top", type=int, default=10)
    lb.add_argument("--days", type=int, default=14)
    args = p.parse_args(argv)
    if args.cmd == "sweep":
        print(json.dumps(sweep(max_combos=args.max_combos,
                               time_budget_secs=args.time_budget_secs,
                               offset=args.offset, run_id=args.run_id),
                         indent=2, default=str))
    else:
        print(json.dumps(leaderboard(top=args.top, days=args.days),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
