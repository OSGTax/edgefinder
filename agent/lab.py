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
- Ranking is by the WORST half's excess return (maximin), not the best —
  the number a skeptic would quote.
- The leaderboard always reports how many combos were tested alongside the
  winners. Picking 5 of 300 tested means expect shrinkage live; saying so
  is the difference between research and marketing.
- Bars are loaded ONCE for a broad candidate pool and reused across the
  whole grid (loading dominates backtest cost); each universe bucket's
  actual symbol list is then ranked POINT-IN-TIME per half, not off today's
  dollar volume — see _pit_universe.

CLI (JSON out, like every agent tool):
  python -m agent.lab sweep --max-combos 80 --time-budget-secs 2400
  python -m agent.lab leaderboard --top 10
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

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
# a rule that only works on megacaps is riding fame, not edge — and deep-
# history tests on today's top names carry survivorship shine that mid-tier
# slices partially deflate.
UNIVERSES = ("top20", "top40", "top60", "mid200")
# Broad superset the universe pool loads bars for, wide enough to cover any
# top20/40/60/mid200 slice AS RANKED AT ANY POINT-IN-TIME in the backtest —
# see _pit_universe. A name outside this pool today (thin/delisted since)
# is invisible to the Lab even if it was genuinely liquid decades ago; that
# residual gap is a data-availability limit of the hot set, not something
# this fix can close, and is disclosed rather than hidden.
POOL_SIZE = 300


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


def _universe_pool_symbols() -> list[str]:
    """The broad candidate pool, loaded once and PIT-ranked per half by
    ``_pit_universe`` below — see POOL_SIZE."""
    from agent import data

    syms = data.universe(POOL_SIZE)
    if "SPY" not in syms:
        syms = [*syms, "SPY"]  # regime gauge for regime_momentum; harmless else
    return syms


def _pit_universe(pool_bars: dict, name: str, as_of) -> list[str]:
    """Point-in-time symbol list for universe ``name``, ranked by dollar
    volume using ONLY bars dated <= ``as_of`` (a trailing 190-day window,
    matching ``agent.data.universe``'s own default).

    This is the fix for a real bug: the old ``_universe_symbols`` ranked by
    TODAY's dollar volume and reused that ONE list across the entire
    2006-2026 backtest window. That reserves a permanent universe slot for
    every eventual future winner (SOXL, TQQQ, ARM, APP, NBIS, ...) for
    years before they existed or were liquid, which a top-K momentum rule
    then rides for its whole real run — the Lab itself flagged the result
    (momentum:3 monthly/top60 printing +5520% out-sample excess against a
    +44% in-sample score, a 30-55x gap) as implausible on 2026-07-15. Using
    ``rank_top_universe`` (the same PIT machinery live trading already uses
    for its own universe scans) instead of a static today's-list closes
    that look-ahead bias for good.
    """
    from edgefinder.engine.data import rank_top_universe

    rank_start = as_of - timedelta(days=190)
    if name == "mid200":
        syms = rank_top_universe(pool_bars, as_of, 200, rank_offset=40,
                                 rank_start=rank_start)
    else:
        n = {"top20": 20, "top40": 40, "top60": 60}[name]
        syms = rank_top_universe(pool_bars, as_of, n, rank_start=rank_start)
    if "SPY" not in syms and "SPY" in pool_bars:
        syms = [*syms, "SPY"]
    return syms


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
    bench = data.spy_series_df()
    pool_syms = _universe_pool_symbols()
    pool_bars_raw = data.load_bars(pool_syms, div_adjust=True, source="auto")
    pool_bars = {s: df for s, df in pool_bars_raw.items()
                if df is not None and len(df) > 210}
    # Each half's universe is ranked ONCE (as of the day before that half
    # starts trading — see _pit_universe) and reused across every combo that
    # shares the same universe bucket, same caching shape as the old
    # per-universe bars_cache.
    uni_cache: dict[str, dict] = {}
    tested = qualified = errors = 0
    results: list[dict] = []

    for combo in order:
        if tested >= max_combos or (time.time() - t0) > time_budget_secs:
            break
        uni = combo["universe"]
        try:
            if uni not in uni_cache:
                in_syms = _pit_universe(pool_bars, uni,
                                        IN_SAMPLE_START - timedelta(days=1))
                out_syms = _pit_universe(pool_bars, uni,
                                         SPLIT_DATE - timedelta(days=1))
                uni_cache[uni] = {
                    "in": {s: pool_bars[s] for s in in_syms if s in pool_bars},
                    "out": {s: pool_bars[s] for s in out_syms if s in pool_bars},
                }
            in_bars, out_bars = uni_cache[uni]["in"], uni_cache[uni]["out"]
            if len(out_bars) < 5:
                errors += 1
                continue
            # Two honest halves: in-sample is bounded by TRIMMING bars at the
            # split (no peeking), out-sample simply starts trading there.
            # Each half also trades its OWN point-in-time universe, not
            # today's — see _pit_universe.
            half_a = _bounded_half(in_bars, bench, combo, backtest_tool)
            half_b = backtest_tool.run_prepared(
                out_bars, bench, combo["rule"], schedule=combo["schedule"],
                start=SPLIT_DATE)
            verdict = score_combo(half_a, half_b)
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
            entries.append({"rule": spec.get("rule"),
                            "universe": spec.get("universe"),
                            "schedule": spec.get("schedule"),
                            "score": res.get("score"),
                            "in_sample_excess_pct": res.get("in_sample_excess_pct"),
                            "out_sample_excess_pct": res.get("out_sample_excess_pct"),
                            "sharpe_out": res.get("sharpe_out"),
                            "max_dd_out": res.get("max_dd_out")})
    entries.sort(key=lambda e: -(e.get("score") or -999))
    return {"window_days": days, "combos_tested": tested,
            "qualified": len(entries),
            "honesty": f"top picks are {min(top, len(entries))} of "
                       f"{tested} tested — expect live shrinkage toward "
                       "the worst-half number",
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
