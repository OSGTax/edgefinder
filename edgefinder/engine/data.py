"""Bar loading for the portfolio engine — the one seam between DB and engine.

Phase 4 (two-tier storage) will put the R2 Parquet bar-store behind this same
interface; everything above it (walk-forward, CLI, future promotion runs)
already speaks only ``{symbol: DataFrame[date, open, high, low, close,
volume]}``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from edgefinder.db.models import DailyBar, IndexDaily

_IN_CHUNK = 500  # keep IN() clauses bounded for the pooler

# Symbol-reuse contamination found by the 2026-06-10 fidelity audit: rows
# under a ticker that belong to a DIFFERENT security (recycled symbol).
# {symbol: first date the rows are the real company} — earlier rows dropped.
# META < 2022-06-09 is the Roundhill Ball Metaverse ETF, not Meta Platforms
# (FB renamed to META on 2022-06-09).
CONTAMINATED_BEFORE: dict[str, date] = {"META": date(2022, 6, 9)}


def load_bars(
    db: Session, symbols: list[str], start=None, end=None,
    split_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """Bulk-load daily bars for ``symbols`` as the engine's input shape.

    ``split_adjust`` (default ON — this is what makes the engine's stated
    "split-adjusted" price basis TRUE): daily_bars stores raw as-traded
    prints, so bars before each split ex-date are divided by the cumulative
    ratio (volume multiplied) using ticker_splits. Without it, every split
    inside a backtest window is a fake ±60-99% one-day move.
    """
    rows: list = []
    for i in range(0, len(symbols), _IN_CHUNK):
        q = db.query(
            DailyBar.symbol, DailyBar.date, DailyBar.open, DailyBar.high,
            DailyBar.low, DailyBar.close, DailyBar.volume,
        ).filter(DailyBar.symbol.in_(symbols[i:i + _IN_CHUNK]))
        if start is not None:
            q = q.filter(DailyBar.date >= start)
        if end is not None:
            q = q.filter(DailyBar.date <= end)
        rows.extend(q.all())

    by_symbol: dict[str, list] = {}
    for sym, dt, o, h, lo, c, v in rows:
        cutoff = CONTAMINATED_BEFORE.get(sym)
        if cutoff is not None and dt < cutoff:
            continue   # a different security traded under this ticker
        by_symbol.setdefault(sym, []).append((dt, o, h, lo, c, v))

    cols = ["date", "open", "high", "low", "close", "volume"]
    bars = {
        sym: pd.DataFrame(data, columns=cols)
        for sym, data in by_symbol.items()
    }
    if split_adjust and bars:
        bars = adjust_for_splits(bars, load_splits(db, list(bars)))
    return bars


def load_splits(db: Session, symbols: list[str]) -> dict[str, list[tuple]]:
    """{symbol: [(execution_date, factor), ...]} sorted; factor = to/from."""
    from edgefinder.db.models import TickerSplit

    out: dict[str, list[tuple]] = {}
    for i in range(0, len(symbols), _IN_CHUNK):
        rows = (db.query(TickerSplit.symbol, TickerSplit.execution_date,
                         TickerSplit.split_from, TickerSplit.split_to)
                .filter(TickerSplit.symbol.in_(symbols[i:i + _IN_CHUNK])).all())
        for sym, ex, f, t in rows:
            if not f or not t or f <= 0 or t <= 0 or f == t:
                continue
            ex_date = date.fromisoformat(str(ex)[:10])
            out.setdefault(sym, []).append((ex_date, t / f))
    for sym in out:
        out[sym].sort()
    return out


def adjust_for_splits(
    bars_by_symbol: dict[str, pd.DataFrame],
    splits: dict[str, list[tuple]],
) -> dict[str, pd.DataFrame]:
    """Back-adjust OHLCV for stock splits (price / factor, volume * factor
    for every bar strictly BEFORE each execution date). Dollar volume is
    split-invariant, so liquidity ranking and cost-model ADV are unaffected.
    Symbols without splits pass through unchanged (same object).
    """
    import bisect as _bisect

    out: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        events = splits.get(sym)
        if not events:
            out[sym] = df
            continue
        df = df.sort_values("date").reset_index(drop=True)
        dates = list(df["date"])
        price_f = pd.Series(1.0, index=df.index)
        vol_f = pd.Series(1.0, index=df.index)
        touched = False
        for ex_date, factor in events:
            idx = _bisect.bisect_left(dates, ex_date)
            if idx <= 0 or factor <= 0:
                continue   # split predates our history (or bogus row)
            price_f.iloc[:idx] /= factor
            vol_f.iloc[:idx] *= factor
            touched = True
        if not touched:
            out[sym] = df
            continue
        adj = df.copy()
        for col in ("open", "high", "low", "close"):
            adj[col] = df[col] * price_f
        adj["volume"] = df["volume"] * vol_f
        out[sym] = adj
    return out


def adjust_dividends_for_splits(
    dividends: dict[str, list[tuple]],
    splits: dict[str, list[tuple]],
) -> dict[str, list[tuple]]:
    """Scale dividend cash amounts onto the split-adjusted share basis.

    A dividend is declared per share AS OF ITS EX-DATE; splits executing
    AFTER that ex-date shrink the adjusted share, so the cash amount must
    shrink by the same cumulative factor — otherwise a pre-split dividend
    reads as an N-times yield against split-adjusted prices.
    """
    out: dict[str, list[tuple]] = {}
    for sym, divs in dividends.items():
        events = splits.get(sym)
        if not events:
            out[sym] = divs
            continue
        adjusted = []
        for ex_date, amount in divs:
            factor = 1.0
            for split_date, f in events:
                if split_date > ex_date:
                    factor *= f
            adjusted.append((ex_date, amount / factor))
        out[sym] = adjusted
    return out


def load_bars_from_store(
    symbols: list[str] | None, start=None, end=None,
) -> dict[str, pd.DataFrame]:
    """Read bars from the R2 Parquet store instead of the DB.

    Same return shape as :func:`load_bars`. The store is a verified mirror of
    daily_bars (see edgefinder/data/barstore.py); callers opt in explicitly.
    ``symbols=None`` loads the ENTIRE manifest — the full-market frame set
    that PIT universe resolution needs once the DB no longer carries breadth
    history. The symbol-reuse quarantine (CONTAMINATED_BEFORE) is applied
    here exactly as in the DB loader — the store mirrors raw rows, so the
    META pre-2022-06-09 ETF rows exist in it too.
    """
    from edgefinder.data.barstore import BarStore

    store = BarStore()
    if symbols is None:
        symbols = sorted(store.read_manifest())
    bars = store.load(symbols)
    for sym, cutoff in CONTAMINATED_BEFORE.items():
        if sym in bars:
            df = bars[sym]
            bars[sym] = df[df["date"] >= cutoff].reset_index(drop=True)
    if start is not None or end is not None:
        out = {}
        for sym, df in bars.items():
            if start is not None:
                df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]
            out[sym] = df.reset_index(drop=True)
        bars = out
    return {s: df for s, df in bars.items() if len(df)}


def rank_top_universe(
    bars_by_symbol: dict[str, pd.DataFrame], as_of, top_n: int,
    rank_offset: int = 0, rank_start=None, alive_days: int = 30,
) -> list[str]:
    """``resolve_universe('top')`` semantics computed from in-memory frames —
    used when bars come from the R2 store instead of the DB.

    Same rules as the SQL path (backtest/jobs.resolve_universe): rank by mean
    dollar volume over bars dated <= ``as_of`` (and >= ``rank_start`` when
    given — the trailing window), require a bar within ``alive_days`` of
    as_of (the graveyard gate), then take ``top_n`` after ``rank_offset``.
    Dollar volume is split-invariant, so raw store frames rank identically
    to raw DB rows. Ties break by symbol for determinism.
    """
    from datetime import timedelta

    alive_cutoff = as_of - timedelta(days=alive_days)
    scored: list[tuple[float, str]] = []
    for sym, df in bars_by_symbol.items():
        d = df["date"]
        mask = d <= as_of
        if rank_start is not None:
            mask = mask & (d >= rank_start)
        sub = df[mask]
        if not len(sub) or sub["date"].iloc[-1] < alive_cutoff:
            continue
        dv = float((sub["close"] * sub["volume"]).mean())
        scored.append((dv, sym))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [s for _, s in scored[rank_offset: rank_offset + top_n]]


def load_dividends(db: Session, symbols: list[str]) -> dict[str, list[tuple]]:
    """{symbol: [(ex_date, cash_amount), ...]} sorted by ex_date."""
    from edgefinder.db.models import DividendRecord

    out: dict[str, list[tuple]] = {}
    for i in range(0, len(symbols), _IN_CHUNK):
        rows = (db.query(DividendRecord.symbol, DividendRecord.ex_date,
                         DividendRecord.cash_amount)
                .filter(DividendRecord.symbol.in_(symbols[i:i + _IN_CHUNK]))
                .order_by(DividendRecord.symbol, DividendRecord.ex_date).all())
        for sym, ex, amt in rows:
            out.setdefault(sym, []).append((ex, amt))
    return out


def adjust_for_dividends(
    bars_by_symbol: dict[str, pd.DataFrame],
    dividends: dict[str, list[tuple]],
) -> dict[str, pd.DataFrame]:
    """Back-adjust OHLC to a total-return basis (CRSP-style).

    For each ex-date, all prices strictly BEFORE it are scaled by
    ``1 - dividend / close_on_the_day_before_ex`` — the standard
    back-adjustment that makes a buy-and-hold of the adjusted series equal
    the dividend-reinvested return. The LAST close is unchanged (adjusted
    and raw series converge at the present). Volume is untouched. Symbols
    without dividends pass through unchanged (same object).
    """
    out: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        divs = dividends.get(sym)
        if not divs:
            out[sym] = df
            continue
        df = df.sort_values("date").reset_index(drop=True)
        dates = list(df["date"])
        factor = pd.Series(1.0, index=df.index)
        import bisect as _bisect
        for ex_date, amount in divs:
            idx = _bisect.bisect_left(dates, ex_date)
            # idx == 0: ex-date precedes our history (nothing to adjust);
            # idx == len: a DECLARED future dividend that has not gone ex —
            # adjusting on it would rescale the entire series including the
            # last close on information the market hasn't priced yet
            if idx <= 0 or idx >= len(dates):
                continue
            prev_close = float(df["close"].iloc[idx - 1])
            if prev_close <= 0 or amount >= prev_close:
                continue                      # bad print / bogus dividend
            factor.iloc[:idx] *= (1.0 - amount / prev_close)
        adj = df.copy()
        for col in ("open", "high", "low", "close"):
            adj[col] = df[col] * factor
        # the raw close rides along: cost models must size liquidity (dollar
        # ADV vs absolute thresholds) on prices as they actually printed, not
        # on levels rescaled by FUTURE dividends
        adj["close_raw"] = df["close"]
        out[sym] = adj
    return out


def spy_series(db: Session) -> pd.DataFrame:
    """Longest available SPY series for benchmarking/regime tagging.

    Unions daily_bars with index_daily (daily_bars wins on overlap) because
    SPY coverage is split across both tables. Real opens are kept where
    available — the engine anchors the benchmark return at the first bar's
    open to match the strategy's first fill; missing opens fall back to the
    close. High/low are filled with the close (nothing reads them).
    """
    def _to_date(x):
        return x.date() if hasattr(x, "date") else x

    by_date: dict = {}   # date -> (open or None, close)
    for d, c in (db.query(IndexDaily.date, IndexDaily.close)
                 .filter(IndexDaily.symbol == "SPY").all()):
        if c:
            by_date[_to_date(d)] = (None, float(c))
    for d, o, c in (db.query(DailyBar.date, DailyBar.open, DailyBar.close)
                    .filter(DailyBar.symbol == "SPY").all()):
        if c:
            by_date[_to_date(d)] = (float(o) if o else None, float(c))

    cols = ["date", "open", "high", "low", "close", "volume"]
    if not by_date:
        return pd.DataFrame(columns=cols)
    items = sorted(by_date.items())
    closes = [c for _, (_, c) in items]
    return pd.DataFrame({
        "date": [d for d, _ in items],
        "open": [o if o else c for _, (o, c) in items],
        "high": closes, "low": closes, "close": closes,
        "volume": [0.0] * len(items),
    })
