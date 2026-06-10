"""Bar loading for the portfolio engine — the one seam between DB and engine.

Phase 4 (two-tier storage) will put the R2 Parquet bar-store behind this same
interface; everything above it (walk-forward, CLI, future promotion runs)
already speaks only ``{symbol: DataFrame[date, open, high, low, close,
volume]}``.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from edgefinder.db.models import DailyBar, IndexDaily

_IN_CHUNK = 500  # keep IN() clauses bounded for the pooler


def load_bars(
    db: Session, symbols: list[str], start=None, end=None,
) -> dict[str, pd.DataFrame]:
    """Bulk-load daily bars for ``symbols`` as the engine's input shape."""
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
        by_symbol.setdefault(sym, []).append((dt, o, h, lo, c, v))

    cols = ["date", "open", "high", "low", "close", "volume"]
    return {
        sym: pd.DataFrame(data, columns=cols)
        for sym, data in by_symbol.items()
    }


def load_bars_from_store(
    symbols: list[str], start=None, end=None,
) -> dict[str, pd.DataFrame]:
    """Read bars from the R2 Parquet store instead of the DB.

    Same return shape as :func:`load_bars`. The store is a verified mirror of
    daily_bars (see edgefinder/data/barstore.py); callers opt in explicitly.
    """
    from edgefinder.data.barstore import BarStore

    bars = BarStore().load(symbols)
    if start is not None or end is not None:
        out = {}
        for sym, df in bars.items():
            if start is not None:
                df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]
            out[sym] = df.reset_index(drop=True)
        bars = out
    return bars


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
