"""Symbol chart data — bars (+ optional indicator series) and event markers.

Powers the Symbol Workstation. All times are UTC-midnight epoch seconds
(the dashboard-wide chart time standard). One bar load serves both the
candles and the indicator series (no double DB/R2 read).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from dashboard.symbol_service import get_bars

router = APIRouter()

RANGE_DAYS = {
    "1m": 31, "3m": 93, "6m": 186, "1y": 366, "2y": 731, "5y": 1827,
    "max": None,
}

# indicator series surfaced to the chart (subset of IndicatorSnapshot)
_INDICATOR_FIELDS = (
    "ema_9", "ema_21", "ema_50", "ema_200",
    "rsi",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower",
    "atr", "adx", "volume_ratio",
)


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _clean(v):
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) or math.isinf(f) else round(f, 6)


@router.get("/search")
def symbol_search(
    q: str = Query(..., min_length=1, max_length=12),
    limit: int = Query(10, le=25),
    db: Session = Depends(get_db),
):
    """Prefix search over every symbol we hold bars for (chart-page search).

    Replaces the retired workbench's /api/research/search — the tracked
    universe IS the searchable set now.
    """
    import re

    from edgefinder.db.models import DailyBar

    pat = re.sub(r"[^A-Z0-9.\-]", "", q.upper())
    if not pat:
        return {"results": []}
    rows = (db.query(DailyBar.symbol)
            .filter(DailyBar.symbol.like(pat + "%"))
            .distinct().order_by(DailyBar.symbol).limit(limit).all())
    return {"results": [{"symbol": r[0]} for r in rows]}


@router.get("/{symbol}/fundamentals")
def symbol_fundamentals(symbol: str, db: Session = Depends(get_db)):
    """Company research from SEC EDGAR point-in-time filings.

    Public-domain source — display anything. Returns the latest filing's
    price-free values, price ratios computed against our latest stored
    close (the price is named, the honesty convention everywhere here),
    and per-filing series for trend charts. Funds/ETFs don't file company
    fundamentals — ``covered: false`` with a plain explanation.
    """
    from agent.edgar import price_ratios
    from edgefinder.db.models import DailyBar, FundamentalsPit

    sym = symbol.upper()
    rows = (db.query(FundamentalsPit)
            .filter(FundamentalsPit.symbol == sym)
            .order_by(FundamentalsPit.filed).all())
    if not rows:
        return {"symbol": sym, "covered": False,
                "note": "No SEC company filings for this symbol — funds and "
                        "ETFs don't file company fundamentals, and newly "
                        "tracked stocks arrive with the nightly refresh."}
    latest = rows[-1]
    data = latest.data if isinstance(latest.data, dict) else {}
    bar = (db.query(DailyBar.date, DailyBar.close)
           .filter(DailyBar.symbol == sym)
           .order_by(desc(DailyBar.date)).first())
    close = float(bar[1]) if bar and bar[1] else None
    ratios = price_ratios(data, close)

    def pick(d, keys):
        return {k: d.get(k) for k in keys}

    series = [{"filed": str(r.filed), "form": r.form,
               **pick(r.data if isinstance(r.data, dict) else {}, (
                   "_revenue_ttm", "_net_income_ttm", "_fcf_ttm",
                   "earnings_per_share", "return_on_equity",
                   "debt_to_equity"))}
              for r in rows]
    return {
        "symbol": sym, "covered": True,
        "filings": len(rows),
        "first_filed": str(rows[0].filed), "latest_filed": str(latest.filed),
        "latest_form": latest.form,
        "price_used": close, "price_as_of": str(bar[0]) if bar else None,
        "snapshot": {**pick(data, (
            "earnings_per_share", "return_on_equity", "return_on_assets",
            "debt_to_equity", "current_ratio", "quick_ratio",
            "revenue_growth", "earnings_growth", "free_cash_flow",
            "_revenue_ttm", "_net_income_ttm", "_shares")), **ratios},
        "series": series,
    }


@router.get("/{symbol}/bars")
def symbol_bars(
    symbol: str,
    rng: str = Query("1y", alias="range", pattern="^(1m|3m|6m|1y|2y|5y|max)$"),
    indicators: bool = Query(False),
    db: Session = Depends(get_db),
):
    days = RANGE_DAYS[rng]
    start = date.today() - timedelta(days=days) if days else None

    # indicators need warmup history (ema_200 = 200 trading days); load
    # extra leading bars, compute, then clip the response to the range
    load_start = start
    if indicators and start is not None:
        load_start = start - timedelta(days=320)

    frame, meta = get_bars(db, symbol, load_start)
    if not len(frame):
        raise HTTPException(404, f"no bars for {symbol}")

    out = {
        "symbol": symbol.upper(),
        "range": rng,
        "source": meta["source"],
        "truncated": meta["truncated"],
        "live_through": meta.get("live_through"),
        "basis": "split-adjusted",
        "bars": [],
    }

    dates = list(frame["date"])
    epochs = [_epoch(d) for d in dates]
    clip_from = 0
    if start is not None:
        while clip_from < len(dates) and dates[clip_from] < start:
            clip_from += 1

    o, hi, lo, c, v = (frame[k] for k in ("open", "high", "low", "close", "volume"))
    out["bars"] = [
        {"time": epochs[i], "open": _clean(o.iloc[i]), "high": _clean(hi.iloc[i]),
         "low": _clean(lo.iloc[i]), "close": _clean(c.iloc[i]),
         "volume": _clean(v.iloc[i])}
        for i in range(clip_from, len(dates))
    ]

    if indicators:
        from edgefinder.data.indicator_engine import compute_snapshot_series

        snaps = compute_snapshot_series(
            frame[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
        series: dict[str, list] = {f: [] for f in _INDICATOR_FIELDS}
        for i in range(clip_from, len(snaps)):
            t = epochs[i]
            snap = snaps[i]
            for f in _INDICATOR_FIELDS:
                val = _clean(getattr(snap, f, None))
                if val is not None:
                    series[f].append({"time": t, "value": val})
        out["indicators"] = {f: pts for f, pts in series.items() if pts}

    return out


@router.get("/{symbol}/events")
def symbol_events(
    symbol: str,
    days: int = Query(3650, le=15000),
    db: Session = Depends(get_db),
):
    """Chart event markers: dividends, splits, news (sparse)."""
    # Post-Alpaca cutover: cash dividends live in the `dividends` table
    # (DividendRecord) written by agent.refresh from Alpaca corporate
    # announcements. The old TickerDividend/`ticker_dividends` table is stale
    # (Polygon-era, no live writes), so reading from it left the desk's
    # dividend markers silently empty. Splits + news writers are unchanged.
    from edgefinder.db.models import DividendRecord, TickerNews, TickerSplit

    sym = symbol.upper()
    cutoff = date.today() - timedelta(days=days)

    dividends = []
    for r in (db.query(DividendRecord)
              .filter(DividendRecord.symbol == sym,
                      DividendRecord.ex_date >= cutoff)
              .order_by(DividendRecord.ex_date).all()):
        if r.ex_date:
            dividends.append({"time": _epoch(r.ex_date),
                              "cash_amount": r.cash_amount,
                              "pay_date": None})

    splits = []
    for r in (db.query(TickerSplit)
              .filter(TickerSplit.symbol == sym,
                      TickerSplit.execution_date >= str(cutoff))
              .order_by(TickerSplit.execution_date).all()):
        d = _parse_date(r.execution_date)
        if d and r.split_from and r.split_to:
            splits.append({"time": _epoch(d), "from": r.split_from,
                           "to": r.split_to,
                           "ratio": f"{r.split_to:g}:{r.split_from:g}"})

    news = []
    for r in (db.query(TickerNews)
              .filter(TickerNews.symbol == sym)
              .order_by(TickerNews.published_utc.desc()).limit(100).all()):
        d = _parse_date(str(r.published_utc))
        if d:
            news.append({"time": _epoch(d), "title": r.title,
                         "url": r.article_url,
                         "publisher": r.publisher_name})
    news.reverse()

    # Live headline top-up: stored news advances with the nightly for the
    # maintained universe only — off-universe names read straight from
    # Alpaca at view time (read-only, TTL-cached, dedup by URL).
    live = _live_news(sym)
    if live:
        seen = {n["url"] for n in news if n.get("url")}
        fresh = [n for n in live if n.get("url") not in seen]
        if fresh:
            news = sorted(news + fresh, key=lambda n: n["time"])

    return {"symbol": sym, "dividends": dividends, "splits": splits, "news": news}


_news_cache = None


def _live_news(symbol: str) -> list[dict] | None:
    """Latest headlines for one symbol from Alpaca's news API (Benzinga).
    None when data creds are absent or the call fails — stored news still
    serves. Cached ~15 min per symbol; never written to ticker_news."""
    global _news_cache
    from config.settings import settings

    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        return None
    if _news_cache is None:
        from dashboard.ttl_cache import TTLCache

        _news_cache = TTLCache(maxsize=256, ttl_seconds=900)
    hit = _news_cache.get(symbol)
    if hit is not None:
        return hit
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(settings.alpaca_api_key,
                            settings.alpaca_api_secret)
        req = NewsRequest(symbols=symbol, limit=20)
        items = (client.get_news(req).data or {}).get("news") or []
        out = []
        for n in items:
            d = _parse_date(getattr(n, "created_at", None))
            if d:
                out.append({"time": _epoch(d),
                            "title": getattr(n, "headline", None),
                            "url": getattr(n, "url", None),
                            "publisher": getattr(n, "source", None)
                            or "Benzinga"})
    except Exception:  # noqa: BLE001 — the top-up is additive, never a 500
        return None
    _news_cache.set(symbol, out)
    return out


_quote_cache = None


def _fetch_snapshot(symbol: str) -> dict | None:
    """One live Alpaca snapshot → the research header's payload: last trade,
    bid/ask, today's developing daily bar, previous close. None without data
    creds or on any API error — the page falls back to stored closes."""
    from config.settings import settings

    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        return None
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest

        try:
            feed = DataFeed(settings.alpaca_data_feed or "sip")
        except ValueError:
            feed = DataFeed.SIP
        client = StockHistoricalDataClient(settings.alpaca_api_key,
                                           settings.alpaca_api_secret)
        snap = client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbol, feed=feed)
        ).get(symbol)
        if snap is None:
            return None
        lt, lq = snap.latest_trade, snap.latest_quote
        day, prev = snap.daily_bar, snap.previous_daily_bar
        last = (float(lt.price) if lt and lt.price
                else float(day.close) if day else None)
        prev_close = float(prev.close) if prev and prev.close else None
        return {
            "last": last,
            "last_ts": str(lt.timestamp) if lt else None,
            "bid": float(lq.bid_price) if lq and lq.bid_price else None,
            "ask": float(lq.ask_price) if lq and lq.ask_price else None,
            "prev_close": prev_close,
            "day_change_pct": (round((last / prev_close - 1) * 100, 2)
                               if last and prev_close else None),
            "day_bar": ({"time": _epoch(day.timestamp.date()),
                         "open": float(day.open), "high": float(day.high),
                         "low": float(day.low), "close": float(day.close),
                         "volume": float(day.volume)} if day else None),
        }
    except Exception:  # noqa: BLE001 — the live layer is additive, never a 500
        return None


@router.get("/{symbol}/quote")
def symbol_quote(symbol: str):
    """Live research quote: latest trade/quote + today's developing daily
    bar straight from Alpaca (read-only, any symbol), ~8s TTL per symbol so
    every open viewer shares one upstream call. ``available: false`` — never
    a 500 — without creds or on errors; the page then shows stored closes."""
    import re as _re

    global _quote_cache
    sym = _re.sub(r"[^A-Z0-9.\-]", "", symbol.upper())[:12]
    if not sym:
        raise HTTPException(404, "bad symbol")
    if _quote_cache is None:
        from dashboard.ttl_cache import TTLCache

        _quote_cache = TTLCache(maxsize=512, ttl_seconds=8)
    hit = _quote_cache.get(sym)
    if hit is not None:
        return hit
    data = _fetch_snapshot(sym)
    out = ({"symbol": sym, "available": True, **data} if data
           else {"symbol": sym, "available": False})
    _quote_cache.set(sym, out)
    return out


def _parse_date(s) -> date | None:
    if not s:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None
