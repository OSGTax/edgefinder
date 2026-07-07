"""Pure-function tests for the agent data-freshness seams:
- agent.data.merge_fresh_bars (R2 frames topped up with newer DB rows)
- agent.refresh.alpaca_news_to_rows (Alpaca news → ticker_news rows)
- agent.refresh.alpaca_corp_actions_to_rows (corporate actions → rows)
"""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd


def _frame(dates, close=100.0):
    return pd.DataFrame({
        "date": dates,
        "open": [close] * len(dates), "high": [close] * len(dates),
        "low": [close] * len(dates), "close": [close] * len(dates),
        "volume": [1000.0] * len(dates),
    })


def test_merge_fresh_bars_appends_only_newer():
    from agent.data import merge_fresh_bars

    base = {"SPY": _frame([date(2026, 6, 17), date(2026, 6, 18)])}
    fresh = {"SPY": _frame([date(2026, 6, 18), date(2026, 6, 19),
                            date(2026, 6, 22)], close=101.0)}
    out = merge_fresh_bars(base, fresh)
    got = list(out["SPY"]["date"])
    assert got == [date(2026, 6, 17), date(2026, 6, 18),
                   date(2026, 6, 19), date(2026, 6, 22)]
    # the overlapping 6/18 row kept the base (archive) values
    assert float(out["SPY"]["close"].iloc[1]) == 100.0
    assert list(out["SPY"].columns) == list(base["SPY"].columns)


def test_merge_fresh_bars_adds_missing_symbol_and_ignores_empty():
    from agent.data import merge_fresh_bars

    base = {"SPY": _frame([date(2026, 6, 18)])}
    fresh = {"NEWCO": _frame([date(2026, 6, 19)]), "EMPTY": _frame([])}
    out = merge_fresh_bars(base, fresh)
    assert "NEWCO" in out and len(out["NEWCO"]) == 1
    assert "EMPTY" not in out
    assert len(out["SPY"]) == 1  # untouched


def test_alpaca_news_to_rows_filters_and_normalizes():
    from agent.refresh import alpaca_news_to_rows

    ts = datetime(2026, 7, 7, 16, 29, 43, tzinfo=timezone.utc)
    items = [
        SimpleNamespace(headline="Apple tops", author="A. Writer", created_at=ts,
                        summary="sum", url="http://x", source="benzinga",
                        symbols=["AAPL", "ZZZZ"]),
        SimpleNamespace(headline=None, created_at=ts, symbols=["AAPL"]),  # skipped
        SimpleNamespace(headline="Apple tops", author=None, created_at=ts,
                        summary=None, url=None, source=None,
                        symbols=["AAPL"]),  # duplicate key skipped
    ]
    rows = alpaca_news_to_rows(items, {"AAPL", "GOOGL"})
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "AAPL" and r["title"] == "Apple tops"
    assert r["published_utc"] == "2026-07-07T16:29:43Z"  # Polygon-shaped key
    assert r["publisher_name"] == "benzinga"


def test_alpaca_corp_actions_to_rows():
    from agent.refresh import alpaca_corp_actions_to_rows

    data = {
        "cash_dividends": [
            SimpleNamespace(symbol="GOOGL", ex_date=date(2026, 6, 8), rate=0.22),
            SimpleNamespace(symbol="BAD", ex_date=None, rate=0.5),  # skipped
        ],
        "forward_splits": [
            SimpleNamespace(symbol="NVDA", ex_date=date(2026, 6, 10),
                            new_rate=4.0, old_rate=1.0),
            SimpleNamespace(symbol="ODD", ex_date=date(2026, 6, 11),
                            new_rate=3.0, old_rate=2.0),  # 3:2 ratio
        ],
        "reverse_splits": [
            SimpleNamespace(symbol="TINY", ex_date=date(2026, 6, 12),
                            new_rate=1.0, old_rate=10.0),
        ],
    }
    divs, splits = alpaca_corp_actions_to_rows(data)
    assert divs == [{"symbol": "GOOGL", "ex_date": "2026-06-08",
                     "cash_amount": 0.22}]
    assert {"symbol": "NVDA", "execution_date": "2026-06-10",
            "split_from": 1, "split_to": 4} in splits
    assert {"symbol": "ODD", "execution_date": "2026-06-11",
            "split_from": 2, "split_to": 3} in splits
    assert {"symbol": "TINY", "execution_date": "2026-06-12",
            "split_from": 10, "split_to": 1} in splits
