"""Tests for edgefinder/data/flatfiles.py (no network)."""

import gzip
from datetime import date

import pandas as pd

from edgefinder.data.flatfiles import (
    FlatFilesClient,
    day_aggs_to_rows,
    parse_aggs_csv,
)

NS = int(pd.Timestamp("2026-05-26", tz="UTC").value)

CSV_LINES = [
    "ticker,volume,open,close,high,low,window_start,transactions",
    f"NVDA,1000.0,216.54,214.86,218.18,212.00,{NS},2826923",
    f"AAPL,2000.5,150.0,151.0,152.0,149.5,{NS},123456",
]


def _gz(lines):
    return gzip.compress("\n".join(lines).encode())


def test_parse_aggs_csv_gzipped():
    df = parse_aggs_csv(_gz(CSV_LINES))
    assert list(df["ticker"]) == ["NVDA", "AAPL"]
    assert df["close"].iloc[0] == 214.86
    assert df["date"].iloc[0] == date(2026, 5, 26)
    assert "timestamp" in df.columns


def test_parse_aggs_csv_plain_text():
    df = parse_aggs_csv("\n".join(CSV_LINES))
    assert len(df) == 2
    assert df["date"].iloc[1] == date(2026, 5, 26)


def test_key_for():
    c = FlatFilesClient(access_key_id="x", secret_access_key="y")
    assert (
        c.key_for("day_aggs", date(2026, 5, 26))
        == "us_stocks_sip/day_aggs_v1/2026/05/2026-05-26.csv.gz"
    )
    assert (
        c.key_for("minute_aggs", date(2026, 1, 2))
        == "us_stocks_sip/minute_aggs_v1/2026/01/2026-01-02.csv.gz"
    )


def test_day_aggs_to_rows():
    rows = day_aggs_to_rows(parse_aggs_csv(_gz(CSV_LINES)))
    assert rows[0]["symbol"] == "NVDA"
    assert rows[0]["date"] == date(2026, 5, 26)
    assert rows[0]["transactions"] == 2826923
    assert rows[0]["volume"] == 1000.0
    assert rows[0]["source"] == "flatfiles"


def test_read_aggs_symbol_filter(monkeypatch):
    c = FlatFilesClient(access_key_id="x", secret_access_key="y")
    monkeypatch.setattr(c, "download", lambda *a, **k: _gz(CSV_LINES))
    df = c.read_day_aggs(date(2026, 5, 26), symbols=["nvda"])
    assert list(df["ticker"]) == ["NVDA"]
    assert df["date"].iloc[0] == date(2026, 5, 26)
