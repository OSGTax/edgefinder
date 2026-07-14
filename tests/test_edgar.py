"""SEC EDGAR fundamentals pipeline: tag waterfalls, Q4/TTM math, per-filing
PIT rows (restatement honesty), price-at-decision ratios, the store-based
PITFundamentals loader, idempotent ingest, and the validation harness.

The invariant under test everywhere: a row keyed to filed-date F contains
ONLY knowledge filed on or before F — later restatements are invisible to F.
"""

from __future__ import annotations

from datetime import date

import pytest


def _fact(end, val, filed, start=None, form="10-Q"):
    d = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        d["start"] = start
    return d


def make_companyfacts():
    """Two fiscal years of a Dec-FYE filer, incl. ONE RESTATEMENT: Q3-2020
    revenue originally filed 2020-10-30 as 12.0, restated in the 10-K filed
    2021-02-25 as 13.0."""
    rev = [
        _fact("2019-06-29", 9.0, "2019-07-30", "2019-03-31"),
        _fact("2019-09-28", 9.5, "2019-10-30", "2019-06-30"),
        _fact("2019-12-31", 38.0, "2020-02-25", "2019-01-01", form="10-K"),
        _fact("2019-03-30", 9.0, "2019-04-30", "2019-01-01"),
        _fact("2020-03-28", 10.0, "2020-04-30", "2020-01-01"),
        _fact("2020-06-27", 11.0, "2020-07-30", "2020-03-29"),
        _fact("2020-09-26", 12.0, "2020-10-30", "2020-06-28"),
        _fact("2020-09-26", 13.0, "2021-02-25", "2020-06-28", form="10-K"),
        _fact("2020-12-31", 50.0, "2021-02-25", "2020-01-01", form="10-K"),
    ]
    ni = [
        _fact("2019-03-30", 2.0, "2019-04-30", "2019-01-01"),
        _fact("2019-06-29", 2.0, "2019-07-30", "2019-03-31"),
        _fact("2019-09-28", 2.0, "2019-10-30", "2019-06-30"),
        _fact("2019-12-31", 8.0, "2020-02-25", "2019-01-01", form="10-K"),
        _fact("2020-03-28", 2.5, "2020-04-30", "2020-01-01"),
        _fact("2020-06-27", 2.5, "2020-07-30", "2020-03-29"),
        _fact("2020-09-26", 2.5, "2020-10-30", "2020-06-28"),
        _fact("2020-12-31", 10.0, "2021-02-25", "2020-01-01", form="10-K"),
    ]
    assets = [_fact("2020-09-26", 100.0, "2020-10-30"),
              _fact("2020-12-31", 110.0, "2021-02-25", form="10-K")]
    equity = [_fact("2020-09-26", 40.0, "2020-10-30"),
              _fact("2020-12-31", 44.0, "2021-02-25", form="10-K")]
    ca = [_fact("2020-09-26", 30.0, "2020-10-30")]
    cl = [_fact("2020-09-26", 15.0, "2020-10-30")]
    inv = [_fact("2020-09-26", 6.0, "2020-10-30")]
    debt = [_fact("2020-09-26", 20.0, "2020-10-30")]
    return {
        "cik": 12345, "entityName": "Testco",
        "facts": {
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
                _fact("2020-09-26", 10.0, "2020-10-30"),
            ]}}},
            "us-gaap": {
                # revenue arrives under a DEPRECATED tag to prove the waterfall
                "SalesRevenueNet": {"units": {"USD": rev}},
                "NetIncomeLoss": {"units": {"USD": ni}},
                "Assets": {"units": {"USD": assets}},
                "StockholdersEquity": {"units": {"USD": equity}},
                "AssetsCurrent": {"units": {"USD": ca}},
                "LiabilitiesCurrent": {"units": {"USD": cl}},
                "InventoryNet": {"units": {"USD": inv}},
                "LongTermDebtNoncurrent": {"units": {"USD": debt}},
            },
        },
    }


# ── pure normalization ──


def test_pit_rows_restatement_honesty():
    from agent.edgar import pit_rows

    rows = pit_rows("TST", 12345, make_companyfacts())
    by_filed = {str(r["filed"]): r for r in rows}

    # As of the ORIGINAL Q3 filing (2020-10-30): Q3 revenue is 12.0 (the
    # restatement is filed months later and must be invisible here), and
    # Q4-2019 is derived from the FY-2019 10-K: 38 - (9 + 9 + 9.5) = 10.5.
    # TTM at 2020-10-30 = Q4'19(10.5) + Q1'20(10) + Q2'20(11) + Q3'20(12).
    oct_row = by_filed["2020-10-30"]
    assert oct_row["data"]["_revenue_ttm"] == pytest.approx(43.5)

    # As of the 10-K (2021-02-25): Q3 is RESTATED to 13.0; Q4'20 derived from
    # FY50 minus restated quarters: 50 - (10+11+13) = 16.
    feb_row = by_filed["2021-02-25"]
    # TTM = Q1'20..Q4'20 with restated Q3 = 10+11+13+16 = 50 (the FY, exactly)
    assert feb_row["data"]["_revenue_ttm"] == pytest.approx(50.0)

    # Balance-sheet ratios at the October filing:
    d = oct_row["data"]
    assert d["current_ratio"] == pytest.approx(2.0)          # 30/15
    assert d["quick_ratio"] == pytest.approx(1.6)            # (30-6)/15
    assert d["debt_to_equity"] == pytest.approx(0.5)         # 20/40
    assert d["_shares"] == 10.0
    # EPS-TTM = TTM net income / shares = (8-2-2-2 + 2.5*3) / 10 = 9.5/10
    assert d["earnings_per_share"] == pytest.approx(0.95)


def test_tag_waterfall_prefers_modern_tag():
    from agent.edgar import _collect

    facts = {"us-gaap": {
        "Revenues": {"units": {"USD": [_fact("2020-03-28", 99.0, "2020-04-30",
                                             "2020-01-01")]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            {"units": {"USD": [_fact("2020-03-28", 100.0, "2020-04-30",
                                     "2020-01-01")]}},
    }}
    got = _collect(facts, "revenue")
    assert got[0]["val"] == 100.0  # the modern tag outranks Revenues


def test_price_ratios_at_decision_time():
    from agent.edgar import price_ratios

    data = {"_shares": 10.0, "_net_income_ttm": 9.5, "_revenue_ttm": 43.5,
            "_book_equity": 40.0, "_fcf_ttm": 5.0, "_ebitda_ttm": 14.0,
            "_total_debt": 20.0, "_cash": 8.0}
    r = price_ratios(data, price=19.0)
    assert r["market_cap"] == pytest.approx(190.0)
    assert r["price_to_earnings"] == pytest.approx(20.0)     # 190/9.5
    assert r["price_to_book"] == pytest.approx(4.75)         # 190/40
    assert r["enterprise_value"] == pytest.approx(202.0)     # 190+20-8
    assert r["ev_to_ebitda"] == pytest.approx(202.0 / 14.0)
    assert price_ratios(data, price=None) == {}
    neg = price_ratios({**data, "_net_income_ttm": -1.0}, price=19.0)
    assert neg["price_to_earnings"] is None  # no P/E on losses, not a negative


# ── loader + ingest on sqlite ──


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'edgar.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def _patch_edgar_network(monkeypatch, cfs: dict):
    import agent.edgar as edgar

    monkeypatch.setattr(edgar, "cik_map",
                        lambda: {sym: cf["cik"] for sym, cf in cfs.items()})
    monkeypatch.setattr(edgar, "company_facts",
                        lambda cik: next(cf for cf in cfs.values()
                                         if cf["cik"] == cik))


def test_ingest_writes_and_is_idempotent(store, monkeypatch):
    from agent.edgar import ingest

    _patch_edgar_network(monkeypatch, {"TST": make_companyfacts()})
    s1 = ingest(store, symbols=["TST"])
    assert s1["rows_inserted"] > 0 and s1["errors"] == 0
    s2 = ingest(store, symbols=["TST"])
    assert s2["rows_inserted"] == 0  # rerun inserts nothing new
    rows = store.select("fundamentals_pit", filters={"symbol": "TST"})
    assert {str(r["filed"])[:10] for r in rows} >= {"2020-10-30", "2021-02-25"}


def test_pit_loader_semantics(store, monkeypatch):
    from agent.edgar import ingest
    from edgefinder.data.pit_fundamentals import PITFundamentals

    _patch_edgar_network(monkeypatch, {"TST": make_companyfacts()})
    ingest(store, symbols=["TST"])
    pit = PITFundamentals(store)
    pit.preload(["TST"])

    assert pit.asof("TST", date(2019, 1, 1)) is None      # before coverage
    oct_view = pit.asof("TST", date(2020, 12, 1))          # after Q3 filing
    feb_view = pit.asof("TST", date(2021, 6, 1))           # after the 10-K
    assert oct_view is not None and feb_view is not None
    # Restatement honesty via the raw ingredients:
    assert pit.raw_asof("TST", date(2020, 12, 1))["_revenue_ttm"] == pytest.approx(43.5)
    assert pit.raw_asof("TST", date(2021, 6, 1))["_revenue_ttm"] == pytest.approx(50.0)
    # Price fields are None on hydrated models — never guessed.
    assert oct_view.price_to_earnings is None
    assert oct_view.market_cap is None
    assert oct_view.current_ratio == pytest.approx(2.0)


# ── validation harness ──


def test_validate_agreement_stats(store, monkeypatch):
    from agent.edgar import ingest, validate

    _patch_edgar_network(monkeypatch, {"TST": make_companyfacts()})
    ingest(store, symbols=["TST"])
    # Frozen vendor row shortly after the 10-K: EPS agrees (within 10%),
    # current_ratio disagrees wildly (frozen says 4.0, ours 2.0).
    store.insert("fundamentals_snapshots", {
        "symbol": "TST", "as_of": date(2021, 3, 1),
        "data": {"symbol": "TST", "earnings_per_share": 1.0,
                 "current_ratio": 4.0}}, returning=False)
    store.insert("daily_bars", {
        "symbol": "TST", "date": date(2021, 2, 26), "open": 19, "high": 19,
        "low": 19, "close": 19.0, "volume": 1e6, "source": "test"},
        returning=False)

    rep = validate(store, max_symbols=10)
    eps = rep["metrics"]["earnings_per_share"]
    cr = rep["metrics"]["current_ratio"]
    assert eps["n"] == 1 and eps["agree_share"] == 1.0   # 1.0 vs ours 1.0
    assert cr["n"] == 1 and cr["agree_share"] == 0.0
    assert cr["worst"][0]["symbol"] == "TST"
    assert rep["verdict"] == "FAIL"  # n>=20 gate unmet + disagreement