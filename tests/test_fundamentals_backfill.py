"""Tests for the historical PIT fundamentals backfill (pure logic + writes)."""

from dataclasses import dataclass, field
from datetime import date

import pytest

from edgefinder.data.fundamentals_backfill import (
    backfill,
    build_snapshots,
    effective_date,
)
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import FundamentalsSnapshot


@dataclass
class DP:
    value: float | None


@dataclass
class Income:
    net_income_loss: DP | None = None
    revenues: DP | None = None
    basic_earnings_per_share: DP | None = None


@dataclass
class Balance:
    liabilities: DP | None = None
    equity: DP | None = None
    assets: DP | None = None
    current_assets: DP | None = None
    current_liabilities: DP | None = None


@dataclass
class Financials:
    income_statement: Income = field(default_factory=Income)
    balance_sheet: Balance = field(default_factory=Balance)


@dataclass
class Filing:
    fiscal_period: str
    end_date: str
    filing_date: str | None
    financials: Financials = field(default_factory=Financials)


def _q(end: str, filed: str | None, ni=10.0, rev=100.0, eps=1.0,
       liab=50.0, eq=100.0, assets=200.0, ca=80.0, cl=40.0,
       period="Q1") -> Filing:
    return Filing(
        fiscal_period=period, end_date=end, filing_date=filed,
        financials=Financials(
            income_statement=Income(DP(ni), DP(rev), DP(eps)),
            balance_sheet=Balance(DP(liab), DP(eq), DP(assets), DP(ca), DP(cl))))


def _quarters(n: int, start_year=2019, ni=10.0, ni_step=0.0) -> list[Filing]:
    """n consecutive quarters with filing ~40d after period end."""
    out = []
    ends = []
    y, q = start_year, 1
    for i in range(n):
        end_month = q * 3
        end = date(y, end_month, 28)
        filed = end.replace(day=1) + (date(y, end_month, 28) - date(y, end_month, 1))
        out.append(_q(str(end), str(end + (date(y, 1, 10) - date(y, 1, 1)) * 4),
                      ni=ni + i * ni_step, period=f"Q{q}"))
        ends.append(end)
        q += 1
        if q == 5:
            q, y = 1, y + 1
    return out


class TestEffectiveDate:
    def test_filing_date_wins(self):
        f = _q("2024-03-31", "2024-05-02")
        assert effective_date(f) == date(2024, 5, 2)

    def test_missing_filing_date_imputes_sec_deadline(self):
        q = Filing("Q1", "2024-03-31", None)
        assert effective_date(q) == date(2024, 5, 15)        # +45d
        fy = Filing("FY", "2024-12-31", None)
        assert effective_date(fy) == date(2025, 3, 16)       # +75d


class TestBuildSnapshots:
    def test_ttm_requires_four_quarters(self):
        snaps = build_snapshots("X", _quarters(3), since=date(2019, 1, 1))
        assert len(snaps) == 3
        for _, data in snaps:
            assert data["earnings_per_share"] is None        # <4 quarters
            assert data["debt_to_equity"] == 0.5             # point-in-time, no TTM needed

    def test_ttm_and_growth_math(self):
        # 8 quarters, net income rising 10,11,...,17: TTM at q8 = 14+15+16+17 = 62
        # TTM at q4 (prior-year) = 10+11+12+13 = 46; growth = (62-46)/46
        snaps = build_snapshots("X", _quarters(8, ni=10.0, ni_step=1.0),
                                since=date(2019, 1, 1))
        _, last = snaps[-1]
        assert last["earnings_per_share"] == 4.0             # 4 x eps 1.0
        assert last["earnings_growth"] == pytest.approx((62 - 46) / 46, abs=1e-4)
        assert last["return_on_equity"] == pytest.approx(62 / 100, abs=1e-4)
        assert last["return_on_assets"] == pytest.approx(62 / 200, abs=1e-4)
        assert last["current_ratio"] == 2.0

    def test_no_future_data_in_any_snapshot(self):
        # snapshot i must not change when later filings are appended
        eight = _quarters(8, ni=10.0, ni_step=1.0)
        snaps_8 = build_snapshots("X", eight, since=date(2019, 1, 1))
        snaps_6 = build_snapshots("X", eight[:6], since=date(2019, 1, 1))
        assert snaps_8[:6] == snaps_6

    def test_amended_filings_first_wins(self):
        q1 = _q("2024-03-31", "2024-05-01", ni=10.0)
        amend = _q("2024-03-31", "2024-08-01", ni=99.0)      # same period, later filing
        snaps = build_snapshots("X", [q1, amend], since=date(2019, 1, 1))
        assert len(snaps) == 1                                # one snapshot per period
        assert snaps[0][0] == date(2024, 5, 1)                # the FIRST filing's date

    def test_same_day_catchup_filings_keep_latest_period(self):
        # a delinquent filer files three quarters on one day (seen live: ADMP
        # 2021-11-22) — one snapshot, from the most recent period end
        f1 = _q("2021-03-31", "2021-11-22", ni=1.0, period="Q1")
        f2 = _q("2021-06-30", "2021-11-22", ni=2.0, period="Q2")
        f3 = _q("2021-09-30", "2021-11-22", ni=3.0, period="Q3")
        snaps = build_snapshots("X", [f1, f2, f3], since=date(2019, 1, 1))
        assert len(snaps) == 1
        assert snaps[0][0] == date(2021, 11, 22)
        # the kept snapshot has all three quarters in its trailing window
        # (debt_to_equity comes from the LATEST period's balance sheet)
        assert snaps[0][1]["debt_to_equity"] == 0.5

    def test_since_filters_and_annual_rows_ignored(self):
        fy = Filing("FY", "2023-12-31", "2024-02-15")
        old = _q("2018-03-31", "2018-05-01")
        snaps = build_snapshots("X", [fy, old], since=date(2019, 1, 1))
        assert snaps == []


class TestBackfillWrites:
    def test_idempotent_writes(self):
        engine = get_engine(url="sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = get_session_factory(engine)()

        class FakeVX:
            def list_stock_financials(self, ticker, **kw):
                return iter(_quarters(8, ni=10.0, ni_step=1.0))

        class FakeClient:
            vx = FakeVX()

        r1 = backfill(session, FakeClient(), ["AAA"], since=date(2019, 1, 1))
        assert r1["rows_written"] == 8
        r2 = backfill(session, FakeClient(), ["AAA"], since=date(2019, 1, 1))
        assert r2["rows_written"] == 0                        # skip-existing
        assert session.query(FundamentalsSnapshot).count() == 8
        session.close()
