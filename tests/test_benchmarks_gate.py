"""Trading-day gate on BenchmarkService.collect_daily.

Regression tests for the phantom-row incident: the 4:10 PM collector ran on
Memorial Day 2026-05-25 (a weekday) and stored the prior Friday's prices
under the holiday date. The gate must skip weekends and Polygon-reported
closed days, and fail OPEN (collect anyway) when the holiday check errors.

These live outside test_market.py because that file is excluded from CI
(known pre-existing flake) — a gate test that never runs gates nothing.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from edgefinder.db.models import IndexDaily
from edgefinder.market.benchmarks import BenchmarkService

MONDAY_HOLIDAY = date(2026, 5, 25)   # Memorial Day — the real incident
SATURDAY = date(2026, 5, 23)
TUESDAY = date(2026, 5, 26)


@pytest.fixture
def provider():
    p = MagicMock()
    p.get_latest_price.return_value = 100.0
    p.get_market_holidays.return_value = [
        {"date": "2026-05-25", "status": "closed", "name": "Memorial Day"},
        {"date": "2026-11-27", "status": "early-close", "name": "Thanksgiving"},
    ]
    return p


def _row_count(session) -> int:
    return session.query(IndexDaily).count()


def test_weekend_is_skipped(provider, db_session):
    svc = BenchmarkService(provider, db_session)
    assert svc.collect_daily(as_of=SATURDAY) == 0
    assert _row_count(db_session) == 0
    provider.get_latest_price.assert_not_called()


def test_weekday_holiday_is_skipped(provider, db_session):
    svc = BenchmarkService(provider, db_session)
    assert svc.collect_daily(as_of=MONDAY_HOLIDAY) == 0
    assert _row_count(db_session) == 0
    provider.get_latest_price.assert_not_called()


def test_regular_trading_day_collects(provider, db_session):
    svc = BenchmarkService(provider, db_session)
    assert svc.collect_daily(as_of=TUESDAY) > 0
    assert _row_count(db_session) > 0


def test_early_close_day_still_collects(provider, db_session):
    svc = BenchmarkService(provider, db_session)
    assert svc.collect_daily(as_of=date(2026, 11, 27)) > 0


def test_holiday_check_failure_fails_open(provider, db_session):
    provider.get_market_holidays.side_effect = RuntimeError("api down")
    svc = BenchmarkService(provider, db_session)
    assert svc.collect_daily(as_of=TUESDAY) > 0


def test_provider_without_holiday_support_collects(db_session):
    class BareProvider:
        def get_latest_price(self, symbol):
            return 100.0

    svc = BenchmarkService(BareProvider(), db_session)
    assert svc.collect_daily(as_of=TUESDAY) > 0
