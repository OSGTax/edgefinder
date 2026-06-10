"""dividends_backfill.add_record — manual single-record patches.

Exists for gaps the upstream vendor lacks entirely (Polygon has no TLT
2015-08 dividend at all). Must be idempotent and refuse junk amounts.
"""

from datetime import date

import pytest

from edgefinder.data.dividends_backfill import add_record
from edgefinder.db.models import DividendRecord

EX = date(2015, 8, 3)


def test_add_inserts_row(db_session):
    out = add_record(db_session, "TLT", EX, 0.267)
    assert out["status"] == "added"
    row = db_session.query(DividendRecord).filter_by(symbol="TLT", ex_date=EX).one()
    assert row.cash_amount == 0.267


def test_add_is_idempotent_and_never_overwrites(db_session):
    add_record(db_session, "TLT", EX, 0.267)
    out = add_record(db_session, "TLT", EX, 9.99)   # wrong amount, must not stick
    assert out["status"] == "exists"
    assert out["cash_amount"] == 0.267
    row = db_session.query(DividendRecord).filter_by(symbol="TLT", ex_date=EX).one()
    assert row.cash_amount == 0.267


def test_add_rejects_nonpositive_amount(db_session):
    with pytest.raises(ValueError):
        add_record(db_session, "TLT", EX, 0.0)
    with pytest.raises(ValueError):
        add_record(db_session, "TLT", EX, -1.0)
