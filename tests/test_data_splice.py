"""The auto-source seam must prefer the DB's fresh tail over a stale R2 archive."""

from __future__ import annotations

from datetime import date

import pandas as pd

from agent.data import _splice_db_tail


def _frame(rows):
    return pd.DataFrame(
        [{"date": date(2026, 6, d), "open": c, "high": c, "low": c,
          "close": c, "volume": 1_000} for d, c in rows])


def test_db_tail_extends_stale_r2():
    r2 = {"MU": _frame([(16, 1134.0), (17, 1120.0), (18, 1130.0)])}   # stale @ Jun 18
    db = {"MU": _frame([(18, 1130.0), (19, 980.0), (20, 938.0)])}      # fresh tail
    out = _splice_db_tail(r2, db)["MU"]
    assert list(out["date"])[-1] == date(2026, 6, 20)                 # newest wins
    assert float(out.iloc[-1]["close"]) == 938.0
    assert len(out) == 5                                              # union, Jun 18 deduped


def test_db_wins_on_shared_date():
    r2 = {"X": _frame([(1, 10.0), (2, 11.0)])}
    db = {"X": _frame([(2, 99.0)])}                                   # correction on Jun 2
    out = _splice_db_tail(r2, db)["X"]
    assert float(out[out["date"] == date(2026, 6, 2)]["close"].iloc[0]) == 99.0


def test_db_only_symbol_passes_through():
    out = _splice_db_tail({}, {"NEW": _frame([(1, 5.0)])})
    assert "NEW" in out and len(out["NEW"]) == 1


def test_empty_db_leaves_r2_untouched():
    r2 = {"A": _frame([(1, 1.0)])}
    out = _splice_db_tail(r2, {})
    assert list(out["A"]["close"]) == [1.0]
