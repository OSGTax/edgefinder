"""Tests for the intraday pilot menu resolver (seeded daily bars, no network)."""

import json
from datetime import date, timedelta

from edgefinder.data.barstore import DB_PROTECTED_ETFS
from edgefinder.db.models import DailyBar

from scripts.resolve_intraday_menu import resolve_menu

AS_OF = date(2024, 6, 28)


def seed(session):
    """SPY calendar + ranked stocks: HOTT > WARM > COLD; DEAD delisted."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(180)]
    days = [d for d in days if d.weekday() < 5 and d <= AS_OF]
    spec = {"SPY": 50.0, "HOTT": 1000.0, "WARM": 500.0, "COLD": 10.0,
            "DEAD": 9999.0}
    for sym, dollar_vol_m in spec.items():
        for d in days:
            if sym == "DEAD" and d > date(2024, 3, 1):
                continue          # delisted — fails the 30-day alive gate
            session.add(DailyBar(symbol=sym, date=d, open=100.0, high=101.0,
                                 low=99.0, close=100.0,
                                 volume=dollar_vol_m * 10_000, source="test"))
    session.commit()


class TestResolveMenu:
    def test_top_n_plus_etfs_deterministic(self, db_session):
        seed(db_session)
        menu = resolve_menu(db_session, AS_OF, top_n=2, rank_window=126)
        # top-2 by trailing dollar volume (DEAD excluded by the alive gate,
        # even though its lifetime dollar volume would rank first)
        expected = sorted({"HOTT", "WARM"}
                          | {e.upper() for e in DB_PROTECTED_ETFS})
        assert menu["symbols"] == expected
        assert "DEAD" not in menu["symbols"]
        assert menu["as_of"] == str(AS_OF)
        assert menu["top_n"] == 2 and menu["rank_window"] == 126
        assert "resolve_universe semantics" in menu["criteria"]
        # deterministic: same inputs, same list
        again = resolve_menu(db_session, AS_OF, top_n=2, rank_window=126)
        assert again["symbols"] == menu["symbols"]

    def test_etfs_dedupe_against_ranked_names(self, db_session):
        seed(db_session)
        # SPY ranks inside the top-4 AND is a protected ETF — appears once
        menu = resolve_menu(db_session, AS_OF, top_n=4, rank_window=126)
        assert menu["symbols"].count("SPY") == 1

    def test_menu_json_is_valid_and_frozen(self):
        # the committed file is the pre-registration: criteria fixed; the
        # symbols list was frozen by the first real MENU-mode run (PHASE 1
        # complete, 2026-06-12 — 52 symbols incl. the protected ETFs).
        with open("intraday/menu.json") as f:
            menu = json.load(f)
        assert isinstance(menu["symbols"], list) and len(menu["symbols"]) > 0
        # the protected ETF menu is always present in the frozen list
        for etf in DB_PROTECTED_ETFS:
            assert etf.upper() in menu["symbols"]
        assert menu["frozen_at"] == "2026-06-12"
        assert "trailing-126-trading-day" in menu["criteria"]
        assert menu["etfs"] == list(DB_PROTECTED_ETFS)
