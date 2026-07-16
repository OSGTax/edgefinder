"""F5: equity corporate actions on the live book — settle() folds splits and
dividends into open positions via 0-price ledger adjustment rows.

Conventions under test:
- A split changes the UNIT COUNT, never the basis: shares move by
  held × (to/from − 1) at price 0 / dollars 0, cost is untouched, so
  avg_price rebases by the ratio and market value / unrealized P&L are
  unchanged when the quote rebases. No realized P&L is fabricated.
- A dividend credits shares-held-going-INTO-the-ex-date × cash_amount as a
  SELL of 0 shares with dollars=credit (ex-date basis; pay-date cash is the
  documented simplification).
- Idempotent: every adjustment row carries its (src, date) key in
  fill_quote; a second settle() books nothing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

TODAY = date.today()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'corp.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401 — ticker_splits / dividends
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def _buy(store, symbol, shares, price, days_ago=5, run_id="T"):
    """Backdated BUY so corp actions dated after it sit inside the lookback
    (settle bounds the scan to the current lot's earliest open fill)."""
    from agent import ledger
    store.insert("desk_trades", {
        "account": "agent", "run_id": run_id, "symbol": symbol, "side": "BUY",
        "shares": shares, "price": price, "dollars": round(shares * price, 2),
        "ts": datetime.utcnow() - timedelta(days=days_ago)}, returning=False)
    ledger.rebuild_positions(store)


def _split(store, symbol, days_ago, frm, to):
    store.insert("ticker_splits", {
        "symbol": symbol, "execution_date": str(TODAY - timedelta(days=days_ago)),
        "split_from": frm, "split_to": to}, returning=False)


def test_forward_split_rebases_without_pnl(store):
    """10:1 forward split: 10 sh @ $500 become 100 sh @ $50 avg — market
    value and unrealized P&L unchanged when the quote drops 10x."""
    from agent import ledger
    _buy(store, "NVDA", 10, 500.0, days_ago=5)         # $5,000 basis
    _split(store, "NVDA", 2, frm=1, to=10)
    out = ledger.settle(store)
    assert out["corp_actions"]["splits"] == 1
    pos = store.select("desk_positions", filters={"symbol": "NVDA"})[0]
    assert pos["shares"] == pytest.approx(100.0)
    assert pos["avg_price"] == pytest.approx(50.0)     # 100 × 50 = the same $5,000
    assert ledger.cash(store) == 100_000.0 - 5_000.0   # a split moves no cash
    st = ledger.mark(store, prices={"NVDA": 50.0})     # quote rebased 10x down
    row = next(p for p in st["positions"] if p["symbol"] == "NVDA")
    assert row["market_value"] == pytest.approx(5_000.0)
    assert row["unrealized_pnl"] == pytest.approx(0.0)
    _, by_symbol = ledger._realized_pnl(ledger._trades(store, "agent"))
    assert by_symbol.get("NVDA", 0.0) == pytest.approx(0.0)  # no phantom loss
    adj = store.select("desk_trades", filters={"symbol": "NVDA",
                                               "run_id": "settlement"})[0]
    assert adj["side"] == "BUY" and adj["price"] == 0.0 and adj["dollars"] == 0.0
    assert adj["fill_quote"]["src"] == "split_adjustment"
    assert adj["fill_quote"]["ratio"] == "10:1"


def test_reverse_split_rebases_without_pnl(store):
    """1:10 reverse split: 100 sh @ $5 become 10 sh @ $50 avg — MV and
    unrealized unchanged when the quote rises 10x, no fake realized loss."""
    from agent import ledger
    _buy(store, "XYZ", 100, 5.0, days_ago=5)           # $500 basis
    _split(store, "XYZ", 2, frm=10, to=1)
    out = ledger.settle(store)
    assert out["corp_actions"]["splits"] == 1
    pos = store.select("desk_positions", filters={"symbol": "XYZ"})[0]
    assert pos["shares"] == pytest.approx(10.0)
    assert pos["avg_price"] == pytest.approx(50.0)
    st = ledger.mark(store, prices={"XYZ": 50.0})      # quote rebased 10x up
    row = next(p for p in st["positions"] if p["symbol"] == "XYZ")
    assert row["market_value"] == pytest.approx(500.0)
    assert row["unrealized_pnl"] == pytest.approx(0.0)
    _, by_symbol = ledger._realized_pnl(ledger._trades(store, "agent"))
    assert by_symbol.get("XYZ", 0.0) == pytest.approx(0.0)
    adj = store.select("desk_trades", filters={"symbol": "XYZ",
                                               "run_id": "settlement"})[0]
    assert adj["side"] == "SELL" and adj["dollars"] == 0.0
    assert adj["shares"] == pytest.approx(90.0)


def test_dividend_pays_on_shares_held_asof_ex_date(store):
    """Credit = shares held going INTO the ex-date × cash_amount — a trim
    AFTER the ex-date doesn't shrink the entitlement, and the credit is
    pure cash (no share movement, avg untouched)."""
    from agent import ledger
    _buy(store, "AAPL", 100, 200.0, days_ago=10)
    store.insert("desk_trades", {                       # trimmed AFTER ex-date
        "account": "agent", "run_id": "T2", "symbol": "AAPL", "side": "SELL",
        "shares": 40, "price": 200.0, "dollars": 8_000.0,
        "ts": datetime.utcnow() - timedelta(days=2)}, returning=False)
    ledger.rebuild_positions(store)
    store.insert("dividends", {"symbol": "AAPL",
                               "ex_date": TODAY - timedelta(days=5),
                               "cash_amount": 0.26}, returning=False)
    out = ledger.settle(store)
    corp = out["corp_actions"]
    assert corp["dividends"] == 1
    assert corp["details"][0]["credit"] == pytest.approx(26.0)  # 100 held at ex
    assert ledger.cash(store) == pytest.approx(
        100_000.0 - 20_000.0 + 8_000.0 + 26.0)
    pos = store.select("desk_positions", filters={"symbol": "AAPL"})[0]
    assert pos["shares"] == pytest.approx(60.0)         # dividend moved no shares
    assert pos["avg_price"] == pytest.approx(200.0)
    adj = store.select("desk_trades", filters={"symbol": "AAPL",
                                               "run_id": "settlement"})[0]
    assert adj["fill_quote"]["src"] == "dividend"
    assert adj["fill_quote"]["shares_asof"] == pytest.approx(100.0)


def test_dividend_after_split_pays_on_post_split_shares(store):
    """A split and a later ex-date processed in ONE settle run: the replay
    dates the split adjustment by its execution date, so the dividend pays
    on the post-split share count."""
    from agent import ledger
    _buy(store, "NVDA", 10, 500.0, days_ago=10)
    _split(store, "NVDA", 6, frm=1, to=10)              # → 100 shares
    store.insert("dividends", {"symbol": "NVDA",
                               "ex_date": TODAY - timedelta(days=2),
                               "cash_amount": 0.10}, returning=False)
    out = ledger.settle(store)
    corp = out["corp_actions"]
    assert corp["splits"] == 1 and corp["dividends"] == 1
    div = next(d for d in corp["details"] if d["action"] == "dividend")
    assert div["credit"] == pytest.approx(10.0)         # 100 × $0.10, not 10 × $0.10
    assert ledger.cash(store) == pytest.approx(100_000.0 - 5_000.0 + 10.0)


def test_settle_corp_actions_idempotent(store):
    """Re-running settle() books NOTHING the second time — every adjustment
    row's (src, date) key is read back and skipped."""
    from agent import ledger
    _buy(store, "NVDA", 10, 500.0, days_ago=5)
    _split(store, "NVDA", 2, frm=1, to=10)
    store.insert("dividends", {"symbol": "NVDA",
                               "ex_date": TODAY - timedelta(days=3),
                               "cash_amount": 0.01}, returning=False)
    first = ledger.settle(store)
    assert first["corp_actions"]["splits"] == 1
    assert first["corp_actions"]["dividends"] == 1
    n_trades = len(store.select("desk_trades"))
    cash_after = ledger.cash(store)
    second = ledger.settle(store)
    assert second["corp_actions"] == {"splits": 0, "dividends": 0, "details": []}
    assert len(store.select("desk_trades")) == n_trades
    assert ledger.cash(store) == cash_after
    pos = store.select("desk_positions", filters={"symbol": "NVDA"})[0]
    assert pos["shares"] == pytest.approx(100.0)        # applied exactly once


def test_corp_actions_skip_crypto_and_closed_positions(store):
    """Crypto pairs never have corp actions; a closed lot gets nothing even
    if a row exists in its window."""
    from agent import ledger
    _buy(store, "BTC/USD", 0.5, 60_000.0, days_ago=5)
    _split(store, "BTC/USD", 2, frm=1, to=10)           # nonsense row — ignored
    # an equity bought and fully sold before settle runs
    _buy(store, "OLD", 10, 10.0, days_ago=8)
    store.insert("desk_trades", {
        "account": "agent", "run_id": "T2", "symbol": "OLD", "side": "SELL",
        "shares": 10, "price": 10.0, "dollars": 100.0,
        "ts": datetime.utcnow() - timedelta(days=4)}, returning=False)
    ledger.rebuild_positions(store)
    _split(store, "OLD", 2, frm=1, to=2)
    out = ledger.settle(store)
    assert out["corp_actions"] == {"splits": 0, "dividends": 0, "details": []}
