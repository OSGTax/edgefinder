"""Seed a LOCAL SQLite database with demo data so every dashboard page
renders offline (charts, trades, lab runs, ops) without Polygon or prod.

    python scripts/seed_demo_data.py --db-url sqlite:///data/demo.db
    DATABASE_URL=sqlite:///data/demo.db uvicorn dashboard.app:app --reload

NEVER points at production by default; refuses non-SQLite URLs.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, ".")

from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db.models import (
    DailyBar,
    IndexDaily,
    StrategyAccount,
    StrategySnapshot,
    SystemHeartbeat,
    TickerDividend,
    TickerSplit,
    TradeRecord,
    ValidationRun,
)

SYMBOLS = {"SPY": 450.0, "QQQ": 380.0, "AAPL": 190.0, "NVDA": 480.0, "TSLA": 240.0}


def seed(db_url: str) -> None:
    if not db_url.startswith("sqlite"):
        raise SystemExit("refusing to seed a non-SQLite database")
    engine = get_engine(url=db_url)
    Base.metadata.create_all(engine)
    s = get_session_factory(engine)()
    rng = random.Random(7)

    # ── 2 years of daily bars + index closes ──
    start = date.today() - timedelta(days=730)
    for sym, px in SYMBOLS.items():
        d, price = start, px * 0.7
        while d <= date.today():
            if d.weekday() < 5:
                drift = rng.gauss(0.0004, 0.015)
                o = price
                price = max(1.0, price * (1 + drift))
                hi, lo = max(o, price) * 1.008, min(o, price) * 0.992
                s.add(DailyBar(symbol=sym, date=d, open=o, high=hi, low=lo,
                               close=price, volume=rng.uniform(2e7, 9e7),
                               source="demo"))
                if sym in ("SPY", "QQQ"):
                    s.add(IndexDaily(symbol=sym,
                                     date=datetime.combine(d, datetime.min.time()),
                                     close=price, change_pct=drift * 100))
            d += timedelta(days=1)
    s.add(TickerDividend(symbol="AAPL",
                         ex_dividend_date=str(date.today() - timedelta(days=40)),
                         pay_date=str(date.today() - timedelta(days=26)),
                         cash_amount=0.24))
    s.add(TickerSplit(symbol="NVDA",
                      execution_date=str(date.today() - timedelta(days=300)),
                      split_from=1, split_to=4))

    # ── strategies: accounts + equity snapshots + trades ──
    for i, (strat, cap) in enumerate([("coward", 5000.0), ("gambler", 5000.0),
                                      ("equal_weight", 100000.0)]):
        equity = cap
        for back in range(60, -1, -1):
            ts = datetime.now(timezone.utc) - timedelta(days=back)
            equity *= 1 + rng.gauss(0.0006, 0.008)
            s.add(StrategySnapshot(strategy_name=strat, timestamp=ts,
                                   cash=equity * 0.4, positions_value=equity * 0.6,
                                   total_equity=equity, drawdown_pct=0.0,
                                   total_return_pct=(equity / cap - 1) * 100))
        s.add(StrategyAccount(strategy_name=strat, starting_capital=cap,
                              cash_balance=equity * 0.4,
                              open_positions_value=equity * 0.6,
                              total_equity=equity, peak_equity=equity * 1.02,
                              drawdown_pct=2.0,
                              realized_pnl=equity - cap, is_paused=False))
        for n in range(14):
            sym = rng.choice(list(SYMBOLS))
            entry = SYMBOLS[sym] * rng.uniform(0.85, 1.1)
            pnl = rng.gauss(40, 120)
            opened = datetime.now(timezone.utc) - timedelta(days=rng.randint(2, 55))
            closed = n >= 2
            s.add(TradeRecord(
                trade_id=f"demo-{strat}-{n}", strategy_name=strat, symbol=sym,
                direction="LONG", trade_type="SWING",
                entry_price=round(entry, 2), shares=max(1, int(1000 / entry)),
                stop_loss=round(entry * 0.95, 2), target=round(entry * 1.15, 2),
                confidence=70.0, entry_time=opened,
                status="CLOSED" if closed else "OPEN",
                exit_price=round(entry * (1 + pnl / 1000), 2) if closed else None,
                exit_time=opened + timedelta(days=rng.randint(1, 8)) if closed else None,
                pnl_dollars=round(pnl, 2) if closed else None,
                pnl_percent=round(pnl / 10, 2) if closed else None,
                exit_reason="TARGET" if closed and pnl > 0 else "STOP" if closed else None,
                entry_reasoning=f"demo signal on {sym}",
                exit_reasoning="demo exit" if closed else None,
            ))

    # ── lab: a few validation runs (hunt-style labels) ──
    for n, (name, label, excess, folds, all_met, holdout) in enumerate([
        ("xsec_mom_12_1", "hunt-r1:top500", 9.17, "4/6", False,
         {"window": "2025-12-05..2026-06-09", "passes": None}),
        ("deep_value_pe10", "hunt-r1:top500-pit", 8.71, "4/6", False, None),
        ("buy_and_hold_spy", "hunt-r1:null", -0.02, "0/38", False, None),
    ]):
        s.add(ValidationRun(
            strategy_name=name, universe=label, git_sha="demo0000",
            run_at=datetime.now(timezone.utc) - timedelta(hours=n),
            config={"engine": "v2", "schedule": "monthly", "cost_bps": 2.0,
                    "num_folds": int(folds.split("/")[1]), "costed": True,
                    "prices": "split+dividend-adjusted (total return)",
                    "holdout_window": "2025-12-05..2026-06-09",
                    "rebalance_band": 0.0, "fundamentals": "none"},
            oos={"mean_excess_vs_spy_pct": excess, "folds_beating_spy": folds,
                 "mean_excess_sharpe": -0.5, "folds_higher_sharpe": "1/6",
                 "mean_sharpe": 1.1, "total_trades": 900,
                 "total_return_pct": 120.0, "mean_drawdown_reduction_pct": -10.0},
            criteria={"mode": "risk_adjusted", "all_met": all_met},
            holdout=holdout, verdict="FAIL",
        ))

    # ── ops heartbeats ──
    for comp in ("intraday_cycle", "v2_portfolio_cycle", "nightly_scan"):
        s.add(SystemHeartbeat(component=comp,
                              last_run_at=datetime.now(timezone.utc),
                              ok=True, detail={"demo": True}))

    s.commit()
    s.close()
    print(f"demo data seeded into {db_url}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-url", default="sqlite:///data/demo.db")
    seed(p.parse_args().db_url)
