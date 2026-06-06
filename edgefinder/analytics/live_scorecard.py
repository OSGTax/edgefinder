"""Live-vs-SPY scorecard — the offline validation bar applied to LIVE results.

Computes, per strategy and rolling window, the SAME criteria the walk-forward
lab uses (``edgefinder/backtest/walkforward.py`` ``criteria`` block):
Sharpe > 0 AND beats SPY AND >= N trades — but from *live* stored data only:

- daily equity series: last ``strategy_snapshots`` mark per ET trading day
- benchmark: SPY closes from ``index_daily`` (the EOD benchmark collector's
  table — fresher than ``daily_bars`` for a window ending today)
- trade stats: CLOSED rows in ``trades`` with ``exit_time`` in the window

Every number is recomputable from those three tables — that is the point:
the dashboard's "Live Proof" card must be verifiable from stored data, not
from in-memory marks.

Implementation notes (hard-won; see HANDOFF 2026-06-05):
- All relevant DB timestamps are NAIVE UTC. Snapshot marks are grouped into
  ET days by attaching UTC then converting to America/New_York.
- The equity and SPY series are INNER-JOINED on common dates: this drops
  today-before-the-16:10-ET-benchmark-collection, outage gaps, weekends,
  and any day either leg is missing.
- Sharpe mirrors the backtest lab's definition (annualized, daily returns,
  population stdev) via the shared helper so live and offline numbers are
  directly comparable.
- Portable SQL only (no DISTINCT ON / AT TIME ZONE): fetch ordered rows and
  reduce in Python, as the existing endpoints do — tests run on SQLite.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from edgefinder.backtest.daily_backtest import _sharpe  # single Sharpe definition
from edgefinder.db.models import IndexDaily, StrategySnapshot, TradeRecord

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Same bar as the offline lab (walkforward.run_walkforward pass_min_trades).
DEFAULT_PASS_MIN_TRADES = 30


def _naive_utc_cutoff(now: datetime, days: int) -> datetime:
    """Window start as naive UTC (DB columns are timestamp-without-tz)."""
    return (now - timedelta(days=days)).replace(tzinfo=None)


def _daily_equity(session: Session, strategy: str, cutoff: datetime) -> dict:
    """{ET date -> last equity mark of that day} from strategy_snapshots."""
    rows = (
        session.query(StrategySnapshot.timestamp, StrategySnapshot.total_equity)
        .filter(
            StrategySnapshot.strategy_name == strategy,
            StrategySnapshot.timestamp >= cutoff,
        )
        .order_by(StrategySnapshot.timestamp)
        .all()
    )
    by_day: dict = {}
    for ts, equity in rows:
        if ts is None or equity is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        day = ts.astimezone(_ET).date()
        by_day[day] = float(equity)  # ordered ascending → last mark wins
    return by_day


def _spy_closes(session: Session, cutoff: datetime) -> dict:
    """{date -> SPY close} from index_daily (dates stored as midnight-naive)."""
    rows = (
        session.query(IndexDaily.date, IndexDaily.close)
        .filter(IndexDaily.symbol == "SPY", IndexDaily.date >= cutoff)
        .order_by(IndexDaily.date)
        .all()
    )
    out: dict = {}
    for d, close in rows:
        if d is None or not close:
            continue
        out[d.date() if hasattr(d, "date") else d] = float(close)
    return out


def _trade_stats(session: Session, strategy: str, cutoff: datetime) -> dict:
    """Closed-trade stats over the window (exit_time-based, naive UTC)."""
    rows = (
        session.query(TradeRecord.pnl_dollars)
        .filter(
            TradeRecord.strategy_name == strategy,
            TradeRecord.status == "CLOSED",
            TradeRecord.exit_time.is_not(None),
            TradeRecord.exit_time >= cutoff,
        )
        .all()
    )
    pnls = [float(r[0]) for r in rows if r[0] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "closed": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "realized_pnl": round(sum(pnls), 2),
    }


def compute_scorecard(
    session: Session,
    strategy: str,
    *,
    days: int = 90,
    pass_min_trades: int = DEFAULT_PASS_MIN_TRADES,
    now: datetime | None = None,
) -> dict:
    """One strategy's live scorecard over the trailing ``days`` window."""
    now = now or datetime.now(timezone.utc)
    cutoff = _naive_utc_cutoff(now, days)

    equity_by_day = _daily_equity(session, strategy, cutoff)
    spy_by_day = _spy_closes(session, cutoff)

    # Inner join on common ET trading days (weekdays only — a stale holiday
    # quote occasionally lands on both legs and would add a flat no-op day).
    common = sorted(
        d for d in equity_by_day.keys() & spy_by_day.keys() if d.weekday() < 5
    )

    trades = _trade_stats(session, strategy, cutoff)

    base = {
        "strategy_name": strategy,
        "days": days,
        "min_trades_threshold": pass_min_trades,
        "trades": trades["closed"],
        "trade_stats": trades,
    }

    if len(common) < 2:
        return base | {
            "status": "insufficient_data",
            "window": {"start": None, "end": None, "points": len(common)},
            "sharpe": None,
            "return_pct": None,
            "spy_return_pct": None,
            "excess_vs_spy_pct": None,
            "criteria": {
                "sharpe_positive": False,
                "beats_spy": False,
                "min_trades_met": trades["closed"] >= pass_min_trades,
                "all_met": False,
            },
            "verdict": "FAIL",
        }

    equity = [equity_by_day[d] for d in common]
    spy = [spy_by_day[d] for d in common]

    return_pct = (
        round((equity[-1] - equity[0]) / equity[0] * 100, 2) if equity[0] > 0 else None
    )
    spy_return_pct = (
        round((spy[-1] - spy[0]) / spy[0] * 100, 2) if spy[0] > 0 else None
    )
    excess = (
        round(return_pct - spy_return_pct, 2)
        if return_pct is not None and spy_return_pct is not None
        else None
    )
    sharpe = _sharpe(equity)  # annualized, daily returns — lab definition

    sharpe_positive = sharpe is not None and sharpe > 0
    beats_spy = excess is not None and excess > 0
    min_trades_met = trades["closed"] >= pass_min_trades
    all_met = bool(sharpe_positive and beats_spy and min_trades_met)

    return base | {
        "status": "ok",
        "window": {
            "start": common[0].isoformat(),
            "end": common[-1].isoformat(),
            "points": len(common),
        },
        "sharpe": sharpe,
        "return_pct": return_pct,
        "spy_return_pct": spy_return_pct,
        "excess_vs_spy_pct": excess,
        "criteria": {
            "sharpe_positive": sharpe_positive,
            "beats_spy": beats_spy,
            "min_trades_met": min_trades_met,
            "all_met": all_met,
        },
        "verdict": "PASS" if all_met else "FAIL",
    }


def compute_all_scorecards(
    session: Session,
    *,
    days: int = 90,
    pass_min_trades: int = DEFAULT_PASS_MIN_TRADES,
    strategies: list[str] | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """Scorecards for the live strategy set.

    Defaults to the ``live_strategies`` allowlist — the strategies actually
    trading. Research candidates are registered in the StrategyRegistry too,
    but they are lab-only by definition and have no live evidence to score;
    listing them on the Live Proof panel just reads as noise.
    """
    if strategies is None:
        from config.settings import settings

        strategies = sorted(settings.live_strategies)
    return [
        compute_scorecard(
            session, s, days=days, pass_min_trades=pass_min_trades, now=now
        )
        for s in strategies
    ]
