"""Self-running paper trading for portfolio (engine-v2) strategies.

The live counterpart of engine/backtest.py with the SAME decision semantics:
the rebalance context is built by the same code path (prepare_bars +
_build_context) from data through the last completed trading day, and fills
happen at today's price. One isolated paper account per promoted strategy,
persisted to the same strategy_accounts/trades tables the dashboard reads
(trades flow through TradeJournal, so the hash chain stays intact).

Deliberate deviations from the old per-ticker arena's account rules:
- No per-trade risk caps, max-position caps, PDT, or cooldowns — a portfolio
  strategy's target weights ARE its risk policy (long-only, sum <= 1,
  enforced here). The old caps encode a per-ticker risk model and would
  block e.g. a 7-position equal-weight at ~14% per name.
- The account integrity formula is unchanged (CLAUDE.md):
  cash = starting_capital + sum(closed pnl) - sum(open cost basis),
  recomputed from the trades table at the start of every cycle (self-heal).
- A partial rebalance sell SPLITS a lot: the sold shares close as a normal
  closed trade and the remainder reopens with the ORIGINAL entry price/time
  (entry_reasoning notes the split), so realized P&L stays exact and every
  row remains an honest open->close unit.

Cadence: the runner executes once per trading day (scheduled shortly after
the open). A strategy trades on its schedule's boundary days (month/week
change vs the previous trading day — the engine's _is_rebalance semantics)
plus its very first cycle (mirroring the engine's forced first-bar
rebalance). Re-true deltas smaller than REBALANCE_BAND of equity are
skipped (dust/churn guard) unless they open or fully close a position.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from edgefinder.core.models import Direction, Trade, TradeStatus, TradeType
from edgefinder.db.models import DailyBar, PromotedStrategy, StrategyAccount, SystemHeartbeat
from edgefinder.engine.backtest import _build_context, prepare_bars
from edgefinder.engine.data import load_bars
from edgefinder.engine.strategies import make_strategy_factory
from edgefinder.trading.journal import TradeJournal

logger = logging.getLogger(__name__)

# Portfolio accounts start at $100k, not the old arena's $5k: a $5k book
# cannot hold even ONE share of each name in a 7-ETF equal weight at 2026
# prices (SPY ~$739 > the $714 sleeve), which would leave ~29% permanently
# uninvested and distort every result. Comparability holds WITHIN the v2
# tier (every v2 account starts equal); the dashboard shows per-strategy
# accounts, so the two tiers are never summed.
STARTING_CAPITAL = 100_000.0
REBALANCE_BAND = 0.01          # skip re-trues smaller than 1% of equity
SLIPPAGE_BPS = 5.0             # paper-fill slippage, both sides
HISTORY_DAYS = 450             # calendar days of bars to load (>=210 trading)
HEARTBEAT = "v2_portfolio_cycle"


def _memoized(price_fn):
    cache: dict[str, float | None] = {}

    def lookup(symbol: str):
        if symbol not in cache:
            cache[symbol] = price_fn(symbol)
        return cache[symbol]
    return lookup


def _is_rebalance_day(today: date, prev_trading_day: date, schedule: str) -> bool:
    """The engine's _is_rebalance semantics, keyed to real calendar days."""
    if schedule == "daily":
        return True
    if schedule == "weekly":
        return (today.isocalendar()[1] != prev_trading_day.isocalendar()[1]
                or today.year != prev_trading_day.year)
    if schedule == "monthly":
        return today.month != prev_trading_day.month or today.year != prev_trading_day.year
    raise ValueError(f"unknown schedule {schedule!r}")


def _record_heartbeat(session_factory, ok: bool, detail: dict) -> None:
    """Best-effort heartbeat upsert on its own session (never breaks a cycle)."""
    try:
        session = session_factory()
        try:
            hb = (session.query(SystemHeartbeat)
                  .filter(SystemHeartbeat.component == HEARTBEAT).one_or_none())
            now = datetime.now(timezone.utc)
            if hb is None:
                session.add(SystemHeartbeat(
                    component=HEARTBEAT, last_run_at=now, ok=ok, detail=detail))
            else:
                hb.last_run_at, hb.ok, hb.detail = now, ok, detail
            session.commit()
        finally:
            session.close()
    except Exception:
        logger.exception("v2 heartbeat upsert failed")


def ensure_recent_bars(session, provider, symbols: list[str], today: date) -> int:
    """Append any missing recent daily bars for ``symbols`` to daily_bars.

    This is also how the permanent data asset grows for traded universes —
    nothing else currently appends daily_bars in production. Best-effort:
    a fetch failure leaves the existing history in place.
    """
    if provider is None:
        return 0
    added = 0
    for sym in symbols:
        try:
            latest = (session.query(DailyBar.date).filter(DailyBar.symbol == sym)
                      .order_by(DailyBar.date.desc()).first())
            latest_date = latest[0] if latest else None
            if latest_date is not None and latest_date >= today - timedelta(days=1):
                continue
            start = (latest_date + timedelta(days=1)) if latest_date else today - timedelta(days=HISTORY_DAYS)
            fetch = getattr(provider, "get_bars_fresh", provider.get_bars)
            df = fetch(sym, "day", start, today)
            if df is None or not len(df):
                continue
            for ts, row in df.iterrows():   # timestamp is the frame's index
                dt = ts.date() if hasattr(ts, "date") else ts
                if dt >= today:        # never ingest a partial in-progress bar
                    continue
                if latest_date is not None and dt <= latest_date:
                    continue
                session.add(DailyBar(
                    symbol=sym, date=dt, open=float(row["open"]),
                    high=float(row["high"]), low=float(row["low"]),
                    close=float(row["close"]), volume=float(row["volume"]),
                    source="polygon_aggs"))
                added += 1
        except Exception:
            logger.exception("bar refresh failed for %s", sym)
    if added:
        session.commit()
    return added


def _open_lots(journal: TradeJournal, strategy_name: str) -> list:
    return journal.get_open_trades(strategy_name)


def _recalc_cash(journal: TradeJournal, strategy_name: str) -> float:
    """CLAUDE.md integrity formula, recomputed from the trades table."""
    closed = journal.get_closed_trades(strategy_name)
    open_lots = _open_lots(journal, strategy_name)
    realized = sum(t.pnl_dollars or 0.0 for t in closed)
    open_cost = sum(t.entry_price * t.shares for t in open_lots)
    return STARTING_CAPITAL + realized - open_cost


def _close_lot(journal: TradeJournal, lot, shares: int, price: float,
               now: datetime, reason: str) -> float:
    """Close ``shares`` of an open lot (splitting it if partial).

    Returns realized P&L. All writes go through the journal: the lot row
    closes for the sold shares; any remainder reopens as a fresh OPEN row
    carrying the ORIGINAL entry price/time, so realized P&L stays exact and
    every row remains an honest open->close unit.
    """
    remainder = lot.shares - shares
    pnl = (price - lot.entry_price) * shares
    pnl_pct = (price / lot.entry_price - 1) * 100 if lot.entry_price else 0.0

    if remainder > 0:
        split = Trade(
            trade_id=str(uuid.uuid4()), strategy_name=lot.strategy_name,
            symbol=lot.symbol, direction=Direction.LONG, trade_type=TradeType.SWING,
            entry_price=lot.entry_price, shares=remainder,
            stop_loss=0.0, target=0.0, confidence=1.0,
            entry_time=lot.entry_time,
            entry_reasoning=f"rebalance split of {lot.trade_id} ({shares} sold)",
        )
        journal.log_trade(split, commit=False)

    close = Trade(
        trade_id=lot.trade_id, strategy_name=lot.strategy_name,
        symbol=lot.symbol, direction=Direction.LONG, trade_type=TradeType.SWING,
        entry_price=lot.entry_price, shares=shares,
        stop_loss=0.0, target=0.0, confidence=1.0,
        entry_time=lot.entry_time, status=TradeStatus.CLOSED,
        exit_price=round(price, 4), exit_time=now,
        pnl_dollars=round(pnl, 2), pnl_percent=round(pnl_pct, 2),
        exit_reason=reason,
    )
    journal.log_trade(close, commit=False)
    return pnl


def run_portfolio_cycle(
    session_factory,
    *,
    provider=None,
    today: date | None = None,
    price_fn=None,
    dry_run: bool = False,
) -> dict:
    """Run one daily cycle over every active promoted strategy.

    ``price_fn(symbol) -> float | None`` supplies fill prices (defaults to
    the provider's latest price, memoized per cycle; tests inject a stub).
    ``dry_run`` computes and reports intended trades without persisting
    anything (its trade list is approximate when sells fund buys).
    """
    # the trading day is an ET concept — a UTC date would roll over at 8 PM
    # ET and an evening manual run would trade "tomorrow"
    today = today or datetime.now(ET).date()
    summary: dict = {"date": str(today), "strategies": {}, "dry_run": dry_run}

    session = session_factory()
    try:
        promos = (session.query(PromotedStrategy)
                  .filter(PromotedStrategy.active.is_(True)).all())
        if not promos:
            summary["note"] = "no active promoted strategies"
            _record_heartbeat(session_factory, True, {"skip": "none promoted",
                                                      "date": str(today)})
            return summary

        if price_fn is None:
            if provider is None:
                raise ValueError("need a provider or an explicit price_fn")
            price_fn = provider.get_latest_price
        price_fn = _memoized(price_fn)   # one quote per symbol per cycle

        ok = True
        for promo in promos:
            try:
                summary["strategies"][promo.strategy_name] = _run_one(
                    session, promo, provider=provider, today=today,
                    price_fn=price_fn, dry_run=dry_run)
            except Exception as e:
                ok = False
                logger.exception("v2 cycle failed for %s", promo.strategy_name)
                summary["strategies"][promo.strategy_name] = {"error": str(e)}
        _record_heartbeat(session_factory, ok, {
            "date": str(today), "dry_run": dry_run,
            "strategies": {k: v.get("action", "error")
                           for k, v in summary["strategies"].items()}})
        return summary
    finally:
        session.close()


def _run_one(session, promo, *, provider, today, price_fn, dry_run) -> dict:
    symbols = list(promo.symbols or [])
    factory = make_strategy_factory(promo.spec)
    strategy = factory()

    ensure_recent_bars(session, provider, symbols, today)

    # cheap schedule gate BEFORE the expensive load + indicator precompute —
    # on a hold day only the account mark runs
    latest = (session.query(DailyBar.date)
              .filter(DailyBar.symbol.in_(symbols), DailyBar.date < today)
              .order_by(DailyBar.date.desc()).first())
    if latest is None:
        return {"action": "skip", "reason": "no completed bars before today"}
    decision_date = latest[0]

    journal = TradeJournal(session)
    open_lots = _open_lots(journal, promo.strategy_name)
    has_history = bool(open_lots or journal.get_closed_trades(promo.strategy_name))
    if has_history and not _is_rebalance_day(today, decision_date, promo.schedule):
        _mark_account(session, promo.strategy_name, journal, price_fn, dry_run)
        return {"action": "hold", "reason": f"not a {promo.schedule} boundary"}

    bars = load_bars(session, symbols, start=today - timedelta(days=HISTORY_DAYS))
    bars = {s: df for s, df in bars.items() if len(df)}
    if not bars:
        return {"action": "skip", "reason": "no bars"}
    prep, _ = prepare_bars(bars)

    ctx = _build_context(prep, decision_date, None)
    weights = strategy.rebalance(ctx) or {}
    weights = {s: w for s, w in weights.items() if w and w > 0}
    total = sum(weights.values())
    if total > 1.0:
        weights = {s: w / total for s, w in weights.items()}

    prices: dict[str, float] = {}
    for s in set(symbols) | set(t.symbol for t in open_lots):
        px = price_fn(s)
        if px is None or px <= 0:    # stale/halted: fall back to last close
            px = prep[s]["last_close"] if s in prep else 0.0
        prices[s] = px

    cash = _recalc_cash(journal, promo.strategy_name)
    current: dict[str, int] = {}
    for t in open_lots:
        current[t.symbol] = current.get(t.symbol, 0) + t.shares
    equity = cash + sum(sh * prices.get(s, 0.0) for s, sh in current.items())

    slip = SLIPPAGE_BPS / 1e4
    now = datetime.now(timezone.utc)
    actions: list[dict] = []

    # sells first (raise cash) — anything over target, or fully exited names
    for sym in sorted(set(current) | set(weights)):
        px = prices.get(sym, 0.0)
        if px <= 0:
            continue
        target_shares = int((weights.get(sym, 0.0) * equity) / px)
        delta = target_shares - current.get(sym, 0)
        delta_value = abs(delta) * px
        opens_or_closes = (current.get(sym, 0) == 0) or (target_shares == 0)
        if delta >= 0 or (delta_value < REBALANCE_BAND * equity and not opens_or_closes):
            continue
        to_sell = -delta
        fill = px * (1 - slip)
        if not dry_run:
            for lot in sorted((t for t in open_lots if t.symbol == sym),
                              key=lambda t: t.entry_time):
                if to_sell <= 0:
                    break
                n = min(lot.shares, to_sell)
                _close_lot(journal, lot, n, fill,
                           now, "REBALANCE")
                to_sell -= n
            open_lots = _open_lots(journal, promo.strategy_name)
            current[sym] = sum(t.shares for t in open_lots if t.symbol == sym)
            cash = _recalc_cash(journal, promo.strategy_name)
        actions.append({"side": "SELL", "symbol": sym, "shares": -delta,
                        "price": round(fill, 4)})

    # then buys, capped by cash
    for sym in sorted(weights):
        px = prices.get(sym, 0.0)
        if px <= 0:
            continue
        target_shares = int((weights[sym] * equity) / px)
        delta = target_shares - current.get(sym, 0)
        delta_value = delta * px
        opens = current.get(sym, 0) == 0
        if delta <= 0 or (delta_value < REBALANCE_BAND * equity and not opens):
            continue
        fill = px * (1 + slip)
        affordable = int(cash / fill) if fill > 0 else 0
        buy = min(delta, affordable)
        if buy <= 0:
            continue
        if not dry_run:
            trade = Trade(
                trade_id=str(uuid.uuid4()), strategy_name=promo.strategy_name,
                symbol=sym, direction=Direction.LONG, trade_type=TradeType.SWING,
                entry_price=round(fill, 4), shares=buy,
                stop_loss=0.0, target=0.0, confidence=1.0, entry_time=now,
                entry_reasoning=(f"v2 portfolio rebalance: target weight "
                                 f"{weights[sym]:.4f}, decision date {decision_date}"),
            )
            journal.log_trade(trade, commit=False)
            cash -= buy * fill
        actions.append({"side": "BUY", "symbol": sym, "shares": buy,
                        "price": round(fill, 4)})

    if not dry_run:
        session.commit()
        _mark_account(session, promo.strategy_name, journal, price_fn, dry_run)

    return {"action": "rebalance" if actions else "hold",
            "decision_date": str(decision_date),
            "weights": {s: round(w, 4) for s, w in weights.items()},
            "trades": actions}


def _mark_account(session, strategy_name: str, journal: TradeJournal,
                  price_fn, dry_run: bool) -> None:
    """Upsert the strategy_accounts row from the trades table + live prices."""
    if dry_run:
        return
    cash = _recalc_cash(journal, strategy_name)
    open_lots = _open_lots(journal, strategy_name)
    closed = journal.get_closed_trades(strategy_name)
    positions_value = 0.0
    for t in open_lots:
        px = price_fn(t.symbol)
        positions_value += (px if px and px > 0 else t.entry_price) * t.shares
    equity = cash + positions_value

    acct = (session.query(StrategyAccount)
            .filter(StrategyAccount.strategy_name == strategy_name).one_or_none())
    if acct is None:
        acct = StrategyAccount(strategy_name=strategy_name,
                               starting_capital=STARTING_CAPITAL,
                               peak_equity=STARTING_CAPITAL)
        session.add(acct)
    acct.cash_balance = round(cash, 2)
    acct.open_positions_value = round(positions_value, 2)
    acct.total_equity = round(equity, 2)
    acct.peak_equity = round(max(acct.peak_equity or STARTING_CAPITAL, equity), 2)
    acct.drawdown_pct = round(
        (acct.peak_equity - equity) / acct.peak_equity * 100, 2) if acct.peak_equity else 0.0
    acct.realized_pnl = round(sum(t.pnl_dollars or 0.0 for t in closed), 2)
    session.commit()


def main(argv: list[str] | None = None) -> None:
    """Manual cycle runner: ``python -m edgefinder.engine.live [--dry-run]``."""
    import argparse
    import json

    from edgefinder.data.polygon import PolygonDataProvider
    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="compute and print intended trades; persist nothing")
    p.add_argument("--date", default=None,
                   help="simulate a cycle date (YYYY-MM-DD; default today UTC)")
    args = p.parse_args(argv)

    today = date.fromisoformat(args.date) if args.date else None
    summary = run_portfolio_cycle(
        get_session_factory(get_engine()),
        provider=PolygonDataProvider(),
        today=today,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
