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
- The account integrity formula follows CLAUDE.md, extended with dividend
  cash credits (the live counterpart of the lab's total-return adjustment):
  cash = starting_capital + sum(closed pnl) + sum(dividend credits)
       - sum(open cost basis),
  recomputed from the trades + dividend_credits tables every cycle
  (self-heal). Credits are written when an ex-date passes while lots are
  held (one row per strategy/symbol/ex_date, never recomputed).
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

import pandas as pd
from sqlalchemy import func

from edgefinder.agents.journal import record_observation
from edgefinder.core.models import Direction, Trade, TradeStatus, TradeType
from edgefinder.data.pit_fundamentals import PITFundamentals
from edgefinder.db.models import (
    DailyBar,
    DividendCredit,
    DividendRecord,
    PromotedStrategy,
    StrategyAccount,
    SystemHeartbeat,
)
from edgefinder.engine.backtest import _build_context, prepare_bars
from edgefinder.engine.data import (
    adjust_for_splits,
    load_bars,
    load_bars_from_store,
    load_splits,
    parse_universe_spec,
    resolve_universe,
    trailing_rank_start,
)
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
# cross-sectional universes need ranking breadth + indicator warmup; 550
# calendar days covers the 126-trading-day rank window AND the engine's
# 210-trading-day indicator warmup with margin
UNIVERSE_HISTORY_DAYS = 550
# shrink guard: refuse to trade a resolution carrying fewer than this
# fraction of top_n names (a thin store/manifest would otherwise quietly
# concentrate the book into whatever happened to load)
UNIVERSE_MIN_FRACTION = 0.9
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


def ensure_recent_dividends(session, provider, symbols: list[str]) -> int:
    """Top up the dividends table for the traded universe (best-effort).

    Live credits can only be as fresh as the dividends table, and nothing
    else updates it in production. Polygon declares ex-dates ahead of time,
    so a daily idempotent top-up (skip-existing per (symbol, ex_date)) keeps
    credits exact. Needs the raw Polygon REST client; absent (tests, stub
    providers), it skips cleanly.
    """
    client = getattr(provider, "_client", None)
    if client is None:  # CachedDataProvider wraps the real provider
        client = getattr(getattr(provider, "_provider", None), "_client", None)
    if client is None or not symbols:
        return 0
    try:
        from edgefinder.data.dividends_backfill import backfill

        return int(backfill(session, client, symbols).get("rows_written", 0))
    except Exception:
        logger.exception("dividend refresh failed")
        return 0


def _credit_dividends(session, strategy_name: str, open_lots, today: date) -> float:
    """Credit cash for dividends whose ex-date passed while lots were held.

    Ex-date convention: a lot earns a dividend when it was OPENED strictly
    before the ex-date (bought yesterday or earlier = held at the prior
    close, which is what gets the payment). Cycles run daily and trades only
    happen at cycle time AFTER this step, so the open lots seen here are the
    holdings as of every ex-date being credited — including ex-dates from
    missed cycles, which self-heal on the next run.

    One credit row per (strategy, symbol, ex_date). Splits from partial
    sells carry the ORIGINAL entry time, so they can't double-credit.
    Returns dollars credited by this call.
    """
    by_symbol: dict[str, list] = {}
    for t in open_lots:
        by_symbol.setdefault(t.symbol, []).append(t)

    credited = 0.0
    for symbol, lots in by_symbol.items():
        earliest_entry = min(t.entry_time for t in lots).date()
        divs = (session.query(DividendRecord)
                .filter(DividendRecord.symbol == symbol,
                        DividendRecord.ex_date > earliest_entry,
                        DividendRecord.ex_date <= today).all())
        if not divs:
            continue
        already = {r[0] for r in session.query(DividendCredit.ex_date)
                   .filter(DividendCredit.strategy_name == strategy_name,
                           DividendCredit.symbol == symbol).all()}
        for div in divs:
            if div.ex_date in already:
                continue
            shares = sum(t.shares for t in lots
                         if t.entry_time.date() < div.ex_date)
            if shares <= 0:
                continue
            amount = round(shares * div.cash_amount, 2)
            session.add(DividendCredit(
                strategy_name=strategy_name, symbol=symbol,
                ex_date=div.ex_date, shares=shares, amount=amount))
            credited += amount
            logger.info("%s: dividend credit %s %s x%d = $%.2f",
                        strategy_name, symbol, div.ex_date, shares, amount)
    if credited:
        session.commit()
    return credited


def _open_lots(journal: TradeJournal, strategy_name: str) -> list:
    return journal.get_open_trades(strategy_name)


def _recalc_cash(session, journal: TradeJournal, strategy_name: str) -> float:
    """CLAUDE.md integrity formula, recomputed from the trades table
    (+ dividend credits, the v2 extension)."""
    closed = journal.get_closed_trades(strategy_name)
    open_lots = _open_lots(journal, strategy_name)
    realized = sum(t.pnl_dollars or 0.0 for t in closed)
    open_cost = sum(t.entry_price * t.shares for t in open_lots)
    credits = session.query(
        func.coalesce(func.sum(DividendCredit.amount), 0.0)
    ).filter(DividendCredit.strategy_name == strategy_name).scalar() or 0.0
    return STARTING_CAPITAL + realized + credits - open_cost


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

        # one cycle-scoped cache for the full-market R2 load + per-spec
        # universe resolutions — ~1000+ frames must load ONCE per cycle and
        # be shared by every universe strategy, never re-loaded per strategy
        universe_cache: dict = {}

        ok = True
        for promo in promos:
            try:
                summary["strategies"][promo.strategy_name] = _run_one(
                    session, promo, provider=provider, today=today,
                    price_fn=price_fn, dry_run=dry_run,
                    universe_cache=universe_cache)
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


def _spy_calendar(session, as_of: date) -> list[date]:
    """SPY's trading dates through ``as_of`` — the NYSE calendar the
    validator plans on (SPY is a protected full-history symbol)."""
    rows = (session.query(DailyBar.date)
            .filter(DailyBar.symbol == "SPY", DailyBar.date <= as_of)
            .order_by(DailyBar.date).all())
    return [r[0] for r in rows]


def _load_frames(session, symbols: list[str], today: date) -> dict[str, pd.DataFrame]:
    """RAW frames for ``symbols``: R2 deep history topped up with any
    fresher rows already in the DB.

    Targeted by design — the first live cycle proved a full-manifest store
    load does not fit the production instance's memory; ranking happens in
    SQL over the DB hot set instead (see _resolve_universe_frames), so only
    the resolved universe (+ held names) ever needs frames.

    Recency top-up: R2 syncs nightly at 19:00 ET while the 04:15 UTC
    bars-nightly ingest can leave the DB one day fresher at the 9:45 cycle —
    frames whose last date trails the latest DB bar get the newer DB rows
    appended (raw, deduped by date).
    """
    frames = load_bars_from_store(
        symbols, start=today - timedelta(days=UNIVERSE_HISTORY_DAYS))
    db_latest = (session.query(func.max(DailyBar.date))
                 .filter(DailyBar.date < today).scalar())
    if db_latest is None:
        return frames
    stale = {s: df["date"].iloc[-1] for s, df in frames.items()
             if len(df) and df["date"].iloc[-1] < db_latest}
    if not stale:
        return frames
    fresh = load_bars(session, sorted(stale),
                      start=min(stale.values()) + timedelta(days=1),
                      end=db_latest, split_adjust=False)   # raw, like the store
    topped = 0
    for sym, last in stale.items():
        add = fresh.get(sym)
        if add is None or not len(add):
            continue
        add = add[add["date"] > last]                       # dedupe by date
        if not len(add):
            continue
        frames[sym] = (pd.concat(
            [frames[sym], add[["date", "open", "high", "low", "close", "volume"]]],
            ignore_index=True).sort_values("date").reset_index(drop=True))
        topped += 1
    if topped:
        logger.info("universe frames: topped up %d symbols from the DB "
                    "(store lagged %s)", topped, db_latest)
    return frames


def _resolve_universe_frames(session, promo, decision_date: date, today: date,
                             *, lot_symbols: list[str], dry_run: bool,
                             cache: dict):
    """Resolve a cross-sectional promo's PIT universe and build its frames.

    Returns ``(resolved, frames, fundamentals, note)``; ``resolved`` is None
    when the shrink guard fired with no last-good fallback (the caller must
    not trade). Mirrors the validator's --universe semantics — but the
    RANKING runs in SQL over daily_bars (resolve_universe, the store
    ranker's parity-tested twin; the hot set holds the nightly top-1000, a
    strict superset of any top-500) so the cycle never loads full-market
    frames: the production instance OOM'd on that during the first live
    cycle (2026-06-11, no heartbeat, no resolutions persisted). Frames are
    then loaded from R2 ONLY for the resolved set + held names,
    split-adjusted, with PIT fundamentals handed to the context.
    """
    top_n, rank_offset = parse_universe_spec(promo.universe_spec)
    rank_window = promo.rank_window if promo.rank_window is not None else 126

    key = (promo.universe_spec, rank_window)
    resolved = cache.setdefault("resolved", {}).get(key)
    if resolved is None:
        calendar = _spy_calendar(session, decision_date)
        rank_start = (trailing_rank_start(calendar, decision_date, rank_window)
                      if calendar else None)
        resolved = resolve_universe(session, "top", [], top_n,
                                    as_of=decision_date,
                                    rank_offset=rank_offset,
                                    rank_start=rank_start)
        cache["resolved"][key] = resolved

    note = None
    if len(resolved) < UNIVERSE_MIN_FRACTION * top_n:
        message = (f"{promo.strategy_name}: universe {promo.universe_spec} "
                   f"resolved only {len(resolved)}/{top_n} names as of "
                   f"{decision_date} — refusing to trade on it")
        logger.error(message)
        if not dry_run:
            record_observation(
                session, agent_name="engine.live", severity="CRITICAL",
                category="live_universe", message=message,
                metadata={"strategy": promo.strategy_name,
                          "universe": promo.universe_spec,
                          "resolved": len(resolved), "top_n": top_n,
                          "decision_date": str(decision_date)})
        last_good = list(getattr(promo, "resolved_symbols", None) or [])
        if not last_good:
            return None, None, None, message
        resolved = last_good
        note = (f"shrunken universe ({message.split(': ', 1)[1]}); fell back "
                f"to the last good resolution of {promo.resolved_at}")
    elif not dry_run:
        promo.resolved_symbols = list(resolved)
        promo.resolved_at = today
        session.commit()

    # frames for the resolved set PLUS any held names that dropped out of
    # it — their last close backs the sell fill if the quote source fails;
    # cached per cycle so the twelve top:500 promos share one R2 load
    needed = sorted(set(resolved) | set(lot_symbols))
    have: dict = cache.setdefault("frames", {})
    missing = [s for s in needed if s not in have]
    if missing:
        have.update(_load_frames(session, missing, today))
    frames = {s: have[s] for s in needed if s in have and len(have[s])}
    frames = adjust_for_splits(frames, load_splits(session, list(frames)))

    # PIT fundamentals, exactly as the validator hands them to the context:
    # without these, value/growth sleeves silently see None and go all-cash
    pit = cache.get("pit")
    if pit is None:
        pit = cache["pit"] = PITFundamentals(session)
    pit.preload(needed)
    return list(resolved), frames, pit, note


def _run_one(session, promo, *, provider, today, price_fn, dry_run,
             universe_cache=None) -> dict:
    # a paused account never trades — neither the paper runner nor a real
    # (live_manual) book. It is still MARKED so its equity curve stays
    # continuous; the dry-run path yields no weights, so a paused book also
    # produces no real-money ticket (engine/live_ticket.dry_run_weights).
    acct = (session.query(StrategyAccount)
            .filter(StrategyAccount.strategy_name == promo.strategy_name)
            .one_or_none())
    if acct is not None and acct.is_paused:
        _mark_account(session, promo.strategy_name, TradeJournal(session),
                      price_fn, dry_run)
        return {"action": "skip", "reason": "account paused"}

    factory = make_strategy_factory(promo.spec)
    strategy = factory()

    basis = getattr(promo, "prices_basis", None)
    if basis and "dividend-adjusted" in basis:
        # the live runner trades RAW prices; a strategy validated on
        # total-return prices sees systematically different signals on
        # high-yield names — disclosed, not blocked (paper tier)
        logger.warning("%s was validated on '%s' but live trades raw prices",
                       promo.strategy_name, basis)

    journal = TradeJournal(session)
    open_lots = _open_lots(journal, promo.strategy_name)
    universe_spec = getattr(promo, "universe_spec", None)
    if universe_spec:
        # cross-sectional universe: per-symbol Polygon top-ups over 500+
        # names would be 500 REST calls; bars-nightly already ingests the
        # top-1000 into the DB. Refresh only the held names (their bars
        # back the cheap account mark + dividend credits).
        refresh_symbols = sorted({t.symbol for t in open_lots})
    else:
        refresh_symbols = list(promo.symbols or [])
    ensure_recent_bars(session, provider, refresh_symbols, today)
    ensure_recent_dividends(session, provider, refresh_symbols)

    # cheap schedule gate BEFORE the expensive load + indicator precompute —
    # on a hold day only the account mark runs (universe strategies must
    # never touch the R2 store on hold days)
    latest_q = session.query(DailyBar.date).filter(DailyBar.date < today)
    if not universe_spec:
        latest_q = latest_q.filter(DailyBar.symbol.in_(refresh_symbols))
    latest = latest_q.order_by(DailyBar.date.desc()).first()
    if latest is None:
        return {"action": "skip", "reason": "no completed bars before today"}
    decision_date = latest[0]

    # credit ex-dates crossed since the last cycle BEFORE any trading or
    # marking — runs on hold days too (ex-dates rarely land on boundaries)
    if not dry_run:
        _credit_dividends(session, promo.strategy_name, open_lots, today)
    has_history = bool(open_lots or journal.get_closed_trades(promo.strategy_name))
    if has_history and not _is_rebalance_day(today, decision_date, promo.schedule):
        _mark_account(session, promo.strategy_name, journal, price_fn, dry_run)
        return {"action": "hold", "reason": f"not a {promo.schedule} boundary"}

    universe_note = None
    fundamentals = None
    if universe_spec:
        symbols, bars, fundamentals, universe_note = _resolve_universe_frames(
            session, promo, decision_date, today,
            lot_symbols=refresh_symbols, dry_run=dry_run,
            cache=universe_cache if universe_cache is not None else {})
        if symbols is None:
            return {"action": "error", "reason": universe_note}
    else:
        symbols = refresh_symbols
        bars = load_bars(session, symbols,
                         start=today - timedelta(days=HISTORY_DAYS))

    bars = {s: df for s, df in bars.items() if len(df)}
    if not bars:
        return {"action": "skip", "reason": "no bars"}
    prep, _ = prepare_bars(bars)

    # the decision context covers ONLY the (resolved) universe — the
    # validator restricts each window's bars to its PIT universe the same
    # way, so a held name that dropped out gets no weight and is sold below;
    # its frame stays in prep so the sell fill can fall back to last close
    # current book + cash BEFORE the decision, so a stateful strategy sees
    # its holdings (as weights, at decision-date closes — look-ahead-free,
    # identical to the backtest's _build_context) and can choose to HOLD.
    cash = _recalc_cash(session, journal, promo.strategy_name)
    current: dict[str, int] = {}
    for t in open_lots:
        current[t.symbol] = current.get(t.symbol, 0) + t.shares

    universe_set = set(symbols)
    ctx = _build_context({s: p for s, p in prep.items() if s in universe_set},
                         decision_date, fundamentals,
                         holdings_shares=current, cash=cash)
    weights = strategy.rebalance(ctx) or {}
    weights = {s: w for s, w in weights.items() if w and w > 0}
    total = sum(weights.values())
    if total > 1.0:
        weights = {s: w / total for s, w in weights.items()}

    prices: dict[str, float] = {}
    for s in set(weights) | set(t.symbol for t in open_lots):
        px = price_fn(s)
        if px is None or px <= 0:    # stale/halted: fall back to last close
            px = prep[s]["last_close"] if s in prep else 0.0
        prices[s] = px

    equity = cash + sum(sh * prices.get(s, 0.0) for s, sh in current.items())

    slip = SLIPPAGE_BPS / 1e4
    # trade timestamps carry the CYCLE's trading day, not the wall-clock
    # date: a --date/simulated run (or an evening manual run after the UTC
    # rollover) must not stamp trades onto a different day than the cycle
    # they belong to — dividend eligibility compares entry date vs ex-date
    now = datetime.combine(today, datetime.now(timezone.utc).timetz())
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
            cash = _recalc_cash(session, journal, promo.strategy_name)
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

    result = {"action": "rebalance" if actions else "hold",
              "decision_date": str(decision_date),
              "weights": {s: round(w, 4) for s, w in weights.items()},
              "trades": actions}
    if universe_spec:
        result["universe"] = {"spec": universe_spec, "size": len(symbols)}
        if universe_note:
            result["universe"]["note"] = universe_note
    return result


def _mark_account(session, strategy_name: str, journal: TradeJournal,
                  price_fn, dry_run: bool) -> None:
    """Upsert the strategy_accounts row from the trades table + live prices."""
    if dry_run:
        return
    cash = _recalc_cash(session, journal, strategy_name)
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
