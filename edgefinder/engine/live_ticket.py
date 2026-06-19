"""Order-ticket + reconciliation engine for the REAL-MONEY book.

The unattended paper runner (``engine/live._run_one``) *simulates* fills.
This module instead *proposes* an order ticket for a human to approve and a
broker (Robinhood via its MCP server) to place. The sizing discipline is the
same sells-before-buys / weight->size logic as the paper runner, but:

- it returns an order TICKET (a list of ``OrderLine``) and simulates nothing;
- it is **notional-aware** (fractional shares) so a small real account can
  hold the full basket — a $2,000 book cannot buy one whole share of a $739
  ETF, but it can buy $X notional of it;
- broker state (cash, holdings, prices) is passed IN, so the module never
  touches the DB, the network, or the clock and is fully unit-testable.

The real book NEVER mixes with paper: it is tagged
``PromotedStrategy.execution_mode = "live_manual"`` and its trades carry
``broker`` / ``broker_order_id`` for audit + idempotency.

The monthly ritual (see ``reviews/REAL-MONEY-RUNBOOK.md``):
1. read the real account (cash + holdings) from Robinhood via MCP;
2. get the strategy's target weights from a dry-run cycle (``dry_run_weights``
   — zero drift from the proven decision path);
3. ``propose_orders`` -> a ticket diffing target vs the REAL holdings;
4. a human approves;
5. place the approved orders via RH MCP, capturing each order id + fill;
6. write the fills back to ``trades`` under the real book's namespace;
7. ``reconcile`` -> assert DB book == RH actual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

# Same dust/churn band as the paper runner (engine/live.REBALANCE_BAND): skip
# re-trues smaller than 1% of equity unless they open or fully close a name.
REBALANCE_BAND = 0.01
# Robinhood's fractional/notional order floor; never propose a smaller order.
MIN_NOTIONAL = 1.00
# Refuse to build a ticket if the freshest daily bar is older than this many
# calendar days (a stale data asset would size the book off a wrong picture).
MAX_STALENESS_DAYS = 4


# ── order ticket ────────────────────────────────────────────


@dataclass
class OrderLine:
    """One proposed BUY or SELL, notional-aware.

    ``notional`` (dollars) is the primary instruction for a fractional broker;
    ``shares`` is the estimate at ``ref_price`` for a whole-share broker and
    for human sanity-checking. ``ref_price`` is a reference quote, not a
    guaranteed fill — the actual fill price is written back after placement.
    """

    side: str               # "BUY" | "SELL"
    symbol: str
    notional: float         # dollars to trade (always > 0)
    shares: float           # positive share estimate at ref_price
    ref_price: float
    target_weight: float
    current_shares: float
    target_shares: float
    reason: str


@dataclass
class OrderTicket:
    """A full proposed rebalance for one book — the thing a human approves."""

    strategy_name: str
    equity: float
    cash_before: float
    cash_after: float                       # projected, at ref prices
    lines: list[OrderLine] = field(default_factory=list)
    projected_book: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not self.lines

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_noop"] = self.is_noop
        return d


def propose_orders(
    strategy_name: str,
    target_weights: dict[str, float],
    holdings: dict[str, float],
    cash: float,
    prices: dict[str, float],
    *,
    fractional: bool = True,
    rebalance_band: float = REBALANCE_BAND,
    min_notional: float = MIN_NOTIONAL,
) -> OrderTicket:
    """Diff target weights against actual broker holdings -> an order ticket.

    Mirrors ``engine/live._run_one`` sizing: equity = cash + market value of
    holdings; each name targets ``weight * equity``; re-trues under
    ``rebalance_band`` of equity are skipped unless they open or fully close a
    position; SELLS come first (they raise the cash the BUYS draw on); BUYS are
    capped by projected cash. Simulates nothing and persists nothing.

    ``holdings`` / ``target_weights`` may carry fractional share counts. With
    ``fractional=False`` orders round down to whole shares (whole-share broker).
    """
    weights = {s: float(w) for s, w in target_weights.items() if w and w > 0}
    total = sum(weights.values())
    if total > 1.0:                              # never lever past fully invested
        weights = {s: w / total for s, w in weights.items()}
    holdings = {s: float(sh) for s, sh in holdings.items() if sh}

    warnings: list[str] = []
    tradable = set(weights) | set(holdings)
    for s in sorted(tradable):
        if prices.get(s, 0.0) <= 0:
            warnings.append(f"no price for {s} — skipped (cannot size)")

    equity = cash + sum(sh * prices.get(s, 0.0) for s, sh in holdings.items())

    def size(value: float, px: float) -> float:
        sh = value / px
        return sh if fractional else float(int(sh))

    book = dict(holdings)
    proj_cash = cash
    sells: list[OrderLine] = []
    buys: list[OrderLine] = []

    # SELLS first — anything over target, or a name being fully exited.
    for sym in sorted(tradable):
        px = prices.get(sym, 0.0)
        if px <= 0:
            continue
        w = weights.get(sym, 0.0)
        cur = book.get(sym, 0.0)
        delta_value = w * equity - cur * px
        full_exit = w <= 0 and cur > 0
        if delta_value >= 0:
            continue
        if abs(delta_value) < rebalance_band * equity and not full_exit:
            continue
        sell_shares = cur if full_exit else size(-delta_value, px)
        sell_shares = min(sell_shares, cur)      # never sell more than held
        notional = sell_shares * px
        if sell_shares <= 0 or notional < min_notional:
            continue
        target_shares = cur - sell_shares
        sells.append(OrderLine(
            "SELL", sym, round(notional, 2), sell_shares, px, w, cur,
            target_shares,
            reason=("exit (target 0%)" if full_exit
                    else f"trim to {w:.2%} target")))
        book[sym] = target_shares
        proj_cash += notional

    # BUYS — capped by projected cash (sell proceeds included).
    for sym in sorted(weights):
        px = prices.get(sym, 0.0)
        if px <= 0:
            continue
        w = weights[sym]
        cur = book.get(sym, 0.0)
        delta_value = w * equity - cur * px
        opens = cur == 0
        if delta_value <= 0:
            continue
        if delta_value < rebalance_band * equity and not opens:
            continue
        spend = min(delta_value, proj_cash)
        if spend < min_notional:
            continue
        buy_shares = size(spend, px)
        notional = buy_shares * px
        if buy_shares <= 0 or notional < min_notional:
            continue
        target_shares = cur + buy_shares
        buys.append(OrderLine(
            "BUY", sym, round(notional, 2), buy_shares, px, w, cur,
            target_shares,
            reason=("open (new position)" if opens
                    else f"add to {w:.2%} target")))
        book[sym] = target_shares
        proj_cash -= notional

    return OrderTicket(
        strategy_name=strategy_name,
        equity=round(equity, 2),
        cash_before=round(cash, 2),
        cash_after=round(proj_cash, 2),
        lines=sells + buys,
        projected_book={s: round(sh, 6) for s, sh in book.items() if sh > 1e-9},
        warnings=warnings,
    )


# ── reconciliation ──────────────────────────────────────────


@dataclass
class PositionDiff:
    symbol: str
    db_shares: float
    broker_shares: float

    @property
    def delta(self) -> float:               # broker - db
        return self.broker_shares - self.db_shares


@dataclass
class ReconcileReport:
    clean: bool
    position_diffs: list[PositionDiff]
    db_cash: float | None
    broker_cash: float | None

    @property
    def cash_diff(self) -> float | None:    # broker - db
        if self.db_cash is None or self.broker_cash is None:
            return None
        return self.broker_cash - self.db_cash

    def summary(self) -> str:
        if self.clean:
            return "reconciliation CLEAN (DB book == broker)"
        parts = [f"{d.symbol}: db={d.db_shares:g} broker={d.broker_shares:g} "
                 f"(Δ{d.delta:+g})" for d in self.position_diffs]
        if self.cash_diff is not None and abs(self.cash_diff) > 0:
            parts.append(f"cash: db=${self.db_cash:.2f} "
                         f"broker=${self.broker_cash:.2f} (Δ{self.cash_diff:+.2f})")
        return "reconciliation MISMATCH — " + "; ".join(parts)


def reconcile(
    db_positions: dict[str, float],
    broker_positions: dict[str, float],
    *,
    db_cash: float | None = None,
    broker_cash: float | None = None,
    share_tolerance: float = 1e-4,
    cash_tolerance: float = 0.01,
) -> ReconcileReport:
    """Compare the DB real-book positions/cash against the broker's actual.

    Returns only the mismatches. ``clean`` is True iff every position matches
    within ``share_tolerance`` and (when both cash figures are given) cash
    matches within ``cash_tolerance``.
    """
    diffs: list[PositionDiff] = []
    for sym in sorted(set(db_positions) | set(broker_positions)):
        d = float(db_positions.get(sym, 0.0))
        b = float(broker_positions.get(sym, 0.0))
        if abs(b - d) > share_tolerance:
            diffs.append(PositionDiff(sym, d, b))
    report = ReconcileReport(
        clean=False, position_diffs=diffs, db_cash=db_cash, broker_cash=broker_cash)
    cash_ok = report.cash_diff is None or abs(report.cash_diff) <= cash_tolerance
    report.clean = not diffs and cash_ok
    return report


# ── decision-path glue (zero-drift target weights) ──────────


def data_staleness_days(session, today: date | None = None,
                        benchmark: str = "SPY") -> int:
    """Calendar days between today and the freshest daily bar for ``benchmark``.

    SPY is a protected full-history symbol present every trading day, so its
    latest bar is the data asset's freshness watermark. A missing series
    returns a large number (treated as stale) so the gate fails closed.
    """
    from sqlalchemy import func

    from edgefinder.db.models import DailyBar

    latest = (session.query(func.max(DailyBar.date))
              .filter(DailyBar.symbol == benchmark).scalar())
    if latest is None:
        return 10 ** 6
    return ((today or date.today()) - latest).days


def assert_data_fresh(session, *, today: date | None = None,
                      max_staleness_days: int = MAX_STALENESS_DAYS) -> int:
    """Raise if the data asset is staler than ``max_staleness_days``.

    The freshness gate: refuse to build a real-money ticket on stale data.
    Returns the staleness in days when fresh enough.
    """
    days = data_staleness_days(session, today)
    if days > max_staleness_days:
        raise StaleDataError(
            f"data asset is {days}d stale (max {max_staleness_days}d) — "
            "refusing to build a real-money ticket")
    return days


class StaleDataError(RuntimeError):
    """Raised by the freshness gate when daily bars are too old to trade on."""


def dry_run_weights(session_factory, strategy_name: str, *, provider=None,
                    price_fn=None, today: date | None = None) -> dict[str, float]:
    """Target weights for one promoted book, from a dry-run cycle.

    Reuses the EXACT proven decision path (``run_portfolio_cycle(dry_run=True)``
    resolves the universe, builds the context, calls ``strategy.rebalance``)
    so the real book's target never drifts from the paper book's. A paused
    book (``is_paused``) yields no weights -> this raises, so a paused book
    produces no ticket.

    NOTE: stateless target-weight strategies only (the 12 monthly finalists,
    e.g. ``growth_value_barbell``). A *stateful* strategy's weights depend on
    current holdings; the dry cycle sees the PAPER holdings, not the real
    book's — real-book support for stateful strategies is a later enhancement.
    """
    from edgefinder.engine.live import run_portfolio_cycle

    summary = run_portfolio_cycle(
        session_factory, provider=provider, price_fn=price_fn,
        today=today, dry_run=True)
    s = summary.get("strategies", {}).get(strategy_name)
    if s is None:
        raise ValueError(f"{strategy_name!r} is not an active promoted strategy")
    if "weights" not in s:
        raise ValueError(
            f"no target weights for {strategy_name!r} "
            f"(action={s.get('action')}, reason={s.get('reason')})")
    return {k: float(v) for k, v in s["weights"].items() if v and v > 0}


# ── CLI: the no-money dry ritual helper ─────────────────────


def main(argv: list[str] | None = None) -> None:
    """Build (and print) a proposed ticket WITHOUT placing anything.

        python -m edgefinder.engine.live_ticket --strategy growth_value_barbell \
            --cash 2000 --holdings '{}'

    Pass the REAL account's cash + holdings (read from Robinhood via MCP) to
    preview the exact ticket the ritual would propose, then STOP. With no
    ``--holdings`` it assumes a flat (all-cash) account — the first-funding
    case (buy the whole basket).
    """
    import argparse
    import json

    from edgefinder.data.polygon import PolygonDataProvider
    from edgefinder.db.engine import get_engine, get_session_factory
    from edgefinder.db.models import DailyBar

    p = argparse.ArgumentParser(description="Preview a real-money order ticket")
    p.add_argument("--strategy", required=True, help="promoted strategy_name")
    p.add_argument("--cash", type=float, required=True, help="real account cash")
    p.add_argument("--holdings", default="{}",
                   help='JSON {"SYM": shares} of the real account (default flat)')
    p.add_argument("--whole-shares", action="store_true",
                   help="round to whole shares (default: fractional/notional)")
    p.add_argument("--date", default=None, help="cycle date (YYYY-MM-DD)")
    args = p.parse_args(argv)

    today = date.fromisoformat(args.date) if args.date else None
    holdings = {k.upper(): float(v) for k, v in json.loads(args.holdings).items()}

    engine = get_engine()
    sf = get_session_factory(engine)
    provider = PolygonDataProvider()

    session = sf()
    try:
        stale = assert_data_fresh(session, today=today)
        weights = dry_run_weights(sf, args.strategy, provider=provider, today=today)

        def last_close(sym: str) -> float:
            row = (session.query(DailyBar.close).filter(DailyBar.symbol == sym)
                   .order_by(DailyBar.date.desc()).first())
            return float(row[0]) if row else 0.0

        prices = {}
        for sym in set(weights) | set(holdings):
            px = provider.get_latest_price(sym)
            prices[sym] = float(px) if px and px > 0 else last_close(sym)
    finally:
        session.close()

    ticket = propose_orders(
        args.strategy, weights, holdings, args.cash, prices,
        fractional=not args.whole_shares)
    out = ticket.to_dict()
    out["data_staleness_days"] = stale
    print(json.dumps(out, indent=2, default=str))
    print("\n*** DRY PREVIEW — no orders placed. Approve in the ritual to trade. ***")


if __name__ == "__main__":
    main()
