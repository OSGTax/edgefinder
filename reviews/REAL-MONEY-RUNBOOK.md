# Real-money runbook — the monthly, manual-approve rebalance on Robinhood

> **One small book. Manual approval on every order. No unattended real-money
> execution, ever.** The Render scheduler keeps running the *paper* fleet; the
> real book trades ONLY inside a Claude session where a human approves the
> ticket. This document is the operational source of truth for that ritual.

## What this is (and is not)

EdgeFinder is a paper-simulation system. The unattended runner
(`engine/live._run_one`) *simulates* fills and must **never** place a real
order. Going live is a **monthly, Claude-mediated, human-approved rebalance
ritual** on a single small Robinhood account, using Robinhood's MCP server for
execution. The 12 finalists rebalance monthly, so this is a once-a-month event.

- **Broker:** Robinhood, via its MCP server.
- **Control:** manual-approve — the system *proposes* an order ticket; a human
  approves before anything is placed.
- **Scope (v1):** one book (`live_core`), small capital you set.

The real book is tagged so it never mixes with paper:
`PromotedStrategy.execution_mode = "live_manual"` and every real fill carries
`broker` + `broker_order_id` (audit + idempotency). The dashboard, the
live-vs-lab scorecard, and the watchdog then track it exactly like a paper book.

## One-time setup

1. **Connect the broker MCP** (in your terminal, once per machine):
   ```
   claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
   ```
   Then start Claude Code and run `/mcp` to authenticate.
2. **Pin the RH tool names.** On first `/mcp` connect, list the Robinhood
   tools and record the EXACT names below — the ritual calls these by name.
   Expected surface (verify, then fill in):

   | Need | Robinhood MCP tool (verify on connect) |
   |---|---|
   | account cash / buying power | `…` |
   | current holdings / positions | `…` |
   | latest quote (optional; we also have Polygon) | `…` |
   | place order (notional/fractional, BUY/SELL) | `…` |
   | order status / fill detail | `…` |

   > Until these are verified, run the **DRY RITUAL** only (no orders placed).
3. **Create the real book** (one row), small starting capital:
   ```
   python -m edgefinder.engine.promote --spec growth_value_barbell \
       --universe top:500 --finalist --name live_core
   # then mark it real-money + set starting capital to your funded amount:
   #   PromotedStrategy.execution_mode = "live_manual"
   #   StrategyAccount(strategy_name="live_core", starting_capital=<funded $>)
   ```
   (`growth_value_barbell` — best risk profile in the hunt, ~Sharpe 2.1 / ~21%
   worst-DD. An equal-weight blend of the 12 is the alternative; the ticket
   engine supports either.)

## The monthly ritual

Run on a rebalance boundary (first trading day of the month). In a Claude
session with the Robinhood MCP connected:

1. **Read the real account (RH MCP):** cash/buying power + current holdings
   `{symbol: shares}`.
2. **Preview the ticket** with the real account state:
   ```
   python -m edgefinder.engine.live_ticket --strategy live_core \
       --cash <RH cash> --holdings '<RH holdings JSON>'
   ```
   This: (a) runs the **data-freshness gate** (`assert_data_fresh` — refuses on
   stale Polygon data), (b) computes target weights from the proven decision
   path (`dry_run_weights` → `run_portfolio_cycle(dry_run=True)`, zero drift
   from paper), (c) diffs target vs RH holdings with `propose_orders`
   (notional/fractional, sells-before-buys), and (d) prints the ticket + the
   projected post-trade book. **It places nothing.**
3. **Human approves** the ticket. Checklist:
   - [ ] data staleness is 0–1 trading days
   - [ ] no `warnings` (every traded name has a price)
   - [ ] `cash_after` ≥ 0 and the projected book matches the target weights
   - [ ] total BUY notional ≤ available cash + SELL proceeds
   - [ ] the book is `live_core` (NOT a paper strategy)
4. **Place approved orders via RH MCP**, one line at a time, SELLS first.
   Use **notional** (dollar) orders for fractional shares. Capture each
   `broker_order_id` and the actual fill price/shares.
5. **Write fills back** to the `trades` table under `strategy_name="live_core"`
   via `TradeJournal`, with `broker="robinhood"` and the captured
   `broker_order_id` on each row (the actual fill price/shares, not the ticket
   estimate). This keeps `_recalc_cash` and the dashboard exact.
6. **Reconcile.** Re-read RH holdings + cash and run:
   ```python
   from edgefinder.engine.live_ticket import reconcile
   report = reconcile(db_positions, rh_positions, db_cash=…, broker_cash=…)
   ```
   - If `report.clean`: record a heartbeat `real_book_reconcile:live_core`
     with `ok=True`.
   - If NOT clean: record the same heartbeat with `ok=False` and
     `detail={"summary": report.summary()}`. The **watchdog** escalates this to
     a CRITICAL observation (and the alerts pipeline to a GitHub issue). Do not
     trade again until it reads clean.

## Abort / kill

- **Before placing:** just don't approve — nothing was sent.
- **Mid-ritual:** stop placing further lines; write back only the orders that
  actually filled (with their `broker_order_id`), then reconcile to capture the
  true state.
- **Halt the book entirely:** set `StrategyAccount(strategy_name="live_core").is_paused = True`.
  A paused book produces **no ticket** (`dry_run_weights` yields no weights)
  and the paper cycle skips it (still marks its equity curve).
- **Cancel an in-flight order:** use the broker's app or the RH MCP
  cancel/order-status tool; then reconcile.

## Safety rails (built in)

- **Data-freshness gate** — `assert_data_fresh` refuses a ticket on stale bars
  (default > 4 calendar days), failing closed if SPY has no bars.
- **`is_paused` honored** — wired into `engine/live._run_one`; a paused book
  never trades and yields no ticket.
- **Reconciliation check** — `watchdog.check_real_book_reconciliation` turns a
  recorded DB-vs-broker mismatch into a CRITICAL alert.
- **Small capital** is the primary risk control. Scale only after one clean
  ritual AND the live-vs-lab scorecard shows the real book tracks the lab.

## Known v1 limits

- **Stateless target-weight strategies only.** `dry_run_weights` reads the
  PAPER book's holdings when building context, so a *stateful* strategy's
  weights would not reflect the real book. `growth_value_barbell` and the other
  monthly finalists are stateless → fine. Real-book support for stateful
  strategies is a later enhancement.
- **Write-back + placement are manual** (done in-session via the RH MCP tools);
  no automated order router. That is the point of v1.
