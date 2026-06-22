"""EdgeFinder autonomous trading agent — the greenfield rebuild.

A self-directed AI stock picker + paper trader that runs on Claude Code
Routines. The Routine session IS the agent: each cycle it reads its own
evolving strategy + the market, reasons in the open, trades a $100k paper
book with full discretion, grounds ideas with backtests, and narrates
everything to the trading-desk page.

Packages:
  agent.models        — the desk_* tables (the agent's own clean schema)
  agent.data          — data-access over the kept R2/Postgres bar layer
  agent.market        — observe CLI (regime / quote / history / news / universe)
  agent.backtest_tool — ground ideas in history (CLI)
  agent.ledger        — the paper book (record / mark / state)
  agent.brain         — strategy state / journal / thinking / decision writers
"""
