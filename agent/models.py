"""The autonomous trading agent's own tables (greenfield rebuild).

A clean namespace (``desk_*``) that lives in the SAME database as the kept
market-data tables but shares none of the retired trading/app schema. The
agent reads and writes only these; the market-data tables (daily_bars,
dividends, ticker_splits, fundamentals_snapshots, ticker_news, index_daily)
are read-only inputs reached through the kept data-access layer.

Source-of-truth rule (mirrors the old engine's discipline): ``desk_trades``
is append-only and authoritative for cash. The portfolio/positions/equity
rows are recomputable projections — ``agent.ledger`` rebuilds them from the
trade ledger on every mark, so a corrupted projection can always be healed.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from edgefinder.db.engine import Base

# The single paper book the agent runs. One account, full discretion.
ACCOUNT = "agent"
STARTING_CAPITAL = 100_000.0


# ── desk_trades — append-only fill ledger (source of truth for cash) ──


class DeskTrade(Base):
    """One executed paper fill. Append-only; never updated or deleted.

    Cash is always ``STARTING_CAPITAL + Σ(SELL dollars) - Σ(BUY dollars)``
    over this table, so the ledger is auditable and self-correcting.
    """

    __tablename__ = "desk_trades"
    __table_args__ = (Index("idx_desk_trades_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    shares: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)  # fill price after costs
    dollars: Mapped[float] = mapped_column(Float)  # signed-agnostic gross = shares*price
    rationale: Mapped[str | None] = mapped_column(Text)


# ── desk_positions — open lots (projection, rebuilt from the ledger) ──


class DeskPosition(Base):
    """Current open lot per symbol. Rebuilt from desk_trades on every mark."""

    __tablename__ = "desk_positions"
    __table_args__ = (
        UniqueConstraint("account", "symbol", name="uq_desk_pos_account_symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    shares: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)  # cost basis per share
    last_price: Mapped[float | None] = mapped_column(Float)  # latest mark
    opened_at: Mapped[datetime | None] = mapped_column(DateTime)
    marked_at: Mapped[datetime | None] = mapped_column(DateTime)


# ── desk_equity — equity-curve time series (one row per mark) ──


class DeskEquity(Base):
    """Account equity snapshot for the curve. Append-only series."""

    __tablename__ = "desk_equity"
    __table_args__ = (Index("idx_desk_equity_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    return_pct: Mapped[float] = mapped_column(Float)


# ── desk_strategy_state — the agent's current, evolving strategy ──


class DeskStrategyState(Base):
    """The agent's living strategy. One row per version; latest is current.

    The agent rewrites this when it adopts/evolves an approach. ``version``
    increments on a real pivot (a journal entry should accompany it).
    """

    __tablename__ = "desk_strategy_state"
    __table_args__ = (Index("idx_desk_state_account_ver", "account", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(80))
    thesis: Mapped[str | None] = mapped_column(Text)  # plain-English approach
    rules: Mapped[dict | None] = mapped_column(JSON)   # structured selection rules
    params: Mapped[dict | None] = mapped_column(JSON)  # knobs the agent tunes
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


# ── desk_journal — pivots, tweaks, and notes (the agent's diary) ──


class DeskJournal(Base):
    """Why the strategy changed. The narrative audit of the agent's evolution."""

    __tablename__ = "desk_journal"
    __table_args__ = (Index("idx_desk_journal_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # pivot | tweak | note
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str | None] = mapped_column(Text)
    version_from: Mapped[int | None] = mapped_column(Integer)
    version_to: Mapped[int | None] = mapped_column(Integer)


# ── desk_thinking — per-run narration feed (the "live thinking") ──


class DeskThinking(Base):
    """Streamed reasoning lines for one run. Powers the live thinking panel."""

    __tablename__ = "desk_thinking"
    __table_args__ = (Index("idx_desk_thinking_run_ts", "run_id", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    phase: Mapped[str | None] = mapped_column(String(30))  # observe|research|decide|execute
    text: Mapped[str] = mapped_column(Text)


# ── desk_decisions — one decision record per run ──


class DeskDecision(Base):
    """The agent's decision for a run: regime, picks, target book, watchlist.

    ``picks`` is a list of per-name dossiers (symbol, action, why_now,
    rationale, evidence, news) — the chart-forward holdings panel reads it.
    """

    __tablename__ = "desk_decisions"
    __table_args__ = (
        UniqueConstraint("account", "run_id", name="uq_desk_decision_run"),
        Index("idx_desk_decision_account_ts", "account", "ts"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    decision_date: Mapped[date | None] = mapped_column(Date)
    regime: Mapped[str | None] = mapped_column(String(40))
    summary: Mapped[str | None] = mapped_column(Text)
    target_weights: Mapped[dict | None] = mapped_column(JSON)
    picks: Mapped[list | None] = mapped_column(JSON)
    watchlist: Mapped[list | None] = mapped_column(JSON)
    strategy_version: Mapped[int | None] = mapped_column(Integer)


# ── desk_backtests — grounding evidence the agent ran ──


class DeskBacktest(Base):
    """A backtest the agent ran to ground an idea. Evidence panel reads these."""

    __tablename__ = "desk_backtests"
    __table_args__ = (Index("idx_desk_backtest_account_ts", "account", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    label: Mapped[str] = mapped_column(String(120))
    spec: Mapped[dict | None] = mapped_column(JSON)    # symbols/rule/schedule/window
    result: Mapped[dict | None] = mapped_column(JSON)  # return/sharpe/dd/excess vs SPY


# ── desk_changelog — what the app-evolution routine shipped ("What's New") ──


class DeskChangelog(Base):
    """One user-facing improvement the agent made to the dashboard itself.

    The end-of-day app-evolution routine, when it ships a genuinely useful
    change to what /desk shows, records a row here: a short ``title`` and a
    plain-English ``detail`` explaining the feature and why it helps. The page
    lights a "NEW" badge for entries inside the spotlight window and lists them
    in the "What's New" panel — so users (and the owner) can see how the app is
    growing and read what each addition does.
    """

    __tablename__ = "desk_changelog"
    __table_args__ = (Index("idx_desk_changelog_ts", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="feature")  # feature|improvement|data|disclaimer|fix
    title: Mapped[str] = mapped_column(String(160))
    detail: Mapped[str | None] = mapped_column(Text)  # the explanation users read
    version: Mapped[str | None] = mapped_column(String(20))  # app version at ship time
    run_id: Mapped[str | None] = mapped_column(String(40))


# Idempotent CREATE TABLE IF NOT EXISTS DDL for render_start.py (Render skips
# create_all). Postgres-flavored; SQLite ignores the JSON type harmlessly.
DESK_TABLE_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS desk_trades (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        run_id VARCHAR(40),
        symbol VARCHAR(10) NOT NULL,
        side VARCHAR(4) NOT NULL,
        shares INTEGER NOT NULL,
        price FLOAT NOT NULL,
        dollars FLOAT NOT NULL,
        rationale TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_trades_account_ts ON desk_trades (account, ts)",
    "CREATE INDEX IF NOT EXISTS idx_desk_trades_run ON desk_trades (run_id)",
    """CREATE TABLE IF NOT EXISTS desk_positions (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        symbol VARCHAR(10) NOT NULL,
        shares INTEGER NOT NULL,
        avg_price FLOAT NOT NULL,
        last_price FLOAT,
        opened_at TIMESTAMP,
        marked_at TIMESTAMP,
        CONSTRAINT uq_desk_pos_account_symbol UNIQUE (account, symbol)
    )""",
    """CREATE TABLE IF NOT EXISTS desk_equity (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        cash FLOAT NOT NULL,
        positions_value FLOAT NOT NULL,
        equity FLOAT NOT NULL,
        return_pct FLOAT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_equity_account_ts ON desk_equity (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_strategy_state (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        version INTEGER DEFAULT 1,
        name VARCHAR(80) NOT NULL,
        thesis TEXT,
        rules JSON,
        params JSON,
        updated_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_state_account_ver ON desk_strategy_state (account, version)",
    """CREATE TABLE IF NOT EXISTS desk_journal (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        kind VARCHAR(20) NOT NULL,
        title VARCHAR(200) NOT NULL,
        body TEXT,
        version_from INTEGER,
        version_to INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_journal_account_ts ON desk_journal (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_thinking (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40) NOT NULL,
        ts TIMESTAMP DEFAULT NOW(),
        phase VARCHAR(30),
        text TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_thinking_run_ts ON desk_thinking (run_id, ts)",
    """CREATE TABLE IF NOT EXISTS desk_decisions (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40) NOT NULL,
        ts TIMESTAMP DEFAULT NOW(),
        decision_date DATE,
        regime VARCHAR(40),
        summary TEXT,
        target_weights JSON,
        picks JSON,
        watchlist JSON,
        strategy_version INTEGER,
        CONSTRAINT uq_desk_decision_run UNIQUE (account, run_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_decision_account_ts ON desk_decisions (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_backtests (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40),
        ts TIMESTAMP DEFAULT NOW(),
        label VARCHAR(120) NOT NULL,
        spec JSON,
        result JSON
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_backtest_account_ts ON desk_backtests (account, ts)",
    """CREATE TABLE IF NOT EXISTS desk_changelog (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        kind VARCHAR(20) DEFAULT 'feature',
        title VARCHAR(160) NOT NULL,
        detail TEXT,
        version VARCHAR(20),
        run_id VARCHAR(40)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_changelog_ts ON desk_changelog (ts)",
]
