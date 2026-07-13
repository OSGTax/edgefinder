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
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    shares: Mapped[float] = mapped_column(Float)  # fractional shares (v7)
    price: Mapped[float] = mapped_column(Float)  # fill price after costs
    dollars: Mapped[float] = mapped_column(Float)  # signed-agnostic gross = shares*price
    rationale: Mapped[str | None] = mapped_column(Text)
    # The live-quote snapshot the fill priced off: {bid, ask, mid, t, src}.
    # The honesty contract's receipt — every live fill carries one.
    fill_quote: Mapped[dict | None] = mapped_column(JSON)


# ── desk_positions — open lots (projection, rebuilt from the ledger) ──


class DeskPosition(Base):
    """Current open lot per symbol. Rebuilt from desk_trades on every mark."""

    __tablename__ = "desk_positions"
    __table_args__ = (
        UniqueConstraint("account", "symbol", name="uq_desk_pos_account_symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    shares: Mapped[float] = mapped_column(Float)  # fractional shares (v7)
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
    # Candidates that LOST the slot this run: [{symbol, why_not}]. Graded by
    # the weekly reflection alongside the picks — "the thing I didn't buy did
    # X" doubles the learning signal at zero risk. NOTE: a dev database
    # created before v8.15 lacks this column and the ORM will error reading
    # desk_decisions — rerun scripts/setup_db.py (prod self-heals via the
    # idempotent ALTER in DESK_TABLE_DDL on deploy).
    rejected: Mapped[list | None] = mapped_column(JSON)
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


# ── desk_options_snap — the options IV data bank (one row per underlying/day) ──


class DeskOptionsSnap(Base):
    """Daily options snapshot per underlying: ATM IV, straddle-implied expected
    move, 25-delta skew. Written once/day by the agent's refresh — accumulates
    into the IV history the charts plot and the agent reasons over (IV rank
    becomes computable as the bank grows)."""

    __tablename__ = "desk_options_snap"
    __table_args__ = (
        UniqueConstraint("symbol", "snap_date", name="uq_desk_optsnap_sym_date"),
        Index("idx_desk_optsnap_sym_date", "symbol", "snap_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    snap_date: Mapped[date] = mapped_column(Date)
    spot: Mapped[float | None] = mapped_column(Float)
    atm_iv: Mapped[float | None] = mapped_column(Float)
    expected_move_pct: Mapped[float | None] = mapped_column(Float)
    skew_25d: Mapped[float | None] = mapped_column(Float)
    dte: Mapped[int | None] = mapped_column(Integer)
    expiry: Mapped[str | None] = mapped_column(String(10))


# ── desk_wiki — the agent's self-curated lessons wiki (system-prompt learning) ──


class DeskWiki(Base):
    """One curated page of the agent's lessons wiki.

    Karpathy-style "system prompt learning": a small, size-capped set of pages
    (playbook / lessons / mistakes / market-notes) the agent READS at the start
    of every cycle and REVISES from measured outcomes — knowledge accumulating
    in curated context, not weights. Pages are edited IN PLACE (fixed slugs are
    the curation constraint; no append-only sprawl); every edit writes a
    desk_journal note (kind="wiki"), which is the audit trail — prior body text
    is deliberately not retained. Caps enforced by agent.brain (the only writer).
    """

    __tablename__ = "desk_wiki"
    __table_args__ = (
        UniqueConstraint("account", "slug", name="uq_desk_wiki_account_slug"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    slug: Mapped[str] = mapped_column(String(40), index=True)  # playbook|lessons|mistakes|market-notes
    title: Mapped[str | None] = mapped_column(String(80))
    body: Mapped[str] = mapped_column(Text)  # markdown-lite, hard-capped by the tool
    revision: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now())
    updated_run_id: Mapped[str | None] = mapped_column(String(40))


# ── desk_briefs — the nightly research pack ──


class DeskBrief(Base):
    """One nightly research pack, precomputed while the whole-market data is
    already in hand (the data-refresh routine builds it right after the
    ingest). The hourly trading cycle reads ONE dense payload — regime,
    ranked universe, movers, trend roster, headlines, data-coverage verdict —
    instead of re-deriving it with a dozen exploratory scans, so its context
    goes to deciding, not gathering. One row per (account, brief_date),
    rebuilt in place; written only by ``agent.market brief-build``.
    """

    __tablename__ = "desk_briefs"
    __table_args__ = (
        UniqueConstraint("account", "brief_date", name="uq_desk_brief_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    brief_date: Mapped[date] = mapped_column(Date, index=True)
    built_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    payload: Mapped[dict] = mapped_column(JSON)


# ── the attention system: tripwires + planned wakes ──


class DeskWatch(Base):
    """One standing tripwire the brain armed on its way out.

    The always-on streamer sweeps armed wires against the live tape every few
    seconds and marks trips; the brain reads them FIRST at its next wake.
    Cheap code watches continuously so expensive judgment only shows up when
    something it named actually happened. Written only by ``agent.brain
    watch-set`` / ``watch-clear``; the streamer's only write is the trip.
    """

    __tablename__ = "desk_watch"

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40))
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    kind: Mapped[str] = mapped_column(String(10))  # above | below
    level: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    armed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    until: Mapped[datetime | None] = mapped_column(DateTime)  # expiry (UTC)
    # armed | tripped | expired | disarmed
    status: Mapped[str] = mapped_column(String(10), default="armed", index=True)
    tripped_at: Mapped[datetime | None] = mapped_column(DateTime)
    tripped_price: Mapped[float | None] = mapped_column(Float)


class DeskWake(Base):
    """One self-scheduled check-in the brain planned (and why).

    The budget ledger for the attention system: ``agent.brain wake-plan``
    enforces the per-day cap and minimum gap against these rows BEFORE the
    skill arms the actual one-shot trigger, and the desk shows the owner
    when the trader plans to look next. Append-only.
    """

    __tablename__ = "desk_wakes"

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str] = mapped_column(String(30), default=ACCOUNT, index=True)
    run_id: Mapped[str | None] = mapped_column(String(40))
    at: Mapped[datetime] = mapped_column(DateTime, index=True)  # UTC fire time
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# Idempotent CREATE TABLE IF NOT EXISTS DDL for render_start.py (Render skips
# create_all). Postgres-flavored; SQLite ignores the JSON type harmlessly.
DESK_TABLE_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS desk_trades (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        ts TIMESTAMP DEFAULT NOW(),
        run_id VARCHAR(40),
        symbol VARCHAR(24) NOT NULL,
        side VARCHAR(4) NOT NULL,
        shares FLOAT NOT NULL,
        price FLOAT NOT NULL,
        dollars FLOAT NOT NULL,
        rationale TEXT,
        fill_quote JSON
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_trades_account_ts ON desk_trades (account, ts)",
    "CREATE INDEX IF NOT EXISTS idx_desk_trades_run ON desk_trades (run_id)",
    """CREATE TABLE IF NOT EXISTS desk_positions (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        symbol VARCHAR(24) NOT NULL,
        shares FLOAT NOT NULL,
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
        rejected JSON,
        strategy_version INTEGER,
        CONSTRAINT uq_desk_decision_run UNIQUE (account, run_id)
    )""",
    # Additive upgrade for desk_decisions tables created before v8.15.
    "ALTER TABLE desk_decisions ADD COLUMN IF NOT EXISTS rejected JSON",
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
    """CREATE TABLE IF NOT EXISTS desk_options_snap (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        snap_date DATE NOT NULL,
        spot FLOAT,
        atm_iv FLOAT,
        expected_move_pct FLOAT,
        skew_25d FLOAT,
        dte INTEGER,
        expiry VARCHAR(10),
        CONSTRAINT uq_desk_optsnap_sym_date UNIQUE (symbol, snap_date)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_optsnap_sym_date ON desk_options_snap (symbol, snap_date)",
    """CREATE TABLE IF NOT EXISTS desk_wiki (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        slug VARCHAR(40) NOT NULL,
        title VARCHAR(80),
        body TEXT NOT NULL,
        revision INTEGER DEFAULT 1,
        updated_at TIMESTAMP DEFAULT NOW(),
        updated_run_id VARCHAR(40),
        CONSTRAINT uq_desk_wiki_account_slug UNIQUE (account, slug)
    )""",
    """CREATE TABLE IF NOT EXISTS desk_briefs (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        brief_date DATE NOT NULL,
        built_at TIMESTAMP DEFAULT NOW(),
        payload JSON NOT NULL,
        CONSTRAINT uq_desk_brief_date UNIQUE (account, brief_date)
    )""",
    # Same lockdown as every other desk_* table (scripts/enable_rls.sql):
    # RLS on, zero policies — anon/authenticated denied; the owning postgres
    # role (Render/agent) bypasses. Without this a new public-schema table is
    # world-writable through the Supabase Data API. Idempotent.
    "ALTER TABLE desk_briefs ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_watch (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40),
        symbol VARCHAR(24) NOT NULL,
        kind VARCHAR(10) NOT NULL,
        level FLOAT NOT NULL,
        reason TEXT NOT NULL,
        armed_at TIMESTAMP DEFAULT NOW(),
        until TIMESTAMP,
        status VARCHAR(10) DEFAULT 'armed',
        tripped_at TIMESTAMP,
        tripped_price FLOAT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_watch_status ON desk_watch (account, status)",
    "ALTER TABLE desk_watch ENABLE ROW LEVEL SECURITY",
    """CREATE TABLE IF NOT EXISTS desk_wakes (
        id SERIAL PRIMARY KEY,
        account VARCHAR(30) DEFAULT 'agent',
        run_id VARCHAR(40),
        at TIMESTAMP NOT NULL,
        reason TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_desk_wakes_at ON desk_wakes (account, at)",
    "ALTER TABLE desk_wakes ENABLE ROW LEVEL SECURITY",
]
