"""EdgeFinder v2 — SQLAlchemy 2.0 ORM models (10 tables).

All models use Mapped + mapped_column (SQLAlchemy 2.0 style).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from edgefinder.db.engine import Base


# ── 1. tickers ────────────────────────────────────────────


class Ticker(Base):
    __tablename__ = "tickers"
    __table_args__ = (UniqueConstraint("symbol", name="uq_tickers_symbol"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    company_name: Mapped[str | None] = mapped_column(String(200))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(200))
    market_cap: Mapped[float | None] = mapped_column(Float)
    last_price: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="scanner")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    scan_batch: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    fundamentals: Mapped[Fundamental | None] = relationship(
        back_populates="ticker_rel", uselist=False
    )


# ── 2. fundamentals ──────────────────────────────────────


class Fundamental(Base):
    __tablename__ = "fundamentals"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"), unique=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)

    # Core ratios (computed from raw financials)
    peg_ratio: Mapped[float | None] = mapped_column(Float)
    earnings_growth: Mapped[float | None] = mapped_column(Float)
    debt_to_equity: Mapped[float | None] = mapped_column(Float)
    revenue_growth: Mapped[float | None] = mapped_column(Float)
    institutional_pct: Mapped[float | None] = mapped_column(Float)
    fcf_yield: Mapped[float | None] = mapped_column(Float)
    price_to_tangible_book: Mapped[float | None] = mapped_column(Float)
    short_interest: Mapped[float | None] = mapped_column(Float)
    ev_to_ebitda: Mapped[float | None] = mapped_column(Float)
    current_ratio: Mapped[float | None] = mapped_column(Float)

    # Extended ratios
    price_to_earnings: Mapped[float | None] = mapped_column(Float)
    price_to_book: Mapped[float | None] = mapped_column(Float)
    return_on_equity: Mapped[float | None] = mapped_column(Float)
    return_on_assets: Mapped[float | None] = mapped_column(Float)
    dividend_yield: Mapped[float | None] = mapped_column(Float)
    free_cash_flow: Mapped[float | None] = mapped_column(Float)
    quick_ratio: Mapped[float | None] = mapped_column(Float)

    # Short interest details
    short_shares: Mapped[int | None] = mapped_column(Integer)
    days_to_cover: Mapped[float | None] = mapped_column(Float)

    # Dividends
    dividend_amount: Mapped[float | None] = mapped_column(Float)
    ex_dividend_date: Mapped[str | None] = mapped_column(String(20))

    # News sentiment
    news_sentiment: Mapped[str | None] = mapped_column(String(20))

    # Technical indicators (from Massive API)
    rsi_14: Mapped[float | None] = mapped_column(Float)
    ema_21: Mapped[float | None] = mapped_column(Float)
    sma_50: Mapped[float | None] = mapped_column(Float)
    macd_value: Mapped[float | None] = mapped_column(Float)
    macd_signal: Mapped[float | None] = mapped_column(Float)
    macd_histogram: Mapped[float | None] = mapped_column(Float)

    raw_data: Mapped[dict | None] = mapped_column(JSON)

    scan_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    ticker_rel: Mapped[Ticker] = relationship(back_populates="fundamentals")


# ── 3. trades ────────────────────────────────────────────


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    strategy_name: Mapped[str] = mapped_column(String(50), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    direction: Mapped[str] = mapped_column(String(5))
    trade_type: Mapped[str] = mapped_column(String(5))

    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float)
    shares: Mapped[int] = mapped_column(Integer)
    stop_loss: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)

    entry_time: Mapped[datetime] = mapped_column(DateTime)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="OPEN", index=True)

    pnl_dollars: Mapped[float | None] = mapped_column(Float)
    pnl_percent: Mapped[float | None] = mapped_column(Float)
    r_multiple: Mapped[float | None] = mapped_column(Float)
    exit_reason: Mapped[str | None] = mapped_column(String(30))

    market_snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("market_snapshots.id")
    )
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    sentiment_data: Mapped[dict | None] = mapped_column(JSON)
    technical_signals: Mapped[dict | None] = mapped_column(JSON)

    sequence_num: Mapped[int | None] = mapped_column(Integer, index=True)
    integrity_hash: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    market_snapshot: Mapped[MarketSnapshotRecord | None] = relationship()


# ── 4. market_snapshots ──────────────────────────────────


class MarketSnapshotRecord(Base):
    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    spy_price: Mapped[float] = mapped_column(Float)
    spy_change_pct: Mapped[float] = mapped_column(Float)
    qqq_price: Mapped[float] = mapped_column(Float)
    qqq_change_pct: Mapped[float] = mapped_column(Float)
    iwm_price: Mapped[float] = mapped_column(Float)
    iwm_change_pct: Mapped[float] = mapped_column(Float)
    dia_price: Mapped[float] = mapped_column(Float)
    dia_change_pct: Mapped[float] = mapped_column(Float)
    vix_level: Mapped[float] = mapped_column(Float)

    market_regime: Mapped[str] = mapped_column(String(20), default="sideways")
    sector_performance: Mapped[dict | None] = mapped_column(JSON)
    advance_decline_ratio: Mapped[float | None] = mapped_column(Float)


# ── 5. strategy_accounts ────────────────────────────────


class StrategyAccount(Base):
    __tablename__ = "strategy_accounts"
    __table_args__ = (
        UniqueConstraint("strategy_name", name="uq_strategy_accounts_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(50), index=True)
    starting_capital: Mapped[float] = mapped_column(Float, default=5000.0)
    cash_balance: Mapped[float] = mapped_column(Float, default=5000.0)
    open_positions_value: Mapped[float] = mapped_column(Float, default=0.0)
    total_equity: Mapped[float] = mapped_column(Float, default=5000.0)
    peak_equity: Mapped[float] = mapped_column(Float, default=5000.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    pdt_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


# ── 6. strategy_snapshots ───────────────────────────────


class StrategySnapshot(Base):
    __tablename__ = "strategy_snapshots"
    __table_args__ = (
        Index("idx_strat_snap_name_ts", "strategy_name", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(50))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_equity: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float)
    total_return_pct: Mapped[float] = mapped_column(Float)


# ── 7. index_daily ──────────────────────────────────────


class IndexDaily(Base):
    __tablename__ = "index_daily"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_index_daily_symbol_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[datetime] = mapped_column(DateTime)
    close: Mapped[float] = mapped_column(Float)
    change_pct: Mapped[float] = mapped_column(Float)


# ── 8. manual_injections ────────────────────────────────


class ManualInjection(Base):
    __tablename__ = "manual_injections"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    target_strategy: Mapped[str | None] = mapped_column(String(50))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 10. strategy_parameters ─────────────────────────────


class StrategyParameterLog(Base):
    __tablename__ = "strategy_parameters"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_name: Mapped[str] = mapped_column(String(50), index=True)
    param_name: Mapped[str] = mapped_column(String(100))
    old_value: Mapped[str | None] = mapped_column(String(200))
    new_value: Mapped[str | None] = mapped_column(String(200))
    changed_by: Mapped[str] = mapped_column(String(30))
    changed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 11. ticker_strategy_qualifications ─────────────────


class TickerStrategyQualification(Base):
    """Per-strategy qualification for each ticker.

    Tracks which strategies each stock qualifies for, with an optional
    composite score (0-100) for ranked watchlist generation.
    """
    __tablename__ = "ticker_strategy_qualifications"
    __table_args__ = (
        UniqueConstraint("ticker_id", "strategy_name", name="uq_ticker_strategy"),
        Index("idx_tsq_strategy_qualified", "strategy_name", "qualified"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker_id: Mapped[int] = mapped_column(Integer, ForeignKey("tickers.id"))
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    strategy_name: Mapped[str] = mapped_column(String(50), index=True)
    qualified: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    scan_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 12. ticker_news ────────────────────────────────────


class TickerNews(Base):
    """News articles for tickers, accumulated over time."""

    __tablename__ = "ticker_news"
    __table_args__ = (
        UniqueConstraint("symbol", "title", "published_utc", name="uq_ticker_news"),
        Index("idx_ticker_news_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10))
    title: Mapped[str] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(200))
    published_utc: Mapped[str | None] = mapped_column(String(30))
    article_url: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    publisher_name: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 13. ticker_dividends ───────────────────────────────


class TickerDividend(Base):
    """Dividend history for tickers."""

    __tablename__ = "ticker_dividends"
    __table_args__ = (
        UniqueConstraint("symbol", "ex_dividend_date", name="uq_ticker_dividend"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    ex_dividend_date: Mapped[str] = mapped_column(String(20))
    pay_date: Mapped[str | None] = mapped_column(String(20))
    cash_amount: Mapped[float | None] = mapped_column(Float)
    declaration_date: Mapped[str | None] = mapped_column(String(20))
    frequency: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 14. ticker_splits ──────────────────────────────────


class TickerSplit(Base):
    """Stock split history for tickers."""

    __tablename__ = "ticker_splits"
    __table_args__ = (
        UniqueConstraint("symbol", "execution_date", name="uq_ticker_split"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    execution_date: Mapped[str] = mapped_column(String(20))
    split_from: Mapped[int | None] = mapped_column(Integer)
    split_to: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 15. trade_context ──────────────────────────────────


class TradeContext(Base):
    """Rich market context captured at trade time for later AI analysis."""

    __tablename__ = "trade_context"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[str] = mapped_column(String(36), ForeignKey("trades.trade_id"), unique=True)
    recent_news: Mapped[dict | None] = mapped_column(JSON)
    sector_prices: Mapped[dict | None] = mapped_column(JSON)
    related_tickers: Mapped[dict | None] = mapped_column(JSON)
    short_interest: Mapped[dict | None] = mapped_column(JSON)
    dividends: Mapped[dict | None] = mapped_column(JSON)
    indicators: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 16. agent_observations ─────────────────────────────


class AgentObservation(Base):
    """Findings reported by management agents (watchdog, strategist, ...).

    Observations are read by the dashboard for a unified audit view and
    by other agents to decide whether a finding needs follow-up action.
    """

    __tablename__ = "agent_observations"
    __table_args__ = (
        Index("idx_agent_obs_agent_ts", "agent_name", "timestamp"),
        Index("idx_agent_obs_unresolved", "resolved_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(50), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    severity: Mapped[str] = mapped_column(String(10))  # INFO | WARN | ERROR | CRITICAL
    category: Mapped[str] = mapped_column(String(50), index=True)
    message: Mapped[str] = mapped_column(Text)
    obs_metadata: Mapped[dict | None] = mapped_column(JSON)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(50), nullable=True)


# ── 17. agent_actions ──────────────────────────────────


class AgentAction(Base):
    """Actions taken by management agents — diagnoses, proposals, PRs.

    Actions are the write-side counterpart to observations: every change
    an agent makes to the system (commit, PR, comment, diagnostic write)
    lands here so postmortems read a single timeline of agent activity.
    """

    __tablename__ = "agent_actions"
    __table_args__ = (
        Index("idx_agent_act_agent_ts", "agent_name", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(50), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    action_type: Mapped[str] = mapped_column(String(30))  # diagnose | propose_param | propose_strategy | open_pr | comment
    summary: Mapped[str] = mapped_column(String(500))
    files_touched: Mapped[list | None] = mapped_column(JSON)
    commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pr_url: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | submitted | merged | rejected
    observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_observations.id"), nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# ── 18. agent_memory ───────────────────────────────────


class AgentMemory(Base):
    """Persistent memory for each management agent — the agent's
    "learning" across ticks.

    The reasoning step reads this content before calling the LLM and
    may rewrite it to capture new patterns, known false positives, or
    recent resolutions. One row per agent_name.
    """

    __tablename__ = "agent_memory"
    __table_args__ = (
        UniqueConstraint("agent_name", name="uq_agent_memory_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(50), index=True)
    content: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
