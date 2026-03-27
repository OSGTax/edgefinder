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

    # Lynch metrics
    peg_ratio: Mapped[float | None] = mapped_column(Float)
    earnings_growth: Mapped[float | None] = mapped_column(Float)
    debt_to_equity: Mapped[float | None] = mapped_column(Float)
    revenue_growth: Mapped[float | None] = mapped_column(Float)
    institutional_pct: Mapped[float | None] = mapped_column(Float)
    lynch_score: Mapped[float | None] = mapped_column(Float)
    lynch_category: Mapped[str | None] = mapped_column(String(50))

    # Burry metrics
    fcf_yield: Mapped[float | None] = mapped_column(Float)
    price_to_tangible_book: Mapped[float | None] = mapped_column(Float)
    short_interest: Mapped[float | None] = mapped_column(Float)
    ev_to_ebitda: Mapped[float | None] = mapped_column(Float)
    current_ratio: Mapped[float | None] = mapped_column(Float)
    burry_score: Mapped[float | None] = mapped_column(Float)

    composite_score: Mapped[float | None] = mapped_column(Float)
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


# ── 8. sentiment_readings ───────────────────────────────


class SentimentReading(Base):
    __tablename__ = "sentiment_readings"
    __table_args__ = (
        Index("idx_sentiment_symbol_ts", "symbol", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(20))
    score: Mapped[float] = mapped_column(Float)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    is_trending: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── 9. manual_injections ────────────────────────────────


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
