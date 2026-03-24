"""
EdgeFinder Database Models and Helpers
======================================
All database models and connection management.
Supports PostgreSQL (via DATABASE_URL) with SQLite fallback for local dev.
"""

import os
import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Boolean, Text, JSON, UniqueConstraint, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from config import settings

logger = logging.getLogger(__name__)
Base = declarative_base()


def _get_database_url(db_path: str | None = None) -> str:
    """
    Determine the database URL to use.

    Priority:
    1. DATABASE_URL env var (Render PostgreSQL provides this)
    2. db_path argument (for tests / explicit override)
    3. settings.DATABASE_PATH (SQLite fallback for local dev)
    """
    database_url = os.getenv("DATABASE_URL", "")

    # Render provides postgres:// but SQLAlchemy 2.x requires postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    if database_url:
        return database_url

    # SQLite fallback
    path = db_path or settings.DATABASE_PATH
    if path == ":memory:":
        return "sqlite:///:memory:"
    return f"sqlite:///{path}"


# ── DATABASE MODELS ──────────────────────────────────────────

class WatchlistStock(Base):
    """A stock that passed fundamental screening."""
    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("ticker", name="uq_watchlist_ticker"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    company_name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(200))
    market_cap = Column(Float)
    price = Column(Float)

    # Lynch Scores
    peg_ratio = Column(Float)
    earnings_growth = Column(Float)
    debt_to_equity = Column(Float)
    revenue_growth = Column(Float)
    institutional_pct = Column(Float)
    lynch_category = Column(String(50))       # fast_grower, stalwart, turnaround, etc.
    lynch_score = Column(Float)               # 0-100

    # Burry Scores
    fcf_yield = Column(Float)
    price_to_tangible_book = Column(Float)
    short_interest = Column(Float)
    ev_to_ebitda = Column(Float)
    current_ratio = Column(Float)
    burry_score = Column(Float)               # 0-100

    # Composite
    composite_score = Column(Float)           # 0-100 (Lynch + Burry weighted)

    # Meta
    scan_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    notes = Column(Text)


class Trade(Base):
    """A simulated trade record."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(36), unique=True, nullable=False)  # UUID
    ticker = Column(String(10), nullable=False, index=True)
    direction = Column(String(5))              # LONG or SHORT
    trade_type = Column(String(5))             # DAY or SWING

    entry_price = Column(Float)
    exit_price = Column(Float)
    shares = Column(Integer)
    stop_loss = Column(Float)
    target = Column(Float)

    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED

    pnl_dollars = Column(Float)
    pnl_percent = Column(Float)
    r_multiple = Column(Float)

    fundamental_score = Column(Float)
    technical_signals = Column(JSON)           # Dict of signals that fired
    news_sentiment = Column(Float)
    confidence_score = Column(Float)

    exit_reason = Column(String(30))           # STOP_HIT, TARGET_HIT, etc.
    market_conditions = Column(JSON)           # SPY trend, VIX, etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Signal(Base):
    """A technical signal event (traded or skipped)."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(20))           # BUY, SELL
    trade_type = Column(String(5))             # DAY or SWING
    confidence = Column(Float)
    indicators = Column(JSON)                  # Which indicators fired
    was_traded = Column(Boolean, default=False)
    trade_id = Column(String(36))              # Links to Trade if executed
    reason_skipped = Column(String(200))       # Why signal was not traded
    timestamp = Column(DateTime, default=datetime.utcnow)


class StrategyParameter(Base):
    """Tracks parameter changes made by the optimizer."""
    __tablename__ = "strategy_parameters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_name = Column(String(100), nullable=False)
    old_value = Column(String(100))
    new_value = Column(String(100))
    reason = Column(Text)
    changed_at = Column(DateTime, default=datetime.utcnow)


class AccountSnapshot(Base):
    """Daily account balance snapshot for equity curve."""
    __tablename__ = "account_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    cash = Column(Float)
    positions_value = Column(Float)
    total_value = Column(Float)
    open_positions = Column(Integer)
    peak_value = Column(Float)
    drawdown_pct = Column(Float)


class Suggestion(Base):
    """User-submitted suggestion for system improvements."""
    __tablename__ = "suggestions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False)   # feature, strategy, bug, other
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(20), default="new")      # new, reviewing, implemented, declined
    created_at = Column(DateTime, default=datetime.utcnow)


# ── ARENA MODELS (Multi-Strategy) ──────────────────────────

class ArenaTradeLog(Base):
    """Immutable audit log for arena trades. One entry per execution."""
    __tablename__ = "arena_trade_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(36), unique=True, nullable=False, index=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    strategy_version = Column(String(20))
    ticker = Column(String(10), nullable=False, index=True)
    action = Column(String(5))                    # BUY, SELL
    direction = Column(String(5))                 # LONG
    trade_type = Column(String(5))                # DAY, SWING

    signal_price = Column(Float)                  # Price at signal time
    execution_price = Column(Float)               # Final price after slippage
    exit_price = Column(Float)                    # Price at close (NULL if open)
    slippage = Column(Float)
    shares = Column(Integer)
    stop_loss = Column(Float)
    target = Column(Float)
    confidence = Column(Float)

    signal_timestamp = Column(DateTime)
    execution_timestamp = Column(DateTime)
    exit_timestamp = Column(DateTime)

    pnl_dollars = Column(Float)
    pnl_percent = Column(Float)
    r_multiple = Column(Float)
    exit_reason = Column(String(30))

    price_source = Column(String(30))             # alpaca, yfinance, etc.
    bar_data_at_decision = Column(JSON)           # OHLCV at decision time
    market_regime = Column(String(20))            # bull, bear, sideways
    signal_overlap = Column(Integer, default=0)   # Other strategies with same signal
    position_overlap = Column(Integer, default=0) # Other strategies holding ticker

    status = Column(String(20), default="OPEN")   # OPEN, CLOSED
    extra_data = Column(JSON)                     # Strategy-specific context
    created_at = Column(DateTime, default=datetime.utcnow)


class ArenaSnapshot(Base):
    """Per-strategy equity snapshot for arena comparison."""
    __tablename__ = "arena_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    cash = Column(Float)
    positions_value = Column(Float)
    total_equity = Column(Float)
    peak_equity = Column(Float)
    drawdown_pct = Column(Float)
    open_positions = Column(Integer)
    realized_pnl = Column(Float)
    unrealized_pnl = Column(Float)
    total_return_pct = Column(Float)
    is_paused = Column(Boolean, default=False)


# ── DATABASE ENGINE ──────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_engine(db_path: str | None = None, echo: bool = False):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = _get_database_url(db_path)
        is_sqlite = url.startswith("sqlite")

        if is_sqlite:
            # Ensure directory exists for SQLite file
            if ":memory:" not in url:
                path = db_path or settings.DATABASE_PATH
                os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

            kwargs = {
                "echo": echo,
                "connect_args": {"check_same_thread": False},
            }
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool

            _engine = create_engine(url, **kwargs)

            # Enable WAL mode for better concurrent access (SQLite only)
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        else:
            # PostgreSQL
            _engine = create_engine(
                url,
                echo=echo,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )

        db_type = "PostgreSQL" if not is_sqlite else "SQLite"
        logger.info(f"Database engine: {db_type}")

    return _engine


def get_session(db_path: str | None = None) -> Session:
    """Get a new database session."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine(db_path)
        _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


def init_db(db_path: str | None = None, echo: bool = False):
    """Create all tables."""
    engine = get_engine(db_path, echo)
    Base.metadata.create_all(engine)
    url = _get_database_url(db_path)
    if url.startswith("postgresql"):
        logger.info("Database initialized (PostgreSQL)")
    else:
        logger.info(f"Database initialized at {db_path or settings.DATABASE_PATH}")
    return engine


def reset_engine():
    """Reset the global engine (useful for testing)."""
    global _engine, _SessionFactory
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
