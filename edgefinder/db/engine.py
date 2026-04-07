"""EdgeFinder v2 — SQLAlchemy 2.0 engine and session management."""

from __future__ import annotations

import logging
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from config.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base for all ORM models."""
    pass


def _ensure_ssl(url: str) -> str:
    """Append sslmode=require to PostgreSQL URLs that don't already specify it."""
    if "sslmode" in url:
        return url
    separator = "&" if "?" in url else "?"
    return url + separator + "sslmode=require"


def get_engine(url: str | None = None, echo: bool | None = None):
    """Create SQLAlchemy engine.

    Priority for URL:
    1. Explicit url parameter
    2. DATABASE_URL env var (Render provides this)
    3. settings.database_url (defaults to SQLite)

    Supabase notes:
    - Use the Transaction mode pooler URL (port 6543) from your dashboard
    - SSL is enforced automatically (sslmode=require appended if missing)
    - pool_recycle handles Supavisor retiring idle connections
    """
    db_url = url or os.getenv("DATABASE_URL", "") or settings.database_url

    # Render/Supabase may use postgres:// but SQLAlchemy 2.x requires postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    is_sqlite = db_url.startswith("sqlite")
    _echo = echo if echo is not None else settings.db_echo

    if is_sqlite and os.getenv("RENDER"):
        logger.warning(
            "WARNING: Using SQLite on Render — data WILL BE LOST on redeploy! "
            "Set DATABASE_URL to a PostgreSQL connection string."
        )

    if is_sqlite:
        kwargs: dict = {
            "echo": _echo,
            "connect_args": {"check_same_thread": False},
        }
        if ":memory:" in db_url:
            kwargs["poolclass"] = StaticPool

        engine = create_engine(db_url, **kwargs)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # Enforce SSL for all PostgreSQL connections (Supabase requires it)
        db_url = _ensure_ssl(db_url)

        is_supabase = "pooler.supabase.com" in db_url
        if is_supabase:
            logger.info("Supabase pooler detected — using pool_recycle=%ds", settings.db_pool_recycle)

        engine = create_engine(
            db_url,
            echo=_echo,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=settings.db_pool_recycle,
        )

    logger.info("Database engine created: %s", "SQLite" if is_sqlite else "PostgreSQL")
    return engine


def get_session_factory(engine=None) -> sessionmaker[Session]:
    """Get a sessionmaker bound to the given (or default) engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)
