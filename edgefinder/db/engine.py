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


def get_engine(url: str | None = None, echo: bool | None = None):
    """Create SQLAlchemy engine.

    Priority for URL:
    1. Explicit url parameter
    2. DATABASE_URL env var (Render provides this)
    3. settings.database_url (defaults to SQLite)
    """
    db_url = url or os.getenv("DATABASE_URL", "") or settings.database_url

    # Render uses postgres:// but SQLAlchemy 2.x requires postgresql://
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
        engine = create_engine(
            db_url,
            echo=_echo,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
        )

    logger.info("Database engine created: %s", "SQLite" if is_sqlite else "PostgreSQL")
    return engine


def get_session_factory(engine=None) -> sessionmaker[Session]:
    """Get a sessionmaker bound to the given (or default) engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)
