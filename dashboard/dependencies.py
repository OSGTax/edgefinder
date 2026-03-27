"""EdgeFinder v2 — FastAPI dependency injection.

Provides DB sessions, data providers, and services to route handlers.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from edgefinder.db.engine import get_engine, get_session_factory

_engine = None
_session_factory = None


def _get_session_factory():
    global _engine, _session_factory
    if _session_factory is None:
        _engine = get_engine()
        _session_factory = get_session_factory(_engine)
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session, closes after request."""
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
