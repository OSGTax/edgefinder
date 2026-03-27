"""Shared test fixtures for EdgeFinder v2."""

import pytest
from sqlalchemy.orm import sessionmaker

from edgefinder.db.engine import Base, get_engine


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with all tables created."""
    engine = get_engine(url="sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Session for a single test, rolled back after."""
    session_factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = session_factory()
    yield session
    session.rollback()
    session.close()
