"""Shared test fixtures for EdgeFinder v2."""

import pytest
from sqlalchemy.orm import sessionmaker

from edgefinder.db.engine import Base, get_engine


@pytest.fixture(autouse=True)
def _agent_test_isolation(monkeypatch):
    """Pin every test to the pg/SQLite transport and never the live REST DB.

    The dev/Routine environment may export SUPABASE_URL + service-role key (and
    R2_* creds). Without this, ``agent.store.transport()`` would resolve to
    "rest" and the agent tools would read/write the *production* Supabase
    project from the test suite. Force pg, strip the cloud creds, and clear the
    cached store/session between tests so each gets a clean per-test engine.
    """
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
              "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)

    def _reset():
        import agent.data as agent_data
        import agent.store as agent_store

        agent_store.get_store.cache_clear()
        agent_data._session_factory = None

    _reset()
    yield
    _reset()


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
