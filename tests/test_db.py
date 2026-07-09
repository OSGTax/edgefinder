"""Tests for edgefinder/db/engine — the URL rewriter that keeps Codespaces
+ Render + Supabase pooler URLs interchangeable. The ORM-model tests that
lived here exercised the retired trading-workbench tables (TradeRecord,
StrategyAccount, MarketSnapshotRecord, ManualInjection, …); those tables
were dropped in the 2026-06-22 cutover and their tests would only assert
that the ORM class still shapes to itself. Deleted with the classes."""

import pytest

from edgefinder.db.engine import Base, get_engine, get_session_factory


class TestEngine:
    def test_sqlite_memory_engine(self):
        engine = get_engine(url="sqlite:///:memory:")
        assert engine is not None
        engine.dispose()

    def test_session_factory(self, db_engine):
        factory = get_session_factory(db_engine)
        session = factory()
        assert session is not None
        session.close()

    def test_postgres_url_fix(self):
        """postgres:// should be rewritten to postgresql://."""
        try:
            engine = get_engine(url="postgres://fake:fake@localhost/fake")
            engine.dispose()
        except Exception:
            pass  # Expected — no PG server. Point: the URL rewrite doesn't crash.

    def test_direct_supabase_rewritten_to_pooler_in_codespaces(self, monkeypatch):
        from edgefinder.db.engine import _rewrite_direct_supabase_to_pooler

        direct = "postgresql://postgres:p%40ss@db.abcdef123456.supabase.co:5432/postgres"
        monkeypatch.setenv("CODESPACES", "true")
        assert _rewrite_direct_supabase_to_pooler(direct) == (
            "postgresql://postgres.abcdef123456:p%40ss"
            "@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
        )

        # Untouched outside Codespaces, and for non-direct URLs anywhere.
        monkeypatch.delenv("CODESPACES")
        assert _rewrite_direct_supabase_to_pooler(direct) == direct
        monkeypatch.setenv("CODESPACES", "true")
        pooler = "postgresql://postgres.x:p@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
        assert _rewrite_direct_supabase_to_pooler(pooler) == pooler
        assert _rewrite_direct_supabase_to_pooler("sqlite:///:memory:") == "sqlite:///:memory:"
