"""Tests for _load_watchlists — qualifications + manual injection merging."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

from edgefinder.db.models import (
    ManualInjection,
    Ticker,
    TickerStrategyQualification,
)


def _seed_qualification(
    session,
    symbol: str,
    strategy_name: str,
    score: float | None,
    qualified: bool = True,
) -> None:
    ticker = session.query(Ticker).filter_by(symbol=symbol).first()
    if ticker is None:
        ticker = Ticker(symbol=symbol, company_name=symbol, is_active=True, source="test")
        session.add(ticker)
        session.flush()
    session.add(
        TickerStrategyQualification(
            ticker_id=ticker.id,
            symbol=symbol,
            strategy_name=strategy_name,
            qualified=qualified,
            score=score,
            scan_date=datetime.now(timezone.utc),
        )
    )
    session.flush()


@pytest.fixture
def services_module_with_session(db_engine):
    """Wire dashboard.services._session_factory to the in-memory test engine."""
    import dashboard.services as services

    factory = sessionmaker(bind=db_engine, expire_on_commit=False)

    original_factory = services._session_factory
    services._session_factory = factory
    try:
        yield services, factory
    finally:
        services._session_factory = original_factory


class TestLoadWatchlists:
    def test_qualified_tickers_ranked_by_score_desc(self, services_module_with_session):
        services, factory = services_module_with_session
        session = factory()
        try:
            _seed_qualification(session, "AAA", "alpha", score=50.0)
            _seed_qualification(session, "BBB", "alpha", score=90.0)
            _seed_qualification(session, "CCC", "alpha", score=70.0)
            session.commit()
        finally:
            session.close()

        result = services._load_watchlists()
        assert result["alpha"] == ["BBB", "CCC", "AAA"]

    def test_unqualified_rows_are_ignored(self, services_module_with_session):
        services, factory = services_module_with_session
        session = factory()
        try:
            _seed_qualification(session, "XXX", "alpha", score=None, qualified=False)
            session.commit()
        finally:
            session.close()

        result = services._load_watchlists()
        assert "alpha" not in result

    def test_manual_injection_targeted_at_single_strategy(
        self, services_module_with_session
    ):
        services, factory = services_module_with_session
        session = factory()
        try:
            _seed_qualification(session, "AAA", "alpha", score=90.0)
            session.add(ManualInjection(symbol="INJ", target_strategy="alpha"))
            session.commit()
        finally:
            session.close()

        result = services._load_watchlists()
        assert "INJ" in result["alpha"]
        # INJ should come after qualified AAA (qualifications fill first).
        assert result["alpha"].index("AAA") < result["alpha"].index("INJ")

    def test_manual_injection_null_target_fans_out_to_all_strategies(
        self, services_module_with_session
    ):
        services, factory = services_module_with_session
        registry_names = ["alpha", "bravo"]
        session = factory()
        try:
            session.add(ManualInjection(symbol="FANOUT", target_strategy=None))
            session.commit()
        finally:
            session.close()

        with patch(
            "edgefinder.strategies.base.StrategyRegistry.list_names",
            classmethod(lambda cls: list(registry_names)),
        ):
            result = services._load_watchlists()

        assert result.get("alpha") == ["FANOUT"]
        assert result.get("bravo") == ["FANOUT"]

    def test_expired_injection_is_skipped(self, services_module_with_session):
        services, factory = services_module_with_session
        session = factory()
        try:
            session.add(
                ManualInjection(
                    symbol="STALE",
                    target_strategy="alpha",
                    expires_at=datetime.now(timezone.utc) - timedelta(days=1),
                )
            )
            session.commit()
        finally:
            session.close()

        result = services._load_watchlists()
        assert "alpha" not in result

    def test_duplicate_symbol_from_injection_is_deduped(
        self, services_module_with_session
    ):
        services, factory = services_module_with_session
        session = factory()
        try:
            _seed_qualification(session, "AAA", "alpha", score=90.0)
            session.add(ManualInjection(symbol="AAA", target_strategy="alpha"))
            session.commit()
        finally:
            session.close()

        result = services._load_watchlists()
        assert result["alpha"].count("AAA") == 1
