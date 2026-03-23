"""
EdgeFinder Test Fixtures (conftest.py)
======================================
Shared fixtures for all test modules.
Provides mock data so tests don't hit real APIs.
"""

import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import init_db, reset_engine
from modules.scanner import FundamentalData


@pytest.fixture(autouse=True)
def reset_db_engine():
    """Reset the database engine before each test to avoid state leakage."""
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
def in_memory_db():
    """Create a fresh in-memory database for testing."""
    reset_engine()
    engine = init_db(":memory:")
    yield engine
    reset_engine()


@pytest.fixture
def sample_strong_stock() -> FundamentalData:
    """A stock that should score very high on both Lynch and Burry criteria."""
    return FundamentalData(
        ticker="GREAT",
        company_name="Great Company Inc",
        sector="Technology",
        industry="Software",
        market_cap=5_000_000_000,
        price=50.00,
        avg_volume=2_000_000,
        peg_ratio=0.8,
        earnings_growth=0.25,
        earnings_quarterly_growth=0.30,
        debt_to_equity=0.3,
        revenue_growth=0.20,
        institutional_pct=0.45,
        free_cashflow=500_000_000,
        fcf_yield=0.10,
        price_to_tangible_book=0.9,
        short_interest=0.18,
        ev_to_ebitda=5.0,
        current_ratio=2.5,
    )


@pytest.fixture
def sample_weak_stock() -> FundamentalData:
    """A stock that should score very low on both criteria."""
    return FundamentalData(
        ticker="WEAK",
        company_name="Weak Company Inc",
        sector="Consumer Cyclical",
        industry="Retail",
        market_cap=1_000_000_000,
        price=15.00,
        avg_volume=800_000,
        peg_ratio=3.0,
        earnings_growth=-0.10,
        earnings_quarterly_growth=-0.05,
        debt_to_equity=2.5,
        revenue_growth=-0.05,
        institutional_pct=0.85,
        free_cashflow=-50_000_000,
        fcf_yield=-0.05,
        price_to_tangible_book=5.0,
        short_interest=0.02,
        ev_to_ebitda=25.0,
        current_ratio=0.6,
    )


@pytest.fixture
def sample_moderate_stock() -> FundamentalData:
    """A stock that should score moderately (around 50-65)."""
    return FundamentalData(
        ticker="MODS",
        company_name="Moderate Stock Corp",
        sector="Healthcare",
        industry="Biotechnology",
        market_cap=8_000_000_000,
        price=75.00,
        avg_volume=1_500_000,
        peg_ratio=1.2,
        earnings_growth=0.12,
        debt_to_equity=0.7,
        revenue_growth=0.08,
        institutional_pct=0.55,
        free_cashflow=400_000_000,
        fcf_yield=0.05,
        price_to_tangible_book=1.5,
        short_interest=0.08,
        ev_to_ebitda=10.0,
        current_ratio=1.8,
    )


@pytest.fixture
def sample_missing_data_stock() -> FundamentalData:
    """A stock with many None fields (common with yfinance)."""
    return FundamentalData(
        ticker="MISS",
        company_name="Missing Data Inc",
        sector="Industrials",
        industry="Machinery",
        market_cap=2_000_000_000,
        price=30.00,
        avg_volume=600_000,
        peg_ratio=None,
        earnings_growth=0.18,
        debt_to_equity=None,
        revenue_growth=None,
        institutional_pct=None,
        free_cashflow=None,
        fcf_yield=None,
        price_to_tangible_book=None,
        short_interest=None,
        ev_to_ebitda=7.0,
        current_ratio=2.0,
    )


@pytest.fixture
def sample_penny_stock() -> FundamentalData:
    """Should be filtered out by pre-screening."""
    return FundamentalData(
        ticker="PENY",
        company_name="Penny Corp",
        sector="Technology",
        industry="Software",
        market_cap=50_000_000,
        price=2.50,
        avg_volume=100_000,
    )


@pytest.fixture
def sample_utility_stock() -> FundamentalData:
    """Should be filtered out by excluded sectors."""
    return FundamentalData(
        ticker="UTIL",
        company_name="Utility Corp",
        sector="Utilities",
        industry="Electric Utilities",
        market_cap=10_000_000_000,
        price=45.00,
        avg_volume=3_000_000,
    )


@pytest.fixture
def multiple_stocks(
    sample_strong_stock,
    sample_weak_stock,
    sample_moderate_stock,
    sample_missing_data_stock,
):
    """A list of stocks for batch testing."""
    return [
        sample_strong_stock,
        sample_weak_stock,
        sample_moderate_stock,
        sample_missing_data_stock,
    ]
