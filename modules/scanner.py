"""
EdgeFinder Module 1: Fundamental Scanner
=========================================
Screens US-listed stocks through Peter Lynch and Michael Burry criteria.
Produces a ranked watchlist of 50-100 candidates.

Runs nightly after market close (4:30 PM ET).
Rotates through sectors daily so the full universe is covered weekly.
Watchlist entries persist across scan days (7-day TTL).
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from config import settings
from modules.database import WatchlistStock, get_session, init_db

logger = logging.getLogger(__name__)


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class FundamentalData:
    """Raw fundamental data for a single stock."""
    ticker: str
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    price: float = 0.0

    # Lynch fields
    peg_ratio: Optional[float] = None
    earnings_growth: Optional[float] = None      # trailing EPS growth
    earnings_quarterly_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    institutional_pct: Optional[float] = None

    # Burry fields
    free_cashflow: Optional[float] = None
    fcf_yield: Optional[float] = None
    tangible_book_value: Optional[float] = None
    price_to_tangible_book: Optional[float] = None
    short_interest: Optional[float] = None       # short % of float
    ev_to_ebitda: Optional[float] = None
    current_ratio: Optional[float] = None

    # Meta
    avg_volume: Optional[float] = None
    fetch_errors: list = field(default_factory=list)


@dataclass
class ScoredStock:
    """A stock with Lynch, Burry, and composite scores."""
    data: FundamentalData
    lynch_score: float = 0.0
    burry_score: float = 0.0
    composite_score: float = 0.0
    lynch_category: str = "unknown"
    score_breakdown: dict = field(default_factory=dict)


# ── TICKER UNIVERSE ──────────────────────────────────────────

_data_service = None


def _init_data_service():
    """Lazy-init DataService for scanner use. Loads API keys from env."""
    global _data_service
    if _data_service is not None:
        return _data_service
    try:
        from services.data_service import DataService
        _data_service = DataService()
        return _data_service
    except Exception as e:
        logger.warning(f"DataService unavailable: {e}")
        return None


def _load_env():
    """Ensure environment variables from secrets.env are loaded."""
    secrets_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "secrets.env"
    )
    if os.path.exists(secrets_path):
        load_dotenv(secrets_path)


def get_todays_sectors() -> list[str]:
    """
    Return the sectors to scan today based on the day-of-week rotation.
    Returns None on weekends (no scan).
    """
    weekday = datetime.now().weekday()  # 0=Mon, 4=Fri
    rotation = settings.SCAN_SECTOR_ROTATION
    if weekday < len(rotation):
        return rotation[weekday]
    return []  # Weekend


def get_ticker_universe(sectors: list[str] | None = None) -> list[str]:
    """
    Get the universe of US-listed stock tickers to scan.

    Args:
        sectors: Optional list of sectors to filter by (e.g. ["Technology"]).
                 If None, returns full universe (all sectors).

    Priority:
    1. FMP screener with sector filter (when sectors specified) — via DataService
    2. Alpaca Assets API for full universe (when no sector filter) — via DataService
    3. FMP screener unfiltered — via DataService
    4. Wikipedia S&P lists
    5. Hardcoded fallback

    All FMP/Alpaca calls route through the shared DataService to ensure
    a single rate-limit counter and benefit from caching.
    """
    _load_env()
    tickers = set()
    ds = _init_data_service()

    # When sector-filtering, FMP screener is best (supports sector param)
    if sectors and ds and ds.fmp and ds.fmp.has_budget(len(sectors) * 3):
        try:
            for sector in sectors:
                for exchange in ["NYSE", "NASDAQ", "AMEX"]:
                    results = ds.fmp.get_stock_screener(
                        market_cap_min=300_000_000,
                        volume_min=500_000,
                        price_min=5.0,
                        price_max=500.0,
                        exchange=exchange,
                        sector=sector,
                        limit=3000,
                    )
                    if results:
                        symbols = [r["symbol"] for r in results if r.get("symbol")]
                        tickers.update(symbols)
                logger.info(f"FMP screener: {sector} → {len(tickers)} tickers so far")
            if tickers:
                logger.info(f"Sector scan universe: {len(tickers)} tickers for {sectors}")
                return sorted(tickers)
        except Exception as e:
            logger.warning(f"FMP sector screener failed: {e}")

    # Full universe: Alpaca Assets API (free, fast, ~8000+)
    if ds and ds.alpaca:
        try:
            assets = ds.alpaca.get_tradeable_assets(
                exchanges=["NYSE", "NASDAQ", "AMEX"]
            )
            if assets:
                tickers = {a["symbol"] for a in assets}
                logger.info(f"Alpaca universe: {len(tickers)} tradeable equities")
                return sorted(tickers)
        except Exception as e:
            logger.warning(f"Alpaca assets unavailable: {e}")
    else:
        # Try raw env vars as fallback if DataService didn't init Alpaca
        try:
            alpaca_key = os.getenv("ALPACA_API_KEY", "").strip()
            alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
            if alpaca_key and alpaca_secret:
                from services.alpaca_client import AlpacaClient
                alpaca = AlpacaClient(alpaca_key, alpaca_secret)
                assets = alpaca.get_tradeable_assets(
                    exchanges=["NYSE", "NASDAQ", "AMEX"]
                )
                if assets:
                    tickers = {a["symbol"] for a in assets}
                    logger.info(f"Alpaca universe: {len(tickers)} tradeable equities")
                    return sorted(tickers)
        except Exception as e:
            logger.warning(f"Alpaca assets unavailable: {e}")

    # Fallback: FMP screener unfiltered (only if budget allows)
    if ds and ds.fmp and ds.fmp.has_budget(3):
        try:
            for exchange in ["NYSE", "NASDAQ", "AMEX"]:
                results = ds.fmp.get_stock_screener(
                    market_cap_min=300_000_000,
                    volume_min=500_000,
                    price_min=5.0,
                    price_max=500.0,
                    exchange=exchange,
                    limit=5000,
                )
                if results:
                    symbols = [r["symbol"] for r in results if r.get("symbol")]
                    tickers.update(symbols)
            if tickers:
                logger.info(f"FMP universe: {len(tickers)} total tickers")
                return sorted(tickers)
        except Exception as e:
            logger.warning(f"FMP screener unavailable: {e}")

    # Fallback: Wikipedia S&P lists
    try:
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        tickers.update(sp500["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    try:
        sp400 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        )[0]
        tickers.update(sp400["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    try:
        sp600 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        )[0]
        col = "Symbol" if "Symbol" in sp600.columns else sp600.columns[0]
        tickers.update(sp600[col].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass

    if tickers:
        logger.info(f"Wikipedia universe: {len(tickers)} total tickers")
        return sorted(tickers)

    # Last resort
    logger.warning("Using hardcoded fallback ticker list")
    return sorted(set(settings.SCANNER_DEFAULT_TICKERS))


# ── DATA FETCHING ────────────────────────────────────────────

def fetch_fundamental_data(ticker: str, max_retries: int = 3) -> Optional[FundamentalData]:
    """
    Fetch fundamental data for a single ticker via DataService (FMP).

    Uses Cache → FMP pipeline. No yfinance — fast and deterministic.
    Returns None if FMP has no data for this ticker.
    """
    try:
        ds = _init_data_service()
        if ds is None:
            logger.warning(f"{ticker}: DataService unavailable")
            return None

        info = ds.get_fundamentals(ticker)
        if not info or not info.get("market_cap"):
            return None

        data = FundamentalData(ticker=ticker)

        # Basic info
        data.company_name = info.get("company_name", ticker)
        data.sector = info.get("sector", "Unknown")
        data.industry = info.get("industry", "Unknown")
        data.market_cap = _safe_float(info.get("market_cap"))
        data.price = _safe_float(info.get("price"))
        data.avg_volume = _safe_float(info.get("avg_volume"))

        # Lynch fields
        data.peg_ratio = _safe_float(info.get("peg_ratio"))
        data.earnings_growth = _safe_float(info.get("earnings_growth"))
        data.debt_to_equity = _safe_float(info.get("debt_to_equity"))
        data.revenue_growth = _safe_float(info.get("revenue_growth"))
        data.institutional_pct = _safe_float(info.get("institutional_pct"))

        # Burry fields
        data.fcf_yield = _safe_float(info.get("fcf_yield"))
        data.ev_to_ebitda = _safe_float(info.get("ev_to_ebitda"))
        data.current_ratio = _safe_float(info.get("current_ratio"))
        data.short_interest = _safe_float(info.get("short_interest"))
        data.price_to_tangible_book = _safe_float(info.get("price_to_tangible_book"))

        return data

    except Exception as e:
        logger.warning(f"{ticker}: Fetch error — {e}")
        return None


def fetch_batch(tickers: list[str], batch_size: int = 50) -> list[FundamentalData]:
    """
    Fetch fundamental data for a list of tickers via DataService (FMP + cache).
    Budget-aware: caps the number of tickers based on FMP requests remaining.
    Each ticker requires ~3 FMP calls (profile + metrics + ratios).
    """
    ds = _init_data_service()

    # Budget check: cap tickers to what FMP can handle
    if ds and ds.fmp:
        remaining = ds.fmp.requests_remaining
        max_tickers = remaining // 3  # 3 FMP calls per ticker
        if max_tickers < len(tickers):
            logger.warning(
                f"FMP budget: {remaining} requests remaining, "
                f"capping scan from {len(tickers)} to {max_tickers} tickers"
            )
            tickers = tickers[:max_tickers]
        logger.info(f"FMP budget: {remaining} remaining, scanning {len(tickers)} tickers")
    elif ds:
        logger.info("FMP not available — scanning with cache only")

    results = []
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = math.ceil(total / batch_size)
        logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} tickers)")

        for ticker in batch:
            data = fetch_fundamental_data(ticker)
            if data:
                results.append(data)

        logger.info(f"Progress: {len(results)} fetched so far ({i + len(batch)}/{total})")

    logger.info(f"Successfully fetched {len(results)}/{total} tickers")
    return results


# ── PRE-SCREENING FILTERS ───────────────────────────────────

def passes_prescreen(data: FundamentalData) -> bool:
    """
    Quick filter: does this stock meet basic requirements?
    Eliminates junk before we waste time scoring.
    """
    # Market cap
    if not data.market_cap or data.market_cap < settings.SCANNER_MIN_MARKET_CAP:
        return False
    if data.market_cap > settings.SCANNER_MAX_MARKET_CAP:
        return False

    # Price
    if not data.price or data.price < settings.SCANNER_MIN_PRICE:
        return False
    if data.price > settings.SCANNER_MAX_PRICE:
        return False

    # Volume
    if not data.avg_volume or data.avg_volume < settings.SCANNER_MIN_AVG_VOLUME:
        return False

    # Excluded sectors
    if data.sector in settings.SCANNER_EXCLUDED_SECTORS:
        return False

    return True


# ── LYNCH SCORING ────────────────────────────────────────────

def classify_lynch_category(data: FundamentalData) -> str:
    """
    Classify stock into Lynch's categories based on growth characteristics.
    Categories: slow_grower, stalwart, fast_grower, cyclical, turnaround, asset_play
    """
    eg = data.earnings_growth
    rg = data.revenue_growth

    if eg is None and rg is None:
        return "unclassified"

    growth = eg if eg is not None else (rg if rg is not None else 0)

    # Turnaround: negative earnings but improving
    if eg is not None and eg < 0 and data.earnings_quarterly_growth is not None:
        if data.earnings_quarterly_growth > 0:
            return "turnaround"

    # Growth categories (checked before asset_play so growth dominates)
    if growth >= 0.25:
        return "fast_grower"
    elif growth >= 0.10:
        return "stalwart"

    # Asset play: trading below tangible book (only if not a clear grower)
    if data.price_to_tangible_book is not None and data.price_to_tangible_book < 1.0:
        return "asset_play"

    if growth >= 0:
        return "slow_grower"
    else:
        return "cyclical"


def score_lynch(data: FundamentalData) -> tuple[float, str, dict]:
    """
    Score a stock on Peter Lynch criteria (0-100).
    Returns: (score, category, breakdown_dict)
    """
    breakdown = {}
    total = 0.0

    # PEG Ratio (25%)
    if data.peg_ratio is not None and data.peg_ratio > 0:
        if data.peg_ratio <= settings.LYNCH_PEG_IDEAL:
            peg_score = 100.0
        elif data.peg_ratio <= settings.LYNCH_PEG_MAX:
            # Linear interpolation between ideal and max
            peg_score = 100.0 * (settings.LYNCH_PEG_MAX - data.peg_ratio) / (
                settings.LYNCH_PEG_MAX - settings.LYNCH_PEG_IDEAL
            )
        else:
            peg_score = 0.0
        breakdown["peg"] = round(peg_score, 1)
        total += peg_score * settings.LYNCH_PEG_WEIGHT
    else:
        breakdown["peg"] = None

    # Earnings Growth (20%)
    if data.earnings_growth is not None:
        if data.earnings_growth >= settings.LYNCH_EARNINGS_GROWTH_MIN * 2:
            eg_score = 100.0
        elif data.earnings_growth >= settings.LYNCH_EARNINGS_GROWTH_MIN:
            eg_score = 50.0 + 50.0 * (
                (data.earnings_growth - settings.LYNCH_EARNINGS_GROWTH_MIN) /
                settings.LYNCH_EARNINGS_GROWTH_MIN
            )
        elif data.earnings_growth > 0:
            eg_score = 50.0 * (data.earnings_growth / settings.LYNCH_EARNINGS_GROWTH_MIN)
        else:
            eg_score = 0.0
        breakdown["earnings_growth"] = round(eg_score, 1)
        total += eg_score * settings.LYNCH_EARNINGS_GROWTH_WEIGHT
    else:
        breakdown["earnings_growth"] = None

    # Debt-to-Equity (15%)
    if data.debt_to_equity is not None:
        if data.debt_to_equity <= settings.LYNCH_DEBT_TO_EQUITY_PREFERRED:
            de_score = 100.0
        elif data.debt_to_equity <= settings.LYNCH_DEBT_TO_EQUITY_MAX:
            de_score = 100.0 * (settings.LYNCH_DEBT_TO_EQUITY_MAX - data.debt_to_equity) / (
                settings.LYNCH_DEBT_TO_EQUITY_MAX - settings.LYNCH_DEBT_TO_EQUITY_PREFERRED
            )
        else:
            de_score = 0.0
        breakdown["debt_to_equity"] = round(de_score, 1)
        total += de_score * settings.LYNCH_DEBT_TO_EQUITY_WEIGHT
    else:
        breakdown["debt_to_equity"] = None

    # Revenue Growth (15%)
    if data.revenue_growth is not None:
        if data.revenue_growth >= settings.LYNCH_REVENUE_GROWTH_MIN * 2:
            rg_score = 100.0
        elif data.revenue_growth >= settings.LYNCH_REVENUE_GROWTH_MIN:
            rg_score = 50.0 + 50.0 * (
                (data.revenue_growth - settings.LYNCH_REVENUE_GROWTH_MIN) /
                settings.LYNCH_REVENUE_GROWTH_MIN
            )
        elif data.revenue_growth > 0:
            rg_score = 50.0 * (data.revenue_growth / settings.LYNCH_REVENUE_GROWTH_MIN)
        else:
            rg_score = 0.0
        breakdown["revenue_growth"] = round(rg_score, 1)
        total += rg_score * settings.LYNCH_REVENUE_GROWTH_WEIGHT
    else:
        breakdown["revenue_growth"] = None

    # Institutional Ownership (10%)
    if data.institutional_pct is not None:
        if settings.LYNCH_INSTITUTIONAL_MIN <= data.institutional_pct <= settings.LYNCH_INSTITUTIONAL_MAX:
            inst_score = 100.0
        elif data.institutional_pct < settings.LYNCH_INSTITUTIONAL_MIN:
            inst_score = 100.0 * data.institutional_pct / settings.LYNCH_INSTITUTIONAL_MIN
        else:
            # Overcrowded — penalize steeply
            over = data.institutional_pct - settings.LYNCH_INSTITUTIONAL_MAX
            inst_score = max(0, 100.0 - (over * 500))  # drops fast above 70%
        breakdown["institutional"] = round(inst_score, 1)
        total += inst_score * settings.LYNCH_INSTITUTIONAL_WEIGHT
    else:
        breakdown["institutional"] = None

    # Category (15%)
    category = classify_lynch_category(data)
    cat_scores = {
        "fast_grower": 100, "stalwart": 75, "turnaround": 60,
        "asset_play": 50, "cyclical": 30, "slow_grower": 20, "unclassified": 10
    }
    cat_score = cat_scores.get(category, 10)
    breakdown["category"] = cat_score
    total += cat_score * settings.LYNCH_CATEGORY_WEIGHT

    # Normalize: account for missing fields by scaling up
    available_weight = sum(
        w for name, w in [
            ("peg", settings.LYNCH_PEG_WEIGHT),
            ("earnings_growth", settings.LYNCH_EARNINGS_GROWTH_WEIGHT),
            ("debt_to_equity", settings.LYNCH_DEBT_TO_EQUITY_WEIGHT),
            ("revenue_growth", settings.LYNCH_REVENUE_GROWTH_WEIGHT),
            ("institutional", settings.LYNCH_INSTITUTIONAL_WEIGHT),
        ]
        if breakdown.get(name) is not None
    ) + settings.LYNCH_CATEGORY_WEIGHT  # Category is always available

    if available_weight > 0:
        score = total / available_weight
    else:
        score = 0

    return min(100, max(0, score)), category, breakdown


# ── BURRY SCORING ────────────────────────────────────────────

def score_burry(data: FundamentalData) -> tuple[float, dict]:
    """
    Score a stock on Michael Burry criteria (0-100).
    Returns: (score, breakdown_dict)
    """
    breakdown = {}
    total = 0.0

    # Free Cash Flow Yield (30%)
    if data.fcf_yield is not None:
        if data.fcf_yield >= settings.BURRY_FCF_YIELD_STRONG:
            fcf_score = 100.0
        elif data.fcf_yield >= settings.BURRY_FCF_YIELD_ACCEPTABLE:
            fcf_score = 50.0 + 50.0 * (
                (data.fcf_yield - settings.BURRY_FCF_YIELD_ACCEPTABLE) /
                (settings.BURRY_FCF_YIELD_STRONG - settings.BURRY_FCF_YIELD_ACCEPTABLE)
            )
        elif data.fcf_yield > 0:
            fcf_score = 50.0 * (data.fcf_yield / settings.BURRY_FCF_YIELD_ACCEPTABLE)
        else:
            fcf_score = 0.0
        breakdown["fcf_yield"] = round(fcf_score, 1)
        total += fcf_score * settings.BURRY_FCF_YIELD_WEIGHT
    else:
        breakdown["fcf_yield"] = None

    # Price to Tangible Book (25%)
    if data.price_to_tangible_book is not None and data.price_to_tangible_book > 0:
        if data.price_to_tangible_book <= settings.BURRY_PRICE_TO_TANGIBLE_BOOK_DEEP:
            ptb_score = 100.0
        elif data.price_to_tangible_book <= settings.BURRY_PRICE_TO_TANGIBLE_BOOK_VALUE:
            ptb_score = 50.0 + 50.0 * (
                (settings.BURRY_PRICE_TO_TANGIBLE_BOOK_VALUE - data.price_to_tangible_book) /
                (settings.BURRY_PRICE_TO_TANGIBLE_BOOK_VALUE - settings.BURRY_PRICE_TO_TANGIBLE_BOOK_DEEP)
            )
        else:
            ptb_score = max(0, 50.0 - (data.price_to_tangible_book - settings.BURRY_PRICE_TO_TANGIBLE_BOOK_VALUE) * 10)
        breakdown["price_to_tangible_book"] = round(ptb_score, 1)
        total += ptb_score * settings.BURRY_PRICE_TO_TANGIBLE_BOOK_WEIGHT
    else:
        breakdown["price_to_tangible_book"] = None

    # Short Interest (15%) — contrarian signal
    if data.short_interest is not None:
        if data.short_interest >= settings.BURRY_SHORT_INTEREST_CONTRARIAN:
            si_score = 80.0  # High short interest is interesting, not 100% positive
        elif data.short_interest >= 0.05:
            si_score = 50.0
        else:
            si_score = 30.0  # Low short interest is neutral
        breakdown["short_interest"] = round(si_score, 1)
        total += si_score * settings.BURRY_SHORT_INTEREST_WEIGHT
    else:
        breakdown["short_interest"] = None

    # EV/EBITDA (15%)
    if data.ev_to_ebitda is not None and data.ev_to_ebitda > 0:
        if data.ev_to_ebitda <= settings.BURRY_EV_TO_EBITDA_MAX:
            ev_score = 100.0 * (1 - data.ev_to_ebitda / (settings.BURRY_EV_TO_EBITDA_MAX * 2))
            ev_score = max(50, ev_score)
        else:
            ev_score = max(0, 50.0 - (data.ev_to_ebitda - settings.BURRY_EV_TO_EBITDA_MAX) * 5)
        breakdown["ev_to_ebitda"] = round(ev_score, 1)
        total += ev_score * settings.BURRY_EV_TO_EBITDA_WEIGHT
    else:
        breakdown["ev_to_ebitda"] = None

    # Current Ratio (15%)
    if data.current_ratio is not None:
        if data.current_ratio >= settings.BURRY_CURRENT_RATIO_MIN:
            cr_score = min(100, 60.0 + 40.0 * (
                (data.current_ratio - settings.BURRY_CURRENT_RATIO_MIN) / settings.BURRY_CURRENT_RATIO_MIN
            ))
        elif data.current_ratio >= 1.0:
            cr_score = 40.0
        else:
            cr_score = max(0, 40.0 * data.current_ratio)
        breakdown["current_ratio"] = round(cr_score, 1)
        total += cr_score * settings.BURRY_CURRENT_RATIO_WEIGHT
    else:
        breakdown["current_ratio"] = None

    # Normalize for missing fields
    available_weight = sum(
        w for name, w in [
            ("fcf_yield", settings.BURRY_FCF_YIELD_WEIGHT),
            ("price_to_tangible_book", settings.BURRY_PRICE_TO_TANGIBLE_BOOK_WEIGHT),
            ("short_interest", settings.BURRY_SHORT_INTEREST_WEIGHT),
            ("ev_to_ebitda", settings.BURRY_EV_TO_EBITDA_WEIGHT),
            ("current_ratio", settings.BURRY_CURRENT_RATIO_WEIGHT),
        ]
        if breakdown.get(name) is not None
    )

    if available_weight > 0:
        score = total / available_weight
    else:
        score = 0

    return min(100, max(0, score)), breakdown


# ── COMPOSITE SCORING ────────────────────────────────────────

def score_stock(data: FundamentalData) -> ScoredStock:
    """Score a stock with both Lynch and Burry frameworks and compute composite."""
    lynch_score, category, lynch_breakdown = score_lynch(data)
    burry_score, burry_breakdown = score_burry(data)

    composite = (lynch_score * settings.LYNCH_WEIGHT) + (burry_score * settings.BURRY_WEIGHT)

    return ScoredStock(
        data=data,
        lynch_score=round(lynch_score, 2),
        burry_score=round(burry_score, 2),
        composite_score=round(composite, 2),
        lynch_category=category,
        score_breakdown={"lynch": lynch_breakdown, "burry": burry_breakdown},
    )


# ── FULL SCAN PIPELINE ──────────────────────────────────────

def run_scan(
    tickers: list[str] | None = None,
    sectors: list[str] | None = None,
    save_to_db: bool = True,
    verbose: bool = False,
) -> list[ScoredStock]:
    """
    Run the full fundamental scan pipeline.

    1. Get ticker universe (or use provided list), optionally filtered by sector
    2. Fetch fundamental data for each
    3. Pre-screen (market cap, price, volume filters)
    4. Score each stock (Lynch + Burry)
    5. Rank by composite score
    6. Save top candidates to database (appends, doesn't replace other sectors)

    Args:
        tickers: Explicit list of tickers. If None, uses get_ticker_universe().
        sectors: Sector filter for universe. E.g. ["Technology", "Healthcare"].
        save_to_db: Persist results to database.
        verbose: Print detailed output.

    Returns: List of ScoredStock objects, sorted by composite score descending.
    """
    logger.info("=" * 60)
    logger.info("EDGEFINDER FUNDAMENTAL SCAN STARTING")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Step 1: Get tickers
    if tickers is None:
        tickers = get_ticker_universe(sectors=sectors)

    # Cap universe to what FMP budget allows (3 calls per ticker)
    ds = _init_data_service()
    if ds and ds.fmp:
        max_scannable = ds.fmp.requests_remaining // 3
        if len(tickers) > max_scannable:
            logger.info(
                f"Capping universe from {len(tickers)} to {max_scannable} tickers "
                f"(FMP budget: {ds.fmp.requests_remaining} remaining)"
            )
            tickers = tickers[:max_scannable]

    if sectors:
        logger.info(f"Scanning {len(tickers)} tickers (sectors: {', '.join(sectors)})")
    else:
        logger.info(f"Scanning {len(tickers)} tickers (full universe)")

    # Step 2: Fetch data
    all_data = fetch_batch(tickers)
    logger.info(f"Fetched data for {len(all_data)} tickers")

    # Step 3: Pre-screen
    screened = [d for d in all_data if passes_prescreen(d)]
    logger.info(f"Passed pre-screen: {len(screened)} tickers "
                f"(filtered {len(all_data) - len(screened)})")

    # Step 4: Score
    scored = [score_stock(d) for d in screened]
    logger.info(f"Scored {len(scored)} stocks")

    # Step 5: Filter and rank
    watchlist = [
        s for s in scored
        if s.composite_score >= settings.WATCHLIST_MIN_COMPOSITE_SCORE
    ]
    watchlist.sort(key=lambda s: s.composite_score, reverse=True)
    watchlist = watchlist[:settings.WATCHLIST_MAX_SIZE]
    logger.info(f"Watchlist candidates: {len(watchlist)} "
                f"(min score: {settings.WATCHLIST_MIN_COMPOSITE_SCORE})")

    # Step 6: Save to database
    if save_to_db and watchlist:
        _save_watchlist(watchlist)

    # Summary
    if watchlist:
        logger.info("-" * 60)
        logger.info("TOP 10 CANDIDATES:")
        for i, s in enumerate(watchlist[:10], 1):
            logger.info(
                f"  {i:>2}. {s.data.ticker:<6} | "
                f"Composite: {s.composite_score:>5.1f} | "
                f"Lynch: {s.lynch_score:>5.1f} | "
                f"Burry: {s.burry_score:>5.1f} | "
                f"Cat: {s.lynch_category:<12} | "
                f"${s.data.price:>8.2f}"
            )
        logger.info("-" * 60)

    logger.info("SCAN COMPLETE")
    return watchlist


# ── DATABASE PERSISTENCE ─────────────────────────────────────

def _build_notes(s: ScoredStock) -> str:
    """Build plain English scoring summary using actual data values."""
    parts = []
    d = s.data

    # Category in plain English
    cat_names = {
        "fast_grower": "Fast grower", "stalwart": "Stalwart",
        "turnaround": "Turnaround", "asset_play": "Asset play",
        "cyclical": "Cyclical", "slow_grower": "Slow grower",
        "unclassified": "Unclassified",
    }
    parts.append(cat_names.get(s.lynch_category, s.lynch_category))

    if d.peg_ratio is not None:
        qual = "great" if d.peg_ratio <= 1.0 else "ok" if d.peg_ratio <= 1.5 else "high"
        parts.append(f"PEG {d.peg_ratio:.1f} ({qual})")
    if d.earnings_growth is not None:
        parts.append(f"Earnings {d.earnings_growth:+.0%}/yr")
    if d.revenue_growth is not None:
        parts.append(f"Revenue {d.revenue_growth:+.0%}/yr")
    if d.debt_to_equity is not None:
        qual = "low" if d.debt_to_equity <= 0.5 else "moderate" if d.debt_to_equity <= 1.0 else "high"
        parts.append(f"D/E {d.debt_to_equity:.1f} ({qual})")
    if d.fcf_yield is not None:
        qual = "strong" if d.fcf_yield >= 0.08 else "good" if d.fcf_yield >= 0.05 else "weak"
        parts.append(f"FCF yield {d.fcf_yield:.1%} ({qual})")
    if d.ev_to_ebitda is not None:
        qual = "cheap" if d.ev_to_ebitda <= 8 else "fair" if d.ev_to_ebitda <= 15 else "expensive"
        parts.append(f"EV/EBITDA {d.ev_to_ebitda:.1f} ({qual})")
    if d.current_ratio is not None:
        qual = "healthy" if d.current_ratio >= 1.5 else "ok" if d.current_ratio >= 1.0 else "tight"
        parts.append(f"Current ratio {d.current_ratio:.1f} ({qual})")

    return ". ".join(parts) + "."


def _save_watchlist(watchlist: list[ScoredStock]):
    """
    Save watchlist to database. Only deactivates tickers being re-scanned,
    preserving entries from other sector rotations. Entries older than 7 days
    are deactivated automatically (stale data).
    """
    try:
        session = get_session()

        # Expire stale entries older than 7 days
        from sqlalchemy import or_
        cutoff = datetime.utcnow() - timedelta(days=7)
        stale_count = session.query(WatchlistStock).filter(
            WatchlistStock.is_active == True,  # noqa: E712
            WatchlistStock.scan_date < cutoff,
            or_(
                WatchlistStock.notes == None,  # noqa: E711
                ~WatchlistStock.notes.like("MANUAL%"),
            ),
        ).update({"is_active": False}, synchronize_session="fetch")
        if stale_count:
            logger.info(f"Expired {stale_count} watchlist entries older than 7 days")

        # Deactivate only tickers we're about to re-scan (not the whole list)
        scanned_tickers = [s.data.ticker for s in watchlist]
        if scanned_tickers:
            session.query(WatchlistStock).filter(
                WatchlistStock.is_active == True,  # noqa: E712
                WatchlistStock.ticker.in_(scanned_tickers),
                or_(
                    WatchlistStock.notes == None,  # noqa: E711
                    ~WatchlistStock.notes.like("MANUAL%"),
                ),
            ).update({"is_active": False}, synchronize_session="fetch")

        # Insert new watchlist entries
        for s in watchlist:
            entry = WatchlistStock(
                ticker=s.data.ticker,
                company_name=s.data.company_name,
                sector=s.data.sector,
                industry=s.data.industry,
                market_cap=s.data.market_cap,
                price=s.data.price,
                peg_ratio=s.data.peg_ratio,
                earnings_growth=s.data.earnings_growth,
                debt_to_equity=s.data.debt_to_equity,
                revenue_growth=s.data.revenue_growth,
                institutional_pct=s.data.institutional_pct,
                lynch_category=s.lynch_category,
                lynch_score=s.lynch_score,
                fcf_yield=s.data.fcf_yield,
                price_to_tangible_book=s.data.price_to_tangible_book,
                short_interest=s.data.short_interest,
                ev_to_ebitda=s.data.ev_to_ebitda,
                current_ratio=s.data.current_ratio,
                burry_score=s.burry_score,
                composite_score=s.composite_score,
                scan_date=datetime.utcnow(),
                is_active=True,
                notes=_build_notes(s),
            )
            session.add(entry)

        session.commit()
        logger.info(f"Saved {len(watchlist)} stocks to watchlist (persistent across sectors)")
    except Exception as e:
        logger.error(f"Failed to save watchlist: {e}")
        session.rollback()
    finally:
        session.close()


# ── UTILITIES ────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float, handling None/NaN/Inf."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def get_active_watchlist() -> list[dict]:
    """Get the current active watchlist from the database."""
    session = get_session()
    try:
        stocks = session.query(WatchlistStock).filter(
            WatchlistStock.is_active == True  # noqa: E712
        ).order_by(WatchlistStock.composite_score.desc()).all()

        return [
            {
                "ticker": s.ticker,
                "company_name": s.company_name,
                "sector": s.sector,
                "composite_score": s.composite_score,
                "lynch_score": s.lynch_score,
                "burry_score": s.burry_score,
                "lynch_category": s.lynch_category,
                "price": s.price,
                "market_cap": s.market_cap,
                "scan_date": s.scan_date.isoformat() if s.scan_date else None,
            }
            for s in stocks
        ]
    finally:
        session.close()
