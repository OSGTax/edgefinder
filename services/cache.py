"""
Data Cache Layer
================
SQLite-backed cache for market data. Lives in data/cache.db — completely
separate from trade data in data/edgefinder.db.

Cache can be deleted and rebuilt at any time without losing trade history.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, event, Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

CacheBase = declarative_base()


class CachedBar(CacheBase):
    """Cached OHLCV bar data."""
    __tablename__ = "cached_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1Min, 5Min, 1Day, etc.
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    vwap = Column(Float)
    trade_count = Column(Integer)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_bars_ticker_tf_ts", "ticker", "timeframe", "timestamp", unique=True),
    )


class CachedFundamental(CacheBase):
    """Cached fundamental data (profile, metrics, ratios)."""
    __tablename__ = "cached_fundamentals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    data_type = Column(String(30), nullable=False)  # profile, metrics, ratios, etc.
    data_json = Column(Text, nullable=False)  # JSON-serialized data
    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_fund_ticker_type", "ticker", "data_type", unique=True),
    )


# Default TTLs (how long cached data stays fresh)
DEFAULT_BAR_TTL = {
    "1Min": timedelta(minutes=5),
    "5Min": timedelta(minutes=15),
    "15Min": timedelta(minutes=30),
    "1Hour": timedelta(hours=2),
    "1Day": timedelta(hours=18),  # Daily bars refresh once per day
}

DEFAULT_FUNDAMENTAL_TTL = {
    "profile": timedelta(days=7),
    "metrics": timedelta(days=1),
    "ratios": timedelta(days=1),
    "income_statement": timedelta(days=7),
    "balance_sheet": timedelta(days=7),
    "cash_flow": timedelta(days=7),
    "enterprise_value": timedelta(days=1),
    "screener": timedelta(hours=12),
    "earnings_calendar": timedelta(hours=6),
}


class DataCache:
    """SQLite-backed cache for market data."""

    def __init__(self, cache_path: str = "data/cache.db"):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)

        if cache_path == ":memory:":
            self._engine = create_engine(
                "sqlite:///:memory:",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            self._engine = create_engine(
                f"sqlite:///{cache_path}",
                connect_args={"check_same_thread": False},
            )

        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        CacheBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info(f"Data cache initialized at {cache_path}")

    def _get_session(self) -> Session:
        return self._Session()

    # ── BAR CACHE ───────────────────────────────────────────

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached bars if they exist and are fresh.

        Returns DataFrame or None if cache miss.
        """
        ttl = DEFAULT_BAR_TTL.get(timeframe, timedelta(hours=1))
        cutoff = datetime.utcnow() - ttl

        session = self._get_session()
        try:
            query = session.query(CachedBar).filter(
                CachedBar.ticker == ticker,
                CachedBar.timeframe == timeframe,
                CachedBar.fetched_at >= cutoff,
            )
            if start:
                query = query.filter(CachedBar.timestamp >= start)
            if end:
                query = query.filter(CachedBar.timestamp <= end)

            query = query.order_by(CachedBar.timestamp)
            rows = query.all()

            if not rows:
                return None

            data = [{
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "vwap": r.vwap,
                "trade_count": r.trade_count,
            } for r in rows]

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            logger.debug(f"Cache hit: {len(df)} bars for {ticker} ({timeframe})")
            return df

        finally:
            session.close()

    def store_bars(self, ticker: str, timeframe: str, df: pd.DataFrame) -> int:
        """
        Store bars in the cache. Upserts (replaces existing bars for same timestamp).

        Returns number of bars stored.
        """
        if df is None or df.empty:
            return 0

        session = self._get_session()
        now = datetime.utcnow()
        count = 0

        try:
            for ts, row in df.iterrows():
                ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

                # Upsert: delete existing then insert
                session.query(CachedBar).filter(
                    CachedBar.ticker == ticker,
                    CachedBar.timeframe == timeframe,
                    CachedBar.timestamp == ts_dt,
                ).delete()

                bar = CachedBar(
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=ts_dt,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                    vwap=float(row.get("vwap", 0)) if pd.notna(row.get("vwap")) else None,
                    trade_count=int(row.get("trade_count", 0)) if pd.notna(row.get("trade_count")) else None,
                    fetched_at=now,
                )
                session.add(bar)
                count += 1

            session.commit()
            logger.debug(f"Cached {count} bars for {ticker} ({timeframe})")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Error caching bars for {ticker}: {e}")
            return 0
        finally:
            session.close()

    # ── FUNDAMENTAL CACHE ───────────────────────────────────

    def get_fundamental(self, ticker: str, data_type: str) -> Optional[dict | list]:
        """
        Get cached fundamental data if fresh.

        Returns parsed JSON data or None if cache miss.
        """
        ttl = DEFAULT_FUNDAMENTAL_TTL.get(data_type, timedelta(days=1))
        cutoff = datetime.utcnow() - ttl

        session = self._get_session()
        try:
            row = session.query(CachedFundamental).filter(
                CachedFundamental.ticker == ticker,
                CachedFundamental.data_type == data_type,
                CachedFundamental.fetched_at >= cutoff,
            ).first()

            if row:
                logger.debug(f"Cache hit: {data_type} for {ticker}")
                return json.loads(row.data_json)
            return None

        finally:
            session.close()

    def store_fundamental(self, ticker: str, data_type: str, data: dict | list) -> bool:
        """Store fundamental data in cache. Upserts."""
        session = self._get_session()
        try:
            # Delete existing
            session.query(CachedFundamental).filter(
                CachedFundamental.ticker == ticker,
                CachedFundamental.data_type == data_type,
            ).delete()

            entry = CachedFundamental(
                ticker=ticker,
                data_type=data_type,
                data_json=json.dumps(data),
                fetched_at=datetime.utcnow(),
            )
            session.add(entry)
            session.commit()
            logger.debug(f"Cached {data_type} for {ticker}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error caching {data_type} for {ticker}: {e}")
            return False
        finally:
            session.close()

    # ── CACHE MANAGEMENT ────────────────────────────────────

    def clear_ticker(self, ticker: str) -> int:
        """Clear all cached data for a ticker. Returns rows deleted."""
        session = self._get_session()
        try:
            bars = session.query(CachedBar).filter(CachedBar.ticker == ticker).delete()
            funds = session.query(CachedFundamental).filter(
                CachedFundamental.ticker == ticker
            ).delete()
            session.commit()
            total = bars + funds
            logger.info(f"Cleared {total} cached entries for {ticker}")
            return total
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing cache for {ticker}: {e}")
            return 0
        finally:
            session.close()

    def clear_stale(self) -> int:
        """Remove all expired cache entries. Returns rows deleted."""
        session = self._get_session()
        total = 0
        try:
            # Clear stale bars
            for tf, ttl in DEFAULT_BAR_TTL.items():
                cutoff = datetime.utcnow() - ttl
                count = session.query(CachedBar).filter(
                    CachedBar.timeframe == tf,
                    CachedBar.fetched_at < cutoff,
                ).delete()
                total += count

            # Clear stale fundamentals
            for dt, ttl in DEFAULT_FUNDAMENTAL_TTL.items():
                cutoff = datetime.utcnow() - ttl
                count = session.query(CachedFundamental).filter(
                    CachedFundamental.data_type == dt,
                    CachedFundamental.fetched_at < cutoff,
                ).delete()
                total += count

            session.commit()
            if total > 0:
                logger.info(f"Cleared {total} stale cache entries")
            return total

        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing stale cache: {e}")
            return 0
        finally:
            session.close()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        session = self._get_session()
        try:
            bar_count = session.query(CachedBar).count()
            fund_count = session.query(CachedFundamental).count()
            bar_tickers = session.query(CachedBar.ticker).distinct().count()
            fund_tickers = session.query(CachedFundamental.ticker).distinct().count()

            return {
                "cached_bars": bar_count,
                "cached_fundamentals": fund_count,
                "bar_tickers": bar_tickers,
                "fundamental_tickers": fund_tickers,
            }
        finally:
            session.close()

    def clear_all(self) -> int:
        """
        Nuclear option: clear entire cache. Trade data is NOT affected.
        Only call this if cache is corrupted or you want a fresh start.
        """
        session = self._get_session()
        try:
            bars = session.query(CachedBar).delete()
            funds = session.query(CachedFundamental).delete()
            session.commit()
            total = bars + funds
            logger.warning(f"Cleared entire cache: {total} entries")
            return total
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing cache: {e}")
            return 0
        finally:
            session.close()
