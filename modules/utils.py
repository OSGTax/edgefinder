"""
EdgeFinder Utilities
====================
Shared helpers for timezone conversion and trade integrity.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone — used for all display timestamps
ET = ZoneInfo("America/New_York")


def to_eastern(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to Eastern Time ISO 8601 string.

    Assumes naive datetimes are UTC. Returns None for None input.
    Output includes UTC offset (e.g., '2025-03-24T10:30:00-04:00').

    Args:
        dt: A datetime object (timezone-aware or naive-UTC).

    Returns:
        ISO 8601 string in Eastern Time, or None.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).isoformat()


def compute_trade_hash(trade_data: dict, prev_hash: str = "") -> str:
    """Compute a SHA-256 integrity hash for a trade record.

    The hash covers the critical fields that determine trade honesty:
    trade_id, ticker, action, execution_price, shares, execution_timestamp,
    and strategy_name. It also chains to the previous trade's hash,
    so inserting, deleting, or reordering records breaks the chain.

    Args:
        trade_data: Dict with trade fields.
        prev_hash: The integrity_hash of the previous trade in the chain.
                   Empty string for the first trade.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    # Normalize execution_timestamp to ISO string for deterministic hashing
    exec_ts = trade_data.get("execution_timestamp")
    if isinstance(exec_ts, datetime):
        if exec_ts.tzinfo is None:
            exec_ts = exec_ts.replace(tzinfo=timezone.utc)
        exec_ts = exec_ts.isoformat()
    elif exec_ts is None:
        exec_ts = ""

    # Build the canonical string to hash
    parts = [
        str(trade_data.get("trade_id", "")),
        str(trade_data.get("ticker", "")),
        str(trade_data.get("action", "")),
        str(trade_data.get("execution_price", "")),
        str(trade_data.get("shares", "")),
        str(exec_ts),
        str(trade_data.get("strategy_name", "")),
        str(prev_hash),
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_hash_chain(trades: list[dict]) -> dict:
    """Verify the integrity of a sequence of trade records.

    Args:
        trades: List of trade dicts ordered by sequence_num ascending.
                Each must have 'integrity_hash' and the fields used by
                compute_trade_hash().

    Returns:
        Dict with:
            valid: bool — True if entire chain is intact
            total_trades: int — number of trades checked
            first_break_at: int or None — sequence_num where chain breaks
            message: str — human-readable summary
    """
    prev_hash = ""
    for i, trade in enumerate(trades):
        expected = compute_trade_hash(trade, prev_hash)
        stored = trade.get("integrity_hash", "")
        if expected != stored:
            seq = trade.get("sequence_num", i + 1)
            return {
                "valid": False,
                "total_trades": len(trades),
                "first_break_at": seq,
                "message": (
                    f"Chain broken at sequence {seq} "
                    f"(trade_id={trade.get('trade_id', '?')}, "
                    f"ticker={trade.get('ticker', '?')}). "
                    f"Expected hash {expected[:12]}..., "
                    f"got {stored[:12]}..."
                ),
            }
        prev_hash = stored

    return {
        "valid": True,
        "total_trades": len(trades),
        "first_break_at": None,
        "message": f"All {len(trades)} trades verified — chain intact.",
    }
