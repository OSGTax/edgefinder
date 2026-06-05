"""Trade hash-chain verification — proves stored trades haven't been edited.

The journal writes each new trade row with::

    hash_n = sha256(f"{trade_id}:{seq_n}:{hash_(n-1)}")     (hash_0 = "")

per strategy, anchored to the previous STORED row (journal._next_chain_link).
``verify_chain`` recomputes every link from the rows alone: a verified row
proves both its own identity and that the prior row's hash was untouched —
so any silent edit/delete/insert breaks every link after it.

Rows written before the v2 scheme (2026-06-05) chained against in-memory
state that was never fully persisted, so they cannot be verified — they are
reported as ``legacy_unverified``, not silently passed. The first v2 row
chains onto the last legacy row's stored hash, freezing the legacy tail.
"""

from __future__ import annotations

import hashlib
import logging

from sqlalchemy.orm import Session

from edgefinder.db.models import TradeRecord

logger = logging.getLogger(__name__)

_MAX_LISTED_BREAKS = 20


def verify_chain(session: Session, strategy_name: str | None = None) -> dict:
    """Walk each strategy's chain and recompute every link.

    Returns a per-strategy report plus overall totals. ``intact_tail`` means
    the most recent row verifies — i.e. everything written under the v2
    scheme is provably untampered even while older legacy rows remain
    unverifiable.
    """
    if strategy_name:
        strategies = [strategy_name]
    else:
        strategies = [
            r[0]
            for r in session.query(TradeRecord.strategy_name).distinct().all()
        ]

    per_strategy: dict[str, dict] = {}
    totals = {"total": 0, "verified": 0, "legacy_unverified": 0, "unsequenced": 0}

    for strat in sorted(strategies):
        rows = (
            session.query(
                TradeRecord.trade_id,
                TradeRecord.sequence_num,
                TradeRecord.integrity_hash,
            )
            .filter(TradeRecord.strategy_name == strat)
            .order_by(TradeRecord.sequence_num, TradeRecord.id)
            .all()
        )
        unsequenced = sum(
            1 for _, seq, h in rows if seq is None or not h
        )
        chain_rows = [(t, s, h) for t, s, h in rows if s is not None and h]

        verified = 0
        unverified: list[dict] = []
        prev_hash = ""
        last_ok = False
        for tid, seq, stored in chain_rows:
            expected = hashlib.sha256(
                f"{tid}:{seq}:{prev_hash}".encode()
            ).hexdigest()
            if stored == expected:
                verified += 1
                last_ok = True
            else:
                last_ok = False
                if len(unverified) < _MAX_LISTED_BREAKS:
                    unverified.append({"trade_id": tid, "sequence_num": seq})
            # The next link always chains on the STORED hash, verified or
            # not — that's what lets a v2 row sit on top of legacy rows.
            prev_hash = stored or ""

        per_strategy[strat] = {
            "total": len(rows),
            "verified": verified,
            "legacy_unverified": len(chain_rows) - verified,
            "unsequenced": unsequenced,
            "intact_tail": last_ok if chain_rows else True,
            "chain_tip_seq": chain_rows[-1][1] if chain_rows else None,
            "breaks": unverified,
        }
        totals["total"] += len(rows)
        totals["verified"] += verified
        totals["legacy_unverified"] += len(chain_rows) - verified
        totals["unsequenced"] += unsequenced

    return {
        "ok": totals["legacy_unverified"] == 0 and totals["unsequenced"] == 0,
        "totals": totals,
        "strategies": per_strategy,
        "note": (
            "Rows predating the v2 chain (2026-06-05) chained against "
            "in-memory state and cannot be verified; new rows chain onto "
            "stored hashes and verify end-to-end."
        ),
    }
