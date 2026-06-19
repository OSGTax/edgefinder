"""The research-agent account's Strategy seam.

``AnalystStrategy`` is a normal ``Strategy`` (rebalance(ctx) -> weights) but it
does NOT compute anything in-cycle. The heavy research (screen -> backtest ->
news -> LLM synthesis) runs in a separate daily job (edgefinder/agents/analyst)
that writes an ``agent_decisions`` row; this class simply READS that row's
target weights for the cycle's decision date and hands them to the proven live
engine, which diffs them against the current book and executes the deltas
(so hold / add / trim / sell all fall out for free).

Decoupling the slow brain from the fast hands is the whole point: the live
9:45 cycle that also trades the other strategies must never block on an
agentic research loop.

Fail-safe by construction: if no fresh decision exists (the research job
hasn't run, or a DB error), it HOLDS the current book (returns ``ctx.holdings``)
rather than liquidating. A backtest of this spec finds no decisions and simply
holds cash — this strategy is LIVE-only by nature (its edge accrues forward,
it is not a sealed-backtest strategy).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from edgefinder.engine.strategy import RebalanceContext

logger = logging.getLogger(__name__)

DEFAULT_NAME = "ai_analyst"
# A decision older than this many calendar days is stale — hold the book
# rather than trade on a week-old view (covers weekends/holidays with margin).
MAX_DECISION_STALENESS_DAYS = 6


class AnalystStrategy:
    """Trade the persisted daily decision of the research agent account."""

    def __init__(self, name: str = DEFAULT_NAME,
                 session_factory: Any | None = None,
                 max_staleness_days: int = MAX_DECISION_STALENESS_DAYS) -> None:
        self._name = name
        self._sf = session_factory
        self._max_staleness = max_staleness_days

    @property
    def name(self) -> str:
        return self._name

    def _factory(self):
        if self._sf is not None:
            return self._sf
        from edgefinder.db.engine import get_engine, get_session_factory
        self._sf = get_session_factory(get_engine())
        return self._sf

    def _load_target(self, as_of: date) -> dict[str, float] | None:
        """The latest decision's target weights on/before ``as_of`` (or None).

        Returns None on a stale (older than max_staleness) or missing decision,
        or on any DB error — the caller then holds the current book."""
        from edgefinder.db.models import AgentDecision
        session = self._factory()()
        try:
            row = (session.query(AgentDecision)
                   .filter(AgentDecision.strategy_name == self._name,
                           AgentDecision.decision_date <= as_of)
                   .order_by(AgentDecision.decision_date.desc())
                   .first())
            if row is None:
                return None
            if (as_of - row.decision_date) > timedelta(days=self._max_staleness):
                logger.warning("%s: latest decision %s is stale as of %s — holding",
                               self._name, row.decision_date, as_of)
                return None
            return dict(row.target_weights or {})
        finally:
            session.close()

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        try:
            target = self._load_target(ctx.date)
        except Exception:
            logger.warning("%s: decision lookup failed — holding current book",
                           self._name, exc_info=True)
            target = None
        if target is None:
            # never liquidate on a missing/failed decision: hold what we have
            return dict(ctx.holdings)
        # only names actually tradable in today's context, positive weight
        return {s: float(w) for s, w in target.items()
                if w and w > 0 and s in ctx.assets}


def make_analyst_strategy_factory(name: str = DEFAULT_NAME,
                                  session_factory: Any | None = None):
    """Spec factory for ``ai_analyst`` / ``ai_analyst:<name>``."""
    def _factory() -> AnalystStrategy:
        return AnalystStrategy(name=name, session_factory=session_factory)
    return _factory
