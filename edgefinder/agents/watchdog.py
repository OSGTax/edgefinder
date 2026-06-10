"""Watchdog agent — health monitor for the trading system.

Runs a set of invariant checks against the DB and records findings to
the agent_observations table. Designed to run every N minutes via cron,
a GitHub Action, or the /watchdog-tick Claude Code skill. Reconciles
against unresolved observations so a cleared condition auto-resolves
and a persisting one doesn't create duplicates.

CLI:
    python -m edgefinder.agents.watchdog

The module is pure-DB — it does not read logs, call APIs, or touch
filesystem state beyond the kill-switch config. That keeps it cheap and
lets every check be unit-tested against an in-memory SQLite fixture.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.agents.config import get_agent_config
from edgefinder.agents.journal import record_observation
from edgefinder.db.models import (
    AgentObservation,
    DividendCredit,
    StrategyAccount,
    SystemHeartbeat,
    TradeRecord,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "watchdog"

# Active window: 1 hour before market open to 1 hour after market close, ET.
# Hard-coded to real US equity market hours (9:30-16:00 ET). If the user
# wants pre-market/after-hours monitoring, expose these as settings.
_ACTIVE_WINDOW_START_ET = time(8, 30)  # 1h before 9:30 open
_ACTIVE_WINDOW_END_ET = time(17, 0)    # 1h after 16:00 close
_ET = ZoneInfo("America/New_York")


def is_in_active_window(now: datetime | None = None) -> bool:
    """True if current time (ET) is Mon-Fri, 08:30 to 17:00 inclusive."""
    current = (now or datetime.now(timezone.utc)).astimezone(_ET)
    if current.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return _ACTIVE_WINDOW_START_ET <= current.time() <= _ACTIVE_WINDOW_END_ET


@dataclass(frozen=True)
class ObservationSpec:
    """One finding from a single check, not yet persisted.

    `dedup_key` identifies the thing being observed across ticks (e.g.
    ("cash_drift", "alpha")). If the same key is present next tick, the
    existing row is kept open; if it disappears, the row is resolved.
    """

    severity: str
    category: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dedup_key: tuple[str, ...] = ()


# ── Invariant checks ────────────────────────────────────────


def check_cash_drift(
    session: Session, drift_threshold_pct: float
) -> list[ObservationSpec]:
    """Flag strategies whose stored cash_balance differs from the
    recomputed correct_cash by more than `drift_threshold_pct`.

    correct_cash = starting_capital + sum(closed_pnl) + sum(dividend_credits)
                 - sum(open_cost)

    The dividend_credits term is the v2 extension (engine/live credits cash
    for ex-dates crossed while holding); old-arena strategies simply have no
    credit rows, so the formula stays correct for both tiers.
    """
    specs: list[ObservationSpec] = []
    for account in session.query(StrategyAccount).all():
        realized = session.query(
            func.coalesce(func.sum(TradeRecord.pnl_dollars), 0.0)
        ).filter(
            TradeRecord.strategy_name == account.strategy_name,
            TradeRecord.status == "CLOSED",
        ).scalar() or 0.0
        open_cost = session.query(
            func.coalesce(
                func.sum(TradeRecord.entry_price * TradeRecord.shares), 0.0
            )
        ).filter(
            TradeRecord.strategy_name == account.strategy_name,
            TradeRecord.status == "OPEN",
        ).scalar() or 0.0
        credits = session.query(
            func.coalesce(func.sum(DividendCredit.amount), 0.0)
        ).filter(
            DividendCredit.strategy_name == account.strategy_name,
        ).scalar() or 0.0

        correct_cash = account.starting_capital + realized + credits - open_cost
        diff = account.cash_balance - correct_cash
        if account.starting_capital <= 0:
            continue
        drift_pct = abs(diff) / account.starting_capital
        if drift_pct < drift_threshold_pct:
            continue

        severity = "CRITICAL" if drift_pct >= 0.10 else "WARN"
        specs.append(ObservationSpec(
            severity=severity,
            category="cash_drift",
            message=(
                f"{account.strategy_name}: DB cash ${account.cash_balance:.2f} "
                f"vs correct ${correct_cash:.2f} (diff ${diff:.2f}, "
                f"{drift_pct * 100:.2f}% of starting capital)"
            ),
            metadata={
                "key": account.strategy_name,
                "db_cash": round(account.cash_balance, 2),
                "correct_cash": round(correct_cash, 2),
                "diff": round(diff, 2),
                "drift_pct": round(drift_pct, 4),
            },
            dedup_key=("cash_drift", account.strategy_name),
        ))
    return specs


def check_negative_cash(session: Session) -> list[ObservationSpec]:
    """Any strategy with cash_balance < 0 is immediately CRITICAL."""
    specs: list[ObservationSpec] = []
    for account in session.query(StrategyAccount).filter(
        StrategyAccount.cash_balance < 0
    ).all():
        specs.append(ObservationSpec(
            severity="CRITICAL",
            category="negative_cash",
            message=(
                f"{account.strategy_name}: cash_balance ${account.cash_balance:.2f} "
                "— account over-leveraged"
            ),
            metadata={
                "key": account.strategy_name,
                "cash": round(account.cash_balance, 2),
            },
            dedup_key=("negative_cash", account.strategy_name),
        ))
    return specs


def check_account_paused(session: Session) -> list[ObservationSpec]:
    """Any paused account gets an observation — the dashboard already
    shows this but the watchdog makes it auditable over time.
    """
    specs: list[ObservationSpec] = []
    for account in session.query(StrategyAccount).filter(
        StrategyAccount.is_paused.is_(True)
    ).all():
        specs.append(ObservationSpec(
            severity="WARN",
            category="account_paused",
            message=(
                f"{account.strategy_name}: paused "
                f"(drawdown {account.drawdown_pct * 100:.1f}%, "
                f"cash ${account.cash_balance:.2f})"
            ),
            metadata={
                "key": account.strategy_name,
                "drawdown_pct": round(account.drawdown_pct, 4),
                "cash": round(account.cash_balance, 2),
            },
            dedup_key=("account_paused", account.strategy_name),
        ))
    return specs


def check_high_drawdown(
    session: Session, warn_pct: float, critical_pct: float
) -> list[ObservationSpec]:
    """Pre-warning before the drawdown circuit breaker fires at 20%."""
    specs: list[ObservationSpec] = []
    for account in session.query(StrategyAccount).filter(
        StrategyAccount.drawdown_pct >= warn_pct,
        StrategyAccount.is_paused.is_(False),
    ).all():
        severity = "CRITICAL" if account.drawdown_pct >= critical_pct else "WARN"
        specs.append(ObservationSpec(
            severity=severity,
            category="high_drawdown",
            message=(
                f"{account.strategy_name}: drawdown "
                f"{account.drawdown_pct * 100:.1f}% (breaker at "
                f"{settings.drawdown_circuit_breaker_pct * 100:.0f}%)"
            ),
            metadata={
                "key": account.strategy_name,
                "drawdown_pct": round(account.drawdown_pct, 4),
                "peak_equity": round(account.peak_equity, 2),
            },
            dedup_key=("high_drawdown", account.strategy_name),
        ))
    return specs


# ── Cycle liveness ──────────────────────────────────────────

# The intraday cycle only runs Mon-Fri 09:30-16:00 ET (the scheduler /
# the intraday cron). Liveness is only meaningful inside that window —
# outside it, returning no spec lets persist_checks auto-resolve any
# open stall and never alerts overnight/weekends.
_MARKET_OPEN_ET = time(9, 30)
_MARKET_CLOSE_ET = time(16, 0)


def _in_market_hours(now: datetime, grace_min: int) -> bool:
    """True if `now` (any tz) is Mon-Fri and within 09:30+grace .. 16:00 ET.

    Compares full tz-aware datetimes (not bare times) so DST is handled
    correctly by zoneinfo for the given date.
    """
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Sat=5, Sun=6
        return False
    open_dt = (
        datetime.combine(et.date(), _MARKET_OPEN_ET, tzinfo=_ET)
        + timedelta(minutes=grace_min)
    )
    close_dt = datetime.combine(et.date(), _MARKET_CLOSE_ET, tzinfo=_ET)
    return open_dt <= et <= close_dt


def check_cycle_liveness(
    session: Session,
    stale_minutes: int,
    open_grace_minutes: int,
    now: datetime | None = None,
    component: str = "intraday_cycle",
) -> list[ObservationSpec]:
    """Flag a stalled intraday trading cycle.

    The cycle writes a SystemHeartbeat at the end of every run — success,
    controlled skip, or failure. We alert only during ET market hours, so
    overnight/weekend silence and a clean 16:00 close never fire, and any
    open stall auto-resolves once the window ends:
      - no heartbeat row at all       → CRITICAL "missing"
      - last_run_at older than stale  → CRITICAL "stale"
      - latest run errored (ok=False) → CRITICAL "error"

    A controlled skip is written fresh with ok=True, so a market holiday
    reads as healthy without this check needing a holiday calendar (which
    it couldn't obtain in a bare runner anyway).
    """
    now = now or datetime.now(timezone.utc)
    if not _in_market_hours(now, open_grace_minutes):
        return []

    hb = session.query(SystemHeartbeat).filter(
        SystemHeartbeat.component == component
    ).one_or_none()

    if hb is None:
        return [ObservationSpec(
            severity="CRITICAL",
            category="cycle_liveness",
            message=(
                f"No heartbeat for '{component}' during market hours — the "
                "intraday cycle has never run (entries/exits are not firing)."
            ),
            metadata={"key": component, "reason": "missing"},
            dedup_key=("cycle_liveness", component),
        )]

    last = hb.last_run_at
    if last.tzinfo is None:  # SQLite/Postgres round-trip naive datetimes
        last = last.replace(tzinfo=timezone.utc)
    age_min = (now - last).total_seconds() / 60.0

    if age_min > stale_minutes:
        return [ObservationSpec(
            severity="CRITICAL",
            category="cycle_liveness",
            message=(
                f"Intraday cycle STALLED: last heartbeat {age_min:.1f} min ago "
                f"(threshold {stale_minutes}m). Open positions are unmonitored "
                "— no stop/target checks are running."
            ),
            metadata={
                "key": component, "reason": "stale",
                "age_min": round(age_min, 1), "last_run_at": last.isoformat(),
            },
            dedup_key=("cycle_liveness", component),
        )]

    if not hb.ok:
        err = (hb.detail or {}).get("error", "unknown")
        return [ObservationSpec(
            severity="CRITICAL",
            category="cycle_liveness",
            message=f"Intraday cycle ran but ERRORED: {err}",
            metadata={"key": component, "reason": "error", "error": err},
            dedup_key=("cycle_liveness", component),
        )]

    return []


# ── Orchestration ──────────────────────────────────────────


def run_checks(
    session: Session, config_settings: dict[str, Any] | None = None
) -> list[ObservationSpec]:
    """Run every check and return the combined specs.

    config_settings comes from agents.config.get_agent_config("watchdog").settings
    and supplies thresholds. Safe defaults are used for any missing keys.
    """
    cfg = config_settings or {}
    drift_threshold = float(cfg.get("drift_threshold_pct", 0.01))
    drawdown_warn = float(cfg.get("drawdown_warn_pct", 0.15))
    drawdown_critical = float(cfg.get("drawdown_critical_pct", 0.18))

    specs: list[ObservationSpec] = []
    specs.extend(check_cash_drift(session, drift_threshold))
    specs.extend(check_negative_cash(session))
    specs.extend(check_account_paused(session))
    specs.extend(check_high_drawdown(session, drawdown_warn, drawdown_critical))
    if bool(cfg.get("liveness_enabled", settings.liveness_enabled)):
        specs.extend(check_cycle_liveness(
            session,
            stale_minutes=int(cfg.get("liveness_stale_minutes", settings.liveness_stale_minutes)),
            open_grace_minutes=int(
                cfg.get("liveness_open_grace_minutes", settings.liveness_open_grace_minutes)
            ),
        ))
    return specs


def persist_checks(
    session: Session,
    specs: list[ObservationSpec],
    agent_name: str = AGENT_NAME,
) -> dict[str, int]:
    """Reconcile specs against unresolved observations and commit.

    - New dedup_key   → insert a new observation
    - Existing key    → keep the prior row open (no duplicate)
    - Missing key     → auto-resolve the prior row (condition cleared)
    """
    unresolved = session.query(AgentObservation).filter(
        AgentObservation.agent_name == agent_name,
        AgentObservation.resolved_at.is_(None),
    ).all()

    def _key_of(obs: AgentObservation) -> tuple[str, ...]:
        meta_key = (obs.obs_metadata or {}).get("key")
        return (obs.category, meta_key) if meta_key is not None else (obs.category,)

    existing_by_key: dict[tuple[str, ...], AgentObservation] = {
        _key_of(obs): obs for obs in unresolved
    }
    current_keys: set[tuple[str, ...]] = set()

    new_count = 0
    kept_count = 0
    for spec in specs:
        key = spec.dedup_key or (spec.category,)
        current_keys.add(key)
        if key in existing_by_key:
            kept_count += 1
            continue
        record_observation(
            session,
            agent_name=agent_name,
            severity=spec.severity,
            category=spec.category,
            message=spec.message,
            metadata=spec.metadata,
            commit=False,
        )
        new_count += 1

    resolved_count = 0
    now = datetime.now(timezone.utc)
    for key, obs in existing_by_key.items():
        if key in current_keys:
            continue
        obs.resolved_at = now
        obs.resolved_by = agent_name
        resolved_count += 1

    session.commit()
    return {"new": new_count, "kept": kept_count, "resolved": resolved_count}


def run_once(
    session: Session, agent_name: str = AGENT_NAME
) -> tuple[list[ObservationSpec], dict[str, int]]:
    """One full tick: load config, run checks, persist."""
    cfg = get_agent_config(agent_name)
    specs = run_checks(session, cfg.settings)
    summary = persist_checks(session, specs, agent_name=agent_name)
    return specs, summary


# ── CLI ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one watchdog tick")
    parser.add_argument(
        "--agent-name",
        default=AGENT_NAME,
        help="Agent identifier used for observations (default: watchdog)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if agent-config.json disables the watchdog",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks and print findings but do not persist",
    )
    parser.add_argument(
        "--ignore-window",
        action="store_true",
        help="Run even outside the active market window (Mon-Fri 08:30-17:00 ET)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.ignore_window and not is_in_active_window():
        logger.info(
            "[%s] outside active window (Mon-Fri 08:30-17:00 ET) — exiting cleanly "
            "(use --ignore-window to override)",
            args.agent_name,
        )
        return 0

    cfg = get_agent_config(args.agent_name)
    if not cfg.enabled and not args.force:
        logger.info(
            "[%s] disabled by .claude/agent-config.json — exiting cleanly "
            "(use --force to override)",
            args.agent_name,
        )
        return 0

    from edgefinder.db.engine import get_engine, get_session_factory

    engine = get_engine()
    session = get_session_factory(engine)()
    try:
        specs = run_checks(session, cfg.settings)
        if args.dry_run:
            for s in specs:
                logger.info("[DRY] %s %s: %s", s.severity, s.category, s.message)
            logger.info("[DRY] %d findings (not persisted)", len(specs))
            return 0

        summary = persist_checks(session, specs, agent_name=args.agent_name)
        for s in specs:
            logger.info("[%s] %s %s: %s", args.agent_name, s.severity, s.category, s.message)
        logger.info(
            "[%s] tick done — new=%d kept=%d resolved=%d",
            args.agent_name, summary["new"], summary["kept"], summary["resolved"],
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
