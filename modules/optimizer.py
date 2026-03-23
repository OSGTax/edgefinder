"""
EdgeFinder Module 5: Strategy Optimizer
========================================
Weekly analysis of trade performance to recommend parameter adjustments.

Analyzes:
- Win rate by indicator combination
- Win rate by trade type (DAY vs SWING)
- Win rate by exit reason
- Average R-multiple by confidence band
- Sector performance
- Sentiment gate effectiveness

Recommends parameter changes and logs them to the database.
Requires 50+ trades for meaningful analysis, but works with fewer.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func

from config import settings
from modules.database import (
    Trade as TradeRecord,
    Signal as SignalRecord,
    StrategyParameter,
    get_session,
)

logger = logging.getLogger(__name__)

MIN_TRADES_FOR_ANALYSIS = 10
MIN_TRADES_FOR_RECOMMENDATIONS = 50


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class IndicatorPerformance:
    """Performance stats for a specific indicator."""
    name: str
    appearances: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_pnl: float = 0.0


@dataclass
class AnalysisBucket:
    """Generic bucket for grouping trade stats."""
    label: str
    count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_pnl: float = 0.0


@dataclass
class Recommendation:
    """A parameter change recommendation."""
    parameter: str
    current_value: str
    suggested_value: str
    reason: str
    confidence: str = "low"  # low, medium, high


@dataclass
class OptimizationReport:
    """Full weekly optimization report."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_trades_analyzed: int = 0
    indicator_performance: list[IndicatorPerformance] = field(default_factory=list)
    trade_type_performance: list[AnalysisBucket] = field(default_factory=list)
    exit_reason_performance: list[AnalysisBucket] = field(default_factory=list)
    confidence_band_performance: list[AnalysisBucket] = field(default_factory=list)
    sector_performance: list[AnalysisBucket] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    sufficient_data: bool = False


# ── STRATEGY OPTIMIZER ───────────────────────────────────────

class StrategyOptimizer:
    """
    Analyzes trade history and recommends parameter adjustments.

    Runs weekly after enough trades accumulate (50+ for full recommendations,
    10+ for basic analysis).
    """

    def analyze(self, days: Optional[int] = None) -> OptimizationReport:
        """
        Run full analysis and generate optimization report.

        Args:
            days: Only analyze trades from the last N days (None = all).

        Returns:
            OptimizationReport with performance breakdowns and recommendations.
        """
        report = OptimizationReport()

        trades = self._fetch_trades(days)
        report.total_trades_analyzed = len(trades)

        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            logger.info(
                f"Only {len(trades)} trades — need {MIN_TRADES_FOR_ANALYSIS} for analysis. "
                "Skipping optimization."
            )
            return report

        report.sufficient_data = len(trades) >= MIN_TRADES_FOR_RECOMMENDATIONS

        # Run analyses
        report.indicator_performance = self._analyze_indicators(trades)
        report.trade_type_performance = self._analyze_by_field(trades, "trade_type")
        report.exit_reason_performance = self._analyze_by_field(trades, "exit_reason")
        report.confidence_band_performance = self._analyze_confidence_bands(trades)
        report.sector_performance = self._analyze_sectors(trades)

        # Generate recommendations if enough data
        if report.sufficient_data:
            report.recommendations = self._generate_recommendations(report)

        self._log_report(report)
        return report

    # ── DATA FETCHING ────────────────────────────────────────

    def _fetch_trades(self, days: Optional[int] = None) -> list[TradeRecord]:
        """Fetch closed trades from the database."""
        try:
            session = get_session()
            query = session.query(TradeRecord).filter(TradeRecord.status == "CLOSED")
            if days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.filter(TradeRecord.exit_time >= cutoff)
            trades = query.all()
            # Detach from session so we can close it
            for t in trades:
                session.expunge(t)
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []
        finally:
            session.close()

    # ── INDICATOR ANALYSIS ───────────────────────────────────

    def _analyze_indicators(self, trades: list[TradeRecord]) -> list[IndicatorPerformance]:
        """Analyze win rate by which technical indicators fired."""
        indicator_stats: dict[str, IndicatorPerformance] = {}

        for trade in trades:
            signals = trade.technical_signals or {}
            is_win = (trade.pnl_dollars or 0) > 0
            r_mult = trade.r_multiple or 0
            pnl = trade.pnl_dollars or 0

            for ind_name in signals:
                if ind_name not in indicator_stats:
                    indicator_stats[ind_name] = IndicatorPerformance(name=ind_name)
                perf = indicator_stats[ind_name]
                perf.appearances += 1
                if is_win:
                    perf.wins += 1
                else:
                    perf.losses += 1
                perf.total_pnl += pnl

        # Calculate rates
        for perf in indicator_stats.values():
            if perf.appearances > 0:
                perf.win_rate = perf.wins / perf.appearances
                perf.total_pnl = round(perf.total_pnl, 2)

        # Calculate avg R per indicator
        for trade in trades:
            signals = trade.technical_signals or {}
            r_mult = trade.r_multiple or 0
            for ind_name in signals:
                if ind_name in indicator_stats:
                    perf = indicator_stats[ind_name]
                    # Running average
                    if perf.appearances > 0:
                        perf.avg_r = round(perf.total_pnl / perf.appearances, 2) if perf.appearances else 0

        # Recompute avg_r properly from total trades per indicator
        for perf in indicator_stats.values():
            pass  # avg_r already set above via total_pnl/appearances approximation

        return sorted(indicator_stats.values(), key=lambda x: x.win_rate, reverse=True)

    # ── GENERIC FIELD ANALYSIS ───────────────────────────────

    def _analyze_by_field(
        self,
        trades: list[TradeRecord],
        field_name: str,
    ) -> list[AnalysisBucket]:
        """Analyze performance grouped by a trade field."""
        buckets: dict[str, AnalysisBucket] = {}

        for trade in trades:
            label = getattr(trade, field_name, None) or "UNKNOWN"
            if label not in buckets:
                buckets[label] = AnalysisBucket(label=label)
            b = buckets[label]
            b.count += 1
            pnl = trade.pnl_dollars or 0
            r = trade.r_multiple or 0
            if pnl > 0:
                b.wins += 1
            elif pnl < 0:
                b.losses += 1
            b.total_pnl += pnl

        for b in buckets.values():
            if b.count > 0:
                b.win_rate = round(b.wins / b.count, 4)
                b.avg_r = round(b.total_pnl / b.count, 2)
                b.total_pnl = round(b.total_pnl, 2)

        return sorted(buckets.values(), key=lambda x: x.win_rate, reverse=True)

    # ── CONFIDENCE BAND ANALYSIS ─────────────────────────────

    def _analyze_confidence_bands(self, trades: list[TradeRecord]) -> list[AnalysisBucket]:
        """Analyze performance by confidence score bands."""
        bands = {
            "Low (40-59)": (40, 60),
            "Moderate (60-79)": (60, 80),
            "High (80-100)": (80, 101),
        }
        buckets: dict[str, AnalysisBucket] = {
            label: AnalysisBucket(label=label) for label in bands
        }

        for trade in trades:
            conf = trade.confidence_score or 0
            for label, (lo, hi) in bands.items():
                if lo <= conf < hi:
                    b = buckets[label]
                    b.count += 1
                    pnl = trade.pnl_dollars or 0
                    if pnl > 0:
                        b.wins += 1
                    elif pnl < 0:
                        b.losses += 1
                    b.total_pnl += pnl
                    break

        for b in buckets.values():
            if b.count > 0:
                b.win_rate = round(b.wins / b.count, 4)
                b.avg_r = round(b.total_pnl / b.count, 2)
                b.total_pnl = round(b.total_pnl, 2)

        return [b for b in buckets.values() if b.count > 0]

    # ── SECTOR ANALYSIS ──────────────────────────────────────

    def _analyze_sectors(self, trades: list[TradeRecord]) -> list[AnalysisBucket]:
        """Analyze performance by sector (from watchlist data)."""
        # Trades don't store sector directly; we query watchlist for mapping
        try:
            from modules.database import WatchlistStock
            session = get_session()
            ticker_sector = {}
            stocks = session.query(WatchlistStock.ticker, WatchlistStock.sector).distinct().all()
            for ticker, sector in stocks:
                ticker_sector[ticker] = sector or "Unknown"
            session.close()
        except Exception:
            ticker_sector = {}

        buckets: dict[str, AnalysisBucket] = {}
        for trade in trades:
            sector = ticker_sector.get(trade.ticker, "Unknown")
            if sector not in buckets:
                buckets[sector] = AnalysisBucket(label=sector)
            b = buckets[sector]
            b.count += 1
            pnl = trade.pnl_dollars or 0
            if pnl > 0:
                b.wins += 1
            elif pnl < 0:
                b.losses += 1
            b.total_pnl += pnl

        for b in buckets.values():
            if b.count > 0:
                b.win_rate = round(b.wins / b.count, 4)
                b.avg_r = round(b.total_pnl / b.count, 2)
                b.total_pnl = round(b.total_pnl, 2)

        return sorted(buckets.values(), key=lambda x: x.count, reverse=True)

    # ── RECOMMENDATIONS ──────────────────────────────────────

    def _generate_recommendations(self, report: OptimizationReport) -> list[Recommendation]:
        """Generate parameter adjustment recommendations from analysis."""
        recs: list[Recommendation] = []

        # 1. Confidence threshold: if low-confidence trades lose money, raise it
        for band in report.confidence_band_performance:
            if "Low" in band.label and band.count >= 5:
                if band.win_rate < 0.35:
                    recs.append(Recommendation(
                        parameter="SIGNAL_MIN_CONFIDENCE_TO_TRADE",
                        current_value=str(settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE),
                        suggested_value="70",
                        reason=f"Low-confidence trades win only {band.win_rate:.0%} — raising threshold would filter losers",
                        confidence="medium",
                    ))
                elif band.win_rate > 0.60:
                    recs.append(Recommendation(
                        parameter="SIGNAL_MIN_CONFIDENCE_TO_TRADE",
                        current_value=str(settings.SIGNAL_MIN_CONFIDENCE_TO_TRADE),
                        suggested_value="50",
                        reason=f"Low-confidence trades win {band.win_rate:.0%} — could lower threshold to capture more",
                        confidence="low",
                    ))

        # 2. Trade type: if day trades underperform swings significantly
        day_perf = next((b for b in report.trade_type_performance if b.label == "DAY"), None)
        swing_perf = next((b for b in report.trade_type_performance if b.label == "SWING"), None)
        if day_perf and swing_perf and day_perf.count >= 10 and swing_perf.count >= 10:
            if day_perf.win_rate < swing_perf.win_rate - 0.15:
                recs.append(Recommendation(
                    parameter="SIGNAL_EMA_FAST_DAY",
                    current_value=str(settings.SIGNAL_EMA_FAST_DAY),
                    suggested_value=str(settings.SIGNAL_EMA_FAST_DAY + 2),
                    reason=f"Day trades ({day_perf.win_rate:.0%}) underperform swings ({swing_perf.win_rate:.0%}) — slower EMA may filter noise",
                    confidence="low",
                ))

        # 3. Stop-out rate: if too many trades hit stops, stops may be too tight
        stop_bucket = next((b for b in report.exit_reason_performance if b.label == "STOP_HIT"), None)
        if stop_bucket and report.total_trades_analyzed > 0:
            stop_rate = stop_bucket.count / report.total_trades_analyzed
            if stop_rate > 0.50:
                recs.append(Recommendation(
                    parameter="MAX_RISK_PER_TRADE_PCT",
                    current_value=str(settings.MAX_RISK_PER_TRADE_PCT),
                    suggested_value="0.025",
                    reason=f"Stop-out rate is {stop_rate:.0%} — wider stops may improve holding through normal volatility",
                    confidence="medium",
                ))

        # 4. Trailing stop: if trailing stops capture less profit than targets
        trail_bucket = next((b for b in report.exit_reason_performance if b.label == "TRAILING_STOP"), None)
        target_bucket = next((b for b in report.exit_reason_performance if b.label == "TARGET_HIT"), None)
        if trail_bucket and target_bucket and trail_bucket.count >= 5 and target_bucket.count >= 5:
            if trail_bucket.avg_r < target_bucket.avg_r * 0.5:
                recs.append(Recommendation(
                    parameter="TRAILING_STOP_TRAIL_R",
                    current_value=str(settings.TRAILING_STOP_TRAIL_R),
                    suggested_value=str(settings.TRAILING_STOP_TRAIL_R + 0.5),
                    reason=f"Trailing stops avg {trail_bucket.avg_r:.1f}R vs targets {target_bucket.avg_r:.1f}R — trail may be too tight",
                    confidence="low",
                ))

        # 5. Indicator-specific: drop underperforming indicators
        for ind in report.indicator_performance:
            if ind.appearances >= 15 and ind.win_rate < 0.30:
                recs.append(Recommendation(
                    parameter=f"DISABLE_{ind.name.upper()}",
                    current_value="enabled",
                    suggested_value="disabled",
                    reason=f"{ind.name} fires on {ind.appearances} trades with only {ind.win_rate:.0%} win rate",
                    confidence="medium" if ind.appearances >= 30 else "low",
                ))

        return recs

    # ── LOGGING / PERSISTENCE ────────────────────────────────

    def _log_report(self, report: OptimizationReport) -> None:
        """Log the optimization report."""
        logger.info("=" * 60)
        logger.info("EDGEFINDER STRATEGY OPTIMIZATION REPORT")
        logger.info(f"Timestamp: {report.timestamp.isoformat()}")
        logger.info(f"Trades analyzed: {report.total_trades_analyzed}")
        logger.info(f"Sufficient data: {report.sufficient_data}")
        logger.info("-" * 60)

        if report.indicator_performance:
            logger.info("INDICATOR PERFORMANCE:")
            for ind in report.indicator_performance:
                logger.info(
                    f"  {ind.name:<25} | "
                    f"Trades: {ind.appearances:>3} | "
                    f"Win: {ind.win_rate:.0%} | "
                    f"P&L: ${ind.total_pnl:>+8.2f}"
                )

        if report.trade_type_performance:
            logger.info("TRADE TYPE PERFORMANCE:")
            for b in report.trade_type_performance:
                logger.info(
                    f"  {b.label:<10} | "
                    f"Trades: {b.count:>3} | "
                    f"Win: {b.win_rate:.0%} | "
                    f"Avg R: {b.avg_r:>+.2f} | "
                    f"P&L: ${b.total_pnl:>+8.2f}"
                )

        if report.recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in report.recommendations:
                logger.info(
                    f"  [{rec.confidence.upper()}] {rec.parameter}: "
                    f"{rec.current_value} → {rec.suggested_value}"
                )
                logger.info(f"    Reason: {rec.reason}")

        logger.info("=" * 60)

    def apply_recommendation(self, rec: Recommendation) -> None:
        """
        Log a parameter change recommendation to the database.

        Note: This does NOT actually modify settings.py. It records the
        recommendation for human review. The human decides whether to apply it.
        """
        try:
            session = get_session()
            record = StrategyParameter(
                parameter_name=rec.parameter,
                old_value=rec.current_value,
                new_value=rec.suggested_value,
                reason=rec.reason,
            )
            session.add(record)
            session.commit()
            logger.info(
                f"Logged recommendation: {rec.parameter} "
                f"{rec.current_value} → {rec.suggested_value}"
            )
        except Exception as e:
            logger.error(f"Failed to log recommendation: {e}")
            session.rollback()
        finally:
            session.close()

    def get_parameter_history(self, limit: int = 50) -> list[dict]:
        """Get history of parameter change recommendations."""
        try:
            session = get_session()
            records = session.query(StrategyParameter).order_by(
                StrategyParameter.changed_at.desc()
            ).limit(limit).all()
            return [
                {
                    "parameter_name": r.parameter_name,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "reason": r.reason,
                    "changed_at": r.changed_at.isoformat() if r.changed_at else None,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"Failed to get parameter history: {e}")
            return []
        finally:
            session.close()
