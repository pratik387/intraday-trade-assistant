# services/execution/position_thesis_monitor.py
"""
Position Thesis Monitor
=======================

Automated system to continuously monitor if original trade thesis remains valid
during position lifetime. Tracks:
  - Momentum health (ADX trend, continuation)
  - Volume health (volume ratio maintenance)
  - Structure integrity (category-specific rules)
  - Target reachability (ATR-based probability)

Category-specific logic:
  - Breakout trades: Need momentum continuation, volume maintenance
  - Reversion trades: Benefit from counter-moves, need exhaustion maintenance

Auto-exit when thesis degrades below threshold.

Integration:
  - Lives in ExitExecutor as an additional exit check
  - Uses existing indicator infrastructure (ATR, RSI, ADX, volume)
  - Configuration-driven thresholds (no hardcoded values)

Reference: Professional institutional exit management systems
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import math
import pandas as pd

import logging

def _get_logger():
    """Get logger with fallback for testing environment."""
    try:
        from config.logging_config import get_execution_loggers
        _logger, _ = get_execution_loggers()
        if _logger is not None:
            return _logger
    except Exception:
        pass
    # Fallback to standard logging
    return logging.getLogger(__name__)

logger = _get_logger()


# ---------- Data Models ----------

@dataclass(frozen=True)
class MomentumHealth:
    """Momentum health assessment for a position."""
    score: float  # 0.0 to 1.0 (1.0 = healthy)
    adx_current: float
    adx_at_entry: float
    adx_decline_pct: float  # % decline from entry
    di_alignment: bool  # +DI > -DI for longs, vice versa
    notes: str


@dataclass(frozen=True)
class VolumeHealth:
    """Volume health assessment for a position."""
    score: float  # 0.0 to 1.0 (1.0 = healthy)
    volume_ratio_current: float
    volume_ratio_at_entry: float
    volume_decline_pct: float  # % decline from entry
    notes: str


@dataclass(frozen=True)
class StructureIntegrity:
    """Structure integrity assessment for a position."""
    score: float  # 0.0 to 1.0 (1.0 = intact)
    structure_type: str
    breach_detected: bool
    breach_reason: Optional[str]
    notes: str


@dataclass(frozen=True)
class TargetReachability:
    """Target reachability assessment based on ATR."""
    score: float  # 0.0 to 1.0 (1.0 = highly reachable)
    distance_to_target: float
    atr_current: float
    atr_at_entry: float
    bars_at_current_pace: float  # Estimated bars to reach target
    notes: str


@dataclass
class ThesisHealth:
    """
    Combined thesis health assessment.

    Exit when combined_score < threshold (configurable per category).
    """
    combined_score: float  # 0.0 to 1.0 (1.0 = thesis fully valid)
    momentum: MomentumHealth
    volume: VolumeHealth
    structure: StructureIntegrity
    target: TargetReachability
    should_exit: bool
    exit_reason: Optional[str]
    category: str  # "breakout" or "reversion"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging/persistence."""
        return {
            "combined_score": self.combined_score,
            "should_exit": self.should_exit,
            "exit_reason": self.exit_reason,
            "category": self.category,
            "momentum_score": self.momentum.score,
            "volume_score": self.volume.score,
            "structure_score": self.structure.score,
            "target_score": self.target.score,
        }


# ---------- Trade Category Classification ----------

def classify_setup_category(setup_type: str) -> str:
    """
    Classify setup into thesis monitoring category using central registry.

    Uses config/setup_categories.py as single source of truth.
    Maps pipeline categories to thesis monitoring behavior:
      - BREAKOUT, MOMENTUM → "breakout" (need momentum continuation, stricter threshold)
      - LEVEL, REVERSION → "reversion" (more patient, allow temporary weakness)
    """
    from config.setup_categories import get_category, SetupCategory

    category = get_category(setup_type)

    if category in (SetupCategory.BREAKOUT, SetupCategory.MOMENTUM):
        return "breakout"
    elif category in (SetupCategory.LEVEL, SetupCategory.REVERSION):
        return "reversion"
    else:
        # Unknown setup - default to reversion (more patient/conservative)
        return "reversion"


# ---------- Position Thesis Monitor ----------

class PositionThesisMonitor:
    """
    Monitor trade thesis validity during position lifetime.

    Tracks momentum, volume, structure integrity, and target reachability.
    Applies category-specific logic (breakout vs reversion).
    Signals exit when thesis degrades below threshold.
    """

    def __init__(self, cfg: dict, log=None):
        """
        Initialize thesis monitor with configuration.

        Args:
            cfg: Configuration dict with thesis_monitoring section
            log: Optional logger instance
        """
        self.log = log or logger

        # Load thesis monitoring configuration - KeyError if missing required params
        tm_cfg = cfg["thesis_monitoring"]

        self.enabled = bool(tm_cfg["enabled"])
        self.check_interval_seconds = float(tm_cfg["check_interval_seconds"])

        # Momentum health thresholds
        mom_cfg = tm_cfg["momentum_health"]
        self.breakout_min_adx = float(mom_cfg["breakout_min_adx"])
        self.breakout_adx_decline_pct = float(mom_cfg["breakout_adx_decline_pct"])
        self.reversion_rsi_bounce_threshold = float(mom_cfg["reversion_rsi_bounce_threshold"])

        # Volume health thresholds
        vol_cfg = tm_cfg["volume_health"]
        self.breakout_min_volume_ratio = float(vol_cfg["breakout_min_volume_ratio"])
        self.volume_decline_pct = float(vol_cfg["volume_decline_pct"])

        # Structure integrity thresholds
        struct_cfg = tm_cfg["structure_integrity"]
        self.orb_back_inside_exit = bool(struct_cfg["orb_back_inside_or_auto_exit"])
        self.breakout_reversal_atr_mult = float(struct_cfg["breakout_reversal_atr_mult"])
        self.reversion_vwap_breach_atr_mult = float(struct_cfg["reversion_vwap_breach_atr_mult"])

        # Target reachability thresholds
        target_cfg = tm_cfg["target_reachability"]
        self.atr_expansion_factor = float(target_cfg["atr_expansion_factor"])
        self.min_time_to_target_minutes = float(target_cfg["min_time_to_target_minutes"])
        self.expected_atr_per_bar = float(target_cfg["expected_atr_per_bar"])

        # Combined threshold for exit
        self.breakout_exit_threshold = float(tm_cfg["breakout_exit_threshold"])
        self.reversion_exit_threshold = float(tm_cfg["reversion_exit_threshold"])
        self.breakout_structure_breach_cap = float(tm_cfg["breakout_structure_breach_cap"])
        self.reversion_structure_breach_cap = float(tm_cfg["reversion_structure_breach_cap"])

        # Time-based invalidation for mean reversion (research: Alvarez Quant Trading)
        # Progressive time checks - stricter thresholds as time passes
        self.reversion_max_hold_minutes = float(tm_cfg["reversion_max_hold_minutes"])
        self.reversion_min_progress_pct = float(tm_cfg["reversion_min_progress_pct"])
        self.reversion_time_check_interval = float(tm_cfg["reversion_time_check_interval"])
        self.reversion_progress_increment = float(tm_cfg["reversion_progress_increment"])
        self.reversion_score_increment = float(tm_cfg["reversion_score_increment"])
        self.reversion_force_exit_minutes = float(tm_cfg["reversion_force_exit_minutes"])

        # Category-specific weight configuration for combined score
        breakout_weights = tm_cfg["weights_breakout"]
        self.breakout_weights = {
            "momentum": float(breakout_weights["momentum"]),
            "volume": float(breakout_weights["volume"]),
            "structure": float(breakout_weights["structure"]),
            "target": float(breakout_weights["target"]),
        }
        reversion_weights = tm_cfg["weights_reversion"]
        self.reversion_weights = {
            "momentum": float(reversion_weights["momentum"]),
            "volume": float(reversion_weights["volume"]),
            "structure": float(reversion_weights["structure"]),
            "target": float(reversion_weights["target"]),
        }

        # Cache for last check timestamps (avoid excessive computation)
        self._last_check: Dict[str, float] = {}
        # Cache for last thesis health result (return when rate limited to allow exit on subsequent ticks)
        self._last_health: Dict[str, ThesisHealth] = {}
        # Consecutive failure counter - require multiple failures before exit
        self._consecutive_failures: Dict[str, int] = {}
        self.consecutive_failures_required = int(tm_cfg["consecutive_failures_required"])

        self.log.info(
            f"[ThesisMonitor] Initialized: enabled={self.enabled}, "
            f"breakout_threshold={self.breakout_exit_threshold}, "
            f"reversion_threshold={self.reversion_exit_threshold}, "
            f"breakout_breach_cap={self.breakout_structure_breach_cap}, "
            f"reversion_breach_cap={self.reversion_structure_breach_cap}"
        )
        self.log.info(
            f"[ThesisMonitor] Progressive time: base={self.reversion_max_hold_minutes}min, "
            f"interval={self.reversion_time_check_interval}min, "
            f"progress_incr={self.reversion_progress_increment}%, "
            f"score_incr={self.reversion_score_increment}, "
            f"force_exit={self.reversion_force_exit_minutes}min"
        )
        self.log.info(
            f"[ThesisMonitor] Breakout weights: {self.breakout_weights}"
        )
        self.log.info(
            f"[ThesisMonitor] Reversion weights: {self.reversion_weights}"
        )

    def check_thesis(
        self,
        symbol: str,
        side: str,
        current_price: float,
        plan: Dict[str, Any],
        df_5m: Optional[pd.DataFrame] = None,
        tick_ts: Optional[pd.Timestamp] = None,
    ) -> Optional[ThesisHealth]:
        """
        Check if trade thesis remains valid.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            current_price: Current market price
            plan: Trade plan dict with entry indicators, targets, etc.
            df_5m: Optional 5-minute OHLCV DataFrame for live indicator calculation
            tick_ts: Optional tick timestamp for rate limiting (uses simulated time in backtest)

        Returns:
            ThesisHealth if check performed, None if skipped (disabled/cached)
        """
        if not self.enabled:
            return None

        # Rate limit checks using tick timestamp (works correctly in backtest mode)
        # Fall back to wall-clock time if tick_ts not provided
        if tick_ts is not None:
            current_time = tick_ts.timestamp()
        else:
            import time
            current_time = time.time()
        # STICKY BREACH: Once thesis fails (should_exit=True), keep signaling exit
        # Don't allow recovery - the breach indicates trade thesis is compromised
        cached = self._last_health.get(symbol)
        if cached and cached.should_exit:
            if self.log:
                self.log.info(f"[ThesisMonitor] {symbol} STICKY_BREACH active")
            return cached  # Keep returning failed thesis until position closes

        last_check = self._last_check.get(symbol, 0)
        if current_time - last_check < self.check_interval_seconds:
            # Rate limited - return cached result to allow exit on subsequent ticks
            return cached
        self._last_check[symbol] = current_time

        # Get setup type and category (plan may have "setup_type" or "strategy")
        setup_type = plan.get("setup_type") or plan.get("strategy", "unknown")
        category = classify_setup_category(setup_type)

        # Get entry-time indicators from plan
        entry_indicators = plan.get("indicators", {})
        # Get entry price: prefer actual_entry (fill price), then entry_ref_price (planned entry),
        # then entry.reference (from entry dict), finally fall back to current_price
        entry_dict = plan.get("entry", {})

        # Track which source provided entry price for debugging
        entry_price = float(current_price)  # Default fallback
        entry_price_source = "current_price_FALLBACK"
        try:
            if plan.get("actual_entry"):
                entry_price = float(plan.get("actual_entry"))
                entry_price_source = "actual_entry"
            elif plan.get("entry_ref_price"):
                entry_price = float(plan.get("entry_ref_price"))
                entry_price_source = "entry_ref_price"
            elif isinstance(entry_dict, dict) and entry_dict.get("reference"):
                entry_price = float(entry_dict.get("reference"))
                entry_price_source = "entry.reference"
        except (ValueError, TypeError):
            pass  # Keep default fallback

        # Calculate current indicators if df_5m provided
        current_indicators = self._calculate_current_indicators(df_5m) if df_5m is not None else {}

        # Assess each health dimension
        momentum = self._assess_momentum_health(
            side, category, entry_indicators, current_indicators, df_5m
        )
        volume = self._assess_volume_health(
            category, entry_indicators, current_indicators, df_5m
        )
        structure = self._assess_structure_integrity(
            symbol, side, current_price, entry_price, plan, category, df_5m
        )
        target = self._assess_target_reachability(
            side, current_price, plan, entry_indicators, current_indicators
        )

        # Calculate combined score with category-specific weights
        combined_score = self._calculate_combined_score(
            momentum, volume, structure, target, category
        )

        # Determine if should exit
        threshold = (
            self.breakout_exit_threshold if category == "breakout"
            else self.reversion_exit_threshold
        )

        # Check if score is below threshold
        score_below_threshold = combined_score < threshold

        # Track consecutive failures - require multiple before triggering exit
        if score_below_threshold:
            self._consecutive_failures[symbol] = self._consecutive_failures.get(symbol, 0) + 1
        else:
            self._consecutive_failures[symbol] = 0  # Reset on healthy check

        consecutive_count = self._consecutive_failures.get(symbol, 0)

        # Filter 1: Require consecutive failures
        should_exit = (score_below_threshold and
                       consecutive_count >= self.consecutive_failures_required)

        # Filter 5: Reversion progressing - if price closer to target, don't exit
        if should_exit and category == "reversion":
            # Get T1 using standard format: plan["targets"][0]["level"]
            targets = plan.get("targets") or []
            t1_price = None
            if len(targets) > 0:
                try:
                    t1_price = float(targets[0].get("level"))
                except (KeyError, TypeError, ValueError):
                    pass
            if t1_price is not None:
                entry_to_target = abs(entry_price - t1_price)
                current_to_target = abs(current_price - t1_price)
                if current_to_target < entry_to_target * 0.9:  # 10% closer to target
                    should_exit = False  # Trade progressing, let it run
                    if self.log:
                        self.log.info(
                            f"[ThesisMonitor] {symbol} SKIP_EXIT reversion progressing "
                            f"(entry_dist={entry_to_target:.2f} > curr_dist={current_to_target:.2f})"
                        )

        # TIME-BASED INVALIDATION for reversion trades (research: Alvarez Quant Trading)
        # Progressive time checks - stricter thresholds as time passes
        time_based_exit = False
        if category == "reversion" and tick_ts is not None:
            # Get entry time - try multiple sources (matching exit_executor field names)
            entry_time_str = (
                plan.get("entry_ts") or
                plan.get("trigger_ts") or
                plan.get("entry_time") or
                plan.get("fill_time") or
                plan.get("_state", {}).get("entry_time")
            )
            if not entry_time_str:
                # Log once per symbol when entry time not found
                if self.log and symbol not in getattr(self, '_time_warn_logged', set()):
                    if not hasattr(self, '_time_warn_logged'):
                        self._time_warn_logged = set()
                    self._time_warn_logged.add(symbol)
                    self.log.info(
                        f"[ThesisMonitor] {symbol} NO_ENTRY_TIME | "
                        f"plan_keys={list(plan.keys())[:10]}"
                    )
            if entry_time_str:
                try:
                    # Parse entry time
                    if isinstance(entry_time_str, str):
                        entry_ts = pd.Timestamp(entry_time_str)
                    else:
                        entry_ts = pd.Timestamp(entry_time_str)

                    # Calculate time in trade (minutes)
                    time_in_trade_mins = (tick_ts - entry_ts).total_seconds() / 60

                    # Calculate progress toward target
                    targets = plan.get("targets") or []
                    t1_price = None
                    if len(targets) > 0:
                        try:
                            t1_price = float(targets[0].get("level"))
                        except (KeyError, TypeError, ValueError):
                            pass

                    progress_pct = 0.0
                    if t1_price is not None:
                        entry_to_target = abs(entry_price - t1_price)
                        current_to_target = abs(current_price - t1_price)
                        if entry_to_target > 0:
                            progress_pct = (entry_to_target - current_to_target) / entry_to_target * 100

                    # PROGRESSIVE TIME CHECKS - stricter thresholds as time passes
                    # Calculate how many intervals past the base time
                    if time_in_trade_mins > self.reversion_max_hold_minutes:
                        time_past_base = time_in_trade_mins - self.reversion_max_hold_minutes
                        intervals_past = int(time_past_base / self.reversion_time_check_interval)

                        # Progressive thresholds: base + (intervals × increment)
                        required_progress = min(
                            self.reversion_min_progress_pct + (intervals_past * self.reversion_progress_increment),
                            70.0  # Cap at 70%
                        )
                        required_score = min(
                            threshold + (intervals_past * self.reversion_score_increment),
                            0.80  # Cap score threshold at 0.80
                        )

                        # Force exit after reversion_force_exit_minutes regardless of score
                        force_exit = time_in_trade_mins >= self.reversion_force_exit_minutes

                        # Log time in trade at each interval boundary
                        if self.log and int(time_past_base) % int(self.reversion_time_check_interval) < 1:
                            self.log.info(
                                f"[ThesisMonitor] {symbol} TIME_CHECK | "
                                f"time={time_in_trade_mins:.0f}min | intervals={intervals_past} | "
                                f"progress={progress_pct:.1f}% (need {required_progress:.0f}%) | "
                                f"score={combined_score:.2f} (need <{required_score:.2f})"
                            )

                        # Exit conditions:
                        # 1. Force exit after max time
                        # 2. Progress below threshold AND score below threshold
                        if force_exit and progress_pct < 70.0:
                            time_based_exit = True
                            should_exit = True
                            if self.log:
                                self.log.warning(
                                    f"[ThesisMonitor] {symbol} TIME_FORCE_EXIT | "
                                    f"time={time_in_trade_mins:.0f}min >= {self.reversion_force_exit_minutes}min | "
                                    f"progress={progress_pct:.1f}% < 70%"
                                )
                        elif progress_pct < required_progress and combined_score < required_score:
                            time_based_exit = True
                            should_exit = True
                            if self.log:
                                self.log.warning(
                                    f"[ThesisMonitor] {symbol} TIME_BASED_EXIT | "
                                    f"time={time_in_trade_mins:.0f}min (interval {intervals_past}) | "
                                    f"progress={progress_pct:.1f}% < {required_progress:.0f}% | "
                                    f"score={combined_score:.2f} < {required_score:.2f}"
                                )
                except Exception as e:
                    if self.log:
                        self.log.info(f"[ThesisMonitor] {symbol} TIME_CHECK_ERROR | {e}")

        exit_reason = None
        if should_exit:
            if time_based_exit:
                # Time-based exit takes precedence
                exit_reason = f"thesis_failed_time_exceeded_score{combined_score:.2f}"
            else:
                # Determine primary reason for exit based on lowest score
                scores = [
                    (momentum.score, "momentum_degraded"),
                    (volume.score, "volume_dried_up"),
                    (structure.score, "structure_breached"),
                    (target.score, "target_unreachable"),
                ]
                worst = min(scores, key=lambda x: x[0])
                exit_reason = f"thesis_failed_{worst[1]}_score{combined_score:.2f}"

        health = ThesisHealth(
            combined_score=combined_score,
            momentum=momentum,
            volume=volume,
            structure=structure,
            target=target,
            should_exit=should_exit,
            exit_reason=exit_reason,
            category=category,
        )

        # Log assessment - detailed breakdown for exits to enable tuning
        if self.log:
            if should_exit:
                # Detailed exit log with all contributing factors
                self.log.warning(
                    f"THESIS_EXIT | {symbol} | {setup_type} ({category}) | "
                    f"Combined={combined_score:.2f} < threshold={threshold}"
                )
                self.log.warning(
                    f"  MOMENTUM: score={momentum.score:.2f} | "
                    f"ADX: entry={momentum.adx_at_entry:.1f} -> current={momentum.adx_current:.1f} "
                    f"(decline={momentum.adx_decline_pct:.1f}%) | "
                    f"DI_aligned={momentum.di_alignment} | {momentum.notes}"
                )
                self.log.warning(
                    f"  VOLUME: score={volume.score:.2f} | "
                    f"Ratio: entry={volume.volume_ratio_at_entry:.2f} -> current={volume.volume_ratio_current:.2f} "
                    f"(decline={volume.volume_decline_pct:.1f}%) | {volume.notes}"
                )
                self.log.warning(
                    f"  STRUCTURE: score={structure.score:.2f} | "
                    f"type={structure.structure_type} | "
                    f"breach={structure.breach_detected} | "
                    f"reason={structure.breach_reason} | {structure.notes}"
                )
                self.log.warning(
                    f"  TARGET: score={target.score:.2f} | "
                    f"distance={target.distance_to_target:.2f} | "
                    f"ATR: entry={target.atr_at_entry:.2f} -> current={target.atr_current:.2f} | "
                    f"bars_needed={target.bars_at_current_pace:.1f} | {target.notes}"
                )
                # Summary of what caused the exit
                failing_components = []
                if momentum.score < 0.5:
                    failing_components.append(f"momentum({momentum.score:.2f})")
                if volume.score < 0.5:
                    failing_components.append(f"volume({volume.score:.2f})")
                if structure.score < 0.5:
                    failing_components.append(f"structure({structure.score:.2f})")
                if target.score < 0.5:
                    failing_components.append(f"target({target.score:.2f})")
                self.log.warning(
                    f"  PRIMARY_FACTORS: {', '.join(failing_components) if failing_components else 'combined_degradation'}"
                )
            else:
                # Brief log for healthy positions (INFO for analysis, can revert to DEBUG later)
                has_current = bool(current_indicators)
                self.log.info(
                    f"[ThesisMonitor] {symbol} | {setup_type} ({category}) | "
                    f"Score: {combined_score:.2f} (threshold={threshold}) | "
                    f"M={momentum.score:.2f} V={volume.score:.2f} "
                    f"S={structure.score:.2f} T={target.score:.2f} | "
                    f"entry={entry_price:.2f} current={current_price:.2f} ({entry_price_source}) | "
                    f"{'LIVE' if has_current else 'DEFAULTS'}"
                )

        # Cache health result for rate-limited checks
        self._last_health[symbol] = health
        return health

    def _calculate_current_indicators(self, df_5m: pd.DataFrame) -> Dict[str, float]:
        """Calculate current indicators from 5m bars."""
        if df_5m is None:
            if self.log:
                self.log.debug("[ThesisMonitor] df_5m is None - using defaults")
            return {}
        if len(df_5m) < 14:
            if self.log:
                self.log.debug(f"[ThesisMonitor] df_5m has {len(df_5m)} bars (<14) - using defaults")
            return {}

        try:
            from services.indicators.indicators import (
                calculate_atr, calculate_adx_with_di, calculate_rsi, volume_ratio
            )

            atr = calculate_atr(df_5m, period=14)
            adx, plus_di, minus_di = calculate_adx_with_di(df_5m, period=14)
            rsi = calculate_rsi(df_5m['close'], period=14)
            vol_ratio = volume_ratio(df_5m, lookback=20)

            return {
                "atr": atr,
                "adx": float(adx.iloc[-1]) if len(adx) > 0 else 20.0,
                "plus_di": float(plus_di.iloc[-1]) if len(plus_di) > 0 else 25.0,
                "minus_di": float(minus_di.iloc[-1]) if len(minus_di) > 0 else 25.0,
                "rsi": float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0,
                "volume_ratio": vol_ratio,
            }
        except Exception as e:
            if self.log:
                self.log.debug(f"[ThesisMonitor] Indicator calc failed: {e}")
            return {}

    def _assess_momentum_health(
        self,
        side: str,
        category: str,
        entry_indicators: Dict[str, float],
        current_indicators: Dict[str, float],
        df_5m: Optional[pd.DataFrame],
    ) -> MomentumHealth:
        """
        Assess momentum health based on ADX trend and DI alignment.

        Breakout trades: Need ADX sustained or rising, DI aligned with direction
        Reversion trades: RSI moving toward neutral, exhaustion fading
        """
        adx_entry = float(entry_indicators.get("adx", 25))
        adx_current = float(current_indicators.get("adx", adx_entry))
        plus_di = float(current_indicators.get("plus_di", 25))
        minus_di = float(current_indicators.get("minus_di", 25))
        rsi_current = float(current_indicators.get("rsi", 50))

        # Calculate ADX decline
        adx_decline_pct = max(0, (adx_entry - adx_current) / max(adx_entry, 1)) * 100

        # DI alignment check
        side_upper = side.upper()
        di_alignment = (
            (plus_di > minus_di) if side_upper == "BUY"
            else (minus_di > plus_di)
        )

        # Calculate score based on category
        if category == "breakout":
            # Breakouts need momentum continuation
            score = 1.0
            notes_parts = []

            # ADX level check
            if adx_current < self.breakout_min_adx:
                score -= 0.3
                notes_parts.append(f"ADX<{self.breakout_min_adx}")

            # ADX decline check
            if adx_decline_pct > self.breakout_adx_decline_pct:
                decline_penalty = min(0.4, adx_decline_pct / 100)
                score -= decline_penalty
                notes_parts.append(f"ADX_decline={adx_decline_pct:.0f}%")

            # DI alignment check
            if not di_alignment:
                score -= 0.2
                notes_parts.append("DI_misaligned")

            notes = "; ".join(notes_parts) if notes_parts else "momentum_healthy"

        else:  # reversion
            # Reversions: RSI should move TOWARD neutral from entry extreme
            # Thesis fails if RSI moves AWAY from neutral (wrong direction)
            rsi_entry = float(entry_indicators.get("rsi", 50))
            score = 1.0
            notes_parts = []

            if side_upper == "BUY":
                # Long reversion from oversold: RSI should RISE toward neutral
                # Thesis fails if RSI drops further (more oversold = wrong direction)
                if rsi_current < rsi_entry - 10:
                    # RSI dropped 10+ points from entry = going wrong direction
                    score -= 0.4
                    notes_parts.append(f"RSI_dropping({rsi_entry:.0f}->{rsi_current:.0f})")
                elif rsi_current < 30 and rsi_entry < 35:
                    # Still deeply oversold with no progress
                    score -= 0.2
                    notes_parts.append("RSI_stuck_oversold")
                elif rsi_current > rsi_entry + 15:
                    # RSI rose 15+ points = reversion working well
                    notes_parts.append(f"RSI_recovering({rsi_entry:.0f}->{rsi_current:.0f})")
            else:
                # Short reversion from overbought: RSI should FALL toward neutral
                # Thesis fails if RSI rises further (more overbought = wrong direction)
                if rsi_current > rsi_entry + 10:
                    # RSI rose 10+ points from entry = going wrong direction
                    score -= 0.4
                    notes_parts.append(f"RSI_rising({rsi_entry:.0f}->{rsi_current:.0f})")
                elif rsi_current > 70 and rsi_entry > 65:
                    # Still deeply overbought with no progress
                    score -= 0.2
                    notes_parts.append("RSI_stuck_overbought")
                elif rsi_current < rsi_entry - 15:
                    # RSI fell 15+ points = reversion working well
                    notes_parts.append(f"RSI_declining({rsi_entry:.0f}->{rsi_current:.0f})")

            notes = "; ".join(notes_parts) if notes_parts else "reversion_progressing"

        return MomentumHealth(
            score=max(0, min(1, score)),
            adx_current=adx_current,
            adx_at_entry=adx_entry,
            adx_decline_pct=adx_decline_pct,
            di_alignment=di_alignment,
            notes=notes,
        )

    def _assess_volume_health(
        self,
        category: str,
        entry_indicators: Dict[str, float],
        current_indicators: Dict[str, float],
        df_5m: Optional[pd.DataFrame],
    ) -> VolumeHealth:
        """
        Assess volume health based on volume ratio maintenance.

        Breakout trades: Need sustained volume (institutional follow-through)
        Reversion trades: Volume exhaustion is expected (less critical)
        """
        vol_ratio_entry = float(entry_indicators.get("volume_ratio", 1.5))
        vol_ratio_current = float(current_indicators.get("volume_ratio", 1.0))

        # Calculate volume decline
        vol_decline_pct = max(0, (vol_ratio_entry - vol_ratio_current) / max(vol_ratio_entry, 0.1)) * 100

        if category == "breakout":
            # Breakouts need volume confirmation
            score = 1.0
            notes_parts = []

            # Volume ratio check
            if vol_ratio_current < self.breakout_min_volume_ratio:
                score -= 0.3
                notes_parts.append(f"vol_ratio<{self.breakout_min_volume_ratio}")

            # Volume decline check
            if vol_decline_pct > self.volume_decline_pct:
                decline_penalty = min(0.4, vol_decline_pct / 100)
                score -= decline_penalty
                notes_parts.append(f"vol_decline={vol_decline_pct:.0f}%")

            notes = "; ".join(notes_parts) if notes_parts else "volume_healthy"

        else:  # reversion
            # Volume exhaustion is expected for reversions - don't penalize decline
            # Only penalize if volume SURGES against the thesis (institutional push against you)
            score = 1.0
            notes_parts = []

            if vol_ratio_current > vol_ratio_entry * 1.5:
                # 50%+ volume increase = someone pushing against your reversion thesis
                score -= 0.4
                notes_parts.append(f"volume_surge_against({vol_ratio_entry:.1f}->{vol_ratio_current:.1f})")
            elif vol_ratio_current < vol_ratio_entry * 0.3:
                # Volume dried up significantly - neutral, reversion may stall
                notes_parts.append("volume_exhausted")

            notes = "; ".join(notes_parts) if notes_parts else "volume_ok_for_reversion"

        return VolumeHealth(
            score=max(0, min(1, score)),
            volume_ratio_current=vol_ratio_current,
            volume_ratio_at_entry=vol_ratio_entry,
            volume_decline_pct=vol_decline_pct,
            notes=notes,
        )

    def _assess_structure_integrity(
        self,
        symbol: str,
        side: str,
        current_price: float,
        entry_price: float,
        plan: Dict[str, Any],
        category: str,
        df_5m: Optional[pd.DataFrame],
    ) -> StructureIntegrity:
        """
        Assess structure integrity based on category-specific rules.

        Breakout trades: Check if breakout level is holding
        Reversion trades: Check if reversal structure is intact
        """
        setup_type = plan.get("setup_type") or plan.get("strategy", "")
        levels = plan.get("levels", {})
        atr_entry = float(plan.get("indicators", {}).get("atr", 1.0))
        side_upper = side.upper()

        score = 1.0
        breach_detected = False
        breach_reason = None
        notes_parts = []

        if category == "breakout":
            # Check if breakout level is being retested or lost
            orh = levels.get("ORH") or levels.get("orh")
            orl = levels.get("ORL") or levels.get("orl")
            breakout_level = levels.get("breakout_level")

            if "orb" in setup_type.lower():
                # ORB specific: Check if price closed back inside OR
                if side_upper == "BUY" and orh is not None:
                    if current_price < orh:
                        breach_detected = True
                        breach_reason = "price_back_below_orh"
                        score = 0.2
                        notes_parts.append("ORB_long_failed_below_ORH")
                    elif current_price < orh + (orh - orl) * 0.1 if orl else current_price < orh * 1.001:
                        score -= 0.2
                        notes_parts.append("price_near_orh_danger_zone")

                elif side_upper == "SELL" and orl is not None:
                    if current_price > orl:
                        breach_detected = True
                        breach_reason = "price_back_above_orl"
                        score = 0.2
                        notes_parts.append("ORB_short_failed_above_ORL")
                    elif current_price > orl - (orh - orl) * 0.1 if orh else current_price > orl * 0.999:
                        score -= 0.2
                        notes_parts.append("price_near_orl_danger_zone")

            elif breakout_level is not None:
                # Generic breakout: Check if level is holding
                if side_upper == "BUY" and current_price < breakout_level:
                    breach_detected = True
                    breach_reason = "price_below_breakout_level"
                    score = 0.3
                    notes_parts.append("breakout_level_lost")
                elif side_upper == "SELL" and current_price > breakout_level:
                    breach_detected = True
                    breach_reason = "price_above_breakout_level"
                    score = 0.3
                    notes_parts.append("breakdown_level_reclaimed")

            # Check for significant adverse move (ATR-based)
            adverse_move = (
                (entry_price - current_price) if side_upper == "BUY"
                else (current_price - entry_price)
            )
            if adverse_move > atr_entry * self.breakout_reversal_atr_mult:
                score -= 0.3
                notes_parts.append(f"adverse_move>{self.breakout_reversal_atr_mult}xATR")

        else:  # reversion
            # Reversion: Check if counter-move is progressing
            # Price should be moving toward VWAP/mean, not away
            # Get VWAP - prefer current VWAP from df_5m, fallback to entry-time VWAP
            indicators = plan.get("indicators", {})
            entry_vwap = levels.get("vwap") or levels.get("VWAP") or indicators.get("vwap") or indicators.get("VWAP")

            # Try to get current VWAP from df_5m
            current_vwap = None
            if df_5m is not None and len(df_5m) > 0 and "vwap" in df_5m.columns:
                current_vwap = float(df_5m["vwap"].iloc[-1])

            # Use current VWAP for comparison, fallback to entry VWAP
            vwap = current_vwap if current_vwap is not None else entry_vwap

            if vwap is not None:
                # Check if moving toward VWAP using ATR-based breach threshold
                entry_to_vwap = abs(entry_price - vwap)
                current_to_vwap = abs(current_price - vwap)

                # ATR-based breach: price moved X ATR further from VWAP than at entry
                breach_threshold = atr_entry * self.reversion_vwap_breach_atr_mult
                distance_increase = current_to_vwap - entry_to_vwap

                # Log VWAP comparison for reversion trades
                if self.log:
                    self.log.info(
                        f"[ThesisMonitor] {symbol} VWAP_CHECK | "
                        f"entry={entry_price:.2f} current={current_price:.2f} vwap={vwap:.2f} | "
                        f"entry_dist={entry_to_vwap:.2f} curr_dist={current_to_vwap:.2f} | "
                        f"increase={distance_increase:.2f} threshold={breach_threshold:.2f} ({self.reversion_vwap_breach_atr_mult}xATR)"
                    )

                if distance_increase > breach_threshold:
                    # Moving away from VWAP by more than threshold ATR - thesis failing
                    score -= 0.4
                    breach_detected = True
                    breach_reason = f"moving_away_from_vwap_{distance_increase/atr_entry:.1f}xATR"
                    notes_parts.append("reversion_failing_away_from_mean")
                elif current_to_vwap < entry_to_vwap * 0.5:
                    # Good progress toward VWAP (50%+ closer)
                    notes_parts.append("reversion_progressing_well")
            else:
                # No VWAP available - can't assess reversion structure
                notes_parts.append("no_vwap_available")
                if self.log:
                    self.log.info(
                        f"[ThesisMonitor] {symbol} NO_VWAP | "
                        f"levels_keys={list(levels.keys())} indicators_keys={list(indicators.keys())} | "
                        f"entry_vwap={entry_vwap} current_vwap={current_vwap}"
                    )

        notes = "; ".join(notes_parts) if notes_parts else "structure_intact"

        return StructureIntegrity(
            score=max(0, min(1, score)),
            structure_type=setup_type,
            breach_detected=breach_detected,
            breach_reason=breach_reason,
            notes=notes,
        )

    def _assess_target_reachability(
        self,
        side: str,
        current_price: float,
        plan: Dict[str, Any],
        entry_indicators: Dict[str, float],
        current_indicators: Dict[str, float],
    ) -> TargetReachability:
        """
        Assess target reachability based on ATR and price momentum.

        If ATR has contracted significantly, targets become less reachable.
        If price has stalled, target probability decreases.
        """
        # Get targets from plan
        targets = plan.get("targets", [])
        t1 = None
        if targets and len(targets) > 0:
            t1 = targets[0].get("level") if isinstance(targets[0], dict) else targets[0]

        if t1 is None:
            # Fallback: try to get T1 directly
            t1 = plan.get("t1") or plan.get("T1")

        if t1 is None:
            return TargetReachability(
                score=0.7,  # Default moderate score
                distance_to_target=0,
                atr_current=0,
                atr_at_entry=0,
                bars_at_current_pace=0,
                notes="no_target_defined",
            )

        t1 = float(t1)
        atr_entry = float(entry_indicators.get("atr", 1.0))
        atr_current = float(current_indicators.get("atr", atr_entry))
        side_upper = side.upper()

        # Calculate distance to target
        distance_to_target = (
            (t1 - current_price) if side_upper == "BUY"
            else (current_price - t1)
        )

        # If already past target, full score
        if distance_to_target <= 0:
            return TargetReachability(
                score=1.0,
                distance_to_target=0,
                atr_current=atr_current,
                atr_at_entry=atr_entry,
                bars_at_current_pace=0,
                notes="target_reached_or_passed",
            )

        # Estimate bars to reach target at current ATR pace
        bars_at_current_pace = distance_to_target / max(atr_current * self.expected_atr_per_bar, 0.01)

        # Calculate score
        score = 1.0
        notes_parts = []

        # ATR contraction check
        if atr_current < atr_entry / self.atr_expansion_factor:
            atr_penalty = min(0.3, (atr_entry - atr_current) / atr_entry)
            score -= atr_penalty
            notes_parts.append(f"ATR_contracted_{atr_penalty:.0%}")

        # Distance check (too far away)
        if distance_to_target > atr_current * 5:
            distance_penalty = min(0.3, (distance_to_target - atr_current * 5) / (atr_current * 10))
            score -= distance_penalty
            notes_parts.append(f"target_far_{distance_to_target/atr_current:.1f}xATR")

        # Time-based probability (more bars = less likely to reach)
        if bars_at_current_pace > 20:
            time_penalty = min(0.2, (bars_at_current_pace - 20) / 50)
            score -= time_penalty
            notes_parts.append(f"~{bars_at_current_pace:.0f}_bars_needed")

        notes = "; ".join(notes_parts) if notes_parts else "target_reachable"

        return TargetReachability(
            score=max(0, min(1, score)),
            distance_to_target=distance_to_target,
            atr_current=atr_current,
            atr_at_entry=atr_entry,
            bars_at_current_pace=bars_at_current_pace,
            notes=notes,
        )

    def _calculate_combined_score(
        self,
        momentum: MomentumHealth,
        volume: VolumeHealth,
        structure: StructureIntegrity,
        target: TargetReachability,
        category: str,
    ) -> float:
        """
        Calculate weighted combined score with category-specific adjustments.

        Breakouts: Higher weight on momentum and structure
        Reversions: Higher weight on structure and target
        """
        # Use category-specific weights from config
        if category == "breakout":
            weights = self.breakout_weights
        else:  # reversion (includes LEVEL and REVERSION categories)
            weights = self.reversion_weights

        combined = (
            momentum.score * weights["momentum"] +
            volume.score * weights["volume"] +
            structure.score * weights["structure"] +
            target.score * weights["target"]
        )

        # Penalty for critical failure (structure breach)
        # Apply different caps per category - breakout needs hard exit, reversion softer
        if structure.breach_detected:
            if category == "breakout":
                combined = min(combined, self.breakout_structure_breach_cap)  # Hard cap at 0.35
            elif category == "reversion":
                combined = min(combined, self.reversion_structure_breach_cap)  # Softer cap at 0.50

        return max(0, min(1, combined))

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear check cache for a symbol or all symbols."""
        if symbol:
            self._last_check.pop(symbol, None)
            self._last_health.pop(symbol, None)
            self._consecutive_failures.pop(symbol, None)
        else:
            self._last_check.clear()
            self._consecutive_failures.clear()
            self._last_health.clear()
