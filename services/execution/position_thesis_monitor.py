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

# Breakout setups: Need momentum continuation
BREAKOUT_SETUPS = frozenset([
    "orb_breakout_long", "orb_breakout_short",
    "orb_breakdown_long", "orb_breakdown_short",
    "breakout_long", "breakout_short",
    "level_breakout_long", "level_breakout_short",
    "flag_continuation_long", "flag_continuation_short",
    "squeeze_release_long", "squeeze_release_short",
    "momentum_breakout_long", "momentum_breakout_short",
    "range_breakout_long", "range_breakout_short",
    "gap_breakout_long", "gap_breakout_short",
    "resistance_breakout_long", "support_breakdown_short",
    "break_of_structure_long", "break_of_structure_short",
    "change_of_character_long", "change_of_character_short",
    "equilibrium_breakout_long", "equilibrium_breakout_short",
])

# Reversion setups: Benefit from counter-moves
REVERSION_SETUPS = frozenset([
    "failure_fade_long", "failure_fade_short",
    "volume_spike_reversal_long", "volume_spike_reversal_short",
    "gap_fill_long", "gap_fill_short",
    "vwap_mean_reversion_long", "vwap_mean_reversion_short",
    "liquidity_sweep_long", "liquidity_sweep_short",
    "fair_value_gap_long", "fair_value_gap_short",
    "premium_zone_short", "discount_zone_long",
    "order_block_long", "order_block_short",
])


def classify_setup_category(setup_type: str) -> str:
    """Classify setup into category for thesis monitoring."""
    setup_lower = setup_type.lower()
    if setup_lower in BREAKOUT_SETUPS or any(k in setup_lower for k in ["breakout", "breakdown", "break_of", "squeeze", "flag_cont"]):
        return "breakout"
    elif setup_lower in REVERSION_SETUPS or any(k in setup_lower for k in ["fade", "reversal", "fill", "reversion", "sweep", "premium", "discount", "order_block"]):
        return "reversion"
    else:
        # Default to breakout logic (more conservative)
        return "breakout"


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

        # Target reachability thresholds
        target_cfg = tm_cfg["target_reachability"]
        self.atr_expansion_factor = float(target_cfg["atr_expansion_factor"])
        self.min_time_to_target_minutes = float(target_cfg["min_time_to_target_minutes"])

        # Combined threshold for exit
        self.breakout_exit_threshold = float(tm_cfg["breakout_exit_threshold"])
        self.reversion_exit_threshold = float(tm_cfg["reversion_exit_threshold"])

        # Weight configuration for combined score
        weights_cfg = tm_cfg["weights"]
        self.weight_momentum = float(weights_cfg["momentum"])
        self.weight_volume = float(weights_cfg["volume"])
        self.weight_structure = float(weights_cfg["structure"])
        self.weight_target = float(weights_cfg["target"])

        # Cache for last check timestamps (avoid excessive computation)
        self._last_check: Dict[str, float] = {}

        self.log.info(
            f"[ThesisMonitor] Initialized: enabled={self.enabled}, "
            f"breakout_threshold={self.breakout_exit_threshold}, "
            f"reversion_threshold={self.reversion_exit_threshold}"
        )

    def check_thesis(
        self,
        symbol: str,
        side: str,
        current_price: float,
        plan: Dict[str, Any],
        df_5m: Optional[pd.DataFrame] = None,
    ) -> Optional[ThesisHealth]:
        """
        Check if trade thesis remains valid.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            current_price: Current market price
            plan: Trade plan dict with entry indicators, targets, etc.
            df_5m: Optional 5-minute OHLCV DataFrame for live indicator calculation

        Returns:
            ThesisHealth if check performed, None if skipped (disabled/cached)
        """
        if not self.enabled:
            return None

        # Rate limit checks
        import time
        current_time = time.time()
        last_check = self._last_check.get(symbol, 0)
        if current_time - last_check < self.check_interval_seconds:
            return None
        self._last_check[symbol] = current_time

        # Get setup type and category
        setup_type = plan.get("setup_type", "unknown")
        category = classify_setup_category(setup_type)

        # Get entry-time indicators from plan
        entry_indicators = plan.get("indicators", {})
        entry_price = float(plan.get("entry", plan.get("entry_price", current_price)))

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
        should_exit = combined_score < threshold

        exit_reason = None
        if should_exit:
            # Determine primary reason for exit
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
                # Brief log for healthy positions
                self.log.debug(
                    f"[ThesisMonitor] {symbol} | {setup_type} ({category}) | "
                    f"Score: {combined_score:.2f} (threshold={threshold}) | "
                    f"M={momentum.score:.2f} V={volume.score:.2f} "
                    f"S={structure.score:.2f} T={target.score:.2f} | OK"
                )

        return health

    def _calculate_current_indicators(self, df_5m: pd.DataFrame) -> Dict[str, float]:
        """Calculate current indicators from 5m bars."""
        if df_5m is None or len(df_5m) < 14:
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
            # Reversions benefit from exhaustion fading
            score = 1.0
            notes_parts = []

            # RSI should be moving toward neutral (away from extremes)
            if side_upper == "BUY":
                # Long reversion: RSI should be recovering from oversold
                if rsi_current < 30:
                    score -= 0.1  # Still oversold - might continue
                    notes_parts.append("RSI_still_oversold")
                elif rsi_current > 70:
                    score -= 0.4  # Overbought - reversal thesis failing
                    notes_parts.append("RSI_overbought_thesis_failed")
            else:
                # Short reversion: RSI should be declining from overbought
                if rsi_current > 70:
                    score -= 0.1  # Still overbought - might continue
                    notes_parts.append("RSI_still_overbought")
                elif rsi_current < 30:
                    score -= 0.4  # Oversold - reversal thesis failing
                    notes_parts.append("RSI_oversold_thesis_failed")

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
            # Volume exhaustion is expected for reversions
            score = 0.8  # Start with good score (volume decline is OK)
            notes_parts = []

            # Only penalize if volume surges AGAINST the reversion
            if vol_ratio_current > vol_ratio_entry * 1.5:
                score -= 0.3
                notes_parts.append("volume_surge_against_thesis")

            notes = "; ".join(notes_parts) if notes_parts else "volume_exhaustion_expected"

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
        setup_type = plan.get("setup_type", "")
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
            vwap = levels.get("vwap") or levels.get("VWAP")

            if vwap is not None:
                # Check if moving toward VWAP
                entry_to_vwap = abs(entry_price - vwap)
                current_to_vwap = abs(current_price - vwap)

                if current_to_vwap > entry_to_vwap * 1.2:
                    # Moving away from VWAP - thesis failing
                    score -= 0.4
                    breach_detected = True
                    breach_reason = "moving_away_from_vwap"
                    notes_parts.append("reversion_failing_away_from_mean")
                elif current_to_vwap < entry_to_vwap * 0.5:
                    # Good progress toward VWAP
                    notes_parts.append("reversion_progressing_well")

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
        bars_at_current_pace = distance_to_target / max(atr_current * 0.5, 0.01)  # Assume 0.5 ATR per bar average

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
        if category == "breakout":
            # Breakouts: momentum and structure are critical
            weights = {
                "momentum": 0.35,
                "volume": 0.20,
                "structure": 0.30,
                "target": 0.15,
            }
        else:  # reversion
            # Reversions: structure (mean reversion progress) is critical
            weights = {
                "momentum": 0.20,
                "volume": 0.15,
                "structure": 0.40,
                "target": 0.25,
            }

        combined = (
            momentum.score * weights["momentum"] +
            volume.score * weights["volume"] +
            structure.score * weights["structure"] +
            target.score * weights["target"]
        )

        # Penalty for any critical failure (structure breach)
        if structure.breach_detected:
            combined = min(combined, 0.35)

        return max(0, min(1, combined))

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear check cache for a symbol or all symbols."""
        if symbol:
            self._last_check.pop(symbol, None)
        else:
            self._last_check.clear()
