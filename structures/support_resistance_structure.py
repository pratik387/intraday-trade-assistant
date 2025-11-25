"""
Support/Resistance Structure Implementation

This module implements support and resistance level-based trading structures including:
- Support bounce (price bounces off support level - long bias)
- Resistance rejection (price rejects at resistance level - short bias)
- Support breakdown (price breaks below support - short bias)
- Resistance breakout (price breaks above resistance - long bias)

All trading parameters must be explicitly configured - no hardcoded assumptions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    MarketContext, StructureEvent, TradePlan, RiskParams, ExitLevels, StructureAnalysis
)

logger = get_agent_logger()


@dataclass
class SupportResistanceLevels:
    """Support/Resistance level information."""
    support_levels: List[float]
    resistance_levels: List[float]
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    support_distance_pct: Optional[float]
    resistance_distance_pct: Optional[float]
    support_touches: int
    resistance_touches: int
    support_strength: float
    resistance_strength: float


class SupportResistanceStructure(BaseStructure):
    """
    Support/Resistance-based structure detection and strategy planning.

    Handles multiple S/R strategies:
    1. Support bounce - price bounces off support level (long bias)
    2. Resistance rejection - price rejects at resistance level (short bias)
    3. Support breakdown - price breaks below support (short bias)
    4. Resistance breakout - price breaks above resistance (long bias)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Support/Resistance structure with configuration.

        CRITICAL: All trading parameters must be provided in config.
        No hardcoded defaults for trading decisions.
        """
        super().__init__(config)
        self.structure_type = "support_resistance"

        # KeyError if missing trading parameters

        # Level detection parameters
        self.min_touches = config["min_touches"]
        self.bounce_tolerance_pct = config["bounce_tolerance_pct"]
        self.require_volume_spike = config["require_volume_spike"]
        self.min_volume_mult = config["min_volume_mult"]

        # Level strength parameters
        self.min_level_age_bars = config["min_level_age_bars"]
        self.max_level_distance_pct = config["max_level_distance_pct"]
        self.level_merge_tolerance_pct = config["level_merge_tolerance_pct"]

        # Breakout parameters
        self.breakout_buffer_pct = config["breakout_buffer_pct"]
        self.min_breakout_volume_mult = config["min_breakout_volume_mult"]

        # Risk management parameters
        self.min_stop_distance_pct = config["min_stop_distance_pct"]
        self.stop_distance_mult = config["stop_distance_mult"]
        self.level_buffer_mult = config["level_buffer_mult"]

        # Target parameters
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]

        # Confidence scoring
        self.confidence_strong_level = config["confidence_strong_level"]
        self.confidence_weak_level = config["confidence_weak_level"]
        self.confidence_breakout = config["confidence_breakout"]
        self.confidence_bounce = config["confidence_bounce"]

        # Dual-mode entry configuration (Professional S/R Trading)
        self.entry_mode = config["entry_mode"]  # "immediate", "retest", or "conditional"
        self.immediate_entry_distance_pct = config.get("immediate_entry_distance_pct", 0.3)  # For conditional mode
        self.retest_entry_zone_width_atr = config["retest_entry_zone_width_atr"]
        self.retest_timeout_minutes = config["retest_timeout_minutes"]

        # Target structure type filter - each instance only detects its specific structure
        # Valid values: "support_bounce_long", "resistance_bounce_short", "support_breakdown_short", "resistance_breakout_long", or "all"
        self.target_structure_type = config.get("target_structure_type", "all")

        logger.debug(f"S/R: Initialized with min touches: {self.min_touches}, tolerance: {self.bounce_tolerance_pct}%")
        logger.debug(f"S/R: Volume spike required: {self.require_volume_spike}, min mult: {self.min_volume_mult}")
        logger.debug(f"S/R: Dual-mode entry: mode={self.entry_mode}, immediate_dist={self.immediate_entry_distance_pct}%, retest_zone={self.retest_entry_zone_width_atr}ATR")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect Support/Resistance-based structures in the market context."""

        logger.debug(f"SR_DETECT: Starting detection for {context.symbol}")

        try:
            # Extract S/R levels and context
            sr_info = self._extract_support_resistance_levels(context)
            if not sr_info:
                logger.debug(f"SR_DETECT: {context.symbol} - REJECTED: Cannot extract S/R levels")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Support/Resistance levels not available"
                )

            # Detect different S/R strategies based on target_structure_type filter
            events = []
            max_quality = 0.0
            target = self.target_structure_type

            # 1. Support Bounce - only if target matches or "all"
            if target in ("all", "support_bounce_long"):
                bounce_events, bounce_quality = self._detect_support_bounce(context, sr_info)
                events.extend(bounce_events)
                max_quality = max(max_quality, bounce_quality)

            # 2. Resistance Rejection - only if target matches or "all"
            if target in ("all", "resistance_bounce_short"):
                rejection_events, rejection_quality = self._detect_resistance_rejection(context, sr_info)
                events.extend(rejection_events)
                max_quality = max(max_quality, rejection_quality)

            # 3. Support Breakdown - only if target matches or "all"
            if target in ("all", "support_breakdown_short"):
                breakdown_events, breakdown_quality = self._detect_support_breakdown(context, sr_info)
                events.extend(breakdown_events)
                max_quality = max(max_quality, breakdown_quality)

            # 4. Resistance Breakout - only if target matches or "all"
            if target in ("all", "resistance_breakout_long"):
                breakout_events, breakout_quality = self._detect_resistance_breakout(context, sr_info)
                events.extend(breakout_events)
                max_quality = max(max_quality, breakout_quality)

            structure_detected = len(events) > 0
            rejection_reason = None if structure_detected else "No S/R setups detected"

            if structure_detected:
                logger.debug(f"SR_DETECT: ✅ {context.symbol} - DETECTED: {len(events)} events, quality: {max_quality:.2f}")
            else:
                logger.debug(f"SR_DETECT: {context.symbol} - REJECTED: No S/R setups detected")

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=events,
                quality_score=max_quality,
                rejection_reason=rejection_reason
            )

        except Exception as e:
            logger.error(f"S/R: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _extract_support_resistance_levels(self, context: MarketContext) -> Optional[SupportResistanceLevels]:
        """Extract support and resistance levels from market context."""

        try:
            df = context.df_5m
            logger.debug(f"SR_DETECT: {context.symbol} - Extracting S/R levels from {len(df)} bars")
            if len(df) < 20:  # Need sufficient data for S/R analysis
                logger.debug(f"SR_DETECT: {context.symbol} - REJECTED: Insufficient data ({len(df)} bars, need 20)")
                return None

            current_price = float(context.current_price)

            # Extract key levels from context if available
            support_levels = []
            resistance_levels = []

            # Add levels from context (PDH, PDL, etc.)
            if context.pdh:
                resistance_levels.append(context.pdh)
            if context.pdl:
                support_levels.append(context.pdl)

            # Add pivot levels calculated from recent data
            pivot_levels = self._calculate_pivot_levels(df)
            support_levels.extend(pivot_levels['support'])
            resistance_levels.extend(pivot_levels['resistance'])

            # Filter and merge nearby levels
            support_levels = self._merge_nearby_levels(support_levels, current_price)
            resistance_levels = self._merge_nearby_levels(resistance_levels, current_price)

            # Find nearest levels
            support_below = [s for s in support_levels if s < current_price]
            resistance_above = [r for r in resistance_levels if r > current_price]

            nearest_support = max(support_below) if support_below else None
            nearest_resistance = min(resistance_above) if resistance_above else None

            # Calculate distances
            support_distance_pct = None
            resistance_distance_pct = None

            if nearest_support:
                support_distance_pct = abs(current_price - nearest_support) / current_price * 100

            if nearest_resistance:
                resistance_distance_pct = abs(nearest_resistance - current_price) / current_price * 100

            # Calculate level strength (touches and age)
            support_touches = self._count_level_touches(df, nearest_support) if nearest_support else 0
            resistance_touches = self._count_level_touches(df, nearest_resistance) if nearest_resistance else 0

            support_strength = self._calculate_level_strength(df, nearest_support, support_touches)
            resistance_strength = self._calculate_level_strength(df, nearest_resistance, resistance_touches)

            logger.debug(f"SR_DETECT: {context.symbol} - Support: {nearest_support}, Resistance: {nearest_resistance}")
            logger.debug(f"SR_DETECT: {context.symbol} - Support distance: {support_distance_pct:.2f}% (touches={support_touches}), Resistance distance: {resistance_distance_pct:.2f}% (touches={resistance_touches})" if support_distance_pct and resistance_distance_pct else f"SR_DETECT: {context.symbol} - Support touches: {support_touches}, Resistance touches: {resistance_touches}")

            return SupportResistanceLevels(
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                support_distance_pct=support_distance_pct,
                resistance_distance_pct=resistance_distance_pct,
                support_touches=support_touches,
                resistance_touches=resistance_touches,
                support_strength=support_strength,
                resistance_strength=resistance_strength
            )

        except Exception as e:
            logger.error(f"S/R: Error extracting S/R levels for {context.symbol}: {e}")
            return None

    def _calculate_pivot_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate pivot-based support and resistance levels."""

        try:
            levels = {'support': [], 'resistance': []}

            # Use recent high/low for pivot calculation
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            recent_close = df['close'].iloc[-1]

            # Classic pivot calculation
            pivot = (recent_high + recent_low + recent_close) / 3

            # Calculate support and resistance levels
            r1 = 2 * pivot - recent_low
            s1 = 2 * pivot - recent_high
            r2 = pivot + (recent_high - recent_low)
            s2 = pivot - (recent_high - recent_low)

            levels['resistance'].extend([r1, r2])
            levels['support'].extend([s1, s2])

            # Add swing highs and lows as levels
            swing_highs = self._find_swing_points(df['high'], 'high')
            swing_lows = self._find_swing_points(df['low'], 'low')

            levels['resistance'].extend(swing_highs)
            levels['support'].extend(swing_lows)

            return levels

        except Exception as e:
            logger.warning(f"S/R: Error calculating pivot levels: {e}")
            return {'support': [], 'resistance': []}

    def _find_swing_points(self, series: pd.Series, point_type: str) -> List[float]:
        """Find swing highs or lows in price series."""

        try:
            swings = []
            window = 5  # Look for swings in 5-bar windows

            for i in range(window, len(series) - window):
                if point_type == 'high':
                    # Check if current bar is highest in window
                    if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                        swings.append(series.iloc[i])
                else:  # 'low'
                    # Check if current bar is lowest in window
                    if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                        swings.append(series.iloc[i])

            # Return most recent swings
            return swings[-3:] if len(swings) > 3 else swings

        except Exception:
            return []

    def _merge_nearby_levels(self, levels: List[float], current_price: float) -> List[float]:
        """Merge levels that are too close together."""

        if not levels:
            return []

        try:
            # Sort levels
            sorted_levels = sorted(set(levels))
            merged_levels = []

            for level in sorted_levels:
                # Skip levels too far from current price
                distance_pct = abs(level - current_price) / current_price * 100
                if distance_pct > self.max_level_distance_pct:
                    continue

                # Check if this level is too close to existing merged levels
                too_close = False
                for existing in merged_levels:
                    merge_distance_pct = abs(level - existing) / existing * 100
                    if merge_distance_pct < self.level_merge_tolerance_pct:
                        too_close = True
                        break

                if not too_close:
                    merged_levels.append(level)

            return merged_levels

        except Exception:
            return levels

    def _count_level_touches(self, df: pd.DataFrame, level: float) -> int:
        """Count how many times price has touched a level."""

        if level is None:
            return 0

        try:
            tolerance = level * self.bounce_tolerance_pct / 100
            touches = 0

            # Check recent bars for touches
            for i in range(max(0, len(df) - 50), len(df)):  # Last 50 bars
                bar_high = df['high'].iloc[i]
                bar_low = df['low'].iloc[i]

                # Check if bar touched the level
                if bar_low <= level + tolerance and bar_high >= level - tolerance:
                    touches += 1

            return touches

        except Exception:
            return 0

    def _calculate_level_strength(self, df: pd.DataFrame, level: float, touches: int) -> float:
        """Calculate the strength of a support/resistance level."""

        if level is None:
            return 0.0

        try:
            # Base strength from touches
            strength = min(touches * 20, 80)  # Max 80 from touches

            # Bonus for level age (how long it's been respected)
            level_age = self._calculate_level_age(df, level)
            age_bonus = min(level_age * 2, 20)  # Max 20 from age

            total_strength = min(strength + age_bonus, 100)
            return total_strength

        except Exception:
            return 0.0

    def _calculate_level_age(self, df: pd.DataFrame, level: float) -> int:
        """Calculate how many bars ago this level was first established."""

        try:
            tolerance = level * self.bounce_tolerance_pct / 100

            # Look backwards to find first touch
            for i in range(len(df) - 1, -1, -1):
                bar_high = df['high'].iloc[i]
                bar_low = df['low'].iloc[i]

                if bar_low <= level + tolerance and bar_high >= level - tolerance:
                    return len(df) - 1 - i

            return 0

        except Exception:
            return 0

    def _determine_entry_mode(self, distance_pct: float) -> str:
        """Determine actual entry mode based on config and price distance from level.

        For 'conditional' mode:
        - If price is within immediate_entry_distance_pct of level → 'immediate' (price is at optimal zone)
        - If price is further away → 'retest' (wait for pullback to level)

        For 'immediate' or 'retest' modes, return as-is.
        """
        if self.entry_mode == "conditional":
            if distance_pct <= self.immediate_entry_distance_pct:
                return "immediate"
            else:
                return "retest"
        return self.entry_mode

    def _detect_support_bounce(self, context: MarketContext, sr_info: SupportResistanceLevels) -> Tuple[List[StructureEvent], float]:
        """Detect support bounce opportunities (long bias)."""

        events = []
        quality_score = 0.0

        if not sr_info.nearest_support or sr_info.support_touches < self.min_touches:
            logger.debug(f"SR_DETECT: {context.symbol} - Support bounce REJECTED: No valid support (touches={sr_info.support_touches}, need {self.min_touches})")
            return events, quality_score

        current_price = context.current_price
        support_level = sr_info.nearest_support

        # Check if price is near support (within tolerance)
        distance_pct = abs(current_price - support_level) / support_level * 100
        if distance_pct > self.bounce_tolerance_pct:
            logger.debug(f"SR_DETECT: {context.symbol} - Support bounce REJECTED: Price too far from support ({distance_pct:.2f}% > {self.bounce_tolerance_pct}%)")
            return events, quality_score

        # Check if price is above support (bounce scenario)
        if current_price <= support_level:
            logger.debug(f"SR_DETECT: {context.symbol} - Support bounce REJECTED: Price below support (not bounce scenario)")
            return events, quality_score

        # Check volume confirmation if required
        volume_ok = True
        if self.require_volume_spike:
            volume_ok = self._check_volume_confirmation(context)
            logger.debug(f"S/R: {context.symbol} - Volume confirmation: {volume_ok}")

        if volume_ok:
            confidence = self._calculate_institutional_strength(context, sr_info, "support_bounce", "long", volume_ok)

            quality_score = min(95.0, sr_info.support_strength + (confidence * 20))

            # Determine actual entry mode (conditional logic based on distance)
            actual_entry_mode = self._determine_entry_mode(distance_pct)

            # Calculate retest zone for dual-mode entry
            atr = self._get_atr(context)
            retest_zone = None
            if actual_entry_mode == "retest":
                # For longs at support: zone is below current price, near support level
                zone_half_width = self.retest_entry_zone_width_atr * atr
                retest_zone = [
                    support_level - zone_half_width,  # Lower bound
                    support_level + zone_half_width   # Upper bound
                ]
                logger.debug(f"S/R: {context.symbol} - Retest zone for support_bounce_long: [{retest_zone[0]:.2f}, {retest_zone[1]:.2f}]")

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type="support_bounce_long",
                side="long",
                confidence=confidence,
                levels={"support": support_level},
                context={
                    "support_touches": sr_info.support_touches,
                    "support_strength": sr_info.support_strength,
                    "distance_pct": distance_pct,
                    "volume_confirmation": volume_ok,
                    "entry_mode": actual_entry_mode,
                    "retest_zone": retest_zone
                },
                price=current_price
            )
            events.append(event)

            logger.debug(f"S/R: {context.symbol} - Support bounce LONG detected: level {support_level:.2f}, touches {sr_info.support_touches}, confidence {confidence:.2f}, entry_mode={actual_entry_mode}")

        return events, quality_score

    def _detect_resistance_rejection(self, context: MarketContext, sr_info: SupportResistanceLevels) -> Tuple[List[StructureEvent], float]:
        """Detect resistance rejection opportunities (short bias)."""

        events = []
        quality_score = 0.0

        if not sr_info.nearest_resistance or sr_info.resistance_touches < self.min_touches:
            logger.debug(f"S/R: {context.symbol} - No valid resistance for rejection: touches {sr_info.resistance_touches}")
            return events, quality_score

        current_price = context.current_price
        resistance_level = sr_info.nearest_resistance

        # Check if price is near resistance (within tolerance)
        distance_pct = abs(current_price - resistance_level) / resistance_level * 100
        if distance_pct > self.bounce_tolerance_pct:
            logger.debug(f"S/R: {context.symbol} - Price too far from resistance: {distance_pct:.2f}% > {self.bounce_tolerance_pct}%")
            return events, quality_score

        # Check if price is below resistance (rejection scenario)
        if current_price >= resistance_level:
            logger.debug(f"S/R: {context.symbol} - Price above resistance, not rejection scenario")
            return events, quality_score

        # Check volume confirmation if required
        volume_ok = True
        if self.require_volume_spike:
            volume_ok = self._check_volume_confirmation(context)
            logger.debug(f"S/R: {context.symbol} - Volume confirmation: {volume_ok}")

        if volume_ok:
            confidence = self._calculate_institutional_strength(context, sr_info, "resistance_bounce", "short", volume_ok)

            quality_score = min(95.0, sr_info.resistance_strength + (confidence * 20))

            # Determine actual entry mode (conditional logic based on distance)
            actual_entry_mode = self._determine_entry_mode(distance_pct)

            # Calculate retest zone for dual-mode entry
            atr = self._get_atr(context)
            retest_zone = None
            if actual_entry_mode == "retest":
                # For shorts at resistance: zone is above current price, near resistance level
                zone_half_width = self.retest_entry_zone_width_atr * atr
                retest_zone = [
                    resistance_level - zone_half_width,  # Lower bound
                    resistance_level + zone_half_width   # Upper bound
                ]
                logger.debug(f"S/R: {context.symbol} - Retest zone for resistance_bounce_short: [{retest_zone[0]:.2f}, {retest_zone[1]:.2f}]")

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type="resistance_bounce_short",
                side="short",
                confidence=confidence,
                levels={"resistance": resistance_level},
                context={
                    "resistance_touches": sr_info.resistance_touches,
                    "resistance_strength": sr_info.resistance_strength,
                    "distance_pct": distance_pct,
                    "volume_confirmation": volume_ok,
                    "entry_mode": actual_entry_mode,
                    "retest_zone": retest_zone
                },
                price=current_price
            )
            events.append(event)

            logger.debug(f"S/R: {context.symbol} - Resistance rejection SHORT detected: level {resistance_level:.2f}, touches {sr_info.resistance_touches}, confidence {confidence:.2f}, entry_mode={actual_entry_mode}")

        return events, quality_score

    def _detect_support_breakdown(self, context: MarketContext, sr_info: SupportResistanceLevels) -> Tuple[List[StructureEvent], float]:
        """Detect support breakdown opportunities (short bias)."""

        events = []
        quality_score = 0.0

        if not sr_info.nearest_support or sr_info.support_touches < self.min_touches:
            logger.debug(f"S/R: {context.symbol} - No valid support for breakdown")
            return events, quality_score

        current_price = context.current_price
        support_level = sr_info.nearest_support

        # Check if price broke below support with buffer
        breakdown_threshold = support_level * (1 - self.breakout_buffer_pct / 100)
        if current_price > breakdown_threshold:
            logger.debug(f"S/R: {context.symbol} - Price not below breakdown threshold: {current_price:.2f} > {breakdown_threshold:.2f}")
            return events, quality_score

        # Check volume confirmation for breakout
        volume_ok = self._check_breakout_volume_confirmation(context)
        logger.debug(f"S/R: {context.symbol} - Breakdown volume confirmation: {volume_ok}")

        if volume_ok:
            confidence = self._calculate_institutional_strength(context, sr_info, "support_breakout", "long", True)
            quality_score = min(90.0, sr_info.support_strength * 0.8 + (confidence * 25))

            # Distance from broken support level
            breakdown_distance_pct = abs(current_price - support_level) / support_level * 100

            # Determine actual entry mode (conditional logic based on distance from level)
            actual_entry_mode = self._determine_entry_mode(breakdown_distance_pct)

            # Calculate retest zone for retest mode
            atr = self._get_atr(context)
            retest_zone = None
            if actual_entry_mode == "retest":
                # For shorts after breakdown: wait for price to retest (bounce back to) support level
                zone_half_width = self.retest_entry_zone_width_atr * atr
                retest_zone = [
                    support_level - zone_half_width,
                    support_level + zone_half_width
                ]

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type="support_breakdown_short",
                side="short",
                confidence=confidence,
                levels={"support": support_level},
                context={
                    "support_touches": sr_info.support_touches,
                    "support_strength": sr_info.support_strength,
                    "breakdown_distance_pct": breakdown_distance_pct,
                    "volume_confirmation": volume_ok,
                    "entry_mode": actual_entry_mode,
                    "retest_zone": retest_zone
                },
                price=current_price
            )
            events.append(event)

            logger.debug(f"S/R: {context.symbol} - Support breakdown SHORT detected: level {support_level:.2f}, breakdown at {current_price:.2f}, confidence {confidence:.2f}, entry_mode={actual_entry_mode}")

        return events, quality_score

    def _detect_resistance_breakout(self, context: MarketContext, sr_info: SupportResistanceLevels) -> Tuple[List[StructureEvent], float]:
        """Detect resistance breakout opportunities (long bias)."""

        events = []
        quality_score = 0.0

        if not sr_info.nearest_resistance or sr_info.resistance_touches < self.min_touches:
            logger.debug(f"S/R: {context.symbol} - No valid resistance for breakout")
            return events, quality_score

        current_price = context.current_price
        resistance_level = sr_info.nearest_resistance

        # Check if price broke above resistance with buffer
        breakout_threshold = resistance_level * (1 + self.breakout_buffer_pct / 100)
        if current_price < breakout_threshold:
            logger.debug(f"S/R: {context.symbol} - Price not above breakout threshold: {current_price:.2f} < {breakout_threshold:.2f}")
            return events, quality_score

        # Check volume confirmation for breakout
        volume_ok = self._check_breakout_volume_confirmation(context)
        logger.debug(f"S/R: {context.symbol} - Breakout volume confirmation: {volume_ok}")

        if volume_ok:
            confidence = self._calculate_institutional_strength(context, sr_info, "support_breakout", "long", True)
            quality_score = min(90.0, sr_info.resistance_strength * 0.8 + (confidence * 25))

            # Distance from broken resistance level
            breakout_distance_pct = abs(current_price - resistance_level) / resistance_level * 100

            # Determine actual entry mode (conditional logic based on distance from level)
            actual_entry_mode = self._determine_entry_mode(breakout_distance_pct)

            # Calculate retest zone for retest mode
            atr = self._get_atr(context)
            retest_zone = None
            if actual_entry_mode == "retest":
                # For longs after breakout: wait for price to retest (pullback to) resistance level
                zone_half_width = self.retest_entry_zone_width_atr * atr
                retest_zone = [
                    resistance_level - zone_half_width,
                    resistance_level + zone_half_width
                ]

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type="resistance_breakout_long",
                side="long",
                confidence=confidence,
                levels={"resistance": resistance_level},
                context={
                    "resistance_touches": sr_info.resistance_touches,
                    "resistance_strength": sr_info.resistance_strength,
                    "breakout_distance_pct": breakout_distance_pct,
                    "volume_confirmation": volume_ok,
                    "entry_mode": actual_entry_mode,
                    "retest_zone": retest_zone
                },
                price=current_price
            )
            events.append(event)

            logger.debug(f"S/R: {context.symbol} - Resistance breakout LONG detected: level {resistance_level:.2f}, breakout at {current_price:.2f}, confidence {confidence:.2f}, entry_mode={actual_entry_mode}")

        return events, quality_score

    def _check_volume_confirmation(self, context: MarketContext) -> bool:
        """Check if current volume supports the setup."""
        try:
            if context.indicators and 'vol_z' in context.indicators:
                current_vol_z = context.indicators['vol_z']
                return current_vol_z >= self.min_volume_mult

            # Fallback to DataFrame if available
            df = context.df_5m
            if 'vol_z' in df.columns:
                current_vol_z = float(df['vol_z'].iloc[-1])
                return current_vol_z >= self.min_volume_mult

            return False
        except Exception:
            return False

    def _check_breakout_volume_confirmation(self, context: MarketContext) -> bool:
        """Check if current volume supports breakout/breakdown."""
        try:
            if context.indicators and 'vol_z' in context.indicators:
                current_vol_z = context.indicators['vol_z']
                return current_vol_z >= self.min_breakout_volume_mult

            # Fallback to DataFrame if available
            df = context.df_5m
            if 'vol_z' in df.columns:
                current_vol_z = float(df['vol_z'].iloc[-1])
                return current_vol_z >= self.min_breakout_volume_mult

            return False
        except Exception:
            return False

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for S/R setups."""

        logger.debug(f"S/R: Planning long strategy for {context.symbol} - {event.structure_type}")

        # Calculate risk parameters
        risk_params = self.calculate_risk_params(context, event, "long")

        # Calculate exit levels
        exit_levels = self.get_exit_levels(context, event, "long")

        # Determine position size based on risk
        entry_price = context.current_price
        qty, notional = self._calculate_position_size(entry_price, risk_params.hard_sl, context)

        plan = TradePlan(
            symbol=context.symbol,
            side="long",
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=event.confidence,
            notes={
                "sr_level": event.levels.get("support") or event.levels.get("resistance"),
                "setup_reason": f"S/R {event.structure_type}",
                **event.context
            }
        )

        logger.debug(f"S/R: {context.symbol} - Long strategy planned: entry {entry_price:.2f}, SL {risk_params.hard_sl:.2f}, qty {qty}")

        return plan

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for S/R setups."""

        logger.debug(f"S/R: Planning short strategy for {context.symbol} - {event.structure_type}")

        # Calculate risk parameters
        risk_params = self.calculate_risk_params(context, event, "short")

        # Calculate exit levels
        exit_levels = self.get_exit_levels(context, event, "short")

        # Determine position size based on risk
        entry_price = context.current_price
        qty, notional = self._calculate_position_size(entry_price, risk_params.hard_sl, context)

        plan = TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=event.confidence,
            notes={
                "sr_level": event.levels.get("support") or event.levels.get("resistance"),
                "setup_reason": f"S/R {event.structure_type}",
                **event.context
            }
        )

        logger.debug(f"S/R: {context.symbol} - Short strategy planned: entry {entry_price:.2f}, SL {risk_params.hard_sl:.2f}, qty {qty}")

        return plan

    def calculate_risk_params(self, context: MarketContext, event: StructureEvent, side: str) -> RiskParams:
        """Calculate risk parameters for S/R strategies."""

        entry_price = context.current_price
        sr_level = event.levels.get("support") or event.levels.get("resistance", entry_price)

        # Get ATR for stop calculation
        atr = self._get_atr(context)

        if side == "long":
            if "support" in event.structure_type:
                # Long on support bounce - stop below support
                level_stop = sr_level * (1 - self.level_buffer_mult / 100)
            else:
                # Long on resistance breakout - stop below resistance
                level_stop = sr_level * (1 - self.level_buffer_mult / 100)

            atr_stop = entry_price - (atr * self.stop_distance_mult) if atr > 0 else entry_price * 0.99

            # Use more conservative stop
            calculated_stop = min(level_stop, atr_stop)

            # Ensure minimum stop distance
            min_stop = entry_price * (1 - self.min_stop_distance_pct / 100)
            hard_sl = min(calculated_stop, min_stop)

            logger.debug(f"S/R: {context.symbol} - Long stops: Level {level_stop:.2f}, ATR {atr_stop:.2f}, Min {min_stop:.2f}, Final {hard_sl:.2f}")

        else:  # short
            if "resistance" in event.structure_type:
                # Short on resistance rejection - stop above resistance
                level_stop = sr_level * (1 + self.level_buffer_mult / 100)
            else:
                # Short on support breakdown - stop above support
                level_stop = sr_level * (1 + self.level_buffer_mult / 100)

            atr_stop = entry_price + (atr * self.stop_distance_mult) if atr > 0 else entry_price * 1.01

            # Use more conservative stop
            calculated_stop = max(level_stop, atr_stop)

            # Ensure minimum stop distance
            min_stop = entry_price * (1 + self.min_stop_distance_pct / 100)
            hard_sl = max(calculated_stop, min_stop)

            logger.debug(f"S/R: {context.symbol} - Short stops: Level {level_stop:.2f}, ATR {atr_stop:.2f}, Min {min_stop:.2f}, Final {hard_sl:.2f}")

        # Calculate risk per share
        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02  # 2% max risk - should be configurable
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels for S/R strategies."""

        entry_price = context.current_price
        sr_level = event.levels.get("support") or event.levels.get("resistance", entry_price)
        atr = self._get_atr(context)

        # Calculate target distance based on S/R level distance or ATR
        level_distance = abs(entry_price - sr_level)
        atr_distance = atr if atr > 0 else entry_price * 0.01

        # Use the larger of level distance or ATR for target calculation
        target_base_distance = max(level_distance, atr_distance)

        if side == "long":
            target_distance_t1 = target_base_distance * self.target_mult_t1
            target_distance_t2 = target_base_distance * self.target_mult_t2

            t1_target = entry_price + target_distance_t1
            t2_target = entry_price + target_distance_t2

            logger.debug(f"S/R: {context.symbol} - Long targets: T1 {t1_target:.2f} (+{target_distance_t1:.2f}), T2 {t2_target:.2f} (+{target_distance_t2:.2f})")

        else:  # short
            target_distance_t1 = target_base_distance * self.target_mult_t1
            target_distance_t2 = target_base_distance * self.target_mult_t2

            t1_target = entry_price - target_distance_t1
            t2_target = entry_price - target_distance_t2

            logger.debug(f"S/R: {context.symbol} - Short targets: T1 {t1_target:.2f} (-{target_distance_t1:.2f}), T2 {t2_target:.2f} (-{target_distance_t2:.2f})")

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 50, "rr": 1.0},
                {"level": t2_target, "qty_pct": 50, "rr": 2.0}
            ],
            hard_sl=0.0,  # Will be set by risk params
            trail_to=None  # Can be added later
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank the quality of S/R setup."""

        base_score = event.confidence * 100

        # Bonus factors
        bonuses = 0.0

        # Volume confirmation bonus
        if event.context.get("volume_confirmation", False):
            bonuses += 15.0
            logger.debug(f"S/R: {context.symbol} - Volume confirmation bonus: +15")

        # Level strength bonus
        level_strength = event.context.get("support_strength", 0) or event.context.get("resistance_strength", 0)
        if level_strength > 80:
            bonuses += 12.0
            logger.debug(f"S/R: {context.symbol} - Strong level bonus: +12")
        elif level_strength > 60:
            bonuses += 8.0
            logger.debug(f"S/R: {context.symbol} - Good level bonus: +8")

        # Touch count bonus
        touches = event.context.get("support_touches", 0) or event.context.get("resistance_touches", 0)
        if touches >= 4:
            bonuses += 10.0
            logger.debug(f"S/R: {context.symbol} - High touch count bonus: +10")
        elif touches >= 3:
            bonuses += 6.0
            logger.debug(f"S/R: {context.symbol} - Good touch count bonus: +6")

        # Breakout distance bonus (for breakout/breakdown setups)
        if "breakout" in event.structure_type or "breakdown" in event.structure_type:
            breakout_distance = event.context.get("breakout_distance_pct", 0) or event.context.get("breakdown_distance_pct", 0)
            if 0.3 <= breakout_distance <= 1.0:  # Sweet spot for breakouts
                bonuses += 8.0
                logger.debug(f"S/R: {context.symbol} - Breakout distance bonus: +8")

        final_score = min(100.0, base_score + bonuses)

        logger.debug(f"S/R: {context.symbol} - Setup quality: base {base_score:.1f} + bonuses {bonuses:.1f} = {final_score:.1f}")

        return final_score

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for S/R setup execution."""

        # Check market hours (basic validation)
        current_hour = context.timestamp.hour
        if current_hour < 9 or current_hour > 15:
            return False, f"Outside market hours: {current_hour}:xx"

        # Check if we have sufficient data
        if len(context.df_5m) < 20:
            return False, f"Insufficient data: {len(context.df_5m)} bars"

        # All checks passed
        logger.debug(f"S/R: {context.symbol} - Timing validation passed")
        return True, "Timing validated"

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR value with fallback."""
        if context.indicators and 'atr' in context.indicators:
            atr = context.indicators['atr']
            if atr > 0:
                return atr

        # Fallback: calculate simple ATR from recent data
        try:
            df = context.df_5m
            if len(df) >= 14:
                highs = df['high'].tail(14)
                lows = df['low'].tail(14)
                closes = df['close'].tail(15)  # Need one extra for previous close

                true_ranges = []
                for i in range(1, 15):
                    tr = max(
                        highs.iloc[i] - lows.iloc[i],
                        abs(highs.iloc[i] - closes.iloc[i-1]),
                        abs(lows.iloc[i] - closes.iloc[i-1])
                    )
                    true_ranges.append(tr)

                atr = sum(true_ranges) / len(true_ranges)
                logger.debug(f"S/R: {context.symbol} - Calculated fallback ATR: {atr:.3f}")
                return atr
        except Exception as e:
            logger.warning(f"S/R: {context.symbol} - ATR calculation failed: {e}")

        # Final fallback
        fallback_atr = context.current_price * 0.01  # 1% of price
        logger.warning(f"S/R: {context.symbol} - Using emergency ATR fallback: {fallback_atr:.3f}")
        return fallback_atr

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size based on risk management."""

        # This should be enhanced with actual portfolio size and risk management
        # For now, use basic calculation
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0  # Should be configurable

        if risk_per_share > 0:
            max_qty = int(max_risk_amount / risk_per_share)
            # Ensure minimum viable position
            qty = max(1, min(max_qty, 100))  # Min 1, max 100 shares
        else:
            qty = 1

        notional = qty * entry_price

        logger.debug(f"S/R: {context.symbol} - Position calc: risk/share {risk_per_share:.3f}, qty {qty}, notional {notional:.2f}")

        return qty, notional

    def _calculate_institutional_strength(self, context: MarketContext, sr_info: 'SupportResistanceLevels',
                                        setup_type: str, side: str, volume_confirmed: bool) -> float:
        """Calculate institutional-grade strength for support/resistance patterns."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from level quality and touches
            if "support" in setup_type:
                level_strength = sr_info.support_strength / 100.0
                touches = sr_info.support_touches
            else:
                level_strength = sr_info.resistance_strength / 100.0
                touches = sr_info.resistance_touches

            base_strength = max(1.3, vol_z * level_strength * (touches / 3.0) * 0.6)

            # Professional bonuses for institutional-grade S/R patterns
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if volume_confirmed and vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.25  # 25% bonus for volume confirmation
                logger.debug(f"S/R: Volume confirmation bonus applied (vol_z={vol_z:.2f})")

            # Level strength bonuses
            if level_strength >= 0.8:  # Very strong level (80%+)
                strength_multiplier *= 1.3  # 30% bonus for strong levels
                logger.debug(f"S/R: Strong level bonus applied ({level_strength*100:.1f}%)")
            elif level_strength >= 0.6:  # Moderate level (60%+)
                strength_multiplier *= 1.15  # 15% bonus for moderate levels

            # Touch count bonuses (more touches = stronger level)
            if touches >= 4:  # Well-tested level
                strength_multiplier *= 1.25  # 25% bonus for multiple tests
                logger.debug(f"S/R: Multiple touches bonus applied ({touches} touches)")
            elif touches >= 3:  # Tested level
                strength_multiplier *= 1.15  # 15% bonus for tested level

            # Setup-specific bonuses
            if "bounce" in setup_type:
                # Clean level interaction bonus
                distance_pct = context.current_price
                if "support" in setup_type:
                    support_level = sr_info.support_level if hasattr(sr_info, 'support_level') else context.current_price * 0.995
                    distance_pct = abs(context.current_price - support_level) / support_level * 100
                else:
                    resistance_level = sr_info.resistance_level if hasattr(sr_info, 'resistance_level') else context.current_price * 1.005
                    distance_pct = abs(context.current_price - resistance_level) / resistance_level * 100

                if distance_pct <= 0.3:  # Very close to level
                    strength_multiplier *= 1.2  # 20% bonus for clean level interaction
                    logger.debug(f"S/R: Clean level interaction bonus applied")

            elif "breakout" in setup_type:
                # Strong breakout bonus (significant move through level)
                # This would need breakout distance data from context
                strength_multiplier *= 1.1  # 10% base bonus for breakout patterns

            # Market timing bonus (S/R levels work throughout the session)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"S/R: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.5)  # Strong minimum for S/R patterns

            logger.debug(f"S/R: {context.symbol} {side} {setup_type} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"S/R: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold