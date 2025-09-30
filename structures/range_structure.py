"""
Range Structure Implementation

This module implements range-bound trading structures including:
- Range bounce (price bounces off range boundaries)
- Range breakout (price breaks out of established range)
- Range compression (range tightening before breakout)

All trading parameters must be explicitly configured.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    MarketContext, StructureEvent, TradePlan, RiskParams, ExitLevels, StructureAnalysis
)

logger = get_agent_logger()


class RangeStructure(BaseStructure):
    """Range-based structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Range structure with configuration."""
        super().__init__(config)
        self.structure_type = "range"

        # Range parameters
        self.min_range_duration = config["min_range_duration"]
        self.max_range_height_pct = config["max_range_height_pct"]
        self.min_range_height_pct = config["min_range_height_pct"]
        self.bounce_tolerance_pct = config["bounce_tolerance_pct"]
        self.breakout_confirmation_pct = config["breakout_confirmation_pct"]
        self.require_volume_confirmation = config["require_volume_confirmation"]
        self.min_volume_mult = config["min_volume_mult"]

        # Risk management
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.confidence_level = config["confidence_level"]

        logger.info(f"RANGE: Initialized with range duration: {self.min_range_duration} bars, height: {self.min_range_height_pct}-{self.max_range_height_pct}%")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect range-based structures."""
        try:
            df = context.df_5m
            if len(df) < self.min_range_duration + 5:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient data for range analysis"
                )

            events = []
            range_info = self._detect_range(df)

            if range_info:
                current_price = context.current_price

                # Range bounce strategies
                bounce_events = self._detect_range_bounce(context, range_info)
                events.extend(bounce_events)

                # Range breakout strategies
                breakout_events = self._detect_range_breakout(context, range_info)
                events.extend(breakout_events)

                if events:
                    logger.info(f"RANGE: {context.symbol} - Range detected: {range_info['support']:.2f}-{range_info['resistance']:.2f}, {len(events)} events")

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=self._calculate_quality_score(range_info) if range_info else 0.0,
                rejection_reason=None if events else "No valid range patterns detected"
            )

        except Exception as e:
            logger.error(f"RANGE: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _detect_range(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect if price is in a defined range."""
        # Look at recent price action to identify range
        lookback_bars = min(len(df), self.min_range_duration * 2)
        recent_data = df.tail(lookback_bars)

        # Calculate potential range boundaries
        highs = recent_data['high']
        lows = recent_data['low']

        # Find potential resistance (cluster of highs)
        resistance_candidates = highs.rolling(window=5).max()
        resistance = resistance_candidates.quantile(0.95)

        # Find potential support (cluster of lows)
        support_candidates = lows.rolling(window=5).min()
        support = support_candidates.quantile(0.05)

        # Validate range
        range_height_pct = (resistance - support) / support * 100

        if not (self.min_range_height_pct <= range_height_pct <= self.max_range_height_pct):
            return None

        # Check if price has been respecting these levels
        touches_resistance = 0
        touches_support = 0

        for _, row in recent_data.iterrows():
            # Count resistance touches
            if abs(row['high'] - resistance) / resistance * 100 <= self.bounce_tolerance_pct:
                touches_resistance += 1

            # Count support touches
            if abs(row['low'] - support) / support * 100 <= self.bounce_tolerance_pct:
                touches_support += 1

        # Need at least 2 touches on each level to confirm range
        if touches_resistance >= 2 and touches_support >= 2:
            return {
                "support": support,
                "resistance": resistance,
                "range_height_pct": range_height_pct,
                "touches_support": touches_support,
                "touches_resistance": touches_resistance,
                "duration_bars": lookback_bars
            }

        return None

    def _detect_range_bounce(self, context: MarketContext, range_info: Dict[str, Any]) -> List[StructureEvent]:
        """Detect range bounce opportunities."""
        events = []
        current_price = context.current_price
        support = range_info["support"]
        resistance = range_info["resistance"]

        # Support bounce (long)
        support_distance_pct = abs(current_price - support) / support * 100
        if support_distance_pct <= self.bounce_tolerance_pct:
            # Check if we're coming from above (bounce setup)
            if current_price >= support:
                if self._validate_volume_confirmation(context):
                    event = StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type="range_bounce_long",
                        side="long",
                        confidence=self._calculate_institutional_strength(context, range_info, "bounce", "long"),
                        levels={"support": support, "resistance": resistance},
                        context={
                            "range_height_pct": range_info["range_height_pct"],
                            "distance_from_support_pct": support_distance_pct,
                            "touches_support": range_info["touches_support"]
                        },
                        price=current_price
                    )
                    events.append(event)
                    logger.debug(f"RANGE: {context.symbol} - Support bounce long at {current_price:.2f}")

        # Resistance bounce (short)
        resistance_distance_pct = abs(current_price - resistance) / resistance * 100
        if resistance_distance_pct <= self.bounce_tolerance_pct:
            # Check if we're coming from below (bounce setup)
            if current_price <= resistance:
                if self._validate_volume_confirmation(context):
                    event = StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type="range_bounce_short",
                        side="short",
                        confidence=self._calculate_institutional_strength(context, range_info, "bounce", "short"),
                        levels={"support": support, "resistance": resistance},
                        context={
                            "range_height_pct": range_info["range_height_pct"],
                            "distance_from_resistance_pct": resistance_distance_pct,
                            "touches_resistance": range_info["touches_resistance"]
                        },
                        price=current_price
                    )
                    events.append(event)
                    logger.debug(f"RANGE: {context.symbol} - Resistance bounce short at {current_price:.2f}")

        return events

    def _detect_range_breakout(self, context: MarketContext, range_info: Dict[str, Any]) -> List[StructureEvent]:
        """Detect range breakout opportunities."""
        events = []
        current_price = context.current_price
        support = range_info["support"]
        resistance = range_info["resistance"]

        # Resistance breakout (long)
        if current_price > resistance:
            breakout_distance_pct = (current_price - resistance) / resistance * 100
            if breakout_distance_pct >= self.breakout_confirmation_pct:
                if self._validate_volume_confirmation(context):
                    event = StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type="range_breakout_long",
                        side="long",
                        confidence=self._calculate_institutional_strength(context, range_info, "breakout", "long"),
                        levels={"support": support, "resistance": resistance, "breakout_level": resistance},
                        context={
                            "range_height_pct": range_info["range_height_pct"],
                            "breakout_distance_pct": breakout_distance_pct,
                            "range_duration": range_info["duration_bars"]
                        },
                        price=current_price
                    )
                    events.append(event)
                    logger.debug(f"RANGE: {context.symbol} - Range breakout long at {current_price:.2f}")

        # Support breakdown (short)
        if current_price < support:
            breakdown_distance_pct = (support - current_price) / support * 100
            if breakdown_distance_pct >= self.breakout_confirmation_pct:
                if self._validate_volume_confirmation(context):
                    event = StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type="range_breakdown_short",
                        side="short",
                        confidence=self._calculate_institutional_strength(context, range_info, "breakout", "short"),
                        levels={"support": support, "resistance": resistance, "breakdown_level": support},
                        context={
                            "range_height_pct": range_info["range_height_pct"],
                            "breakdown_distance_pct": breakdown_distance_pct,
                            "range_duration": range_info["duration_bars"]
                        },
                        price=current_price
                    )
                    events.append(event)
                    logger.debug(f"RANGE: {context.symbol} - Range breakdown short at {current_price:.2f}")

        return events

    def _validate_volume_confirmation(self, context: MarketContext) -> bool:
        """Validate volume confirmation if required."""
        if not self.require_volume_confirmation:
            return True

        if context.indicators and 'vol_z' in context.indicators:
            vol_z = context.indicators['vol_z']
            return vol_z >= self.min_volume_mult

        return True  # Default to true if no volume data

    def _calculate_quality_score(self, range_info: Dict[str, Any]) -> float:
        """Calculate quality score for range setup."""
        if not range_info:
            return 0.0

        base_score = 60.0

        # Add points for more touches
        touch_score = min(20.0, (range_info["touches_support"] + range_info["touches_resistance"]) * 3)

        # Add points for good range height
        height_pct = range_info["range_height_pct"]
        if 1.0 <= height_pct <= 2.0:  # Ideal range height
            height_score = 15.0
        else:
            height_score = max(0.0, 15.0 - abs(height_pct - 1.5) * 5)

        # Add points for duration
        duration_score = min(5.0, range_info["duration_bars"] / 10)

        return min(100.0, base_score + touch_score + height_score + duration_score)

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for range setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for range setups."""
        return self._plan_strategy(context, event, "short")

    def _plan_strategy(self, context: MarketContext, event: StructureEvent, side: str) -> TradePlan:
        """Common strategy planning logic."""
        entry_price = context.current_price
        risk_params = self.calculate_risk_params(context, event, side)
        exit_levels = self.get_exit_levels(context, event, side)
        qty, notional = self._calculate_position_size(entry_price, risk_params.hard_sl, context)

        return TradePlan(
            symbol=context.symbol,
            side=side,
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=event.confidence,
            notes=event.context
        )

    def calculate_risk_params(self, context: MarketContext, event: StructureEvent, side: str) -> RiskParams:
        """Calculate risk parameters."""
        entry_price = context.current_price

        if "bounce" in event.structure_type:
            # For bounce trades, risk to opposite boundary
            if side == "long":
                hard_sl = event.levels.get("support", entry_price) * 0.995  # Slightly below support
            else:
                hard_sl = event.levels.get("resistance", entry_price) * 1.005  # Slightly above resistance
        else:
            # For breakout trades, risk back into range
            if side == "long":
                hard_sl = event.levels.get("breakout_level", entry_price) * 0.99  # Back below breakout
            else:
                hard_sl = event.levels.get("breakdown_level", entry_price) * 1.01  # Back above breakdown

        risk_per_share = abs(entry_price - hard_sl)
        range_height = abs(event.levels.get("resistance", entry_price) - event.levels.get("support", entry_price))

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=range_height,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels."""
        entry_price = context.current_price
        range_height = abs(event.levels.get("resistance", entry_price) - event.levels.get("support", entry_price))

        if "bounce" in event.structure_type:
            # For bounce trades, target opposite boundary
            if side == "long":
                t1_target = event.levels.get("resistance", entry_price) * 0.995  # Just below resistance
                t2_target = entry_price + range_height * self.target_mult_t2
            else:
                t1_target = event.levels.get("support", entry_price) * 1.005  # Just above support
                t2_target = entry_price - range_height * self.target_mult_t2
        else:
            # For breakout trades, target measured move
            if side == "long":
                t1_target = entry_price + range_height * self.target_mult_t1
                t2_target = entry_price + range_height * self.target_mult_t2
            else:
                t1_target = entry_price - range_height * self.target_mult_t1
                t2_target = entry_price - range_height * self.target_mult_t2

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 50, "rr": 1.0},
                {"level": t2_target, "qty_pct": 50, "rr": 2.0}
            ],
            hard_sl=0.0,
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank range setup quality."""
        base_score = event.confidence * 100
        range_height = event.context.get("range_height_pct", 0)

        # Add points for ideal range height
        if 1.0 <= range_height <= 2.0:
            height_bonus = 10.0
        else:
            height_bonus = max(0.0, 10.0 - abs(range_height - 1.5) * 3)

        return min(100.0, base_score + height_bonus)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing."""
        return True, "Range timing validated"

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0
        qty = max(1, min(int(max_risk_amount / risk_per_share), 100)) if risk_per_share > 0 else 1
        return qty, qty * entry_price

    def _calculate_institutional_strength(self, context: MarketContext, range_info: Dict,
                                        setup_type: str, side: str) -> float:
        """Calculate institutional-grade strength for range patterns."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from range quality and duration
            range_height_pct = range_info.get("range_height_pct", 1.0)
            range_duration = range_info.get("duration", 20)
            range_quality = min(3.0, (range_duration / 10) * (2.0 / range_height_pct))  # Quality from duration/tightness
            base_strength = max(1.4, vol_z * range_quality * 0.3)

            # Professional bonuses for institutional-grade range patterns
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.2  # 20% bonus for volume surge
                logger.debug(f"RANGE: Volume surge bonus applied (vol_z={vol_z:.2f})")

            # Range quality bonuses
            if 1.0 <= range_height_pct <= 2.5:  # Optimal range height (not too tight, not too wide)
                strength_multiplier *= 1.2  # 20% bonus for optimal range
                logger.debug(f"RANGE: Optimal range height bonus applied ({range_height_pct:.2f}%)")

            if range_duration >= 30:  # Well-established range
                strength_multiplier *= 1.15  # 15% bonus for established range
                logger.debug(f"RANGE: Established range bonus applied ({range_duration} bars)")

            # Setup-specific bonuses
            if setup_type == "bounce":
                # Clean bounce bonus (close to boundary)
                if side == "long":
                    distance_pct = range_info.get("distance_from_support_pct", 1.0)
                    if distance_pct <= 0.5:  # Very close to support
                        strength_multiplier *= 1.25  # 25% bonus for clean support bounce
                        logger.debug(f"RANGE: Clean support bounce bonus applied")
                else:
                    distance_pct = range_info.get("distance_from_resistance_pct", 1.0)
                    if distance_pct <= 0.5:  # Very close to resistance
                        strength_multiplier *= 1.25  # 25% bonus for clean resistance bounce
                        logger.debug(f"RANGE: Clean resistance bounce bonus applied")

                # Multiple touches bonus (level strength)
                touches = range_info.get(f"touches_{'support' if side == 'long' else 'resistance'}", 1)
                if touches >= 3:  # Multiple level tests
                    strength_multiplier *= 1.15  # 15% bonus for tested levels
                    logger.debug(f"RANGE: Multiple touches bonus applied ({touches} touches)")

            elif setup_type == "breakout":
                # Clean breakout bonus (significant break distance)
                breakout_distance_pct = range_info.get("breakout_distance_pct", 0.3)
                if breakout_distance_pct >= 0.5:  # Strong breakout
                    strength_multiplier *= 1.3  # 30% bonus for strong breakout
                    logger.debug(f"RANGE: Strong breakout bonus applied ({breakout_distance_pct:.2f}%)")

                # Range maturity bonus (older ranges have stronger breakouts)
                if range_duration >= 50:  # Very mature range
                    strength_multiplier *= 1.2  # 20% bonus for mature range breakout
                    logger.debug(f"RANGE: Mature range breakout bonus applied")

            # Market timing bonus (ranges work well during consolidation periods)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 11 <= current_hour <= 13:  # Mid-session consolidation
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"RANGE: Mid-session consolidation bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.6)  # Strong minimum for range patterns

            logger.debug(f"RANGE: {context.symbol} {side} {setup_type} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"RANGE: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold