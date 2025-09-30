"""
Flag Continuation Structure Implementation

This module implements flag/pennant continuation structures including:
- Flag continuation long (breakout above flag after uptrend)
- Flag continuation short (breakdown below flag after downtrend)
- Trend strength validation and consolidation pattern recognition
- Volume confirmation on breakout from consolidation

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


class FlagContinuationStructure(BaseStructure):
    """Flag/pennant continuation structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Flag Continuation structure with configuration."""
        super().__init__(config)
        self.structure_type = "flag_continuation"

        # KeyError if missing trading parameters

        # Flag pattern parameters
        self.min_consolidation_bars = config["min_consolidation_bars"]
        self.max_consolidation_bars = config["max_consolidation_bars"]
        self.min_trend_strength = config["min_trend_strength"]  # Percentage

        # Pattern recognition parameters
        self.trend_lookback_period = config["trend_lookback_period"]
        self.max_consolidation_range_pct = config["max_consolidation_range_pct"]
        self.breakout_confirmation_pct = config["breakout_confirmation_pct"]

        # Volume confirmation
        self.require_volume_confirmation = config["require_volume_confirmation"]
        self.min_volume_mult = config["min_volume_mult"]

        # Risk management
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.stop_mult = config["stop_mult"]
        self.confidence_strong_flag = config["confidence_strong_flag"]
        self.confidence_weak_flag = config["confidence_weak_flag"]

        logger.info(f"FLAG_CONTINUATION: Initialized with consolidation: {self.min_consolidation_bars}-{self.max_consolidation_bars} bars, trend strength: {self.min_trend_strength}%")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect flag continuation structures."""
        try:
            df = context.df_5m
            min_required_data = self.trend_lookback_period + self.max_consolidation_bars + 2

            if len(df) < min_required_data:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data for flag analysis (need {min_required_data} bars, have {len(df)})"
                )

            events = []

            # Try different consolidation periods within the allowed range
            for consol_period in range(self.min_consolidation_bars, min(self.max_consolidation_bars + 1, len(df) - self.trend_lookback_period)):

                flag_pattern = self._analyze_flag_pattern(df, consol_period)
                if not flag_pattern:
                    continue

                trend_direction, trend_strength_pct, consolidation_info, breakout_info = flag_pattern

                # Create flag continuation event
                if trend_direction == "up" and breakout_info["direction"] == "up":
                    structure_type = "flag_continuation_long"
                    side = "long"
                elif trend_direction == "down" and breakout_info["direction"] == "down":
                    structure_type = "flag_continuation_short"
                    side = "short"
                else:
                    continue  # No valid continuation pattern

                # Volume confirmation if required
                if self.require_volume_confirmation:
                    if not self._validate_volume_confirmation(df):
                        logger.debug(f"FLAG_CONTINUATION: {context.symbol} - Volume confirmation failed")
                        continue

                # Calculate confidence based on pattern quality
                confidence = self._calculate_institutional_strength(context, trend_strength_pct, consolidation_info, side)

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type=structure_type,
                    side=side,
                    confidence=confidence,
                    levels={
                        "flag_high": consolidation_info["high"],
                        "flag_low": consolidation_info["low"],
                        "trend_start": consolidation_info["trend_start_price"]
                    },
                    context={
                        "trend_strength_pct": trend_strength_pct,
                        "consolidation_bars": consol_period,
                        "consolidation_range_pct": consolidation_info["range_pct"],
                        "breakout_confirmation_pct": breakout_info["confirmation_pct"],
                        "pattern_quality": "flag_continuation"
                    },
                    price=context.current_price
                )

                events.append(event)
                logger.debug(f"FLAG_CONTINUATION: {context.symbol} - {structure_type} detected: trend {trend_strength_pct:.2f}%, consol {consol_period} bars")

                # Only return the first (best) pattern found
                break

            quality_score = self._calculate_quality_score(events, df) if events else 0.0

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=quality_score,
                rejection_reason=None if events else "No flag continuation patterns detected"
            )

        except Exception as e:
            logger.error(f"FLAG_CONTINUATION: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _analyze_flag_pattern(self, df: pd.DataFrame, consol_period: int) -> Optional[Tuple[str, float, Dict[str, Any], Dict[str, Any]]]:
        """Analyze potential flag pattern with given consolidation period."""
        try:
            # Split data: trend portion vs consolidation portion
            trend_data = df.iloc[-(self.trend_lookback_period + consol_period):-consol_period]
            consol_data = df.iloc[-consol_period:]
            current_price = df['close'].iloc[-1]

            if len(trend_data) < 5 or len(consol_data) < self.min_consolidation_bars:
                return None

            # 1. Analyze trend strength
            trend_start_price = trend_data['close'].iloc[0]
            trend_end_price = trend_data['close'].iloc[-1]
            trend_strength_pct = ((trend_end_price - trend_start_price) / trend_start_price) * 100

            # Check if trend meets minimum strength requirement
            if abs(trend_strength_pct) < self.min_trend_strength:
                return None

            # Determine trend direction
            trend_direction = "up" if trend_strength_pct > 0 else "down"

            # 2. Analyze consolidation pattern
            consol_high = consol_data['high'].max()
            consol_low = consol_data['low'].min()
            consol_range = consol_high - consol_low
            consol_mid = (consol_high + consol_low) / 2
            consol_range_pct = (consol_range / consol_mid) * 100

            # Check if consolidation is tight enough
            if consol_range_pct > self.max_consolidation_range_pct:
                return None

            # Calculate consolidation tightness (higher is better)
            tightness = max(0.0, (self.max_consolidation_range_pct - consol_range_pct) / self.max_consolidation_range_pct)

            consolidation_info = {
                "high": consol_high,
                "low": consol_low,
                "range_pct": consol_range_pct,
                "tightness": tightness,
                "trend_start_price": trend_start_price
            }

            # 3. Check for breakout
            breakout_direction = None
            confirmation_pct = 0.0

            if trend_direction == "up":
                # Look for breakout above consolidation high
                if current_price > consol_high:
                    breakout_direction = "up"
                    confirmation_pct = ((current_price - consol_high) / consol_high) * 100
            else:
                # Look for breakout below consolidation low
                if current_price < consol_low:
                    breakout_direction = "down"
                    confirmation_pct = ((consol_low - current_price) / consol_low) * 100

            # Check if breakout has sufficient confirmation
            if breakout_direction is None or confirmation_pct < self.breakout_confirmation_pct:
                return None

            breakout_info = {
                "direction": breakout_direction,
                "confirmation_pct": confirmation_pct
            }

            return trend_direction, trend_strength_pct, consolidation_info, breakout_info

        except Exception as e:
            logger.debug(f"FLAG_CONTINUATION: Error analyzing flag pattern: {e}")
            return None

    def _validate_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Validate volume confirmation for flag breakout."""
        if not self.require_volume_confirmation:
            return True

        try:
            # Check if volume on breakout bar is elevated
            if 'vol_z' in df.columns:
                current_vol_z = df['vol_z'].iloc[-1]
                return current_vol_z >= self.min_volume_mult
            else:
                # Fallback to simple volume ratio
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(20, min_periods=10).mean().iloc[-1]
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    return volume_ratio >= self.min_volume_mult
        except Exception as e:
            logger.debug(f"FLAG_CONTINUATION: Volume validation error: {e}")

        return True  # Default to true if can't validate

    def _calculate_institutional_strength(self, context: MarketContext, trend_strength_pct: float,
                                        consolidation_info: Dict, side: str) -> float:
        """Calculate institutional-grade strength for flag continuation patterns."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from trend strength and consolidation quality
            trend_quality = abs(trend_strength_pct) / 100.0
            consolidation_tightness = consolidation_info.get("tightness", 0.5)
            base_strength = max(1.4, vol_z * trend_quality * (1.0 + consolidation_tightness) * 3.0)

            # Professional bonuses for institutional-grade flag patterns
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.2  # 20% bonus for volume surge
                logger.debug(f"FLAG: Volume surge bonus applied (vol_z={vol_z:.2f})")

            # Strong trend bonuses
            if abs(trend_strength_pct) >= 4.0:  # Strong trend (4%+)
                strength_multiplier *= 1.25  # 25% bonus for strong trend
                logger.debug(f"FLAG: Strong trend bonus applied ({trend_strength_pct:.1f}%)")
            elif abs(trend_strength_pct) >= 2.0:  # Moderate trend
                strength_multiplier *= 1.15  # 15% bonus for moderate trend

            # Consolidation quality bonuses
            if consolidation_tightness >= 0.8:  # Very tight consolidation
                strength_multiplier *= 1.3  # 30% bonus for tight flag
                logger.debug(f"FLAG: Tight consolidation bonus applied ({consolidation_tightness:.2f})")
            elif consolidation_tightness >= 0.6:  # Good consolidation
                strength_multiplier *= 1.2  # 20% bonus for good consolidation

            # Flag duration bonus
            flag_duration = consolidation_info.get("duration", 10)
            if 5 <= flag_duration <= 15:  # Optimal flag duration
                strength_multiplier *= 1.15  # 15% bonus for optimal duration
                logger.debug(f"FLAG: Optimal duration bonus applied ({flag_duration} bars)")

            # Market timing bonus (flags work well during trend sessions)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"FLAG: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.7)  # Strong minimum for flag patterns

            logger.debug(f"FLAG: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"FLAG: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold

    def _calculate_quality_score(self, events: List[StructureEvent], df: pd.DataFrame) -> float:
        """Calculate quality score for flag continuation events."""
        if not events:
            return 0.0

        base_score = 70.0  # Higher base for continuation patterns

        # Add points for trend strength
        event = events[0]
        trend_strength = abs(event.context.get("trend_strength_pct", 0))
        trend_score = min(20.0, trend_strength * 2)  # Up to 20 points for 10%+ trends

        # Add points for tight consolidation
        range_pct = event.context.get("consolidation_range_pct", 2.0)
        tightness_score = max(0.0, 10.0 - range_pct * 5)  # Up to 10 points for tight ranges

        # Volume bonus if available
        volume_bonus = 0.0
        try:
            if 'vol_z' in df.columns:
                vol_z = df['vol_z'].iloc[-1]
                volume_bonus = min(5.0, vol_z)
        except:
            pass

        return min(100.0, base_score + trend_score + tightness_score + volume_bonus)

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for flag continuation setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for flag continuation setups."""
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
        """Calculate risk parameters for flag continuation trades."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        # For flag continuations, stop loss is typically at the opposite side of the flag
        if side == "long":
            # Stop below flag low
            flag_low = event.levels.get("flag_low", entry_price)
            hard_sl = flag_low - (atr * self.stop_mult)
        else:
            # Stop above flag high
            flag_high = event.levels.get("flag_high", entry_price)
            hard_sl = flag_high + (atr * self.stop_mult)

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels for flag continuation trades."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        # Flag continuations often target measured moves
        flag_high = event.levels.get("flag_high", entry_price)
        flag_low = event.levels.get("flag_low", entry_price)
        flag_height = flag_high - flag_low

        if side == "long":
            # Target flag height above breakout level
            t1_target = entry_price + (flag_height * self.target_mult_t1)
            t2_target = entry_price + (flag_height * self.target_mult_t2)
        else:
            # Target flag height below breakdown level
            t1_target = entry_price - (flag_height * self.target_mult_t1)
            t2_target = entry_price - (flag_height * self.target_mult_t2)

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 50, "rr": self.target_mult_t1},
                {"level": t2_target, "qty_pct": 50, "rr": self.target_mult_t2}
            ],
            hard_sl=0.0,  # Set in risk_params
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank flag continuation setup quality."""
        base_score = event.confidence * 100
        trend_strength = abs(event.context.get("trend_strength_pct", 0))
        consolidation_bars = event.context.get("consolidation_bars", 0)

        # Bonus for strong trend
        trend_bonus = min(15.0, trend_strength)

        # Bonus for optimal consolidation period
        if 5 <= consolidation_bars <= 8:  # Sweet spot
            consol_bonus = 10.0
        else:
            consol_bonus = max(0.0, 10.0 - abs(consolidation_bars - 6))

        return min(100.0, base_score + trend_bonus + consol_bonus)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for flag continuation trades."""
        # Flag continuations work well throughout the session
        return True, "Flag continuation timing validated"

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR with fallback calculation."""
        try:
            if context.indicators and 'atr' in context.indicators:
                return context.indicators['atr']

            # Fallback ATR calculation
            df = context.df_5m
            atr = df['close'].pct_change().abs().rolling(14, min_periods=5).mean().iloc[-1]
            return max(0.005, float(atr))  # Minimum 0.5%
        except:
            return context.current_price * 0.01  # 1% fallback

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size based on risk management."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0  # Maximum risk per trade
        qty = max(1, min(int(max_risk_amount / risk_per_share), 100)) if risk_per_share > 0 else 1
        return qty, qty * entry_price