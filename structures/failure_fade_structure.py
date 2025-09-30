"""
Failure Fade Structure Implementation

This module implements failure fade (mean reversion) structures including:
- Failed breakout above resistance that fades back (short opportunity)
- Failed breakdown below support that recovers (long opportunity)
- Enhanced failure detection with volume and timing validation

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


class FailureFadeStructure(BaseStructure):
    """Failure fade (mean reversion) structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Failure Fade structure with configuration."""
        super().__init__(config)
        self.structure_type = "failure_fade"

        # KeyError if missing trading parameters
        self.min_pierce_size_pct = config["min_pierce_size_pct"]
        self.max_pierce_size_pct = config["max_pierce_size_pct"]
        self.fade_confirmation_bars = config["fade_confirmation_bars"]
        self.require_volume_confirmation = config["require_volume_confirmation"]
        self.min_volume_mult = config["min_volume_mult"]
        self.min_fade_strength = config["min_fade_strength"]
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.stop_mult = config["stop_mult"]
        # Removed deprecated confidence parameters - now using institutional strength calculations

        logger.debug(f"FAILURE_FADE: Initialized with pierce range: {self.min_pierce_size_pct}-{self.max_pierce_size_pct}%, confirmation: {self.fade_confirmation_bars} bars")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect failure fade structures."""
        try:
            df = context.df_5m
            if len(df) < max(2, self.fade_confirmation_bars + 1):
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient data for failure fade analysis"
                )

            events = []
            levels = self._get_available_levels(context)

            if not levels:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="No key levels available for failure fade analysis"
                )

            # Analyze each level for failure fade patterns
            for level_name, level_value in levels.items():
                if level_value is None or not np.isfinite(level_value):
                    continue

                # Check for failure fade at resistance levels (PDH, ORH) - short opportunity
                if level_name in ("PDH", "ORH"):
                    fade_event = self._detect_resistance_failure_fade(context, df, level_name, level_value)
                    if fade_event:
                        events.append(fade_event)

                # Check for failure fade at support levels (PDL, ORL) - long opportunity
                elif level_name in ("PDL", "ORL"):
                    fade_event = self._detect_support_failure_fade(context, df, level_name, level_value)
                    if fade_event:
                        events.append(fade_event)

            quality_score = self._calculate_quality_score(events, df) if events else 0.0

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=quality_score,
                rejection_reason=None if events else "No failure fade patterns detected"
            )

        except Exception as e:
            logger.error(f"FAILURE_FADE: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _detect_resistance_failure_fade(self, context: MarketContext, df: pd.DataFrame,
                                       level_name: str, level_value: float) -> Optional[StructureEvent]:
        """Detect failure fade at resistance level (short opportunity)."""
        current_price = context.current_price

        # Check if we have a failure pattern:
        # 1. Previous bar(s) pierced above resistance
        # 2. Current bar closed back below resistance

        fade_pattern = self._check_resistance_fade_pattern(df, level_value)
        if not fade_pattern:
            logger.debug(f"FAILURE_FADE: {context.symbol} - No resistance fade pattern at {level_name}")
            return None

        pierce_size_pct, fade_strength = fade_pattern

        # Validate pierce size is within acceptable range
        if not (self.min_pierce_size_pct <= pierce_size_pct <= self.max_pierce_size_pct):
            logger.debug(f"FAILURE_FADE: {context.symbol} - Pierce size {pierce_size_pct:.2f}% outside acceptable range at {level_name}")
            return None

        # Validate fade strength
        if fade_strength < self.min_fade_strength:
            logger.debug(f"FAILURE_FADE: {context.symbol} - Fade strength {fade_strength:.2f} too weak at {level_name}")
            return None

        # Volume confirmation if required
        if self.require_volume_confirmation:
            if not self._validate_volume_confirmation(df):
                logger.debug(f"FAILURE_FADE: {context.symbol} - Volume confirmation failed at {level_name}")
                return None

        # Calculate institutional strength based on fade quality and volume
        confidence = self._calculate_institutional_strength(context, pierce_size_pct, fade_strength, level_name, "short")

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type="failure_fade_short",
            side="short",
            confidence=confidence,
            levels={level_name: level_value},
            context={
                "level_name": level_name,
                "pierce_size_pct": pierce_size_pct,
                "fade_strength": fade_strength,
                "failure_type": "resistance_rejection"
            },
            price=current_price
        )

        logger.info(f"FAILURE_FADE: {context.symbol} - Resistance failure fade {level_name} at {current_price:.2f}, pierce: {pierce_size_pct:.2f}%, fade: {fade_strength:.2f}")
        return event

    def _detect_support_failure_fade(self, context: MarketContext, df: pd.DataFrame,
                                    level_name: str, level_value: float) -> Optional[StructureEvent]:
        """Detect failure fade at support level (long opportunity)."""
        current_price = context.current_price

        # Check if we have a failure pattern:
        # 1. Previous bar(s) pierced below support
        # 2. Current bar closed back above support

        fade_pattern = self._check_support_fade_pattern(df, level_value)
        if not fade_pattern:
            logger.debug(f"FAILURE_FADE: {context.symbol} - No support fade pattern at {level_name}")
            return None

        pierce_size_pct, fade_strength = fade_pattern

        # Validate pierce size is within acceptable range
        if not (self.min_pierce_size_pct <= pierce_size_pct <= self.max_pierce_size_pct):
            logger.debug(f"FAILURE_FADE: {context.symbol} - Pierce size {pierce_size_pct:.2f}% outside acceptable range at {level_name}")
            return None

        # Validate fade strength
        if fade_strength < self.min_fade_strength:
            logger.debug(f"FAILURE_FADE: {context.symbol} - Fade strength {fade_strength:.2f} too weak at {level_name}")
            return None

        # Volume confirmation if required
        if self.require_volume_confirmation:
            if not self._validate_volume_confirmation(df):
                logger.debug(f"FAILURE_FADE: {context.symbol} - Volume confirmation failed at {level_name}")
                return None

        # Calculate institutional strength based on fade quality and volume
        confidence = self._calculate_institutional_strength(context, pierce_size_pct, fade_strength, level_name, "long")

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type="failure_fade_long",
            side="long",
            confidence=confidence,
            levels={level_name: level_value},
            context={
                "level_name": level_name,
                "pierce_size_pct": pierce_size_pct,
                "fade_strength": fade_strength,
                "failure_type": "support_bounce"
            },
            price=current_price
        )

        logger.info(f"FAILURE_FADE: {context.symbol} - Support failure fade {level_name} at {current_price:.2f}, pierce: {pierce_size_pct:.2f}%, fade: {fade_strength:.2f}")
        return event

    def _check_resistance_fade_pattern(self, df: pd.DataFrame, resistance: float) -> Optional[Tuple[float, float]]:
        """Check for resistance failure fade pattern."""
        try:
            if len(df) < 2:
                return None

            current_bar = df.iloc[-1]
            previous_bar = df.iloc[-2]

            # Check if previous bar pierced above resistance
            if previous_bar['high'] <= resistance:
                return None

            # Check if current bar closed back below resistance
            if current_bar['close'] >= resistance:
                return None

            # Calculate pierce size as percentage
            pierce_size = previous_bar['high'] - resistance
            pierce_size_pct = (pierce_size / resistance) * 100

            # Calculate fade strength (how far back it faded)
            fade_size = previous_bar['high'] - current_bar['close']
            max_possible_fade = previous_bar['high'] - resistance
            fade_strength = fade_size / max_possible_fade if max_possible_fade > 0 else 0

            # Additional validation for confirmation bars
            if self.fade_confirmation_bars > 1 and len(df) >= self.fade_confirmation_bars + 1:
                # Check that recent bars confirm the fade
                recent_closes = df['close'].tail(self.fade_confirmation_bars)
                confirmation = (recent_closes < resistance).all()
                if not confirmation:
                    return None

            return pierce_size_pct, fade_strength

        except Exception as e:
            logger.debug(f"FAILURE_FADE: Error checking resistance fade pattern: {e}")
            return None

    def _check_support_fade_pattern(self, df: pd.DataFrame, support: float) -> Optional[Tuple[float, float]]:
        """Check for support failure fade pattern."""
        try:
            if len(df) < 2:
                return None

            current_bar = df.iloc[-1]
            previous_bar = df.iloc[-2]

            # Check if previous bar pierced below support
            if previous_bar['low'] >= support:
                return None

            # Check if current bar closed back above support
            if current_bar['close'] <= support:
                return None

            # Calculate pierce size as percentage
            pierce_size = support - previous_bar['low']
            pierce_size_pct = (pierce_size / support) * 100

            # Calculate fade strength (how far back it faded)
            fade_size = current_bar['close'] - previous_bar['low']
            max_possible_fade = support - previous_bar['low']
            fade_strength = fade_size / max_possible_fade if max_possible_fade > 0 else 0

            # Additional validation for confirmation bars
            if self.fade_confirmation_bars > 1 and len(df) >= self.fade_confirmation_bars + 1:
                # Check that recent bars confirm the fade
                recent_closes = df['close'].tail(self.fade_confirmation_bars)
                confirmation = (recent_closes > support).all()
                if not confirmation:
                    return None

            return pierce_size_pct, fade_strength

        except Exception as e:
            logger.debug(f"FAILURE_FADE: Error checking support fade pattern: {e}")
            return None

    def _get_available_levels(self, context: MarketContext) -> Dict[str, float]:
        """Get available key levels for failure fade analysis."""
        levels = {}

        if context.pdh is not None:
            levels["PDH"] = context.pdh
        if context.pdl is not None:
            levels["PDL"] = context.pdl
        if context.orh is not None:
            levels["ORH"] = context.orh
        if context.orl is not None:
            levels["ORL"] = context.orl

        return levels

    def _validate_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Validate volume confirmation for failure fade."""
        if not self.require_volume_confirmation:
            return True

        try:
            # Check if volume on fade bar is elevated
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
        except:
            pass

        return True  # Default to true if can't validate

    def _calculate_quality_score(self, events: List[StructureEvent], df: pd.DataFrame) -> float:
        """Calculate quality score for failure fade events."""
        if not events:
            return 0.0

        base_score = 65.0  # Higher base for mean reversion setups

        # Add points for fade strength
        total_fade_strength = sum(event.context.get("fade_strength", 0) for event in events)
        fade_score = min(20.0, total_fade_strength * 25)

        # Add points for number of events
        event_score = len(events) * 10

        # Volume bonus if available
        volume_bonus = 0.0
        try:
            if 'vol_z' in df.columns:
                vol_z = df['vol_z'].iloc[-1]
                volume_bonus = min(5.0, vol_z)
        except:
            pass

        return min(100.0, base_score + fade_score + event_score + volume_bonus)

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for failure fade setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for failure fade setups."""
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
        """Calculate risk parameters for failure fade trades."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        # Get the level that failed
        level_name = event.context.get("level_name", "")
        failed_level = None
        for level_key, level_value in event.levels.items():
            if level_key in ["PDH", "PDL", "ORH", "ORL"]:
                failed_level = level_value
                break

        if failed_level is None:
            failed_level = entry_price

        # For failure fades, stop loss is beyond the failed level
        if side == "long":
            # For long fades at support, stop below the low that failed
            hard_sl = failed_level - (atr * self.stop_mult)
        else:
            # For short fades at resistance, stop above the high that failed
            hard_sl = failed_level + (atr * self.stop_mult)

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels for failure fade trades."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        if side == "long":
            t1_target = entry_price + (atr * self.target_mult_t1)
            t2_target = entry_price + (atr * self.target_mult_t2)
        else:
            t1_target = entry_price - (atr * self.target_mult_t1)
            t2_target = entry_price - (atr * self.target_mult_t2)

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 60, "rr": self.target_mult_t1},  # Take more profit early on mean reversion
                {"level": t2_target, "qty_pct": 40, "rr": self.target_mult_t2}
            ],
            hard_sl=0.0,  # Set in risk_params
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank failure fade setup quality."""
        base_score = event.confidence * 100
        fade_strength = event.context.get("fade_strength", 0)
        pierce_size = event.context.get("pierce_size_pct", 0)

        # Bonus for strong fade
        fade_bonus = fade_strength * 15

        # Optimal pierce size bonus (not too small, not too large)
        if 0.2 <= pierce_size <= 0.8:
            pierce_bonus = 10.0
        else:
            pierce_bonus = max(0.0, 10.0 - abs(pierce_size - 0.5) * 10)

        return min(100.0, base_score + fade_bonus + pierce_bonus)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for failure fade trades."""
        # Failure fades work well throughout the session, especially when levels are respected
        return True, "Failure fade timing validated"

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

    def _calculate_institutional_strength(self, context: MarketContext, pierce_size_pct: float,
                                        fade_strength: float, level_name: str, side: str) -> float:
        """Calculate institutional-grade strength for failure fade setups."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from fade quality and volume (institutional volume threshold ≥1.5)
            base_strength = max(1.0, vol_z * fade_strength * 2.0)  # Scale fade strength by vol_z

            # Professional bonuses for institutional-grade failure fades
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.25  # 25% bonus for volume surge
                logger.debug(f"FAILURE_FADE: Volume surge bonus applied (vol_z={vol_z:.2f})")

            if vol_z >= 2.0:  # Strong institutional volume
                strength_multiplier *= 1.3  # Additional 30% bonus
                logger.debug(f"FAILURE_FADE: Strong volume surge bonus applied")

            # Fade quality bonuses (mean reversion strength)
            if fade_strength >= 0.7:  # Strong fade back (70%+ reversion)
                strength_multiplier *= 1.2  # 20% bonus for strong fade
                logger.debug(f"FAILURE_FADE: Strong fade bonus applied (fade={fade_strength:.2f})")
            elif fade_strength >= 0.5:  # Moderate fade back
                strength_multiplier *= 1.1  # 10% bonus for moderate fade

            # Pierce size optimization (institutional preference for clean failures)
            optimal_pierce = 0.3 <= pierce_size_pct <= 0.8  # Optimal failure range
            if optimal_pierce:
                strength_multiplier *= 1.15  # 15% bonus for optimal pierce size
                logger.debug(f"FAILURE_FADE: Optimal pierce size bonus applied ({pierce_size_pct:.2f}%)")

            # Key level significance (PDH/PDL more significant than OR levels)
            if level_name in ["PDH", "PDL"]:  # Previous day levels
                strength_multiplier *= 1.25  # 25% bonus for key daily levels
                logger.debug(f"FAILURE_FADE: Key level bonus applied for {level_name}")
            elif level_name in ["ORH", "ORL"]:  # Opening range levels
                strength_multiplier *= 1.1  # 10% bonus for OR levels

            # Market timing bonus (failure fades work well in trending markets)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"FAILURE_FADE: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.5)  # Minimum viable strength

            logger.debug(f"FAILURE_FADE: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"FAILURE_FADE: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size based on risk management."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0  # Maximum risk per trade
        qty = max(1, min(int(max_risk_amount / risk_per_share), 100)) if risk_per_share > 0 else 1
        return qty, qty * entry_price