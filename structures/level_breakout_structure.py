"""
Level Breakout Structure Implementation

This module implements level-based breakout structures including:
- Breakouts above resistance levels (PDH, ORH) with volume and hold confirmation
- Breakdowns below support levels (PDL, ORL) with volume and hold confirmation
- Smart Money Concepts (SMC) filtering to avoid false breakouts
- Session timing adjustments for breakout quality

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


class LevelBreakoutStructure(BaseStructure):
    """Level-based breakout structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Level Breakout structure with configuration."""
        super().__init__(config)
        self.structure_type = "level_breakout"

        # KeyError if missing trading parameters

        # Breakout parameters
        # Required parameters (already validated)
        self.k_atr = config["k_atr"]
        self.hold_bars = config["hold_bars"]
        self.vol_z_required = config["vol_z_required"]

        # KeyError if missing optional trading parameters
        self.min_breakout_atr_mult = config["min_breakout_atr_mult"]
        self.require_breakout_bar_volume_surge = config["require_breakout_bar_volume_surge"]
        self.breakout_volume_min_ratio = config["breakout_volume_min_ratio"]
        self.enable_smc_filtering = config["enable_smc_filtering"]
        self.liquidity_grab_tolerance_pct = config["liquidity_grab_tolerance_pct"]
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.confidence_strong_breakout = config["confidence_strong_breakout"]
        self.confidence_weak_breakout = config["confidence_weak_breakout"]

        logger.debug(f"LEVEL_BREAKOUT: Initialized with k_atr: {self.k_atr}, hold_bars: {self.hold_bars}, vol_z_required: {self.vol_z_required}")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect level breakout structures."""
        logger.debug(f"LEVEL_BREAKOUT_DETECTOR: Starting detection for {context.symbol}")
        try:
            df = context.df_5m
            if len(df) < max(5, self.hold_bars + 1):
                logger.debug(f"LEVEL_BREAKOUT_DETECTOR: {context.symbol} insufficient data (len={len(df)})")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient data for level breakout analysis"
                )

            # Calculate volume Z-score if not present
            if 'vol_z' not in df.columns:
                df = df.copy()
                df['vol_z'] = self._calculate_vol_z(df)

            events = []
            levels = self._get_available_levels(context)

            if not levels:
                logger.debug(f"LEVEL_BREAKOUT_DETECTOR: {context.symbol} no levels available")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="No key levels available for breakout analysis"
                )

            atr = self._calculate_atr(df)
            current_price = context.current_price
            vol_z_current = float(df['vol_z'].iloc[-1])

            # Apply time-based volume adjustment
            vol_z_required = self._get_time_adjusted_vol_threshold(context.timestamp)

            for level_name, level_value in levels.items():
                if level_value is None or not np.isfinite(level_value):
                    continue

                # Long breakouts above resistance levels (PDH, ORH)
                if level_name in ("PDH", "ORH") and current_price > level_value:
                    breakout_event = self._detect_long_breakout(
                        context, df, level_name, level_value, atr, vol_z_current, vol_z_required
                    )
                    if breakout_event:
                        events.append(breakout_event)

                # Short breakdowns below support levels (PDL, ORL)
                elif level_name in ("PDL", "ORL") and current_price < level_value:
                    breakdown_event = self._detect_short_breakdown(
                        context, df, level_name, level_value, atr, vol_z_current, vol_z_required
                    )
                    if breakdown_event:
                        events.append(breakdown_event)

            quality_score = self._calculate_quality_score(events, vol_z_current) if events else 0.0
            logger.debug(f"LEVEL_BREAKOUT_DETECTOR: {context.symbol} detection complete - found {len(events)} events, quality: {quality_score:.2f}")

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=quality_score,
                rejection_reason=None if events else "No valid level breakouts detected"
            )

        except Exception as e:
            logger.exception(f"LEVEL_BREAKOUT: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _detect_long_breakout(self, context: MarketContext, df: pd.DataFrame,
                             level_name: str, level_value: float, atr: float,
                             vol_z_current: float, vol_z_required: float) -> Optional[StructureEvent]:
        """Detect long breakout above resistance level."""
        current_price = context.current_price

        # Check breakout conditions
        breakout_size = current_price - level_value
        min_breakout_size = self.min_breakout_atr_mult * atr

        # 1. Price must be above level by k_atr * atr
        price_condition = current_price > level_value + (self.k_atr * atr)

        # 2. Price must hold above level for hold_bars
        if len(df) >= self.hold_bars:
            hold_condition = (df['close'].tail(self.hold_bars) > level_value).all()
        else:
            hold_condition = current_price > level_value

        # 3. Volume condition
        volume_condition = vol_z_current >= vol_z_required

        # 4. Minimum breakout size
        size_condition = breakout_size >= min_breakout_size

        # 5. Volume surge on breakout bar (optional)
        volume_surge_condition = True
        if self.require_breakout_bar_volume_surge:
            volume_surge_condition = self._check_volume_surge(df)

        if not (price_condition and hold_condition and volume_condition and size_condition and volume_surge_condition):
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Long breakout {level_name} failed conditions: "
                        f"price:{price_condition}, hold:{hold_condition}, volume:{volume_condition}, "
                        f"size:{size_condition}, surge:{volume_surge_condition}")
            return None

        # 6. SMC filtering to avoid false breakouts
        if self.enable_smc_filtering and self._is_false_breakout(df, level_value, is_long=True):
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Rejected potential liquidity grab at {level_name}")
            return None

        # Calculate strength with SMC enhancement
        base_strength = vol_z_current
        smc_strength = self._calculate_smc_strength(df, base_strength, level_value, is_long=True)
        session_weight = self._get_session_weight(context.timestamp)
        final_strength = smc_strength * session_weight

        # Calculate institutional-grade strength based on market dynamics
        # Use volume z-score as base strength (like old system's vol_z approach)
        confidence = self._calculate_institutional_strength(
            context=context,
            vol_z=vol_z_current,
            final_strength=final_strength,
            breakout_size_atr=breakout_size / atr,
            level_name=level_name
        )

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type="level_breakout_long",
            side="long",
            confidence=confidence,
            levels={level_name: level_value, "breakout_size": breakout_size},
            context={
                "level_name": level_name,
                "breakout_size_atr": breakout_size / atr,
                "volume_z": vol_z_current,
                "smc_strength": final_strength
            },
            price=current_price
        )

        logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Long breakout {level_name} at {current_price:.2f}, strength: {final_strength:.2f}")
        return event

    def _detect_short_breakdown(self, context: MarketContext, df: pd.DataFrame,
                               level_name: str, level_value: float, atr: float,
                               vol_z_current: float, vol_z_required: float) -> Optional[StructureEvent]:
        """Detect short breakdown below support level."""
        current_price = context.current_price

        # Check breakdown conditions
        breakdown_size = level_value - current_price
        min_breakdown_size = self.min_breakout_atr_mult * atr

        # 1. Price must be below level by k_atr * atr
        price_condition = current_price < level_value - (self.k_atr * atr)

        # 2. Price must hold below level for hold_bars
        if len(df) >= self.hold_bars:
            hold_condition = (df['close'].tail(self.hold_bars) < level_value).all()
        else:
            hold_condition = current_price < level_value

        # 3. Volume condition
        volume_condition = vol_z_current >= vol_z_required

        # 4. Minimum breakdown size
        size_condition = breakdown_size >= min_breakdown_size

        # 5. Volume surge on breakdown bar (optional)
        volume_surge_condition = True
        if self.require_breakout_bar_volume_surge:
            volume_surge_condition = self._check_volume_surge(df)

        if not (price_condition and hold_condition and volume_condition and size_condition and volume_surge_condition):
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Short breakdown {level_name} failed conditions")
            return None

        # 6. SMC filtering to avoid false breakouts
        if self.enable_smc_filtering and self._is_false_breakout(df, level_value, is_long=False):
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Rejected potential liquidity grab at {level_name}")
            return None

        # Calculate strength with SMC enhancement
        base_strength = vol_z_current
        smc_strength = self._calculate_smc_strength(df, base_strength, level_value, is_long=False)
        session_weight = self._get_session_weight(context.timestamp)
        final_strength = smc_strength * session_weight

        # Calculate institutional-grade strength based on market dynamics
        # Use volume z-score as base strength (like old system's vol_z approach)
        confidence = self._calculate_institutional_strength(
            context=context,
            vol_z=vol_z_current,
            final_strength=final_strength,
            breakout_size_atr=breakdown_size / atr,
            level_name=level_name
        )

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type="level_breakout_short",
            side="short",
            confidence=confidence,
            levels={level_name: level_value, "breakdown_size": breakdown_size},
            context={
                "level_name": level_name,
                "breakdown_size_atr": breakdown_size / atr,
                "volume_z": vol_z_current,
                "smc_strength": final_strength
            },
            price=current_price
        )

        logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Short breakdown {level_name} at {current_price:.2f}, strength: {final_strength:.2f}")
        return event

    def _get_available_levels(self, context: MarketContext) -> Dict[str, float]:
        """Get available key levels for breakout analysis."""
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

    def _calculate_vol_z(self, df: pd.DataFrame, window: int = 30, min_periods: int = 10) -> pd.Series:
        """Calculate volume Z-score."""
        volume_mean = df['volume'].rolling(window, min_periods=min_periods).mean()
        volume_std = df['volume'].rolling(window, min_periods=min_periods).std(ddof=0)
        vol_z = (df['volume'] - volume_mean) / volume_std.replace(0, np.nan)
        return vol_z.fillna(0)

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate ATR using percentage returns as proxy."""
        try:
            atr = df['close'].pct_change().abs().rolling(window, min_periods=5).mean().iloc[-1]
            return max(1e-9, float(atr))
        except:
            return 0.01  # 1% fallback

    def _get_time_adjusted_vol_threshold(self, timestamp: pd.Timestamp) -> float:
        """Apply time-based adjustments to volume threshold."""
        base_vol_z = self.vol_z_required

        try:
            time_minutes = timestamp.hour * 60 + timestamp.minute

            # Market hours adjustments
            if time_minutes < 630:  # Before 10:30am
                return base_vol_z * 0.5  # More lenient early market
            elif time_minutes < 720:  # 10:30am - 12:00pm
                return base_vol_z * 0.75  # Moderate adjustment
            else:  # After 12:00pm
                return base_vol_z  # Standard threshold
        except:
            return base_vol_z

    def _check_volume_surge(self, df: pd.DataFrame, window: int = 20) -> bool:
        """Check if current bar has volume surge."""
        try:
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = df['volume'].rolling(window, min_periods=10).mean().iloc[-1]

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                return volume_ratio >= self.breakout_volume_min_ratio
        except:
            pass
        return True  # Default to true if can't calculate

    def _is_false_breakout(self, df: pd.DataFrame, level: float, is_long: bool) -> bool:
        """Check if breakout is likely a false breakout (liquidity grab)."""
        if not self.enable_smc_filtering:
            return False

        try:
            # Look for quick reversal after initial breakout
            recent_bars = df.tail(3)
            tolerance = level * (self.liquidity_grab_tolerance_pct / 100)

            if is_long:
                # For long breakouts, check if price quickly returned below level
                broke_above = (recent_bars['high'] > level + tolerance).any()
                returned_below = (recent_bars['close'] < level).any()
                return broke_above and returned_below
            else:
                # For short breakdowns, check if price quickly returned above level
                broke_below = (recent_bars['low'] < level - tolerance).any()
                returned_above = (recent_bars['close'] > level).any()
                return broke_below and returned_above

        except:
            pass
        return False

    def _calculate_smc_strength(self, df: pd.DataFrame, base_strength: float, level: float, is_long: bool) -> float:
        """Calculate SMC-enhanced strength score."""
        try:
            # Base strength from volume
            smc_strength = base_strength

            # Add strength for clean breakout without immediate reversal
            if not self._is_false_breakout(df, level, is_long):
                smc_strength *= 1.2

            # Add strength for sustained move
            if len(df) >= 3:
                if is_long:
                    sustained = (df['close'].tail(3) > level).all()
                else:
                    sustained = (df['close'].tail(3) < level).all()

                if sustained:
                    smc_strength *= 1.1

            return smc_strength

        except:
            return base_strength

    def _get_session_weight(self, timestamp: pd.Timestamp) -> float:
        """Get session timing weight for breakout quality."""
        try:
            time_minutes = timestamp.hour * 60 + timestamp.minute

            # Higher weight for optimal breakout times
            if 570 <= time_minutes <= 600:  # 9:30-10:00am (opening hour)
                return 1.2
            elif 600 <= time_minutes <= 660:  # 10:00-11:00am (morning momentum)
                return 1.1
            elif 780 <= time_minutes <= 840:  # 1:00-2:00pm (afternoon session)
                return 1.1
            else:
                return 1.0
        except:
            return 1.0

    def _calculate_quality_score(self, events: List[StructureEvent], vol_z: float) -> float:
        """Calculate overall quality score for detected events."""
        if not events:
            return 0.0

        base_score = 60.0
        volume_score = min(25.0, vol_z * 8)  # Up to 25 points for volume
        event_score = len(events) * 10  # 10 points per event

        return min(100.0, base_score + volume_score + event_score)

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for level breakouts."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for level breakouts."""
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
        """Calculate risk parameters for level breakout trades."""
        entry_price = context.current_price
        atr = self._calculate_atr(context.df_5m)

        # Get the breakout/breakdown level
        level_name = event.context.get("level_name", "")
        breakout_level = None
        for level_key, level_value in event.levels.items():
            if level_key in ["PDH", "PDL", "ORH", "ORL"]:
                breakout_level = level_value
                break

        if breakout_level is None:
            breakout_level = entry_price

        # NSE FIX: Set stop loss relative to ENTRY price, not breakout level
        # Previous bug: stop at level±0.5×ATR gave only 0.75×ATR risk from entry
        # New: stop at entry±1.5×ATR gives proper risk management (47.4% hit rate per NSE data)
        if side == "long":
            hard_sl = entry_price - (atr * 1.5)  # Stop 1.5×ATR below entry
        else:
            hard_sl = entry_price + (atr * 1.5)  # Stop 1.5×ATR above entry

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels for level breakout trades."""
        entry_price = context.current_price
        atr = self._calculate_atr(context.df_5m)

        if side == "long":
            t1_target = entry_price + (atr * self.target_mult_t1)
            t2_target = entry_price + (atr * self.target_mult_t2)
        else:
            t1_target = entry_price - (atr * self.target_mult_t1)
            t2_target = entry_price - (atr * self.target_mult_t2)

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 50, "rr": self.target_mult_t1},
                {"level": t2_target, "qty_pct": 50, "rr": self.target_mult_t2}
            ],
            hard_sl=0.0,  # Set in risk_params
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank level breakout setup quality."""
        base_score = event.confidence * 100
        smc_strength = event.context.get("smc_strength", 1.0)
        volume_z = event.context.get("volume_z", 1.0)

        # Bonus for strong SMC signals
        smc_bonus = min(15.0, smc_strength * 3)
        volume_bonus = min(10.0, volume_z * 2)

        return min(100.0, base_score + smc_bonus + volume_bonus)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for level breakout trades."""
        # Level breakouts can occur throughout the session
        return True, "Level breakout timing validated"

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size based on risk management."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0  # Maximum risk per trade
        qty = max(1, min(int(max_risk_amount / risk_per_share), 100)) if risk_per_share > 0 else 1
        return qty, qty * entry_price

    def _calculate_institutional_strength(self, context: MarketContext, vol_z: float,
                                        final_strength: float, breakout_size_atr: float,
                                        level_name: str) -> float:
        """
        Calculate institutional-grade strength for level breakouts.

        Based on professional trading criteria:
        - Volume surge (vol_z as base strength like old system)
        - Breakout quality (ATR-based confirmation)
        - Level significance (PDH/PDL vs ORH/ORL)
        - Market timing context

        Returns strength in range 0.8-6.0+ to pass regime gates
        """

        # Base strength from volume participation (like old system's vol_z)
        base_strength = max(0.8, vol_z)  # Minimum viable strength

        logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Base strength from vol_z {vol_z:.2f}: {base_strength:.2f}")

        # Professional criteria multipliers
        strength_multiplier = 1.0

        # 1. Volume confirmation bonus (institutional 1.5-2× requirement)
        if vol_z >= 1.5:
            strength_multiplier *= 1.2  # 20% bonus for institutional volume
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Institutional volume bonus: 1.2x")

        if vol_z >= 2.0:
            strength_multiplier *= 1.3  # 30% bonus for strong volume surge
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Strong volume surge bonus: 1.3x")

        # 2. Breakout size quality (institutional traders prefer >1 ATR moves)
        if breakout_size_atr >= 1.0:
            strength_multiplier *= 1.15  # 15% bonus for significant breakout
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Significant breakout bonus ({breakout_size_atr:.1f} ATR): 1.15x")

        if breakout_size_atr >= 1.5:
            strength_multiplier *= 1.1  # Additional 10% for very large breakout
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Large breakout bonus: 1.1x")

        # 3. Level significance bonus
        key_levels = ["PDH", "PDL"]  # Previous day levels more significant
        session_levels = ["ORH", "ORL"]  # Opening range levels less significant

        if level_name in key_levels:
            strength_multiplier *= 1.25  # 25% bonus for key daily levels
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Key level bonus ({level_name}): 1.25x")
        elif level_name in session_levels:
            strength_multiplier *= 1.1  # 10% bonus for session levels
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Session level bonus ({level_name}): 1.1x")

        # 4. SMC strength integration (already calculated)
        if final_strength > 1.5:
            strength_multiplier *= 1.1  # 10% bonus for strong SMC confirmation
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - SMC confirmation bonus: 1.1x")

        # 5. Session timing context (peak liquidity hours)
        if hasattr(context, 'timestamp'):
            hour = context.timestamp.hour
            if 9 <= hour <= 11 or 14 <= hour <= 16:  # Peak trading hours IST
                strength_multiplier *= 1.1  # 10% bonus for peak liquidity
                logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Peak hours bonus (hour {hour}): 1.1x")

        # Calculate final institutional strength
        institutional_strength = base_strength * strength_multiplier

        logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Institutional strength: "
                   f"base {base_strength:.2f} × multiplier {strength_multiplier:.2f} = {institutional_strength:.2f}")

        return institutional_strength