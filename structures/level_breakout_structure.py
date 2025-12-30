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

        # Track which specific setup type this detector is for (e.g., "level_breakout_long")
        self.configured_setup_type = config.get("_setup_name", None)

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

        # NEW: Dual-mode configuration (aggressive + retest)
        # All configuration must come from config file - no defaults
        self.entry_mode = config["entry_mode"]  # "immediate", "retest", or "pending" (legacy)
        self.aggressive_min_volume_z = config["aggressive_min_volume_z"]
        self.aggressive_min_conviction = config["aggressive_min_conviction"]
        self.retest_entry_zone_width_atr = config["retest_entry_zone_width_atr"]
        self.retest_timeout_minutes = config["retest_timeout_minutes"]
        self.allow_both_modes = config["allow_both_modes"]

        # Stop loss parameters - Pro trader: SL at breakout level + ATR buffer
        self.sl_atr_multiplier = config["sl_atr_multiplier"]  # ATR multiplier for stop loss
        self.min_stop_distance_pct = config["min_stop_distance_pct"]  # Minimum SL distance as % of price

        # Track traded breakouts to prevent double exposure
        self.traded_breakouts_today = set()

        logger.debug(f"LEVEL_BREAKOUT: Initialized with DUAL-MODE CONFIG: entry_mode={self.entry_mode}, aggressive_vol_z={self.aggressive_min_volume_z}, aggressive_conviction={self.aggressive_min_conviction}, allow_both_modes={self.allow_both_modes}, retest_zone_width={self.retest_entry_zone_width_atr}ATR")


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

            # Filter events to only include those matching configured setup type
            if self.configured_setup_type and events:
                filtered_events = [e for e in events if e.structure_type == self.configured_setup_type]
                if len(filtered_events) < len(events):
                    logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Filtered {len(events)}→{len(filtered_events)} events (configured for {self.configured_setup_type})")
                events = filtered_events

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

        # INSTITUTIONAL FILTERS - Based on spike test analysis
        # Analysis showed conviction filter successfully blocks losers (44.6% win rate)
        # while level_cleanness incorrectly blocked winners (55.1% win rate)

        # 7. Timing filter - RE-ENABLED but EXEMPT ORB trades (9:15-9:45 AM filter, but allow ORH/ORL)
        timing_valid, timing_rejection = self._check_institutional_timing(context.timestamp, level_name)
        if not timing_valid:
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {timing_rejection}")
            return None

        # 8. Candle conviction filter - KEEP (successfully blocks 2563 losing trades, saves Rs.2359)
        conviction_valid, conviction_rejection = self._check_candle_conviction(df, is_long=True)
        if not conviction_valid:
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {conviction_rejection}")
            return None

        # 9. Volume accumulation filter - DISABLED (spike tests showed zero impact)
        # accumulation_valid, accumulation_rejection = self._check_volume_accumulation(df)
        # if not accumulation_valid:
        #     logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {accumulation_rejection}")
        #     return None

        # 10. Level cleanness filter - DISABLED (blocked 527 winners worth Rs.1965, 55.1% win rate)
        # cleanness_valid, cleanness_rejection = self._check_level_cleanness(df, level_value, is_long=True)
        # if not cleanness_valid:
        #     logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {cleanness_rejection}")
        #     return None

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

        # NEW: Determine entry mode based on configuration and breakout quality
        entry_mode = self._determine_entry_mode(vol_z_current, confidence, context.symbol, level_name, context.timestamp)

        # Skip if already traded or doesn't meet criteria
        if entry_mode is None:
            return None

        # Use orb_level_breakout_* for ORH/ORL to ensure is_orb check passes in breakout_pipeline
        # This allows ORB-related level breakouts to use relaxed chop regime filtering
        structure_type = "orb_level_breakout_long" if level_name in ("ORH", "ORL") else "level_breakout_long"

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type=structure_type,
            side="long",
            confidence=confidence,
            levels={level_name: level_value, "breakout_size": breakout_size},
            context={
                "level_name": level_name,
                "breakout_size_atr": breakout_size / atr,
                "volume_z": vol_z_current,
                "smc_strength": final_strength,
                "entry_mode": entry_mode,  # NEW: Mark entry mode
                "retest_zone": [level_value - (self.retest_entry_zone_width_atr * atr),
                               level_value + (self.retest_entry_zone_width_atr * atr)] if entry_mode == "retest" else None
            },
            price=current_price
        )

        logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - Long breakout {level_name} at {current_price:.2f}, strength: {final_strength:.2f}, entry_mode: {entry_mode}")
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

        # INSTITUTIONAL FILTERS - Based on spike test analysis
        # Analysis showed conviction filter successfully blocks losers (44.6% win rate)
        # while level_cleanness incorrectly blocked winners (55.1% win rate)

        # 7. Timing filter - RE-ENABLED but EXEMPT ORB trades (9:15-9:45 AM filter, but allow ORH/ORL)
        timing_valid, timing_rejection = self._check_institutional_timing(context.timestamp, level_name)
        if not timing_valid:
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {timing_rejection}")
            return None

        # 8. Candle conviction filter - KEEP (successfully blocks 2563 losing trades, saves Rs.2359)
        conviction_valid, conviction_rejection = self._check_candle_conviction(df, is_long=False)
        if not conviction_valid:
            logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {conviction_rejection}")
            return None

        # 9. Volume accumulation filter - DISABLED (spike tests showed zero impact)
        # accumulation_valid, accumulation_rejection = self._check_volume_accumulation(df)
        # if not accumulation_valid:
        #     logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {accumulation_rejection}")
        #     return None

        # 10. Level cleanness filter - DISABLED (blocked 527 winners worth Rs.1965, 55.1% win rate)
        # cleanness_valid, cleanness_rejection = self._check_level_cleanness(df, level_value, is_long=False)
        # if not cleanness_valid:
        #     logger.debug(f"LEVEL_BREAKOUT: {context.symbol} - {cleanness_rejection}")
        #     return None

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

        # Use orb_level_breakout_* for ORH/ORL to ensure is_orb check passes in breakout_pipeline
        # This allows ORB-related level breakouts to use relaxed chop regime filtering
        structure_type = "orb_level_breakout_short" if level_name in ("ORH", "ORL") else "level_breakout_short"

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type=structure_type,
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
        """Calculate ATR using percentage returns, converted to absolute price units."""
        try:
            # Calculate percentage-based ATR
            atr_pct = df['close'].pct_change().abs().rolling(window, min_periods=5).mean().iloc[-1]

            # Convert to absolute price units using current price
            current_price = df['close'].iloc[-1]
            atr_absolute = atr_pct * current_price

            return max(0.01, float(atr_absolute))  # Minimum 0.01 points
        except:
            # Fallback: 1% of current price
            current_price = df['close'].iloc[-1] if len(df) > 0 else 100.0
            return max(0.01, current_price * 0.01)

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
        """
        Get session timing weight for breakout quality.

        INDIAN MARKET CHARACTERISTICS:
        - 9:15-9:45am: RETAIL NOISE - Reject entirely via timing filter
        - 9:45-10:30am: Early institutional - Neutral weight
        - 10:30am-2:00pm: PEAK INSTITUTIONAL - Bonus weight
        - 2:00pm-3:30pm: Late session - Neutral weight
        """
        try:
            time_minutes = timestamp.hour * 60 + timestamp.minute

            # INSTITUTIONAL PEAK HOURS (10:30am - 2:00pm)
            # This is when real institutions trade, not retail noise
            if 630 <= time_minutes <= 840:  # 10:30am-2:00pm
                return 1.2  # 20% bonus for institutional hours

            # Early institutional hours (9:45am-10:30am)
            elif 585 <= time_minutes < 630:  # 9:45am-10:30am
                return 1.0  # Neutral (retail noise has subsided)

            # Late session (2:00pm-3:30pm)
            elif 840 < time_minutes <= 930:  # 2:00pm-3:30pm
                return 1.0  # Neutral

            # Should never reach here due to timing filter rejecting pre-9:45am
            else:
                return 0.8  # Penalize edge cases
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
        qty, notional = 0, 0.0  # Pipeline overrides with proper sizing

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
        # Use configured sl_atr_multiplier - no default, must be in config
        if side == "long":
            hard_sl = entry_price - (atr * self.sl_atr_multiplier)  # Stop sl_atr_multiplier×ATR below entry
        else:
            hard_sl = entry_price + (atr * self.sl_atr_multiplier)  # Stop sl_atr_multiplier×ATR above entry

        # Enforce minimum stop distance
        min_stop_distance = entry_price * (self.min_stop_distance_pct / 100.0)
        risk_per_share = abs(entry_price - hard_sl)
        if risk_per_share < min_stop_distance:
            if side == "long":
                hard_sl = entry_price - min_stop_distance
            else:
                hard_sl = entry_price + min_stop_distance
            risk_per_share = min_stop_distance

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

    def _check_institutional_timing(self, timestamp: pd.Timestamp, level_name: str = None) -> Tuple[bool, str]:
        """
        INSTITUTIONAL TIMING FILTER for Indian markets.

        REJECT breakouts during retail noise periods (9:15-9:45am).
        EXCEPT for ORB (Opening Range Breakout) trades - ORH/ORL are valid 9:30-10:30.

        Institutional traders wait for retail frenzy to subside before committing capital,
        BUT ORB is a legitimate strategy where the opening range (9:15-9:30) defines levels
        and breakouts from 9:35-10:30 are high-probability institutional moves.

        Returns (is_valid, rejection_reason)
        """
        try:
            time_minutes = timestamp.hour * 60 + timestamp.minute

            # EXEMPT ORB trades (ORH/ORL) from timing filter
            # ORB is a valid institutional strategy with 9:30 opening range
            if level_name in ("ORH", "ORL"):
                return True, ""  # Allow ORB trades at any time

            # CRITICAL: Reject pre-institutional hours (9:15-9:45am) for PDH/PDL
            # This is when retail traders chase gaps and create false breakouts
            if 555 <= time_minutes < 585:  # 9:15am - 9:45am
                return False, "Rejected: Pre-institutional hours (9:15-9:45am retail noise, non-ORB)"

            # Accept institutional trading hours (9:45am onwards)
            return True, ""

        except:
            # If can't determine time, be conservative and accept
            return True, ""

    def _check_candle_conviction(self, df: pd.DataFrame, is_long: bool) -> Tuple[bool, str]:
        """
        CANDLE CONVICTION FILTER.

        Professional traders want to see CONVICTION in the breakout candle:
        - For longs: Close in top 70% of bar range (not doji/upper wick)
        - For shorts: Close in bottom 30% of bar range (not doji/lower wick)

        Weak candles (close in middle/opposite end) indicate indecision = high failure rate

        Returns (is_valid, rejection_reason)
        """
        try:
            current_bar = df.iloc[-1]
            bar_high = float(current_bar['high'])
            bar_low = float(current_bar['low'])
            bar_close = float(current_bar['close'])

            bar_range = bar_high - bar_low

            if bar_range < 1e-9:  # No range = doji
                return False, "Rejected: Doji candle (no conviction)"

            # Calculate where close is in the bar (0 = low, 1 = high)
            close_position = (bar_close - bar_low) / bar_range

            if is_long:
                # Long breakouts need close in top 30% (close_position > 0.7)
                if close_position < 0.7:
                    return False, f"Rejected: Weak long candle (close at {close_position:.1%} of range, need >70%)"
            else:
                # Short breakouts need close in bottom 30% (close_position < 0.3)
                if close_position > 0.3:
                    return False, f"Rejected: Weak short candle (close at {close_position:.1%} of range, need <30%)"

            return True, ""

        except:
            # If can't calculate, be lenient and accept
            return True, ""

    def _check_volume_accumulation(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[bool, str]:
        """
        VOLUME ACCUMULATION FILTER.

        Institutional breakouts are preceded by accumulation/distribution.
        Check that prior 3-5 bars show above-average volume (vol_z > 1.0).

        This filters out breakouts that happen on a single volume spike with no buildup.

        Returns (is_valid, rejection_reason)
        """
        try:
            if 'vol_z' not in df.columns or len(df) < lookback:
                return True, ""  # Can't check, be lenient

            # Check prior bars (not including current bar)
            prior_bars = df['vol_z'].iloc[-(lookback+1):-1]  # Last 5 bars before current

            # Calculate how many bars had institutional volume (vol_z > 1.0)
            institutional_bars = (prior_bars > 1.0).sum()

            # Need at least 3 out of last 5 bars to have elevated volume
            min_required = 3

            if institutional_bars < min_required:
                return False, f"Rejected: No volume accumulation ({institutional_bars}/{lookback} bars with vol_z>1.0, need >={min_required})"

            return True, ""

        except:
            return True, ""

    def _check_level_cleanness(self, df: pd.DataFrame, level_value: float, is_long: bool, lookback: int = 20) -> Tuple[bool, str]:
        """
        LEVEL CLEANNESS FILTER.

        Professional traders prefer CLEAN levels that haven't been tested multiple times.
        A level with many touches (>3 in last 20 bars) is a support/resistance zone,
        not a clean breakout level. These often fail or chop around.

        Returns (is_valid, rejection_reason)
        """
        try:
            if len(df) < lookback:
                return True, ""  # Not enough data, be lenient

            recent_bars = df.tail(lookback)

            # Define "touch" tolerance (within 0.5% of level)
            tolerance = level_value * 0.005

            # Count how many bars touched the level
            if is_long:
                # For long breakouts, count bars that got close to resistance from below
                touches = ((recent_bars['high'] >= level_value - tolerance) &
                          (recent_bars['high'] <= level_value + tolerance)).sum()
            else:
                # For short breakouts, count bars that got close to support from above
                touches = ((recent_bars['low'] >= level_value - tolerance) &
                          (recent_bars['low'] <= level_value + tolerance)).sum()

            max_allowed_touches = 3

            if touches > max_allowed_touches:
                return False, f"Rejected: Level not clean ({touches} touches in last {lookback} bars, max {max_allowed_touches})"

            return True, ""

        except:
            return True, ""

    def _determine_entry_mode(self, vol_z: float, confidence: float, symbol: str, level_name: str, timestamp: datetime) -> str:
        """
        Determine entry mode (immediate vs retest) based on configuration and breakout quality.

        Returns:
            "immediate" - Enter immediately on detection (aggressive mode)
            "retest" - Wait for pullback to level (retest mode)
            "pending" - Legacy mode (tight entry zone around current price)
        """
        # Check if this breakout was already traded today
        breakout_key = f"{symbol}_{level_name}_{timestamp.date()}"

        if breakout_key in self.traded_breakouts_today:
            logger.debug(f"LEVEL_BREAKOUT: {symbol} - {level_name} already traded today, skipping")
            return None  # Signal to skip this detection

        # If specific mode is configured, use it
        if self.entry_mode == "immediate":
            # Aggressive mode: Check if meets high-conviction criteria
            if vol_z >= self.aggressive_min_volume_z and confidence >= self.aggressive_min_conviction:
                logger.debug(f"LEVEL_BREAKOUT: {symbol} - AGGRESSIVE mode (vol_z={vol_z:.2f}, conf={confidence:.2f})")
                self.traded_breakouts_today.add(breakout_key)
                return "immediate"
            elif self.allow_both_modes:
                # Doesn't meet aggressive criteria, try retest mode
                logger.debug(f"LEVEL_BREAKOUT: {symbol} - Failed aggressive criteria, falling back to RETEST mode")
                return "retest"
            else:
                # Doesn't meet criteria and no fallback allowed
                logger.debug(f"LEVEL_BREAKOUT: {symbol} - Failed aggressive criteria (vol_z={vol_z:.2f}<{self.aggressive_min_volume_z}, conf={confidence:.2f}<{self.aggressive_min_conviction})")
                return None

        elif self.entry_mode == "retest":
            # Retest mode: Always wait for pullback
            logger.debug(f"LEVEL_BREAKOUT: {symbol} - RETEST mode (will wait for pullback)")
            return "retest"

        else:
            # Legacy pending mode (current behavior)
            return "pending"