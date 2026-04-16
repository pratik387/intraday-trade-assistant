"""
Volume Structure Implementation

This module implements volume-based trading structures including:
- Volume spike reversal (exhaustion after volume spike)
- Volume breakout (breakout with volume confirmation)
- Volume dry-up (low volume before move)

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


class VolumeStructure(BaseStructure):
    """Volume-based structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Volume structure with configuration."""
        super().__init__(config)
        self.structure_type = "volume"

        # Track which specific setup type this detector is for (e.g., "volume_spike_reversal_long")
        # This ensures we only produce signals for the configured direction
        self.configured_setup_type = config.get("_setup_name", None)

        # Volume parameters
        self.min_volume_spike_mult = config["min_volume_spike_mult"]
        self.min_body_size_pct = config["min_body_size_pct"]
        self.reversal_threshold_pct = config["reversal_threshold_pct"]

        # Risk management
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.confidence_level = config["confidence_level"]

        # Stop loss parameters - Pro trader: SL at spike candle extreme + ATR buffer
        self.spike_sl_buffer_atr = config["spike_sl_buffer_atr"]  # ATR buffer beyond spike candle extreme
        self.min_stop_distance_pct = config["min_stop_distance_pct"]  # Minimum SL distance as % of price

        # P2: Wick rejection ratio — filter out momentum candles masquerading as exhaustion
        self.min_rejection_wick_ratio = config["min_rejection_wick_ratio"]
        self.wick_bonus_threshold = config["wick_bonus_threshold"]

        # P3: S/R proximity — boost confidence when spike is at key level
        self.sr_proximity_atr_mult = config["sr_proximity_atr_mult"]
        self.sr_confluence_bonus = config["sr_confluence_bonus"]

        # P5: Volume ratio dual check — prevent illiquid z-score noise
        self.min_volume_ratio = config["min_volume_ratio"]

        # P6: Multi-bar exhaustion pattern
        self.multi_bar_vol_threshold = config["multi_bar_vol_threshold"]
        self.multi_bar_lookback = config["multi_bar_lookback"]
        self.multi_bar_exhaustion_bonus = config["multi_bar_exhaustion_bonus"]

        # Config-driven cap blocking (default empty — no cap blocks unless explicitly set).
        # Mirrors RangeStructure / SupportResistanceStructure / FHMStructure pattern.
        # Per audit/06-volume_structure.md P1 #1.
        self.blocked_cap_segments = set(config.get("blocked_cap_segments", []))

        logger.debug(f"VOLUME: Initialized with min spike: {self.min_volume_spike_mult}x")
        logger.debug(f"VOLUME: SL params - spike_buffer: {self.spike_sl_buffer_atr}ATR, min_distance: {self.min_stop_distance_pct}%")
        logger.debug(f"VOLUME: Wick filter: min_ratio={self.min_rejection_wick_ratio}, S/R proximity: {self.sr_proximity_atr_mult} ATR")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect volume-based structures with structural context."""
        try:
            df = context.df_5m
            if len(df) < 10 or 'vol_z' not in df.columns:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient volume data"
                )

            # Config-driven cap blocking — applies to all volume-spike-reversal setups.
            # Per audit/06-volume_structure.md P1 #1.
            cap_segment = getattr(context, 'cap_segment', None)
            if cap_segment in self.blocked_cap_segments:
                logger.debug(f"VOLUME_BLOCK: {context.symbol} | Cap={cap_segment} in blocked_cap_segments, skipping")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"cap_segment {cap_segment} blocked"
                )

            events = []

            # Volume spike reversal detection
            current_vol_z = float(df['vol_z'].iloc[-1])
            if current_vol_z < self.min_volume_spike_mult:
                return StructureAnalysis(
                    structure_detected=False, events=[], quality_score=0.0,
                    rejection_reason="No volume patterns detected"
                )

            current_price = context.current_price
            candle = df.iloc[-1]
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            body_size_pct = abs(current_price - open_price) / open_price * 100

            if body_size_pct < self.min_body_size_pct:
                return StructureAnalysis(
                    structure_detected=False, events=[], quality_score=0.0,
                    rejection_reason="Body size too small"
                )

            # Determine reversal direction
            if current_price > open_price:  # Up bar, expect reversal down
                structure_type = "volume_spike_reversal_short"
                side = "short"
            else:  # Down bar, expect reversal up
                structure_type = "volume_spike_reversal_long"
                side = "long"

            # Setup type filter
            if self.configured_setup_type and structure_type != self.configured_setup_type:
                logger.debug(f"VOLUME: {context.symbol} - Skipping {structure_type} (configured for {self.configured_setup_type})")
                return StructureAnalysis(
                    structure_detected=False, events=[], quality_score=0.0,
                    rejection_reason=f"Direction mismatch: {structure_type} != {self.configured_setup_type}"
                )

            # --- P5: Volume ratio dual check (prevent illiquid z-score noise) ---
            if len(df) >= 20:
                vol_median = float(df['volume'].tail(20).median())
                vol_ratio = float(candle['volume'] / vol_median) if vol_median > 0 else 0.0
            else:
                vol_ratio = float(current_vol_z)  # Fallback to z-score as proxy

            if vol_ratio < self.min_volume_ratio:
                logger.debug(f"VOLUME: {context.symbol} - vol_ratio {vol_ratio:.1f}x < {self.min_volume_ratio}x (illiquid noise)")
                return StructureAnalysis(
                    structure_detected=False, events=[], quality_score=0.0,
                    rejection_reason=f"Volume ratio {vol_ratio:.1f}x below minimum {self.min_volume_ratio}x"
                )

            # --- P2: Wick rejection ratio (filter momentum candles) ---
            total_range = high_price - low_price
            wick_ratio = 0.0
            if total_range > 0:
                if side == "long":  # Down bar → want lower wick (rejection of lows)
                    lower_wick = min(open_price, current_price) - low_price
                    wick_ratio = lower_wick / total_range
                else:  # Up bar → want upper wick (rejection of highs)
                    upper_wick = high_price - max(open_price, current_price)
                    wick_ratio = upper_wick / total_range

            if wick_ratio < self.min_rejection_wick_ratio:
                logger.debug(f"VOLUME: {context.symbol} - wick_ratio {wick_ratio:.2f} < {self.min_rejection_wick_ratio} (momentum candle, not exhaustion)")
                return StructureAnalysis(
                    structure_detected=False, events=[], quality_score=0.0,
                    rejection_reason=f"Wick ratio {wick_ratio:.2f} below minimum {self.min_rejection_wick_ratio}"
                )

            # --- P3: S/R proximity check (structural confluence) ---
            atr = self._get_atr(context)
            proximity_threshold = atr * self.sr_proximity_atr_mult
            nearest_level_name = None
            nearest_level_dist = float('inf')

            key_levels = []
            if hasattr(context, 'pdh') and context.pdh and not np.isnan(context.pdh):
                key_levels.append(("PDH", context.pdh))
            if hasattr(context, 'pdl') and context.pdl and not np.isnan(context.pdl):
                key_levels.append(("PDL", context.pdl))
            if hasattr(context, 'orh') and context.orh and not np.isnan(context.orh):
                key_levels.append(("ORH", context.orh))
            if hasattr(context, 'orl') and context.orl and not np.isnan(context.orl):
                key_levels.append(("ORL", context.orl))
            if 'vwap' in df.columns:
                vwap_val = float(df['vwap'].iloc[-1])
                if not np.isnan(vwap_val):
                    key_levels.append(("VWAP", vwap_val))

            for name, level in key_levels:
                dist = abs(current_price - level)
                if dist < nearest_level_dist:
                    nearest_level_dist = dist
                    nearest_level_name = name

            at_key_level = nearest_level_dist <= proximity_threshold if key_levels else False

            # --- P6: Multi-bar exhaustion pattern ---
            multi_bar_exhaustion = False
            elevated_bar_count = 0
            if len(df) >= self.multi_bar_lookback:
                recent = df.tail(self.multi_bar_lookback)
                elevated_bar_count = int((recent['vol_z'] >= self.multi_bar_vol_threshold).sum())
                closes = recent['close'].values
                opens = recent['open'].values
                if side == "long":  # Consecutive down bars
                    direction_consistent = sum(1 for c, o in zip(closes, opens) if c < o)
                else:  # Consecutive up bars
                    direction_consistent = sum(1 for c, o in zip(closes, opens) if c > o)
                multi_bar_exhaustion = (elevated_bar_count >= 2 and direction_consistent >= 2)

            # Build enriched event context
            event_context = {
                "volume_spike": current_vol_z,
                "body_size_pct": body_size_pct,
                "wick_ratio": round(wick_ratio, 3),
                "vol_ratio": round(vol_ratio, 1),
                "at_key_level": at_key_level,
                "nearest_level": nearest_level_name,
                "nearest_level_dist_atr": round(nearest_level_dist / atr, 2) if atr > 0 else 0.0,
                "multi_bar_exhaustion": multi_bar_exhaustion,
                "elevated_bar_count": elevated_bar_count,
            }

            confidence = self._calculate_institutional_strength(
                context, current_vol_z, body_size_pct, side,
                wick_ratio=wick_ratio, at_key_level=at_key_level,
                multi_bar_exhaustion=multi_bar_exhaustion
            )

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type=structure_type,
                side=side,
                confidence=confidence,
                levels={"reversal_level": current_price},
                context=event_context,
                price=current_price
            )
            events.append(event)
            logger.debug(
                f"VOLUME: {context.symbol} - Spike reversal {side.upper()}: vol_z={current_vol_z:.1f}, "
                f"body={body_size_pct:.1f}%, wick={wick_ratio:.2f}, vol_ratio={vol_ratio:.1f}x, "
                f"at_level={nearest_level_name if at_key_level else 'none'}, multi_bar={multi_bar_exhaustion}"
            )

            return StructureAnalysis(
                structure_detected=True,
                events=events,
                quality_score=min(80.0, current_vol_z * 15),
                rejection_reason=None
            )

        except Exception as e:
            logger.error(f"VOLUME: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for volume setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for volume setups."""
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
        """Calculate risk parameters using pro trader spike candle-based SL."""
        entry_price = context.current_price
        atr = self._get_atr(context)
        df = context.df_5m

        # Pro trader: SL at spike candle extreme + ATR buffer
        # For reversal trades, the spike candle extreme is the logical invalidation point
        if len(df) >= 1:
            spike_candle_low = float(df['low'].iloc[-1])
            spike_candle_high = float(df['high'].iloc[-1])

            if side == "long":
                # Long reversal: SL below the spike candle low (the selling climax low)
                hard_sl = spike_candle_low - (atr * self.spike_sl_buffer_atr)
            else:
                # Short reversal: SL above the spike candle high (the buying climax high)
                hard_sl = spike_candle_high + (atr * self.spike_sl_buffer_atr)
        else:
            # Fallback if no candle data
            if side == "long":
                hard_sl = entry_price - (atr * self.spike_sl_buffer_atr)
            else:
                hard_sl = entry_price + (atr * self.spike_sl_buffer_atr)

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
        """Calculate exit levels."""
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
                {"level": t1_target, "qty_pct": 50, "rr": 1.0},
                {"level": t2_target, "qty_pct": 50, "rr": 2.0}
            ],
            hard_sl=0.0,
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank volume setup quality."""
        base_score = event.confidence * 100
        volume_spike = event.context.get("volume_spike", 0)
        return min(100.0, base_score + volume_spike * 5)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing."""
        return True, "Volume timing validated"

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR with fallback."""
        if context.indicators and 'atr' in context.indicators:
            return context.indicators['atr']
        return context.current_price * 0.01

    def _calculate_institutional_strength(self, context: MarketContext, vol_z: float,
                                        body_size_pct: float, side: str, *,
                                        wick_ratio: float = 0.0,
                                        at_key_level: bool = False,
                                        multi_bar_exhaustion: bool = False) -> float:
        """Calculate institutional-grade strength for volume patterns."""
        try:
            # Base strength from volume spike magnitude (institutional volume threshold ≥3.0)
            base_strength = max(1.5, vol_z * 0.8)  # Scale vol_z to strength

            # Professional bonuses for institutional-grade volume patterns
            strength_multiplier = 1.0

            # Exceptional volume surge bonus (institutional participation)
            if vol_z >= 5.0:
                strength_multiplier *= 1.4
            elif vol_z >= 3.0:
                strength_multiplier *= 1.2

            # Body size significance (shows conviction)
            if body_size_pct >= 3.0:
                strength_multiplier *= 1.25
            elif body_size_pct >= 2.0:
                strength_multiplier *= 1.15

            # P2: Wick rejection bonus — strong wicks confirm exhaustion
            if wick_ratio >= self.wick_bonus_threshold:
                strength_multiplier *= 1.2  # 20% bonus for strong rejection wick
                logger.debug(f"VOLUME: Strong wick rejection bonus (ratio={wick_ratio:.2f})")

            # P3: S/R confluence bonus — spike at key level is higher probability
            if at_key_level:
                strength_multiplier *= self.sr_confluence_bonus
                logger.debug(f"VOLUME: S/R confluence bonus applied ({self.sr_confluence_bonus}x)")

            # P6: Multi-bar exhaustion bonus — sustained climactic volume
            if multi_bar_exhaustion:
                strength_multiplier *= self.multi_bar_exhaustion_bonus
                logger.debug(f"VOLUME: Multi-bar exhaustion bonus ({self.multi_bar_exhaustion_bonus}x)")

            # Market timing bonus (volume patterns work best during active hours)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:
                strength_multiplier *= 1.1

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier
            final_strength = max(final_strength, 1.8)

            logger.debug(f"VOLUME: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"VOLUME: Error calculating institutional strength: {e}")
            return 1.8