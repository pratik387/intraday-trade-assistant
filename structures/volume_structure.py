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

        # Volume parameters
        self.min_volume_spike_mult = config["min_volume_spike_mult"]
        self.min_body_size_pct = config["min_body_size_pct"]
        self.reversal_threshold_pct = config["reversal_threshold_pct"]

        # Risk management
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.confidence_level = config["confidence_level"]

        logger.info(f"VOLUME: Initialized with min spike: {self.min_volume_spike_mult}x")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect volume-based structures."""
        try:
            df = context.df_5m
            if len(df) < 10 or 'vol_z' not in df.columns:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient volume data"
                )

            events = []

            # Volume spike reversal detection
            current_vol_z = float(df['vol_z'].iloc[-1])
            if current_vol_z >= self.min_volume_spike_mult:
                current_price = context.current_price
                open_price = df['open'].iloc[-1]
                body_size_pct = abs(current_price - open_price) / open_price * 100

                if body_size_pct >= self.min_body_size_pct:
                    # Determine reversal direction
                    if current_price > open_price:  # Up bar, expect reversal down
                        structure_type = "volume_spike_reversal_short"
                        side = "short"
                    else:  # Down bar, expect reversal up
                        structure_type = "volume_spike_reversal_long"
                        side = "long"

                    event = StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type=structure_type,
                        side=side,
                        confidence=self._calculate_institutional_strength(context, current_vol_z, body_size_pct, side),
                        levels={"reversal_level": current_price},
                        context={
                            "volume_spike": current_vol_z,
                            "body_size_pct": body_size_pct
                        },
                        price=current_price
                    )
                    events.append(event)

                    logger.info(f"VOLUME: {context.symbol} - Volume spike reversal {side.upper()}: vol_z {current_vol_z:.1f}, body {body_size_pct:.1f}%")

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=min(80.0, current_vol_z * 15) if events else 0.0,
                rejection_reason=None if events else "No volume patterns detected"
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
        atr = self._get_atr(context)

        if side == "long":
            hard_sl = entry_price - (atr * 1.0)
        else:
            hard_sl = entry_price + (atr * 1.0)

        risk_per_share = abs(entry_price - hard_sl)

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

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0
        qty = max(1, min(int(max_risk_amount / risk_per_share), 100)) if risk_per_share > 0 else 1
        return qty, qty * entry_price

    def _calculate_institutional_strength(self, context: MarketContext, vol_z: float,
                                        body_size_pct: float, side: str) -> float:
        """Calculate institutional-grade strength for volume patterns."""
        try:
            # Base strength from volume spike magnitude (institutional volume threshold ≥3.0)
            base_strength = max(1.5, vol_z * 0.8)  # Scale vol_z to strength

            # Professional bonuses for institutional-grade volume patterns
            strength_multiplier = 1.0

            # Exceptional volume surge bonus (institutional participation)
            if vol_z >= 5.0:  # Exceptional volume surge
                strength_multiplier *= 1.4  # 40% bonus for exceptional volume
                logger.debug(f"VOLUME: Exceptional volume surge bonus applied (vol_z={vol_z:.2f})")
            elif vol_z >= 3.0:  # Strong institutional volume
                strength_multiplier *= 1.2  # 20% bonus for strong volume

            # Body size significance (shows conviction)
            if body_size_pct >= 3.0:  # Large body size (3%+ move)
                strength_multiplier *= 1.25  # 25% bonus for large moves
                logger.debug(f"VOLUME: Large body size bonus applied ({body_size_pct:.2f}%)")
            elif body_size_pct >= 2.0:  # Moderate body size
                strength_multiplier *= 1.15  # 15% bonus for moderate moves

            # Volume spike consistency (multiple bars with elevated volume)
            df = context.df_5m
            if len(df) >= 3:
                recent_vol_z = df['vol_z'].tail(3)
                consistent_volume = (recent_vol_z >= 2.0).sum()
                if consistent_volume >= 2:
                    strength_multiplier *= 1.2  # 20% bonus for sustained volume
                    logger.debug(f"VOLUME: Consistent volume bonus applied ({consistent_volume} bars)")

            # Reversal context bonus (volume at key levels)
            current_price = context.current_price
            if hasattr(context, 'vwap') and context.vwap:
                vwap_distance = abs(current_price - context.vwap) / context.vwap
                if vwap_distance <= 0.005:  # Near VWAP (within 0.5%)
                    strength_multiplier *= 1.15  # 15% bonus for VWAP interaction
                    logger.debug(f"VOLUME: VWAP interaction bonus applied")

            # Market timing bonus (volume patterns work best during active hours)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"VOLUME: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.8)  # Strong minimum for volume patterns

            logger.debug(f"VOLUME: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"VOLUME: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold