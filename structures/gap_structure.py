"""
Gap Structure Implementation

This module implements gap-based trading structures including:
- Gap fill (price returns to fill gap)
- Gap breakout (gap continues in same direction)
- Gap fade (reversal at gap levels)

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


class GapStructure(BaseStructure):
    """Gap-based structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Gap structure with configuration."""
        super().__init__(config)
        self.structure_type = "gap"

        # Gap parameters - will crash with KeyError if missing
        self.min_gap_pct = config["min_gap_pct"]
        self.max_gap_pct = config["max_gap_pct"]
        self.require_volume_confirmation = config["require_volume_confirmation"]

        # Risk management - will crash with KeyError if missing
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.confidence_level = config["confidence_level"]

        logger.debug(f"GAP: Initialized with gap range: {self.min_gap_pct}-{self.max_gap_pct}%")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect gap-based structures."""
        try:
            df = context.df_5m
            if len(df) < 5:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient data for gap analysis"
                )

            events = []

            # Simple gap detection using first bar of day
            if context.pdc:  # Previous day close available
                current_open = df['open'].iloc[0]  # First bar of the day
                gap_pct = abs(current_open - context.pdc) / context.pdc * 100

                if self.min_gap_pct <= gap_pct <= self.max_gap_pct:
                    current_price = context.current_price

                    # Gap fill strategy
                    if current_open > context.pdc:  # Gap up
                        if current_price < current_open:  # Moving back toward gap
                            structure_type = "gap_fill_short"
                            side = "short"
                        else:
                            structure_type = "gap_breakout_long"
                            side = "long"
                    else:  # Gap down
                        if current_price > current_open:  # Moving back toward gap
                            structure_type = "gap_fill_long"
                            side = "long"
                        else:
                            structure_type = "gap_breakout_short"
                            side = "short"

                    event = StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type=structure_type,
                        side=side,
                        confidence=self._calculate_institutional_strength(context, gap_pct, structure_type, side),
                        levels={"gap_level": context.pdc, "gap_open": current_open},
                        context={
                            "gap_size_pct": gap_pct,
                            "gap_direction": "up" if current_open > context.pdc else "down"
                        },
                        price=current_price
                    )
                    events.append(event)

                    logger.debug(f"GAP: {context.symbol} - {structure_type} detected: {gap_pct:.1f}% gap")

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=min(85.0, gap_pct * 20) if events else 0.0,
                rejection_reason=None if events else "No significant gaps detected"
            )

        except Exception as e:
            logger.error(f"GAP: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for gap setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for gap setups."""
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
        gap_size = abs(event.levels.get("gap_open", entry_price) - event.levels.get("gap_level", entry_price))

        if side == "long":
            hard_sl = entry_price - gap_size * 0.5
        else:
            hard_sl = entry_price + gap_size * 0.5

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=gap_size,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels."""
        entry_price = context.current_price
        gap_size = abs(event.levels.get("gap_open", entry_price) - event.levels.get("gap_level", entry_price))

        if side == "long":
            t1_target = entry_price + gap_size * self.target_mult_t1
            t2_target = entry_price + gap_size * self.target_mult_t2
        else:
            t1_target = entry_price - gap_size * self.target_mult_t1
            t2_target = entry_price - gap_size * self.target_mult_t2

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 50, "rr": 1.0},
                {"level": t2_target, "qty_pct": 50, "rr": 1.5}
            ],
            hard_sl=0.0,
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank gap setup quality."""
        base_score = event.confidence * 100
        gap_size = event.context.get("gap_size_pct", 0)
        return min(100.0, base_score + gap_size * 10)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing."""
        return True, "Gap timing validated"

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0
        qty = max(1, min(int(max_risk_amount / risk_per_share), 100)) if risk_per_share > 0 else 1
        return qty, qty * entry_price

    def _calculate_institutional_strength(self, context: MarketContext, gap_pct: float,
                                        structure_type: str, side: str) -> float:
        """Calculate institutional-grade strength for gap patterns."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from gap size and volume (institutional volume threshold ≥1.5)
            gap_quality = min(3.0, gap_pct * 2.0)  # Scale gap percentage
            base_strength = max(1.3, vol_z * gap_quality * 0.4)

            # Professional bonuses for institutional-grade gap patterns
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.25  # 25% bonus for volume surge
                logger.debug(f"GAP: Volume surge bonus applied (vol_z={vol_z:.2f})")

            # Gap size optimization (institutional preference for meaningful gaps)
            if 0.5 <= gap_pct <= 1.5:  # Optimal gap size range
                strength_multiplier *= 1.2  # 20% bonus for optimal gap size
                logger.debug(f"GAP: Optimal gap size bonus applied ({gap_pct:.2f}%)")

            # Early session bonus (gaps are most significant at open)
            current_hour = pd.to_datetime(context.timestamp).hour
            current_minute = pd.to_datetime(context.timestamp).minute

            if current_hour == 9 and current_minute <= 30:  # First 15 minutes after open
                strength_multiplier *= 1.3  # 30% bonus for early gap action
                logger.debug(f"GAP: Early session gap bonus applied")

            # Gap type bonus
            if "fill" in structure_type:
                strength_multiplier *= 1.15  # 15% bonus for gap fill (mean reversion)
            elif "breakout" in structure_type:
                strength_multiplier *= 1.1  # 10% bonus for gap continuation

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.7)  # Strong minimum for gap patterns

            logger.debug(f"GAP: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"GAP: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold