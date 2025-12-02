"""
Squeeze Release Structure Implementation

This module implements squeeze release (volatility expansion) structures including:
- Squeeze release long (expansion with upward momentum)
- Squeeze release short (expansion with downward momentum)
- Volatility regime detection using width (std/mean) analysis
- Momentum confirmation for directional bias

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


class SqueezeReleaseStructure(BaseStructure):
    """Squeeze release (volatility expansion) structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Squeeze Release structure with configuration."""
        super().__init__(config)
        self.structure_type = "squeeze_release"

        # KeyError if missing trading parameters

        # Squeeze detection parameters
        self.width_window = config["width_window"]
        self.expansion_ratio = config["expansion_ratio"]
        self.width_calculation_period = config["width_calculation_period"]
        self.recent_width_bars = config["recent_width_bars"]

        # Momentum parameters
        self.momentum_period = config["momentum_period"]
        self.min_momentum_threshold = config["min_momentum_threshold"]

        # Volume confirmation
        self.require_volume_confirmation = config["require_volume_confirmation"]
        self.min_volume_mult = config["min_volume_mult"]

        # Risk management
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.stop_mult = config["stop_mult"]
        self.confidence_strong_expansion = config["confidence_strong_expansion"]
        self.confidence_weak_expansion = config["confidence_weak_expansion"]

        logger.debug(f"SQUEEZE_RELEASE: Initialized with width_window: {self.width_window}, expansion_ratio: {self.expansion_ratio}")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect squeeze release structures."""
        try:
            logger.debug(f"SQUEEZE_DETECT: Starting detection for {context.symbol}")
            df = context.df_5m
            min_required_bars = self.width_window + self.recent_width_bars + self.width_calculation_period
            logger.debug(f"SQUEEZE_DETECT: {context.symbol} - Checking bars: have {len(df)}, need {min_required_bars} (width_window={self.width_window})")

            if len(df) < min_required_bars:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data for squeeze analysis (need {min_required_bars} bars, have {len(df)})"
                )

            events = []

            # Calculate volatility width (std/mean)
            width_series = self._calculate_volatility_width(df)
            if width_series is None:
                logger.debug(f"SQUEEZE_DETECT: {context.symbol} - REJECTED: Could not calculate volatility width")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Could not calculate volatility width"
                )

            # Check for squeeze release pattern
            squeeze_release_info = self._detect_squeeze_release_pattern(df, width_series, context.symbol)
            if not squeeze_release_info:
                logger.debug(f"SQUEEZE_DETECT: {context.symbol} - REJECTED: No squeeze release pattern detected")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="No squeeze release pattern detected"
                )

            expansion_ratio_actual, momentum, momentum_direction = squeeze_release_info
            logger.debug(f"SQUEEZE_DETECT: {context.symbol} - Pattern found: expansion_ratio={expansion_ratio_actual:.2f}, momentum={momentum:.4f}, direction={momentum_direction}")

            # Validate momentum threshold
            if abs(momentum) < self.min_momentum_threshold:
                logger.debug(f"SQUEEZE_DETECT: {context.symbol} - REJECTED: Momentum {abs(momentum):.4f} below threshold {self.min_momentum_threshold}")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Momentum {momentum:.4f} below threshold {self.min_momentum_threshold}"
                )

            # Volume confirmation if required
            if self.require_volume_confirmation:
                if not self._validate_volume_confirmation(df):
                    logger.debug(f"SQUEEZE_DETECT: {context.symbol} - REJECTED: Volume confirmation failed (require_volume_confirmation=true, min_volume_mult={self.min_volume_mult})")
                    return StructureAnalysis(
                        structure_detected=False,
                        events=[],
                        quality_score=0.0,
                        rejection_reason="Volume confirmation failed"
                    )

            # Create squeeze release event
            structure_type = "squeeze_release_long" if momentum_direction == "long" else "squeeze_release_short"
            side = momentum_direction

            # Get volume information for strength calculation
            vol_z = df.get('vol_z', pd.Series([1.0])).iloc[-1] if 'vol_z' in df.columns else 1.0

            # Calculate confidence based on expansion strength
            confidence = self._calculate_institutional_strength(context, expansion_ratio_actual, momentum, vol_z, momentum_direction)

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type=structure_type,
                side=side,
                confidence=confidence,
                levels={"squeeze_level": context.current_price},
                context={
                    "expansion_ratio": expansion_ratio_actual,
                    "momentum": momentum,
                    "momentum_period": self.momentum_period,
                    "width_expansion": "volatility_breakout"
                },
                price=context.current_price
            )

            events.append(event)

            logger.debug(f"SQUEEZE_DETECT: ✅ {context.symbol} - {structure_type} DETECTED with expansion ratio: {expansion_ratio_actual:.2f}, momentum: {momentum:.4f}, confidence: {confidence:.2f}")

            quality_score = self._calculate_quality_score(expansion_ratio_actual, abs(momentum), df)

            return StructureAnalysis(
                structure_detected=True,
                events=events,
                quality_score=quality_score,
                rejection_reason=None
            )

        except Exception as e:
            logger.error(f"SQUEEZE_RELEASE: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _calculate_volatility_width(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate volatility width (std/mean) series."""
        try:
            # Calculate rolling standard deviation and mean
            period = self.width_calculation_period
            std_series = df['close'].rolling(period, min_periods=period).std(ddof=0)
            mean_series = df['close'].rolling(period, min_periods=period).mean()

            # Calculate width (std/mean) - avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                width_series = std_series / mean_series.replace(0, np.nan)

            # Replace infinite values with NaN
            width_series = width_series.replace([np.inf, -np.inf], np.nan)

            return width_series

        except Exception as e:
            logger.debug(f"SQUEEZE_RELEASE: Error calculating volatility width: {e}")
            return None

    def _detect_squeeze_release_pattern(self, df: pd.DataFrame, width_series: pd.Series, symbol: str = "UNKNOWN") -> Optional[Tuple[float, float, str]]:
        """Detect squeeze release pattern from width series."""
        try:
            # Calculate recent width (average of last N bars)
            recent_width = width_series.iloc[-self.recent_width_bars:].mean()

            # Calculate prior width (average from width_window ago to recent_width_bars ago)
            start_idx = -(self.width_window + self.recent_width_bars)
            end_idx = -self.recent_width_bars
            prior_width = width_series.iloc[start_idx:end_idx].mean()

            # Check for NaN values
            if pd.isna(recent_width) or pd.isna(prior_width) or prior_width == 0:
                return None

            # Calculate expansion ratio
            expansion_ratio_actual = recent_width / prior_width

            # Check if expansion exceeds threshold
            if expansion_ratio_actual < self.expansion_ratio:
                logger.debug(f"SQUEEZE_DETECT: {symbol} - Expansion ratio {expansion_ratio_actual:.2f} below threshold {self.expansion_ratio} (recent_width={recent_width:.6f}, prior_width={prior_width:.6f})")
                return None

            # Calculate momentum for direction
            momentum_change = df['close'].pct_change(self.momentum_period).iloc[-1]
            momentum_direction = "long" if momentum_change > 0 else "short"

            logger.debug(f"SQUEEZE_RELEASE: Expansion ratio: {expansion_ratio_actual:.2f}, momentum: {momentum_change:.4f}, direction: {momentum_direction}")

            return expansion_ratio_actual, momentum_change, momentum_direction

        except Exception as e:
            logger.debug(f"SQUEEZE_RELEASE: Error detecting squeeze pattern: {e}")
            return None

    def _validate_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Validate volume confirmation for squeeze release."""
        if not self.require_volume_confirmation:
            return True

        try:
            # Check if volume on expansion is elevated
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
            logger.debug(f"SQUEEZE_RELEASE: Volume validation error: {e}")

        return True  # Default to true if can't validate

    def _calculate_quality_score(self, expansion_ratio: float, momentum: float, df: pd.DataFrame) -> float:
        """Calculate quality score for squeeze release."""
        base_score = 60.0

        # Score based on expansion strength
        expansion_score = min(25.0, (expansion_ratio - 1.0) * 50)  # Up to 25 points

        # Score based on momentum strength
        momentum_score = min(10.0, momentum * 1000)  # Up to 10 points for 1% momentum

        # Volume bonus if available
        volume_bonus = 0.0
        try:
            if 'vol_z' in df.columns:
                vol_z = df['vol_z'].iloc[-1]
                volume_bonus = min(5.0, vol_z)
        except:
            pass

        return min(100.0, base_score + expansion_score + momentum_score + volume_bonus)

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for squeeze release setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for squeeze release setups."""
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
        """Calculate risk parameters for squeeze release trades."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        # For squeeze releases, use ATR-based stops
        if side == "long":
            hard_sl = entry_price - (atr * self.stop_mult)
        else:
            hard_sl = entry_price + (atr * self.stop_mult)

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels for squeeze release trades."""
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
                {"level": t1_target, "qty_pct": 50, "rr": self.target_mult_t1},
                {"level": t2_target, "qty_pct": 50, "rr": self.target_mult_t2}
            ],
            hard_sl=0.0,  # Set in risk_params
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank squeeze release setup quality."""
        base_score = event.confidence * 100
        expansion_ratio = event.context.get("expansion_ratio", 1.0)
        momentum = abs(event.context.get("momentum", 0.0))

        # Bonus for strong expansion
        expansion_bonus = min(15.0, (expansion_ratio - 1.0) * 30)

        # Bonus for strong momentum
        momentum_bonus = min(10.0, momentum * 500)

        return min(100.0, base_score + expansion_bonus + momentum_bonus)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for squeeze release trades."""
        # Squeeze releases can occur throughout the session
        return True, "Squeeze release timing validated"

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

    def _calculate_institutional_strength(self, context: MarketContext, expansion_ratio: float,
                                        momentum: float, vol_z: float, direction: str) -> float:
        """Calculate institutional-grade strength for squeeze release patterns."""
        try:
            # Base strength from expansion ratio and momentum
            expansion_quality = min(3.0, expansion_ratio)
            momentum_quality = abs(momentum) * 50.0  # Scale momentum
            base_strength = max(1.6, vol_z * expansion_quality * momentum_quality * 0.4)

            # Professional bonuses for institutional-grade squeeze patterns
            strength_multiplier = 1.0

            # Exceptional expansion bonus
            if expansion_ratio >= 2.0:  # 2x or more expansion
                strength_multiplier *= 1.35  # 35% bonus for exceptional expansion
                logger.debug(f"SQUEEZE: Exceptional expansion bonus applied ({expansion_ratio:.2f}x)")
            elif expansion_ratio >= 1.5:  # 1.5x expansion
                strength_multiplier *= 1.2  # 20% bonus for strong expansion

            # Strong momentum bonus
            if abs(momentum) >= 0.02:  # 2%+ momentum
                strength_multiplier *= 1.25  # 25% bonus for strong momentum
                logger.debug(f"SQUEEZE: Strong momentum bonus applied ({abs(momentum)*100:.1f}%)")
            elif abs(momentum) >= 0.01:  # 1%+ momentum
                strength_multiplier *= 1.15  # 15% bonus for moderate momentum

            # Volume surge bonus (institutional participation)
            if vol_z >= 2.0:  # Strong institutional volume
                strength_multiplier *= 1.25  # 25% bonus for volume surge
                logger.debug(f"SQUEEZE: Volume surge bonus applied (vol_z={vol_z:.2f})")
            elif vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.15  # 15% bonus for institutional volume

            # Market timing bonus (squeezes can occur throughout the session)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"SQUEEZE: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.7)  # Strong minimum for squeeze patterns

            logger.debug(f"SQUEEZE: {context.symbol} {direction} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"SQUEEZE: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold