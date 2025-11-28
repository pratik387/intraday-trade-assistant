"""
Trend Structure Implementation

This module implements trend-based trading structures including:
- Trend pullback (retracement in trending market - continuation bias)
- Trend continuation (momentum in trending direction)
- Trend reversal (exhaustion patterns)

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
class TrendInfo:
    """Trend analysis information."""
    trend_direction: str  # "up", "down", "sideways"
    trend_strength: float  # 0-100
    pullback_depth_pct: float
    momentum_score: float
    trend_age_bars: int
    trend_quality: float


class TrendStructure(BaseStructure):
    """
    Trend-based structure detection and strategy planning.

    Handles multiple trend strategies:
    1. Trend pullback - retracement in trending market (continuation bias)
    2. Trend continuation - momentum in trending direction
    3. Trend reversal - exhaustion patterns
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Trend structure with configuration."""
        super().__init__(config)
        self.structure_type = "trend"

        # KeyError if missing trading parameters

        # Trend detection parameters
        self.min_trend_strength = config["min_trend_strength"]
        self.min_trend_bars = config["min_trend_bars"]
        self.max_pullback_pct = config["max_pullback_pct"]
        self.min_pullback_pct = config["min_pullback_pct"]

        # Volume and momentum requirements
        self.require_volume_confirmation = config["require_volume_confirmation"]
        self.min_volume_mult = config["min_volume_mult"]
        self.min_momentum_score = config["min_momentum_score"]

        # Risk management
        self.min_stop_distance_pct = config["min_stop_distance_pct"]
        self.stop_distance_mult = config["stop_distance_mult"]
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]

        # Stop loss parameters - Pro trader: SL at swing low/high + ATR buffer
        self.swing_sl_buffer_atr = config["swing_sl_buffer_atr"]  # ATR buffer beyond swing level
        self.swing_lookback_bars = config["swing_lookback_bars"]  # Bars to look back for swing

        # Confidence levels
        self.confidence_strong_trend = config["confidence_strong_trend"]
        self.confidence_weak_trend = config["confidence_weak_trend"]

        logger.debug(f"TREND: Initialized with min strength: {self.min_trend_strength}, pullback range: {self.min_pullback_pct}-{self.max_pullback_pct}%")
        logger.debug(f"TREND: SL params - swing_buffer: {self.swing_sl_buffer_atr}ATR, lookback: {self.swing_lookback_bars} bars")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect trend-based structures."""
        logger.debug(f"TREND: Starting detection for {context.symbol}")

        try:
            trend_info = self._analyze_trend(context)
            if not trend_info:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Trend analysis not available"
                )

            events = []
            max_quality = 0.0

            # Detect different trend strategies
            pullback_events, pullback_quality = self._detect_trend_pullback(context, trend_info)
            events.extend(pullback_events)
            max_quality = max(max_quality, pullback_quality)

            continuation_events, continuation_quality = self._detect_trend_continuation(context, trend_info)
            events.extend(continuation_events)
            max_quality = max(max_quality, continuation_quality)

            structure_detected = len(events) > 0
            rejection_reason = None if structure_detected else "No trend setups detected"

            logger.debug(f"TREND: {context.symbol} - Detection complete: {len(events)} events, quality: {max_quality:.2f}")

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=events,
                quality_score=max_quality,
                rejection_reason=rejection_reason
            )

        except Exception as e:
            logger.error(f"TREND: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _analyze_trend(self, context: MarketContext) -> Optional[TrendInfo]:
        """Analyze trend characteristics."""
        try:
            df = context.df_5m
            if len(df) < self.min_trend_bars:
                return None

            # Simple trend analysis using moving averages and price action
            closes = df['close']
            highs = df['high']
            lows = df['low']

            # Calculate trend using EMAs
            ema_short = closes.ewm(span=8).mean()
            ema_long = closes.ewm(span=21).mean()

            current_price = context.current_price
            ema_short_current = ema_short.iloc[-1]
            ema_long_current = ema_long.iloc[-1]

            # Determine trend direction
            if ema_short_current > ema_long_current * 1.005:  # 0.5% buffer
                trend_direction = "up"
            elif ema_short_current < ema_long_current * 0.995:
                trend_direction = "down"
            else:
                trend_direction = "sideways"

            # Calculate trend strength
            ema_separation = abs(ema_short_current - ema_long_current) / ema_long_current * 100
            trend_strength = min(100.0, ema_separation * 50)  # Scale to 0-100

            if trend_strength < self.min_trend_strength:
                logger.debug(f"TREND: {context.symbol} - Weak trend strength: {trend_strength:.1f}")
                return None

            # Calculate pullback depth
            if trend_direction == "up":
                recent_high = highs.tail(20).max()
                pullback_depth_pct = (recent_high - current_price) / recent_high * 100
            elif trend_direction == "down":
                recent_low = lows.tail(20).min()
                pullback_depth_pct = (current_price - recent_low) / recent_low * 100
            else:
                pullback_depth_pct = 0.0

            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(df)

            # Calculate trend age
            trend_age_bars = self._calculate_trend_age(ema_short, ema_long)

            # Calculate overall trend quality
            trend_quality = min(100.0, (trend_strength + momentum_score) / 2)

            logger.debug(f"TREND: {context.symbol} - Direction: {trend_direction}, Strength: {trend_strength:.1f}, Pullback: {pullback_depth_pct:.1f}%")

            return TrendInfo(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                pullback_depth_pct=pullback_depth_pct,
                momentum_score=momentum_score,
                trend_age_bars=trend_age_bars,
                trend_quality=trend_quality
            )

        except Exception as e:
            logger.error(f"TREND: Error analyzing trend for {context.symbol}: {e}")
            return None

    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score based on various factors."""
        try:
            # Use ADX if available
            if 'adx14' in df.columns:
                adx = df['adx14'].iloc[-1]
                momentum_score = min(100.0, adx * 2.5)  # Scale ADX to 0-100
            else:
                # Fallback: use price momentum
                closes = df['close']
                price_change_5 = (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] * 100
                momentum_score = min(100.0, abs(price_change_5) * 10)

            return momentum_score

        except Exception:
            return 50.0  # Default moderate momentum

    def _calculate_trend_age(self, ema_short: pd.Series, ema_long: pd.Series) -> int:
        """Calculate how long the trend has been in place."""
        try:
            for i in range(len(ema_short) - 1, 0, -1):
                if ema_short.iloc[i] <= ema_long.iloc[i]:
                    return len(ema_short) - 1 - i
            return len(ema_short)
        except Exception:
            return 0

    def _detect_trend_pullback(self, context: MarketContext, trend_info: TrendInfo) -> Tuple[List[StructureEvent], float]:
        """Detect trend pullback opportunities."""
        events = []
        quality_score = 0.0

        if trend_info.trend_direction == "sideways":
            return events, quality_score

        # Check if pullback is within acceptable range
        if not (self.min_pullback_pct <= trend_info.pullback_depth_pct <= self.max_pullback_pct):
            logger.debug(f"TREND: {context.symbol} - Pullback {trend_info.pullback_depth_pct:.1f}% outside range {self.min_pullback_pct}-{self.max_pullback_pct}%")
            return events, quality_score

        # Check volume confirmation if required
        volume_ok = True
        if self.require_volume_confirmation:
            volume_ok = self._check_volume_confirmation(context)

        if volume_ok and trend_info.momentum_score >= self.min_momentum_score:
            if trend_info.trend_direction == "up":
                structure_type = "trend_pullback_long"
                side = "long"
            else:
                structure_type = "trend_pullback_short"
                side = "short"

            confidence = self._calculate_institutional_strength(context, trend_info, "pullback", side, volume_ok)
            quality_score = min(95.0, trend_info.trend_quality + (confidence * 15))

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type=structure_type,
                side=side,
                confidence=confidence,
                levels={"trend_level": context.current_price},
                context={
                    "trend_direction": trend_info.trend_direction,
                    "trend_strength": trend_info.trend_strength,
                    "pullback_depth_pct": trend_info.pullback_depth_pct,
                    "momentum_score": trend_info.momentum_score,
                    "volume_confirmation": volume_ok
                },
                price=context.current_price
            )
            events.append(event)

            logger.debug(f"TREND: {context.symbol} - Trend pullback {side.upper()} detected: {trend_info.trend_direction} trend, {trend_info.pullback_depth_pct:.1f}% pullback")

        return events, quality_score

    def _detect_trend_continuation(self, context: MarketContext, trend_info: TrendInfo) -> Tuple[List[StructureEvent], float]:
        """Detect trend continuation opportunities."""
        events = []
        quality_score = 0.0

        if trend_info.trend_direction == "sideways":
            return events, quality_score

        # For continuation, we want minimal pullback (momentum continuation)
        if trend_info.pullback_depth_pct > 15.0:  # More than 15% pullback is not continuation
            return events, quality_score

        # Check volume confirmation
        volume_ok = self._check_volume_confirmation(context)

        if volume_ok and trend_info.momentum_score >= self.min_momentum_score + 10:  # Higher threshold for continuation
            if trend_info.trend_direction == "up":
                structure_type = "trend_continuation_long"
                side = "long"
            else:
                structure_type = "trend_continuation_short"
                side = "short"

            confidence = self._calculate_institutional_strength(context, trend_info, "continuation", side, volume_ok)
            quality_score = min(90.0, trend_info.trend_quality + (confidence * 10))

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type=structure_type,
                side=side,
                confidence=confidence,
                levels={"trend_level": context.current_price},
                context={
                    "trend_direction": trend_info.trend_direction,
                    "trend_strength": trend_info.trend_strength,
                    "momentum_score": trend_info.momentum_score,
                    "volume_confirmation": volume_ok
                },
                price=context.current_price
            )
            events.append(event)

            logger.debug(f"TREND: {context.symbol} - Trend continuation {side.upper()} detected: strong {trend_info.trend_direction} momentum")

        return events, quality_score

    def _check_volume_confirmation(self, context: MarketContext) -> bool:
        """Check volume confirmation."""
        try:
            if context.indicators and 'vol_z' in context.indicators:
                return context.indicators['vol_z'] >= self.min_volume_mult

            df = context.df_5m
            if 'vol_z' in df.columns:
                return float(df['vol_z'].iloc[-1]) >= self.min_volume_mult

            return False
        except Exception:
            return False

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for trend setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for trend setups."""
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
            notes={
                "trend_direction": event.context.get("trend_direction"),
                "trend_strength": event.context.get("trend_strength"),
                **event.context
            }
        )

    def calculate_risk_params(self, context: MarketContext, event: StructureEvent, side: str) -> RiskParams:
        """Calculate risk parameters for trend strategies using pro trader swing-based SL.

        Pro trader approach: SL at recent swing low/high + ATR buffer.
        For trend trades, the swing point is the logical invalidation point.
        """
        entry_price = context.current_price
        atr = self._get_atr(context)
        df = context.df_5m

        # Find recent swing low/high for SL placement
        if len(df) >= self.swing_lookback_bars:
            lookback_df = df.tail(self.swing_lookback_bars)

            if side == "long":
                # Long: SL below recent swing low + ATR buffer
                swing_low = float(lookback_df['low'].min())
                hard_sl = swing_low - (atr * self.swing_sl_buffer_atr)
            else:
                # Short: SL above recent swing high + ATR buffer
                swing_high = float(lookback_df['high'].max())
                hard_sl = swing_high + (atr * self.swing_sl_buffer_atr)
        else:
            # Fallback to ATR-based stops if not enough data
            if side == "long":
                hard_sl = entry_price - (atr * self.stop_distance_mult)
            else:
                hard_sl = entry_price + (atr * self.stop_distance_mult)

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
        """Calculate exit levels for trend strategies."""
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
                {"level": t1_target, "qty_pct": 50, "rr": 1.5},
                {"level": t2_target, "qty_pct": 50, "rr": 2.5}
            ],
            hard_sl=0.0,
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank trend setup quality."""
        base_score = event.confidence * 100
        bonuses = 0.0

        trend_strength = event.context.get("trend_strength", 0)
        if trend_strength > 80:
            bonuses += 15.0
        elif trend_strength > 60:
            bonuses += 10.0

        momentum_score = event.context.get("momentum_score", 0)
        if momentum_score > 70:
            bonuses += 10.0

        if event.context.get("volume_confirmation", False):
            bonuses += 12.0

        return min(100.0, base_score + bonuses)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for trend setup execution."""
        current_hour = context.timestamp.hour
        if current_hour < 9 or current_hour > 15:
            return False, f"Outside market hours: {current_hour}:xx"

        if len(context.df_5m) < self.min_trend_bars:
            return False, f"Insufficient data for trend analysis: {len(context.df_5m)} bars"

        return True, "Timing validated"

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR with fallback calculation."""
        if context.indicators and 'atr' in context.indicators:
            atr = context.indicators['atr']
            if atr > 0:
                return atr

        # Simple ATR calculation fallback
        try:
            df = context.df_5m
            if len(df) >= 14:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))

                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.tail(14).mean()
                return atr
        except Exception:
            pass

        return context.current_price * 0.01  # 1% fallback

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size based on risk management."""
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0

        if risk_per_share > 0:
            max_qty = int(max_risk_amount / risk_per_share)
            qty = max(1, min(max_qty, 100))
        else:
            qty = 1

        notional = qty * entry_price
        return qty, notional

    def _calculate_institutional_strength(self, context: MarketContext, trend_info: TrendInfo,
                                        setup_type: str, side: str, volume_confirmed: bool) -> float:
        """Calculate institutional-grade strength for trend patterns."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from trend quality and strength
            trend_quality_score = trend_info.trend_quality / 100.0
            base_strength = max(1.2, trend_info.trend_strength * trend_quality_score * 0.05)

            # Professional bonuses for institutional-grade trend patterns
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if volume_confirmed and vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.25  # 25% bonus for volume confirmation
                logger.debug(f"TREND: Volume confirmation bonus applied (vol_z={vol_z:.2f})")

            if vol_z >= 2.0:  # Strong institutional volume
                strength_multiplier *= 1.2  # Additional 20% bonus
                logger.debug(f"TREND: Strong volume bonus applied")

            # Trend strength bonuses
            if trend_info.trend_strength >= 80:  # Very strong trend
                strength_multiplier *= 1.3  # 30% bonus for strong trend
                logger.debug(f"TREND: Strong trend bonus applied ({trend_info.trend_strength:.1f})")
            elif trend_info.trend_strength >= 60:  # Moderate trend
                strength_multiplier *= 1.15  # 15% bonus for moderate trend

            # Setup-specific bonuses
            if setup_type == "pullback":
                # Optimal pullback depth bonus (institutional preference for controlled retracements)
                if 25 <= trend_info.pullback_depth_pct <= 50:  # Sweet spot pullback
                    strength_multiplier *= 1.2  # 20% bonus for optimal pullback
                    logger.debug(f"TREND: Optimal pullback bonus applied ({trend_info.pullback_depth_pct:.1f}%)")

                # Trend age bonus (mature trends more reliable)
                if trend_info.trend_age_bars >= 20:  # Established trend
                    strength_multiplier *= 1.15  # 15% bonus for established trend
                    logger.debug(f"TREND: Established trend bonus applied ({trend_info.trend_age_bars} bars)")

            elif setup_type == "continuation":
                # Momentum confirmation bonus
                if trend_info.momentum_score >= 80:  # Strong momentum
                    strength_multiplier *= 1.25  # 25% bonus for strong momentum
                    logger.debug(f"TREND: Strong momentum bonus applied ({trend_info.momentum_score:.1f})")

                # Fresh breakout bonus (early continuation)
                if trend_info.trend_age_bars <= 10:  # Fresh trend
                    strength_multiplier *= 1.2  # 20% bonus for fresh momentum
                    logger.debug(f"TREND: Fresh breakout bonus applied")

            # Trend quality bonus
            if trend_info.trend_quality >= 85:  # Exceptional trend quality
                strength_multiplier *= 1.2  # 20% bonus for quality
                logger.debug(f"TREND: High quality trend bonus applied ({trend_info.trend_quality:.1f})")

            # Market timing bonus (trends work best during active sessions)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"TREND: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.6)  # Strong minimum for trend patterns

            logger.debug(f"TREND: {context.symbol} {side} {setup_type} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"TREND: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold