"""
Momentum Structure Implementation

This module implements momentum-based trading structures including:
- Momentum breakout long/short (strong price momentum + volume)
- Trend continuation long/short (consistent directional pressure)
- Early market momentum patterns without specific level requirements
- Volume and momentum confirmation for all patterns

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


class MomentumStructure(BaseStructure):
    """Momentum-based structure detection and strategy planning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Momentum structure with configuration."""
        super().__init__(config)
        self.structure_type = "momentum"

        # KeyError if missing trading parameters

        # Momentum breakout parameters
        self.min_momentum_3bar_pct = config["min_momentum_3bar_pct"]  # 3-bar momentum threshold
        self.min_momentum_1bar_pct = config["min_momentum_1bar_pct"]  # Last bar momentum
        self.min_momentum_2bar_pct = config["min_momentum_2bar_pct"]  # 2-bar cumulative momentum

        # Trend continuation parameters
        self.min_trend_5bar_pct = config["min_trend_5bar_pct"]  # 5-bar trend momentum
        self.min_trend_3bar_pct = config["min_trend_3bar_pct"]  # 3-bar trend bias
        self.min_positive_bars = config["min_positive_bars"]

        # Volume parameters
        self.vol_z_breakout_mult = config["vol_z_breakout_mult"]
        self.vol_z_continuation_mult = config["vol_z_continuation_mult"]
        self.min_volume_surge_ratio = config["min_volume_surge_ratio"]

        # Risk management
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.stop_mult = config["stop_mult"]
        self.confidence_strong_momentum = config["confidence_strong_momentum"]
        self.confidence_weak_momentum = config["confidence_weak_momentum"]

        logger.debug(f"MOMENTUM: Initialized with 3-bar threshold: {self.min_momentum_3bar_pct}%, 5-bar trend: {self.min_trend_5bar_pct}%")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect momentum-based structures."""
        try:
            df = context.df_5m
            if len(df) < 10:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Insufficient data for momentum analysis"
                )

            # Calculate momentum indicators
            df_calc = self._calculate_momentum_indicators(df)
            if df_calc is None:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Could not calculate momentum indicators"
                )

            events = []

            # Get time-adjusted volume threshold
            vol_z_required = self._get_time_adjusted_vol_threshold(context.timestamp)

            # Detect momentum breakout patterns
            momentum_events = self._detect_momentum_breakouts(context, df_calc, vol_z_required)
            events.extend(momentum_events)

            # Detect trend continuation patterns
            trend_events = self._detect_trend_continuations(context, df_calc, vol_z_required)
            events.extend(trend_events)

            quality_score = self._calculate_quality_score(events, df_calc) if events else 0.0

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=quality_score,
                rejection_reason=None if events else "No momentum patterns detected"
            )

        except Exception as e:
            logger.error(f"MOMENTUM: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate momentum and volume indicators."""
        try:
            df_calc = df.copy()

            # Price momentum indicators
            df_calc['returns_1'] = df_calc['close'].pct_change()
            df_calc['returns_3'] = df_calc['close'].pct_change(3)  # 3-bar momentum
            df_calc['returns_5'] = df_calc['close'].pct_change(5)  # 5-bar momentum

            # Volume indicators
            df_calc['vol_ma'] = df_calc['volume'].rolling(10, min_periods=5).mean()
            df_calc['vol_surge'] = df_calc['volume'] / df_calc['vol_ma'].replace(0, np.nan)

            # Volume Z-score if not present
            if 'vol_z' not in df_calc.columns:
                df_calc['vol_z'] = self._calculate_vol_z(df_calc)

            return df_calc

        except Exception as e:
            logger.debug(f"MOMENTUM: Error calculating indicators: {e}")
            return None

    def _detect_momentum_breakouts(self, context: MarketContext, df: pd.DataFrame, vol_z_required: float) -> List[StructureEvent]:
        """Detect momentum breakout patterns."""
        events = []

        try:
            last_bar = df.iloc[-1]

            # Convert percentage thresholds to decimal
            momentum_3bar_threshold = self.min_momentum_3bar_pct / 100.0
            momentum_1bar_threshold = self.min_momentum_1bar_pct / 100.0
            momentum_2bar_threshold = self.min_momentum_2bar_pct / 100.0

            # Momentum Breakout Long
            if self._check_momentum_breakout_long(df, last_bar, momentum_3bar_threshold, momentum_1bar_threshold, momentum_2bar_threshold, vol_z_required):
                confidence = self._calculate_institutional_strength(context, abs(last_bar['returns_3']), last_bar.get('vol_z', 1.0), "long")

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type="momentum_breakout_long",
                    side="long",
                    confidence=confidence,
                    levels={"momentum_level": context.current_price},
                    context={
                        "momentum_3bar_pct": last_bar['returns_3'] * 100,
                        "momentum_1bar_pct": last_bar['returns_1'] * 100,
                        "vol_z": last_bar.get('vol_z', 1.0),
                        "vol_surge": last_bar.get('vol_surge', 1.0),
                        "pattern_type": "momentum_breakout"
                    },
                    price=context.current_price
                )
                events.append(event)
                logger.debug(f"MOMENTUM: {context.symbol} - Momentum breakout long: 3-bar {last_bar['returns_3']*100:.2f}%")

            # Momentum Breakout Short
            elif self._check_momentum_breakout_short(df, last_bar, momentum_3bar_threshold, momentum_1bar_threshold, momentum_2bar_threshold, vol_z_required):
                confidence = self._calculate_institutional_strength(context, abs(last_bar['returns_3']), last_bar.get('vol_z', 1.0), "short")

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type="momentum_breakout_short",
                    side="short",
                    confidence=confidence,
                    levels={"momentum_level": context.current_price},
                    context={
                        "momentum_3bar_pct": last_bar['returns_3'] * 100,
                        "momentum_1bar_pct": last_bar['returns_1'] * 100,
                        "vol_z": last_bar.get('vol_z', 1.0),
                        "vol_surge": last_bar.get('vol_surge', 1.0),
                        "pattern_type": "momentum_breakout"
                    },
                    price=context.current_price
                )
                events.append(event)
                logger.debug(f"MOMENTUM: {context.symbol} - Momentum breakout short: 3-bar {last_bar['returns_3']*100:.2f}%")

        except Exception as e:
            logger.debug(f"MOMENTUM: Error detecting momentum breakouts: {e}")

        return events

    def _detect_trend_continuations(self, context: MarketContext, df: pd.DataFrame, vol_z_required: float) -> List[StructureEvent]:
        """Detect trend continuation patterns."""
        events = []

        try:
            last_bar = df.iloc[-1]

            # Convert percentage thresholds to decimal
            trend_5bar_threshold = self.min_trend_5bar_pct / 100.0
            trend_3bar_threshold = self.min_trend_3bar_pct / 100.0

            # Trend Continuation Long
            if self._check_trend_continuation_long(df, last_bar, trend_5bar_threshold, trend_3bar_threshold, vol_z_required):
                confidence = self._calculate_institutional_strength(context, abs(last_bar['returns_5']), last_bar.get('vol_z', 1.0), "long")

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type="trend_continuation_long",
                    side="long",
                    confidence=confidence,
                    levels={"trend_level": context.current_price},
                    context={
                        "trend_5bar_pct": last_bar['returns_5'] * 100,
                        "trend_3bar_sum_pct": df['returns_1'].tail(3).sum() * 100,
                        "positive_bars": (df['returns_1'] > 0).tail(3).sum(),
                        "vol_z": last_bar.get('vol_z', 1.0),
                        "pattern_type": "trend_continuation"
                    },
                    price=context.current_price
                )
                events.append(event)
                logger.debug(f"MOMENTUM: {context.symbol} - Trend continuation long: 5-bar {last_bar['returns_5']*100:.2f}%")

            # Trend Continuation Short
            elif self._check_trend_continuation_short(df, last_bar, trend_5bar_threshold, trend_3bar_threshold, vol_z_required):
                confidence = self._calculate_institutional_strength(context, abs(last_bar['returns_5']), last_bar.get('vol_z', 1.0), "short")

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type="trend_continuation_short",
                    side="short",
                    confidence=confidence,
                    levels={"trend_level": context.current_price},
                    context={
                        "trend_5bar_pct": last_bar['returns_5'] * 100,
                        "trend_3bar_sum_pct": df['returns_1'].tail(3).sum() * 100,
                        "negative_bars": (df['returns_1'] < 0).tail(3).sum(),
                        "vol_z": last_bar.get('vol_z', 1.0),
                        "pattern_type": "trend_continuation"
                    },
                    price=context.current_price
                )
                events.append(event)
                logger.debug(f"MOMENTUM: {context.symbol} - Trend continuation short: 5-bar {last_bar['returns_5']*100:.2f}%")

        except Exception as e:
            logger.debug(f"MOMENTUM: Error detecting trend continuations: {e}")

        return events

    def _check_momentum_breakout_long(self, df: pd.DataFrame, last_bar: pd.Series, momentum_3bar_threshold: float,
                                     momentum_1bar_threshold: float, momentum_2bar_threshold: float, vol_z_required: float) -> bool:
        """Check conditions for momentum breakout long."""
        try:
            # 1. Strong 3-bar upward momentum
            if last_bar['returns_3'] <= momentum_3bar_threshold:
                return False

            # 2. Last bar positive
            if last_bar['returns_1'] <= momentum_1bar_threshold:
                return False

            # 3. 2-bar cumulative momentum
            if df['returns_1'].tail(2).sum() <= momentum_2bar_threshold:
                return False

            # 4. Volume confirmation (relaxed for momentum)
            vol_z = last_bar.get('vol_z', 1.0)
            if vol_z < vol_z_required * self.vol_z_breakout_mult:
                return False

            # 5. Volume surge
            vol_surge = last_bar.get('vol_surge', 1.0)
            if vol_surge < self.min_volume_surge_ratio:
                return False

            return True

        except Exception as e:
            logger.debug(f"MOMENTUM: Error checking momentum breakout long: {e}")
            return False

    def _check_momentum_breakout_short(self, df: pd.DataFrame, last_bar: pd.Series, momentum_3bar_threshold: float,
                                      momentum_1bar_threshold: float, momentum_2bar_threshold: float, vol_z_required: float) -> bool:
        """Check conditions for momentum breakout short."""
        try:
            # 1. Strong 3-bar downward momentum
            if last_bar['returns_3'] >= -momentum_3bar_threshold:
                return False

            # 2. Last bar negative
            if last_bar['returns_1'] >= -momentum_1bar_threshold:
                return False

            # 3. 2-bar cumulative momentum (negative)
            if df['returns_1'].tail(2).sum() >= -momentum_2bar_threshold:
                return False

            # 4. Volume confirmation (relaxed for momentum)
            vol_z = last_bar.get('vol_z', 1.0)
            if vol_z < vol_z_required * self.vol_z_breakout_mult:
                return False

            # 5. Volume surge
            vol_surge = last_bar.get('vol_surge', 1.0)
            if vol_surge < self.min_volume_surge_ratio:
                return False

            return True

        except Exception as e:
            logger.debug(f"MOMENTUM: Error checking momentum breakout short: {e}")
            return False

    def _check_trend_continuation_long(self, df: pd.DataFrame, last_bar: pd.Series, trend_5bar_threshold: float,
                                      trend_3bar_threshold: float, vol_z_required: float) -> bool:
        """Check conditions for trend continuation long."""
        try:
            # 1. Strong 5-bar upward trend
            if last_bar['returns_5'] <= trend_5bar_threshold:
                return False

            # 2. 3-bar upward bias
            if df['returns_1'].tail(3).sum() <= trend_3bar_threshold:
                return False

            # 3. At least 2 of last 3 bars positive
            positive_bars = (df['returns_1'] > 0).tail(3).sum()
            if positive_bars < self.min_positive_bars:
                return False

            # 4. Volume confirmation (more relaxed for continuation)
            vol_z = last_bar.get('vol_z', 1.0)
            if vol_z < vol_z_required * self.vol_z_continuation_mult:
                return False

            return True

        except Exception as e:
            logger.debug(f"MOMENTUM: Error checking trend continuation long: {e}")
            return False

    def _check_trend_continuation_short(self, df: pd.DataFrame, last_bar: pd.Series, trend_5bar_threshold: float,
                                       trend_3bar_threshold: float, vol_z_required: float) -> bool:
        """Check conditions for trend continuation short."""
        try:
            # 1. Strong 5-bar downward trend
            if last_bar['returns_5'] >= -trend_5bar_threshold:
                return False

            # 2. 3-bar downward bias
            if df['returns_1'].tail(3).sum() >= -trend_3bar_threshold:
                return False

            # 3. At least 2 of last 3 bars negative
            negative_bars = (df['returns_1'] < 0).tail(3).sum()
            if negative_bars < self.min_positive_bars:
                return False

            # 4. Volume confirmation (more relaxed for continuation)
            vol_z = last_bar.get('vol_z', 1.0)
            if vol_z < vol_z_required * self.vol_z_continuation_mult:
                return False

            return True

        except Exception as e:
            logger.debug(f"MOMENTUM: Error checking trend continuation short: {e}")
            return False

    def _calculate_institutional_strength(self, context: MarketContext, momentum_strength: float,
                                        vol_z: float, side: str) -> float:
        """Calculate institutional-grade strength for momentum patterns."""
        try:
            # Base strength from momentum and volume (institutional volume threshold ≥1.5)
            base_strength = max(1.5, vol_z * momentum_strength * 15.0)  # Scale momentum strength

            # Professional bonuses for institutional-grade momentum patterns
            strength_multiplier = 1.0

            # Exceptional momentum bonus
            if momentum_strength >= 0.03:  # 3%+ momentum (exceptional)
                strength_multiplier *= 1.4  # 40% bonus for exceptional momentum
                logger.debug(f"MOMENTUM: Exceptional momentum bonus applied ({momentum_strength*100:.1f}%)")
            elif momentum_strength >= 0.02:  # 2%+ momentum (strong)
                strength_multiplier *= 1.25  # 25% bonus for strong momentum
            elif momentum_strength >= 0.01:  # 1%+ momentum (moderate)
                strength_multiplier *= 1.15  # 15% bonus for moderate momentum

            # Volume surge bonus (institutional participation)
            if vol_z >= 2.5:  # Exceptional volume surge
                strength_multiplier *= 1.3  # 30% bonus for exceptional volume
                logger.debug(f"MOMENTUM: Exceptional volume bonus applied (vol_z={vol_z:.2f})")
            elif vol_z >= 1.5:  # Strong institutional volume
                strength_multiplier *= 1.2  # 20% bonus for strong volume

            # Market timing bonus (momentum works best during active hours)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"MOMENTUM: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.8)  # Strong minimum for momentum patterns

            logger.debug(f"MOMENTUM: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"MOMENTUM: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold

    # Removed deprecated _calculate_trend_confidence method - now using institutional strength

    def _get_time_adjusted_vol_threshold(self, timestamp: pd.Timestamp) -> float:
        """Get time-adjusted volume threshold."""
        base_vol_z = 1.5  # Default base threshold

        try:
            time_minutes = timestamp.hour * 60 + timestamp.minute

            # Time-based adjustments for momentum patterns
            if time_minutes < 630:  # Before 10:30am
                return base_vol_z * 0.5  # More lenient early market
            elif time_minutes < 720:  # 10:30am - 12:00pm
                return base_vol_z * 0.75  # Moderate adjustment
            else:  # After 12:00pm
                return base_vol_z  # Standard threshold
        except:
            return base_vol_z

    def _calculate_vol_z(self, df: pd.DataFrame, window: int = 30, min_periods: int = 10) -> pd.Series:
        """Calculate volume Z-score."""
        volume_mean = df['volume'].rolling(window, min_periods=min_periods).mean()
        volume_std = df['volume'].rolling(window, min_periods=min_periods).std(ddof=0)
        vol_z = (df['volume'] - volume_mean) / volume_std.replace(0, np.nan)
        return vol_z.fillna(0)

    def _calculate_quality_score(self, events: List[StructureEvent], df: pd.DataFrame) -> float:
        """Calculate quality score for momentum events."""
        if not events:
            return 0.0

        base_score = 65.0
        event = events[0]

        # Score based on momentum strength
        if "momentum_3bar_pct" in event.context:
            momentum_pct = abs(event.context["momentum_3bar_pct"])
            momentum_score = min(20.0, momentum_pct * 10)  # Up to 20 points for 2% momentum
        elif "trend_5bar_pct" in event.context:
            trend_pct = abs(event.context["trend_5bar_pct"])
            momentum_score = min(15.0, trend_pct * 6)  # Up to 15 points for trend
        else:
            momentum_score = 0.0

        # Volume score
        vol_z = event.context.get("vol_z", 1.0)
        volume_score = min(10.0, vol_z * 3)

        # Event count bonus
        event_score = len(events) * 5

        return min(100.0, base_score + momentum_score + volume_score + event_score)

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for momentum setups."""
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for momentum setups."""
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
        """Calculate risk parameters for momentum trades."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        # For momentum trades, use ATR-based stops
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
        """Calculate exit levels for momentum trades."""
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
        """Rank momentum setup quality."""
        base_score = event.confidence * 100

        # Bonus for strong momentum
        if "momentum_3bar_pct" in event.context:
            momentum_pct = abs(event.context["momentum_3bar_pct"])
            momentum_bonus = min(15.0, momentum_pct * 7)
        elif "trend_5bar_pct" in event.context:
            trend_pct = abs(event.context["trend_5bar_pct"])
            momentum_bonus = min(10.0, trend_pct * 4)
        else:
            momentum_bonus = 0.0

        # Volume bonus
        vol_z = event.context.get("vol_z", 1.0)
        volume_bonus = min(10.0, vol_z * 3)

        return min(100.0, base_score + momentum_bonus + volume_bonus)

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for momentum trades."""
        # Momentum patterns work well throughout the session, especially early market
        return True, "Momentum timing validated"

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