"""
VWAP Structure Implementation

This module implements VWAP-based trading structures including:
- VWAP mean reversion (when price stretched from VWAP)
- VWAP reclaim (when price reclaims VWAP after being below)
- VWAP lose (when price loses VWAP support)

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
class VWAPLevels:
    """VWAP-specific level information."""
    current_vwap: float
    price_distance_bps: float
    price_distance_atr: float
    above_vwap_bars: int
    below_vwap_bars: int
    volume_confirmation: bool


class VWAPStructure(BaseStructure):
    """
    VWAP-based structure detection and strategy planning.

    Handles multiple VWAP strategies:
    1. Mean reversion when price is stretched from VWAP
    2. VWAP reclaim after being below (long bias)
    3. VWAP lose when losing support (short bias)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VWAP structure with configuration.

        CRITICAL: All trading parameters must be provided in config.
        No hardcoded defaults for trading decisions.
        """
        super().__init__(config)
        self.structure_type = "vwap"

        # KeyError if missing trading parameters

        # Mean reversion parameters
        self.min_distance_bps = config["min_distance_bps"]
        self.max_distance_bps = config["max_distance_bps"]
        self.require_oversold_rsi = config["require_oversold_rsi"]
        self.oversold_rsi_threshold = config["oversold_rsi_threshold"]
        self.overbought_rsi_threshold = config["overbought_rsi_threshold"]

        # VWAP reclaim parameters
        self.min_bars_above_vwap = config["min_bars_above_vwap"]
        self.reclaim_volume_confirmation = config["reclaim_volume_confirmation"]
        self.min_volume_mult = config["min_volume_mult"]

        # Risk management parameters
        self.min_stop_distance_pct = config["min_stop_distance_pct"]
        self.stop_distance_mult = config["stop_distance_mult"]
        self.vwap_buffer_mult = config["vwap_buffer_mult"]

        # Target parameters
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]

        # Confidence scoring
        self.confidence_strong_signal = config["confidence_strong_signal"]
        self.confidence_weak_signal = config["confidence_weak_signal"]

        logger.info(f"VWAP: Initialized with mean reversion distance: {self.min_distance_bps}-{self.max_distance_bps} bps")
        logger.info(f"VWAP: Reclaim requirements - bars above: {self.min_bars_above_vwap}, volume conf: {self.reclaim_volume_confirmation}")


    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect VWAP-based structures in the market context."""

        logger.debug(f"VWAP: Starting detection for {context.symbol}")

        try:
            # Extract VWAP levels and context
            vwap_info = self._extract_vwap_levels(context)
            if not vwap_info:
                logger.debug(f"VWAP: {context.symbol} - Cannot extract VWAP levels")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="VWAP data not available"
                )

            # Detect different VWAP strategies
            events = []
            max_quality = 0.0

            # 1. VWAP Mean Reversion
            mean_rev_events, mean_rev_quality = self._detect_mean_reversion(context, vwap_info)
            events.extend(mean_rev_events)
            max_quality = max(max_quality, mean_rev_quality)

            # 2. VWAP Reclaim
            reclaim_events, reclaim_quality = self._detect_vwap_reclaim(context, vwap_info)
            events.extend(reclaim_events)
            max_quality = max(max_quality, reclaim_quality)

            # 3. VWAP Lose (short bias)
            lose_events, lose_quality = self._detect_vwap_lose(context, vwap_info)
            events.extend(lose_events)
            max_quality = max(max_quality, lose_quality)

            structure_detected = len(events) > 0
            rejection_reason = None if structure_detected else "No VWAP setups detected"

            logger.debug(f"VWAP: {context.symbol} - Detection complete: {len(events)} events, quality: {max_quality:.2f}")

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=events,
                quality_score=max_quality,
                rejection_reason=rejection_reason
            )

        except Exception as e:
            logger.error(f"VWAP: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _extract_vwap_levels(self, context: MarketContext) -> Optional[VWAPLevels]:
        """Extract VWAP-related information from market context."""

        try:
            df = context.df_5m
            if len(df) < 10 or 'vwap' not in df.columns:
                logger.debug(f"VWAP: {context.symbol} - Insufficient data or missing VWAP column")
                return None

            current_vwap = float(df['vwap'].iloc[-1])
            current_price = float(context.current_price)

            # Calculate distance from VWAP
            price_distance_bps = abs(current_price - current_vwap) / current_vwap * 10000

            # Calculate distance in ATR terms if available
            price_distance_atr = 0.0
            if 'atr' in df.columns and context.indicators and 'atr' in context.indicators:
                atr = context.indicators['atr']
                if atr > 0:
                    price_distance_atr = abs(current_price - current_vwap) / atr

            # Count consecutive bars above/below VWAP
            above_vwap_bars = self._count_consecutive_above_vwap(df)
            below_vwap_bars = self._count_consecutive_below_vwap(df)

            # Check volume confirmation
            volume_confirmation = self._check_volume_confirmation(df)

            logger.debug(f"VWAP: {context.symbol} - Price: {current_price:.2f}, VWAP: {current_vwap:.2f}, Distance: {price_distance_bps:.1f} bps")
            logger.debug(f"VWAP: {context.symbol} - Above bars: {above_vwap_bars}, Below bars: {below_vwap_bars}, Vol conf: {volume_confirmation}")

            return VWAPLevels(
                current_vwap=current_vwap,
                price_distance_bps=price_distance_bps,
                price_distance_atr=price_distance_atr,
                above_vwap_bars=above_vwap_bars,
                below_vwap_bars=below_vwap_bars,
                volume_confirmation=volume_confirmation
            )

        except Exception as e:
            logger.error(f"VWAP: Error extracting VWAP levels for {context.symbol}: {e}")
            return None

    def _count_consecutive_above_vwap(self, df: pd.DataFrame) -> int:
        """Count consecutive bars where close > VWAP."""
        try:
            close_prices = df['close'].values
            vwap_values = df['vwap'].values

            count = 0
            for i in range(len(close_prices) - 1, -1, -1):
                if close_prices[i] > vwap_values[i]:
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 0

    def _count_consecutive_below_vwap(self, df: pd.DataFrame) -> int:
        """Count consecutive bars where close < VWAP."""
        try:
            close_prices = df['close'].values
            vwap_values = df['vwap'].values

            count = 0
            for i in range(len(close_prices) - 1, -1, -1):
                if close_prices[i] < vwap_values[i]:
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 0

    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if current volume supports the setup."""
        try:
            if 'vol_z' in df.columns:
                current_vol_z = float(df['vol_z'].iloc[-1])
                return current_vol_z >= self.min_volume_mult
            return False
        except Exception:
            return False

    def _detect_mean_reversion(self, context: MarketContext, vwap_info: VWAPLevels) -> Tuple[List[StructureEvent], float]:
        """Detect VWAP mean reversion opportunities."""

        events = []
        quality_score = 0.0

        # Check if price is stretched enough from VWAP
        if not (self.min_distance_bps <= vwap_info.price_distance_bps <= self.max_distance_bps):
            logger.debug(f"VWAP: {context.symbol} - Price distance {vwap_info.price_distance_bps:.1f} bps outside range {self.min_distance_bps}-{self.max_distance_bps}")
            return events, quality_score

        current_price = context.current_price

        # Mean reversion long (price below VWAP)
        if current_price < vwap_info.current_vwap:
            logger.debug(f"VWAP: {context.symbol} - Checking mean reversion long setup")

            # Check RSI if required
            rsi_ok = True
            if self.require_oversold_rsi:
                if context.indicators and 'rsi14' in context.indicators:
                    rsi = context.indicators['rsi14']
                    rsi_ok = rsi <= self.oversold_rsi_threshold
                    logger.debug(f"VWAP: {context.symbol} - RSI: {rsi:.1f}, Oversold threshold: {self.oversold_rsi_threshold}, OK: {rsi_ok}")
                else:
                    rsi_ok = False
                    logger.debug(f"VWAP: {context.symbol} - RSI required but not available")

            if rsi_ok:
                # Calculate institutional-grade strength for mean reversion
                strength = self._calculate_institutional_strength(context, vwap_info, setup_type="mean_reversion")
                quality_score = min(90.0, vwap_info.price_distance_bps * 0.5 + (strength * 25))

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type="vwap_mean_reversion_long",
                    side="long",
                    confidence=strength,
                    levels={"vwap": vwap_info.current_vwap},
                    context={
                        "distance_bps": vwap_info.price_distance_bps,
                        "volume_confirmation": vwap_info.volume_confirmation,
                        "rsi_oversold": rsi_ok
                    },
                    price=context.current_price
                )
                events.append(event)

                logger.debug(f"VWAP: {context.symbol} - Mean reversion LONG detected: distance {vwap_info.price_distance_bps:.1f} bps, confidence {strength:.2f}")

        # Mean reversion short (price above VWAP)
        elif current_price > vwap_info.current_vwap:
            logger.debug(f"VWAP: {context.symbol} - Checking mean reversion short setup")

            # Check RSI if required
            rsi_ok = True
            if self.require_oversold_rsi:  # Use same config for overbought
                if context.indicators and 'rsi14' in context.indicators:
                    rsi = context.indicators['rsi14']
                    rsi_ok = rsi >= self.overbought_rsi_threshold
                    logger.debug(f"VWAP: {context.symbol} - RSI: {rsi:.1f}, Overbought threshold: {self.overbought_rsi_threshold}, OK: {rsi_ok}")
                else:
                    rsi_ok = False
                    logger.debug(f"VWAP: {context.symbol} - RSI required but not available")

            if rsi_ok:
                # Calculate institutional-grade strength for mean reversion short
                strength = self._calculate_institutional_strength(context, vwap_info, setup_type="mean_reversion")
                quality_score = min(90.0, vwap_info.price_distance_bps * 0.5 + (strength * 25))

                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type="vwap_mean_reversion_short",
                    side="short",
                    confidence=strength,
                    levels={"vwap": vwap_info.current_vwap},
                    context={
                        "distance_bps": vwap_info.price_distance_bps,
                        "volume_confirmation": vwap_info.volume_confirmation,
                        "rsi_overbought": rsi_ok
                    },
                    price=context.current_price
                )
                events.append(event)

                logger.debug(f"VWAP: {context.symbol} - Mean reversion SHORT detected: distance {vwap_info.price_distance_bps:.1f} bps, confidence {strength:.2f}")

        return events, quality_score

    def _detect_vwap_reclaim(self, context: MarketContext, vwap_info: VWAPLevels) -> Tuple[List[StructureEvent], float]:
        """Detect VWAP reclaim setups (bullish)."""

        events = []
        quality_score = 0.0

        # Must be above VWAP now
        if context.current_price <= vwap_info.current_vwap:
            logger.debug(f"VWAP: {context.symbol} - Price not above VWAP for reclaim setup")
            return events, quality_score

        # Must have been below VWAP recently and now above for required bars
        if vwap_info.above_vwap_bars < self.min_bars_above_vwap:
            logger.debug(f"VWAP: {context.symbol} - Above VWAP bars {vwap_info.above_vwap_bars} < required {self.min_bars_above_vwap}")
            return events, quality_score

        # Check volume confirmation if required
        volume_ok = True
        if self.reclaim_volume_confirmation:
            volume_ok = vwap_info.volume_confirmation
            logger.debug(f"VWAP: {context.symbol} - Volume confirmation required: {volume_ok}")

        if volume_ok:
            # Calculate institutional-grade strength based on market dynamics
            strength = self._calculate_institutional_strength(context, vwap_info)
            quality_score = min(85.0, vwap_info.above_vwap_bars * 10 + (strength * 20))

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type="vwap_reclaim_long",
                side="long",
                confidence=strength,
                levels={"vwap": vwap_info.current_vwap},
                context={
                    "above_vwap_bars": vwap_info.above_vwap_bars,
                    "volume_confirmation": vwap_info.volume_confirmation,
                    "distance_bps": vwap_info.price_distance_bps
                },
                price=context.current_price
            )
            events.append(event)

            logger.debug(f"VWAP: {context.symbol} - VWAP reclaim LONG detected: {vwap_info.above_vwap_bars} bars above, confidence {strength:.2f}")

        return events, quality_score

    def _detect_vwap_lose(self, context: MarketContext, vwap_info: VWAPLevels) -> Tuple[List[StructureEvent], float]:
        """Detect VWAP lose setups (bearish)."""

        events = []
        quality_score = 0.0

        # Must be below VWAP now
        if context.current_price >= vwap_info.current_vwap:
            logger.debug(f"VWAP: {context.symbol} - Price not below VWAP for lose setup")
            return events, quality_score

        # Must have been above VWAP recently and now below for some bars
        if vwap_info.below_vwap_bars < 2:  # At least 2 bars below
            logger.debug(f"VWAP: {context.symbol} - Below VWAP bars {vwap_info.below_vwap_bars} < 2")
            return events, quality_score

        # Check volume confirmation if required
        volume_ok = True
        if self.reclaim_volume_confirmation:  # Use same setting
            volume_ok = vwap_info.volume_confirmation
            logger.debug(f"VWAP: {context.symbol} - Volume confirmation for lose: {volume_ok}")

        if volume_ok:
            # Calculate institutional-grade strength based on market dynamics
            strength = self._calculate_institutional_strength(context, vwap_info, setup_type="lose")
            quality_score = min(80.0, vwap_info.below_vwap_bars * 8 + (strength * 20))

            event = StructureEvent(
                symbol=context.symbol,
                timestamp=context.timestamp,
                structure_type="vwap_lose_short",
                side="short",
                confidence=strength,
                levels={"vwap": vwap_info.current_vwap},
                context={
                    "below_vwap_bars": vwap_info.below_vwap_bars,
                    "volume_confirmation": vwap_info.volume_confirmation,
                    "distance_bps": vwap_info.price_distance_bps
                },
                price=context.current_price
            )
            events.append(event)

            logger.debug(f"VWAP: {context.symbol} - VWAP lose SHORT detected: {vwap_info.below_vwap_bars} bars below, confidence {strength:.2f}")

        return events, quality_score

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan long strategy for VWAP setups."""

        logger.debug(f"VWAP: Planning long strategy for {context.symbol} - {event.structure_type}")

        # Calculate risk parameters
        risk_params = self.calculate_risk_params(context, event, "long")

        # Calculate exit levels
        exit_levels = self.get_exit_levels(context, event, "long")

        # Determine position size based on risk
        entry_price = context.current_price
        qty, notional = self._calculate_position_size(entry_price, risk_params.hard_sl, context)

        plan = TradePlan(
            symbol=context.symbol,
            side="long",
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=event.confidence,
            notes={
                "vwap_level": event.levels.get("vwap"),
                "setup_reason": f"VWAP {event.structure_type}",
                **event.context
            }
        )

        logger.info(f"VWAP: {context.symbol} - Long strategy planned: entry {entry_price:.2f}, SL {risk_params.hard_sl:.2f}, qty {qty}")

        return plan

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        """Plan short strategy for VWAP setups."""

        logger.debug(f"VWAP: Planning short strategy for {context.symbol} - {event.structure_type}")

        # Calculate risk parameters
        risk_params = self.calculate_risk_params(context, event, "short")

        # Calculate exit levels
        exit_levels = self.get_exit_levels(context, event, "short")

        # Determine position size based on risk
        entry_price = context.current_price
        qty, notional = self._calculate_position_size(entry_price, risk_params.hard_sl, context)

        plan = TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=event.confidence,
            notes={
                "vwap_level": event.levels.get("vwap"),
                "setup_reason": f"VWAP {event.structure_type}",
                **event.context
            }
        )

        logger.info(f"VWAP: {context.symbol} - Short strategy planned: entry {entry_price:.2f}, SL {risk_params.hard_sl:.2f}, qty {qty}")

        return plan

    def calculate_risk_params(self, context: MarketContext, event: StructureEvent, side: str) -> RiskParams:
        """Calculate risk parameters for VWAP strategies."""

        entry_price = context.current_price
        vwap_level = event.levels.get("vwap", entry_price)

        # Get ATR for stop calculation
        atr = self._get_atr(context)

        if side == "long":
            # Long stop: below VWAP with buffer
            vwap_stop = vwap_level * (1 - self.vwap_buffer_mult * 0.01)
            atr_stop = entry_price - (atr * self.stop_distance_mult) if atr > 0 else entry_price * 0.99

            # Use the lower (more conservative) stop
            calculated_stop = min(vwap_stop, atr_stop)

            # Ensure minimum stop distance
            min_stop = entry_price * (1 - self.min_stop_distance_pct * 0.01)
            hard_sl = min(calculated_stop, min_stop)

            logger.debug(f"VWAP: {context.symbol} - Long stops: VWAP {vwap_stop:.2f}, ATR {atr_stop:.2f}, Min {min_stop:.2f}, Final {hard_sl:.2f}")

        else:  # short
            # Short stop: above VWAP with buffer
            vwap_stop = vwap_level * (1 + self.vwap_buffer_mult * 0.01)
            atr_stop = entry_price + (atr * self.stop_distance_mult) if atr > 0 else entry_price * 1.01

            # Use the higher (more conservative) stop
            calculated_stop = max(vwap_stop, atr_stop)

            # Ensure minimum stop distance
            min_stop = entry_price * (1 + self.min_stop_distance_pct * 0.01)
            hard_sl = max(calculated_stop, min_stop)

            logger.debug(f"VWAP: {context.symbol} - Short stops: VWAP {vwap_stop:.2f}, ATR {atr_stop:.2f}, Min {min_stop:.2f}, Final {hard_sl:.2f}")

        # Calculate risk per share
        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02  # 2% max risk - should be configurable
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate exit levels for VWAP strategies."""

        entry_price = context.current_price
        vwap_level = event.levels.get("vwap", entry_price)
        atr = self._get_atr(context)

        if side == "long":
            # Target calculation for longs
            target_distance_t1 = atr * self.target_mult_t1 if atr > 0 else entry_price * 0.01
            target_distance_t2 = atr * self.target_mult_t2 if atr > 0 else entry_price * 0.015

            t1_target = entry_price + target_distance_t1
            t2_target = entry_price + target_distance_t2

            logger.debug(f"VWAP: {context.symbol} - Long targets: T1 {t1_target:.2f} (+{target_distance_t1:.2f}), T2 {t2_target:.2f} (+{target_distance_t2:.2f})")

        else:  # short
            # Target calculation for shorts
            target_distance_t1 = atr * self.target_mult_t1 if atr > 0 else entry_price * 0.01
            target_distance_t2 = atr * self.target_mult_t2 if atr > 0 else entry_price * 0.015

            t1_target = entry_price - target_distance_t1
            t2_target = entry_price - target_distance_t2

            logger.debug(f"VWAP: {context.symbol} - Short targets: T1 {t1_target:.2f} (-{target_distance_t1:.2f}), T2 {t2_target:.2f} (-{target_distance_t2:.2f})")

        return ExitLevels(
            targets=[
                {"level": t1_target, "qty_pct": 50, "rr": 1.0},
                {"level": t2_target, "qty_pct": 50, "rr": 2.0}
            ],
            hard_sl=0.0,  # Will be set by risk params
            trail_to=None  # Can be added later
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        """Rank the quality of VWAP setup."""

        base_score = event.confidence * 100

        # Bonus factors
        bonuses = 0.0

        # Volume confirmation bonus
        if event.context.get("volume_confirmation", False):
            bonuses += 15.0
            logger.debug(f"VWAP: {context.symbol} - Volume confirmation bonus: +15")

        # Distance from VWAP bonus (sweet spot)
        distance_bps = event.context.get("distance_bps", 0)
        if 30 <= distance_bps <= 150:  # Sweet spot for mean reversion
            bonuses += 10.0
            logger.debug(f"VWAP: {context.symbol} - Distance sweet spot bonus: +10")

        # RSI confirmation bonus
        if event.context.get("rsi_oversold") or event.context.get("rsi_overbought"):
            bonuses += 12.0
            logger.debug(f"VWAP: {context.symbol} - RSI confirmation bonus: +12")

        # VWAP reclaim duration bonus
        above_bars = event.context.get("above_vwap_bars", 0)
        if above_bars >= 5:
            bonuses += min(10.0, above_bars * 1.5)
            logger.debug(f"VWAP: {context.symbol} - Reclaim duration bonus: +{min(10.0, above_bars * 1.5)}")

        final_score = min(100.0, base_score + bonuses)

        logger.debug(f"VWAP: {context.symbol} - Setup quality: base {base_score:.1f} + bonuses {bonuses:.1f} = {final_score:.1f}")

        return final_score

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        """Validate timing for VWAP setup execution."""

        # Check market hours (basic validation)
        current_hour = context.timestamp.hour
        if current_hour < 9 or current_hour > 15:
            return False, f"Outside market hours: {current_hour}:xx"

        # Check if we have sufficient data
        if len(context.df_5m) < 20:
            return False, f"Insufficient data: {len(context.df_5m)} bars"

        # Check if VWAP is stable (not first few bars of session)
        if len(context.df_5m) < 12:  # Less than 1 hour of data
            return False, "VWAP not yet stable (< 1 hour of data)"

        # All checks passed
        logger.debug(f"VWAP: {context.symbol} - Timing validation passed")
        return True, "Timing validated"

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR value with fallback."""
        if context.indicators and 'atr' in context.indicators:
            atr = context.indicators['atr']
            if atr > 0:
                return atr

        # Fallback: calculate simple ATR from recent data
        try:
            df = context.df_5m
            if len(df) >= 14:
                highs = df['high'].tail(14)
                lows = df['low'].tail(14)
                closes = df['close'].tail(15)  # Need one extra for previous close

                true_ranges = []
                for i in range(1, 15):
                    tr = max(
                        highs.iloc[i] - lows.iloc[i],
                        abs(highs.iloc[i] - closes.iloc[i-1]),
                        abs(lows.iloc[i] - closes.iloc[i-1])
                    )
                    true_ranges.append(tr)

                atr = sum(true_ranges) / len(true_ranges)
                logger.debug(f"VWAP: {context.symbol} - Calculated fallback ATR: {atr:.3f}")
                return atr
        except Exception as e:
            logger.warning(f"VWAP: {context.symbol} - ATR calculation failed: {e}")

        # Final fallback
        fallback_atr = context.current_price * 0.01  # 1% of price
        logger.warning(f"VWAP: {context.symbol} - Using emergency ATR fallback: {fallback_atr:.3f}")
        return fallback_atr

    def _calculate_position_size(self, entry_price: float, stop_loss: float, context: MarketContext) -> Tuple[int, float]:
        """Calculate position size based on risk management."""

        # This should be enhanced with actual portfolio size and risk management
        # For now, use basic calculation
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = 1000.0  # Should be configurable

        if risk_per_share > 0:
            max_qty = int(max_risk_amount / risk_per_share)
            # Ensure minimum viable position
            qty = max(1, min(max_qty, 100))  # Min 1, max 100 shares
        else:
            qty = 1

        notional = qty * entry_price

        logger.debug(f"VWAP: {context.symbol} - Position calc: risk/share {risk_per_share:.3f}, qty {qty}, notional {notional:.2f}")

        return qty, notional

    def _calculate_institutional_strength(self, context: MarketContext, vwap_info: VWAPLevels, setup_type: str = "reclaim") -> float:
        """
        Calculate institutional-grade strength based on market dynamics.

        Based on professional trading criteria from institutional benchmarks:
        - Volume surge multiplier (≥1.5-2× for strong setups)
        - Structure quality bonuses
        - Market timing context

        Returns strength in range 0.5-4.0+ to match old system's vol_z scoring
        """

        # Get volume multiplier from context (current volume vs average)
        volume_mult = getattr(vwap_info, 'volume_multiplier', 1.0)
        if hasattr(context, 'indicators') and context.indicators and 'volume_mult' in context.indicators:
            volume_mult = float(context.indicators['volume_mult'])

        # Base strength from volume participation (like old system's vol_z)
        # Scale to institutional benchmarks: 1.5x = 1.5, 2.0x = 2.0, etc.
        base_strength = max(0.5, volume_mult)

        logger.debug(f"VWAP: {context.symbol} - Base strength from volume_mult {volume_mult:.2f}: {base_strength:.2f}")

        # Professional criteria bonuses (matching old system logic)
        strength_multiplier = 1.0

        # 1. Volume confirmation bonus (20% like old system)
        if vwap_info.volume_confirmation and volume_mult >= 1.5:
            strength_multiplier *= 1.2
            logger.debug(f"VWAP: {context.symbol} - Volume confirmation bonus: 1.2x")

        # 2. Strong volume surge bonus (institutional 2× threshold)
        if volume_mult >= 2.0:
            strength_multiplier *= 1.3  # 30% bonus for strong surge
            logger.debug(f"VWAP: {context.symbol} - Strong volume surge bonus: 1.3x")

        # 3. Setup-specific structure quality bonuses
        if setup_type == "reclaim":
            # VWAP reclaim - bars above confirmation
            bars_above = getattr(vwap_info, 'above_vwap_bars', 0)
            if bars_above >= 4:  # Strong reclaim pattern
                strength_multiplier *= 1.1  # 10% bonus for sustained reclaim
                logger.debug(f"VWAP: {context.symbol} - Sustained reclaim bonus ({bars_above} bars): 1.1x")

        elif setup_type == "lose":
            # VWAP lose - bars below confirmation
            bars_below = getattr(vwap_info, 'below_vwap_bars', 0)
            if bars_below >= 3:  # Strong lose pattern
                strength_multiplier *= 1.1  # 10% bonus for sustained lose
                logger.debug(f"VWAP: {context.symbol} - Sustained lose bonus ({bars_below} bars): 1.1x")

        elif setup_type == "mean_reversion":
            # Mean reversion - distance from VWAP
            distance_bps = getattr(vwap_info, 'price_distance_bps', 0)
            if distance_bps >= 200:  # Significant distance (2× min institutional threshold)
                strength_multiplier *= 1.15  # 15% bonus for extreme deviation
                logger.debug(f"VWAP: {context.symbol} - Extreme deviation bonus ({distance_bps:.1f} bps): 1.15x")

        # 4. Session timing context (like old system's session_weight)
        # Higher weight during peak liquidity hours
        if hasattr(context, 'timestamp'):
            hour = context.timestamp.hour
            if 9 <= hour <= 11 or 14 <= hour <= 16:  # Peak trading hours IST
                strength_multiplier *= 1.1  # 10% bonus for peak liquidity
                logger.debug(f"VWAP: {context.symbol} - Peak hours bonus (hour {hour}): 1.1x")

        # Calculate final institutional strength
        final_strength = base_strength * strength_multiplier

        # Ensure minimum threshold for regime gate compatibility
        # Institutional setups need ≥2.0 strength to pass regime gate
        final_strength = max(final_strength, 0.8)  # Minimum viable strength

        logger.info(f"VWAP: {context.symbol} - Institutional strength calculation: "
                   f"base {base_strength:.2f} × multiplier {strength_multiplier:.2f} = {final_strength:.2f}")

        return final_strength