# structures/ict_structure_compatible.py
"""
ICT (Inner Circle Trader) Structure Implementation - Fully Compatible Version.

Handles all ICT concepts including Order Blocks, Fair Value Gaps, Liquidity Sweeps,
Premium/Discount Zones, Break of Structure, and Change of Character patterns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, time

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    StructureEvent,
    TradePlan,
    RiskParams,
    ExitLevels,
    MarketContext,
    StructureAnalysis
)

@dataclass
class StrategyPlanResult:
    """Result of strategy planning with rejection tracking."""
    plan: Optional[TradePlan]
    rejected: bool
    rejection_reason: Optional[str] = None

logger = get_agent_logger()


class ICTStructureCompatible(BaseStructure):
    """
    ICT (Inner Circle Trader) structure implementation - fully compatible version.

    Detects institutional trading patterns including Order Blocks, Fair Value Gaps,
    Liquidity Sweeps, Premium/Discount Zones, Break of Structure, and Change of Character.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize ICT structure with configuration."""
        super().__init__(config)

        # ICT-specific configuration - KeyError if missing trading parameters

        # Order Blocks parameters
        self.ob_min_move_pct = config["order_block_min_move_pct"] / 100.0
        self.ob_min_volume_surge = config["order_block_min_volume_surge"]
        self.ob_lookback_bars = config["order_block_lookback"]
        self.ob_test_tolerance_pct = config["order_block_test_tolerance_pct"] / 100.0

        # Fair Value Gap parameters
        self.fvg_min_gap_pct = config["fvg_min_gap_pct"] / 100.0
        self.fvg_max_gap_pct = config["fvg_max_gap_pct"] / 100.0
        self.fvg_require_volume = config["fvg_require_volume_spike"]
        self.fvg_min_volume_mult = config["fvg_min_volume_mult"]
        self.fvg_fill_tolerance_pct = config["fvg_fill_tolerance_pct"] / 100.0

        # Liquidity Sweep parameters
        self.sweep_min_distance_pct = config["sweep_min_distance_pct"] / 100.0
        self.sweep_max_distance_pct = config["sweep_max_distance_pct"] / 100.0
        self.sweep_min_volume_surge = config["sweep_min_volume_surge"]
        self.sweep_min_wick_ratio = config["sweep_min_wick_ratio"]
        self.sweep_reversal_bars = config["sweep_reversal_bars"]

        # Premium/Discount parameters
        self.premium_threshold_pct = config["premium_threshold_pct"] / 100.0
        self.discount_threshold_pct = config["discount_threshold_pct"] / 100.0
        self.range_lookback_bars = config["range_lookback_bars"]

        # Break of Structure parameters
        self.bos_min_structure_bars = config["bos_min_structure_bars"]
        self.bos_min_break_pct = config["bos_min_break_pct"] / 100.0
        self.bos_volume_confirmation = config["bos_volume_confirmation"]

        # Change of Character parameters
        self.choch_momentum_periods = config["choch_momentum_periods"]
        self.choch_min_momentum_change_pct = config["choch_min_momentum_change_pct"] / 100.0
        self.choch_volume_threshold = config["choch_volume_threshold"]

        # Confidence levels
        self.confidence_strong_signal = config["confidence_strong_signal"]
        self.confidence_medium_signal = config["confidence_medium_signal"]
        self.confidence_weak_signal = config["confidence_weak_signal"]

        logger.debug(f"ICT: Initialized with config - OB move: {self.ob_min_move_pct*100:.1f}%, "
                   f"FVG gap: {self.fvg_min_gap_pct*100:.2f}%-{self.fvg_max_gap_pct*100:.1f}%, "
                   f"Sweep vol: {self.sweep_min_volume_surge}x")

        # Session timing for ICT concepts
        self.session_start = time(9, 15)  # Market open
        self.london_session_start = time(2, 0)  # London session (IST)
        self.new_york_session_start = time(7, 0)  # NY session (IST)

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        """
        Detect ICT patterns in market data.

        Args:
            market_context: Market data and context

        Returns:
            StructureAnalysis with detected ICT events
        """
        symbol = market_context.symbol
        logger.debug(f"ICT: Starting detection for {symbol}")

        try:
            df = market_context.df_5m
            if df is None:
                logger.debug(f"ICT: {symbol} - No 5m data available")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="No 5m data available"
                )

            if len(df) < 20:
                logger.debug(f"ICT: {symbol} - Insufficient data: {len(df)} bars < 20 minimum")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data: {len(df)} bars < 20 minimum"
                )

            events = []

            # Prepare data with volume indicators
            df_with_indicators = self._add_volume_indicators(df)

            # Define key levels for liquidity sweeps
            levels = {
                "PDH": market_context.pdh,
                "PDL": market_context.pdl,
                "ORH": market_context.orh,
                "ORL": market_context.orl
            }

            # Detect all ICT patterns
            order_block_events = self._detect_order_blocks(df_with_indicators, market_context)
            fvg_events = self._detect_fair_value_gaps(df_with_indicators, market_context)
            sweep_events = self._detect_liquidity_sweeps(df_with_indicators, market_context, levels)
            premium_discount_events = self._detect_premium_discount_zones(df_with_indicators, market_context, levels)
            bos_events = self._detect_break_of_structure(df_with_indicators, market_context)
            choch_events = self._detect_change_of_character(df_with_indicators, market_context)

            # Combine all events
            all_events = (order_block_events + fvg_events + sweep_events +
                         premium_discount_events + bos_events + choch_events)

            if all_events:
                logger.debug(f"ICT: {symbol} - Detected {len(all_events)} ICT patterns")

            # Calculate quality score
            quality_score = self._calculate_ict_quality_score(all_events, df_with_indicators, market_context)
            structure_detected = len(all_events) > 0

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=all_events,
                quality_score=quality_score
            )

        except Exception as e:
            logger.exception(f"ICT: {symbol} - Detection error: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {str(e)}"
            )

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis indicators to dataframe."""
        d = df.copy()

        try:
            # Volume z-score
            vol_mean = d['volume'].rolling(20, min_periods=10).mean()
            vol_std = d['volume'].rolling(20, min_periods=10).std()
            d['vol_z'] = (d['volume'] - vol_mean) / vol_std.replace(0, np.nan)

            # Volume surge ratio
            d['vol_ma10'] = d['volume'].rolling(10, min_periods=5).mean()
            d['vol_surge'] = d['volume'] / d['vol_ma10']

            # Price returns for momentum analysis
            d['returns_1'] = d['close'].pct_change()
            d['returns_3'] = d['close'].pct_change(3)
            d['returns_5'] = d['close'].pct_change(5)

            # Fill NaN values
            d['vol_z'].fillna(0, inplace=True)
            d['vol_surge'].fillna(1, inplace=True)
            d['returns_1'].fillna(0, inplace=True)
            d['returns_3'].fillna(0, inplace=True)
            d['returns_5'].fillna(0, inplace=True)

        except Exception as e:
            logger.error(f"ICT: Error adding volume indicators: {e}")

        return d

    def _detect_order_blocks(self, df: pd.DataFrame, market_context: MarketContext) -> List[StructureEvent]:
        """Detect Order Blocks - institutional accumulation/distribution zones."""
        events = []

        try:
            current_price = market_context.current_price
            current_bar_idx = len(df) - 1

            # Look for significant moves in recent history
            search_start = max(5, current_bar_idx - self.ob_lookback_bars)

            for move_start_idx in range(search_start, current_bar_idx - 2):
                # Check for significant move (5-bar window)
                move_bars = df.iloc[move_start_idx:move_start_idx + 5]
                if len(move_bars) < 3:
                    continue

                move_start_price = move_bars['close'].iloc[0]
                move_end_price = move_bars['close'].iloc[-1]
                move_pct = (move_end_price - move_start_price) / move_start_price

                # Check volume confirmation
                move_had_volume = (move_bars['vol_surge'] > self.ob_min_volume_surge).any()

                if abs(move_pct) >= self.ob_min_move_pct and move_had_volume:
                    # Find the last opposing candle before this move
                    ob_candle_idx = self._find_opposing_candle(df, move_start_idx, move_pct)

                    if ob_candle_idx is not None:
                        event = self._create_order_block_event(df, ob_candle_idx, move_pct,
                                                             current_price, current_bar_idx, market_context)
                        if event:
                            events.append(event)

        except Exception as e:
            logger.error(f"ICT: Order block detection error: {e}")

        return events

    def _find_opposing_candle(self, df: pd.DataFrame, move_start_idx: int, move_pct: float) -> Optional[int]:
        """Find the last opposing candle before a significant move."""
        for look_idx in range(move_start_idx - 1, max(0, move_start_idx - 8), -1):
            candle = df.iloc[look_idx]

            if move_pct > 0:  # Bullish move - look for last bearish candle
                if candle['close'] < candle['open']:
                    return look_idx
            else:  # Bearish move - look for last bullish candle
                if candle['close'] > candle['open']:
                    return look_idx

        return None

    def _create_order_block_event(self, df: pd.DataFrame, ob_candle_idx: int, move_pct: float,
                                current_price: float, current_bar_idx: int,
                                market_context: MarketContext) -> Optional[StructureEvent]:
        """Create order block event if current price is testing the zone."""
        ob_candle = df.iloc[ob_candle_idx]
        ob_high = ob_candle['high']
        ob_low = ob_candle['low']

        if move_pct > 0:  # Bearish OB (resistance zone)
            if (ob_low <= current_price <= ob_high * (1 + self.ob_test_tolerance_pct)):
                bars_since_ob = current_bar_idx - ob_candle_idx
                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))
                confidence = min(self.confidence_strong_signal, abs(move_pct) * 100 * time_decay / 3.0)

                return StructureEvent(
                    symbol=market_context.symbol,
                    timestamp=market_context.timestamp,
                    structure_type='order_block_short',
                    side='short',
                    confidence=confidence,
                    levels={'entry': ob_high, 'stop': ob_high * 1.01, 'target': ob_low},
                    context={
                        'ob_high': ob_high,
                        'ob_low': ob_low,
                        'move_pct': move_pct * 100,
                        'bars_since_formation': bars_since_ob,
                        'pattern_type': 'bearish_order_block'
                    },
                    price=current_price,
                    indicators=market_context.indicators
                )
        else:  # Bullish OB (support zone)
            if (ob_low * (1 - self.ob_test_tolerance_pct) <= current_price <= ob_high):
                bars_since_ob = current_bar_idx - ob_candle_idx
                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))
                confidence = min(self.confidence_strong_signal, abs(move_pct) * 100 * time_decay / 3.0)

                return StructureEvent(
                    symbol=market_context.symbol,
                    timestamp=market_context.timestamp,
                    structure_type='order_block_long',
                    side='long',
                    confidence=confidence,
                    levels={'entry': ob_low, 'stop': ob_low * 0.99, 'target': ob_high},
                    context={
                        'ob_high': ob_high,
                        'ob_low': ob_low,
                        'move_pct': move_pct * 100,
                        'bars_since_formation': bars_since_ob,
                        'pattern_type': 'bullish_order_block'
                    },
                    price=current_price,
                    indicators=market_context.indicators
                )

        return None

    def _detect_fair_value_gaps(self, df: pd.DataFrame, market_context: MarketContext) -> List[StructureEvent]:
        """Detect Fair Value Gaps - price imbalances indicating institutional activity."""
        events = []

        try:
            current_price = market_context.current_price
            lookback_bars = min(20, len(df) - 3)

            for i in range(2, lookback_bars):
                if i >= len(df) - 1:
                    continue

                candle_before = df.iloc[i - 1]
                candle_middle = df.iloc[i]
                candle_after = df.iloc[i + 1]

                # Check volume condition
                if self.fvg_require_volume:
                    vol_surge = df['vol_surge'].iloc[i]
                    if vol_surge < self.fvg_min_volume_mult:
                        continue

                # Bullish FVG
                if (candle_before['high'] < candle_after['low'] and
                    candle_middle['close'] > candle_middle['open']):

                    event = self._create_fvg_event('long', candle_before, candle_after,
                                                 df, i, current_price, market_context)
                    if event:
                        events.append(event)

                # Bearish FVG
                elif (candle_before['low'] > candle_after['high'] and
                      candle_middle['close'] < candle_middle['open']):

                    event = self._create_fvg_event('short', candle_before, candle_after,
                                                 df, i, current_price, market_context)
                    if event:
                        events.append(event)

        except Exception as e:
            logger.error(f"ICT: Fair Value Gap detection error: {e}")

        return events

    def _create_fvg_event(self, side: str, candle_before: pd.Series, candle_after: pd.Series,
                         df: pd.DataFrame, gap_index: int, current_price: float,
                         market_context: MarketContext) -> Optional[StructureEvent]:
        """Create Fair Value Gap event if conditions are met."""
        if side == 'long':
            gap_size = candle_after['low'] - candle_before['high']
            gap_pct = gap_size / candle_before['high']
            fvg_top = candle_after['low']
            fvg_bottom = candle_before['high']
        else:
            gap_size = candle_before['low'] - candle_after['high']
            gap_pct = gap_size / candle_after['high']
            fvg_top = candle_before['low']
            fvg_bottom = candle_after['high']

        # Check gap size
        if not (self.fvg_min_gap_pct <= gap_pct <= self.fvg_max_gap_pct):
            return None

        # Check if current price is testing this FVG
        if not (fvg_bottom * (1 - self.fvg_fill_tolerance_pct) <= current_price <=
                fvg_top * (1 + self.fvg_fill_tolerance_pct)):
            return None

        # Calculate confidence
        volume_strength = df['vol_surge'].iloc[gap_index]
        confidence = min(self.confidence_strong_signal, gap_pct * 500 + volume_strength * 0.1)

        return StructureEvent(
            symbol=market_context.symbol,
            timestamp=market_context.timestamp,
            structure_type=f'fair_value_gap_{side}',
            side=side,
            confidence=confidence,
            levels={'entry': fvg_bottom if side == 'long' else fvg_top,
                   'stop': fvg_bottom * 0.995 if side == 'long' else fvg_top * 1.005,
                   'target': fvg_top if side == 'long' else fvg_bottom},
            context={
                'fvg_top': fvg_top,
                'fvg_bottom': fvg_bottom,
                'gap_size_pct': gap_pct * 100,
                'volume_surge': volume_strength,
                'pattern_type': f'{side}_fair_value_gap'
            },
            price=current_price,
            indicators=market_context.indicators
        )

    def _detect_liquidity_sweeps(self, df: pd.DataFrame, market_context: MarketContext,
                               levels: Dict[str, float]) -> List[StructureEvent]:
        """Detect Liquidity Sweeps - stop hunt patterns."""
        events = []

        try:
            current_price = market_context.current_price
            lookback_bars = min(10, len(df) - self.sweep_reversal_bars)

            for level_name, level_price in levels.items():
                if level_price is None or level_price <= 0 or not np.isfinite(level_price):
                    continue

                for i in range(lookback_bars, len(df) - 1):
                    sweep_bar = df.iloc[i]

                    # Calculate wick ratios
                    bar_range = sweep_bar['high'] - sweep_bar['low']
                    if bar_range <= 0:
                        continue

                    upper_wick = sweep_bar['high'] - max(sweep_bar['open'], sweep_bar['close'])
                    lower_wick = min(sweep_bar['open'], sweep_bar['close']) - sweep_bar['low']
                    upper_wick_ratio = upper_wick / bar_range
                    lower_wick_ratio = lower_wick / bar_range

                    # Check volume surge
                    vol_surge = df['vol_surge'].iloc[i]
                    if vol_surge < self.sweep_min_volume_surge:
                        continue

                    event = self._check_liquidity_sweep(df, i, level_name, level_price,
                                                      upper_wick_ratio, lower_wick_ratio,
                                                      current_price, market_context)
                    if event:
                        events.append(event)

        except Exception as e:
            logger.error(f"ICT: Liquidity sweep detection error: {e}")

        return events

    def _check_liquidity_sweep(self, df: pd.DataFrame, sweep_idx: int, level_name: str,
                             level_price: float, upper_wick_ratio: float, lower_wick_ratio: float,
                             current_price: float, market_context: MarketContext) -> Optional[StructureEvent]:
        """Check for specific liquidity sweep pattern."""
        sweep_bar = df.iloc[sweep_idx]

        # Bullish liquidity sweep (hunt sell stops below support)
        if level_name in ["PDL", "ORL"]:
            sweep_distance = level_price - sweep_bar['low']
            sweep_pct = sweep_distance / level_price

            if (self.sweep_min_distance_pct <= sweep_pct <= self.sweep_max_distance_pct and
                lower_wick_ratio >= self.sweep_min_wick_ratio and
                sweep_bar['close'] > level_price):

                # Check for reversal confirmation
                if self._check_sweep_reversal(df, sweep_idx, level_price, 'long'):
                    confidence = min(self.confidence_strong_signal, sweep_pct * 200 + lower_wick_ratio * 2)

                    return StructureEvent(
                        symbol=market_context.symbol,
                        timestamp=market_context.timestamp,
                        structure_type='liquidity_sweep_long',
                        side='long',
                        confidence=confidence,
                        levels={'entry': level_price, 'stop': sweep_bar['low'], 'target': level_price * 1.01},
                        context={
                            'level_name': level_name,
                            'level_price': level_price,
                            'sweep_low': sweep_bar['low'],
                            'sweep_distance_pct': sweep_pct * 100,
                            'wick_ratio': lower_wick_ratio,
                            'pattern_type': 'bullish_liquidity_sweep'
                        },
                        price=current_price,
                        indicators=market_context.indicators
                    )

        # Bearish liquidity sweep (hunt buy stops above resistance)
        elif level_name in ["PDH", "ORH"]:
            sweep_distance = sweep_bar['high'] - level_price
            sweep_pct = sweep_distance / level_price

            if (self.sweep_min_distance_pct <= sweep_pct <= self.sweep_max_distance_pct and
                upper_wick_ratio >= self.sweep_min_wick_ratio and
                sweep_bar['close'] < level_price):

                # Check for reversal confirmation
                if self._check_sweep_reversal(df, sweep_idx, level_price, 'short'):
                    confidence = min(self.confidence_strong_signal, sweep_pct * 200 + upper_wick_ratio * 2)

                    return StructureEvent(
                        symbol=market_context.symbol,
                        timestamp=market_context.timestamp,
                        structure_type='liquidity_sweep_short',
                        side='short',
                        confidence=confidence,
                        levels={'entry': level_price, 'stop': sweep_bar['high'], 'target': level_price * 0.99},
                        context={
                            'level_name': level_name,
                            'level_price': level_price,
                            'sweep_high': sweep_bar['high'],
                            'sweep_distance_pct': sweep_pct * 100,
                            'wick_ratio': upper_wick_ratio,
                            'pattern_type': 'bearish_liquidity_sweep'
                        },
                        price=current_price,
                        indicators=market_context.indicators
                    )

        return None

    def _check_sweep_reversal(self, df: pd.DataFrame, sweep_idx: int, level_price: float, side: str) -> bool:
        """Check for quick reversal confirmation after liquidity sweep."""
        reversal_bars = df.iloc[sweep_idx+1:sweep_idx+1+self.sweep_reversal_bars]
        if len(reversal_bars) == 0:
            return False

        if side == 'long':
            # Price should stay above level and move higher
            above_level = (reversal_bars['low'] > level_price * 0.998).all()
            moved_higher = reversal_bars['close'].iloc[-1] > df.iloc[sweep_idx]['close']
            return above_level and moved_higher
        else:
            # Price should stay below level and move lower
            below_level = (reversal_bars['high'] < level_price * 1.002).all()
            moved_lower = reversal_bars['close'].iloc[-1] < df.iloc[sweep_idx]['close']
            return below_level and moved_lower

    def _detect_premium_discount_zones(self, df: pd.DataFrame, market_context: MarketContext,
                                     levels: Dict[str, float]) -> List[StructureEvent]:
        """Detect Premium/Discount zone positioning."""
        events = []

        try:
            # Calculate recent range
            recent_data = df.tail(self.range_lookback_bars)
            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            range_size = range_high - range_low

            if range_size <= 0:
                return events

            current_price = market_context.current_price

            # Calculate position in range (0 = bottom, 1 = top)
            range_position = (current_price - range_low) / range_size

            # Premium zone (top 30% of range)
            if range_position >= self.premium_threshold_pct:
                confidence = min(self.confidence_medium_signal, (range_position - self.premium_threshold_pct) * 10)

                events.append(StructureEvent(
                    symbol=market_context.symbol,
                    timestamp=market_context.timestamp,
                    structure_type='premium_zone_short',
                    side='short',
                    confidence=confidence,
                    levels={'entry': current_price, 'stop': range_high * 1.005, 'target': range_low},
                    context={
                        'range_high': range_high,
                        'range_low': range_low,
                        'range_position_pct': range_position * 100,
                        'zone_type': 'premium',
                        'pattern_type': 'premium_zone_positioning'
                    },
                    price=current_price,
                    indicators=market_context.indicators
                ))

            # Discount zone (bottom 30% of range)
            elif range_position <= self.discount_threshold_pct:
                confidence = min(self.confidence_medium_signal, (self.discount_threshold_pct - range_position) * 10)

                events.append(StructureEvent(
                    symbol=market_context.symbol,
                    timestamp=market_context.timestamp,
                    structure_type='discount_zone_long',
                    side='long',
                    confidence=confidence,
                    levels={'entry': current_price, 'stop': range_low * 0.995, 'target': range_high},
                    context={
                        'range_high': range_high,
                        'range_low': range_low,
                        'range_position_pct': range_position * 100,
                        'zone_type': 'discount',
                        'pattern_type': 'discount_zone_positioning'
                    },
                    price=current_price,
                    indicators=market_context.indicators
                ))

        except Exception as e:
            logger.error(f"ICT: Premium/Discount zone detection error: {e}")

        return events

    def _detect_break_of_structure(self, df: pd.DataFrame, market_context: MarketContext) -> List[StructureEvent]:
        """Detect Break of Structure - trend change confirmations."""
        events = []

        try:
            if len(df) < self.bos_min_structure_bars + 5:
                return events

            current_price = market_context.current_price

            # Find recent swing highs and lows
            swing_highs = self._find_swing_points(df, 'high')
            swing_lows = self._find_swing_points(df, 'low')

            # Check for bullish BOS (break above recent swing high)
            if swing_highs:
                recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
                break_distance = current_price - recent_high
                break_pct = break_distance / recent_high

                if break_pct >= self.bos_min_break_pct:
                    # Check volume confirmation if required
                    volume_confirmed = True
                    if self.bos_volume_confirmation:
                        recent_vol_z = df['vol_z'].tail(3).max()
                        volume_confirmed = recent_vol_z >= 1.5

                    if volume_confirmed:
                        confidence = min(self.confidence_strong_signal, break_pct * 200)

                        events.append(StructureEvent(
                            symbol=market_context.symbol,
                            timestamp=market_context.timestamp,
                            structure_type='break_of_structure_long',
                            side='long',
                            confidence=confidence,
                            levels={'entry': current_price, 'stop': recent_high, 'target': current_price * 1.02},
                            context={
                                'broken_level': recent_high,
                                'break_distance_pct': break_pct * 100,
                                'structure_type': 'swing_high',
                                'pattern_type': 'bullish_break_of_structure'
                            },
                            price=current_price,
                            indicators=market_context.indicators
                        ))

            # Check for bearish BOS (break below recent swing low)
            if swing_lows:
                recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]
                break_distance = recent_low - current_price
                break_pct = break_distance / recent_low

                if break_pct >= self.bos_min_break_pct:
                    # Check volume confirmation if required
                    volume_confirmed = True
                    if self.bos_volume_confirmation:
                        recent_vol_z = df['vol_z'].tail(3).max()
                        volume_confirmed = recent_vol_z >= 1.5

                    if volume_confirmed:
                        confidence = min(self.confidence_strong_signal, break_pct * 200)

                        events.append(StructureEvent(
                            symbol=market_context.symbol,
                            timestamp=market_context.timestamp,
                            structure_type='break_of_structure_short',
                            side='short',
                            confidence=confidence,
                            levels={'entry': current_price, 'stop': recent_low, 'target': current_price * 0.98},
                            context={
                                'broken_level': recent_low,
                                'break_distance_pct': break_pct * 100,
                                'structure_type': 'swing_low',
                                'pattern_type': 'bearish_break_of_structure'
                            },
                            price=current_price,
                            indicators=market_context.indicators
                        ))

        except Exception as e:
            logger.error(f"ICT: Break of Structure detection error: {e}")

        return events

    def _find_swing_points(self, df: pd.DataFrame, price_type: str) -> List[float]:
        """Find swing highs or lows in the data."""
        swing_points = []
        lookback = min(self.bos_min_structure_bars, len(df) - 2)

        for i in range(2, len(df) - 2):
            if i < lookback:
                continue

            window = df.iloc[i-2:i+3]
            center_value = window[price_type].iloc[2]

            if price_type == 'high':
                # Swing high: center is highest in window
                if center_value == window[price_type].max():
                    swing_points.append(center_value)
            else:
                # Swing low: center is lowest in window
                if center_value == window[price_type].min():
                    swing_points.append(center_value)

        return swing_points

    def _detect_change_of_character(self, df: pd.DataFrame, market_context: MarketContext) -> List[StructureEvent]:
        """Detect Change of Character - momentum shift detection."""
        events = []

        try:
            if len(df) < max(self.choch_momentum_periods) + 5:
                return events

            current_price = market_context.current_price

            # Calculate momentum changes for different periods
            momentum_changes = {}
            for period in self.choch_momentum_periods:
                if len(df) >= period + 1:
                    old_momentum = df['returns_3'].iloc[-(period+1)]
                    current_momentum = df['returns_3'].iloc[-1]
                    momentum_change = current_momentum - old_momentum
                    momentum_changes[period] = momentum_change

            # Check for significant momentum shift
            significant_changes = [abs(change) >= self.choch_min_momentum_change_pct
                                 for change in momentum_changes.values()]

            if any(significant_changes):
                # Determine direction of character change
                avg_momentum_change = np.mean(list(momentum_changes.values()))
                side = 'long' if avg_momentum_change > 0 else 'short'

                # Check volume confirmation
                recent_vol_z = df['vol_z'].tail(3).max()
                if recent_vol_z >= self.choch_volume_threshold:
                    confidence = min(self.confidence_strong_signal, abs(avg_momentum_change) * 100)

                    events.append(StructureEvent(
                        symbol=market_context.symbol,
                        timestamp=market_context.timestamp,
                        structure_type=f'change_of_character_{side}',
                        side=side,
                        confidence=confidence,
                        levels={'entry': current_price,
                               'stop': current_price * (0.98 if side == 'long' else 1.02),
                               'target': current_price * (1.02 if side == 'long' else 0.98)},
                        context={
                            'momentum_changes': momentum_changes,
                            'avg_momentum_change_pct': avg_momentum_change * 100,
                            'volume_confirmation': recent_vol_z,
                            'pattern_type': f'{side}_change_of_character'
                        },
                        price=current_price,
                        indicators=market_context.indicators
                    ))

        except Exception as e:
            logger.error(f"ICT: Change of Character detection error: {e}")

        return events

    def _calculate_ict_quality_score(self, events: List[StructureEvent], df: pd.DataFrame,
                                   market_context: MarketContext) -> float:
        """Calculate quality score based on multiple ICT confirmations."""
        if not events:
            return 0.0

        try:
            score = 0.0

            # Base score from number of confirmations
            num_events = len(events)
            score += min(40.0, num_events * 10.0)

            # Volume confirmation
            recent_vol_z = df['vol_z'].tail(3).max()
            if recent_vol_z >= 2.0:
                score += 20.0
            elif recent_vol_z >= 1.5:
                score += 10.0

            # Pattern diversity bonus
            pattern_types = set(event.structure_type.split('_')[0] for event in events)
            if len(pattern_types) >= 3:
                score += 20.0
            elif len(pattern_types) >= 2:
                score += 10.0

            # Confidence-based adjustment
            avg_confidence = np.mean([event.confidence for event in events])
            score += avg_confidence * 20.0

            return min(100.0, score)

        except Exception as e:
            logger.error(f"ICT: Quality score calculation error: {e}")
            return 0.0

    # Abstract method implementations for BaseStructure compatibility

    def plan_long_strategy(self, market_context: MarketContext) -> StrategyPlanResult:
        """Plan long strategy based on ICT analysis."""
        try:
            analysis = self.detect(market_context)
            long_events = [e for e in analysis.events if e.side == 'long']

            if not long_events:
                return StrategyPlanResult(
                    plan=None,
                    rejected=True,
                    rejection_reason="No ICT long signals detected"
                )

            # Use the highest confidence event
            best_event = max(long_events, key=lambda e: e.confidence)

            # Create trade plan from event
            trade_plan = self._create_trade_plan_from_event(best_event, market_context)

            return StrategyPlanResult(
                plan=trade_plan,
                rejected=False
            )

        except Exception as e:
            logger.error(f"ICT: Long strategy planning error: {e}")
            return StrategyPlanResult(
                plan=None,
                rejected=True,
                rejection_reason=f"Strategy planning error: {str(e)}"
            )

    def plan_short_strategy(self, market_context: MarketContext) -> StrategyPlanResult:
        """Plan short strategy based on ICT analysis."""
        try:
            analysis = self.detect(market_context)
            short_events = [e for e in analysis.events if e.side == 'short']

            if not short_events:
                return StrategyPlanResult(
                    plan=None,
                    rejected=True,
                    rejection_reason="No ICT short signals detected"
                )

            # Use the highest confidence event
            best_event = max(short_events, key=lambda e: e.confidence)

            # Create trade plan from event
            trade_plan = self._create_trade_plan_from_event(best_event, market_context)

            return StrategyPlanResult(
                plan=trade_plan,
                rejected=False
            )

        except Exception as e:
            logger.error(f"ICT: Short strategy planning error: {e}")
            return StrategyPlanResult(
                plan=None,
                rejected=True,
                rejection_reason=f"Strategy planning error: {str(e)}"
            )

    def _create_trade_plan_from_event(self, event: StructureEvent, market_context: MarketContext) -> TradePlan:
        """Create trade plan from structure event."""
        # Calculate position sizing
        atr = market_context.indicators.get('atr', 1.0) if market_context.indicators else 1.0
        entry_price = event.levels.get('entry', event.price)
        stop_loss = event.levels.get('stop', entry_price * (0.99 if event.side == 'long' else 1.01))
        take_profit = event.levels.get('target', entry_price * (1.02 if event.side == 'long' else 0.98))

        # Risk parameters
        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_per_share=abs(entry_price - stop_loss),
            atr=atr
        )

        # Exit levels
        exit_levels = ExitLevels(
            targets=[{
                'level': take_profit,
                'qty_pct': 100,
                'rr': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
            }],
            hard_sl=stop_loss
        )

        # Basic position sizing (could be enhanced)
        qty = 100  # Default quantity
        notional = entry_price * qty

        return TradePlan(
            symbol=market_context.symbol,
            side=event.side,
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=event.confidence,
            timestamp=market_context.timestamp,
            market_context=market_context
        )

    def calculate_risk_params(self, market_context: MarketContext, side: str) -> RiskParams:
        """Calculate risk parameters for ICT setups."""
        atr = market_context.indicators.get('atr', 1.0) if market_context.indicators else 1.0
        current_price = market_context.current_price

        if side == 'long':
            hard_sl = current_price - (atr * 2.0)
        else:
            hard_sl = current_price + (atr * 2.0)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=abs(current_price - hard_sl),
            atr=atr
        )

    def get_exit_levels(self, market_context: MarketContext, side: str) -> ExitLevels:
        """Get exit levels for ICT setups."""
        atr = market_context.indicators.get('atr', 1.0) if market_context.indicators else 1.0
        current_price = market_context.current_price

        if side == 'long':
            hard_sl = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 2.0)
        else:
            hard_sl = current_price + (atr * 2.0)
            take_profit = current_price - (atr * 2.0)

        return ExitLevels(
            targets=[{
                'level': take_profit,
                'qty_pct': 100,
                'rr': abs(take_profit - current_price) / abs(current_price - hard_sl)
            }],
            hard_sl=hard_sl
        )

    def rank_setup_quality(self, market_context: MarketContext) -> float:
        """Rank the quality of ICT setups."""
        analysis = self.detect(market_context)
        return analysis.quality_score

    def validate_timing(self, market_context: MarketContext) -> bool:
        """Validate timing for ICT setups."""
        # ICT setups are valid during regular trading hours
        current_time = market_context.timestamp.time()
        return time(9, 15) <= current_time <= time(15, 15)