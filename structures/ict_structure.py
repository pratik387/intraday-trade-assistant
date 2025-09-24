#!/usr/bin/env python3
"""
ICT (Inner Circle Trader) Structure Implementation

Comprehensive structure covering all ICT concepts:
- Order Blocks: Institutional accumulation/distribution zones
- Fair Value Gaps: Price imbalances indicating institutional activity
- Liquidity Sweeps: Stop hunt patterns before institutional moves
- Premium/Discount Zones: Market structure positioning
- Break of Structure: Trend change confirmations
- Change of Character: Momentum shift detection
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from .base_structure import BaseStructure
from .data_models import StructureEvent, TradePlan, RiskParams, ExitLevels, MarketContext, StructureAnalysis

from config.logging_config import get_agent_logger
logger = get_agent_logger()

class ICTStructure(BaseStructure):
    """Comprehensive ICT structure covering all Inner Circle Trader concepts."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Order Blocks parameters
        self.ob_min_move_pct = config.get("order_block_min_move_pct", 0.5) / 100.0
        self.ob_min_volume_surge = config.get("order_block_min_volume_surge", 1.5)
        self.ob_lookback_bars = config.get("order_block_lookback", 20) or 20
        self.ob_test_tolerance = config.get("order_block_test_tolerance_pct", 0.15) / 100.0

        # Fair Value Gap parameters
        self.fvg_min_gap_pct = config.get("fvg_min_gap_pct", 0.1) / 100.0
        self.fvg_max_gap_pct = config.get("fvg_max_gap_pct", 1.5) / 100.0
        self.fvg_require_volume = config.get("fvg_require_volume_spike", True)
        self.fvg_min_volume_mult = config.get("fvg_min_volume_mult", 1.5)
        self.fvg_fill_tolerance = config.get("fvg_fill_tolerance_pct", 0.05) / 100.0

        # Liquidity Sweep parameters
        self.sweep_min_distance_pct = config.get("sweep_min_distance_pct", 0.05) / 100.0
        self.sweep_max_distance_pct = config.get("sweep_max_distance_pct", 0.3) / 100.0
        self.sweep_min_volume_surge = config.get("sweep_min_volume_surge", 2.0)
        self.sweep_min_wick_ratio = config.get("sweep_min_wick_ratio", 0.4)
        self.sweep_reversal_bars = config.get("sweep_reversal_bars", 3) or 3

        # Premium/Discount parameters
        self.premium_threshold = config.get("premium_threshold_pct", 70.0) / 100.0
        self.discount_threshold = config.get("discount_threshold_pct", 30.0) / 100.0
        self.range_lookback_bars = config.get("range_lookback_bars", 50) or 50

        # Break of Structure parameters
        self.bos_min_structure_bars = config.get("bos_min_structure_bars", 10) or 10
        self.bos_min_break_pct = config.get("bos_min_break_pct", 0.2) / 100.0
        self.bos_volume_confirmation = config.get("bos_volume_confirmation", True)

        # Change of Character parameters
        self.choch_momentum_periods = config.get("choch_momentum_periods", [3, 5, 8]) or [3, 5, 8]
        self.choch_min_momentum_change = config.get("choch_min_momentum_change_pct", 1.0) / 100.0
        self.choch_volume_threshold = config.get("choch_volume_threshold", 1.8)

        # Risk and exit parameters
        self.risk_pct = config.get("risk_pct", 1.0) / 100.0
        self.reward_risk_ratio = config.get("reward_risk_ratio", 2.0)
        self.max_bars_hold = config.get("max_bars_hold", 12) or 12

        logger.info(f"ICT structure initialized - OB move: {self.ob_min_move_pct*100:.1f}%, "
                   f"FVG gap: {self.fvg_min_gap_pct*100:.2f}%-{self.fvg_max_gap_pct*100:.1f}%, "
                   f"Sweep vol: {self.sweep_min_volume_surge}x")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect all ICT patterns and return comprehensive analysis."""
        logger.debug(f"ICT_DETECTOR: Starting detection for {context.symbol}")
        events = []

        try:
            df = context.df_5m
            if df is None or len(df) < 20:
                logger.debug(f"ICT_DETECTOR: {context.symbol} insufficient data (len={len(df) if df is not None else 0})")
                return StructureAnalysis(structure_detected=False, events=[], quality_score=0.0)

            # Prepare data
            d = df.copy()
            d = self._add_volume_indicators(d)

            # Define key levels for liquidity sweeps
            levels = {
                "PDH": context.pdh,
                "PDL": context.pdl,
                "ORH": context.orh,
                "ORL": context.orl
            }

            # Detect all ICT patterns
            logger.debug(f"ICT_DETECTOR: {context.symbol} running pattern detection")
            order_block_events = self._detect_order_blocks(d, context)
            fvg_events = self._detect_fair_value_gaps(d, context)
            sweep_events = self._detect_liquidity_sweeps(d, context, levels)
            premium_discount_events = self._detect_premium_discount_zones(d, context, levels)
            bos_events = self._detect_break_of_structure(d, context)
            choch_events = self._detect_change_of_character(d, context)

            logger.info(f"ICT_DETECTOR: {context.symbol} pattern counts - OB:{len(order_block_events)} FVG:{len(fvg_events)} "
                       f"Sweep:{len(sweep_events)} P/D:{len(premium_discount_events)} BOS:{len(bos_events)} CHOCH:{len(choch_events)}")

            # Combine all events
            all_events = (order_block_events + fvg_events + sweep_events +
                         premium_discount_events + bos_events + choch_events)

            # Calculate quality score based on multiple confirmations
            quality_score = self._calculate_ict_quality_score(all_events, d, context)
            structure_detected = len(all_events) > 0

            return StructureAnalysis(structure_detected=structure_detected, events=all_events, quality_score=quality_score)

        except Exception as e:
            logger.error(f"ICT detection error: {e}")
            return StructureAnalysis(structure_detected=False, events=[], quality_score=0.0)

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis indicators."""
        d = df.copy()

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

        return d

    def _detect_order_blocks(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Order Blocks - institutional accumulation/distribution zones."""
        events = []

        try:
            current_price = context.current_price
            current_bar_idx = len(df) - 1

            # Look for significant moves in recent history
            search_start = max(5, current_bar_idx - self.ob_lookback_bars)

            for move_start_idx in range(search_start, current_bar_idx - 2):
                # Check for significant move
                move_bars = df.iloc[move_start_idx:move_start_idx + 5]
                if len(move_bars) < 3:
                    continue

                move_start_price = move_bars['close'].iloc[0]
                move_end_price = move_bars['close'].iloc[-1]
                move_pct = (move_end_price - move_start_price) / move_start_price

                # Check volume confirmation
                move_had_volume = (move_bars['vol_surge'] > self.ob_min_volume_surge).any()

                if abs(move_pct) >= self.ob_min_move_pct and move_had_volume:
                    # Find last opposing candle before move
                    ob_candle_idx = self._find_opposing_candle(df, move_start_idx, move_pct)

                    if ob_candle_idx is not None:
                        event = self._create_order_block_event(df, ob_candle_idx, move_pct,
                                                             current_price, current_bar_idx, context)
                        if event:
                            events.append(event)

        except Exception as e:
            logger.error(f"Order block detection error: {e}")

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
                                context: MarketContext) -> Optional[StructureEvent]:
        """Create order block event if current price is testing the zone."""
        ob_candle = df.iloc[ob_candle_idx]
        ob_high = ob_candle['high']
        ob_low = ob_candle['low']

        if move_pct > 0:  # Bearish OB (resistance zone)
            if (ob_low <= current_price <= ob_high * (1 + self.ob_test_tolerance)):
                bars_since_ob = current_bar_idx - ob_candle_idx
                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))
                strength = min(3.0, abs(move_pct) * 100 * time_decay)

                trade_plan = self._create_order_block_trade_plan(
                    'short', ob_high, ob_low, context.current_price, context.indicators.get('atr', 1.0)
                )

                return StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type='order_block_short',
                    side='short',
                    confidence=self._calculate_institutional_strength(context, strength, "order_block", "short", move_pct, bars_since_ob),
                    levels={'entry': ob_high, 'stop': ob_high + (context.indicators.get('atr', 1.0) * 1.5), 'target': ob_high - (context.indicators.get('atr', 1.0) * 2.0)},
                    context={
                        'ob_high': ob_high,
                        'ob_low': ob_low,
                        'move_pct': move_pct * 100,
                        'bars_since_formation': bars_since_ob,
                        'pattern_type': 'bearish_order_block'
                    },
                    price=context.current_price,
                    volume=None,
                    indicators=context.indicators
                )
        else:  # Bullish OB (support zone)
            if (ob_low * (1 - self.ob_test_tolerance) <= current_price <= ob_high):
                bars_since_ob = current_bar_idx - ob_candle_idx
                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))
                strength = min(3.0, abs(move_pct) * 100 * time_decay)

                trade_plan = self._create_order_block_trade_plan(
                    'long', ob_high, ob_low, context.current_price, context.indicators.get('atr', 1.0)
                )

                return StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type='order_block_long',
                    side='long',
                    confidence=self._calculate_institutional_strength(context, strength, "order_block", "long", move_pct, bars_since_ob),
                    levels={'entry': ob_low, 'stop': ob_low - (context.indicators.get('atr', 1.0) * 1.5), 'target': ob_low + (context.indicators.get('atr', 1.0) * 2.0)},
                    context={
                        'ob_high': ob_high,
                        'ob_low': ob_low,
                        'move_pct': move_pct * 100,
                        'bars_since_formation': bars_since_ob,
                        'pattern_type': 'bullish_order_block'
                    },
                    price=ob_low
                )

        return None

    def _detect_fair_value_gaps(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Fair Value Gaps - price imbalances indicating institutional activity."""
        events = []

        try:
            current_price = context.current_price
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
                                                 df, i, current_price, context)
                    if event:
                        events.append(event)

                # Bearish FVG
                elif (candle_before['low'] > candle_after['high'] and
                      candle_middle['close'] < candle_middle['open']):

                    event = self._create_fvg_event('short', candle_before, candle_after,
                                                 df, i, current_price, context)
                    if event:
                        events.append(event)

        except Exception as e:
            logger.error(f"FVG detection error: {e}")

        return events

    def _create_fvg_event(self, direction: str, candle_before: pd.Series, candle_after: pd.Series,
                         df: pd.DataFrame, gap_index: int, current_price: float,
                         context: MarketContext) -> Optional[StructureEvent]:
        """Create Fair Value Gap event if conditions are met."""
        if direction == 'long':
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
        if not (fvg_bottom * (1 - self.fvg_fill_tolerance) <= current_price <=
                fvg_top * (1 + self.fvg_fill_tolerance)):
            return None

        # Calculate strength
        volume_strength = df['vol_surge'].iloc[gap_index]
        strength = min(3.0, gap_pct * 500 + volume_strength * 0.5)

        trade_plan = self._create_fvg_trade_plan(direction, fvg_top, fvg_bottom,
                                               current_price, context.indicators.get('atr', 1.0))

        return StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type=f'fair_value_gap_{direction}',
            side=direction,
            confidence=self._calculate_institutional_strength(context, strength, "fair_value_gap", direction, gap_pct, 0),
            levels={'entry': fvg_bottom if direction == 'long' else fvg_top,
                   'support': fvg_bottom, 'resistance': fvg_top},
            context={
                'fvg_top': fvg_top,
                'fvg_bottom': fvg_bottom,
                'gap_size_pct': gap_pct * 100,
                'volume_surge': volume_strength,
                'pattern_type': f'{direction}_fair_value_gap'
            },
            price=fvg_bottom if direction == 'long' else fvg_top
        )

    def _detect_liquidity_sweeps(self, df: pd.DataFrame, context: MarketContext,
                               levels: Dict[str, float]) -> List[StructureEvent]:
        """Detect Liquidity Sweeps - stop hunt patterns."""
        events = []

        try:
            current_price = context.current_price
            lookback_bars = min(10, len(df) - self.sweep_reversal_bars)

            for level_name, level_price in levels.items():
                if level_price <= 0 or not np.isfinite(level_price):
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
                                                      current_price, context)
                    if event:
                        events.append(event)

        except Exception as e:
            logger.error(f"Liquidity sweep detection error: {e}")

        return events

    def _check_liquidity_sweep(self, df: pd.DataFrame, sweep_idx: int, level_name: str,
                             level_price: float, upper_wick_ratio: float, lower_wick_ratio: float,
                             current_price: float, context: MarketContext) -> Optional[StructureEvent]:
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
                    trade_plan = self._create_sweep_trade_plan('long', level_price, sweep_bar['low'],
                                                             current_price, context.indicators.get('atr', 1.0))

                    return StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type='liquidity_sweep_long',
                        side='long',
                        confidence=self._calculate_institutional_strength(context, sweep_pct * 200 + lower_wick_ratio * 2, "liquidity_sweep", "long", sweep_pct, 0),
                        levels={'entry': level_price, 'sweep_level': level_price, 'sweep_low': sweep_bar['low']},
                        context={
                            'level_name': level_name,
                            'level_price': level_price,
                            'sweep_low': sweep_bar['low'],
                            'sweep_distance_pct': sweep_pct * 100,
                            'wick_ratio': lower_wick_ratio,
                            'pattern_type': 'bullish_liquidity_sweep'
                        },
                        price=level_price
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
                    trade_plan = self._create_sweep_trade_plan('short', level_price, sweep_bar['high'],
                                                             current_price, context.indicators.get('atr', 1.0))

                    return StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type='liquidity_sweep_short',
                        side='short',
                        confidence=self._calculate_institutional_strength(context, sweep_pct * 200 + upper_wick_ratio * 2, "liquidity_sweep", "short", sweep_pct, 0),
                        levels={'entry': level_price, 'sweep_level': level_price, 'sweep_high': sweep_bar['high']},
                        context={
                            'level_name': level_name,
                            'level_price': level_price,
                            'sweep_high': sweep_bar['high'],
                            'sweep_distance_pct': sweep_pct * 100,
                            'wick_ratio': upper_wick_ratio,
                            'pattern_type': 'bearish_liquidity_sweep'
                        },
                        price=level_price
                    )

        return None

    def _check_sweep_reversal(self, df: pd.DataFrame, sweep_idx: int, level_price: float, direction: str) -> bool:
        """Check for quick reversal confirmation after liquidity sweep."""
        reversal_bars = df.iloc[sweep_idx+1:sweep_idx+1+self.sweep_reversal_bars]
        if len(reversal_bars) == 0:
            return False

        if direction == 'long':
            # Price should stay above level and move higher
            above_level = (reversal_bars['low'] > level_price * 0.998).all()
            moved_higher = reversal_bars['close'].iloc[-1] > df.iloc[sweep_idx]['close']
            return above_level and moved_higher
        else:
            # Price should stay below level and move lower
            below_level = (reversal_bars['high'] < level_price * 1.002).all()
            moved_lower = reversal_bars['close'].iloc[-1] < df.iloc[sweep_idx]['close']
            return below_level and moved_lower

    def _detect_premium_discount_zones(self, df: pd.DataFrame, context: MarketContext,
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

            current_price = context.current_price

            # Calculate position in range (0 = bottom, 1 = top)
            range_position = (current_price - range_low) / range_size

            # Premium zone (top 30% of range)
            if range_position >= self.premium_threshold:
                trade_plan = self._create_premium_discount_trade_plan('short', range_high, range_low,
                                                                    current_price, context.indicators.get('atr', 1.0))

                events.append(StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type='premium_zone_short',
                    side='short',
                    confidence=self._calculate_institutional_strength(context, (range_position - self.premium_threshold) * 10, "premium_zone", "short", range_position, 0),
                    levels={'entry': current_price, 'range_high': range_high, 'range_low': range_low},
                    context={
                        'range_high': range_high,
                        'range_low': range_low,
                        'range_position_pct': range_position * 100,
                        'zone_type': 'premium',
                        'pattern_type': 'premium_zone_positioning'
                    },
                    price=current_price
                ))

            # Discount zone (bottom 30% of range)
            elif range_position <= self.discount_threshold:
                trade_plan = self._create_premium_discount_trade_plan('long', range_high, range_low,
                                                                    current_price, context.indicators.get('atr', 1.0))

                events.append(StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type='discount_zone_long',
                    side='long',
                    confidence=self._calculate_institutional_strength(context, (self.discount_threshold - range_position) * 10, "discount_zone", "long", range_position, 0),
                    levels={'entry': current_price, 'range_high': range_high, 'range_low': range_low},
                    context={
                        'range_high': range_high,
                        'range_low': range_low,
                        'range_position_pct': range_position * 100,
                        'zone_type': 'discount',
                        'pattern_type': 'discount_zone_positioning'
                    },
                    price=current_price
                ))

        except Exception as e:
            logger.error(f"Premium/Discount zone detection error: {e}")

        return events

    def _detect_break_of_structure(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Break of Structure - trend change confirmations."""
        events = []

        try:
            if len(df) < self.bos_min_structure_bars + 5:
                return events

            current_price = context.current_price

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
                        trade_plan = self._create_bos_trade_plan('long', recent_high, current_price,
                                                               context.indicators.get('atr', 1.0))

                        events.append(StructureEvent(
                            symbol=context.symbol,
                            timestamp=context.timestamp,
                            structure_type='break_of_structure_long',
                            side='long',
                            confidence=self._calculate_institutional_strength(context, break_pct * 200, "break_of_structure", "long", break_pct, 0),
                            levels={'entry': current_price, 'broken_level': recent_high},
                            context={
                                'broken_level': recent_high,
                                'break_distance_pct': break_pct * 100,
                                'structure_type': 'swing_high',
                                'pattern_type': 'bullish_break_of_structure'
                            },
                            price=recent_high
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
                        trade_plan = self._create_bos_trade_plan('short', recent_low, current_price,
                                                               context.indicators.get('atr', 1.0))

                        events.append(StructureEvent(
                            symbol=context.symbol,
                            timestamp=context.timestamp,
                            structure_type='break_of_structure_short',
                            side='short',
                            confidence=self._calculate_institutional_strength(context, break_pct * 200, "break_of_structure", "short", break_pct, 0),
                            levels={'entry': current_price, 'broken_level': recent_low},
                            context={
                                'broken_level': recent_low,
                                'break_distance_pct': break_pct * 100,
                                'structure_type': 'swing_low',
                                'pattern_type': 'bearish_break_of_structure'
                            },
                            price=recent_low
                        ))

        except Exception as e:
            logger.error(f"Break of Structure detection error: {e}")

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

    def _detect_change_of_character(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Change of Character - momentum shift detection."""
        events = []

        try:
            if len(df) < max(self.choch_momentum_periods) + 5:
                return events

            current_price = context.current_price

            # Calculate momentum changes for different periods
            momentum_changes = {}
            for period in self.choch_momentum_periods:
                if len(df) >= period + 1:
                    old_momentum = df['returns_3'].iloc[-(period+1)]
                    current_momentum = df['returns_3'].iloc[-1]
                    momentum_change = current_momentum - old_momentum
                    momentum_changes[period] = momentum_change

            # Check for significant momentum shift
            significant_changes = [abs(change) >= self.choch_min_momentum_change
                                 for change in momentum_changes.values()]

            if any(significant_changes):
                # Determine direction of character change
                avg_momentum_change = np.mean(list(momentum_changes.values()))
                direction = 'long' if avg_momentum_change > 0 else 'short'

                # Check volume confirmation
                recent_vol_z = df['vol_z'].tail(3).max()
                if recent_vol_z >= self.choch_volume_threshold:
                    trade_plan = self._create_choch_trade_plan(direction, current_price,
                                                             context.indicators.get('atr', 1.0))

                    events.append(StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type=f'change_of_character_{direction}',
                        side=direction,
                        confidence=self._calculate_institutional_strength(context, abs(avg_momentum_change) * 100, "change_of_character", direction, abs(avg_momentum_change), 0),
                        levels={'entry': current_price, 'momentum_shift': current_price},
                        context={
                            'momentum_changes': momentum_changes,
                            'avg_momentum_change_pct': avg_momentum_change * 100,
                            'volume_confirmation': recent_vol_z,
                            'pattern_type': f'{direction}_change_of_character'
                        },
                        price=current_price
                    ))

        except Exception as e:
            logger.error(f"Change of Character detection error: {e}")

        return events

    def _calculate_ict_quality_score(self, events: List[StructureEvent], df: pd.DataFrame,
                                   context: MarketContext) -> float:
        """Calculate quality score based on multiple ICT confirmations."""
        if not events:
            return 0.0

        try:
            score = 0.0

            # Base score from number of confirmations
            num_events = len(events)
            score += min(2.0, num_events * 0.5)

            # Volume confirmation
            recent_vol_z = df['vol_z'].tail(3).max()
            if recent_vol_z >= 2.0:
                score += 1.0
            elif recent_vol_z >= 1.5:
                score += 0.5

            # Pattern diversity bonus
            pattern_types = set(event.structure_type.split('_')[0] for event in events)
            if len(pattern_types) >= 3:
                score += 1.0
            elif len(pattern_types) >= 2:
                score += 0.5

            # Strength-based adjustment
            avg_strength = np.mean([event.confidence for event in events])
            score += min(1.0, avg_strength / 3.0)

            return min(5.0, score)

        except Exception as e:
            logger.error(f"ICT quality score calculation error: {e}")
            return 0.0

    def _create_order_block_trade_plan(self, direction: str, ob_high: float, ob_low: float,
                                     current_price: float, atr: float) -> TradePlan:
        """Create trade plan for Order Block setup."""
        if direction == 'long':
            entry_price = ob_low
            stop_loss = ob_low - (atr * 1.5)
            take_profit = entry_price + ((entry_price - stop_loss) * self.reward_risk_ratio)
        else:
            entry_price = ob_high
            stop_loss = ob_high + (atr * 1.5)
            take_profit = entry_price - ((stop_loss - entry_price) * self.reward_risk_ratio)

        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

        exit_levels = ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

        return TradePlan(
            symbol="",
            side=direction,
            structure_type="order_block",
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=100,
            notional=100.0 * entry_price,
            confidence=0.7,
            notes=f"Order Block {direction} at {entry_price:.2f}"
        )

    def _create_fvg_trade_plan(self, direction: str, fvg_top: float, fvg_bottom: float,
                             current_price: float, atr: float) -> TradePlan:
        """Create trade plan for Fair Value Gap setup."""
        if direction == 'long':
            entry_price = fvg_bottom
            stop_loss = fvg_bottom - (atr * 1.0)
            take_profit = fvg_top + (atr * 1.5)
        else:
            entry_price = fvg_top
            stop_loss = fvg_top + (atr * 1.0)
            take_profit = fvg_bottom - (atr * 1.5)

        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

        exit_levels = ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

        return TradePlan(
            symbol="",
            side=direction,
            structure_type="fvg",
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=100,
            notional=100.0 * entry_price,
            confidence=0.7,
            notes=f"Fair Value Gap {direction} fill at {entry_price:.2f}"
        )

    def _create_sweep_trade_plan(self, direction: str, level_price: float, sweep_extreme: float,
                               current_price: float, atr: float) -> TradePlan:
        """Create trade plan for Liquidity Sweep setup."""
        if direction == 'long':
            entry_price = level_price
            stop_loss = sweep_extreme - (atr * 0.5)
            take_profit = entry_price + ((entry_price - stop_loss) * self.reward_risk_ratio)
        else:
            entry_price = level_price
            stop_loss = sweep_extreme + (atr * 0.5)
            take_profit = entry_price - ((stop_loss - entry_price) * self.reward_risk_ratio)

        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

        exit_levels = ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

        return TradePlan(
            symbol="",
            side=direction,
            structure_type="fvg",
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=100,
            notional=100.0 * entry_price,
            confidence=0.7,
            notes=f"Liquidity Sweep {direction} reversal at {entry_price:.2f}"
        )

    def _create_premium_discount_trade_plan(self, direction: str, range_high: float, range_low: float,
                                          current_price: float, atr: float) -> TradePlan:
        """Create trade plan for Premium/Discount zone setup."""
        if direction == 'long':
            entry_price = current_price
            stop_loss = range_low - (atr * 0.5)
            take_profit = range_high
        else:
            entry_price = current_price
            stop_loss = range_high + (atr * 0.5)
            take_profit = range_low

        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

        exit_levels = ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

        return TradePlan(
            symbol="",
            side=direction,
            structure_type="fvg",
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=100,
            notional=100.0 * entry_price,
            confidence=0.7,
            notes=f"{'Discount' if direction == 'long' else 'Premium'} zone {direction} at {entry_price:.2f}"
        )

    def _create_bos_trade_plan(self, direction: str, broken_level: float, current_price: float,
                             atr: float) -> TradePlan:
        """Create trade plan for Break of Structure setup."""
        if direction == 'long':
            entry_price = current_price
            stop_loss = broken_level - (atr * 0.5)
            take_profit = current_price + ((current_price - stop_loss) * self.reward_risk_ratio)
        else:
            entry_price = current_price
            stop_loss = broken_level + (atr * 0.5)
            take_profit = current_price - ((stop_loss - current_price) * self.reward_risk_ratio)

        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

        exit_levels = ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

        return TradePlan(
            symbol="",
            side=direction,
            structure_type="fvg",
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=100,
            notional=100.0 * entry_price,
            confidence=0.7,
            notes=f"Break of Structure {direction} continuation at {entry_price:.2f}"
        )

    def _create_choch_trade_plan(self, direction: str, current_price: float, atr: float) -> TradePlan:
        """Create trade plan for Change of Character setup."""
        if direction == 'long':
            entry_price = current_price
            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + ((current_price - stop_loss) * self.reward_risk_ratio)
        else:
            entry_price = current_price
            stop_loss = current_price + (atr * 2.0)
            take_profit = current_price - ((stop_loss - current_price) * self.reward_risk_ratio)

        risk_params = RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

        exit_levels = ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

        return TradePlan(
            symbol="",
            side=direction,
            structure_type="fvg",
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=100,
            notional=100.0 * entry_price,
            confidence=0.7,
            notes=f"Change of Character {direction} momentum shift at {entry_price:.2f}"
        )

    def _calculate_institutional_strength(self, context: MarketContext, base_strength: float,
                                        pattern_type: str, side: str, pattern_value: float,
                                        time_factor: int = 0) -> float:
        """Calculate institutional-grade strength for ICT patterns."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from pattern quality and volume (institutional volume threshold ≥1.5)
            institutional_base = max(1.0, vol_z * (base_strength / 100.0) * 2.0)

            # Professional bonuses for institutional-grade ICT patterns
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.2  # 20% bonus for volume surge
                logger.debug(f"ICT: Volume surge bonus applied (vol_z={vol_z:.2f})")

            if vol_z >= 2.0:  # Strong institutional volume
                strength_multiplier *= 1.3  # Additional 30% bonus
                logger.debug(f"ICT: Strong volume surge bonus applied")

            # Pattern-specific bonuses
            if pattern_type == "order_block":
                # Time decay bonus (fresher order blocks are stronger)
                if time_factor <= 10:  # Recent formation
                    strength_multiplier *= 1.25  # 25% bonus for fresh OB
                    logger.debug(f"ICT: Fresh order block bonus applied ({time_factor} bars)")

                # Significant move bonus
                if abs(pattern_value) >= 0.015:  # 1.5%+ move
                    strength_multiplier *= 1.2  # 20% bonus for significant move

            elif pattern_type == "fair_value_gap":
                # Optimal gap size bonus (not too small, not too large)
                if 0.002 <= abs(pattern_value) <= 0.01:  # 0.2%-1.0% gap
                    strength_multiplier *= 1.2  # 20% bonus for optimal gap
                    logger.debug(f"ICT: Optimal FVG size bonus applied ({pattern_value*100:.2f}%)")

            elif pattern_type == "liquidity_sweep":
                # Clean sweep bonus (good wick ratio + distance)
                if 0.001 <= abs(pattern_value) <= 0.005:  # Clean sweep range
                    strength_multiplier *= 1.25  # 25% bonus for clean sweep
                    logger.debug(f"ICT: Clean liquidity sweep bonus applied")

            elif pattern_type in ["premium_zone", "discount_zone"]:
                # Extreme positioning bonus (closer to range boundaries)
                if abs(pattern_value) >= 0.25:  # Extreme positioning
                    strength_multiplier *= 1.15  # 15% bonus for extreme positioning
                    logger.debug(f"ICT: Extreme zone positioning bonus applied")

            elif pattern_type == "break_of_structure":
                # Significant break bonus
                if abs(pattern_value) >= 0.003:  # 0.3%+ break
                    strength_multiplier *= 1.2  # 20% bonus for significant break
                    logger.debug(f"ICT: Significant BOS bonus applied ({pattern_value*100:.2f}%)")

            elif pattern_type == "change_of_character":
                # Strong momentum shift bonus
                if abs(pattern_value) >= 0.02:  # 2%+ momentum change
                    strength_multiplier *= 1.25  # 25% bonus for strong momentum shift
                    logger.debug(f"ICT: Strong CHOCH bonus applied ({pattern_value*100:.2f}%)")

            # Market timing bonus (ICT patterns work best during active hours)
            current_hour = pd.to_datetime(context.timestamp).hour
            if 10 <= current_hour <= 14:  # Main trading session
                strength_multiplier *= 1.1  # 10% timing bonus
                logger.debug(f"ICT: Market timing bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = institutional_base * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.5)  # Minimum viable strength

            logger.debug(f"ICT: {context.symbol} {side} {pattern_type} - Base: {institutional_base:.2f}, "
                        f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.error(f"ICT: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold

    # Abstract method implementations for BaseStructure compatibility

    def plan_long_strategy(self, context: MarketContext) -> Optional[TradePlan]:
        """Plan long strategy based on ICT analysis."""
        analysis = self.detect(context)
        long_events = [e for e in analysis.events if e.direction == 'long']
        if long_events:
            return long_events[0].trade_plan
        return None

    def plan_short_strategy(self, context: MarketContext) -> Optional[TradePlan]:
        """Plan short strategy based on ICT analysis."""
        analysis = self.detect(context)
        short_events = [e for e in analysis.events if e.direction == 'short']
        if short_events:
            return short_events[0].trade_plan
        return None

    def calculate_risk_params(self, context: MarketContext, direction: str) -> RiskParams:
        """Calculate risk parameters for ICT setups."""
        atr = context.indicators.get('atr', 1.0)

        if direction == 'long':
            stop_loss = context.current_price - (atr * 2.0)
        else:
            stop_loss = context.current_price + (atr * 2.0)

        return RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=atr * 0.01
        )

    def get_exit_levels(self, context: MarketContext, direction: str) -> ExitLevels:
        """Get exit levels for ICT setups."""
        atr = context.indicators.get('atr', 1.0)

        if direction == 'long':
            stop_loss = context.current_price - (atr * 2.0)
            take_profit = context.current_price + (atr * self.reward_risk_ratio * 2.0)
        else:
            stop_loss = context.current_price + (atr * 2.0)
            take_profit = context.current_price - (atr * self.reward_risk_ratio * 2.0)

        return ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss,
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext) -> float:
        """Rank the quality of ICT setups."""
        analysis = self.detect(context)
        return analysis.quality_score

    def validate_timing(self, context: MarketContext) -> bool:
        """Validate timing for ICT setups."""
        # ICT setups are valid during regular trading hours
        current_hour = context.timestamp.hour
        return 9 <= current_hour <= 15  # 9:00 AM to 3:00 PM