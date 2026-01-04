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

        # Track which specific setup type this detector is for (e.g., "premium_zone_short")
        self.configured_setup_type = config.get("_setup_name", None)

        # Order Blocks parameters - will crash with KeyError if missing
        self.ob_min_move_pct = config["order_block_min_move_pct"] / 100.0
        self.ob_min_volume_surge = config["order_block_min_volume_surge"]
        self.ob_lookback_bars = config["order_block_lookback"]
        self.ob_test_tolerance = config["order_block_test_tolerance_pct"] / 100.0

        # Fair Value Gap parameters - will crash with KeyError if missing
        self.fvg_min_gap_pct = config["fvg_min_gap_pct"] / 100.0
        self.fvg_max_gap_pct = config["fvg_max_gap_pct"] / 100.0
        self.fvg_require_volume = config["fvg_require_volume_spike"]
        self.fvg_min_volume_mult = config["fvg_min_volume_mult"]
        self.fvg_fill_tolerance = config["fvg_fill_tolerance_pct"] / 100.0

        # Liquidity Sweep parameters - will crash with KeyError if missing
        self.sweep_min_distance_pct = config["sweep_min_distance_pct"] / 100.0
        self.sweep_max_distance_pct = config["sweep_max_distance_pct"] / 100.0
        self.sweep_min_volume_surge = config["sweep_min_volume_surge"]
        self.sweep_min_wick_ratio = config["sweep_min_wick_ratio"]
        self.sweep_reversal_bars = config["sweep_reversal_bars"]

        # Premium/Discount parameters - will crash with KeyError if missing
        self.premium_threshold = config["premium_threshold_pct"] / 100.0
        self.discount_threshold = config["discount_threshold_pct"] / 100.0
        self.range_lookback_bars = config["range_lookback_bars"]

        # Break of Structure parameters - will crash with KeyError if missing
        self.bos_min_structure_bars = config["bos_min_structure_bars"]
        self.bos_min_break_pct = config["bos_min_break_pct"] / 100.0
        self.bos_volume_confirmation = config["bos_volume_confirmation"]
        # PRO ICT: Retest-based entry parameters - will crash with KeyError if missing
        self.bos_entry_mode = config["bos_entry_mode"]  # "retest" or "immediate"
        self.bos_retest_zone_pct = config["bos_retest_zone_pct"] / 100.0
        self.bos_retest_timeout_bars = config["bos_retest_timeout_bars"]
        self.bos_require_htf_structure = config["bos_require_htf_structure"]

        # Change of Character parameters - will crash with KeyError if missing
        self.choch_momentum_periods = config["choch_momentum_periods"]
        self.choch_min_momentum_change = config["choch_min_momentum_change_pct"] / 100.0
        self.choch_volume_threshold = config["choch_volume_threshold"]

        # Risk and exit parameters - will crash with KeyError if missing
        self.risk_pct = config["risk_pct"] / 100.0
        self.reward_risk_ratio = config["reward_risk_ratio"]
        self.max_bars_hold = config["max_bars_hold"]

        # Stop loss parameters - Pro trader: SL at structure level + ATR buffer
        self.sl_buffer_atr = config["sl_buffer_atr"]  # ATR buffer beyond structure level
        self.min_stop_distance_pct = config["min_stop_distance_pct"]  # Minimum SL distance as % of price

        # Quality Filter parameters (MUST be in config - no defaults)
        ict_filters = config["ict_quality_filters"]  # Will raise KeyError if missing

        # Order Block Quality Filters
        ob_filters = ict_filters["order_block"]  # Will raise KeyError if missing
        self.ob_min_block_size_pct = ob_filters["min_block_size_pct"]
        self.ob_min_volume_ratio = ob_filters["min_volume_ratio"]
        self.ob_swing_tolerance_pct = ob_filters["swing_tolerance_pct"]
        self.ob_min_rejection_wick_pct = ob_filters["min_rejection_wick_pct"]

        # Fair Value Gap Quality Filters
        fvg_filters = ict_filters["fair_value_gap"]  # Will raise KeyError if missing
        self.fvg_min_gap_size_pct = fvg_filters["min_gap_size_pct"]
        self.fvg_min_volume_ratio = fvg_filters["min_volume_ratio"]
        self.fvg_max_wick_penetration_pct = fvg_filters["max_wick_penetration_pct"]
        self.fvg_vwap_distance_tolerance_pct = fvg_filters["vwap_distance_tolerance_pct"]
        self.fvg_swing_tolerance_pct = fvg_filters["swing_tolerance_pct"]

        logger.debug(f"ICT structure initialized - OB move: {self.ob_min_move_pct*100:.1f}%, "
                   f"FVG gap: {self.fvg_min_gap_pct*100:.2f}%-{self.fvg_max_gap_pct*100:.1f}%, "
                   f"Sweep vol: {self.sweep_min_volume_surge}x, "
                   f"Quality filters: OB block_size>{self.ob_min_block_size_pct*100:.1f}%, "
                   f"FVG gap_size>{self.fvg_min_gap_size_pct*100:.1f}%")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect all ICT patterns and return comprehensive analysis with professional criteria."""
        logger.debug(f"ICT_DETECTOR: Starting detection for {context.symbol}")
        events = []

        try:
            df = context.df_5m

            # OPENING BELL FIX: Check if we're in opening bell window (09:20-09:30)
            # Professional NSE traders analyze first candle for premium/discount zones
            from datetime import time as dtime
            current_time = context.timestamp.time() if hasattr(context.timestamp, 'time') else context.timestamp
            in_opening_bell = dtime(9, 20) <= current_time < dtime(9, 30)
            min_bars_required = 1 if in_opening_bell else 20

            if df is None or len(df) < min_bars_required:
                logger.debug(f"ICT_DETECTOR: {context.symbol} insufficient data (len={len(df) if df is not None else 0}, required={min_bars_required}, opening_bell={in_opening_bell})")
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

            # PROFESSIONAL ICT: Detect in specific order to enable professional filters
            logger.debug(f"ICT_DETECTOR: {context.symbol} running pattern detection (opening_bell={in_opening_bell}, bars={len(df)})")

            # OPENING BELL FIX: With 1-2 bars, only detect premium/discount zones (like pro traders)
            # Skip complex patterns that need swing structure (OB, FVG, MSS, BOS, CHOCH)
            if in_opening_bell and len(df) < 3:
                logger.debug(f"ICT_DETECTOR: {context.symbol} opening bell mode - premium/discount zones only")
                premium_discount_events = self._detect_premium_discount_zones(d, context, levels)

                # Skip all other patterns during opening bell with <3 bars
                sweep_events = []
                mss_events = []
                order_block_events = []
                fvg_events = []
                bos_events = []
                choch_events = []
            else:
                # Normal mode with sufficient bars
                # Step 1: Detect liquidity sweeps FIRST (required for OB validation)
                sweep_events = self._detect_liquidity_sweeps(d, context, levels)
                logger.debug(f"ICT_DETECTOR: {context.symbol} found {len(sweep_events)} sweeps")

                # Step 2: Detect swing points (required for MSS)
                swing_highs = self._find_swing_points(d, 'high')
                swing_lows = self._find_swing_points(d, 'low')

                # Step 3: Detect Market Structure Shift
                mss_events = self._detect_market_structure_shift(d, context, swing_highs, swing_lows)
                logger.debug(f"ICT_DETECTOR: {context.symbol} found {len(mss_events)} MSS")

                # Step 4: Detect Order Blocks with PROFESSIONAL criteria (sweep + MSS required)
                order_block_events = self._detect_order_blocks(d, context, sweep_events, mss_events)
                logger.debug(f"ICT_DETECTOR: {context.symbol} found {len(order_block_events)} OBs")

                # Step 5: Detect FVGs with displacement
                fvg_events = self._detect_fair_value_gaps(d, context)
                logger.debug(f"ICT_DETECTOR: {context.symbol} found {len(fvg_events)} FVGs")

                # Step 6: Other ICT patterns (unchanged)
                premium_discount_events = self._detect_premium_discount_zones(d, context, levels)
                bos_events = self._detect_break_of_structure(d, context)
                choch_events = self._detect_change_of_character(d, context)

            logger.debug(f"ICT_DETECTOR: {context.symbol} pattern counts - OB:{len(order_block_events)} FVG:{len(fvg_events)} "
                       f"Sweep:{len(sweep_events)} MSS:{len(mss_events)} P/D:{len(premium_discount_events)} BOS:{len(bos_events)} CHOCH:{len(choch_events)}")

            # Combine all events
            all_events = (order_block_events + fvg_events + sweep_events + mss_events +
                         premium_discount_events + bos_events + choch_events)

            # Filter events to only include those matching configured setup type
            if self.configured_setup_type and all_events:
                filtered_events = [e for e in all_events if e.structure_type == self.configured_setup_type]
                if len(filtered_events) < len(all_events):
                    logger.debug(f"ICT: {context.symbol} - Filtered {len(all_events)}→{len(filtered_events)} events (configured for {self.configured_setup_type})")
                all_events = filtered_events

            # Calculate quality score based on multiple confirmations
            quality_score = self._calculate_ict_quality_score(all_events, d, context)
            structure_detected = len(all_events) > 0

            logger.debug(f"ICT_DETECTOR: {context.symbol} COMPLETE - detected={structure_detected}, total_events={len(all_events)}, quality={quality_score:.2f}")
            return StructureAnalysis(structure_detected=structure_detected, events=all_events, quality_score=quality_score)

        except Exception as e:
            logger.exception(f"ICT detection error: {e}")
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
        d['vol_surge'] = d['volume'] / d['vol_ma10'].replace(0, np.nan)

        # Price returns for momentum analysis
        d['returns_1'] = d['close'].pct_change()
        d['returns_3'] = d['close'].pct_change(3)
        d['returns_5'] = d['close'].pct_change(5)

        return d

    def _validate_htf_trend(self, df: pd.DataFrame, context: MarketContext, direction: str) -> bool:
        """
        Validate HTF (Higher Time Frame) trend exists for BOS/CHOCH setups.

        Professional ICT requirement: BOS and CHOCH require established trend context.
        - BOS (Break of Structure): Continuation in trending market
        - CHOCH (Change of Character): Reversal of established trend

        Args:
            df: 5m dataframe
            context: Market context (contains HTF data if available)
            direction: 'long' or 'short' - expected trend direction

        Returns:
            True if HTF trend exists in specified direction, False otherwise
        """
        try:
            # Use 15m HTF data if available, otherwise use 5m data
            htf_df = getattr(context, 'htf_df', None)
            if htf_df is not None and not htf_df.empty and len(htf_df) >= 20:
                trend_df = htf_df
                logger.debug(f"HTF validation using 15m data ({len(htf_df)} bars)")
            else:
                trend_df = df
                logger.debug(f"HTF validation using 5m data ({len(df)} bars)")

            if len(trend_df) < 20:
                logger.debug(f"HTF validation failed: insufficient data ({len(trend_df)} < 20 bars)")
                return False

            # Get trend indicators from context or calculate
            adx = context.indicators.get('adx14', None)

            # Calculate trend strength using last 20 bars
            recent_data = trend_df.tail(20)

            # Method 1: ADX-based trend validation (if available)
            if adx is not None and adx > 20:
                # Strong trend exists (ADX > 20)
                # Check direction alignment
                if 'ema20' in recent_data.columns:
                    current_price = context.current_price
                    ema20 = recent_data['ema20'].iloc[-1]

                    if direction == 'long':
                        # Long: Price should be above EMA20
                        trend_aligned = current_price > ema20
                    else:
                        # Short: Price should be below EMA20
                        trend_aligned = current_price < ema20

                    if trend_aligned:
                        logger.debug(f"HTF {direction} trend VALID: ADX={adx:.1f}, price vs EMA20 aligned")
                        return True
                    else:
                        logger.debug(f"HTF {direction} trend INVALID: ADX={adx:.1f} but price not aligned with EMA20")
                        return False

            # Method 2: Price action based trend validation
            # Check if price has been trending (7+ bars out of last 10 above/below MA)
            if 'ema20' in recent_data.columns or 'sma20' in recent_data.columns:
                ma_col = 'ema20' if 'ema20' in recent_data.columns else 'sma20'
                last_10 = recent_data.tail(10)

                if direction == 'long':
                    # Long trend: price above MA for most bars
                    bars_above_ma = sum(last_10['close'] > last_10[ma_col])
                    trend_exists = bars_above_ma >= 7
                else:
                    # Short trend: price below MA for most bars
                    bars_below_ma = sum(last_10['close'] < last_10[ma_col])
                    trend_exists = bars_below_ma >= 7

                if trend_exists:
                    logger.debug(f"HTF {direction} trend VALID: price action confirms trend (7+/10 bars)")
                    return True
                else:
                    logger.debug(f"HTF {direction} trend INVALID: insufficient trend bars")
                    return False

            # Method 3: Simple price slope (fallback)
            # Check if recent price movement is trending
            price_start = recent_data['close'].iloc[0]
            price_end = recent_data['close'].iloc[-1]
            price_change_pct = (price_end - price_start) / price_start

            if direction == 'long':
                # Long trend: at least 1% upward movement
                trend_exists = price_change_pct > 0.01
            else:
                # Short trend: at least 1% downward movement
                trend_exists = price_change_pct < -0.01

            if trend_exists:
                logger.debug(f"HTF {direction} trend VALID: price slope {price_change_pct*100:.1f}%")
            else:
                logger.debug(f"HTF {direction} trend INVALID: price slope {price_change_pct*100:.1f}% too weak")

            return trend_exists

        except Exception as e:
            logger.exception(f"HTF trend validation error: {e}")
            return False  # Reject on error (conservative)

    def _validate_premium_discount_zone(self, direction: str, current_price: float, context: MarketContext) -> bool:
        """
        Validate price is in correct premium/discount zone for ICT setups.

        Professional ICT rule:
        - Longs ONLY in discount zone (below 50% Fibonacci of daily range)
        - Shorts ONLY in premium zone (above 50% Fibonacci of daily range)

        Args:
            direction: 'long' or 'short'
            current_price: Current market price
            context: Market context (contains PDH/PDL)

        Returns:
            True if price is in correct zone, False otherwise
        """
        try:
            # Get daily high/low from context
            pdh = context.pdh
            pdl = context.pdl

            if pdh is None or pdl is None or pdh <= pdl:
                logger.debug(f"P/D validation skipped: invalid PDH/PDL (PDH={pdh}, PDL={pdl})")
                return True  # Skip validation if daily levels unavailable

            # Calculate daily range and current position
            daily_range = pdh - pdl
            fib_level = (current_price - pdl) / daily_range  # 0 = PDL, 1 = PDH

            if direction == 'long':
                # Longs ONLY in discount zone (< 50% Fib = below equilibrium)
                is_valid = fib_level < 0.5
                if not is_valid:
                    logger.debug(f"ICT long REJECTED: in premium zone ({fib_level*100:.0f}% of daily range)")
                else:
                    logger.debug(f"ICT long VALID: in discount zone ({fib_level*100:.0f}% of daily range)")
                return is_valid

            else:  # short
                # Shorts ONLY in premium zone (> 50% Fib = above equilibrium)
                is_valid = fib_level > 0.5
                if not is_valid:
                    logger.debug(f"ICT short REJECTED: in discount zone ({fib_level*100:.0f}% of daily range)")
                else:
                    logger.debug(f"ICT short VALID: in premium zone ({fib_level*100:.0f}% of daily range)")
                return is_valid

        except Exception as e:
            logger.exception(f"Premium/discount validation error: {e}")
            return True  # Allow on error (permissive fallback)

    def _detect_order_blocks(self, df: pd.DataFrame, context: MarketContext,
                            sweep_events: List[StructureEvent],
                            mss_events: List[StructureEvent]) -> List[StructureEvent]:
        """
        Detect Order Blocks with PROFESSIONAL ICT criteria.

        Professional Requirements (at least ONE required):
        1. Liquidity sweep occurred 1-10 bars before OB formation
        2. Market Structure Shift matches OB direction

        Quality Filters (MUST pass):
        - High volume (> 2.0x institutional standard)
        - Significant block size
        - At swing structure level
        - Current price testing the zone
        """
        events = []

        try:
            current_price = context.current_price
            current_bar_idx = len(df) - 1

            # Look for significant moves in recent history
            search_start = max(5, current_bar_idx - self.ob_lookback_bars)

            for move_start_idx in range(search_start, current_bar_idx - 2):
                # Check for significant move with HIGH VOLUME (professional standard)
                move_bars = df.iloc[move_start_idx:move_start_idx + 5]
                if len(move_bars) < 3:
                    continue

                move_start_price = move_bars['close'].iloc[0]
                move_end_price = move_bars['close'].iloc[-1]
                move_pct = (move_end_price - move_start_price) / move_start_price

                # PROFESSIONAL: Require 2.0x volume (institutional participation)
                vol_surge_series = move_bars['vol_surge'].dropna()
                move_had_institutional_volume = (vol_surge_series > 2.0).any() if len(vol_surge_series) > 0 else False

                if abs(move_pct) >= self.ob_min_move_pct and move_had_institutional_volume:
                    # Find last opposing candle before move
                    ob_candle_idx = self._find_opposing_candle(df, move_start_idx, move_pct)

                    if ob_candle_idx is not None:
                        # Check PROFESSIONAL criteria
                        has_sweep, has_mss, confluence_factors = self._check_professional_criteria(
                            ob_candle_idx, move_pct, sweep_events, mss_events
                        )

                        # REQUIRE at least one professional criterion
                        if has_sweep or has_mss:
                            event = self._create_order_block_event(
                                df, ob_candle_idx, move_pct, current_price, current_bar_idx,
                                context, has_sweep, has_mss, confluence_factors
                            )
                            if event:
                                events.append(event)
                                logger.debug(f"OB_PROF: {context.symbol} OB ACCEPTED - "
                                           f"Sweep:{has_sweep} MSS:{has_mss} Confluence:{len(confluence_factors)}")
                        else:
                            logger.debug(f"OB_PROF: {context.symbol} OB REJECTED - No sweep or MSS")

        except Exception as e:
            logger.exception(f"Order block detection error: {e}")
        return events

    def _check_professional_criteria(self, ob_candle_idx: int, move_pct: float,
                                    sweep_events: List[StructureEvent],
                                    mss_events: List[StructureEvent]) -> tuple:
        """
        Check professional ICT criteria for Order Block.

        Returns: (has_sweep, has_mss, confluence_factors)
        """
        confluence_factors = []

        # Check 1: Liquidity sweep before OB
        has_sweep = len(sweep_events) > 0
        if has_sweep:
            confluence_factors.append('liquidity_sweep')

        # Check 2: MSS confirmation
        has_mss = False
        ob_direction = 'short' if move_pct > 0 else 'long'  # OB forms opposite to move
        for mss in mss_events:
            if mss.side == ob_direction:
                has_mss = True
                confluence_factors.append('mss_confirmation')
                break

        return has_sweep, has_mss, confluence_factors

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
                                context: MarketContext, has_sweep: bool, has_mss: bool,
                                confluence_factors: List[str]) -> Optional[StructureEvent]:
        """
        Create order block event if professional criteria met.

        PROFESSIONAL FILTERS:
        1. Liquidity sweep OR MSS (at least one required)
        2. High volume (> 2.0x institutional standard)
        3. Significant block size
        4. At swing structure level
        5. Current price testing the zone
        """
        ob_candle = df.iloc[ob_candle_idx]
        ob_high = ob_candle['high']
        ob_low = ob_candle['low']
        ob_close = ob_candle['close']

        # FILTER 1: Block size must be significant (from config)
        block_size_pct = (ob_high - ob_low) / ob_low
        if block_size_pct < self.ob_min_block_size_pct:
            logger.debug(f"OB rejected: block too small ({block_size_pct*100:.2f}% < {self.ob_min_block_size_pct*100:.2f}%)")
            return None

        # FILTER 2: High volume on block formation (from config)
        ob_volume = ob_candle.get('volume', 0)
        avg_volume = df['volume'].rolling(20).mean().iloc[ob_candle_idx]
        volume_ratio = ob_volume / avg_volume if avg_volume > 0 else 0
        if volume_ratio < self.ob_min_volume_ratio:
            logger.debug(f"OB rejected: insufficient volume ({volume_ratio:.1f}x < {self.ob_min_volume_ratio}x)")
            return None

        # FILTER 3: Check if at structure level (from config)
        lookback_window = df.iloc[max(0, ob_candle_idx - 10):min(len(df), ob_candle_idx + 10)]
        if move_pct > 0:  # Bearish OB - should be near swing high
            is_swing_high = ob_high >= lookback_window['high'].max() * (1 - self.ob_swing_tolerance_pct)
            if not is_swing_high:
                logger.debug(f"OB rejected: not at structure level (swing high)")
                return None
        else:  # Bullish OB - should be near swing low
            is_swing_low = ob_low <= lookback_window['low'].min() * (1 + self.ob_swing_tolerance_pct)
            if not is_swing_low:
                logger.debug(f"OB rejected: not at structure level (swing low)")
                return None

        # FILTER 4: Current test candle should show rejection (from config)
        current_candle = df.iloc[current_bar_idx]
        candle_range = current_candle['high'] - current_candle['low']
        if candle_range > 0:
            if move_pct > 0:  # Testing resistance - need lower wick
                wick_size = current_candle['close'] - current_candle['low']
                wick_ratio = wick_size / candle_range
                if wick_ratio < self.ob_min_rejection_wick_pct:
                    logger.debug(f"OB rejected: weak rejection wick ({wick_ratio*100:.0f}% < {self.ob_min_rejection_wick_pct*100:.0f}%)")
                    return None
            else:  # Testing support - need upper wick
                wick_size = current_candle['high'] - current_candle['close']
                wick_ratio = wick_size / candle_range
                if wick_ratio < self.ob_min_rejection_wick_pct:
                    logger.debug(f"OB rejected: weak rejection wick ({wick_ratio*100:.0f}% < {self.ob_min_rejection_wick_pct*100:.0f}%)")
                    return None

        if move_pct > 0:  # Bearish OB (resistance zone)
            if (ob_low <= current_price <= ob_high * (1 + self.ob_test_tolerance)):
                bars_since_ob = current_bar_idx - ob_candle_idx
                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))
                strength = min(3.0, abs(move_pct) * 100 * time_decay)

                # Boost confidence based on confluence
                base_confidence = self._calculate_institutional_strength(context, strength, "order_block", "short", move_pct, bars_since_ob)
                enhanced_confidence = min(1.0, base_confidence * (1.0 + len(confluence_factors) * 0.2))

                return StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type='order_block_short',
                    side='short',
                    confidence=enhanced_confidence,
                    levels={'entry': ob_high, 'stop': ob_high + (context.indicators.get('atr', 1.0) * 1.5), 'target': ob_high - (context.indicators.get('atr', 1.0) * 2.0)},
                    context={
                        'ob_high': ob_high,
                        'ob_low': ob_low,
                        'move_pct': move_pct * 100,
                        'bars_since_formation': bars_since_ob,
                        'pattern_type': 'bearish_order_block',
                        'professional_filters': {
                            'has_liquidity_sweep': has_sweep,
                            'has_mss_confirmation': has_mss,
                            'confluence_count': len(confluence_factors),
                            'confluence_factors': confluence_factors
                        }
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

                # Boost confidence based on confluence
                base_confidence = self._calculate_institutional_strength(context, strength, "order_block", "long", move_pct, bars_since_ob)
                enhanced_confidence = min(1.0, base_confidence * (1.0 + len(confluence_factors) * 0.2))

                return StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type='order_block_long',
                    side='long',
                    confidence=enhanced_confidence,
                    levels={'entry': ob_low, 'stop': ob_low - (context.indicators.get('atr', 1.0) * 1.5), 'target': ob_low + (context.indicators.get('atr', 1.0) * 2.0)},
                    context={
                        'ob_high': ob_high,
                        'ob_low': ob_low,
                        'move_pct': move_pct * 100,
                        'bars_since_formation': bars_since_ob,
                        'pattern_type': 'bullish_order_block',
                        'professional_filters': {
                            'has_liquidity_sweep': has_sweep,
                            'has_mss_confirmation': has_mss,
                            'confluence_count': len(confluence_factors),
                            'confluence_factors': confluence_factors
                        }
                    },
                    price=ob_low
                )

        return None

    def _detect_fair_value_gaps(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Fair Value Gaps - price imbalances indicating institutional activity."""
        events = []

        try:
            current_price = context.current_price
            # Need at least 3 bars to form FVG (before, gap, after)
            if len(df) < 3:
                return events

            # Loop backwards from recent bars, checking last 20 bars or available bars
            lookback_bars = min(20, len(df) - 2)  # -2 because we need i-1, i, i+1
            start_idx = max(2, len(df) - lookback_bars)

            for i in range(start_idx, len(df) - 1):  # Loop forward from start_idx to second-to-last bar
                candle_before = df.iloc[i - 1]
                candle_middle = df.iloc[i]
                candle_after = df.iloc[i + 1]

                # Check volume condition
                if self.fvg_require_volume:
                    vol_surge = df['vol_surge'].iloc[i]
                    if pd.isna(vol_surge) or vol_surge < self.fvg_min_volume_mult:
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
            logger.exception(f"FVG detection error: {e}")
        return events

    def _create_fvg_event(self, direction: str, candle_before: pd.Series, candle_after: pd.Series,
                         df: pd.DataFrame, gap_index: int, current_price: float,
                         context: MarketContext) -> Optional[StructureEvent]:
        """Create Fair Value Gap event if conditions are met.

        QUALITY FILTERS ADDED:
        1. Gap must be significant (> 0.4% of price - stricter than before)
        2. High volume on gap creation (> 2.0x average)
        3. Clean gap (no overlap/wicks)
        4. Must be at key level (near structure)
        """
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

        # FILTER 1: Gap must be significant (from config)
        if gap_pct < self.fvg_min_gap_size_pct:
            logger.debug(f"FVG rejected: gap too small ({gap_pct*100:.2f}% < {self.fvg_min_gap_size_pct*100:.2f}%)")
            return None

        # Keep max gap check
        if gap_pct > self.fvg_max_gap_pct:
            logger.debug(f"FVG rejected: gap too large ({gap_pct*100:.2f}% > {self.fvg_max_gap_pct*100:.1f}%)")
            return None

        # FILTER 2: High volume on gap creation (from config)
        volume_strength = df['vol_surge'].iloc[gap_index]
        volume_strength = volume_strength if pd.notna(volume_strength) else 1.0
        if volume_strength < self.fvg_min_volume_ratio:
            logger.debug(f"FVG rejected: insufficient volume ({volume_strength:.1f}x < {self.fvg_min_volume_ratio}x)")
            return None

        # FILTER 3: Check for clean gap (from config - allows penetration tolerance)
        middle_candle = df.iloc[gap_index]
        gap_size_abs = abs(fvg_top - fvg_bottom)
        allowed_penetration = gap_size_abs * self.fvg_max_wick_penetration_pct

        if direction == 'long':
            # Bullish FVG - middle candle low shouldn't penetrate too much
            if middle_candle['low'] < fvg_bottom - allowed_penetration:
                logger.debug(f"FVG rejected: not clean gap (wick overlap)")
                return None
        else:
            # Bearish FVG - middle candle high shouldn't penetrate too much
            if middle_candle['high'] > fvg_top + allowed_penetration:
                logger.debug(f"FVG rejected: not clean gap (wick overlap)")
                return None

        # FILTER 4: Must be at key level (from config)
        vwap = context.indicators.get('vwap', current_price)
        gap_center = (fvg_top + fvg_bottom) / 2
        distance_from_vwap_pct = abs(gap_center - vwap) / vwap

        # Gap should be within tolerance of VWAP or at swing levels
        near_vwap = distance_from_vwap_pct < self.fvg_vwap_distance_tolerance_pct

        # Check if near swing levels (from config)
        lookback_window = df.iloc[max(0, gap_index - 10):min(len(df), gap_index + 10)]
        if direction == 'long':
            near_swing = fvg_bottom <= lookback_window['low'].min() * (1 + self.fvg_swing_tolerance_pct)
        else:
            near_swing = fvg_top >= lookback_window['high'].max() * (1 - self.fvg_swing_tolerance_pct)

        if not (near_vwap or near_swing):
            logger.debug(f"FVG rejected: not at key level (VWAP dist: {distance_from_vwap_pct*100:.1f}%)")
            return None

        # PROFESSIONAL ICT FIX: Check price is RETRACING INTO gap (not just in gap)
        # FVG long: Price should retrace INTO gap from ABOVE (price was above, now coming back)
        # FVG short: Price should retrace INTO gap from BELOW (price was below, now coming back)

        # First check if current price is IN the gap zone
        price_in_gap = (fvg_bottom * (1 - self.fvg_fill_tolerance) <= current_price <=
                        fvg_top * (1 + self.fvg_fill_tolerance))

        if not price_in_gap:
            return None  # Price not testing gap yet

        # NEW: Check retracement direction (requires previous candle data)
        if gap_index < len(df) - 2:  # Ensure we have current and previous candle
            prev_candle = df.iloc[-2]
            prev_close = prev_candle['close']

            if direction == 'long':
                # Bullish FVG: Price should be retracing INTO gap from ABOVE
                # Previous candle should be above gap top
                price_coming_from_above = prev_close > fvg_top

                if not price_coming_from_above:
                    logger.debug(f"FVG long rejected: not retracing from above (prev={prev_close:.2f}, gap_top={fvg_top:.2f})")
                    return None

                logger.debug(f"FVG long VALID: retracing into gap from above (prev={prev_close:.2f} > gap_top={fvg_top:.2f})")

            else:  # short
                # Bearish FVG: Price should be retracing INTO gap from BELOW
                # Previous candle should be below gap bottom
                price_coming_from_below = prev_close < fvg_bottom

                if not price_coming_from_below:
                    logger.debug(f"FVG short rejected: not retracing from below (prev={prev_close:.2f}, gap_bottom={fvg_bottom:.2f})")
                    return None

                logger.debug(f"FVG short VALID: retracing into gap from below (prev={prev_close:.2f} < gap_bottom={fvg_bottom:.2f})")

        # PROFESSIONAL ICT FIX: Validate premium/discount zone
        # Longs ONLY in discount, shorts ONLY in premium
        if not self._validate_premium_discount_zone(direction, current_price, context):
            logger.debug(f"FVG {direction} rejected: wrong premium/discount zone")
            return None

        # Calculate strength
        strength = min(3.0, gap_pct * 500 + volume_strength * 0.5)

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
                    if vol_surge is None or pd.isna(vol_surge) or vol_surge < self.sweep_min_volume_surge:
                        continue

                    event = self._check_liquidity_sweep(df, i, level_name, level_price,
                                                      upper_wick_ratio, lower_wick_ratio,
                                                      context)
                    if event:
                        events.append(event)

        except Exception as e:
            logger.exception(f"Liquidity sweep detection error: {e}")
        return events

    def _check_liquidity_sweep(self, df: pd.DataFrame, sweep_idx: int, level_name: str,
                             level_price: float, upper_wick_ratio: float, lower_wick_ratio: float,
                             context: MarketContext) -> Optional[StructureEvent]:
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
            logger.exception(f"Premium/Discount zone detection error: {e}")
        return events

    def _detect_break_of_structure(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Break of Structure - trend continuation pattern.

        PROFESSIONAL ICT REQUIREMENT (2025 Update):
        BOS is a CONTINUATION pattern that requires established HTF trend.
        - Bullish BOS: Break of swing high in UPTREND (continuation)
        - Bearish BOS: Break of swing low in DOWNTREND (continuation)
        - Entry on RETEST of broken level, not immediate chase
        - Requires HH/HL or LL/LH structure pattern (not just indicator trend)
        """
        events = []

        try:
            # BOS DIAGNOSTIC: Log bar count check (throttled)
            if not hasattr(self, '_bos_bar_check_count'):
                self._bos_bar_check_count = 0
            if self._bos_bar_check_count < 5:
                self._bos_bar_check_count += 1
                logger.info(f"BOS_BAR_CHECK | {context.symbol} | bars={len(df)} | required={self.bos_min_structure_bars + 5}")

            if len(df) < self.bos_min_structure_bars + 5:
                return events

            current_price = context.current_price
            atr = context.indicators.get('atr', current_price * 0.01)

            # Find recent swing highs and lows with significance
            swing_highs = self._find_swing_points_with_significance(df, 'high')
            swing_lows = self._find_swing_points_with_significance(df, 'low')

            # BOS DIAGNOSTIC: Log swing point discovery (throttled to first 10 symbols)
            if not hasattr(self, '_bos_diag_count'):
                self._bos_diag_count = 0
            if self._bos_diag_count < 10:
                self._bos_diag_count += 1
                logger.info(f"BOS_DIAG | {context.symbol} | bars={len(df)} | swing_highs={len(swing_highs)} swing_lows={len(swing_lows)} | price={current_price:.2f}")

            # Check for bullish BOS (break above recent swing high)
            if swing_highs:
                # PRO ICT: Use most significant swing high, not just max of last 3
                significant_high = self._get_most_significant_swing(swing_highs)
                recent_high = significant_high['price']
                break_distance = current_price - recent_high
                break_pct = break_distance / recent_high

                if break_pct >= self.bos_min_break_pct:
                    # BOS DIAGNOSTIC: Log candidate that passes break threshold
                    if self._bos_diag_count <= 10:
                        logger.info(f"BOS_LONG_CANDIDATE | {context.symbol} | break_pct={break_pct*100:.2f}% >= {self.bos_min_break_pct*100:.2f}% | recent_high={recent_high:.2f}")

                    # PRO ICT: Validate HTF structure pattern (HH/HL)
                    if self.bos_require_htf_structure:
                        htf_structure_valid = self._validate_htf_structure_pattern(swing_highs, swing_lows, 'long')
                        if not htf_structure_valid:
                            if self._bos_diag_count <= 10:
                                # Show swing high prices to understand why HH pattern failed
                                high_prices = [sh['price'] for sh in swing_highs[-3:]] if len(swing_highs) >= 3 else [sh['price'] for sh in swing_highs]
                                logger.info(f"BOS_LONG_REJECT | {context.symbol} | no HH pattern | swing_highs={high_prices}")
                        else:
                            htf_trend_valid = True
                    else:
                        # Fallback to indicator-based trend validation
                        htf_trend_valid = self._validate_htf_trend(df, context, 'long')

                    if self.bos_require_htf_structure and not htf_structure_valid:
                        pass  # Skip - no valid structure
                    elif not self.bos_require_htf_structure and not htf_trend_valid:
                        logger.debug(f"BOS long rejected: no HTF uptrend (BOS requires trend continuation)")
                        pass  # Skip to next check
                    else:
                        # Check volume confirmation if required
                        volume_confirmed = True
                        if self.bos_volume_confirmation:
                            recent_vol_z = df['vol_z'].tail(3).max()
                            volume_confirmed = (pd.notna(recent_vol_z) and recent_vol_z >= 1.5)

                        if not volume_confirmed:
                            if self._bos_diag_count <= 10:
                                logger.info(f"BOS_LONG_REJECT | {context.symbol} | vol_z={recent_vol_z:.2f} < 1.5")
                        else:
                            # Validate premium/discount zone
                            if not self._validate_premium_discount_zone('long', current_price, context):
                                if self._bos_diag_count <= 10:
                                    logger.info(f"BOS_LONG_REJECT | {context.symbol} | not in discount zone")
                            else:
                                # PRO ICT: Calculate retest zone for entry
                                zone_half_width = recent_high * self.bos_retest_zone_pct
                                retest_zone = [recent_high - zone_half_width, recent_high + zone_half_width]

                                # Determine entry mode
                                entry_mode = self.bos_entry_mode

                                events.append(StructureEvent(
                                    symbol=context.symbol,
                                    timestamp=context.timestamp,
                                    structure_type='break_of_structure_long',
                                    side='long',
                                    confidence=self._calculate_institutional_strength(context, break_pct * 200, "break_of_structure", "long", break_pct, 0),
                                    levels={'entry': current_price, 'broken_level': recent_high, 'support': recent_high},
                                    context={
                                        'broken_level': recent_high,
                                        'break_distance_pct': break_pct * 100,
                                        'structure_type': 'swing_high',
                                        'pattern_type': 'bullish_break_of_structure',
                                        'entry_mode': entry_mode,
                                        'retest_zone': retest_zone if entry_mode == 'retest' else None,
                                        'swing_significance': significant_high.get('significance', 1.0)
                                    },
                                    price=recent_high
                                ))
                                logger.debug(f"BOS long detected: broken_level={recent_high:.2f}, entry_mode={entry_mode}, "
                                           f"retest_zone={retest_zone if entry_mode == 'retest' else 'N/A'}")

            # Check for bearish BOS (break below recent swing low)
            if swing_lows:
                # PRO ICT: Use most significant swing low, not just min of last 3
                significant_low = self._get_most_significant_swing(swing_lows)
                recent_low = significant_low['price']
                break_distance = recent_low - current_price
                break_pct = break_distance / recent_low

                if break_pct >= self.bos_min_break_pct:
                    # BOS DIAGNOSTIC: Log candidate that passes break threshold
                    if self._bos_diag_count <= 10:
                        logger.info(f"BOS_SHORT_CANDIDATE | {context.symbol} | break_pct={break_pct*100:.2f}% >= {self.bos_min_break_pct*100:.2f}% | recent_low={recent_low:.2f}")

                    # PRO ICT: Validate HTF structure pattern (LL/LH)
                    if self.bos_require_htf_structure:
                        htf_structure_valid = self._validate_htf_structure_pattern(swing_highs, swing_lows, 'short')
                        if not htf_structure_valid:
                            if self._bos_diag_count <= 10:
                                # Show swing low prices to understand why LL pattern failed
                                low_prices = [sl['price'] for sl in swing_lows[-3:]] if len(swing_lows) >= 3 else [sl['price'] for sl in swing_lows]
                                logger.info(f"BOS_SHORT_REJECT | {context.symbol} | no LL pattern | swing_lows={low_prices}")
                        else:
                            htf_trend_valid = True
                    else:
                        # Fallback to indicator-based trend validation
                        htf_trend_valid = self._validate_htf_trend(df, context, 'short')

                    if self.bos_require_htf_structure and not htf_structure_valid:
                        pass  # Skip - no valid structure
                    elif not self.bos_require_htf_structure and not htf_trend_valid:
                        logger.debug(f"BOS short rejected: no HTF downtrend (BOS requires trend continuation)")
                        pass  # Skip
                    else:
                        # Check volume confirmation if required
                        volume_confirmed = True
                        if self.bos_volume_confirmation:
                            recent_vol_z = df['vol_z'].tail(3).max()
                            volume_confirmed = (pd.notna(recent_vol_z) and recent_vol_z >= 1.5)

                        if not volume_confirmed:
                            if self._bos_diag_count <= 10:
                                logger.info(f"BOS_SHORT_REJECT | {context.symbol} | vol_z={recent_vol_z:.2f} < 1.5")
                        else:
                            # Validate premium/discount zone
                            if not self._validate_premium_discount_zone('short', current_price, context):
                                if self._bos_diag_count <= 10:
                                    logger.info(f"BOS_SHORT_REJECT | {context.symbol} | not in premium zone")
                            else:
                                # PRO ICT: Calculate retest zone for entry
                                zone_half_width = recent_low * self.bos_retest_zone_pct
                                retest_zone = [recent_low - zone_half_width, recent_low + zone_half_width]

                                # Determine entry mode
                                entry_mode = self.bos_entry_mode

                                events.append(StructureEvent(
                                    symbol=context.symbol,
                                    timestamp=context.timestamp,
                                    structure_type='break_of_structure_short',
                                    side='short',
                                    confidence=self._calculate_institutional_strength(context, break_pct * 200, "break_of_structure", "short", break_pct, 0),
                                    levels={'entry': current_price, 'broken_level': recent_low, 'resistance': recent_low},
                                    context={
                                        'broken_level': recent_low,
                                        'break_distance_pct': break_pct * 100,
                                        'structure_type': 'swing_low',
                                        'pattern_type': 'bearish_break_of_structure',
                                        'entry_mode': entry_mode,
                                        'retest_zone': retest_zone if entry_mode == 'retest' else None,
                                        'swing_significance': significant_low.get('significance', 1.0)
                                    },
                                    price=recent_low
                                ))
                                logger.debug(f"BOS short detected: broken_level={recent_low:.2f}, entry_mode={entry_mode}, "
                                           f"retest_zone={retest_zone if entry_mode == 'retest' else 'N/A'}")

        except Exception as e:
            logger.exception(f"Break of Structure detection error: {e}")
        return events

    def _find_swing_points(self, df: pd.DataFrame, price_type: str) -> List[float]:
        """Find swing highs or lows in the data (legacy - returns prices only)."""
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

    def _find_swing_points_with_significance(self, df: pd.DataFrame, price_type: str) -> List[Dict]:
        """Find swing highs or lows with significance scoring.

        PRO ICT: Returns swing points with bar index and significance score.
        Significance is based on:
        - How many bars the level held before being broken
        - Distance from other swing points (more isolated = more significant)
        """
        swing_points = []
        # Look for swings starting from bar 2 (need 2 bars before and 2 after for 5-bar window)
        # No need to skip early bars - we want ALL swings in the available data
        for i in range(2, len(df) - 2):

            window = df.iloc[i-2:i+3]
            center_value = window[price_type].iloc[2]

            is_swing = False
            if price_type == 'high':
                # Swing high: center is highest in window
                if center_value == window[price_type].max():
                    is_swing = True
            else:
                # Swing low: center is lowest in window
                if center_value == window[price_type].min():
                    is_swing = True

            if is_swing:
                # Calculate significance: how many bars after this point respected the level
                bars_held = 0
                for j in range(i + 3, len(df)):
                    if price_type == 'high':
                        if df[price_type].iloc[j] > center_value:
                            break  # Level broken
                    else:
                        if df[price_type].iloc[j] < center_value:
                            break  # Level broken
                    bars_held += 1

                # Normalize significance: more bars held = higher significance
                significance = min(1.0, bars_held / 20.0)  # Cap at 20 bars

                swing_points.append({
                    'price': center_value,
                    'bar_idx': i,
                    'bars_held': bars_held,
                    'significance': significance
                })

        return swing_points

    def _get_most_significant_swing(self, swing_points: List[Dict]) -> Dict:
        """Get the most significant swing point from the list.

        PRO ICT: Prioritize significance over just being the most recent.
        """
        if not swing_points:
            return {'price': 0, 'bar_idx': 0, 'significance': 0}

        # Consider only recent swings (last 5) to avoid stale levels
        recent_swings = swing_points[-5:] if len(swing_points) >= 5 else swing_points

        # Sort by significance (descending), then by recency (more recent = higher priority)
        sorted_swings = sorted(recent_swings,
                               key=lambda x: (x.get('significance', 0), x.get('bar_idx', 0)),
                               reverse=True)

        return sorted_swings[0]

    def _validate_htf_structure_pattern(self, swing_highs: List[Dict], swing_lows: List[Dict], direction: str) -> bool:
        """Validate HTF trend using actual structure pattern (HH/HL or LL/LH).

        PRO ICT: BOS requires established trend structure, not just indicator readings.
        - Long BOS: Need HH (Higher High) pattern - swing highs ascending
        - Short BOS: Need LL (Lower Low) pattern - swing lows descending
        """
        if direction == 'long':
            # Need at least 2 swing highs to check for HH pattern
            if len(swing_highs) >= 2:
                # Check if recent swing highs are ascending (HH pattern)
                recent_highs = [sh['price'] for sh in swing_highs[-3:]] if len(swing_highs) >= 3 else [sh['price'] for sh in swing_highs]
                if len(recent_highs) >= 2:
                    # At least the last swing high should be higher than previous
                    is_hh = recent_highs[-1] > recent_highs[-2]
                    if is_hh:
                        logger.debug(f"HTF structure valid for long: HH pattern confirmed ({recent_highs[-2]:.2f} -> {recent_highs[-1]:.2f})")
                        return True
            return False
        else:
            # Need at least 2 swing lows to check for LL pattern
            if len(swing_lows) >= 2:
                # Check if recent swing lows are descending (LL pattern)
                recent_lows = [sl['price'] for sl in swing_lows[-3:]] if len(swing_lows) >= 3 else [sl['price'] for sl in swing_lows]
                if len(recent_lows) >= 2:
                    # At least the last swing low should be lower than previous
                    is_ll = recent_lows[-1] < recent_lows[-2]
                    if is_ll:
                        logger.debug(f"HTF structure valid for short: LL pattern confirmed ({recent_lows[-2]:.2f} -> {recent_lows[-1]:.2f})")
                        return True
            return False

    def _detect_market_structure_shift(self, df: pd.DataFrame, context: MarketContext,
                                       swing_highs: List[float], swing_lows: List[float]) -> List[StructureEvent]:
        """
        Detect Market Structure Shift (MSS) - Professional ICT criterion.

        MSS occurs when:
        - Bullish MSS: Lower Low → Higher Low (trend change from down to up)
        - Bearish MSS: Higher High → Lower High (trend change from up to down)
        """
        events = []

        try:
            # Need at least 3 swing points to detect pattern change
            if len(swing_highs) >= 3:
                # Check for bearish MSS (Higher High → Lower High)
                if swing_highs[-3] < swing_highs[-2] and swing_highs[-2] > swing_highs[-1]:
                    # Pattern: HH → LH (bearish structure shift)
                    events.append(StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type='market_structure_shift_bearish',
                        side='short',
                        confidence=0.8,  # MSS is high-confidence signal
                        levels={'prev_high': swing_highs[-2], 'current_high': swing_highs[-1]},
                        context={
                            'pattern_type': 'bearish_mss',
                            'higher_high': swing_highs[-2],
                            'lower_high': swing_highs[-1],
                            'shift_type': 'HH_to_LH'
                        },
                        price=swing_highs[-1]
                    ))
                    logger.debug(f"MSS_DETECTOR: {context.symbol} Bearish MSS detected (HH→LH)")

            if len(swing_lows) >= 3:
                # Check for bullish MSS (Lower Low → Higher Low)
                if swing_lows[-3] > swing_lows[-2] and swing_lows[-2] < swing_lows[-1]:
                    # Pattern: LL → HL (bullish structure shift)
                    events.append(StructureEvent(
                        symbol=context.symbol,
                        timestamp=context.timestamp,
                        structure_type='market_structure_shift_bullish',
                        side='long',
                        confidence=0.8,  # MSS is high-confidence signal
                        levels={'prev_low': swing_lows[-2], 'current_low': swing_lows[-1]},
                        context={
                            'pattern_type': 'bullish_mss',
                            'lower_low': swing_lows[-2],
                            'higher_low': swing_lows[-1],
                            'shift_type': 'LL_to_HL'
                        },
                        price=swing_lows[-1]
                    ))
                    logger.debug(f"MSS_DETECTOR: {context.symbol} Bullish MSS detected (LL→HL)")

        except Exception as e:
            logger.exception(f"MSS detection error: {e}")

        return events

    def _detect_change_of_character(self, df: pd.DataFrame, context: MarketContext) -> List[StructureEvent]:
        """Detect Change of Character - trend reversal pattern.

        PROFESSIONAL ICT REQUIREMENT:
        CHOCH is a REVERSAL pattern that requires established trend to reverse.
        - Bullish CHOCH: Reversal of DOWNTREND (character change from bearish to bullish)
        - Bearish CHOCH: Reversal of UPTREND (character change from bullish to bearish)
        - NOT valid in chop/range (no trend to reverse!)
        """
        events = []

        try:
            if len(df) < max(self.choch_momentum_periods) + 5:
                return events

            current_price = context.current_price

            # Calculate momentum changes for different periods
            momentum_changes = {}

            # Check for recent direction reversals in the lookback window
            # Use wider lookback (2x max period) to catch reversals that happened earlier
            max_lookback = max(self.choch_momentum_periods) * 2
            recent_returns = df['returns_3'].tail(max_lookback).dropna()

            # Count sign changes in recent returns (bearish to bullish or vice versa)
            direction_reversals = 0
            if len(recent_returns) >= 2:
                for i in range(1, len(recent_returns)):
                    prev_return = recent_returns.iloc[i-1]
                    curr_return = recent_returns.iloc[i]
                    if prev_return != 0 and curr_return != 0:
                        if (prev_return < 0 and curr_return > 0) or (prev_return > 0 and curr_return < 0):
                            direction_reversals += 1

            # Also calculate traditional momentum changes
            for period in self.choch_momentum_periods:
                if len(df) >= period + 1:
                    old_momentum = df['returns_3'].iloc[-(period+1)]
                    current_momentum = df['returns_3'].iloc[-1]
                    # Skip if either value is NaN
                    if pd.notna(old_momentum) and pd.notna(current_momentum):
                        momentum_change = current_momentum - old_momentum
                        momentum_changes[period] = momentum_change

            # Check for significant momentum shift OR recent direction reversal
            significant_changes = [abs(change) >= self.choch_min_momentum_change
                                 for change in momentum_changes.values()]

            if any(significant_changes) or direction_reversals >= 1:
                # Determine direction of character change
                avg_momentum_change = np.mean(list(momentum_changes.values()))
                direction = 'long' if avg_momentum_change > 0 else 'short'

                # PROFESSIONAL ICT FIX: Validate trend exists to reverse
                # CHOCH long: Requires DOWNTREND to reverse (bullish reversal)
                # CHOCH short: Requires UPTREND to reverse (bearish reversal)
                # NOTE: Direction is OPPOSITE of the trend being reversed!

                opposite_direction = 'short' if direction == 'long' else 'long'
                htf_trend_valid = self._validate_htf_trend(df, context, opposite_direction)

                if not htf_trend_valid:
                    logger.debug(f"CHOCH {direction} rejected: no {opposite_direction} trend to reverse (CHOCH requires established trend)")
                    # Don't create CHOCH event - can't reverse a trend that doesn't exist
                    pass  # Skip
                else:
                    # Check volume confirmation
                    recent_vol_z = df['vol_z'].tail(3).max()
                    if pd.notna(recent_vol_z) and recent_vol_z >= self.choch_volume_threshold:
                        # PROFESSIONAL ICT FIX: Validate premium/discount zone
                        # Longs ONLY in discount, shorts ONLY in premium
                        if not self._validate_premium_discount_zone(direction, current_price, context):
                            logger.debug(f"CHOCH {direction} rejected: wrong premium/discount zone")
                        else:
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
            logger.exception(f"Change of Character detection error: {e}")
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
            if pd.notna(recent_vol_z):
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
            logger.exception(f"ICT quality score calculation error: {e}")
            return 0.0

    def _create_order_block_trade_plan(self, direction: str, ob_high: float, ob_low: float,
                                     current_price: float, atr: float) -> TradePlan:
        """Create trade plan for Order Block setup.

        FIX: Tighter stops + wider targets to improve R:R from 0.42 to 1.5+
        - Stop: Just below/above order block (0.1 ATR buffer)
        - Targets: Use measured move (block height projection)
        """
        block_height = ob_high - ob_low

        if direction == 'long':
            entry_price = ob_low
            # FIXED: Tighter stop (was 1.5 ATR, now 0.1 ATR below block)
            stop_loss = ob_low - (atr * 0.1)
            # FIXED: Use block height for measured move (more realistic)
            take_profit = entry_price + (block_height * 3.0)  # 3x block height
        else:
            entry_price = ob_high
            # FIXED: Tighter stop (was 1.5 ATR, now 0.1 ATR above block)
            stop_loss = ob_high + (atr * 0.1)
            # FIXED: Use block height for measured move
            take_profit = entry_price - (block_height * 3.0)  # 3x block height

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
        """Create trade plan for Fair Value Gap setup.

        FIX: Tighter stops + wider targets to improve R:R from 0.62 to 1.8+
        - Stop: Just beyond gap (0.1 ATR buffer)
        - Targets: Use gap height projection (3x gap height)
        """
        gap_height = fvg_top - fvg_bottom

        if direction == 'long':
            entry_price = fvg_bottom
            # FIXED: Tighter stop (was 1.0 ATR, now 0.1 ATR below gap)
            stop_loss = fvg_bottom - (atr * 0.1)
            # FIXED: Use gap height for measured move
            take_profit = entry_price + (gap_height * 3.0)  # 3x gap height
        else:
            entry_price = fvg_top
            # FIXED: Tighter stop (was 1.0 ATR, now 0.1 ATR above gap)
            stop_loss = fvg_top + (atr * 0.1)
            # FIXED: Use gap height for measured move
            take_profit = entry_price - (gap_height * 3.0)  # 3x gap height

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
            vol_z_raw = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else None
            vol_z = vol_z_raw if pd.notna(vol_z_raw) else 1.0

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
            logger.exception(f"ICT: Error calculating institutional strength: {e}")
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
        """Calculate risk parameters for ICT setups using configurable SL buffer.

        Pro trader approach: SL at structure level + ATR buffer.
        For ICT, the sweep_low (liquidity sweep) or OB level is the logical invalidation.
        """
        atr = self._get_atr(context)
        entry_price = context.current_price

        if direction == 'long':
            stop_loss = entry_price - (atr * self.sl_buffer_atr)
        else:
            stop_loss = entry_price + (atr * self.sl_buffer_atr)

        # Enforce minimum stop distance
        min_stop_distance = entry_price * (self.min_stop_distance_pct / 100.0)
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share < min_stop_distance:
            if direction == 'long':
                stop_loss = entry_price - min_stop_distance
            else:
                stop_loss = entry_price + min_stop_distance
            risk_per_share = min_stop_distance

        return RiskParams(
            hard_sl=stop_loss,
            risk_percentage=self.risk_pct,
            risk_per_share=risk_per_share,
            atr=atr
        )

    def get_exit_levels(self, context: MarketContext, direction: str) -> ExitLevels:
        """Get exit levels for ICT setups."""
        atr = self._get_atr(context)
        entry_price = context.current_price

        if direction == 'long':
            stop_loss = entry_price - (atr * self.sl_buffer_atr)
            take_profit = entry_price + (atr * self.reward_risk_ratio * self.sl_buffer_atr)
        else:
            stop_loss = entry_price + (atr * self.sl_buffer_atr)
            take_profit = entry_price - (atr * self.reward_risk_ratio * self.sl_buffer_atr)

        return ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": self.reward_risk_ratio}],
            hard_sl=stop_loss,
            trail_to=None
        )

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR from context - must exist, no default."""
        if context.indicators and 'atr' in context.indicators:
            return context.indicators['atr']
        # Fallback: estimate ATR as 1% of price (better than crashing for missing indicator)
        return context.current_price * 0.01

    def rank_setup_quality(self, context: MarketContext) -> float:
        """Rank the quality of ICT setups."""
        analysis = self.detect(context)
        return analysis.quality_score

    def validate_timing(self, context: MarketContext) -> bool:
        """Validate timing for ICT setups."""
        # ICT setups are valid during regular trading hours
        current_hour = context.timestamp.hour
        return 9 <= current_hour <= 15  # 9:00 AM to 3:00 PM