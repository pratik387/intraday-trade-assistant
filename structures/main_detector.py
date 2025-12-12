#!/usr/bin/env python3
"""
Main Detector Coordination Module

This module serves as the primary coordinator for all structure detection,
providing comprehensive structure detection through a unified system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from config.logging_config import get_agent_logger

from .base_structure import BaseStructure
from .data_models import StructureEvent, TradePlan, RiskParams, ExitLevels, MarketContext, StructureAnalysis
from services.gates.trade_decision_gate import SetupCandidate
from .ict_structure import ICTStructure
from .level_breakout_structure import LevelBreakoutStructure
from .failure_fade_structure import FailureFadeStructure
from .squeeze_release_structure import SqueezeReleaseStructure
from .flag_continuation_structure import FlagContinuationStructure
from .momentum_structure import MomentumStructure
from .vwap_structure import VWAPStructure
from .gap_structure import GapStructure
from .orb_structure import ORBStructure
from .support_resistance_structure import SupportResistanceStructure
from .trend_structure import TrendStructure
from .volume_structure import VolumeStructure
from .range_structure import RangeStructure
from .fhm_structure import FHMStructure
from pipelines.base_pipeline import get_cap_segment

logger = get_agent_logger()

class MainDetector(BaseStructure):
    """
    Main detector coordination class that orchestrates all structure detection.

    This provides comprehensive
    structure detection coordination with proper prioritization and conflict resolution.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Extract setups config once and validate
        setups_config = config.get("setups", {})
        if not setups_config:
            logger.exception("MAIN_DETECTOR: No setups configuration found! All detectors will be disabled.")
            self.detectors = {}
            return

        logger.debug(f"MAIN_DETECTOR: Found {len(setups_config)} setup configurations")
        logger.debug(f"MAIN_DETECTOR: Setup names: {list(setups_config.keys())}")
        logger.debug(f"MAIN_DETECTOR: Will initialize {len([s for s in setups_config.values() if s.get('enabled', False)])} enabled detectors")

        # Initialize sub-detectors
        self.detectors = {}

        # Define detector configurations with their classes
        detector_configs = [
            # (setup_name, detector_class, detector_key)
            ("ict_comprehensive", ICTStructure, "ict"),
            ("fvg", ICTStructure, "fvg"),
            ("order_block", ICTStructure, "order_block"),
            # ICT individual setups - these inherit params from ict_comprehensive
            ("order_block_long", ICTStructure, "order_block_long"),
            ("order_block_short", ICTStructure, "order_block_short"),
            ("fair_value_gap_long", ICTStructure, "fair_value_gap_long"),
            ("fair_value_gap_short", ICTStructure, "fair_value_gap_short"),
            ("liquidity_sweep_long", ICTStructure, "liquidity_sweep_long"),
            ("liquidity_sweep_short", ICTStructure, "liquidity_sweep_short"),
            ("premium_zone_short", ICTStructure, "premium_zone_short"),
            ("discount_zone_long", ICTStructure, "discount_zone_long"),
            ("break_of_structure_long", ICTStructure, "break_of_structure_long"),
            ("break_of_structure_short", ICTStructure, "break_of_structure_short"),
            ("change_of_character_long", ICTStructure, "change_of_character_long"),
            ("change_of_character_short", ICTStructure, "change_of_character_short"),
            # NOTE: trend_reversal_long/short and breakout_long/short are NOT mapped here because:
            # - TrendStructure only produces: trend_pullback_long/short, trend_continuation_long/short
            # - LevelBreakoutStructure only produces: level_breakout_long/short
            ("level_breakout_long", LevelBreakoutStructure, "level_breakout_long"),
            ("level_breakout_short", LevelBreakoutStructure, "level_breakout_short"),
            ("failure_fade_long", FailureFadeStructure, "failure_fade_long"),
            ("failure_fade_short", FailureFadeStructure, "failure_fade_short"),
            ("squeeze_release_long", SqueezeReleaseStructure, "squeeze_release_long"),
            ("squeeze_release_short", SqueezeReleaseStructure, "squeeze_release_short"),
            ("flag_continuation_long", FlagContinuationStructure, "flag_continuation_long"),
            ("flag_continuation_short", FlagContinuationStructure, "flag_continuation_short"),
            ("momentum_breakout_long", MomentumStructure, "momentum_breakout_long"),
            ("momentum_breakout_short", MomentumStructure, "momentum_breakout_short"),
            ("vwap_reclaim_long", VWAPStructure, "vwap_reclaim_long"),
            ("vwap_lose_short", VWAPStructure, "vwap_lose_short"),
            ("gap_fill_long", GapStructure, "gap_fill_long"),
            ("gap_fill_short", GapStructure, "gap_fill_short"),
            ("gap_breakout_long", GapStructure, "gap_breakout_long"),
            ("gap_breakout_short", GapStructure, "gap_breakout_short"),
            ("orb_breakout_long", ORBStructure, "orb_breakout_long"),
            ("orb_breakout_short", ORBStructure, "orb_breakout_short"),
            ("orb_breakdown_short", ORBStructure, "orb_breakdown_short"),
            ("orb_breakout", ORBStructure, "orb_breakout"),
            ("orb_breakdown", ORBStructure, "orb_breakdown"),
            ("orb_pullback_long", ORBStructure, "orb_pullback_long"),
            ("orb_pullback_short", ORBStructure, "orb_pullback_short"),
            ("support_bounce_long", SupportResistanceStructure, "support_bounce_long"),
            ("resistance_bounce_short", SupportResistanceStructure, "resistance_bounce_short"),
            ("trend_pullback_long", TrendStructure, "trend_pullback_long"),
            ("trend_pullback_short", TrendStructure, "trend_pullback_short"),
            ("volume_spike_reversal_long", VolumeStructure, "volume_spike_reversal_long"),
            ("volume_spike_reversal_short", VolumeStructure, "volume_spike_reversal_short"),
            ("range_rejection_long", RangeStructure, "range_rejection_long"),
            ("range_rejection_short", RangeStructure, "range_rejection_short"),
            ("vwap_mean_reversion_long", VWAPStructure, "vwap_mean_reversion_long"),
            ("vwap_mean_reversion_short", VWAPStructure, "vwap_mean_reversion_short"),
            ("support_breakdown_short", SupportResistanceStructure, "support_breakdown_short"),
            ("resistance_breakout_long", SupportResistanceStructure, "resistance_breakout_long"),
            ("range_breakout_long", RangeStructure, "range_breakout_long"),
            ("range_breakdown_short", RangeStructure, "range_breakdown_short"),
            ("range_bounce_long", RangeStructure, "range_bounce_long"),
            ("range_bounce_short", RangeStructure, "range_bounce_short"),
            ("trend_continuation_long", TrendStructure, "trend_continuation_long"),
            ("trend_continuation_short", TrendStructure, "trend_continuation_short"),
            # First Hour Momentum (FHM) - captures big movers early
            ("first_hour_momentum_long", FHMStructure, "first_hour_momentum_long"),
            ("first_hour_momentum_short", FHMStructure, "first_hour_momentum_short"),
            # Generic structure types (if needed for fallback)
            ("level_breakout", LevelBreakoutStructure, "level_breakout"),
            ("failure_fade", FailureFadeStructure, "failure_fade"),
            ("flag_continuation", FlagContinuationStructure, "flag_continuation"),
            ("squeeze_release", SqueezeReleaseStructure, "squeeze_release"),
            ("momentum", MomentumStructure, "momentum"),
            ("trend", TrendStructure, "trend"),
            ("volume", VolumeStructure, "volume"),
            ("vwap", VWAPStructure, "vwap"),
            ("gap", GapStructure, "gap"),
            ("range", RangeStructure, "range"),
            ("support_resistance", SupportResistanceStructure, "support_resistance")
        ]

        # ICT setups that should inherit params from ict_comprehensive
        ict_derived_setups = {
            "order_block_long", "order_block_short",
            "fair_value_gap_long", "fair_value_gap_short",
            "liquidity_sweep_long", "liquidity_sweep_short",
            "premium_zone_short", "discount_zone_long",
            "break_of_structure_long", "break_of_structure_short",
            "change_of_character_long", "change_of_character_short"
        }
        ict_base_config = setups_config.get("ict_comprehensive", {})

        # Initialize all detectors uniformly
        for setup_name, detector_class, detector_key in detector_configs:
            setup_config = setups_config.get(setup_name, {})

            # For ICT-derived setups, merge with ict_comprehensive params
            if setup_name in ict_derived_setups and setup_config.get("enabled", False):
                if ict_base_config:
                    # Merge: start with ict_comprehensive, override with specific config
                    merged_config = {**ict_base_config, **setup_config}
                    setup_config = merged_config
                    logger.debug(f"MAIN_DETECTOR: Merged {setup_name} with ict_comprehensive params")
                else:
                    logger.warning(f"MAIN_DETECTOR: {setup_name} enabled but ict_comprehensive not found - skipping")
                    continue

            if setup_config.get("enabled", False):
                try:
                    self.detectors[detector_key] = detector_class(setup_config)
                    logger.debug(f"MAIN_DETECTOR: Initialized {detector_key} from {setup_name}")
                except Exception as e:
                    logger.exception(f"MAIN_DETECTOR: Failed to initialize {setup_name}: {e}")
            else:
                logger.debug(f"MAIN_DETECTOR: {setup_name} disabled (enabled={setup_config.get('enabled', False)})")

        # Detection parameters - read from main_detector section if available
        main_detector_config = setups_config.get("main_detector", {})
        self.max_detections_per_symbol = main_detector_config.get("max_detections_per_symbol",
                                                                config.get("max_detections_per_symbol", 5))
        self.conflict_resolution_enabled = main_detector_config.get("conflict_resolution_enabled",
                                                                  config.get("conflict_resolution_enabled", True))
        self.priority_weights = config.get("priority_weights", {
            'ict': 1.0,                    # Highest priority - institutional concepts
            'level_breakout': 0.9,         # High priority - key level breaks
            'failure_fade': 0.8,           # High priority - mean reversion
            'squeeze_release': 0.7,        # Medium priority - volatility expansion
            'flag_continuation': 0.6,      # Medium priority - trend continuation
            'momentum_breakout': 0.5       # Lower priority - momentum only
        })

        logger.debug(f"MAIN_DETECTOR: Initialization complete with {len(self.detectors)} active detectors: {list(self.detectors.keys())}")

    def detect_setups(self, symbol: str, df5m_tail: pd.DataFrame,
                     levels: Dict[str, float] | None = None) -> List[SetupCandidate]:
        """
        Primary interface for structure detection.

        Args:
            symbol: Trading symbol
            df5m_tail: 5-minute dataframe
            levels: Dict with ORH, ORL, PDH, PDL, PDC levels

        Returns:
            List of SetupCandidate objects ordered by strength
        """
        return self.detect_setups_comprehensive(symbol, df5m_tail, levels or {})

    def detect_setups_comprehensive(self, symbol: str, df: pd.DataFrame,
                                  levels: Dict[str, float]) -> List[SetupCandidate]:
        """
        Comprehensive setup detection that coordinates all structure detectors.

        This is the main entry point for comprehensive structure detection.
        Returns SetupCandidate objects for the trading system.
        """

        try:
            # Reduced from 10 to 4 bars to allow early ORB detection (pro traders use 15-min range = 3 bars)
            # Individual detectors have their own min_bars checks for structures needing more data
            if df is None or len(df) < 4:
                logger.debug(f"MAIN_DETECTOR: {symbol} insufficient data (len={len(df) if df is not None else 0}) - EARLY RETURN")
                return []

            logger.debug(f"MAIN_DETECTOR: {symbol} data check passed, creating market context")

            # Create market context
            context = self._create_market_context(symbol, df, levels)
            logger.debug(f"MAIN_DETECTOR: {symbol} market context created successfully")

            # Run all detectors
            all_detections = {}
            detection_stats = {}

            logger.debug(f"MAIN_DETECTOR: {symbol} starting detector loop with {len(self.detectors)} detectors")

            for detector_name, detector in self.detectors.items():
                try:
                    # Check if detector should run at this time (time-based blacklisting)
                    if not detector.should_detect_at_time(context.timestamp):
                        continue

                    logger.debug(f"DETECTOR_CALL: {symbol} | {detector_name} | CALLING")
                    analysis = detector.detect(context)

                    if analysis.events:
                        all_detections[detector_name] = analysis
                        detection_stats[detector_name] = {
                            'count': len(analysis.events),
                            'quality': analysis.quality_score,
                            'events': analysis.events
                        }
                        logger.debug(f"DETECTOR_RESULT: {symbol} | {detector_name} | ACCEPTED | "
                                   f"events={len(analysis.events)} quality={analysis.quality_score:.2f}")
                    else:
                        logger.debug(f"DETECTOR_RESULT: {symbol} | {detector_name} | REJECTED | no_events")
                except Exception as e:
                    logger.error(f"DETECTOR_ERROR: {symbol} | {detector_name} | ERROR | {str(e)}")
                    logger.exception(f"MAIN_DETECTOR: Error in {detector_name} for {symbol}: {e}")
            # Resolve conflicts and prioritize
            final_events = self._resolve_conflicts_and_prioritize(all_detections, symbol)

            # Convert to SetupCandidate objects
            setup_candidates = self._convert_to_setup_candidates(final_events, symbol, context)

            # Log summary with detector statistics
            total_detectors = len(self.detectors)
            active_detectors = len(all_detections)
            silent_detectors = total_detectors - active_detectors

            logger.debug(f"DETECTOR_SUMMARY: {symbol} | total={total_detectors} active={active_detectors} "
                        f"silent={silent_detectors} setups={len(setup_candidates)}")

            if active_detectors > 0:
                detector_names = ','.join(all_detections.keys())
                logger.debug(f"DETECTOR_ACTIVE_LIST: {symbol} | {detector_names}")

            return setup_candidates

        except Exception as e:
            logger.exception(f"MAIN_DETECTOR: Comprehensive detection failed for {symbol}: {e}")
            return []

    def _create_market_context(self, symbol: str, df: pd.DataFrame,
                             levels: Dict[str, float]) -> MarketContext:
        """Create market context for structure detection."""
        try:
            # Ensure we have the required indicators
            d = df.copy()
            if 'vol_z' not in d.columns:
                vol_mean = d['volume'].rolling(20, min_periods=10).mean()
                vol_std = d['volume'].rolling(20, min_periods=10).std()
                d['vol_z'] = (d['volume'] - vol_mean) / vol_std.replace(0, np.nan)

            if 'atr' not in d.columns:
                # Simple ATR calculation
                high_low = d['high'] - d['low']
                high_close = np.abs(d['high'] - d['close'].shift())
                low_close = np.abs(d['low'] - d['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                d['atr'] = true_range.rolling(14, min_periods=10).mean()

            # Extract indicators
            indicators = {
                'vol_z': float(d['vol_z'].iloc[-1]) if not pd.isna(d['vol_z'].iloc[-1]) else 0.0,
                'atr': float(d['atr'].iloc[-1]) if not pd.isna(d['atr'].iloc[-1]) else 1.0
            }

            # BUGFIX: Use DataFrame timestamp instead of datetime.now() for backtesting compatibility
            bar_timestamp = pd.to_datetime(d.index[-1])

            return MarketContext(
                symbol=symbol,
                current_price=float(d['close'].iloc[-1]),
                timestamp=bar_timestamp,
                df_5m=d,
                session_date=bar_timestamp.date(),
                orh=levels.get('ORH'),
                orl=levels.get('ORL'),
                pdh=levels.get('PDH'),
                pdl=levels.get('PDL'),
                pdc=levels.get('PDC'),
                cap_segment=get_cap_segment(symbol),
                indicators=indicators
            )

        except Exception as e:
            logger.exception(f"MAIN_DETECTOR: Error creating market context for {symbol}: {e}")
            raise

    def _resolve_conflicts_and_prioritize(self, all_detections: Dict[str, StructureAnalysis],
                                        symbol: str) -> List[StructureEvent]:
        """
        Resolve conflicts between different detectors and prioritize the best setups.
        """
        if not self.conflict_resolution_enabled:
            # Simple concatenation without conflict resolution
            final_events = []
            for analysis in all_detections.values():
                final_events.extend(analysis.events)
            return final_events[:self.max_detections_per_symbol]

        try:
            # Group events by direction and proximity
            long_events = []
            short_events = []

            for detector_name, analysis in all_detections.items():
                for event in analysis.events:
                    # Use institutional-grade strength directly without scaling
                    # The detector's strength calculation already incorporates market dynamics

                    # Add detector context for traceability
                    if not hasattr(event, 'context') or event.context is None:
                        event.context = {}
                    event.context['detector_name'] = detector_name

                    if event.side == 'long':
                        long_events.append(event)
                    else:
                        short_events.append(event)

            # Resolve conflicts within each direction
            final_long_events = self._resolve_directional_conflicts(long_events, 'long')
            final_short_events = self._resolve_directional_conflicts(short_events, 'short')

            # Combine and limit
            final_events = final_long_events + final_short_events
            final_events.sort(key=lambda e: e.confidence, reverse=True)

            return final_events[:self.max_detections_per_symbol]

        except Exception as e:
            logger.exception(f"MAIN_DETECTOR: Error resolving conflicts for {symbol}: {e}")
            # Fallback to simple approach
            final_events = []
            for analysis in all_detections.values():
                final_events.extend(analysis.events)
            return final_events[:self.max_detections_per_symbol]

    def _resolve_directional_conflicts(self, events: List[StructureEvent],
                                     direction: str) -> List[StructureEvent]:
        """Resolve conflicts between events of the same direction."""
        if len(events) <= 1:
            return events

        # Group by price proximity (within 0.5%)
        price_groups = []
        tolerance_pct = 0.005  # 0.5%

        for event in events:
            added_to_group = False
            for group in price_groups:
                # Check if event is close to any event in this group
                for group_event in group:
                    price_diff_pct = abs(event.price - group_event.price) / group_event.price
                    if price_diff_pct <= tolerance_pct:
                        group.append(event)
                        added_to_group = True
                        break
                if added_to_group:
                    break

            if not added_to_group:
                price_groups.append([event])

        # Select best event from each group
        final_events = []
        for group in price_groups:
            if len(group) == 1:
                final_events.append(group[0])
            else:
                # Select event with highest confidence
                best_event = max(group, key=lambda e: e.confidence)

                # Merge context information from conflicting events
                if not hasattr(best_event, 'context') or best_event.context is None:
                    best_event.context = {}

                conflicting_detectors = [e.context.get('detector_name', 'unknown') for e in group if e != best_event]
                if conflicting_detectors:
                    best_event.context['resolved_conflicts'] = conflicting_detectors

                final_events.append(best_event)

        return final_events

    def _convert_to_setup_candidates(self, events: List[StructureEvent],
                                   symbol: str, market_context: MarketContext) -> List[SetupCandidate]:
        """Convert StructureEvent objects to SetupCandidate objects for the trading system."""
        setup_candidates = []

        for i, event in enumerate(events):
            try:
                # No quality filtering - let gates and ranking handle filtering like the old system

                # Create reasons list
                reasons = []
                if hasattr(event, 'context') and event.context:
                    detector_name = event.context.get('detector_name', 'unknown')
                    reasons.append(f"detector:{detector_name}")

                    if 'level_name' in event.context:
                        reasons.append(f"level:{event.context['level_name']}")

                    if 'pattern_type' in event.context:
                        reasons.append(f"pattern:{event.context['pattern_type']}")

                # Map structure type to setup type
                setup_type = self._map_structure_to_setup_type(event.structure_type)

                # Extract entry_mode and retest_zone from event context (dual-mode support)
                entry_mode = event.context.get('entry_mode') if event.context else None
                retest_zone = event.context.get('retest_zone') if event.context else None

                if setup_type:
                    setup_candidates.append(SetupCandidate(
                        setup_type=setup_type,
                        strength=float(event.confidence),
                        reasons=reasons,
                        orh=market_context.orh,
                        orl=market_context.orl,
                        entry_mode=entry_mode,
                        retest_zone=retest_zone,
                        cap_segment=market_context.cap_segment
                    ))

                    if entry_mode:
                        logger.debug(f"MAIN_DETECTOR: Converted {event.structure_type} -> {setup_type} "
                                   f"with confidence {event.confidence:.2f}, ENTRY_MODE={entry_mode}, "
                                   f"retest_zone={retest_zone} for {symbol}")
                    else:
                        logger.debug(f"MAIN_DETECTOR: Converted {event.structure_type} -> {setup_type} "
                                   f"with confidence {event.confidence:.2f} for {symbol}")
                else:
                    logger.debug(f"MAIN_DETECTOR: {symbol} no setup type mapping for '{event.structure_type}'")

            except Exception as e:
                logger.exception(f"MAIN_DETECTOR: Error converting event to setup candidate: {e}")
        # Sort by strength descending
        setup_candidates.sort(key=lambda s: s.strength, reverse=True)
        return setup_candidates


    def _map_structure_to_setup_type(self, structure_type: str) -> Optional[str]:
        """Map structure types to setup types for SetupCandidate creation."""

        # Direct mappings for most common cases
        direct_mappings = {
            # Level breakouts
            'level_breakout_long': 'breakout_long',
            'level_breakout_short': 'breakout_short',

            # Failure fades
            'failure_fade_long': 'failure_fade_long',
            'failure_fade_short': 'failure_fade_short',

            # Squeeze release
            'squeeze_release_long': 'squeeze_release_long',
            'squeeze_release_short': 'squeeze_release_short',

            # Flag continuations
            'flag_continuation_long': 'flag_continuation_long',
            'flag_continuation_short': 'flag_continuation_short',

            # Momentum
            'momentum_breakout_long': 'momentum_breakout_long',
            'momentum_breakout_short': 'momentum_breakout_short',
            'momentum_trend_long': 'trend_pullback_long',
            'momentum_trend_short': 'trend_pullback_short',

            # ICT structures
            'order_block_long': 'order_block_long',
            'order_block_short': 'order_block_short',
            'fair_value_gap_long': 'fair_value_gap_long',
            'fair_value_gap_short': 'fair_value_gap_short',
            'liquidity_sweep_long': 'liquidity_sweep_long',
            'liquidity_sweep_short': 'liquidity_sweep_short',
            'premium_zone_short': 'premium_zone_short',
            'discount_zone_long': 'discount_zone_long',
            'break_of_structure_long': 'break_of_structure_long',
            'break_of_structure_short': 'break_of_structure_short',
            'change_of_character_long': 'change_of_character_long',
            'change_of_character_short': 'change_of_character_short',

            # Gap structures
            'gap_fill_long': 'gap_fill_long',
            'gap_fill_short': 'gap_fill_short',
            'gap_breakout_long': 'gap_breakout_long',
            'gap_breakout_short': 'gap_breakout_short',

            # ORB structures
            'orb_breakout_long': 'orb_breakout_long',
            'orb_breakout_short': 'orb_breakout_short',
            'orb_breakdown_short': 'orb_breakdown_short',
            'orb_breakout': 'orb_breakout',
            'orb_breakdown': 'orb_breakdown',
            'orb_pullback_long': 'orb_pullback_long',
            'orb_pullback_short': 'orb_pullback_short',

            # VWAP structures
            'vwap_reclaim_long': 'vwap_reclaim_long',
            'vwap_lose_short': 'vwap_lose_short',
            'vwap_mean_reversion_long': 'vwap_mean_reversion_long',
            'vwap_mean_reversion_short': 'vwap_mean_reversion_short',

            # Trend structures
            'trend_continuation_long': 'trend_continuation_long',
            'trend_continuation_short': 'trend_continuation_short',
            'trend_pullback_long': 'trend_pullback_long',
            'trend_pullback_short': 'trend_pullback_short',
            'trend_reversal_long': 'trend_reversal_long',
            'trend_reversal_short': 'trend_reversal_short',

            # Volume structures
            'volume_spike_reversal_long': 'volume_spike_reversal_long',
            'volume_spike_reversal_short': 'volume_spike_reversal_short',

            # Range structures
            'range_bounce_long': 'range_bounce_long',
            'range_bounce_short': 'range_bounce_short',
            'range_breakdown_short': 'range_breakdown_short',
            'range_breakout_long': 'range_breakout_long',
            'range_rejection_long': 'range_rejection_long',
            'range_rejection_short': 'range_rejection_short',

            # Support/Resistance structures
            'support_bounce_long': 'support_bounce_long',
            'resistance_bounce_short': 'resistance_bounce_short',
            'support_breakdown_short': 'support_breakdown_short',
            'resistance_breakout_long': 'resistance_breakout_long',

            # First Hour Momentum (FHM) structures
            'first_hour_momentum_long': 'first_hour_momentum_long',
            'first_hour_momentum_short': 'first_hour_momentum_short'
        }

        return direct_mappings.get(structure_type)

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """
        Detect method for BaseStructure compatibility.
        This is mainly used for testing and integration.
        """

        try:
            # Extract levels from context
            levels = {
                'ORH': context.orh,
                'ORL': context.orl,
                'PDH': context.pdh,
                'PDL': context.pdl,
                'PDC': context.pdc
            }

            # Run comprehensive detection
            setup_candidates = self.detect_setups_comprehensive(
                context.symbol, context.df_5m, levels
            )

            # Convert back to StructureEvent objects for analysis
            events = []
            for candidate in setup_candidates:
                # Create a basic StructureEvent from SetupCandidate
                event = StructureEvent(
                    symbol=context.symbol,
                    timestamp=context.timestamp,
                    structure_type=candidate.setup_type,
                    side='long' if 'long' in candidate.setup_type else 'short',
                    confidence=candidate.strength,
                    levels={},  # Could be enhanced
                    context={'reasons': candidate.reasons},
                    price=context.current_price
                )
                events.append(event)

            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(events, context)

            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=quality_score
            )

        except Exception as e:
            logger.exception(f"MAIN_DETECTOR: detect method error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0
            )

    def _calculate_overall_quality_score(self, events: List[StructureEvent],
                                       context: MarketContext) -> float:
        """Calculate overall quality score for the detection analysis."""
        if not events:
            return 0.0

        try:
            # Base score from average event confidence
            avg_confidence = np.mean([event.confidence for event in events])
            base_score = min(3.0, avg_confidence)

            # Bonus for multiple confirmations
            if len(events) >= 3:
                base_score += 1.0
            elif len(events) >= 2:
                base_score += 0.5

            # Bonus for diverse detection types
            unique_types = len(set(event.structure_type.split('_')[0] for event in events))
            if unique_types >= 3:
                base_score += 0.5

            # Volume confirmation bonus
            vol_z = context.indicators.get('vol_z', 0)
            if vol_z >= 2.0:
                base_score += 0.5
            elif vol_z >= 1.5:
                base_score += 0.25

            return min(5.0, base_score)

        except Exception as e:
            logger.exception(f"MAIN_DETECTOR: Error calculating quality score: {e}")
            return 1.0

    # Abstract method implementations for BaseStructure compatibility

    def plan_long_strategy(self, context: MarketContext) -> Optional[TradePlan]:
        """Plan long strategy using the best available detector."""
        setup_candidates = self.detect_setups_comprehensive(
            context.symbol, context.df_5m, self._extract_levels_from_context(context)
        )

        long_candidates = [c for c in setup_candidates if 'long' in c.setup_type]
        if long_candidates:
            # Use the detector that found the best long setup
            best_candidate = max(long_candidates, key=lambda c: c.strength)

            # Create a basic trade plan
            atr = context.indicators.get('atr', 1.0)
            entry_price = context.current_price
            stop_loss = entry_price - (atr * 2.0)
            take_profit = entry_price + (atr * 2.0)

            return TradePlan(
                entry_price=entry_price,
                direction='long',
                risk_params=RiskParams(hard_sl=stop_loss, risk_per_share=atr * 0.01),
                exit_levels=ExitLevels(targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}], hard_sl=stop_loss),
                notes=f"Main detector: {best_candidate.setup_type}"
            )
        return None

    def plan_short_strategy(self, context: MarketContext) -> Optional[TradePlan]:
        """Plan short strategy using the best available detector."""
        setup_candidates = self.detect_setups_comprehensive(
            context.symbol, context.df_5m, self._extract_levels_from_context(context)
        )

        short_candidates = [c for c in setup_candidates if 'short' in c.setup_type]
        if short_candidates:
            # Use the detector that found the best short setup
            best_candidate = max(short_candidates, key=lambda c: c.strength)

            # Create a basic trade plan
            atr = context.indicators.get('atr', 1.0)
            entry_price = context.current_price
            stop_loss = entry_price + (atr * 2.0)
            take_profit = entry_price - (atr * 2.0)

            return TradePlan(
                entry_price=entry_price,
                direction='short',
                risk_params=RiskParams(hard_sl=stop_loss, risk_per_share=atr * 0.01),
                exit_levels=ExitLevels(targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}], hard_sl=stop_loss),
                notes=f"Main detector: {best_candidate.setup_type}"
            )
        return None

    def calculate_risk_params(self, context: MarketContext, direction: str) -> RiskParams:
        """Calculate risk parameters using the best available detector."""
        atr = context.indicators.get('atr', 1.0)

        if direction == 'long':
            stop_loss = context.current_price - (atr * 2.0)
        else:
            stop_loss = context.current_price + (atr * 2.0)

        return RiskParams(
            hard_sl=stop_loss,
            risk_per_share=atr * 0.01
        )

    def get_exit_levels(self, context: MarketContext, direction: str) -> ExitLevels:
        """Get exit levels using the best available detector."""
        atr = context.indicators.get('atr', 1.0)

        if direction == 'long':
            stop_loss = context.current_price - (atr * 2.0)
            take_profit = context.current_price + (atr * 2.0)
        else:
            stop_loss = context.current_price + (atr * 2.0)
            take_profit = context.current_price - (atr * 2.0)

        return ExitLevels(
            targets=[{"level": take_profit, "qty_pct": 100, "rr": 2.0}],
            hard_sl=stop_loss
        )

    def rank_setup_quality(self, context: MarketContext) -> float:
        """Rank setup quality using comprehensive analysis."""
        analysis = self.detect(context)
        return analysis.quality_score

    def validate_timing(self, context: MarketContext) -> bool:
        """Validate timing using multiple detectors."""
        # Main detector is valid during regular trading hours
        current_hour = context.timestamp.hour
        return 9 <= current_hour <= 15  # 9:00 AM to 3:00 PM

    def _extract_levels_from_context(self, context: MarketContext) -> Dict[str, float]:
        """Extract levels from market context."""
        levels = {}
        if context.orh is not None:
            levels['ORH'] = context.orh
        if context.orl is not None:
            levels['ORL'] = context.orl
        if context.pdh is not None:
            levels['PDH'] = context.pdh
        if context.pdl is not None:
            levels['PDL'] = context.pdl
        if context.pdc is not None:
            levels['PDC'] = context.pdc
        return levels