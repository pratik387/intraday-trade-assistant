# structures/orb_structure.py
"""
Opening Range Breakout (ORB) Structure Implementation.

Handles ORH/ORL breakouts, pullback strategies, and OR-based risk management.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, time

from config.logging_config import get_agent_logger, get_screener_logger
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


class ORBStructure(BaseStructure):
    """
    Opening Range Breakout structure implementation.

    Detects ORH/ORL breakouts and breakdowns, generates ORB pullback strategies,
    calculates OR-based stop levels, and handles early session timing validation.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize ORB structure with configuration."""
        super().__init__(config)

        # KeyError if missing trading parameters
        self.orb_minutes = config["orb_minutes"]
        self.min_range_pct = config["min_range_pct"] / 100.0
        self.breakout_volume_mult = config["breakout_volume_mult"]
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.breakout_buffer_pct = config["breakout_buffer_pct"] / 100.0
        self.min_stop_distance_pct = config["min_stop_distance_pct"] / 100.0
        self.stop_distance_mult = config["stop_distance_mult"]
        self.confidence_volume_confirmed = config["confidence_volume_confirmed"]
        self.confidence_no_volume = config["confidence_no_volume"]
        self.pullback_zones = config["pullback_zones"]

        logger.debug(f"ORB: Initialized with config - Buffer: {self.breakout_buffer_pct:.3f}%, Stop: {self.min_stop_distance_pct:.3f}%, Targets: {self.target_mult_t1}x/{self.target_mult_t2}x")

        # Session timing
        self.session_start = time(9, 15)  # Market open
        self.orb_cutoff = time(10, 30)    # Latest ORB entry
        self.morning_session_end = time(11, 30)  # End of morning session focus

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        """
        Detect ORB patterns in market data.

        Args:
            market_context: Market data and context

        Returns:
            StructureAnalysis with detected ORB events
        """
        symbol = market_context.symbol
        logger.debug(f"ORB: Starting detection for {symbol}")

        try:
            df = market_context.df_5m
            if df is None:
                logger.debug(f"ORB: {symbol} - No 5m data available")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="No 5m data available"
                )

            if len(df) < 10:
                logger.debug(f"ORB: {symbol} - Insufficient data: {len(df)} bars < 10 minimum")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data: {len(df)} bars < 10 minimum"
                )

            events = []
            orh = market_context.orh
            orl = market_context.orl

            if orh is None:
                logger.debug(f"ORB: {symbol} - ORH not available")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="ORH not available"
                )

            if orl is None:
                logger.debug(f"ORB: {symbol} - ORL not available")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="ORL not available"
                )

            # Check ORB range validity
            mid_price = (orh + orl) / 2
            if mid_price <= 0:
                logger.debug(f"ORB: {symbol} - Invalid price levels: ORH={orh}, ORL={orl}")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Invalid price levels"
                )

            orb_range_pct = (orh - orl) / mid_price
            logger.debug(f"ORB: {symbol} - ORB range: {orb_range_pct:.3f}% (min required: {self.min_range_pct:.3f}%)")

            if orb_range_pct < self.min_range_pct:
                logger.debug(f"ORB: {symbol} - Range too small: {orb_range_pct:.3f}% < {self.min_range_pct:.3f}%")
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"ORB range too small: {orb_range_pct:.3f}% < {self.min_range_pct:.3f}%"
                )

            current_price = market_context.current_price
            current_time = market_context.timestamp

            logger.debug(f"ORB: {symbol} - Price: {current_price:.2f}, ORH: {orh:.2f}, ORL: {orl:.2f}")

            # Detect ORH breakout
            if current_price > orh:
                logger.debug(f"ORB: {symbol} - Price above ORH, checking volume confirmation")
                volume_confirmed = self._check_volume_confirmation(df, "breakout")

                confidence = self._calculate_institutional_strength(market_context, orb_range_pct, volume_confirmed, "long", orh, orl)
                logger.debug(f"ORB: {symbol} - ORH breakout detected | Price: {current_price:.2f} > ORH: {orh:.2f} | Volume confirmed: {volume_confirmed} | Confidence: {confidence:.2f}")

                event = StructureEvent(
                    symbol=market_context.symbol,
                    timestamp=current_time,
                    structure_type="orb_breakout_long",
                    side="long",
                    confidence=confidence,
                    levels={"orh": orh, "orl": orl, "entry": current_price},
                    context={"range_pct": orb_range_pct, "volume_confirmed": volume_confirmed},
                    price=current_price
                )
                events.append(event)

            # Detect ORL breakdown
            elif current_price < orl:
                logger.debug(f"ORB: {symbol} - Price below ORL, checking volume confirmation")
                volume_confirmed = self._check_volume_confirmation(df, "breakdown")

                confidence = self._calculate_institutional_strength(market_context, orb_range_pct, volume_confirmed, "short", orh, orl)
                logger.debug(f"ORB: {symbol} - ORL breakdown detected | Price: {current_price:.2f} < ORL: {orl:.2f} | Volume confirmed: {volume_confirmed} | Confidence: {confidence:.2f}")

                event = StructureEvent(
                    symbol=market_context.symbol,
                    timestamp=current_time,
                    structure_type="orb_breakdown_short",
                    side="short",
                    confidence=confidence,
                    levels={"orh": orh, "orl": orl, "entry": current_price},
                    context={"range_pct": orb_range_pct, "volume_confirmed": volume_confirmed},
                    price=current_price
                )
                events.append(event)

            # Price within OR range
            else:
                logger.debug(f"ORB: {symbol} - Price within OR range: {orl:.2f} <= {current_price:.2f} <= {orh:.2f}")

            # Detect pullback opportunities
            pullback_events = self._detect_pullback_opportunities(market_context)
            events.extend(pullback_events)

            structure_detected = len(events) > 0
            quality_score = self._calculate_quality_score(market_context, events) if structure_detected else 0.0

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=events,
                quality_score=quality_score,
                analysis_notes={
                    "orb_range_pct": orb_range_pct,
                    "orh": orh,
                    "orl": orl,
                    "timing_valid": self.validate_timing(current_time)
                }
            )

        except Exception as e:
            logger.exception(f"ORB detection failed for {market_context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """Generate long ORB strategy."""
        symbol = market_context.symbol
        logger.debug(f"ORB: Planning long strategy for {symbol}")

        try:
            orh = market_context.orh
            orl = market_context.orl

            if orh is None:
                logger.debug(f"ORB: {symbol} - Cannot plan long strategy: ORH not available")
                return None

            if orl is None:
                logger.debug(f"ORB: {symbol} - Cannot plan long strategy: ORL not available")
                return None

            current_price = market_context.current_price
            min_breakout_price = orh * (1.0 - self.breakout_buffer_pct)  # Configurable buffer

            logger.debug(f"ORB: {symbol} - Long planning: Price {current_price:.2f}, Min breakout: {min_breakout_price:.2f}")

            # Only plan long if price is above ORH or in pullback zone
            if current_price < min_breakout_price:  # Not convincingly above ORH
                rejection_reason = f"Price {current_price:.2f} < min breakout {min_breakout_price:.2f}"

                # Log structured rejection for screening stage
                screener_logger = get_screener_logger()
                screener_logger.log_reject(
                    symbol,
                    rejection_reason,
                    timestamp=market_context.timestamp.isoformat(),
                    structure_type="orb",
                    side="long",
                    price=current_price,
                    orh=orh,
                    min_breakout_price=min_breakout_price
                )
                return None

            # Check timing validity
            if not self.validate_timing(market_context.timestamp):
                rejection_reason = f"Invalid timing for ORB at {market_context.timestamp.time()}"

                # Log structured rejection for screening stage
                screener_logger = get_screener_logger()
                screener_logger.log_reject(
                    symbol,
                    rejection_reason,
                    timestamp=market_context.timestamp.isoformat(),
                    structure_type="orb",
                    side="long",
                    price=current_price,
                    orh=orh,
                    current_time=market_context.timestamp.time().isoformat()
                )
                return None

            logger.debug(f"ORB: {symbol} - Long strategy approved: Price {current_price:.2f} > ORH {orh:.2f}")

            # Calculate risk parameters
            risk_params = self.calculate_risk_params(current_price, market_context)

            # Generate exit levels
            trade_plan = TradePlan(
                symbol=market_context.symbol,
                side="long",
                structure_type="orb_breakout",
                entry_price=current_price,
                risk_params=risk_params,
                exit_levels=self._generate_orb_exits(current_price, orh, orl, "long"),
                qty=0,  # Will be calculated later
                notional=0.0,
                confidence=self._calculate_institutional_strength(market_context, 0.0, True, "long", market_context.orh or 0, market_context.orl or 0),
                timestamp=market_context.timestamp,
                market_context=market_context
            )

            # Add entry zone for pullback entries
            trade_plan.entry_zone = {
                "min": orh,
                "max": current_price * 1.005  # Allow small range above current
            }

            return trade_plan

        except Exception as e:
            logger.exception(f"ORB long strategy planning failed for {market_context.symbol}: {e}")
            return None

    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """Generate short ORB strategy."""
        symbol = market_context.symbol
        logger.debug(f"ORB: Planning short strategy for {symbol}")

        try:
            orh = market_context.orh
            orl = market_context.orl

            if orh is None:
                logger.debug(f"ORB: {symbol} - Cannot plan short strategy: ORH not available")
                return None

            if orl is None:
                logger.debug(f"ORB: {symbol} - Cannot plan short strategy: ORL not available")
                return None

            current_price = market_context.current_price
            max_breakdown_price = orl * (1.0 + self.breakout_buffer_pct)  # Configurable buffer

            logger.debug(f"ORB: {symbol} - Short planning: Price {current_price:.2f}, Max breakdown: {max_breakdown_price:.2f}")

            # Only plan short if price is below ORL or in pullback zone
            if current_price > max_breakdown_price:  # Not convincingly below ORL
                rejection_reason = f"Price {current_price:.2f} > max breakdown {max_breakdown_price:.2f}"

                # Log structured rejection for screening stage
                screener_logger = get_screener_logger()
                screener_logger.log_reject(
                    symbol,
                    rejection_reason,
                    timestamp=market_context.timestamp.isoformat(),
                    structure_type="orb",
                    side="short",
                    price=current_price,
                    orl=orl,
                    max_breakdown_price=max_breakdown_price
                )
                return None

            # Check timing validity
            if not self.validate_timing(market_context.timestamp):
                rejection_reason = f"Invalid timing for ORB at {market_context.timestamp.time()}"

                # Log structured rejection for screening stage
                screener_logger = get_screener_logger()
                screener_logger.log_reject(
                    symbol,
                    rejection_reason,
                    timestamp=market_context.timestamp.isoformat(),
                    structure_type="orb",
                    side="short",
                    price=current_price,
                    orl=orl,
                    current_time=market_context.timestamp.time().isoformat()
                )
                return None

            logger.debug(f"ORB: {symbol} - Short strategy approved: Price {current_price:.2f} < ORL {orl:.2f}")

            # Calculate risk parameters
            risk_params = self.calculate_risk_params(current_price, market_context)

            # Generate trade plan
            trade_plan = TradePlan(
                symbol=market_context.symbol,
                side="short",
                structure_type="orb_breakdown",
                entry_price=current_price,
                risk_params=risk_params,
                exit_levels=self._generate_orb_exits(current_price, orh, orl, "short"),
                qty=0,  # Will be calculated later
                notional=0.0,
                confidence=self._calculate_institutional_strength(market_context, 0.0, True, "short", market_context.orh or 0, market_context.orl or 0),
                timestamp=market_context.timestamp,
                market_context=market_context
            )

            # Add entry zone for pullback entries
            trade_plan.entry_zone = {
                "min": current_price * 0.995,  # Allow small range below current
                "max": orl
            }

            return trade_plan

        except Exception as e:
            logger.exception(f"ORB short strategy planning failed for {market_context.symbol}: {e}")
            return None

    def calculate_risk_params(self, entry_price: float, market_context: MarketContext) -> RiskParams:
        """Calculate ORB-specific risk parameters."""
        orh = market_context.orh or entry_price
        orl = market_context.orl or entry_price

        # OR-based stop calculation
        orb_range = orh - orl
        atr = market_context.indicators.get("atr") if market_context.indicators else None

        # Use the larger of OR range or ATR for stop calculation - CONFIGURABLE
        if atr and atr > 0:
            stop_distance = max(orb_range * self.stop_distance_mult, atr * self.stop_distance_mult)
            logger.debug(f"ORB: {market_context.symbol} - Stop distance using ATR: {stop_distance:.3f} (ATR: {atr:.3f}, OR range: {orb_range:.3f})")
        else:
            stop_distance = orb_range * self.stop_distance_mult
            logger.debug(f"ORB: {market_context.symbol} - Stop distance using OR range: {stop_distance:.3f} (no ATR available)")

        # Conservative minimum stop distance - CONFIGURABLE
        min_stop_distance = entry_price * self.min_stop_distance_pct
        stop_distance = max(stop_distance, min_stop_distance)

        if stop_distance == min_stop_distance:
            logger.debug(f"ORB: {market_context.symbol} - Using minimum stop distance: {min_stop_distance:.3f} ({self.min_stop_distance_pct:.3f}% of price)")

        # Calculate hard stop based on OR levels - CONFIGURABLE
        buffer_mult = self.config.get("or_level_buffer_mult", 0.1)  # Default: 10% of OR range as buffer
        if entry_price > orh:  # Long position
            hard_sl = max(orl - (orb_range * buffer_mult), entry_price - stop_distance)
            logger.debug(f"ORB: {market_context.symbol} - Long hard SL: max(ORL-buffer: {orl - (orb_range * buffer_mult):.3f}, entry-stop: {entry_price - stop_distance:.3f}) = {hard_sl:.3f}")
        else:  # Short position
            hard_sl = min(orh + (orb_range * buffer_mult), entry_price + stop_distance)
            logger.debug(f"ORB: {market_context.symbol} - Short hard SL: min(ORH+buffer: {orh + (orb_range * buffer_mult):.3f}, entry+stop: {entry_price + stop_distance:.3f}) = {hard_sl:.3f}")

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            volatility_adj=1.0
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """Get ORB-specific exit levels."""
        return trade_plan.exit_levels

    def rank_setup_quality(self, market_context: MarketContext) -> float:
        """Score ORB setup quality (0-100)."""
        try:
            score = 0.0

            orh = market_context.orh
            orl = market_context.orl

            if orh is None or orl is None:
                return 0.0

            current_price = market_context.current_price

            # Base score from OR range quality
            mid_price = (orh + orl) / 2
            if mid_price <= 0:
                return 0.0

            orb_range_pct = (orh - orl) / mid_price
            if orb_range_pct >= self.min_range_pct:
                score += 30.0

            # Bonus for good range size (optimal around 1-2%)
            if 0.01 <= orb_range_pct <= 0.02:
                score += 20.0
            elif orb_range_pct > 0.005:
                score += 10.0

            # Timing bonus
            if self.validate_timing(market_context.timestamp):
                score += 20.0

            # Volume confirmation
            if market_context.df_5m is not None:
                if self._check_volume_confirmation(market_context.df_5m, "breakout"):
                    score += 15.0

            # Position relative to OR levels
            if current_price > orh:  # Above ORH
                distance_score = min(15.0, ((current_price - orh) / orh) * 100 * 300)  # Bonus for clean break
                score += distance_score
            elif current_price < orl:  # Below ORL
                distance_score = min(15.0, ((orl - current_price) / orl) * 100 * 300)  # Bonus for clean break
                score += distance_score

            return min(100.0, score)

        except Exception as e:
            logger.exception(f"ORB ranking failed for {market_context.symbol}: {e}")
            return 0.0

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """Check if timing is appropriate for ORB strategies."""
        try:
            current_time_of_day = current_time.time()

            # Must be after session start but before cutoff
            return self.session_start <= current_time_of_day <= self.orb_cutoff

        except Exception:
            return False

    # Private helper methods

    def _check_volume_confirmation(self, df: pd.DataFrame, event_type: str) -> bool:
        """Check if current volume supports the ORB event."""
        try:
            if len(df) < 5:
                logger.debug(f"ORB: Volume check failed - insufficient data: {len(df)} bars < 5")
                return False

            current_volume = df["volume"].iloc[-1]
            avg_volume = df["volume"].tail(10).mean()

            # Check for invalid average volume (zero or NaN)
            if pd.isna(avg_volume) or avg_volume <= 0:
                logger.debug(f"ORB: Volume check failed - invalid avg_volume: {avg_volume}")
                return False

            volume_ratio = current_volume / avg_volume
            required_ratio = self.breakout_volume_mult

            volume_confirmed = volume_ratio > required_ratio

            logger.debug(f"ORB: Volume check for {event_type} - Current: {current_volume:.0f}, Avg: {avg_volume:.0f}, Ratio: {volume_ratio:.2f}, Required: {required_ratio:.2f}, Confirmed: {volume_confirmed}")

            return volume_confirmed

        except Exception as e:
            logger.exception(f"ORB: Volume confirmation error: {e}")
            return False

    def _detect_pullback_opportunities(self, market_context: MarketContext) -> List[StructureEvent]:
        """Detect ORB pullback opportunities."""
        events = []

        try:
            orh = market_context.orh
            orl = market_context.orl
            current_price = market_context.current_price

            if orh is None or orl is None:
                return events

            # Check for pullback to ORH after breakout
            if current_price > orh:
                pullback_level = orh + (current_price - orh) * self.pullback_zones["shallow"]
                if orh <= current_price <= pullback_level:
                    event = StructureEvent(
                        symbol=market_context.symbol,
                        timestamp=market_context.timestamp,
                        structure_type="orb_pullback_long",
                        side="long",
                        confidence=self._calculate_institutional_strength(market_context, 0.0, True, "long", market_context.orh or 0, market_context.orl or 0),
                        levels={"orh": orh, "pullback_level": pullback_level},
                        context={"pullback_type": "shallow"},
                        price=current_price
                    )
                    events.append(event)

            # Check for pullback to ORL after breakdown
            elif current_price < orl:
                pullback_level = orl - (orl - current_price) * self.pullback_zones["shallow"]
                if pullback_level <= current_price <= orl:
                    event = StructureEvent(
                        symbol=market_context.symbol,
                        timestamp=market_context.timestamp,
                        structure_type="orb_pullback_short",
                        side="short",
                        confidence=self._calculate_institutional_strength(market_context, 0.0, True, "short", market_context.orh or 0, market_context.orl or 0),
                        levels={"orl": orl, "pullback_level": pullback_level},
                        context={"pullback_type": "shallow"},
                        price=current_price
                    )
                    events.append(event)

        except Exception as e:
            logger.exception(f"ORB pullback detection failed for {market_context.symbol}: {e}")
        return events

    def _generate_orb_exits(self, entry_price: float, orh: float, orl: float, side: str) -> ExitLevels:
        """Generate ORB-specific exit levels."""
        orb_range = orh - orl

        if side == "long":
            # Target based on OR range projection - CONFIGURABLE
            t1_level = entry_price + (orb_range * self.target_mult_t1)
            t2_level = entry_price + (orb_range * self.target_mult_t2)
            logger.debug(f"ORB: Long targets - T1: {t1_level:.2f} ({self.target_mult_t1}x), T2: {t2_level:.2f} ({self.target_mult_t2}x)")

            # Stop below ORL with buffer - CONFIGURABLE
            buffer_mult = self.config.get("or_level_buffer_mult", 0.1)
            hard_sl = orl - (orb_range * buffer_mult)

        else:  # short
            # Target based on OR range projection - CONFIGURABLE
            t1_level = entry_price - (orb_range * self.target_mult_t1)
            t2_level = entry_price - (orb_range * self.target_mult_t2)
            logger.debug(f"ORB: Short targets - T1: {t1_level:.2f} ({self.target_mult_t1}x), T2: {t2_level:.2f} ({self.target_mult_t2}x)")

            # Stop above ORH with buffer - CONFIGURABLE
            buffer_mult = self.config.get("or_level_buffer_mult", 0.1)
            hard_sl = orh + (orb_range * buffer_mult)

        targets = [
            {"level": t1_level, "qty_pct": 50, "rr": abs(t1_level - entry_price) / abs(entry_price - hard_sl)},
            {"level": t2_level, "qty_pct": 50, "rr": abs(t2_level - entry_price) / abs(entry_price - hard_sl)}
        ]

        return ExitLevels(
            targets=targets,
            hard_sl=hard_sl,
            trail_to="t1",  # Trail to T1 after T1 hit
            structure_exit={
                "or_reclaim": True,  # Exit if price reclaims OR levels against us
                "time_exit": "15:15"  # EOD exit
            }
        )

    def _calculate_quality_score(self, market_context: MarketContext, events: List[StructureEvent]) -> float:
        """Calculate overall quality score for detected ORB events."""
        if not events:
            return 0.0

        # Base score from strongest event
        max_confidence = max(event.confidence for event in events)
        base_score = max_confidence * 70.0  # Scale confidence to 0-70

        # Bonus factors
        bonus = 0.0

        # Timing bonus
        if self.validate_timing(market_context.timestamp):
            bonus += 15.0

        # Volume confirmation bonus
        volume_confirmed = any(
            event.context.get("volume_confirmed", False)
            for event in events
        )
        if volume_confirmed:
            bonus += 15.0

        return min(100.0, base_score + bonus)

    def _calculate_institutional_strength(self, context: MarketContext, range_pct: float,
                                        volume_confirmed: bool, side: str, orh: float, orl: float) -> float:
        """Calculate institutional-grade strength for ORB setups."""
        try:
            # Get volume data for institutional validation
            df = context.df_5m
            vol_z = df['vol_z'].iloc[-1] if 'vol_z' in df.columns else 1.0

            # Base strength from range quality and volume (institutional volume threshold ≥1.5)
            range_quality = min(3.0, range_pct * 200)  # Scale range percentage
            base_strength = max(1.0, vol_z * range_quality * 0.5)

            # Professional bonuses for institutional-grade ORB setups
            strength_multiplier = 1.0

            # Volume confirmation bonus (institutional participation)
            if volume_confirmed and vol_z >= 1.5:  # Institutional volume threshold
                strength_multiplier *= 1.3  # 30% bonus for institutional volume
                logger.debug(f"ORB: Institutional volume confirmation bonus applied (vol_z={vol_z:.2f})")

            if vol_z >= 2.0:  # Strong institutional volume
                strength_multiplier *= 1.2  # Additional 20% bonus
                logger.debug(f"ORB: Strong institutional volume bonus applied")

            # Range quality bonuses (opening range significance)
            if range_pct >= 0.015:  # 1.5%+ opening range (significant)
                strength_multiplier *= 1.25  # 25% bonus for significant range
                logger.debug(f"ORB: Significant opening range bonus applied ({range_pct*100:.2f}%)")
            elif range_pct >= 0.01:  # 1.0%+ opening range (moderate)
                strength_multiplier *= 1.1  # 10% bonus for moderate range

            # Early breakout bonus (institutional preference for early moves)
            current_hour = pd.to_datetime(context.timestamp).hour
            current_minute = pd.to_datetime(context.timestamp).minute

            if current_hour == 9 and current_minute <= 45:  # First 30 minutes
                strength_multiplier *= 1.2  # 20% bonus for early breakout
                logger.debug(f"ORB: Early breakout bonus applied ({current_hour}:{current_minute:02d})")
            elif current_hour == 10 and current_minute <= 15:  # Second 30 minutes
                strength_multiplier *= 1.1  # 10% bonus for morning breakout

            # Level significance bonus
            if orh > 0 and orl > 0:
                range_size = orh - orl
                current_price = context.current_price

                # Clean breakout bonus (not too far from level)
                if side == "long" and orh > 0:
                    breakout_distance = (current_price - orh) / orh
                    if breakout_distance <= 0.005:  # Within 0.5% of ORH
                        strength_multiplier *= 1.15  # 15% bonus for clean breakout
                        logger.debug(f"ORB: Clean long breakout bonus applied")
                elif side == "short" and orl > 0:  # short
                    breakout_distance = (orl - current_price) / orl
                    if breakout_distance <= 0.005:  # Within 0.5% of ORL
                        strength_multiplier *= 1.15  # 15% bonus for clean breakout
                        logger.debug(f"ORB: Clean short breakout bonus applied")

            # Market session bonus (ORB works best in morning session)
            if 9 <= current_hour <= 11:  # Peak ORB session
                strength_multiplier *= 1.1  # 10% session timing bonus
                logger.debug(f"ORB: Morning session bonus applied (hour={current_hour})")

            # Apply multipliers and ensure institutional minimum
            final_strength = base_strength * strength_multiplier

            # Institutional minimum for regime gate passage (≥2.0)
            final_strength = max(final_strength, 1.5)  # Minimum viable strength

            logger.debug(f"ORB: {context.symbol} {side} - Base: {base_strength:.2f}, "
                       f"Multiplier: {strength_multiplier:.2f}, Final: {final_strength:.2f}")

            return final_strength

        except Exception as e:
            logger.exception(f"ORB: Error calculating institutional strength: {e}")
            return 1.8  # Safe fallback below regime threshold