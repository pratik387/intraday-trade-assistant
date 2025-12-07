# structures/fhm_structure.py
"""
First Hour Momentum (FHM) Structure Implementation.

Captures big movers early in the session using RVOL, price momentum, and VWAP position.
Pro traders use first hour institutional flow to identify high-conviction setups.

Key conditions:
- Time window: 09:15 - 10:15 (first hour of trading)
- RVOL >= 2.0x (unusual institutional activity)
- Price move >= 1.5% from open (momentum started)
- Volume >= 100k (sufficient liquidity)
- VWAP position confirms direction (long above VWAP, short below VWAP)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import time

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

logger = get_agent_logger()


class FHMStructure(BaseStructure):
    """
    First Hour Momentum structure implementation.

    Detects high-momentum setups in the first hour of trading based on
    RVOL, price movement from open, and VWAP position confirmation.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize FHM structure with configuration."""
        super().__init__(config)

        # Time window config - KeyError if missing (no defaults)
        time_window = config["time_window"]
        self.start_time = self._parse_time(time_window["start"])
        self.end_time = self._parse_time(time_window["end"])

        # Trigger thresholds - KeyError if missing (no defaults)
        triggers = config["triggers"]
        self.min_rvol = triggers["min_rvol"]
        self.min_price_move_pct = triggers["min_price_move_pct"]
        self.min_volume = triggers["min_volume"]

        # VWAP position config - KeyError if missing (no defaults)
        vwap_cfg = config["vwap_position"]
        self.long_requires_above_vwap = vwap_cfg["long_requires_above_vwap"]
        self.short_requires_below_vwap = vwap_cfg["short_requires_below_vwap"]

        # Target and stop config - KeyError if missing (no defaults)
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]
        self.stop_atr_mult = config["stop_atr_mult"]

        # Confidence levels - KeyError if missing (no defaults)
        self.confidence_high_rvol = config["confidence_high_rvol"]
        self.confidence_base = config["confidence_base"]
        self.high_rvol_threshold = config["high_rvol_threshold"]

        # Minimum bars required
        self.min_bars_required = config["min_bars_required"]

        logger.debug(f"FHM: Initialized - Time: {self.start_time}-{self.end_time}, "
                    f"RVOL>={self.min_rvol}x, Move>={self.min_price_move_pct}%, "
                    f"Vol>={self.min_volume}")

    def _parse_time(self, time_str: str) -> time:
        """Parse time string HH:MM to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        """
        Detect FHM patterns in market data.

        Args:
            market_context: Market data and context

        Returns:
            StructureAnalysis with detected FHM events
        """
        symbol = market_context.symbol

        try:
            df = market_context.df_5m

            if df is None or len(df) < self.min_bars_required:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data"
                )

            events = []
            current_time = market_context.timestamp
            current_price = market_context.current_price

            # Check time window
            current_time_of_day = current_time.time()
            in_window = self.start_time <= current_time_of_day <= self.end_time

            if not in_window:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Outside FHM time window ({self.start_time}-{self.end_time})"
                )

            # Calculate RVOL
            rvol = self._calculate_rvol(df)

            if rvol < self.min_rvol:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"RVOL too low: {rvol:.2f}x < {self.min_rvol}x"
                )

            # Calculate price move from open
            open_price = df['open'].iloc[0]
            price_move_pct = ((current_price - open_price) / open_price) * 100
            abs_price_move = abs(price_move_pct)

            if abs_price_move < self.min_price_move_pct:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Price move too small: {abs_price_move:.2f}% < {self.min_price_move_pct}%"
                )

            # Check volume
            current_volume = df['volume'].iloc[-1]

            if current_volume < self.min_volume:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Volume too low: {current_volume:.0f} < {self.min_volume}"
                )

            # Get VWAP
            vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else None

            if price_move_pct > 0:
                # Long candidate
                vwap_check = not self.long_requires_above_vwap or vwap is None or current_price > vwap

                if not vwap_check:
                    return StructureAnalysis(
                        structure_detected=False,
                        events=[],
                        quality_score=0.0,
                        rejection_reason=f"Long requires price above VWAP"
                    )

                confidence = self._calculate_confidence(rvol, abs_price_move, current_volume)
                logger.debug(f"FHM: {symbol} LONG detected | RVOL={rvol:.2f}x, Move={price_move_pct:+.2f}%, Vol={current_volume/1000:.0f}k")

                event = StructureEvent(
                    symbol=symbol,
                    timestamp=current_time,
                    structure_type="first_hour_momentum_long",
                    side="long",
                    confidence=confidence,
                    levels={
                        "open": open_price,
                        "current": current_price,
                        "vwap": vwap
                    },
                    context={
                        "rvol": rvol,
                        "price_move_pct": price_move_pct,
                        "volume": current_volume,
                        "fhm_type": "momentum_long"
                    },
                    price=current_price
                )
                events.append(event)

            else:
                # Short candidate
                vwap_check = not self.short_requires_below_vwap or vwap is None or current_price < vwap

                if not vwap_check:
                    return StructureAnalysis(
                        structure_detected=False,
                        events=[],
                        quality_score=0.0,
                        rejection_reason=f"Short requires price below VWAP"
                    )

                confidence = self._calculate_confidence(rvol, abs_price_move, current_volume)
                logger.debug(f"FHM: {symbol} SHORT detected | RVOL={rvol:.2f}x, Move={price_move_pct:+.2f}%, Vol={current_volume/1000:.0f}k")

                event = StructureEvent(
                    symbol=symbol,
                    timestamp=current_time,
                    structure_type="first_hour_momentum_short",
                    side="short",
                    confidence=confidence,
                    levels={
                        "open": open_price,
                        "current": current_price,
                        "vwap": vwap
                    },
                    context={
                        "rvol": rvol,
                        "price_move_pct": price_move_pct,
                        "volume": current_volume,
                        "fhm_type": "momentum_short"
                    },
                    price=current_price
                )
                events.append(event)

            structure_detected = len(events) > 0
            quality_score = self._calculate_quality_score(rvol, abs_price_move, current_volume) if structure_detected else 0.0

            return StructureAnalysis(
                structure_detected=structure_detected,
                events=events,
                quality_score=quality_score,
                analysis_notes={
                    "rvol": rvol,
                    "price_move_pct": price_move_pct,
                    "volume": current_volume,
                    "timing_valid": True
                }
            )

        except Exception as e:
            logger.exception(f"FHM detection failed for {symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    def _calculate_rvol(self, df: pd.DataFrame) -> float:
        """Calculate Relative Volume (RVOL) - current volume vs 20-bar average."""
        try:
            if len(df) < 5:
                return 1.0

            current_volume = df['volume'].iloc[-1]
            lookback = min(20, len(df) - 1)
            avg_volume = df['volume'].iloc[-lookback-1:-1].mean()

            if pd.isna(avg_volume) or avg_volume <= 0:
                return 1.0

            return current_volume / avg_volume
        except Exception:
            return 1.0

    def _calculate_confidence(self, rvol: float, price_move_pct: float, volume: float) -> float:
        """Calculate confidence based on FHM signals."""
        # Base confidence
        if rvol >= self.high_rvol_threshold:
            confidence = self.confidence_high_rvol
        else:
            confidence = self.confidence_base

        # Bonus for strong RVOL
        if rvol >= 3.0:
            confidence += 0.3
        elif rvol >= 2.5:
            confidence += 0.2

        # Bonus for strong price move
        if price_move_pct >= 2.5:
            confidence += 0.2
        elif price_move_pct >= 2.0:
            confidence += 0.1

        # Bonus for high volume
        if volume >= 500000:
            confidence += 0.2
        elif volume >= 200000:
            confidence += 0.1

        return min(3.0, confidence)  # Cap at 3.0

    def _calculate_quality_score(self, rvol: float, price_move_pct: float, volume: float) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0.0

        # RVOL component (up to 40 points)
        if rvol >= 4.0:
            score += 40.0
        elif rvol >= 3.0:
            score += 30.0
        elif rvol >= 2.0:
            score += 20.0

        # Price move component (up to 30 points)
        if price_move_pct >= 3.0:
            score += 30.0
        elif price_move_pct >= 2.0:
            score += 20.0
        elif price_move_pct >= 1.5:
            score += 10.0

        # Volume component (up to 30 points)
        if volume >= 500000:
            score += 30.0
        elif volume >= 200000:
            score += 20.0
        elif volume >= 100000:
            score += 10.0

        return min(100.0, score)

    def should_detect_at_time(self, current_time: pd.Timestamp) -> bool:
        """
        Override: FHM detection should only happen within the FHM time window.
        """
        try:
            current_time_of_day = current_time.time()
            return self.start_time <= current_time_of_day <= self.end_time
        except Exception:
            return False

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """Check if timing is appropriate for FHM strategies."""
        return self.should_detect_at_time(current_time)

    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """Generate long FHM strategy."""
        symbol = market_context.symbol
        logger.debug(f"FHM: Planning long strategy for {symbol}")

        try:
            current_price = market_context.current_price
            atr = market_context.indicators.get("atr", 1.0) if market_context.indicators else 1.0

            # Calculate risk parameters
            risk_params = self.calculate_risk_params(current_price, market_context)

            # Generate exit levels
            exit_levels = self._generate_fhm_exits(current_price, atr, "long")

            return TradePlan(
                symbol=symbol,
                side="long",
                structure_type="first_hour_momentum",
                entry_price=current_price,
                risk_params=risk_params,
                exit_levels=exit_levels,
                qty=0,  # Will be calculated later
                notional=0.0,
                confidence=2.0,
                timestamp=market_context.timestamp,
                market_context=market_context
            )

        except Exception as e:
            logger.exception(f"FHM long strategy planning failed for {symbol}: {e}")
            return None

    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """Generate short FHM strategy."""
        symbol = market_context.symbol
        logger.debug(f"FHM: Planning short strategy for {symbol}")

        try:
            current_price = market_context.current_price
            atr = market_context.indicators.get("atr", 1.0) if market_context.indicators else 1.0

            # Calculate risk parameters
            risk_params = self.calculate_risk_params(current_price, market_context)

            # Generate exit levels
            exit_levels = self._generate_fhm_exits(current_price, atr, "short")

            return TradePlan(
                symbol=symbol,
                side="short",
                structure_type="first_hour_momentum",
                entry_price=current_price,
                risk_params=risk_params,
                exit_levels=exit_levels,
                qty=0,  # Will be calculated later
                notional=0.0,
                confidence=2.0,
                timestamp=market_context.timestamp,
                market_context=market_context
            )

        except Exception as e:
            logger.exception(f"FHM short strategy planning failed for {symbol}: {e}")
            return None

    def calculate_risk_params(self, entry_price: float, market_context: MarketContext) -> RiskParams:
        """Calculate FHM-specific risk parameters."""
        atr = market_context.indicators.get("atr", 1.0) if market_context.indicators else 1.0

        # Use ATR-based stop
        stop_distance = atr * self.stop_atr_mult

        # Determine direction from context (use open price comparison)
        df = market_context.df_5m
        if df is not None and len(df) > 0:
            open_price = df['open'].iloc[0]
            if entry_price > open_price:
                # Long position
                hard_sl = entry_price - stop_distance
            else:
                # Short position
                hard_sl = entry_price + stop_distance
        else:
            hard_sl = entry_price - stop_distance  # Default to long

        risk_per_share = abs(entry_price - hard_sl)

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            volatility_adj=1.0
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """Get FHM-specific exit levels."""
        return trade_plan.exit_levels

    def rank_setup_quality(self, market_context: MarketContext) -> float:
        """Score FHM setup quality (0-100)."""
        try:
            df = market_context.df_5m
            if df is None or len(df) < self.min_bars_required:
                return 0.0

            rvol = self._calculate_rvol(df)
            open_price = df['open'].iloc[0]
            price_move_pct = abs((market_context.current_price - open_price) / open_price) * 100
            volume = df['volume'].iloc[-1]

            return self._calculate_quality_score(rvol, price_move_pct, volume)

        except Exception as e:
            logger.exception(f"FHM ranking failed for {market_context.symbol}: {e}")
            return 0.0

    def _generate_fhm_exits(self, entry_price: float, atr: float, side: str) -> ExitLevels:
        """Generate FHM-specific exit levels."""
        stop_distance = atr * self.stop_atr_mult

        if side == "long":
            t1_level = entry_price + (atr * self.target_mult_t1)
            t2_level = entry_price + (atr * self.target_mult_t2)
            hard_sl = entry_price - stop_distance
        else:
            t1_level = entry_price - (atr * self.target_mult_t1)
            t2_level = entry_price - (atr * self.target_mult_t2)
            hard_sl = entry_price + stop_distance

        targets = [
            {"level": t1_level, "qty_pct": 60, "rr": abs(t1_level - entry_price) / abs(entry_price - hard_sl)},
            {"level": t2_level, "qty_pct": 40, "rr": abs(t2_level - entry_price) / abs(entry_price - hard_sl)}
        ]

        return ExitLevels(
            targets=targets,
            hard_sl=hard_sl,
            trail_to="t1",
            structure_exit={
                "vwap_cross": True,  # Exit if price crosses VWAP against us
                "time_exit": "10:30"  # Exit if no target hit by end of FHM window
            }
        )
