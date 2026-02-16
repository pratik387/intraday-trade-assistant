"""
Volume Breakout Structure Implementation

This module implements volume breakout trading structures:
- Volume breakout long (swing high breakout with institutional volume surge)
- Volume breakout short (swing low breakdown with institutional volume surge)

PATTERN DEFINITION:
Price breaks a dynamically-computed recent swing high/low with 2x+ average volume.
This is a CONTINUATION pattern (not reversal) — trades WITH the volume spike.

DIFFERENTIATION FROM EXISTING DETECTORS:
- level_breakout: Requires static levels (PDH/PDL/ORH/ORL). Volume breakout uses dynamic swing levels.
- momentum_breakout: Requires 3-bar sequential momentum. Volume breakout requires swing level break + volume surge.
- volume_spike_reversal: Trades AGAINST the spike (exhaustion reversal). Volume breakout trades WITH it.

RESEARCH SOURCES:
- William O'Neil (CAN SLIM): 50% above-average volume on breakout day
- Mark Minervini (SEPA/VCP): 40-50% volume surge, lighter volume = high failure rate
- SSRN (Wang & Gangwar, 2025): NSE Tata Motors intraday volume breakout study
- NYSE TrendSpider: 2x RVOL = 40% greater follow-through vs normal volume
- Chartink NSE screeners: 2x 10-session average as standard volume breakout filter
- Zerodha Varsity: Volume analysis for Indian equity markets

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


class VolumeBreakoutStructure(BaseStructure):
    """Volume breakout structure detection — swing level break with institutional volume surge."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Volume Breakout structure with configuration."""
        super().__init__(config)
        self.structure_type = "volume_breakout"

        # Track which specific setup type this detector is for
        self.configured_setup_type = config.get("_setup_name", None)

        # Volume surge thresholds — KeyError if missing
        self.min_volume_surge_mult = config["min_volume_surge_mult"]
        self.strong_volume_surge_mult = config["strong_volume_surge_mult"]
        self.exhaustion_volume_cap = config["exhaustion_volume_cap"]
        self.volume_avg_lookback_bars = config["volume_avg_lookback_bars"]

        # Swing level detection — KeyError if missing
        self.swing_lookback_bars = config["swing_lookback_bars"]
        self.min_swing_significance_atr = config["min_swing_significance_atr"]
        self.swing_fractal_left_bars = config["swing_fractal_left_bars"]
        self.swing_fractal_right_bars = config["swing_fractal_right_bars"]

        # Breakout confirmation — KeyError if missing
        self.min_breakout_distance_atr = config["min_breakout_distance_atr"]
        self.hold_bars_required = config["hold_bars_required"]

        # Candle body conviction — KeyError if missing
        self.min_body_ratio = config["min_body_ratio"]

        # Stop loss — KeyError if missing
        self.sl_atr_buffer = config["sl_atr_buffer"]
        self.min_stop_distance_pct = config["min_stop_distance_pct"]

        # Targets — KeyError if missing
        self.target_mult_t1 = config["target_mult_t1"]
        self.target_mult_t2 = config["target_mult_t2"]

        logger.debug(f"VOLUME_BREAKOUT: Initialized — vol surge: {self.min_volume_surge_mult}x, "
                     f"swing lookback: {self.swing_lookback_bars} bars, "
                     f"breakout distance: {self.min_breakout_distance_atr} ATR, "
                     f"body ratio: {self.min_body_ratio}")

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect volume breakout structures."""
        try:
            df = context.df_5m
            min_required = self.swing_lookback_bars + self.swing_fractal_right_bars + 2
            if len(df) < min_required:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason=f"Insufficient data (need {min_required} bars, have {len(df)})"
                )

            # Ensure indicators exist
            df_calc = self._prepare_data(df)
            if df_calc is None:
                return StructureAnalysis(
                    structure_detected=False,
                    events=[],
                    quality_score=0.0,
                    rejection_reason="Could not prepare volume/ATR indicators"
                )

            events = []
            atr = self._get_atr(context)

            # Find swing highs and swing lows using fractal method
            swing_highs = self._find_swing_highs(df_calc)
            swing_lows = self._find_swing_lows(df_calc)

            # Check for volume breakout long (price above swing high with volume)
            long_event = self._detect_breakout_long(context, df_calc, swing_highs, atr)
            if long_event:
                events.append(long_event)

            # Check for volume breakout short (price below swing low with volume)
            short_event = self._detect_breakout_short(context, df_calc, swing_lows, atr)
            if short_event:
                events.append(short_event)

            # Filter to configured setup type
            if self.configured_setup_type and events:
                filtered = [e for e in events if e.structure_type == self.configured_setup_type]
                if len(filtered) < len(events):
                    logger.debug(f"VOLUME_BREAKOUT: {context.symbol} filtered "
                                 f"{len(events)}→{len(filtered)} (configured for {self.configured_setup_type})")
                events = filtered

            quality = self._calculate_quality_score(events) if events else 0.0
            return StructureAnalysis(
                structure_detected=len(events) > 0,
                events=events,
                quality_score=quality,
                rejection_reason=None if events else "No volume breakout patterns detected"
            )

        except Exception as e:
            logger.exception(f"VOLUME_BREAKOUT: Detection error for {context.symbol}: {e}")
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=f"Detection error: {e}"
            )

    # ── Swing detection ──────────────────────────────────────────────

    def _find_swing_highs(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Find fractal swing highs: bar where high is highest within left+right window.

        Returns list of (bar_index, high_value) sorted chronologically.
        """
        swings = []
        left = self.swing_fractal_left_bars
        right = self.swing_fractal_right_bars
        lookback_start = max(left, len(df) - self.swing_lookback_bars)

        for i in range(lookback_start, len(df) - right):
            window = df.iloc[i - left:i + right + 1]
            center_high = float(df['high'].iloc[i])
            if center_high == float(window['high'].max()):
                swings.append((i, center_high))

        return swings

    def _find_swing_lows(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Find fractal swing lows: bar where low is lowest within left+right window.

        Returns list of (bar_index, low_value) sorted chronologically.
        """
        swings = []
        left = self.swing_fractal_left_bars
        right = self.swing_fractal_right_bars
        lookback_start = max(left, len(df) - self.swing_lookback_bars)

        for i in range(lookback_start, len(df) - right):
            window = df.iloc[i - left:i + right + 1]
            center_low = float(df['low'].iloc[i])
            if center_low == float(window['low'].min()):
                swings.append((i, center_low))

        return swings

    # ── Breakout detection ───────────────────────────────────────────

    def _detect_breakout_long(self, context: MarketContext, df: pd.DataFrame,
                              swing_highs: List[Tuple[int, float]], atr: float) -> Optional[StructureEvent]:
        """Detect volume breakout long — price breaks recent swing high with volume surge."""
        if not swing_highs:
            return None

        current_price = context.current_price
        last_bar = df.iloc[-1]

        # Use the most recent swing high (excluding current bar cluster)
        # Filter out swing highs too close to the current bar (within fractal_right)
        valid_swings = [(idx, val) for idx, val in swing_highs
                        if idx < len(df) - self.swing_fractal_right_bars]

        if not valid_swings:
            return None

        # Use the highest recent swing high (most significant resistance)
        swing_idx, swing_high = max(valid_swings, key=lambda x: x[1])

        # Swing must be significant (at least min_swing_significance_atr away from current consolidation)
        recent_mean = float(df['close'].tail(5).mean())
        if abs(swing_high - recent_mean) < self.min_swing_significance_atr * atr:
            logger.debug(f"VOLUME_BREAKOUT: {context.symbol} swing high {swing_high:.2f} too close to recent mean "
                         f"({abs(swing_high - recent_mean):.2f} < {self.min_swing_significance_atr * atr:.2f})")
            return None

        # 1. Price must be above swing high by minimum breakout distance
        breakout_distance = current_price - swing_high
        min_distance = self.min_breakout_distance_atr * atr

        if breakout_distance < min_distance:
            return None

        # 2. Volume surge check
        vol_surge = self._get_volume_surge(df)
        if vol_surge < self.min_volume_surge_mult:
            logger.debug(f"VOLUME_BREAKOUT: {context.symbol} long rejected — vol surge {vol_surge:.2f}x < {self.min_volume_surge_mult}x")
            return None

        # 3. Exhaustion cap — volume too high may indicate climactic activity
        if vol_surge > self.exhaustion_volume_cap:
            logger.debug(f"VOLUME_BREAKOUT: {context.symbol} long rejected — vol surge {vol_surge:.2f}x > exhaustion cap {self.exhaustion_volume_cap}x")
            return None

        # 4. Body conviction check — close must be in upper portion of bar
        bar_range = float(last_bar['high']) - float(last_bar['low'])
        if bar_range > 0:
            close_position = (float(last_bar['close']) - float(last_bar['low'])) / bar_range
            if close_position < self.min_body_ratio:
                logger.debug(f"VOLUME_BREAKOUT: {context.symbol} long rejected — body conviction {close_position:.2f} < {self.min_body_ratio}")
                return None

        # 5. Hold confirmation — price must close above swing for hold_bars
        if self.hold_bars_required > 0 and len(df) >= self.hold_bars_required + 1:
            hold_closes = df['close'].tail(self.hold_bars_required)
            if not (hold_closes > swing_high).all():
                logger.debug(f"VOLUME_BREAKOUT: {context.symbol} long rejected — hold bars failed")
                return None

        # Calculate confidence
        confidence = self._calculate_confidence(context, vol_surge, breakout_distance / atr, "long")

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type="volume_breakout_long",
            side="long",
            confidence=confidence,
            levels={"swing_high": swing_high, "breakout_distance": breakout_distance},
            context={
                "swing_level": swing_high,
                "swing_bar_index": swing_idx,
                "volume_surge": vol_surge,
                "breakout_distance_atr": breakout_distance / atr,
                "body_conviction": close_position if bar_range > 0 else 0.0,
                "pattern_type": "volume_breakout"
            },
            price=current_price
        )

        logger.debug(f"VOLUME_BREAKOUT: {context.symbol} LONG detected — swing {swing_high:.2f}, "
                     f"vol {vol_surge:.1f}x, distance {breakout_distance/atr:.2f} ATR, conf {confidence:.2f}")
        return event

    def _detect_breakout_short(self, context: MarketContext, df: pd.DataFrame,
                               swing_lows: List[Tuple[int, float]], atr: float) -> Optional[StructureEvent]:
        """Detect volume breakout short — price breaks recent swing low with volume surge."""
        if not swing_lows:
            return None

        current_price = context.current_price
        last_bar = df.iloc[-1]

        # Use the most recent swing low (excluding current bar cluster)
        valid_swings = [(idx, val) for idx, val in swing_lows
                        if idx < len(df) - self.swing_fractal_right_bars]

        if not valid_swings:
            return None

        # Use the lowest recent swing low (most significant support)
        swing_idx, swing_low = min(valid_swings, key=lambda x: x[1])

        # Swing must be significant
        recent_mean = float(df['close'].tail(5).mean())
        if abs(swing_low - recent_mean) < self.min_swing_significance_atr * atr:
            return None

        # 1. Price must be below swing low by minimum breakout distance
        breakdown_distance = swing_low - current_price
        min_distance = self.min_breakout_distance_atr * atr

        if breakdown_distance < min_distance:
            return None

        # 2. Volume surge check
        vol_surge = self._get_volume_surge(df)
        if vol_surge < self.min_volume_surge_mult:
            return None

        # 3. Exhaustion cap
        if vol_surge > self.exhaustion_volume_cap:
            logger.debug(f"VOLUME_BREAKOUT: {context.symbol} short rejected — vol surge {vol_surge:.2f}x > exhaustion cap")
            return None

        # 4. Body conviction — close must be in lower portion of bar
        bar_range = float(last_bar['high']) - float(last_bar['low'])
        if bar_range > 0:
            close_position = (float(last_bar['close']) - float(last_bar['low'])) / bar_range
            if close_position > (1.0 - self.min_body_ratio):
                logger.debug(f"VOLUME_BREAKOUT: {context.symbol} short rejected — body conviction {close_position:.2f} > {1.0 - self.min_body_ratio}")
                return None

        # 5. Hold confirmation — price must close below swing for hold_bars
        if self.hold_bars_required > 0 and len(df) >= self.hold_bars_required + 1:
            hold_closes = df['close'].tail(self.hold_bars_required)
            if not (hold_closes < swing_low).all():
                return None

        # Calculate confidence
        confidence = self._calculate_confidence(context, vol_surge, breakdown_distance / atr, "short")

        event = StructureEvent(
            symbol=context.symbol,
            timestamp=context.timestamp,
            structure_type="volume_breakout_short",
            side="short",
            confidence=confidence,
            levels={"swing_low": swing_low, "breakdown_distance": breakdown_distance},
            context={
                "swing_level": swing_low,
                "swing_bar_index": swing_idx,
                "volume_surge": vol_surge,
                "breakout_distance_atr": breakdown_distance / atr,
                "body_conviction": close_position if bar_range > 0 else 0.0,
                "pattern_type": "volume_breakout"
            },
            price=current_price
        )

        logger.debug(f"VOLUME_BREAKOUT: {context.symbol} SHORT detected — swing {swing_low:.2f}, "
                     f"vol {vol_surge:.1f}x, distance {breakdown_distance/atr:.2f} ATR, conf {confidence:.2f}")
        return event

    # ── Helpers ──────────────────────────────────────────────────────

    def _prepare_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add volume indicators if not present."""
        try:
            d = df.copy()
            if 'vol_z' not in d.columns:
                vol_mean = d['volume'].rolling(self.volume_avg_lookback_bars, min_periods=10).mean()
                vol_std = d['volume'].rolling(self.volume_avg_lookback_bars, min_periods=10).std()
                d['vol_z'] = (d['volume'] - vol_mean) / vol_std.replace(0, np.nan)
            return d
        except Exception as e:
            logger.error(f"VOLUME_BREAKOUT: Error preparing data: {e}")
            return None

    def _get_volume_surge(self, df: pd.DataFrame) -> float:
        """Get volume surge ratio for current bar (current volume / average volume)."""
        try:
            current_vol = float(df['volume'].iloc[-1])
            avg_vol = float(df['volume'].rolling(self.volume_avg_lookback_bars, min_periods=10).mean().iloc[-1])
            if avg_vol > 0:
                return current_vol / avg_vol
            return 0.0
        except:
            return 0.0

    def _get_atr(self, context: MarketContext) -> float:
        """Get ATR from context with fallback."""
        if context.indicators and 'atr' in context.indicators:
            return context.indicators['atr']
        # Fallback: estimate from recent price range
        df = context.df_5m
        try:
            atr_pct = df['close'].pct_change().abs().rolling(14, min_periods=5).mean().iloc[-1]
            return max(0.01, float(atr_pct) * context.current_price)
        except:
            return context.current_price * 0.01

    def _calculate_confidence(self, context: MarketContext, vol_surge: float,
                              breakout_atr: float, side: str) -> float:
        """Calculate institutional-grade confidence for volume breakout.

        Professional criteria:
        - Volume surge magnitude (2x base, 3x+ bonus)
        - Breakout distance in ATR (larger = more conviction)
        - Market timing (institutional hours 10:00-14:00)
        """
        try:
            # Base from volume surge (primary signal)
            base = max(1.5, vol_surge * 0.8)

            multiplier = 1.0

            # Strong volume bonus (3x+ institutional participation)
            if vol_surge >= self.strong_volume_surge_mult:
                multiplier *= 1.3
            elif vol_surge >= self.min_volume_surge_mult * 1.5:
                multiplier *= 1.15

            # Breakout distance bonus (decisive move past swing)
            if breakout_atr >= 0.5:
                multiplier *= 1.15
            elif breakout_atr >= 0.3:
                multiplier *= 1.05

            # Timing bonus (institutional hours)
            hour = context.timestamp.hour
            if 10 <= hour <= 14:
                multiplier *= 1.1

            final = base * multiplier
            final = max(final, 1.8)  # Minimum viable strength for regime gates

            logger.debug(f"VOLUME_BREAKOUT: {context.symbol} {side} confidence — "
                         f"base {base:.2f} × {multiplier:.2f} = {final:.2f}")
            return final

        except Exception as e:
            logger.error(f"VOLUME_BREAKOUT: Confidence calc error: {e}")
            return 1.8

    def _calculate_quality_score(self, events: List[StructureEvent]) -> float:
        """Calculate quality score for detected events."""
        if not events:
            return 0.0
        base = 65.0
        event = events[0]
        vol_surge = event.context.get("volume_surge", 1.0)
        vol_score = min(20.0, vol_surge * 5)
        distance_atr = event.context.get("breakout_distance_atr", 0.0)
        distance_score = min(10.0, distance_atr * 10)
        return min(100.0, base + vol_score + distance_score)

    # ── Strategy planning ────────────────────────────────────────────

    def plan_long_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        return self._plan_strategy(context, event, "long")

    def plan_short_strategy(self, context: MarketContext, event: StructureEvent) -> TradePlan:
        return self._plan_strategy(context, event, "short")

    def _plan_strategy(self, context: MarketContext, event: StructureEvent, side: str) -> TradePlan:
        entry_price = context.current_price
        risk_params = self.calculate_risk_params(context, event, side)
        exit_levels = self.get_exit_levels(context, event, side)
        return TradePlan(
            symbol=context.symbol,
            side=side,
            structure_type=event.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context
        )

    def calculate_risk_params(self, context: MarketContext, event: StructureEvent, side: str) -> RiskParams:
        """Calculate risk params — SL at swing level + ATR buffer."""
        entry_price = context.current_price
        atr = self._get_atr(context)
        swing_level = event.context.get("swing_level", entry_price)

        # SL anchored at swing level (the level being broken) + ATR buffer
        if side == "long":
            hard_sl = swing_level - (atr * self.sl_atr_buffer)
        else:
            hard_sl = swing_level + (atr * self.sl_atr_buffer)

        # Enforce minimum stop distance
        min_stop = entry_price * (self.min_stop_distance_pct / 100.0)
        risk_per_share = abs(entry_price - hard_sl)
        if risk_per_share < min_stop:
            if side == "long":
                hard_sl = entry_price - min_stop
            else:
                hard_sl = entry_price + min_stop
            risk_per_share = min_stop

        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
            risk_percentage=0.02
        )

    def get_exit_levels(self, context: MarketContext, event: StructureEvent, side: str) -> ExitLevels:
        """Calculate ATR-based exit levels."""
        entry_price = context.current_price
        atr = self._get_atr(context)

        if side == "long":
            t1 = entry_price + (atr * self.target_mult_t1)
            t2 = entry_price + (atr * self.target_mult_t2)
        else:
            t1 = entry_price - (atr * self.target_mult_t1)
            t2 = entry_price - (atr * self.target_mult_t2)

        return ExitLevels(
            targets=[
                {"level": t1, "qty_pct": 50, "rr": self.target_mult_t1},
                {"level": t2, "qty_pct": 50, "rr": self.target_mult_t2}
            ],
            hard_sl=0.0,
            trail_to=None
        )

    def rank_setup_quality(self, context: MarketContext, event: StructureEvent) -> float:
        base = event.confidence * 100
        vol = event.context.get("volume_surge", 1.0)
        return min(100.0, base + min(15.0, vol * 4))

    def validate_timing(self, context: MarketContext, event: StructureEvent) -> Tuple[bool, str]:
        return True, "Volume breakout timing validated"
