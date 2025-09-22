# services/events/structure_events.py
"""
Detects structural intraday events driven by price/volume/VWAP/levels.

Events:
- breakout_long / breakout_short         : PDH/PDL/ORH/ORL break + hold + volume
- vwap_reclaim_long / vwap_lose_short    : VWAP cross + hold + volume
- squeeze_release_long / squeeze_release_short : recent vol expansion vs prior regime
- failure_fade_long / failure_fade_short : failed breakout at PDH/PDL

Strict config (NO in-code defaults):
  entry_config.json must define:
    - k_atr              (e.g., 0.25)
    - hold_bars          (e.g., 1)
    - vol_z_required     (e.g., 1.5)
    - width_window       (e.g., 40)
    - expansion_ratio    (e.g., 1.5)

Assumptions:
- DataFrames have naive IST DatetimeIndex (use utils.time_util.ensure_naive_ist_index).
- Columns expected where applicable: ["open","high","low","close","volume","vwap?","ts?"].
  If "ts" column is absent, the index is used as timestamp.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config.logging_config import get_agent_logger
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index

logger = get_agent_logger()


@dataclass
class StructureEvent:
    setup: str
    ts: pd.Timestamp
    level_name: str
    strength: float


class StructureEventDetector:
    """Structure-based event detector (config-driven; no code defaults)."""

    def __init__(self) -> None:
        # Fail fast if required keys are missing (filters_setup enforces this)
        self.cfg = load_filters()

    # ---- internal helpers -------------------------------------------------

    def _get_time_adjusted_vol_threshold(self, ts: pd.Timestamp) -> float:
        """
        Apply time-based multipliers to vol_z_required for early market conditions.

        Before 10:30am: 50% reduction (more lenient)
        10:30am-12:00pm: 25% reduction
        After 12:00pm: Standard threshold
        """
        base_vol_z = float(self.cfg["vol_z_required"])

        try:
            time_minutes = ts.hour * 60 + ts.minute

            # Market hours: 9:15am = 555 minutes, 10:30am = 630, 12:00pm = 720
            if time_minutes < 630:  # Before 10:30am
                return base_vol_z * 0.5  # 50% reduction for early market
            elif time_minutes < 720:  # 10:30am - 12:00pm
                return base_vol_z * 0.75  # 25% reduction for mid-morning
            else:  # After 12:00pm
                return base_vol_z  # Standard threshold
        except Exception:
            return base_vol_z  # Fallback to standard

    @staticmethod
    def _safe_ts(d: pd.DataFrame) -> pd.Timestamp:
        """Return last timestamp from 'ts' column if present, else from index."""
        if "ts" in d.columns:
            return pd.to_datetime(d["ts"].iloc[-1])
        return pd.to_datetime(d.index[-1])

    @staticmethod
    def _vol_z(d: pd.DataFrame, win: int = 30, minp: int = 10) -> pd.Series:
        """
        Volume Z-score ~ (vol - mean) / std using a rolling window.
        Safe against zero std (returns 0 where std==0).
        """
        mu = d["volume"].rolling(win, min_periods=minp).mean()
        sd = d["volume"].rolling(win, min_periods=minp).std(ddof=0)
        z = (d["volume"] - mu) / sd.replace(0, np.nan)
        return z.fillna(0)

    # ---- detectors --------------------------------------------------------

    def detect_level_breakouts(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Breakouts across PDH/PDL/ORH/ORL with hold & volume confirmation.

        Requires in entry_config.json:
          k_atr, hold_bars, vol_z_required
        """
        try:
            if df is None or df.empty or len(df) < 5:
                return []

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Simple ATR proxy using abs returns mean over 14 bars
            atr = float(
                max(
                    1e-9,
                    d["close"].pct_change().abs().rolling(14, min_periods=5).mean().iloc[-1],
                )
            )
            last = d.iloc[-1]

            k_atr = float(self.cfg["k_atr"])
            min_breakout_atr_mult = float(self.cfg.get("min_breakout_atr_mult", 0.3))
            hold_bars = int(self.cfg["hold_bars"])

            # Use time-adjusted volume threshold
            ts = self._safe_ts(d)
            vol_z_required = self._get_time_adjusted_vol_threshold(ts)

            out: List[StructureEvent] = []
            for name, lvl in (levels or {}).items():
                if lvl is None or not np.isfinite(lvl):
                    continue

                # Long breakout above PDH/ORH
                if name in ("PDH", "ORH"):
                    # Calculate breakout size - distance price moved above level
                    breakout_size = last["close"] - lvl
                    min_breakout_size = min_breakout_atr_mult * atr

                    # Check for volume surge on breakout bar specifically
                    volume_surge_ok = True
                    require_surge = self.cfg.get("require_breakout_bar_volume_surge", True)
                    if require_surge:
                        # Check if current bar (breakout bar) has volume surge
                        current_volume = float(d["volume"].iloc[-1])
                        avg_volume_window = d["volume"].rolling(20, min_periods=10).mean()
                        current_avg_volume = float(avg_volume_window.iloc[-1])

                        if current_avg_volume > 0:
                            volume_ratio = current_volume / current_avg_volume
                            min_volume_ratio = float(self.cfg.get("breakout_volume_min_ratio", 1.4))
                            volume_surge_ok = volume_ratio >= min_volume_ratio

                    cond = (
                        (last["close"] > lvl + k_atr * atr)
                        and (d["close"] > lvl).tail(hold_bars).all()
                        and d["vol_z"].iloc[-1] >= vol_z_required
                        and breakout_size >= min_breakout_size  # Minimum breakout size filter
                        and volume_surge_ok  # Volume surge on breakout bar
                    )
                    if cond:
                        # Apply Smart Money Concepts enhancement
                        base_strength = float(d["vol_z"].iloc[-1])
                        smc_strength = self._calculate_smc_strength(d, base_strength, lvl, is_long=True)

                        # Apply session timing weight
                        ts = self._safe_ts(d)
                        session_weight = self._get_session_weight(ts)
                        final_strength = smc_strength * session_weight

                        # Only create event if SMC criteria are met (no false breakouts)
                        if not self._is_false_breakout(d, lvl, is_long=True):
                            evt = StructureEvent("breakout_long", ts, name, final_strength)
                            out.append(evt)
                            logger.info(f"structure_event: PROFESSIONAL SMC {evt} (strength: {final_strength:.2f})")
                        else:
                            logger.debug(f"structure_event: REJECTED liquidity grab at {name} {lvl}")

                # Short breakdown below PDL/ORL
                if name in ("PDL", "ORL"):
                    # Calculate breakdown size - distance price moved below level
                    breakdown_size = lvl - last["close"]
                    min_breakdown_size = min_breakout_atr_mult * atr

                    # Check for volume surge on breakdown bar specifically
                    volume_surge_ok = True
                    require_surge = self.cfg.get("require_breakout_bar_volume_surge", True)
                    if require_surge:
                        # Check if current bar (breakdown bar) has volume surge
                        current_volume = float(d["volume"].iloc[-1])
                        avg_volume_window = d["volume"].rolling(20, min_periods=10).mean()
                        current_avg_volume = float(avg_volume_window.iloc[-1])

                        if current_avg_volume > 0:
                            volume_ratio = current_volume / current_avg_volume
                            min_volume_ratio = float(self.cfg.get("breakout_volume_min_ratio", 1.4))
                            volume_surge_ok = volume_ratio >= min_volume_ratio

                    cond = (
                        (last["close"] < lvl - k_atr * atr)
                        and (d["close"] < lvl).tail(hold_bars).all()
                        and d["vol_z"].iloc[-1] >= vol_z_required
                        and breakdown_size >= min_breakdown_size  # Minimum breakdown size filter
                        and volume_surge_ok  # Volume surge on breakdown bar
                    )
                    if cond:
                        # Apply Smart Money Concepts enhancement
                        base_strength = float(d["vol_z"].iloc[-1])
                        smc_strength = self._calculate_smc_strength(d, base_strength, lvl, is_long=False)

                        # Apply session timing weight
                        ts = self._safe_ts(d)
                        session_weight = self._get_session_weight(ts)
                        final_strength = smc_strength * session_weight

                        # Only create event if SMC criteria are met (no false breakouts)
                        if not self._is_false_breakout(d, lvl, is_long=False):
                            evt = StructureEvent("breakout_short", ts, name, final_strength)
                            out.append(evt)
                            logger.info(f"structure_event: PROFESSIONAL SMC {evt} (strength: {final_strength:.2f})")
                        else:
                            logger.debug(f"structure_event: REJECTED liquidity grab at {name} {lvl}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_level_breakouts error: {e}")
            return []

    def _get_session_weight(self, ts: pd.Timestamp) -> float:
        """
        Get session-based weight for professional trading.
        London (8-12 IST) and NY (18:30-22:30 IST) get higher weights.
        Ultra-efficient with cached lookups.
        """
        try:
            # Convert to IST hour (cache-friendly)
            hour = ts.hour

            # Professional session weights (institutional trading hours)
            if 8 <= hour <= 12:      # London session
                return 1.3
            elif 18 <= hour <= 22:   # NY session
                return 1.4
            elif 13 <= hour <= 17:   # London-NY overlap
                return 1.2
            else:                    # Asian/dead hours
                return 0.9
        except:
            return 1.0

    def _is_false_breakout(self, df: pd.DataFrame, level: float, is_long: bool, lookback: int = 3) -> bool:
        """
        Detect false breakouts (liquidity grabs) using Smart Money Concepts.

        A false breakout occurs when:
        1. Price breaks through level
        2. But quickly reverses back inside range within lookback bars

        Ultra-efficient check - just compares price to level over last few bars.
        """
        try:
            if len(df) < lookback:
                return False

            recent_closes = df["close"].tail(lookback).values

            if is_long:
                # For long breakouts, check if price fell back below level quickly
                broke_above = any(close > level for close in recent_closes)
                fell_back_below = recent_closes[-1] <= level
                return broke_above and fell_back_below
            else:
                # For short breakdowns, check if price rallied back above level quickly
                broke_below = any(close < level for close in recent_closes)
                rallied_back_above = recent_closes[-1] >= level
                return broke_below and rallied_back_above
        except:
            return False

    def _calculate_smc_strength(self, df: pd.DataFrame, base_strength: float, level: float, is_long: bool) -> float:
        """
        Calculate Smart Money Concepts strength score with professional criteria.

        Factors:
        - Volume spike confirmation
        - Structure break vs liquidity grab
        - Order block confluence
        - Multi-timeframe alignment

        Returns enhanced strength score.
        """
        try:
            strength = base_strength

            # 1. Volume spike detection (1ms)
            if len(df) >= 5:
                recent_vol = df["volume"].tail(5).mean()
                avg_vol = df["volume"].rolling(20, min_periods=10).mean().iloc[-1]
                if recent_vol > avg_vol * 1.5:  # 50% above average
                    strength *= 1.3  # Volume spike bonus

            # 2. Structure break confirmation (1ms)
            # Check if this is continuation of trend vs reversal
            if len(df) >= 10:
                price_trend = df["close"].tail(10).is_monotonic_increasing if is_long else df["close"].tail(10).is_monotonic_decreasing
                if price_trend:
                    strength *= 1.2  # Trend continuation bonus

            # 3. Order block confluence (1ms)
            # Check if breakout occurred from consolidation zone (last opposing candle)
            if len(df) >= 5:
                recent_range = df["high"].tail(5).max() - df["low"].tail(5).min()
                recent_closes = df["close"].tail(5)
                consolidation = recent_range / df["close"].iloc[-1] < 0.02  # Less than 2% range

                if consolidation:
                    strength *= 1.4  # Order block accumulation bonus

            # 4. False breakout penalty
            if self._is_false_breakout(df, level, is_long):
                strength *= 0.5  # Heavy penalty for liquidity grabs

            return float(strength)

        except:
            return base_strength

    def detect_range_deviations(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Institutional range trading strategy for choppy markets.

        Detects:
        - Range boundary deviations (liquidity grabs)
        - Smart money accumulation/distribution zones
        - False breakout reversals

        Based on institutional Smart Money Concepts for sideways markets.
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return []

            d = ensure_naive_ist_index(df.copy())
            out: List[StructureEvent] = []

            # Define range boundaries from recent price action
            lookback_range = 20  # Look back 20 bars for range detection
            range_data = d.tail(lookback_range)

            if len(range_data) < 10:
                return []

            # Calculate range boundaries
            range_high = range_data["high"].max()
            range_low = range_data["low"].min()
            range_mid = (range_high + range_low) / 2
            range_size = range_high - range_low

            # Minimum range size for valid range (prevent noise)
            min_range_pct = 0.015  # 1.5% minimum range
            if range_size / range_mid < min_range_pct:
                return []

            current = d.iloc[-1]
            current_price = current["close"]

            # Professional range deviation parameters
            deviation_threshold = 0.002  # 0.2% beyond range boundary
            volume_multiplier = 1.3      # Volume confirmation requirement

            # Volume confirmation
            recent_vol = d["volume"].tail(3).mean()
            avg_vol = d["volume"].rolling(15, min_periods=10).mean().iloc[-1]
            volume_confirmed = recent_vol > avg_vol * volume_multiplier

            # SESSION TIMING: Boost range strategies during high-volume sessions
            ts = self._safe_ts(d)
            session_weight = self._get_session_weight(ts)

            # RANGE DEVIATION LONG - Buy deviation below range low (institutions accumulating)
            deviation_below = current_price < (range_low * (1 - deviation_threshold))

            if deviation_below and volume_confirmed:
                # Check for reversal signals (smart money defending the range)
                reversal_strength = 0.0

                # Volume surge on deviation = liquidity grab
                if volume_confirmed:
                    reversal_strength += 1.0

                # Price quickly returning to range
                last_3_closes = d["close"].tail(3).values
                if len(last_3_closes) >= 2 and last_3_closes[-1] > last_3_closes[-2]:
                    reversal_strength += 0.5  # Reversal momentum

                # Apply session timing
                final_strength = reversal_strength * session_weight

                if final_strength > 0.8:  # Quality threshold
                    evt = StructureEvent("range_deviation_long", ts, "RANGE_LOW", final_strength)
                    out.append(evt)
                    logger.info(f"structure_event: INSTITUTIONAL RANGE {evt} (deviation below {range_low:.2f})")

            # RANGE DEVIATION SHORT - Sell deviation above range high (institutions distributing)
            deviation_above = current_price > (range_high * (1 + deviation_threshold))

            if deviation_above and volume_confirmed:
                # Check for reversal signals
                reversal_strength = 0.0

                if volume_confirmed:
                    reversal_strength += 1.0

                # Price quickly returning to range
                last_3_closes = d["close"].tail(3).values
                if len(last_3_closes) >= 2 and last_3_closes[-1] < last_3_closes[-2]:
                    reversal_strength += 0.5  # Reversal momentum

                # Apply session timing
                final_strength = reversal_strength * session_weight

                if final_strength > 0.8:  # Quality threshold
                    evt = StructureEvent("range_deviation_short", ts, "RANGE_HIGH", final_strength)
                    out.append(evt)
                    logger.info(f"structure_event: INSTITUTIONAL RANGE {evt} (deviation above {range_high:.2f})")

            # RANGE MEAN REVERSION - Trade back toward range middle
            distance_from_mid = abs(current_price - range_mid) / range_size

            if distance_from_mid > 0.3:  # Near range boundaries
                if current_price > range_mid:  # Near top, expect reversion down
                    reversion_strength = distance_from_mid * session_weight
                    if reversion_strength > 0.4:
                        evt = StructureEvent("range_mean_reversion_short", ts, "RANGE_TOP", reversion_strength)
                        out.append(evt)
                        logger.info(f"structure_event: MEAN REVERSION {evt}")
                else:  # Near bottom, expect reversion up
                    reversion_strength = distance_from_mid * session_weight
                    if reversion_strength > 0.4:
                        evt = StructureEvent("range_mean_reversion_long", ts, "RANGE_BOTTOM", reversion_strength)
                        out.append(evt)
                        logger.info(f"structure_event: MEAN REVERSION {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_range_deviations error: {e}")
            return []

    def detect_vwap_cross_and_hold(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Professional VWAP reclaim/lose detection with institutional-grade criteria.

        Enhanced with:
        - Volume trend confirmation (increasing volume on reclaim)
        - Momentum confirmation (RSI slope alignment)
        - Retest validation (recent pullback before reclaim)
        - Session timing weights (London/NY hours get boost)
        - False breakout filtering

        Requires in entry_config.json:
          hold_bars, vol_z_required
        """
        try:
            if df is None or df.empty or "vwap" not in df.columns:
                return []

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            hold_bars = int(self.cfg["hold_bars"])

            # Use time-adjusted volume threshold
            ts = self._safe_ts(d)
            vol_z_required = self._get_time_adjusted_vol_threshold(ts)

            # Professional enhancement parameters (ultra-efficient)
            volume_trend_window = 3  # Check last 3 bars for volume trend
            retest_window = 10       # Look back 10 bars for retest validation
            momentum_threshold = 0.1 # RSI slope threshold for momentum confirmation

            # Session timing weights (cached for performance)
            session_weight = self._get_session_weight(ts)

            out: List[StructureEvent] = []

            # VWAP RECLAIM LONG - Professional Criteria
            basic_reclaim = (d["close"] > d["vwap"]).tail(hold_bars).all()
            basic_volume = d["vol_z"].iloc[-1] >= vol_z_required

            if basic_reclaim and basic_volume:
                # Professional enhancement checks (< 3ms total)

                # 1. Volume trend confirmation - increasing volume on reclaim
                vol_increasing = len(d) >= volume_trend_window and d["volume"].tail(volume_trend_window).is_monotonic_increasing

                # 2. Momentum confirmation - use existing RSI slope if available
                momentum_bullish = True  # Default true if no RSI data
                if "rsi_slope" in d.columns and len(d) > 0:
                    momentum_bullish = d["rsi_slope"].iloc[-1] > momentum_threshold

                # 3. Retest validation - was price below VWAP recently then reclaimed?
                recent_below_vwap = False
                if len(d) >= retest_window:
                    recent_below_vwap = (d["close"] < d["vwap"]).tail(retest_window).any()

                # 4. Calculate professional strength score
                strength_score = float(d["vol_z"].iloc[-1])

                # Apply professional criteria bonuses
                if vol_increasing:
                    strength_score *= 1.2  # 20% bonus for volume confirmation
                if momentum_bullish:
                    strength_score *= 1.1  # 10% bonus for momentum alignment
                if recent_below_vwap:
                    strength_score *= 1.3  # 30% bonus for proper retest setup

                # Apply session timing weight
                strength_score *= session_weight

                # Only create event if it meets professional criteria
                professional_criteria_met = vol_increasing or momentum_bullish or recent_below_vwap

                if professional_criteria_met:
                    ts = self._safe_ts(d)
                    evt = StructureEvent("vwap_reclaim_long", ts, "VWAP", strength_score)
                    out.append(evt)
                    logger.info(f"structure_event: PROFESSIONAL {evt} (vol_trend:{vol_increasing}, momentum:{momentum_bullish}, retest:{recent_below_vwap})")

            # VWAP LOSE SHORT - Professional Criteria
            basic_lose = (d["close"] < d["vwap"]).tail(hold_bars).all()

            if basic_lose and basic_volume:
                # Professional enhancement checks for short setup

                # 1. Volume trend confirmation - increasing volume on breakdown
                vol_increasing = len(d) >= volume_trend_window and d["volume"].tail(volume_trend_window).is_monotonic_increasing

                # 2. Momentum confirmation - bearish RSI slope
                momentum_bearish = True  # Default true if no RSI data
                if "rsi_slope" in d.columns and len(d) > 0:
                    momentum_bearish = d["rsi_slope"].iloc[-1] < -momentum_threshold

                # 3. Retest validation - was price above VWAP recently then lost?
                recent_above_vwap = False
                if len(d) >= retest_window:
                    recent_above_vwap = (d["close"] > d["vwap"]).tail(retest_window).any()

                # 4. Calculate professional strength score
                strength_score = float(d["vol_z"].iloc[-1])

                # Apply professional criteria bonuses
                if vol_increasing:
                    strength_score *= 1.2  # Volume confirmation
                if momentum_bearish:
                    strength_score *= 1.1  # Momentum alignment
                if recent_above_vwap:
                    strength_score *= 1.3  # Proper retest setup

                # Apply session timing weight
                strength_score *= session_weight

                # Only create event if it meets professional criteria
                professional_criteria_met = vol_increasing or momentum_bearish or recent_above_vwap

                if professional_criteria_met:
                    ts = self._safe_ts(d)
                    evt = StructureEvent("vwap_lose_short", ts, "VWAP", strength_score)
                    out.append(evt)
                    logger.info(f"structure_event: PROFESSIONAL {evt} (vol_trend:{vol_increasing}, momentum:{momentum_bearish}, retest:{recent_above_vwap})")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_vwap_cross_and_hold error: {e}")
            return []

    def detect_squeeze_release(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Volatility regime expansion:
          recent width (σ/μ over last 5 bars) vs prior average width over a window.

        Requires in entry_config.json:
          width_window, expansion_ratio
        """
        try:
            width_window = int(self.cfg["width_window"])
            expansion_ratio = float(self.cfg["expansion_ratio"])

            if df is None or df.empty or len(df) < width_window + 5:
                return []

            d = ensure_naive_ist_index(df.copy())

            # Width = std/mean over 20 bars (min_periods=20) as a simple squeeze proxy
            std20 = d["close"].rolling(20, min_periods=20).std(ddof=0)
            mean20 = d["close"].rolling(20, min_periods=20).mean().replace(0, np.nan)
            width = (std20 / mean20).replace([np.inf, -np.inf], np.nan)

            recent = width.iloc[-5:].mean()
            prior = width.iloc[-(width_window + 5) : -5].mean()

            if pd.isna(prior) or prior == 0 or pd.isna(recent):
                return []

            if recent > expansion_ratio * prior:
                r3 = float(d["close"].pct_change(3).iloc[-1])
                setup = "squeeze_release_long" if r3 > 0 else "squeeze_release_short"
                ts = self._safe_ts(d)
                evt = StructureEvent(setup, ts, "SQUEEZE", abs(r3))
                logger.info(f"structure_event: {evt}")
                return [evt]

            return []

        except Exception as e:
            logger.exception(f"structure_event: detect_squeeze_release error: {e}")
            return []

    def detect_level_failure_fade(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Failure fade: previous bar pierced level, next bar closed back within it.
        PDH -> short fade; PDL -> long fade.
        """
        try:
            if df is None or df.empty or len(df) < 2:
                return []

            d = ensure_naive_ist_index(df.copy())
            last, prev = d.iloc[-1], d.iloc[-2]

            out: List[StructureEvent] = []

            PDH = (levels or {}).get("PDH")
            PDL = (levels or {}).get("PDL")

            if PDH is not None and np.isfinite(PDH):
                if (prev["high"] > PDH) and (last["close"] < PDH):
                    ts = self._safe_ts(d)
                    evt = StructureEvent("failure_fade_short", ts, "PDH", 1.0)
                    out.append(evt)
                    logger.info(f"structure_event: {evt}")

            if PDL is not None and np.isfinite(PDL):
                if (prev["low"] < PDL) and (last["close"] > PDL):
                    ts = self._safe_ts(d)
                    evt = StructureEvent("failure_fade_long", ts, "PDL", 1.0)
                    out.append(evt)
                    logger.info(f"structure_event: {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_level_failure_fade error: {e}")
            return []
        
    # put this inside class StructureEventDetector
    def detect_setups(self, symbol: str, df5m_tail: pd.DataFrame, levels: Dict[str, float] | None):
        """
        Adapter for TradeDecisionGate:
        - runs our specific detectors
        - maps StructureEvent -> SetupCandidate
        - returns List[SetupCandidate] sorted by strength desc
        """
        # local import avoids any chance of circular import at module load time
        from services.gates.trade_decision_gate import SetupCandidate

        d = ensure_naive_ist_index(df5m_tail.copy())

        evts = []
        evts += self.detect_level_breakouts(d, levels or {})
        evts += self.detect_vwap_cross_and_hold(d)
        evts += self.detect_squeeze_release(d)
        evts += self.detect_level_failure_fade(d, levels or {})

        # INSTITUTIONAL RANGE TRADING - Profit from choppy markets like smart money
        evts += self.detect_range_deviations(d, levels or {})
        
        # New setup detections
        evts += self.detect_gap_fills(d, levels or {})
        evts += self.detect_flag_continuations(d)
        evts += self.detect_support_resistance_bounces(d, levels or {})
        evts += self.detect_orb_breakouts(d, levels or {})
        evts += self.detect_vwap_mean_reversions(d)
        evts += self.detect_volume_spike_reversals(d)
        evts += self.detect_trend_pullbacks(d)
        evts += self.detect_range_rejections(d, levels or {})

        # Momentum-based structures (fallback for early market when level-based fails)
        evts += self.detect_momentum_structures(d)

        # Institutional concepts
        evts += self.detect_order_blocks(d)
        evts += self.detect_fair_value_gaps(d)
        evts += self.detect_liquidity_sweeps(d, levels or {})
        evts += self.detect_premium_discount_zones(d, levels or {})
        evts += self.detect_break_of_structure(d)
        evts += self.detect_change_of_character(d)

        setups = []
        for e in evts or []:
            # e.setup should be one of the SetupType literals defined in trade_decision_gate.py:
            #   Traditional: 'breakout_long','breakout_short','vwap_reclaim_long','vwap_lose_short',
            #               'squeeze_release_long','squeeze_release_short','failure_fade_long','failure_fade_short'
            #   Extended: 'gap_fill_long','orb_breakout_long','trend_pullback_long', etc.
            #   Institutional: 'order_block_long','fair_value_gap_long','liquidity_sweep_long',
            #                 'premium_zone_short','break_of_structure_long','change_of_character_long'
            reasons = []
            lvl = getattr(e, "level_name", None)
            if lvl:
                reasons.append(f"level:{lvl}")
            strength = float(getattr(e, "strength", 1.0))
            setups.append(SetupCandidate(setup_type=e.setup, strength=strength, reasons=reasons))

        # strongest first (gate will pick the max anyway)
        setups.sort(key=lambda s: s.strength, reverse=True)
        return setups
    # ================== NEW SETUP DETECTION METHODS ==================
    
    def detect_gap_fills(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Detect gap fill opportunities - price moving to close intraday gaps
        """
        try:
            if len(df) < 5:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            gap_cfg = cfg.get("gap_fill_long", {})
            
            if not gap_cfg.get("enabled", False):
                return []
                
            min_gap_pct = gap_cfg.get("min_gap_pct", 0.3) / 100.0
            max_gap_pct = gap_cfg.get("max_gap_pct", 2.5) / 100.0
            require_volume = gap_cfg.get("require_volume_confirmation", True)
            
            # Get opening price (first bar) and previous close  
            if "PDC" not in levels:
                return []
                
            pdc = levels["PDC"]
            open_price = df["open"].iloc[0]
            current_price = df["close"].iloc[-1]
            current_vol_z = self._vol_z(df).iloc[-1] if require_volume else 2.0
            
            # Calculate gap percentage
            gap_pct = abs(open_price - pdc) / pdc
            
            if min_gap_pct <= gap_pct <= max_gap_pct:
                # Gap up - look for fill (short opportunity)
                if open_price > pdc and current_price < (pdc + open_price) / 2:
                    if not require_volume or current_vol_z > 1.0:
                        events.append(StructureEvent(
                            setup="gap_fill_short",
                            ts=self._safe_ts(df),
                            level_name="GAP_FILL",
                            strength=min(3.0, gap_pct * 10 + current_vol_z)
                        ))
                
                # Gap down - look for fill (long opportunity) 
                elif open_price < pdc and current_price > (pdc + open_price) / 2:
                    if not require_volume or current_vol_z > 1.0:
                        events.append(StructureEvent(
                            setup="gap_fill_long", 
                            ts=self._safe_ts(df),
                            level_name="GAP_FILL",
                            strength=min(3.0, gap_pct * 10 + current_vol_z)
                        ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_gap_fills error: {e}")
            return []
    
    def detect_flag_continuations(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect flag/pennant continuation patterns
        """
        try:
            if len(df) < 15:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            flag_cfg = cfg.get("flag_continuation_long", {})
            
            if not flag_cfg.get("enabled", False):
                return []
                
            min_consol_bars = flag_cfg.get("min_consolidation_bars", 3)
            max_consol_bars = flag_cfg.get("max_consolidation_bars", 12)
            min_trend_strength = flag_cfg.get("min_trend_strength", 1.5)
            
            # Look for prior trend (last 10 bars before consolidation)
            trend_period = 10
            consol_period = min(max_consol_bars, len(df) - trend_period)
            
            if consol_period < min_consol_bars:
                return []
                
            # Split data: trend portion vs consolidation portion
            trend_data = df.iloc[-(trend_period + consol_period):-consol_period]
            consol_data = df.iloc[-consol_period:]
            
            if len(trend_data) < 5 or len(consol_data) < min_consol_bars:
                return []
            
            # Measure trend strength
            trend_start = trend_data["close"].iloc[0] 
            trend_end = trend_data["close"].iloc[-1]
            trend_pct = (trend_end - trend_start) / trend_start
            
            # Measure consolidation tightness
            consol_range = (consol_data["high"].max() - consol_data["low"].min()) / consol_data["close"].mean()
            current_vol_z = self._vol_z(df).iloc[-1]
            
            # Flag continuation long
            if trend_pct > min_trend_strength / 100.0 and consol_range < 0.02:
                if df["close"].iloc[-1] > consol_data["high"].max() * 1.001:  # Breakout above consolidation
                    events.append(StructureEvent(
                        setup="flag_continuation_long",
                        ts=self._safe_ts(df), 
                        level_name="FLAG",
                        strength=min(3.0, abs(trend_pct) * 100 + current_vol_z)
                    ))
            
            # Flag continuation short
            elif trend_pct < -min_trend_strength / 100.0 and consol_range < 0.02:
                if df["close"].iloc[-1] < consol_data["low"].min() * 0.999:  # Breakout below consolidation
                    events.append(StructureEvent(
                        setup="flag_continuation_short",
                        ts=self._safe_ts(df),
                        level_name="FLAG", 
                        strength=min(3.0, abs(trend_pct) * 100 + current_vol_z)
                    ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_flag_continuations error: {e}")
            return []
    
    def detect_support_resistance_bounces(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Detect bounces off key support/resistance levels
        """
        try:
            if len(df) < 5:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            bounce_cfg = cfg.get("support_bounce_long", {})
            
            if not bounce_cfg.get("enabled", False):
                return []
                
            min_touches = bounce_cfg.get("min_touches", 2)
            tolerance_pct = bounce_cfg.get("bounce_tolerance_pct", 0.1) / 100.0
            require_volume = bounce_cfg.get("require_volume_spike", True)
            
            current_price = df["close"].iloc[-1]
            current_vol_z = self._vol_z(df).iloc[-1] if require_volume else 2.0
            
            for level_name, level_price in levels.items():
                if level_price <= 0:
                    continue
                    
                # Count touches near this level in recent history
                touches = 0
                for i in range(max(0, len(df) - 20), len(df)):
                    low_price = df["low"].iloc[i]
                    high_price = df["high"].iloc[i]
                    
                    # Check if price touched this level
                    if (abs(low_price - level_price) / level_price <= tolerance_pct or 
                        abs(high_price - level_price) / level_price <= tolerance_pct):
                        touches += 1
                
                if touches >= min_touches:
                    distance_pct = abs(current_price - level_price) / level_price
                    
                    # Support bounce (long)
                    if (level_name in ["PDL", "ORL"] and 
                        current_price > level_price and 
                        distance_pct <= tolerance_pct):
                        
                        if not require_volume or current_vol_z > 1.5:
                            events.append(StructureEvent(
                                setup="support_bounce_long",
                                ts=self._safe_ts(df),
                                level_name=level_name,
                                strength=min(3.0, touches + current_vol_z)
                            ))
                    
                    # Resistance bounce (short)  
                    elif (level_name in ["PDH", "ORH"] and 
                          current_price < level_price and
                          distance_pct <= tolerance_pct):
                        
                        if not require_volume or current_vol_z > 1.5:
                            events.append(StructureEvent(
                                setup="resistance_bounce_short", 
                                ts=self._safe_ts(df),
                                level_name=level_name,
                                strength=min(3.0, touches + current_vol_z)
                            ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_support_resistance_bounces error: {e}")
            return []
    
    def detect_orb_breakouts(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Detect opening range breakouts
        """
        try:
            if len(df) < 5:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            orb_cfg = cfg.get("orb_breakout_long", {})
            
            if not orb_cfg.get("enabled", False):
                return []
                
            orb_minutes = orb_cfg.get("orb_minutes", 15)
            min_range_pct = orb_cfg.get("min_range_pct", 0.5) / 100.0
            volume_mult = orb_cfg.get("breakout_volume_mult", 1.5)
            
            # Check if we have ORH/ORL levels
            if "ORH" not in levels or "ORL" not in levels:
                return []
                
            orh = levels["ORH"] 
            orl = levels["ORL"]
            orb_range_pct = (orh - orl) / ((orh + orl) / 2)
            
            if orb_range_pct < min_range_pct:
                return []
                
            current_price = df["close"].iloc[-1]
            current_vol_z = self._vol_z(df).iloc[-1]
            
            # ORB breakout long
            if current_price > orh * 1.001:  # Clean break above ORH
                if current_vol_z > volume_mult:
                    events.append(StructureEvent(
                        setup="orb_breakout_long",
                        ts=self._safe_ts(df),
                        level_name="ORH", 
                        strength=min(3.0, orb_range_pct * 100 + current_vol_z)
                    ))
            
            # ORB breakout short
            elif current_price < orl * 0.999:  # Clean break below ORL
                if current_vol_z > volume_mult:
                    events.append(StructureEvent(
                        setup="orb_breakout_short",
                        ts=self._safe_ts(df),
                        level_name="ORL",
                        strength=min(3.0, orb_range_pct * 100 + current_vol_z)
                    ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_orb_breakouts error: {e}")
            return []
    
    def detect_vwap_mean_reversions(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect VWAP mean reversion opportunities when price is stretched
        """
        try:
            if len(df) < 10 or "vwap" not in df.columns:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            mean_rev_cfg = cfg.get("vwap_mean_reversion_long", {})
            
            if not mean_rev_cfg.get("enabled", False):
                return []
                
            min_distance_bps = mean_rev_cfg.get("min_distance_bps", 150) / 10000.0
            max_distance_bps = mean_rev_cfg.get("max_distance_bps", 400) / 10000.0
            require_rsi = mean_rev_cfg.get("require_oversold_rsi", True)
            
            current_price = df["close"].iloc[-1]
            vwap_price = df["vwap"].iloc[-1]
            distance_pct = abs(current_price - vwap_price) / vwap_price
            
            if min_distance_bps <= distance_pct <= max_distance_bps:
                # Calculate simple RSI for last 14 bars
                rsi = 50  # Default neutral
                if require_rsi and len(df) >= 14:
                    closes = df["close"].tail(14)
                    deltas = closes.diff()
                    gains = deltas.clip(lower=0)
                    losses = -deltas.clip(upper=0)
                    avg_gain = gains.mean()
                    avg_loss = losses.mean()
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                
                # Mean reversion long (price below VWAP, oversold)
                if current_price < vwap_price:
                    if not require_rsi or rsi < 35:
                        events.append(StructureEvent(
                            setup="vwap_mean_reversion_long",
                            ts=self._safe_ts(df),
                            level_name="VWAP",
                            strength=min(3.0, distance_pct * 1000 + (50 - rsi) / 10)
                        ))
                
                # Mean reversion short (price above VWAP, overbought)
                elif current_price > vwap_price:
                    short_cfg = cfg.get("vwap_mean_reversion_short", {})
                    short_require_rsi = short_cfg.get("require_overbought_rsi", True)
                    
                    if not short_require_rsi or rsi > 65:
                        events.append(StructureEvent(
                            setup="vwap_mean_reversion_short",
                            ts=self._safe_ts(df), 
                            level_name="VWAP",
                            strength=min(3.0, distance_pct * 1000 + (rsi - 50) / 10)
                        ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_vwap_mean_reversions error: {e}")
            return []
    
    def detect_volume_spike_reversals(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect volume spike exhaustion reversals
        """
        try:
            if len(df) < 10:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            vol_spike_cfg = cfg.get("volume_spike_reversal_long", {})
            
            if not vol_spike_cfg.get("enabled", False):
                return []
                
            min_vol_mult = vol_spike_cfg.get("min_volume_mult", 3.0)
            min_body_pct = vol_spike_cfg.get("min_body_size_pct", 1.0) / 100.0
            
            current_vol_z = self._vol_z(df).iloc[-1]
            last_bar = df.iloc[-1]
            
            if current_vol_z >= min_vol_mult:
                # Calculate body size
                body_size_pct = abs(last_bar["close"] - last_bar["open"]) / last_bar["open"]
                
                if body_size_pct >= min_body_pct:
                    # Volume spike reversal long (big red candle on volume)
                    if last_bar["close"] < last_bar["open"]:
                        events.append(StructureEvent(
                            setup="volume_spike_reversal_long",
                            ts=self._safe_ts(df),
                            level_name="VOLUME_SPIKE", 
                            strength=min(3.0, current_vol_z + body_size_pct * 100)
                        ))
                    
                    # Volume spike reversal short (big green candle on volume)
                    else:
                        events.append(StructureEvent(
                            setup="volume_spike_reversal_short",
                            ts=self._safe_ts(df),
                            level_name="VOLUME_SPIKE",
                            strength=min(3.0, current_vol_z + body_size_pct * 100)
                        ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_volume_spike_reversals error: {e}")
            return []
    
    def detect_trend_pullbacks(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect pullbacks in established trends
        """
        try:
            if len(df) < 15:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            pullback_cfg = cfg.get("trend_pullback_long", {})
            
            if not pullback_cfg.get("enabled", False):
                return []
                
            min_pullback_pct = pullback_cfg.get("min_pullback_pct", 0.5) / 100.0
            max_pullback_pct = pullback_cfg.get("max_pullback_pct", 2.0) / 100.0
            require_trend = pullback_cfg.get("require_trend_confirmation", True)
            
            # Look at trend over last 10 bars
            trend_bars = min(10, len(df) - 3)
            trend_data = df.iloc[-(trend_bars + 3):-3]  # Skip last 3 bars for pullback measurement
            pullback_data = df.tail(3)
            
            if len(trend_data) < 5:
                return []
            
            # Measure trend
            trend_start = trend_data["close"].iloc[0]
            trend_end = trend_data["close"].iloc[-1] 
            trend_pct = (trend_end - trend_start) / trend_start
            
            # Measure pullback from trend high/low
            current_price = df["close"].iloc[-1]
            
            # Uptrend pullback (long opportunity)
            if trend_pct > 0.01:  # 1% uptrend
                trend_high = trend_data["high"].max()
                pullback_pct = (trend_high - current_price) / trend_high
                
                if min_pullback_pct <= pullback_pct <= max_pullback_pct:
                    if not require_trend or trend_pct > 0.02:
                        events.append(StructureEvent(
                            setup="trend_pullback_long",
                            ts=self._safe_ts(df),
                            level_name="TREND_PULLBACK",
                            strength=min(3.0, trend_pct * 100 + pullback_pct * 50)
                        ))
            
            # Downtrend pullback (short opportunity)
            elif trend_pct < -0.01:  # 1% downtrend
                trend_low = trend_data["low"].min()
                pullback_pct = (current_price - trend_low) / trend_low
                
                if min_pullback_pct <= pullback_pct <= max_pullback_pct:
                    if not require_trend or trend_pct < -0.02:
                        events.append(StructureEvent(
                            setup="trend_pullback_short", 
                            ts=self._safe_ts(df),
                            level_name="TREND_PULLBACK",
                            strength=min(3.0, abs(trend_pct) * 100 + pullback_pct * 50)
                        ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_trend_pullbacks error: {e}")
            return []
    
    def detect_momentum_structures(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect momentum-based structure events for early market conditions.

        These are alternative structures when traditional level-based detection fails.
        Combines price momentum + volume confirmation without requiring specific levels.

        Detects:
        - Strong momentum breakouts (price + volume surge)
        - Momentum reversals (exhaustion patterns)
        - Early trend continuation patterns
        """
        try:
            if df is None or df.empty or len(df) < 10:
                return []

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Get time-adjusted volume threshold
            ts = self._safe_ts(d)
            vol_z_required = self._get_time_adjusted_vol_threshold(ts)

            # Calculate momentum indicators
            d["returns_1"] = d["close"].pct_change()
            d["returns_3"] = d["close"].pct_change(3)  # 3-bar momentum
            d["returns_5"] = d["close"].pct_change(5)  # 5-bar momentum

            # Volume momentum
            d["vol_ma"] = d["volume"].rolling(10, min_periods=5).mean()
            d["vol_surge"] = d["volume"] / d["vol_ma"]

            last = d.iloc[-1]
            out: List[StructureEvent] = []

            # Momentum Breakout Long - Strong upward momentum + volume
            if (
                last["returns_3"] > 0.015  # 1.5% move in 3 bars
                and last["returns_1"] > 0.005  # Last bar positive
                and d["returns_1"].tail(2).sum() > 0.01  # 2-bar momentum
                and last["vol_z"] >= vol_z_required * 0.8  # Slightly relaxed volume
                and last["vol_surge"] > 1.3  # Volume surge vs average
            ):
                evt = StructureEvent("momentum_breakout_long", ts, "MOMENTUM", float(last["returns_3"] * 100))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            # Momentum Breakout Short - Strong downward momentum + volume
            elif (
                last["returns_3"] < -0.015  # 1.5% decline in 3 bars
                and last["returns_1"] < -0.005  # Last bar negative
                and d["returns_1"].tail(2).sum() < -0.01  # 2-bar momentum
                and last["vol_z"] >= vol_z_required * 0.8  # Slightly relaxed volume
                and last["vol_surge"] > 1.3  # Volume surge vs average
            ):
                evt = StructureEvent("momentum_breakout_short", ts, "MOMENTUM", float(abs(last["returns_3"]) * 100))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            # Early Trend Continuation Long - Consistent upward pressure
            elif (
                last["returns_5"] > 0.02  # 2% move in 5 bars
                and d["returns_1"].tail(3).sum() > 0.01  # 3-bar upward bias
                and (d["returns_1"] > 0).tail(3).sum() >= 2  # At least 2 of last 3 bars positive
                and last["vol_z"] >= vol_z_required * 0.6  # More relaxed volume for continuation
            ):
                evt = StructureEvent("trend_continuation_long", ts, "TREND", float(last["returns_5"] * 100))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            # Early Trend Continuation Short - Consistent downward pressure
            elif (
                last["returns_5"] < -0.02  # 2% decline in 5 bars
                and d["returns_1"].tail(3).sum() < -0.01  # 3-bar downward bias
                and (d["returns_1"] < 0).tail(3).sum() >= 2  # At least 2 of last 3 bars negative
                and last["vol_z"] >= vol_z_required * 0.6  # More relaxed volume for continuation
            ):
                evt = StructureEvent("trend_continuation_short", ts, "TREND", float(abs(last["returns_5"]) * 100))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_momentum_structures error: {e}")
            return []

    def detect_range_rejections(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Detect rejections at range boundaries
        """
        try:
            if len(df) < 8:
                return []
                
            events = []
            cfg = self.cfg.get("setups", {})
            rejection_cfg = cfg.get("range_rejection_long", {})
            
            if not rejection_cfg.get("enabled", False):
                return []
                
            min_range_bars = rejection_cfg.get("min_range_duration_bars", 5) 
            wick_pct = rejection_cfg.get("rejection_wick_pct", 0.6)
            require_volume = rejection_cfg.get("require_volume_confirmation", True)
            
            if len(df) < min_range_bars:
                return []
            
            # Look for range-bound price action
            range_data = df.tail(min_range_bars)
            range_high = range_data["high"].max()
            range_low = range_data["low"].min()
            range_size = range_high - range_low
            
            if range_size <= 0:
                return []
            
            last_bar = df.iloc[-1]
            current_vol_z = self._vol_z(df).iloc[-1] if require_volume else 2.0
            
            # Calculate wick sizes
            upper_wick = last_bar["high"] - max(last_bar["open"], last_bar["close"])
            lower_wick = min(last_bar["open"], last_bar["close"]) - last_bar["low"]
            body_size = abs(last_bar["close"] - last_bar["open"])
            
            if body_size <= 0:
                return []
            
            upper_wick_ratio = upper_wick / (upper_wick + body_size + lower_wick)
            lower_wick_ratio = lower_wick / (upper_wick + body_size + lower_wick)
            
            # Range rejection long (rejection at range low)
            if (last_bar["low"] <= range_low * 1.002 and  # Near range low
                lower_wick_ratio >= wick_pct and  # Big lower wick
                last_bar["close"] > last_bar["open"]):  # Bullish close
                
                if not require_volume or current_vol_z > 1.0:
                    events.append(StructureEvent(
                        setup="range_rejection_long",
                        ts=self._safe_ts(df),
                        level_name="RANGE_LOW",
                        strength=min(3.0, lower_wick_ratio * 3 + current_vol_z)
                    ))
            
            # Range rejection short (rejection at range high)
            elif (last_bar["high"] >= range_high * 0.998 and  # Near range high
                  upper_wick_ratio >= wick_pct and  # Big upper wick
                  last_bar["close"] < last_bar["open"]):  # Bearish close
                
                if not require_volume or current_vol_z > 1.0:
                    events.append(StructureEvent(
                        setup="range_rejection_short",
                        ts=self._safe_ts(df),
                        level_name="RANGE_HIGH", 
                        strength=min(3.0, upper_wick_ratio * 3 + current_vol_z)
                    ))
            
            return events
            
        except Exception as e:
            logger.exception(f"detect_range_rejections error: {e}")
            return []

    def detect_order_blocks(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect Order Blocks - institutional accumulation/distribution zones.

        Order Blocks are the last opposing candle before a significant move, representing
        areas where institutions placed large orders. They act as strong support/resistance.

        Detection Logic:
        1. Find significant moves (>0.8% + volume surge)
        2. Identify the last opposing candle before the move
        3. Mark that candle's range as an Order Block
        4. Detect when price returns to test the Order Block
        """
        try:
            if df is None or df.empty or len(df) < 10:
                return []

            events = []
            cfg = self.cfg.get("setups", {})
            ob_cfg = cfg.get("order_blocks", {})

            if not ob_cfg.get("enabled", True):  # Default enabled for institutional concepts
                return []

            min_move_pct = ob_cfg.get("min_significant_move_pct", 0.5) / 100.0  # 0.5% (relaxed from 0.8%)
            min_volume_surge = ob_cfg.get("min_volume_surge_ratio", 1.5)  # 1.5x average volume (relaxed from 2x)
            lookback_bars = ob_cfg.get("order_block_lookback", 20)  # Look back 20 bars max
            ob_test_tolerance = ob_cfg.get("test_tolerance_pct", 0.15) / 100.0  # 0.15% tolerance

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)
            d["returns"] = d["close"].pct_change()

            # Calculate volume surge vs recent average
            d["vol_ma10"] = d["volume"].rolling(10, min_periods=5).mean()
            d["vol_surge"] = d["volume"] / d["vol_ma10"]

            current_price = d["close"].iloc[-1]
            current_bar_idx = len(d) - 1

            # Look for significant moves in recent history
            search_start = max(5, current_bar_idx - lookback_bars)

            for move_start_idx in range(search_start, current_bar_idx - 2):
                # Check for bullish significant move (potential bearish OB)
                move_bars = d.iloc[move_start_idx:move_start_idx + 5]  # 5-bar move window
                if len(move_bars) < 3:
                    continue

                move_start_price = move_bars["close"].iloc[0]
                move_end_price = move_bars["close"].iloc[-1]
                move_pct = (move_end_price - move_start_price) / move_start_price

                # Check if this was a significant move with volume
                move_had_volume = (move_bars["vol_surge"] > min_volume_surge).any()

                if abs(move_pct) >= min_move_pct and move_had_volume:
                    # Find the last opposing candle before this move
                    ob_candle_idx = None

                    # Look backwards from move start to find last opposing candle
                    for look_idx in range(move_start_idx - 1, max(0, move_start_idx - 8), -1):
                        candle = d.iloc[look_idx]

                        if move_pct > 0:  # Bullish move - look for last bearish candle (bearish OB)
                            if candle["close"] < candle["open"]:  # Bearish candle
                                ob_candle_idx = look_idx
                                break
                        else:  # Bearish move - look for last bullish candle (bullish OB)
                            if candle["close"] > candle["open"]:  # Bullish candle
                                ob_candle_idx = look_idx
                                break

                    if ob_candle_idx is not None:
                        ob_candle = d.iloc[ob_candle_idx]
                        ob_high = ob_candle["high"]
                        ob_low = ob_candle["low"]

                        # Check if current price is testing this Order Block
                        if move_pct > 0:  # Bearish OB (resistance zone)
                            # Price should be approaching from below to test resistance
                            if (ob_low <= current_price <= ob_high * (1 + ob_test_tolerance) and
                                current_price > move_start_price):  # Above the move start

                                # Strength based on move size and time since OB formation
                                bars_since_ob = current_bar_idx - ob_candle_idx
                                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))  # Decay over 30 bars
                                strength = min(3.0, abs(move_pct) * 100 * time_decay)

                                events.append(StructureEvent(
                                    setup="order_block_short",
                                    ts=self._safe_ts(d),
                                    level_name=f"BEARISH_OB_{ob_candle_idx}",
                                    strength=float(strength)
                                ))

                        else:  # Bullish OB (support zone)
                            # Price should be approaching from above to test support
                            if (ob_low * (1 - ob_test_tolerance) <= current_price <= ob_high and
                                current_price < move_start_price):  # Below the move start

                                # Strength based on move size and time since OB formation
                                bars_since_ob = current_bar_idx - ob_candle_idx
                                time_decay = max(0.5, 1.0 - (bars_since_ob / 30.0))  # Decay over 30 bars
                                strength = min(3.0, abs(move_pct) * 100 * time_decay)

                                events.append(StructureEvent(
                                    setup="order_block_long",
                                    ts=self._safe_ts(d),
                                    level_name=f"BULLISH_OB_{ob_candle_idx}",
                                    strength=float(strength)
                                ))

            # Remove duplicate events (keep strongest)
            if events:
                # Group by setup type and keep strongest
                by_setup = {}
                for evt in events:
                    if evt.setup not in by_setup or evt.strength > by_setup[evt.setup].strength:
                        by_setup[evt.setup] = evt
                events = list(by_setup.values())

                for evt in events:
                    logger.info(f"structure_event: {evt}")

            return events

        except Exception as e:
            logger.exception(f"detect_order_blocks error: {e}")
            return []

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect Fair Value Gaps (FVGs) - price imbalances indicating institutional activity.

        FVGs occur when there's a gap between:
        - High of candle[i-1] and Low of candle[i+1] (bullish FVG)
        - Low of candle[i-1] and High of candle[i+1] (bearish FVG)

        These represent inefficient price areas that institutions often fill.
        """
        try:
            if df is None or df.empty or len(df) < 5:
                return []

            events = []
            cfg = self.cfg.get("setups", {})
            fvg_cfg = cfg.get("fair_value_gaps", {})

            if not fvg_cfg.get("enabled", True):  # Default enabled for institutional concepts
                return []

            min_gap_pct = fvg_cfg.get("min_gap_pct", 0.1) / 100.0  # 0.1% minimum gap
            max_gap_pct = fvg_cfg.get("max_gap_pct", 1.5) / 100.0  # 1.5% maximum gap
            require_volume_spike = fvg_cfg.get("require_volume_spike", True)
            min_volume_mult = fvg_cfg.get("min_volume_mult", 1.5)  # 1.5x average volume
            fvg_fill_tolerance = fvg_cfg.get("fill_tolerance_pct", 0.05) / 100.0  # 0.05% tolerance

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Calculate volume surge vs recent average
            d["vol_ma10"] = d["volume"].rolling(10, min_periods=5).mean()
            d["vol_surge"] = d["volume"] / d["vol_ma10"]

            current_price = d["close"].iloc[-1]
            current_bar_idx = len(d) - 1

            # Look for FVGs in recent history (last 20 bars)
            lookback_bars = min(20, len(d) - 3)

            for i in range(2, lookback_bars):  # Start from bar 2 (need i-1, i, i+1)
                if i >= len(d) - 1:  # Ensure we don't go out of bounds
                    continue

                candle_before = d.iloc[i - 1]
                candle_middle = d.iloc[i]
                candle_after = d.iloc[i + 1]

                # Check for volume spike on the middle candle (aggressive move)
                volume_condition = True
                if require_volume_spike:
                    vol_surge = d["vol_surge"].iloc[i]
                    volume_condition = vol_surge >= min_volume_mult

                if not volume_condition:
                    continue

                # Bullish FVG: Gap between high[i-1] and low[i+1]
                # This happens when price moves up aggressively, leaving a gap below
                if (candle_before["high"] < candle_after["low"] and
                    candle_middle["close"] > candle_middle["open"]):  # Middle candle is bullish

                    gap_size = candle_after["low"] - candle_before["high"]
                    gap_pct = gap_size / candle_before["high"]

                    if min_gap_pct <= gap_pct <= max_gap_pct:
                        # FVG boundaries
                        fvg_top = candle_after["low"]
                        fvg_bottom = candle_before["high"]

                        # Check if current price is testing this FVG (coming back to fill it)
                        if (fvg_bottom * (1 - fvg_fill_tolerance) <= current_price <=
                            fvg_top * (1 + fvg_fill_tolerance)):

                            # Strength based on gap size and volume surge
                            volume_strength = d["vol_surge"].iloc[i]
                            strength = min(3.0, gap_pct * 500 + volume_strength * 0.5)

                            events.append(StructureEvent(
                                setup="fair_value_gap_long",
                                ts=self._safe_ts(d),
                                level_name=f"BULLISH_FVG_{i}",
                                strength=float(strength)
                            ))

                # Bearish FVG: Gap between low[i-1] and high[i+1]
                # This happens when price moves down aggressively, leaving a gap above
                elif (candle_before["low"] > candle_after["high"] and
                      candle_middle["close"] < candle_middle["open"]):  # Middle candle is bearish

                    gap_size = candle_before["low"] - candle_after["high"]
                    gap_pct = gap_size / candle_after["high"]

                    if min_gap_pct <= gap_pct <= max_gap_pct:
                        # FVG boundaries
                        fvg_top = candle_before["low"]
                        fvg_bottom = candle_after["high"]

                        # Check if current price is testing this FVG (coming back to fill it)
                        if (fvg_bottom * (1 - fvg_fill_tolerance) <= current_price <=
                            fvg_top * (1 + fvg_fill_tolerance)):

                            # Strength based on gap size and volume surge
                            volume_strength = d["vol_surge"].iloc[i]
                            strength = min(3.0, gap_pct * 500 + volume_strength * 0.5)

                            events.append(StructureEvent(
                                setup="fair_value_gap_short",
                                ts=self._safe_ts(d),
                                level_name=f"BEARISH_FVG_{i}",
                                strength=float(strength)
                            ))

            # Remove duplicate events (keep strongest)
            if events:
                # Group by setup type and keep strongest
                by_setup = {}
                for evt in events:
                    if evt.setup not in by_setup or evt.strength > by_setup[evt.setup].strength:
                        by_setup[evt.setup] = evt
                events = list(by_setup.values())

                for evt in events:
                    logger.info(f"structure_event: {evt}")

            return events

        except Exception as e:
            logger.exception(f"detect_fair_value_gaps error: {e}")
            return []

    def detect_liquidity_sweeps(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Detect Liquidity Sweeps - institutional stop hunt patterns.

        Liquidity sweeps occur when:
        1. Price briefly breaks above/below a key level (stops triggered)
        2. Volume spikes as stops are hit
        3. Price quickly reverses back within the level (trap)
        4. Often accompanied by long wicks showing rejection

        This is a classic institutional tactic to hunt retail stops before the real move.
        """
        try:
            if df is None or df.empty or len(df) < 5:
                return []

            events = []
            cfg = self.cfg.get("setups", {})
            sweep_cfg = cfg.get("liquidity_sweeps", {})

            if not sweep_cfg.get("enabled", True):  # Default enabled for institutional concepts
                return []

            min_sweep_pct = sweep_cfg.get("min_sweep_distance_pct", 0.05) / 100.0  # 0.05% beyond level
            max_sweep_pct = sweep_cfg.get("max_sweep_distance_pct", 0.3) / 100.0   # 0.3% max sweep
            min_volume_surge = sweep_cfg.get("min_volume_surge", 2.0)  # 2x volume spike
            min_wick_ratio = sweep_cfg.get("min_wick_ratio", 0.4)  # 40% wick vs total range
            quick_reversal_bars = sweep_cfg.get("quick_reversal_bars", 3)  # Reverse within 3 bars

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Calculate volume surge vs recent average
            d["vol_ma10"] = d["volume"].rolling(10, min_periods=5).mean()
            d["vol_surge"] = d["volume"] / d["vol_ma10"]

            current_price = d["close"].iloc[-1]

            # Look for liquidity sweeps in recent bars
            lookback_bars = min(10, len(d) - quick_reversal_bars)

            for level_name, level_price in (levels or {}).items():
                if level_price <= 0 or not np.isfinite(level_price):
                    continue

                # Check recent bars for potential sweeps
                for i in range(lookback_bars, len(d) - 1):
                    sweep_bar = d.iloc[i]

                    # Calculate wick ratios for this bar
                    bar_range = sweep_bar["high"] - sweep_bar["low"]
                    if bar_range <= 0:
                        continue

                    upper_wick = sweep_bar["high"] - max(sweep_bar["open"], sweep_bar["close"])
                    lower_wick = min(sweep_bar["open"], sweep_bar["close"]) - sweep_bar["low"]

                    upper_wick_ratio = upper_wick / bar_range
                    lower_wick_ratio = lower_wick / bar_range

                    # Check volume surge on the sweep bar
                    vol_surge = d["vol_surge"].iloc[i]
                    if vol_surge < min_volume_surge:
                        continue

                    # Bullish liquidity sweep (hunt sell stops below support)
                    if level_name in ["PDL", "ORL"] and level_price > 0:
                        # Check if this bar swept below the level
                        sweep_distance = level_price - sweep_bar["low"]
                        sweep_pct = sweep_distance / level_price

                        if (min_sweep_pct <= sweep_pct <= max_sweep_pct and  # Right sweep distance
                            lower_wick_ratio >= min_wick_ratio and  # Significant lower wick (rejection)
                            sweep_bar["close"] > level_price):  # Closed back above level

                            # Check for quick reversal in subsequent bars
                            reversal_confirmed = False
                            reversal_bars = d.iloc[i+1:i+1+quick_reversal_bars]

                            if len(reversal_bars) > 0:
                                # Check if price stayed above level and moved higher
                                above_level = (reversal_bars["low"] > level_price * 0.998).all()
                                moved_higher = reversal_bars["close"].iloc[-1] > sweep_bar["close"]
                                reversal_confirmed = above_level and moved_higher

                            # Check if current price is still in play
                            currently_relevant = (current_price > level_price * 0.995 and
                                                i >= len(d) - 5)  # Recent sweep

                            if reversal_confirmed and currently_relevant:
                                # Strength based on sweep characteristics
                                wick_strength = lower_wick_ratio * 2
                                volume_strength = min(vol_surge / 2, 1.5)
                                recency_factor = max(0.5, 1.0 - (len(d) - i - 1) / 10.0)

                                strength = min(3.0, (wick_strength + volume_strength + sweep_pct * 200) * recency_factor)

                                events.append(StructureEvent(
                                    setup="liquidity_sweep_long",
                                    ts=self._safe_ts(d),
                                    level_name=f"SWEEP_{level_name}",
                                    strength=float(strength)
                                ))

                    # Bearish liquidity sweep (hunt buy stops above resistance)
                    elif level_name in ["PDH", "ORH"] and level_price > 0:
                        # Check if this bar swept above the level
                        sweep_distance = sweep_bar["high"] - level_price
                        sweep_pct = sweep_distance / level_price

                        if (min_sweep_pct <= sweep_pct <= max_sweep_pct and  # Right sweep distance
                            upper_wick_ratio >= min_wick_ratio and  # Significant upper wick (rejection)
                            sweep_bar["close"] < level_price):  # Closed back below level

                            # Check for quick reversal in subsequent bars
                            reversal_confirmed = False
                            reversal_bars = d.iloc[i+1:i+1+quick_reversal_bars]

                            if len(reversal_bars) > 0:
                                # Check if price stayed below level and moved lower
                                below_level = (reversal_bars["high"] < level_price * 1.002).all()
                                moved_lower = reversal_bars["close"].iloc[-1] < sweep_bar["close"]
                                reversal_confirmed = below_level and moved_lower

                            # Check if current price is still in play
                            currently_relevant = (current_price < level_price * 1.005 and
                                                i >= len(d) - 5)  # Recent sweep

                            if reversal_confirmed and currently_relevant:
                                # Strength based on sweep characteristics
                                wick_strength = upper_wick_ratio * 2
                                volume_strength = min(vol_surge / 2, 1.5)
                                recency_factor = max(0.5, 1.0 - (len(d) - i - 1) / 10.0)

                                strength = min(3.0, (wick_strength + volume_strength + sweep_pct * 200) * recency_factor)

                                events.append(StructureEvent(
                                    setup="liquidity_sweep_short",
                                    ts=self._safe_ts(d),
                                    level_name=f"SWEEP_{level_name}",
                                    strength=float(strength)
                                ))

            # Remove duplicate events (keep strongest per level)
            if events:
                # Group by level_name and keep strongest
                by_level = {}
                for evt in events:
                    if evt.level_name not in by_level or evt.strength > by_level[evt.level_name].strength:
                        by_level[evt.level_name] = evt
                events = list(by_level.values())

                for evt in events:
                    logger.info(f"structure_event: {evt}")

            return events

        except Exception as e:
            logger.exception(f"detect_liquidity_sweeps error: {e}")
            return []

    def detect_premium_discount_zones(self, df: pd.DataFrame, levels: Dict[str, float]) -> List[StructureEvent]:
        """
        Detect Premium/Discount Zone opportunities - daily range positioning analysis.

        Premium Zone: Upper 20% of daily range (sell zone - price expensive)
        Equilibrium: Middle 60% of daily range (neutral zone)
        Discount Zone: Lower 20% of daily range (buy zone - price cheap)

        Institutional traders prefer:
        - Buying in discount zones (cheap prices)
        - Selling in premium zones (expensive prices)
        """
        try:
            if df is None or df.empty or len(df) < 3:
                return []

            events = []
            cfg = self.cfg.get("setups", {})
            prem_disc_cfg = cfg.get("premium_discount_zones", {})

            if not prem_disc_cfg.get("enabled", True):  # Default enabled for institutional concepts
                return []

            premium_threshold = prem_disc_cfg.get("premium_threshold", 0.8)  # Top 20%
            discount_threshold = prem_disc_cfg.get("discount_threshold", 0.2)  # Bottom 20%
            min_range_pct = prem_disc_cfg.get("min_daily_range_pct", 1.0) / 100.0  # 1% min range
            require_volume_confirmation = prem_disc_cfg.get("require_volume_confirmation", True)

            # Get daily high and low (PDH/PDL if available, else intraday high/low)
            daily_high = levels.get("PDH") if "PDH" in levels else df["high"].max()
            daily_low = levels.get("PDL") if "PDL" in levels else df["low"].min()

            if not daily_high or not daily_low or daily_high <= daily_low:
                return []

            daily_range = daily_high - daily_low
            range_pct = daily_range / daily_low

            # Only analyze if we have a meaningful daily range
            if range_pct < min_range_pct:
                return []

            current_price = df["close"].iloc[-1]

            # Calculate current position in daily range (0 = daily low, 1 = daily high)
            range_position = (current_price - daily_low) / daily_range

            # Calculate zone boundaries
            discount_top = daily_low + (daily_range * discount_threshold)
            premium_bottom = daily_low + (daily_range * premium_threshold)

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)
            current_vol_z = d["vol_z"].iloc[-1]

            # Volume confirmation check
            volume_confirmed = True
            if require_volume_confirmation:
                volume_confirmed = current_vol_z > 0.5  # Above average volume

            if not volume_confirmed:
                return []

            # Premium Zone Analysis (sell opportunities)
            if range_position >= premium_threshold:
                # Look for rejection signs in premium zone
                last_bar = df.iloc[-1]
                bar_range = last_bar["high"] - last_bar["low"]

                if bar_range > 0:
                    # Check for upper wick (rejection from highs)
                    upper_wick = last_bar["high"] - max(last_bar["open"], last_bar["close"])
                    upper_wick_ratio = upper_wick / bar_range

                    # Check for bearish close
                    bearish_close = last_bar["close"] < last_bar["open"]

                    # Strong premium zone opportunity
                    if upper_wick_ratio > 0.3 or bearish_close:
                        # Calculate distance from daily high
                        distance_from_high = (daily_high - current_price) / daily_range

                        # Strength based on position in premium zone and rejection signs
                        position_strength = (range_position - premium_threshold) * 5  # 0-1 range
                        rejection_strength = upper_wick_ratio * 2 if upper_wick_ratio > 0.3 else 0
                        bearish_strength = 0.5 if bearish_close else 0

                        strength = min(3.0, position_strength + rejection_strength + bearish_strength + current_vol_z * 0.3)

                        events.append(StructureEvent(
                            setup="premium_zone_short",
                            ts=self._safe_ts(d),
                            level_name="PREMIUM_ZONE",
                            strength=float(strength)
                        ))

            # Discount Zone Analysis (buy opportunities)
            elif range_position <= discount_threshold:
                # Look for support signs in discount zone
                last_bar = df.iloc[-1]
                bar_range = last_bar["high"] - last_bar["low"]

                if bar_range > 0:
                    # Check for lower wick (support from lows)
                    lower_wick = min(last_bar["open"], last_bar["close"]) - last_bar["low"]
                    lower_wick_ratio = lower_wick / bar_range

                    # Check for bullish close
                    bullish_close = last_bar["close"] > last_bar["open"]

                    # Strong discount zone opportunity
                    if lower_wick_ratio > 0.3 or bullish_close:
                        # Calculate distance from daily low
                        distance_from_low = (current_price - daily_low) / daily_range

                        # Strength based on position in discount zone and support signs
                        position_strength = (discount_threshold - range_position) * 5  # 0-1 range
                        support_strength = lower_wick_ratio * 2 if lower_wick_ratio > 0.3 else 0
                        bullish_strength = 0.5 if bullish_close else 0

                        strength = min(3.0, position_strength + support_strength + bullish_strength + current_vol_z * 0.3)

                        events.append(StructureEvent(
                            setup="discount_zone_long",
                            ts=self._safe_ts(d),
                            level_name="DISCOUNT_ZONE",
                            strength=float(strength)
                        ))

            # Equilibrium Zone - look for breakout opportunities
            elif discount_threshold < range_position < premium_threshold:
                # In equilibrium - look for strong directional moves with volume
                recent_momentum = df["close"].pct_change(3).iloc[-1]  # 3-bar momentum

                if abs(recent_momentum) > 0.01 and current_vol_z > 1.0:  # 1% move + volume
                    # Strong move toward premium zone
                    if recent_momentum > 0 and range_position > 0.6:  # Moving up, approaching premium
                        strength = min(3.0, abs(recent_momentum) * 100 + current_vol_z * 0.5)

                        events.append(StructureEvent(
                            setup="equilibrium_breakout_long",
                            ts=self._safe_ts(d),
                            level_name="EQUILIBRIUM",
                            strength=float(strength)
                        ))

                    # Strong move toward discount zone
                    elif recent_momentum < 0 and range_position < 0.4:  # Moving down, approaching discount
                        strength = min(3.0, abs(recent_momentum) * 100 + current_vol_z * 0.5)

                        events.append(StructureEvent(
                            setup="equilibrium_breakout_short",
                            ts=self._safe_ts(d),
                            level_name="EQUILIBRIUM",
                            strength=float(strength)
                        ))

            if events:
                for evt in events:
                    logger.info(f"structure_event: {evt}")

            return events

        except Exception as e:
            logger.exception(f"detect_premium_discount_zones error: {e}")
            return []

    def detect_break_of_structure(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect Break of Structure (BOS) - trend change/continuation patterns.

        BOS occurs when:
        - Bullish BOS: Price breaks above a significant previous high
        - Bearish BOS: Price breaks below a significant previous low

        This indicates either trend continuation (in direction of trend) or
        trend reversal (against previous trend).
        """
        try:
            if df is None or df.empty or len(df) < 10:
                return []

            events = []
            cfg = self.cfg.get("setups", {})
            bos_cfg = cfg.get("break_of_structure", {})

            if not bos_cfg.get("enabled", True):  # Default enabled for institutional concepts
                return []

            lookback_bars = bos_cfg.get("structure_lookback_bars", 15)  # Look back 15 bars for structure
            min_structure_age = bos_cfg.get("min_structure_age_bars", 3)  # Structure must be at least 3 bars old
            min_break_pct = bos_cfg.get("min_break_percentage", 0.1) / 100.0  # 0.1% minimum break
            require_volume_confirmation = bos_cfg.get("require_volume_confirmation", True)
            min_volume_surge = bos_cfg.get("min_volume_surge", 1.5)  # 1.5x average volume

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Calculate volume surge vs recent average
            d["vol_ma10"] = d["volume"].rolling(10, min_periods=5).mean()
            d["vol_surge"] = d["volume"] / d["vol_ma10"]

            current_price = d["close"].iloc[-1]
            current_high = d["high"].iloc[-1]
            current_low = d["low"].iloc[-1]
            current_vol_surge = d["vol_surge"].iloc[-1]
            current_vol_z = d["vol_z"].iloc[-1]

            # Volume confirmation check
            volume_confirmed = True
            if require_volume_confirmation:
                volume_confirmed = current_vol_surge >= min_volume_surge

            if not volume_confirmed:
                return []

            # Look for significant highs and lows in recent history
            search_end = len(d) - 1 - min_structure_age  # Exclude recent bars for structure age
            search_start = max(5, search_end - lookback_bars)

            if search_start >= search_end:
                return []

            # Find significant structure points (swing highs/lows)
            structure_data = d.iloc[search_start:search_end]

            # Identify swing highs (local maxima)
            swing_highs = []
            for i in range(2, len(structure_data) - 2):  # Need 2 bars on each side
                current_bar = structure_data.iloc[i]
                prev_2 = structure_data.iloc[i-2]
                prev_1 = structure_data.iloc[i-1]
                next_1 = structure_data.iloc[i+1]
                next_2 = structure_data.iloc[i+2]

                # Check if this is a swing high (higher than surrounding bars)
                if (current_bar["high"] > prev_2["high"] and
                    current_bar["high"] > prev_1["high"] and
                    current_bar["high"] > next_1["high"] and
                    current_bar["high"] > next_2["high"]):

                    swing_highs.append({
                        "price": current_bar["high"],
                        "index": search_start + i,
                        "timestamp": current_bar.name
                    })

            # Identify swing lows (local minima)
            swing_lows = []
            for i in range(2, len(structure_data) - 2):  # Need 2 bars on each side
                current_bar = structure_data.iloc[i]
                prev_2 = structure_data.iloc[i-2]
                prev_1 = structure_data.iloc[i-1]
                next_1 = structure_data.iloc[i+1]
                next_2 = structure_data.iloc[i+2]

                # Check if this is a swing low (lower than surrounding bars)
                if (current_bar["low"] < prev_2["low"] and
                    current_bar["low"] < prev_1["low"] and
                    current_bar["low"] < next_1["low"] and
                    current_bar["low"] < next_2["low"]):

                    swing_lows.append({
                        "price": current_bar["low"],
                        "index": search_start + i,
                        "timestamp": current_bar.name
                    })

            # Check for Bullish BOS (breaking above recent swing high)
            if swing_highs:
                # Find the most recent significant swing high
                recent_swing_high = max(swing_highs, key=lambda x: x["price"])

                # Check if current price breaks above this swing high
                break_distance = current_high - recent_swing_high["price"]
                break_pct = break_distance / recent_swing_high["price"]

                if break_pct >= min_break_pct:
                    # Calculate strength based on break size, volume, and structure significance
                    bars_since_structure = len(d) - 1 - recent_swing_high["index"]
                    time_factor = max(0.5, 1.0 - (bars_since_structure / 20.0))  # Decay over 20 bars

                    break_strength = min(2.0, break_pct * 500)  # Break percentage contribution
                    volume_strength = min(1.0, current_vol_surge * 0.5)  # Volume contribution
                    structure_strength = min(1.0, len(swing_highs) * 0.2)  # Multiple structures = stronger

                    strength = min(3.0, (break_strength + volume_strength + structure_strength) * time_factor)

                    events.append(StructureEvent(
                        setup="break_of_structure_long",
                        ts=self._safe_ts(d),
                        level_name=f"BOS_HIGH_{recent_swing_high['index']}",
                        strength=float(strength)
                    ))

            # Check for Bearish BOS (breaking below recent swing low)
            if swing_lows:
                # Find the most recent significant swing low
                recent_swing_low = min(swing_lows, key=lambda x: x["price"])

                # Check if current price breaks below this swing low
                break_distance = recent_swing_low["price"] - current_low
                break_pct = break_distance / recent_swing_low["price"]

                if break_pct >= min_break_pct:
                    # Calculate strength based on break size, volume, and structure significance
                    bars_since_structure = len(d) - 1 - recent_swing_low["index"]
                    time_factor = max(0.5, 1.0 - (bars_since_structure / 20.0))  # Decay over 20 bars

                    break_strength = min(2.0, break_pct * 500)  # Break percentage contribution
                    volume_strength = min(1.0, current_vol_surge * 0.5)  # Volume contribution
                    structure_strength = min(1.0, len(swing_lows) * 0.2)  # Multiple structures = stronger

                    strength = min(3.0, (break_strength + volume_strength + structure_strength) * time_factor)

                    events.append(StructureEvent(
                        setup="break_of_structure_short",
                        ts=self._safe_ts(d),
                        level_name=f"BOS_LOW_{recent_swing_low['index']}",
                        strength=float(strength)
                    ))

            if events:
                for evt in events:
                    logger.info(f"structure_event: {evt}")

            return events

        except Exception as e:
            logger.exception(f"detect_break_of_structure error: {e}")
            return []

    def detect_change_of_character(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        Detect Change of Character (CHoCH) - early trend reversal signals.

        CHoCH occurs when:
        - In uptrend: Price fails to make a new high and breaks below previous low
        - In downtrend: Price fails to make a new low and breaks above previous high

        This is an early warning of institutional trend reversal before major BOS.
        """
        try:
            if df is None or df.empty or len(df) < 15:
                return []

            events = []
            cfg = self.cfg.get("setups", {})
            choch_cfg = cfg.get("change_of_character", {})

            if not choch_cfg.get("enabled", True):  # Default enabled for institutional concepts
                return []

            lookback_bars = choch_cfg.get("trend_lookback_bars", 20)  # Look back 20 bars for trend
            min_trend_bars = choch_cfg.get("min_trend_duration_bars", 5)  # Minimum trend duration
            min_failure_pct = choch_cfg.get("min_failure_percentage", 0.15) / 100.0  # 0.15% minimum failure
            require_volume_confirmation = choch_cfg.get("require_volume_confirmation", True)
            min_volume_surge = choch_cfg.get("min_volume_surge", 1.3)  # 1.3x average volume

            d = ensure_naive_ist_index(df.copy())
            d["vol_z"] = self._vol_z(d)

            # Calculate volume surge vs recent average
            d["vol_ma10"] = d["volume"].rolling(10, min_periods=5).mean()
            d["vol_surge"] = d["volume"] / d["vol_ma10"]

            current_price = d["close"].iloc[-1]
            current_high = d["high"].iloc[-1]
            current_low = d["low"].iloc[-1]
            current_vol_surge = d["vol_surge"].iloc[-1]

            # Volume confirmation check
            volume_confirmed = True
            if require_volume_confirmation:
                volume_confirmed = current_vol_surge >= min_volume_surge

            if not volume_confirmed:
                return []

            # Analyze recent trend structure
            trend_data = d.tail(lookback_bars)

            if len(trend_data) < min_trend_bars:
                return []

            # Find swing highs and lows in trend period
            swing_points = []

            # Identify swing points (alternating highs and lows)
            for i in range(2, len(trend_data) - 2):  # Need 2 bars on each side
                current_bar = trend_data.iloc[i]
                prev_2 = trend_data.iloc[i-2]
                prev_1 = trend_data.iloc[i-1]
                next_1 = trend_data.iloc[i+1]
                next_2 = trend_data.iloc[i+2]

                # Check if this is a swing high
                if (current_bar["high"] > prev_2["high"] and
                    current_bar["high"] > prev_1["high"] and
                    current_bar["high"] > next_1["high"] and
                    current_bar["high"] > next_2["high"]):

                    swing_points.append({
                        "type": "high",
                        "price": current_bar["high"],
                        "index": i,
                        "timestamp": current_bar.name
                    })

                # Check if this is a swing low
                elif (current_bar["low"] < prev_2["low"] and
                      current_bar["low"] < prev_1["low"] and
                      current_bar["low"] < next_1["low"] and
                      current_bar["low"] < next_2["low"]):

                    swing_points.append({
                        "type": "low",
                        "price": current_bar["low"],
                        "index": i,
                        "timestamp": current_bar.name
                    })

            # Sort swing points by time
            swing_points.sort(key=lambda x: x["index"])

            if len(swing_points) < 3:  # Need at least 3 swing points for CHoCH
                return []

            # Analyze for CHoCH patterns
            recent_swings = swing_points[-3:]  # Last 3 swing points

            # Bearish CHoCH Pattern: Higher high failed, break below previous low
            # Pattern: Low -> High -> Lower High (failure) -> Break below first low
            if (len(recent_swings) == 3 and
                recent_swings[0]["type"] == "low" and
                recent_swings[1]["type"] == "high" and
                recent_swings[2]["type"] == "high"):

                first_low = recent_swings[0]["price"]
                first_high = recent_swings[1]["price"]
                second_high = recent_swings[2]["price"]

                # Check if second high failed to exceed first high (failure to make new high)
                if second_high < first_high:
                    failure_pct = (first_high - second_high) / first_high

                    # Check if current price breaks below the first low
                    if current_low < first_low and failure_pct >= min_failure_pct:
                        # Calculate strength based on failure size and volume
                        failure_strength = min(2.0, failure_pct * 300)  # Failure percentage
                        volume_strength = min(1.0, current_vol_surge * 0.5)  # Volume contribution
                        structure_strength = min(1.0, len(swing_points) * 0.1)  # More swings = stronger structure

                        strength = min(3.0, failure_strength + volume_strength + structure_strength)

                        events.append(StructureEvent(
                            setup="change_of_character_short",
                            ts=self._safe_ts(d),
                            level_name=f"CHOCH_BEAR_{len(swing_points)}",
                            strength=float(strength)
                        ))

            # Bullish CHoCH Pattern: Lower low failed, break above previous high
            # Pattern: High -> Low -> Higher Low (failure) -> Break above first high
            elif (len(recent_swings) == 3 and
                  recent_swings[0]["type"] == "high" and
                  recent_swings[1]["type"] == "low" and
                  recent_swings[2]["type"] == "low"):

                first_high = recent_swings[0]["price"]
                first_low = recent_swings[1]["price"]
                second_low = recent_swings[2]["price"]

                # Check if second low failed to break below first low (failure to make new low)
                if second_low > first_low:
                    failure_pct = (second_low - first_low) / first_low

                    # Check if current price breaks above the first high
                    if current_high > first_high and failure_pct >= min_failure_pct:
                        # Calculate strength based on failure size and volume
                        failure_strength = min(2.0, failure_pct * 300)  # Failure percentage
                        volume_strength = min(1.0, current_vol_surge * 0.5)  # Volume contribution
                        structure_strength = min(1.0, len(swing_points) * 0.1)  # More swings = stronger structure

                        strength = min(3.0, failure_strength + volume_strength + structure_strength)

                        events.append(StructureEvent(
                            setup="change_of_character_long",
                            ts=self._safe_ts(d),
                            level_name=f"CHOCH_BULL_{len(swing_points)}",
                            strength=float(strength)
                        ))

            # Alternative CHoCH detection: Look for divergence in momentum vs price
            if not events and len(swing_points) >= 2:
                # Get last two same-type swing points
                same_type_swings = [sp for sp in swing_points if sp["type"] == swing_points[-1]["type"]]

                if len(same_type_swings) >= 2:
                    prev_swing = same_type_swings[-2]
                    current_swing = same_type_swings[-1]

                    # Bearish divergence CHoCH (price makes higher high, but with weakness)
                    if (current_swing["type"] == "high" and
                        current_swing["price"] > prev_swing["price"]):

                        # Check if recent momentum is weaker despite higher price
                        recent_momentum = d["close"].pct_change(3).tail(3).mean()
                        if recent_momentum < 0.002:  # Very weak momentum despite new high
                            strength = min(3.0, current_vol_surge + 1.0)

                            events.append(StructureEvent(
                                setup="change_of_character_short",
                                ts=self._safe_ts(d),
                                level_name="CHOCH_DIVERGENCE",
                                strength=float(strength)
                            ))

                    # Bullish divergence CHoCH (price makes lower low, but with strength)
                    elif (current_swing["type"] == "low" and
                          current_swing["price"] < prev_swing["price"]):

                        # Check if recent momentum is stronger despite lower price
                        recent_momentum = d["close"].pct_change(3).tail(3).mean()
                        if recent_momentum > -0.002:  # Less weak momentum despite new low
                            strength = min(3.0, current_vol_surge + 1.0)

                            events.append(StructureEvent(
                                setup="change_of_character_long",
                                ts=self._safe_ts(d),
                                level_name="CHOCH_DIVERGENCE",
                                strength=float(strength)
                            ))

            if events:
                for evt in events:
                    logger.info(f"structure_event: {evt}")

            return events

        except Exception as e:
            logger.exception(f"detect_change_of_character error: {e}")
            return []