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
                    cond = (
                        (last["close"] > lvl + k_atr * atr)
                        and (d["close"] > lvl).tail(hold_bars).all()
                        and d["vol_z"].iloc[-1] >= vol_z_required
                    )
                    if cond:
                        ts = self._safe_ts(d)
                        evt = StructureEvent("breakout_long", ts, name, float(d["vol_z"].iloc[-1]))
                        out.append(evt)
                        logger.info(f"structure_event: {evt}")

                # Short breakdown below PDL/ORL
                if name in ("PDL", "ORL"):
                    cond = (
                        (last["close"] < lvl - k_atr * atr)
                        and (d["close"] < lvl).tail(hold_bars).all()
                        and d["vol_z"].iloc[-1] >= vol_z_required
                    )
                    if cond:
                        ts = self._safe_ts(d)
                        evt = StructureEvent("breakout_short", ts, name, float(d["vol_z"].iloc[-1]))
                        out.append(evt)
                        logger.info(f"structure_event: {evt}")

            return out

        except Exception as e:
            logger.exception(f"structure_event: detect_level_breakouts error: {e}")
            return []

    def detect_vwap_cross_and_hold(self, df: pd.DataFrame) -> List[StructureEvent]:
        """
        VWAP reclaim (long) / lose (short) with volume confirmation.

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
            out: List[StructureEvent] = []

            # Reclaim (close > vwap for hold_bars) + volume
            if (d["close"] > d["vwap"]).tail(hold_bars).all() and d["vol_z"].iloc[-1] >= vol_z_required:
                ts = self._safe_ts(d)
                evt = StructureEvent("vwap_reclaim_long", ts, "VWAP", float(d["vol_z"].iloc[-1]))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

            # Lose (close < vwap for hold_bars) + volume
            if (d["close"] < d["vwap"]).tail(hold_bars).all() and d["vol_z"].iloc[-1] >= vol_z_required:
                ts = self._safe_ts(d)
                evt = StructureEvent("vwap_lose_short", ts, "VWAP", float(d["vol_z"].iloc[-1]))
                out.append(evt)
                logger.info(f"structure_event: {evt}")

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

        setups = []
        for e in evts or []:
            # e.setup should be one of:
            #   'breakout_long','breakout_short','vwap_reclaim_long','vwap_lose_short',
            #   'squeeze_release_long','squeeze_release_short','failure_fade_long','failure_fade_short',
            #   'momentum_breakout_long','momentum_breakout_short','trend_continuation_long','trend_continuation_short',
            #   and many other setup types defined in trade_decision_gate.py SetupType
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