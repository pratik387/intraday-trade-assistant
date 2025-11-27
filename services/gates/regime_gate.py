from __future__ import annotations
"""
regime_gate.py
---------------
Classifies the market regime from 5‑minute index bars and exposes simple
permissions for setup types. No config reads and no hidden defaults in the
caller path: you pass the index DataFrame, it returns a regime label.

Expected input DataFrame (IST‑aligned, 5‑minute close index):
  columns: [open, high, low, close, volume, vwap] (bb_width_proxy optional)

Public API
----------
MarketRegimeGate.compute_regime(df5) -> (regime: str, confidence: float)
MarketRegimeGate.allow_setup(setup_type: str, regime: str) -> bool
MarketRegimeGate.size_multiplier(regime: str, counter_trend: bool=False) -> float

Regime labels
-------------
  "trend_up" | "trend_down" | "chop" | "squeeze"

Setup types (consistent with our structure detectors):
  "breakout_long", "breakout_short",
  "vwap_reclaim_long", "vwap_lose_short",
  "squeeze_release_long", "squeeze_release_short",
  "failure_fade_long", "failure_fade_short",
  "gap_fill_long", "gap_fill_short",
  "flag_continuation_long", "flag_continuation_short",
  "vwap_mean_reversion_long", "vwap_mean_reversion_short",
  "volume_spike_reversal_long", "volume_spike_reversal_short",
  "trend_pullback_long", "trend_pullback_short",
  "range_rejection_long", "range_rejection_short"
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Phase 1: Multi-timeframe regime detection (institutional approach)
try:
    from services.gates.multi_timeframe_regime import (
        DailyRegimeDetector,
        HourlyRegimeDetector,
        MultiTimeframeRegime,
        DailyRegimeResult
    )
    MULTI_TF_AVAILABLE = True
except ImportError:
    MULTI_TF_AVAILABLE = False

@dataclass(frozen=True)
class RegimeMetrics:
    ema_fast: float
    ema_slow: float
    slope_fast: float
    width_proxy: float
    vol_z: float

class MarketRegimeGate:
    """
    Computes regime from 5m index bars and decides if a setup is allowed
    using config-driven, evidence-based thresholds.

    Public:
      compute_regime(df5) -> (regime, metrics_dict)
      allow_setup(setup_type, regime, strength, adx_5m, vol_mult_5m) -> bool
      size_multiplier(regime, counter_trend=False) -> float
    """

    def __init__(self, cfg: Dict, log=None):
        self.log = log
        self.cfg = cfg  # Store config for chop throttle and other features
        # strict: require thresholds to exist (no hidden defaults)
        rt = (cfg.get("regime_thresholds"))
        required = [
            "vwap_min_strength", "vwap_min_adx", "vwap_min_vol_mult",
            "bo_min_adx", "bo_min_vol_mult",
            "ff_min_vol_mult",
        ]
        missing = [k for k in required if k not in rt]
        if missing:
            raise ValueError(f"[regime_gate] Missing regime_thresholds keys: {missing}. "
                             f"Add them under entry_config.json -> regime_thresholds")
        self.VWAP_MIN_STRENGTH = float(rt["vwap_min_strength"])
        self.VWAP_MIN_ADX      = float(rt["vwap_min_adx"])
        self.VWAP_MIN_VOL_MULT = float(rt["vwap_min_vol_mult"])
        self.BO_MIN_ADX        = float(rt["bo_min_adx"])
        self.BO_MIN_VOL_MULT   = float(rt["bo_min_vol_mult"])
        self.FF_MIN_VOL_MULT   = float(rt["ff_min_vol_mult"])

        # Phase 1: Initialize multi-timeframe regime detectors
        if MULTI_TF_AVAILABLE:
            self.daily_detector = DailyRegimeDetector(log=log)
            self.hourly_detector = HourlyRegimeDetector(log=log)
            self.multi_tf_regime = MultiTimeframeRegime(
                daily_detector=self.daily_detector,
                hourly_detector=self.hourly_detector,
                log=log
            )
            if log:
                log.info("Multi-timeframe regime detection initialized (institutional approach)")
        else:
            self.daily_detector = None
            self.hourly_detector = None
            self.multi_tf_regime = None
            if log:
                log.warning("Multi-timeframe regime module not available - using 5m-only regime")

    # ---------------- Regime computation ----------------
    def compute_regime(self, df5: pd.DataFrame) -> Tuple[str, float]:
        """
        INSTITUTIONAL REGIME DETECTION (NSE India professional standards)

        Uses ADX as PRIMARY indicator for regime classification (not EMA slope):
        - ADX < 20: CHOP (ranging/choppy market)
        - ADX >= 25: TREND (trending market, direction from +DI/-DI)
        - Squeeze: Very narrow BB width + ADX < 15

        This aligns with professional trading standards where ADX 25 is the
        preferred threshold (20 produces too many false signals).

        References:
        - Professional ADX thresholds: ADX > 25 = trending, < 20 = ranging
        - John Bollinger: Low volatility (squeeze) precedes high volatility
        - NSE institutional desks: Multi-indicator approach (ADX + BB + EMA)
        """
        if df5 is None or df5.empty:
            return "chop", 0.5

        c = df5["close"].astype(float).copy()
        h = df5["high"].astype(float).copy()
        l = df5["low"].astype(float).copy()

        # Calculate ADX (PRIMARY indicator for regime)
        # Use centralized ADX calculation to avoid duplication across codebase
        adx_period = 14
        if len(df5) >= adx_period + 1:
            from services.indicators.indicators import calculate_adx_with_di

            # Calculate ADX with directional indicators (Wilder's smoothing)
            adx, plus_di, minus_di = calculate_adx_with_di(df5, adx_period)

            adx_value = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
            plus_di_value = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0
            minus_di_value = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0
        else:
            adx_value = 0.0
            plus_di_value = 0.0
            minus_di_value = 0.0

        # EMA for trend direction confirmation (SECONDARY indicator)
        if len(c) < 55:
            ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
            ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
        else:
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()

        # BB width proxy for squeeze detection
        if "bb_width_proxy" in df5.columns:
            width_proxy = float(df5["bb_width_proxy"].tail(20).mean())
        else:
            std20 = float(c.tail(20).std(ddof=0)) if len(c) >= 5 else 0.0
            mean20 = float(c.tail(20).mean()) if len(c) >= 1 else 1.0
            width_proxy = (std20 / mean20) if mean20 else 0.0

        # Volume z-score vs last ~2 hours
        if "volume" in df5.columns and len(df5) >= 10:
            vol = df5["volume"].astype(float)
            recent = vol.tail(min(24, len(vol)))
            mu, sd = float(recent.mean()), float(recent.std(ddof=0))
            vol_z = 0.0 if sd == 0.0 else float((recent.iloc[-1] - mu) / sd)
        else:
            vol_z = 0.0

        m = RegimeMetrics(
            ema_fast=float(ema20.iloc[-1]),
            ema_slow=float(ema50.iloc[-1]) if not np.isnan(ema50.iloc[-1]) else float(ema20.iloc[-1]),
            slope_fast=0.0,  # Not used for regime classification anymore
            width_proxy=width_proxy,
            vol_z=vol_z,
        )

        # SQUEEZE detection: Very narrow BB width + very low ADX
        # Professional standard: BB squeeze = low volatility period
        width_series = (df5["bb_width_proxy"] if "bb_width_proxy" in df5.columns else c.pct_change().abs())
        recent_w = width_series.tail(min(24, len(width_series)))
        if len(recent_w) >= 8:
            q30 = float(recent_w.quantile(0.30))
            is_very_tight = m.width_proxy <= q30
            is_squeeze = is_very_tight and adx_value < 15  # ADX < 15 for squeeze
        else:
            is_squeeze = False

        # REGIME CLASSIFICATION (Professional ADX-based approach)
        if is_squeeze:
            regime = "squeeze"
            confidence = 0.8
        elif adx_value < 20:
            # Low ADX = CHOP (ranging/choppy market)
            # Professional standard: ADX < 20 = weak trend
            regime = "chop"
            confidence = 0.6
        elif adx_value >= 25:
            # High ADX = TREND (trending market)
            # Professional standard: ADX >= 25 = strong trend
            # Use +DI/-DI to determine direction
            if plus_di_value > minus_di_value:
                regime = "trend_up"
            else:
                regime = "trend_down"

            # Higher confidence for stronger trends
            if adx_value >= 40:
                confidence = 0.9  # Very strong trend
            else:
                confidence = 0.75  # Strong trend
        else:
            # Transition zone (ADX 20-25): Use EMA as tiebreaker
            if m.ema_fast > m.ema_slow:
                regime = "trend_up"
            elif m.ema_fast < m.ema_slow:
                regime = "trend_down"
            else:
                regime = "chop"
            confidence = 0.5  # Low confidence in transition zone

        return regime, confidence

    def compute_regime_multi_tf(
        self,
        df5: pd.DataFrame,
        daily_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        Phase 1: Multi-timeframe regime computation (institutional approach).

        Uses daily trend (EMA200, ADX) as primary filter, falls back to 5m if unavailable.
        This prevents -Rs. 4,258 loss from squeeze breakout shorts (audit finding).

        Args:
            df5: 5-minute OHLCV DataFrame
            daily_df: Daily OHLCV DataFrame (210+ bars for EMA200)
            symbol: Symbol name (for logging)

        Returns:
            (regime, confidence, diagnostics)
            - regime: "trend_up" | "trend_down" | "chop" | "squeeze"
            - confidence: 0.0-1.0
            - diagnostics: Dict with daily/hourly/5m breakdown (or None if unavailable)
        """
        # Fallback to 5m-only if multi-TF not available
        if not MULTI_TF_AVAILABLE or self.daily_detector is None:
            regime_5m, conf_5m = self.compute_regime(df5)
            return regime_5m, conf_5m, None

        # Get 5m regime first (always needed)
        regime_5m, conf_5m = self.compute_regime(df5)

        # If no daily data, fall back to 5m only
        if daily_df is None or daily_df.empty or len(daily_df) < 50:
            if self.log:
                self.log.debug(f"Multi-TF regime: {symbol} - insufficient daily data, using 5m-only")
            return regime_5m, conf_5m, None

        try:
            # Get daily regime (primary)
            daily_result = self.daily_detector.classify(daily_df)

            # Get unified multi-TF regime
            unified_regime, unified_conf, diagnostics = self.multi_tf_regime.get_unified_regime(
                daily_df=daily_df,
                df_5m=df5,
                current_5m_regime=regime_5m
            )

            # CRITICAL FIX: Block squeeze breakout shorts (audit report finding)
            # 11 trades, 0% win rate, -Rs. 4,258 loss
            if daily_result.regime == "squeeze" and unified_regime == "squeeze":
                if self.log:
                    self.log.info(
                        f"Multi-TF: {symbol} - Daily squeeze detected "
                        f"(conf={daily_result.confidence:.2f}, prevents breakout_short losses)"
                    )

            return unified_regime, unified_conf, diagnostics

        except Exception as e:
            if self.log:
                self.log.error(f"Multi-TF regime error for {symbol}: {e}, falling back to 5m")
            return regime_5m, conf_5m, None

    # ---------------- Evidence-based permissioning ----------------
    def allow_setup(
        self,
        setup_type: str,
        regime: str,
        strength: float,
        adx_5m: float,
        vol_mult_5m: float,
        cap_segment: str = "unknown",  # NEW: Priority 2 - cap-specific filtering
    ) -> bool:
        """
        Returns True iff the setup is allowed by regime AND passes evidence thresholds AND cap-strategy fit.

        Params:
          setup_type   : 'breakout_long'|'breakout_short'|'vwap_reclaim_long'|'vwap_lose_short'|
                         'squeeze_release_long'|'squeeze_release_short'|'failure_fade_long'|'failure_fade_short'|
                         'gap_fill_long'|'gap_fill_short'|'flag_continuation_long'|'flag_continuation_short'|
                         'support_bounce_long'|'resistance_bounce_short'|'orb_breakout_long'|'orb_breakout_short'|
                         'vwap_mean_reversion_long'|'vwap_mean_reversion_short'|'volume_spike_reversal_long'|
                         'volume_spike_reversal_short'|'trend_pullback_long'|'trend_pullback_short'|
                         'range_rejection_long'|'range_rejection_short'
          regime       : 'trend_up'|'trend_down'|'chop'|'squeeze'
          strength     : structure score (0..5 normalized)
          adx_5m       : 5m ADX
          vol_mult_5m  : last5m_vol / median5m_vol (~2h)
          cap_segment  : 'large_cap'|'mid_cap'|'small_cap'|'micro_cap'|'unknown' (Priority 2: cap-aware filtering)
        """
        # === PRIORITY 2: CAP-STRATEGY FILTERING (institutional evidence) ===
        cap_prefs = self.cfg.get("cap_strategy_preferences", {})
        if cap_prefs.get("enabled", False) and cap_segment != "unknown":
            seg_cfg = cap_prefs.get(cap_segment, {})

            # Check if setup is BLOCKED for this cap segment
            blocked = seg_cfg.get("blocked", [])
            if setup_type in blocked:
                if self.log:
                    self.log.debug(f"CAP_FILTER: blocked {setup_type} for {cap_segment}")
                return False

            # Check if setup is NOT in preferred or allowed lists
            preferred = seg_cfg.get("preferred", [])
            allowed = seg_cfg.get("allowed", [])
            if setup_type not in (preferred + allowed):
                if self.log:
                    self.log.debug(f"CAP_FILTER: {setup_type} not suitable for {cap_segment}")
                return False

        # NA-safe casts
        s = float(strength) if pd.notna(strength) else 0.0
        a = float(adx_5m) if pd.notna(adx_5m) else 0.0
        v = float(vol_mult_5m) if pd.notna(vol_mult_5m) else 0.0

        if regime == "chop":
            # CHOP THROTTLE: Check if this setup is allowed in chop
            quality_filters = self.cfg.get("quality_filters", {})
            chop_throttle = quality_filters.get("chop_throttle", {})

            if chop_throttle.get("enabled", False):
                allowed_setups = chop_throttle.get("allowed_setups", [])
                if setup_type not in allowed_setups:
                    if self.log:
                        self.log.debug(f"CHOP_THROTTLE: blocked {setup_type} in chop regime")
                    return False

            # Prioritized setups for chop (proven winners)
            if setup_type == "orb_pullback_long":
                # Requires strong evidence as per analysis
                return (s >= self.VWAP_MIN_STRENGTH) and (a >= self.VWAP_MIN_ADX) and (v >= self.VWAP_MIN_VOL_MULT)

            if setup_type == "failure_fade_long":
                # Enhanced requirements for better performance
                return (s >= 2.0) and (v >= self.FF_MIN_VOL_MULT * 1.5)  # 1.5x volume requirement

            if setup_type == "range_break_retest_short":
                # Strong retest pattern
                return (s >= self.VWAP_MIN_STRENGTH) and (v >= self.VWAP_MIN_VOL_MULT)

            # Standard setup requirements for chop regime (may be throttled if enabled)
            # VWAP reclaims/loses - standard requirements
            if setup_type in {"vwap_reclaim_long", "vwap_lose_short"}:
                return (s >= self.VWAP_MIN_STRENGTH) and (a >= self.VWAP_MIN_ADX) and (v >= self.VWAP_MIN_VOL_MULT)

            # Failure fades - primary chop strategy
            # INSTITUTIONAL FIX: Require ADX >= 15 to avoid weak/choppy trades
            # Analysis: ADX > 15 filter saves Rs.2,718 (18 hard_sl avoided, 15 winners lost)
            if setup_type in {"failure_fade_long", "failure_fade_short"}:
                return (v >= self.FF_MIN_VOL_MULT) and (a >= 15.0)

            # Mean reversion and range setups - work well in chop
            if setup_type in {"vwap_mean_reversion_long", "vwap_mean_reversion_short"}:
                return (s >= self.VWAP_MIN_STRENGTH) and (v >= self.VWAP_MIN_VOL_MULT)
            if setup_type in {"range_rejection_long", "range_rejection_short"}:
                return v >= self.FF_MIN_VOL_MULT
            if setup_type in {"support_bounce_long", "resistance_bounce_short"}:
                return (s >= self.VWAP_MIN_STRENGTH) and (v >= self.VWAP_MIN_VOL_MULT)

            # Volume spike reversals - can work in chop
            if setup_type in {"volume_spike_reversal_long", "volume_spike_reversal_short"}:
                return v >= self.FF_MIN_VOL_MULT

            # Allow selective breakouts in chop with VERY strong evidence (reduced size via size_multiplier)
            # These get 0.6x size multiplier to manage risk while capturing exceptional setups
            if setup_type in {"breakout_long", "breakout_short"}:
                # Require exceptional strength + very high ADX + high volume
                return (s >= 3.5) and (a >= 35.0) and (v >= 2.5)

            # ORB breakouts disabled - poor performance per diagnostic report

            return False

        if regime == "trend_up":
            if setup_type == "breakout_long":
                return (a >= self.BO_MIN_ADX) and (v >= self.BO_MIN_VOL_MULT)
            if setup_type == "vwap_reclaim_long":
                return (s >= self.VWAP_MIN_STRENGTH) and (a >= self.VWAP_MIN_ADX) and (v >= self.VWAP_MIN_VOL_MULT)
            # INSTITUTIONAL FIX: Require ADX >= 15 for fade strategies
            if setup_type == "failure_fade_short":
                return (v >= self.FF_MIN_VOL_MULT) and (a >= 15.0)
            # Trend continuation setups in uptrend
            if setup_type == "flag_continuation_long":
                return (a >= self.BO_MIN_ADX) and (v >= self.BO_MIN_VOL_MULT)
            if setup_type == "trend_pullback_long":
                return (s >= self.VWAP_MIN_STRENGTH) and (v >= self.VWAP_MIN_VOL_MULT)
            # Support bounce disabled - poor performance per diagnostic report
            if setup_type == "gap_fill_long":
                return v >= self.VWAP_MIN_VOL_MULT
            # Volume spike reversals can work as counter-trend in strong trends
            if setup_type == "volume_spike_reversal_short":
                return v >= self.FF_MIN_VOL_MULT
            return False

        if regime == "trend_down":
            if setup_type == "breakout_short":
                return (a >= self.BO_MIN_ADX) and (v >= self.BO_MIN_VOL_MULT)
            if setup_type == "vwap_lose_short":
                return (s >= self.VWAP_MIN_STRENGTH) and (a >= self.VWAP_MIN_ADX) and (v >= self.VWAP_MIN_VOL_MULT)
            # INSTITUTIONAL FIX: failure_fade_long DISABLED in trend_down
            # Analysis showed 5/5 trades (100%) hit hard_sl in trend_down (-Rs.2,606)
            # Mean reversion doesn't work when catching falling knives
            # Only allow failure_fade_short (fade resistance) in downtrends
            # INSTITUTIONAL FIX: Require ADX >= 15 to avoid weak/choppy trades
            if setup_type == "failure_fade_short":
                return (v >= self.FF_MIN_VOL_MULT) and (a >= 15.0)
            # Trend continuation setups in downtrend
            if setup_type == "flag_continuation_short":
                return (a >= self.BO_MIN_ADX) and (v >= self.BO_MIN_VOL_MULT)
            if setup_type == "trend_pullback_short":
                return (s >= self.VWAP_MIN_STRENGTH) and (v >= self.VWAP_MIN_VOL_MULT)
            # Resistance bounce disabled - poor performance per diagnostic report
            if setup_type == "gap_fill_short":
                return v >= self.VWAP_MIN_VOL_MULT
            # Volume spike reversals can work as counter-trend in strong trends
            if setup_type == "volume_spike_reversal_long":
                return v >= self.FF_MIN_VOL_MULT
            return False

        if regime == "squeeze":
            if setup_type in {"squeeze_release_long", "squeeze_release_short"}:
                return v >= self.VWAP_MIN_VOL_MULT  # need participation on release
            if setup_type in {"breakout_long", "breakout_short"}:
                return (a >= self.BO_MIN_ADX) and (v >= self.BO_MIN_VOL_MULT)
            if setup_type in {"vwap_reclaim_long", "vwap_lose_short"}:
                return (s >= self.VWAP_MIN_STRENGTH) and (a >= self.VWAP_MIN_ADX) and (v >= self.VWAP_MIN_VOL_MULT)
            # ORB breakouts disabled - poor performance per diagnostic report
            if setup_type in {"volume_spike_reversal_long", "volume_spike_reversal_short"}:
                return v >= self.VWAP_MIN_VOL_MULT
            # Gap fills can trigger squeeze releases
            if setup_type in {"gap_fill_long", "gap_fill_short"}:
                return v >= self.VWAP_MIN_VOL_MULT
            return False

        return False

    # ---------------- Sizing bias ----------------
    def size_multiplier(self, regime: str, *, counter_trend: bool = False, setup_type: str = None) -> float:
        if regime in ("trend_up", "trend_down"):
            return 0.70 if counter_trend else 1.15
        if regime == "squeeze":
            return 0.90
        if regime == "chop":
            # Aggressive size reduction for breakouts in chop (ORB breakouts disabled)
            if setup_type in {"breakout_long", "breakout_short"}:
                return 0.60  # More conservative for risky chop breakouts
            # Standard chop reduction for other setups
            return 0.85
        return 0.85  # default chop
