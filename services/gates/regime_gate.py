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
from typing import Dict, Tuple
import numpy as np
import pandas as pd

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

    # ---------------- Regime computation ----------------
    def compute_regime(self, df5: pd.DataFrame) -> Tuple[str, float]:
        if df5 is None or df5.empty:
            return "chop", 0.5

        c = df5["close"].astype(float).copy()

        # EMA warmup guards
        if len(c) < 55:
            ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
            ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
        else:
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()

        # slope over ~30 minutes
        window = 6 if len(ema20) >= 6 else max(2, len(ema20))
        slope_fast = float((ema20.diff().tail(window)).mean())

        # width proxy: std(close,20)/vwap if available, else std/mean
        if "bb_width_proxy" in df5.columns:
            width_proxy = float(df5["bb_width_proxy"].tail(20).mean())
        else:
            std20 = float(c.tail(20).std(ddof=0)) if len(c) >= 5 else 0.0
            mean20 = float(c.tail(20).mean()) if len(c) >= 1 else 1.0
            width_proxy = (std20 / mean20) if mean20 else 0.0

        # volume z vs last ~2 hours
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
            slope_fast=slope_fast,
            width_proxy=width_proxy,
            vol_z=vol_z,
        )

        # squeeze if width very low vs recent distribution
        width_series = (df5["bb_width_proxy"] if "bb_width_proxy" in df5.columns else c.pct_change().abs())
        recent_w = width_series.tail(min(24, len(width_series)))
        if len(recent_w) >= 8:
            q30 = float(recent_w.quantile(0.30))
            is_squeeze = m.width_proxy <= q30
        else:
            is_squeeze = False

        if m.ema_fast > m.ema_slow and m.slope_fast > 0 and not is_squeeze:
            regime = "trend_up"
        elif m.ema_fast < m.ema_slow and m.slope_fast < 0 and not is_squeeze:
            regime = "trend_down"
        elif is_squeeze:
            regime = "squeeze"
        else:
            regime = "chop"

        # Calculate confidence based on how clear the regime signal is
        confidence = 0.5  # base confidence

        # Higher confidence for clear trending regimes
        if regime in ("trend_up", "trend_down"):
            slope_strength = min(abs(m.slope_fast) * 1000, 0.3)  # normalize slope
            confidence = min(0.9, 0.6 + slope_strength)
        elif regime == "squeeze":
            confidence = 0.8  # squeeze is usually pretty clear
        # chop gets default 0.5 confidence

        return regime, confidence

    # ---------------- Evidence-based permissioning ----------------
    def allow_setup(
        self,
        setup_type: str,
        regime: str,
        strength: float,
        adx_5m: float,
        vol_mult_5m: float,
    ) -> bool:
        """
        Returns True iff the setup is allowed by regime AND passes evidence thresholds.

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
        """
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
            if setup_type in {"failure_fade_long", "failure_fade_short"}:
                return v >= self.FF_MIN_VOL_MULT

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
            if setup_type == "failure_fade_short":
                return v >= self.FF_MIN_VOL_MULT
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
            if setup_type == "failure_fade_long":
                return v >= self.FF_MIN_VOL_MULT
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
