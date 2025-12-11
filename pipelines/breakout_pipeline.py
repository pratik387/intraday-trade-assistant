# pipelines/breakout_pipeline.py
"""
Breakout Pipeline - Specialized pipeline for BREAKOUT category setups.

BREAKOUT setups are momentum breaks of key levels:
- orb_breakout, orb_breakdown
- breakout (generic), level_breakout
- flag_continuation
- squeeze_release
- momentum_breakout
- range_breakout, range_breakdown
- gap_breakout
- resistance_breakout, support_breakdown
- break_of_structure, change_of_character
- equilibrium_breakout

Quality Metric: volume_ratio * breakout_strength / normalized_risk
- Measures STRENGTH of the break, not arbitrary distance to levels
- Volume surge confirms institutional participation
- Breakout distance (ATR-normalized) measures momentum

Key Filters (from trade_decision_gate.py):
- Volume surge >= 1.2x average (1.5x for shorts)
- Momentum candle >= 1.5x average (2.0x for shorts)
- ADX >= 20 for trend confirmation
- Time window restrictions (10:30-12:30, 14:15-15:00)

Regime Rules:
- Best in trend_up/trend_down (go with momentum)
- Penalized in chop (false breakouts)
- ORB exception: Allowed in chop before 10:30 (consolidation breakout)
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from config.logging_config import get_agent_logger

from .base_pipeline import (
    BasePipeline,
    ScreeningResult,
    QualityResult,
    GateResult,
    RankingResult,
    EntryResult,
    TargetResult,
    get_cap_segment
)

logger = get_agent_logger()


class BreakoutPipeline(BasePipeline):
    """Pipeline for BREAKOUT category setups."""

    def get_category_name(self) -> str:
        return "BREAKOUT"

    def get_setup_types(self) -> List[str]:
        return self._get("setup_types")

    # ======================== SCREENING ========================

    def screen(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        features: Dict[str, Any],
        levels: Dict[str, float],
        now: pd.Timestamp
    ) -> ScreeningResult:
        """
        Breakout-specific screening filters.

        From trade_decision_gate.py:
        - Momentum consolidation filter (avoid buying tops)
        - Volume surge requirement
        - Time window restrictions
        """
        reasons = []
        passed = True

        # levels used for context in momentum checks
        _ = levels

        logger.debug(f"[BREAKOUT] Screening {symbol} at {now}")

        # Time window check from config
        time_cfg = self._get("screening", "time_windows")
        morning_start = self.parse_time(time_cfg["morning_start"])
        morning_end = self.parse_time(time_cfg["morning_end"])
        afternoon_start = self.parse_time(time_cfg["afternoon_start"])
        afternoon_end = self.parse_time(time_cfg["afternoon_end"])
        opening_start = self.parse_time(time_cfg["opening_bell_start"])
        opening_end = self.parse_time(time_cfg["opening_bell_end"])

        md = now.hour * 60 + now.minute
        morning_in = morning_start <= md <= morning_end
        afternoon_in = afternoon_start <= md <= afternoon_end
        opening_bell = opening_start <= md <= opening_end

        if not (morning_in or afternoon_in or opening_bell):
            reasons.append(f"time_window_fail:{md}")
            passed = False

        # Momentum consolidation filter
        min_bars = self._get("screening", "min_bars_required")
        if len(df5m) >= min_bars:
            close_3_bars_ago = df5m["close"].iloc[-4]
            current_close = df5m["close"].iloc[-1]
            momentum_15min = ((current_close - close_3_bars_ago) / close_3_bars_ago) * 100

            mom_cfg = self._get("screening", "momentum_consolidation")
            momentum_threshold = mom_cfg["threshold_pct"]

            if abs(momentum_15min) > momentum_threshold:
                # Stock is moving fast - check quality indicators
                vwap = df5m["vwap"].iloc[-1] if "vwap" in df5m.columns else current_close
                vwap_deviation = abs((current_close - vwap) / vwap) if vwap > 0 else 0.0
                volume_ratio = features["volume_ratio"]

                vwap_threshold = mom_cfg["vwap_deviation_threshold"]
                volume_threshold = mom_cfg["volume_ratio_threshold"]

                near_vwap = vwap_deviation < vwap_threshold
                volume_confirmed = volume_ratio > volume_threshold

                if near_vwap and volume_confirmed:
                    reasons.append(f"momentum_quality_pass:{momentum_15min:.2f}%,vol:{volume_ratio:.2f}x")
                elif not near_vwap:
                    reasons.append(f"momentum_overextended:{vwap_deviation*100:.2f}%>{vwap_threshold*100:.0f}%")
                    passed = False
                else:
                    reasons.append(f"momentum_no_volume:{volume_ratio:.2f}x<{volume_threshold}x")
                    passed = False
            else:
                reasons.append(f"momentum_consolidation_ok:{momentum_15min:.2f}%")

        # First Hour Momentum (FHM) detection
        # If FHM conditions are met, add to features for downstream processing
        fhm_cfg = self._get("screening", "first_hour_momentum")
        if fhm_cfg and fhm_cfg.get("enabled", False):
            fhm_result = self._check_fhm_conditions(df5m, features, now, fhm_cfg)
            features["fhm_eligible"] = fhm_result.get("eligible", False)
            features["fhm_rvol"] = fhm_result.get("rvol", 0.0)
            features["fhm_price_move_pct"] = fhm_result.get("price_move_pct", 0.0)
            if fhm_result.get("eligible"):
                reasons.append(f"fhm_eligible:rvol{fhm_result['rvol']:.1f}x,move{fhm_result['price_move_pct']:.1f}%")
                logger.info(f"[BREAKOUT] FHM eligible: {symbol} RVOL={fhm_result['rvol']:.2f}x, Move={fhm_result['price_move_pct']:.2f}%")

        return ScreeningResult(passed=passed, reasons=reasons, features=features)

    def _check_fhm_conditions(
        self,
        df5m: pd.DataFrame,
        features: Dict[str, Any],
        now: pd.Timestamp,
        fhm_cfg: Dict
    ) -> Dict[str, Any]:
        """
        Check if First Hour Momentum (FHM) conditions are met.

        FHM captures big movers early using:
        1. Time: 09:15 - 10:15 only (from config)
        2. RVOL: >= min_rvol (from config)
        3. Price move: >= min_price_move_pct (from config)
        4. Volume: >= min_volume_1m (from config)

        RVOL CALCULATION - EXPERIMENTAL:
        ================================
        Uses hybrid approach: avg daily volume × first-hour fraction (35%).
        This is based on research that first hour typically accounts for ~35% of daily volume.

        If this causes quality issues (too many/few FHM triggers), we may need to revert to:
        - Option A: Pure intraday RVOL (compare current bar to recent intraday bars)
        - Option B: Different first-hour fraction (30-40% typical range)
        - Option C: Time-weighted approach (9:30 uses 10%, 10:15 uses 35%)

        Fallback: If avg_daily_volume unavailable, uses intraday volume comparison.

        All parameters MUST be defined in config - no defaults.
        """
        result = {"eligible": False, "rvol": 0.0, "price_move_pct": 0.0, "rvol_method": "unknown"}

        # Check time window - MUST be in config
        time_cfg = fhm_cfg["time_window"]
        start_time = self.parse_time(time_cfg["start"])
        end_time = self.parse_time(time_cfg["end"])

        md = now.hour * 60 + now.minute
        in_window = start_time <= md <= end_time

        if not in_window:
            return result  # Outside first hour

        # Get trigger thresholds - MUST be in config
        triggers = fhm_cfg["triggers"]
        min_rvol = triggers["min_rvol"]
        min_price_move = triggers["min_price_move_pct"]
        min_volume = triggers["min_volume_1m"]

        # ===========================================
        # RVOL CALCULATION - EXPERIMENTAL HYBRID APPROACH
        # ===========================================
        # Uses avg daily volume with first-hour adjustment factor.
        # May need to change if quality degrades - see docstring for alternatives.
        #
        # First hour (9:15-10:15) typically = 35% of daily volume.
        # This is a common observation in Indian markets.
        # If this doesn't work, try 30% or 40%, or use time-weighted approach.
        FIRST_HOUR_VOLUME_FRACTION = 0.35  # EXPERIMENTAL: May need tuning

        avg_daily_volume = features.get("avg_daily_volume")

        if avg_daily_volume is not None and avg_daily_volume > 0 and len(df5m) >= 1:
            # HYBRID RVOL: Use daily volume adjusted for first-hour fraction
            # Expected first hour volume = avg daily vol × 35%
            # Then pro-rate based on how much of first hour has elapsed
            expected_first_hour_vol = avg_daily_volume * FIRST_HOUR_VOLUME_FRACTION

            # Calculate minutes elapsed since market open (9:15)
            market_open_minutes = 9 * 60 + 15  # 9:15 AM
            minutes_since_open = max(5, md - market_open_minutes)  # At least 5 min
            first_hour_duration = 60  # 60 minutes in first hour

            # Expected volume so far = first hour vol × (elapsed / 60)
            elapsed_fraction = min(1.0, minutes_since_open / first_hour_duration)
            expected_vol_so_far = expected_first_hour_vol * elapsed_fraction

            # Session volume = sum of all 5m bars so far
            session_volume = float(df5m["volume"].sum())

            # True RVOL = actual session vol / expected session vol
            rvol = session_volume / expected_vol_so_far if expected_vol_so_far > 0 else 0.0
            result["rvol_method"] = "hybrid_daily"
        else:
            # FALLBACK: Intraday-only RVOL (original approach)
            # Use this if daily_df not available - less accurate early in session
            if len(df5m) >= 5 and "volume" in df5m.columns:
                current_vol = float(df5m["volume"].iloc[-1])
                lookback = min(20, len(df5m) - 1)
                avg_vol = df5m["volume"].iloc[-lookback-1:-1].mean() if lookback > 0 else current_vol
                rvol = current_vol / avg_vol if avg_vol > 0 else 0.0
                result["rvol_method"] = "intraday_fallback"
            else:
                rvol = features.get("volume_ratio", 0.0)
                result["rvol_method"] = "features_fallback"

        result["rvol"] = rvol

        # Calculate price move from open
        if len(df5m) >= 2:
            open_price = float(df5m["open"].iloc[0])  # First bar open = day open
            current_close = float(df5m["close"].iloc[-1])
            price_move_pct = abs((current_close - open_price) / open_price) * 100 if open_price > 0 else 0.0
        else:
            price_move_pct = 0.0

        result["price_move_pct"] = price_move_pct

        # Check current bar volume
        current_bar_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0

        # Check all conditions
        rvol_ok = rvol >= min_rvol
        price_ok = price_move_pct >= min_price_move
        volume_ok = current_bar_volume >= min_volume

        result["eligible"] = rvol_ok and price_ok and volume_ok

        if result["eligible"]:
            logger.debug(f"FHM eligible: RVOL={rvol:.2f}x, Move={price_move_pct:.2f}%, Vol={current_bar_volume/1000:.0f}k")

        return result

    # ======================== QUALITY ========================

    def calculate_quality(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        bias: str,
        levels: Dict[str, float],
        atr: float
    ) -> QualityResult:
        """
        Breakout quality: volume * breakout_strength / normalized_risk

        From planner_internal.py lines 1162-1177:
        - Breakout distance (ATR-normalized) measures momentum
        - Volume ratio confirms institutional participation
        - Quality = strength of the break, not distance to arbitrary levels

        NOTE: ORB setups naturally have LOW structural_rr (near 0) since we enter AT the level.
        The old config handles this with strategy_structural_rr_overrides: 0.25 for orb_* setups.
        """
        logger.debug(f"[BREAKOUT] Calculating quality for {symbol} bias={bias}")
        current_close = float(df5m["close"].iloc[-1])
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)

        # Volume ratio with floor from config
        vol_ratio = self.get_volume_ratio(df5m)
        vol_floor = self._get("quality", "volume_ratio_floor")
        vol_ratio = max(vol_ratio, vol_floor)

        # ADX for trend strength
        adx = float(df5m["adx"].iloc[-1]) if "adx" in df5m.columns and not pd.isna(df5m["adx"].iloc[-1]) else 20.0

        # Breakout strength calculation
        if bias == "long":
            breakout_distance = max(current_close - orh, 0)
        else:
            breakout_distance = max(orl - current_close, 0)

        breakout_strength = breakout_distance / max(atr, 1e-6)

        # Calculate risk per share (simplified)
        rps = atr * 1.5  # Standard risk

        # Structural R:R = volume * breakout_strength / normalized_risk
        structural_rr = (vol_ratio * breakout_strength) / max(rps / atr, 1e-6)

        # Quality status from config thresholds
        quality_cfg = self._get("quality", "quality_thresholds")
        if structural_rr >= quality_cfg["excellent"]["min_structural_rr"] and vol_ratio >= quality_cfg["excellent"]["min_volume_ratio"]:
            quality_status = "excellent"
        elif structural_rr >= quality_cfg["good"]["min_structural_rr"] and vol_ratio >= quality_cfg["good"]["min_volume_ratio"]:
            quality_status = "good"
        elif structural_rr >= quality_cfg["fair"]["min_structural_rr"]:
            quality_status = "fair"
        else:
            quality_status = "poor"

        metrics = {
            "volume_ratio": round(vol_ratio, 2),
            "breakout_distance": round(breakout_distance, 2),
            "breakout_strength": round(breakout_strength, 3),
            "adx": round(adx, 1),
        }

        reasons = [f"vol={vol_ratio:.2f}", f"break_str={breakout_strength:.3f}", f"adx={adx:.1f}"]

        return QualityResult(
            structural_rr=structural_rr,
            quality_status=quality_status,
            metrics=metrics,
            reasons=reasons
        )

    # ======================== GATES ========================

    def validate_gates(
        self,
        symbol: str,
        setup_type: str,
        regime: str,
        df5m: pd.DataFrame,
        df1m: Optional[pd.DataFrame],
        strength: float,
        adx: float,
        vol_mult: float,
        regime_diagnostics: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        Breakout-specific gate validations.

        From trade_decision_gate.py:
        - Volume surge filter (lines 974-1006)
        - Momentum candle filter (lines 1008-1034)
        - Regime allowlist (breakout penalized in chop except ORB before 10:30)
        """
        # strength and vol_mult used for logging/context
        logger.debug(f"[BREAKOUT] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, strength={strength:.2f}, vol={vol_mult:.2f}")

        reasons = []
        passed = True
        size_mult = 1.0
        min_hold = 0

        # ========== CROSS-RUN VALIDATED FILTERS (Dec 2024) ==========
        # These filters were validated across both backtest_20251204 and backtest_20251205
        validated_filters = self._get("gates", "validated_filters") or {}

        # Get current hour from df5m index
        current_time = df5m.index[-1] if hasattr(df5m.index[-1], 'hour') else None
        current_hour = current_time.hour if current_time else 0

        # Get volume from 5m bar for long trade volume filter
        bar5_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0
        is_long = "_long" in setup_type

        # Early FHM detection for bypass logic
        is_fhm = "first_hour_momentum" in setup_type.lower()

        # 1. BREAKOUT_LONG: HOUR = 11 FILTER
        # Evidence: R1: +1,695 Rs (91 trades) | R2: +1,554 Rs (92 trades) - 11AM breakouts have low WR
        if setup_type == "breakout_long":
            filter_cfg = validated_filters.get("breakout_long_hour_11", {})
            blocked_hours = filter_cfg.get("blocked_hours")
            if current_hour in blocked_hours:
                reasons.append(f"breakout_long_blocked:hour{current_hour}_in_blocked_list")
                logger.debug(f"[BREAKOUT] {symbol} breakout_long BLOCKED: Hour {current_hour} is in blocked hours {blocked_hours}")
                return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
            else:
                reasons.append(f"breakout_long_hour_ok:{current_hour}")

        # 2. LONG TRADE VOLUME FILTER (MIN 150K)
        # Evidence: Winners have 2x volume (213k vs 107k)
        # FHM BYPASS: FHM uses RVOL (relative volume) from screening, not absolute volume
        # Small caps with high RVOL (6.97x like MAHLOG) are valid FHM plays even with <150k absolute volume
        if is_long and not is_fhm:
            filter_cfg = validated_filters.get("long_trade_volume")
            min_volume = filter_cfg.get("min_volume")
            if bar5_volume < min_volume:
                reasons.append(f"long_vol_blocked:{bar5_volume/1000:.0f}k<{min_volume/1000:.0f}k")
                logger.debug(f"[BREAKOUT] {symbol} {setup_type} BLOCKED: Long vol {bar5_volume/1000:.0f}k < {min_volume/1000:.0f}k")
                return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
            else:
                reasons.append(f"long_vol_ok:{bar5_volume/1000:.0f}k")
        elif is_long and is_fhm:
            reasons.append(f"fhm_long_vol_bypass:{bar5_volume/1000:.0f}k:uses_rvol")

        # 3. ORB_BREAKOUT_LONG CAP SEGMENT FILTER
        # DATA-DRIVEN (Dec 2024): Large cap 92% WR vs mid/small 43% WR
        # Allow: large_cap always, mid_cap with strength>=0.7 and entry>=35min
        if setup_type == "orb_breakout_long":
            cap_filter_cfg = validated_filters.get("orb_breakout_long_cap_filter", {})
            if cap_filter_cfg.get("enabled"):
                cap_segment = get_cap_segment(symbol)
                allowed_caps = cap_filter_cfg.get("allowed_caps", ["large_cap"])
                entry_minute = current_time.minute if current_time else 30

                if cap_segment in allowed_caps:
                    # Large cap always allowed
                    reasons.append(f"orb_cap_ok:{cap_segment}")
                elif cap_segment == "mid_cap":
                    # Mid cap allowed with additional filters
                    mid_cfg = cap_filter_cfg.get("mid_cap_with_filters", {})
                    if mid_cfg.get("enabled"):
                        min_srr = mid_cfg.get("min_structural_rr")
                        min_entry_min = mid_cfg.get("min_entry_minute")
                        # strength parameter is structural_rr (passed from calculate_quality)
                        structural_rr = strength

                        srr_ok = structural_rr >= min_srr
                        entry_ok = entry_minute >= min_entry_min

                        if srr_ok and entry_ok:
                            reasons.append(f"orb_mid_cap_ok:srr{structural_rr:.2f}>={min_srr},entry{entry_minute}>={min_entry_min}")
                        else:
                            fail_parts = []
                            if not srr_ok:
                                fail_parts.append(f"srr{structural_rr:.2f}<{min_srr}")
                            if not entry_ok:
                                fail_parts.append(f"entry{entry_minute}<{min_entry_min}")
                            reasons.append(f"orb_mid_cap_blocked:{','.join(fail_parts)}")
                            logger.info(f"[BREAKOUT] {symbol} orb_breakout_long BLOCKED: mid_cap requires srr>={min_srr} AND entry>={min_entry_min}min, got srr={structural_rr:.2f}, entry={entry_minute}")
                            return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
                    else:
                        # Mid cap not allowed without filters
                        reasons.append(f"orb_cap_blocked:mid_cap_filters_disabled")
                        logger.info(f"[BREAKOUT] {symbol} orb_breakout_long BLOCKED: mid_cap not allowed (filters disabled)")
                        return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
                else:
                    # Small cap, micro cap, unknown - blocked
                    reasons.append(f"orb_cap_blocked:{cap_segment}")
                    logger.info(f"[BREAKOUT] {symbol} orb_breakout_long BLOCKED: {cap_segment} not in allowed list {allowed_caps}")
                    return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)

        # ORB detection - structure detector already enforces 10:30 cutoff via should_detect_at_time()
        # So ALL ORB setups that reach here are within the valid window - use relaxed thresholds
        is_orb = "orb" in setup_type.lower()
        # Note: is_fhm already defined earlier (line 393) for volume bypass

        # ========== FHM REGIME OVERRIDE ==========
        # Check if First Hour Momentum conditions allow regime bypass
        # When RVOL >= threshold during first hour, institutional flow trumps regime blocking
        # All parameters MUST be in config - no defaults
        fhm_regime_override = False
        fhm_cfg = self._get("screening", "first_hour_momentum")
        if fhm_cfg and fhm_cfg.get("enabled"):
            override_cfg = fhm_cfg.get("regime_override")
            if override_cfg and override_cfg.get("enabled"):
                # Check if in first hour time window - MUST be in config
                time_cfg = fhm_cfg["time_window"]
                start_time = self.parse_time(time_cfg["start"])
                end_time = self.parse_time(time_cfg["end"])
                md = current_time.hour * 60 + current_time.minute if current_time else 0

                if start_time <= md <= end_time:
                    # Calculate RVOL to check for override threshold
                    if len(df5m) >= 5 and "volume" in df5m.columns:
                        current_vol = float(df5m["volume"].iloc[-1])
                        lookback = min(20, len(df5m) - 1)
                        avg_vol = df5m["volume"].iloc[-lookback-1:-1].mean() if lookback > 0 else current_vol
                        rvol = current_vol / avg_vol if avg_vol > 0 else 0.0
                    else:
                        rvol = vol_mult  # Fallback to passed volume ratio

                    # RVOL threshold MUST be in config
                    rvol_threshold = override_cfg["rvol_threshold_for_override"]
                    if rvol >= rvol_threshold:
                        fhm_regime_override = True
                        reasons.append(f"fhm_regime_override:rvol{rvol:.1f}x>={rvol_threshold}x")
                        logger.info(f"[BREAKOUT] FHM regime override for {symbol}: RVOL={rvol:.2f}x >= {rvol_threshold}x, skipping regime blocking")

        # Regime rules from config - HARD GATE for chop (ORB allowed with volume+srr filter, others blocked)
        # Exception: FHM regime override bypasses this check when RVOL >= 3x
        regime_cfg = self._get("gates", "regime_rules")
        if regime in regime_cfg:
            if regime == "chop":
                # FHM OVERRIDE: Skip chop blocking when RVOL >= 3x (institutional flow trumps regime)
                if fhm_regime_override or is_fhm:
                    reasons.append(f"regime_ok:chop_fhm_override")
                elif is_orb:
                    # DATA-DRIVEN FIX (Dec 2024 analysis):
                    # ORB in CHOP: Winners have 283k volume vs Losers 107k volume (+164% diff)
                    # ORB in CHOP: Winners have 1.28 structural_rr vs Losers 2.28 (-44% diff)
                    # Best filter: volume >= 150k AND structural_rr < 1.8 → 64.8% win rate, +5,190 Rs
                    # This turns ORB CHOP from -8,404 Rs loser to +5,190 Rs winner (+13,594 Rs improvement)

                    # Get volume from 5m bar
                    bar5_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0

                    # strength parameter is actually structural_rr (passed from calculate_quality)
                    structural_rr = strength

                    # Get thresholds from config (with defaults from spike test)
                    orb_chop_cfg = self._get("gates", "regime_rules", "chop")
                    min_volume = orb_chop_cfg.get("orb_chop_min_volume")
                    max_structural_rr = orb_chop_cfg.get("orb_chop_max_structural_rr")

                    # ALLOW if volume >= threshold AND structural_rr < threshold
                    volume_ok = bar5_volume >= min_volume
                    srr_ok = structural_rr < max_structural_rr

                    if volume_ok and srr_ok:
                        reasons.append(f"regime_ok:chop_orb_vol{bar5_volume/1000:.0f}k_srr{structural_rr:.2f}")
                    else:
                        # Block trades with low volume OR high structural_rr
                        fail_reasons = []
                        if not volume_ok:
                            fail_reasons.append(f"vol{bar5_volume/1000:.0f}k<{min_volume/1000:.0f}k")
                        if not srr_ok:
                            fail_reasons.append(f"srr{structural_rr:.2f}>={max_structural_rr}")
                        reasons.append(f"regime_blocked:chop_orb_{','.join(fail_reasons)}")
                        passed = False
                else:
                    # Non-ORB breakouts blocked in chop
                    reasons.append("regime_blocked:chop_non_orb")
                    passed = False
            else:
                reasons.append(f"regime_ok:{regime}")

        # Determine if short (stricter filters)
        is_short = "_short" in setup_type

        # Volume surge filter from config
        vol_cfg = self._get("gates", "volume_surge")
        mom_cfg = self._get("gates", "momentum_candle")

        if is_short:
            volume_surge_min = vol_cfg["short"]["min_ratio"]
            momentum_candle_min = mom_cfg["short"]["min_ratio"]
        else:
            volume_surge_min = vol_cfg["long"]["min_ratio"]
            momentum_candle_min = mom_cfg["long"]["min_ratio"]

        # Check 1m volume surge
        # FHM BYPASS: FHM already validated RVOL in screening, skip 1m volume surge check
        lookback = vol_cfg["long"]["lookback_bars"]
        if is_fhm:
            reasons.append(f"fhm_volume_surge_bypass:uses_rvol_from_screening")
        elif df1m is not None and len(df1m) >= 5:
            lookback = min(lookback, len(df1m) - 1)
            if lookback >= 4:
                avg_volume = df1m["volume"].iloc[-lookback-1:-1].mean()
                current_volume = df1m["volume"].iloc[-1]

                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume

                    if volume_ratio < volume_surge_min:
                        reasons.append(f"volume_surge_fail:{volume_ratio:.2f}x<{volume_surge_min}x")
                        passed = False
                    else:
                        reasons.append(f"volume_surge_pass:{volume_ratio:.2f}x")

        # Momentum candle filter
        # FHM BYPASS: FHM uses RVOL-based momentum, not candle analysis
        if is_fhm:
            reasons.append(f"fhm_momentum_candle_bypass:uses_rvol")
        elif df1m is not None and len(df1m) >= 5:
            last_5 = df1m.tail(5)
            candle_sizes = (last_5["high"] - last_5["low"]).values
            current_candle = candle_sizes[-1]
            avg_prev = candle_sizes[:-1].mean()

            if avg_prev > 0:
                candle_ratio = current_candle / avg_prev

                if candle_ratio < momentum_candle_min:
                    reasons.append(f"momentum_candle_fail:{candle_ratio:.2f}x<{momentum_candle_min}x")
                    passed = False
                else:
                    reasons.append(f"momentum_candle_pass:{candle_ratio:.2f}x")

        # ADX gate from config - HARD GATE
        # ORB uses relaxed threshold (15) - ADX takes 14+ bars to stabilize, structure detector enforces time
        # FHM BYPASS: FHM uses RVOL-based momentum instead of ADX (RVOL >= 2x is the signal)
        adx_cfg = self._get("gates", "adx")
        if is_fhm:
            reasons.append(f"fhm_adx_bypass:{adx:.1f}:uses_rvol_instead")
        else:
            base_min_adx = adx_cfg["min_value"]  # 20 for non-ORB
            orb_min_adx = adx_cfg.get("orb_early_min_value")  # 15 for ORB
            min_adx = orb_min_adx if is_orb else base_min_adx
            if adx < min_adx:
                reasons.append(f"adx_low:{adx:.1f}<{min_adx}")
                passed = False
            else:
                reasons.append(f"adx_ok:{adx:.1f}>={min_adx}")

        # RSI weak momentum gate - HARD GATE
        # FHM BYPASS: FHM uses RVOL-based momentum instead of RSI
        # ORB uses relaxed thresholds - RSI erratic at market open
        rsi_cfg = self.cfg["rsi_penalty"]
        rsi = float(df5m["rsi"].iloc[-1]) if "rsi" in df5m.columns else 50.0
        if is_fhm:
            # FHM validates momentum via RVOL in screening, RSI check not applicable
            reasons.append(f"fhm_rsi_bypass:{rsi:.0f}:uses_rvol_instead")
        elif is_short:
            # Short breakouts need RSI weakness (high RSI = no selling pressure)
            threshold = rsi_cfg.get("orb_early_short_threshold") if is_orb else rsi_cfg["short_weak_threshold"]
            if rsi < threshold:
                reasons.append(f"rsi_no_selling_pressure:{rsi:.0f}<{threshold}")
                passed = False
            else:
                reasons.append(f"rsi_ok:{rsi:.0f}>={threshold}")
        else:
            # Long breakouts need RSI strength (low RSI = no buying pressure)
            threshold = rsi_cfg.get("orb_early_long_threshold") if is_orb else rsi_cfg["long_weak_threshold"]
            if rsi < threshold:
                reasons.append(f"rsi_weak_momentum:{rsi:.0f}<{threshold}")
                passed = False
            else:
                reasons.append(f"rsi_ok:{rsi:.0f}>={threshold}")

        return GateResult(passed=passed, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)

    # ======================== RANKING ========================

    def calculate_rank_score(
        self,
        symbol: str,
        intraday_features: Dict[str, Any],
        regime: str,
        daily_trend: Optional[str] = None,
        htf_context: Optional[Dict] = None
    ) -> RankingResult:
        """
        Breakout-specific ranking - ALL 9 COMPONENTS FROM OLD ranker.py _intraday_strength().

        From ranker.py lines 95-131:
        1. s_vol  - Volume ratio: min(vol_ratio / divisor, cap)
        2. s_rsi  - RSI score: max((rsi - mid) / divisor, floor)
        3. s_rsis - RSI slope: max(min(rsi_slope, cap), 0)
        4. s_adx  - ADX score: max((adx - mid) / divisor, floor)
        5. s_adxs - ADX slope: max(min(adx_slope, cap), 0)
        6. s_vwap - VWAP alignment: bonus if aligned, penalty if not
        7. s_dist - Distance from level: near/ok/far scoring
        8. s_sq   - Squeeze percentile: <=50/<=70/>=90 scoring
        9. s_acc  - Acceptance status: excellent/good bonus

        HTF Logic for BREAKOUT:
        - Breakouts want HTF (15m) trend ALIGNED with setup direction
        - +20% boost if HTF trend aligned
        - -10% penalty if HTF trend opposing
        - +10% bonus if HTF volume surge (confirms institutional participation)
        """
        logger.debug(f"[BREAKOUT] Calculating rank score for {symbol} in {regime}")

        # REQUIRED features (core indicators - must exist)
        vol_ratio = float(intraday_features["volume_ratio"])
        rsi = float(intraday_features["rsi"])
        adx = float(intraday_features["adx"])
        above_vwap = bool(intraday_features["above_vwap"])
        bias = intraday_features["bias"]
        # OPTIONAL features (derived/data-dependent - may not always be available)
        rsi_slope = float(intraday_features.get("rsi_slope") or 0.0)
        adx_slope = float(intraday_features.get("adx_slope") or 0.0)
        dist_from_level_bpct = float(intraday_features.get("dist_from_level_bpct") or 9.99)
        squeeze_pctile = intraday_features.get("squeeze_pctile")
        acceptance_status = intraday_features.get("acceptance_status") or "poor"

        # Component scores from config - ALL 9 COMPONENTS
        weights = self._get("ranking", "weights")

        # 1. VOLUME SCORE (s_vol)
        vol_cfg = weights["volume"]
        s_vol = min(vol_ratio / vol_cfg["divisor"], vol_cfg["cap"])

        # 2. RSI SCORE (s_rsi) - PORTED FROM OLD ranker.py line 97
        rsi_cfg = weights["rsi"]
        s_rsi = max((rsi - rsi_cfg["mid"]) / rsi_cfg["divisor"], rsi_cfg["floor"])

        # 3. RSI SLOPE SCORE (s_rsis) - PORTED FROM OLD ranker.py line 98
        rsi_slope_cfg = weights["rsi_slope"]
        s_rsis = max(min(rsi_slope, rsi_slope_cfg["cap"]), 0.0)

        # 4. ADX SCORE (s_adx) - ISSUE 3 FIX: Added cap to prevent high ADX dominating
        adx_cfg = weights["adx"]
        s_adx_raw = max((adx - adx_cfg["mid"]) / adx_cfg["divisor"], adx_cfg["floor"])
        adx_cap = adx_cfg.get("cap")  # Default to no cap if not specified
        s_adx = min(s_adx_raw, adx_cap)

        # 5. ADX SLOPE SCORE (s_adxs) - FROM OLD ranker.py line 100
        adx_slope_cfg = weights["adx_slope"]
        s_adxs = max(min(adx_slope, adx_slope_cfg["cap"]), 0.0)

        # 6. VWAP ALIGNMENT SCORE (s_vwap)
        vwap_cfg = weights["vwap"]
        if bias == "long":
            s_vwap = vwap_cfg["aligned_bonus"] if above_vwap else vwap_cfg["misaligned_penalty"]
        else:
            s_vwap = vwap_cfg["aligned_bonus"] if not above_vwap else vwap_cfg["misaligned_penalty"]

        # 7. DISTANCE FROM LEVEL SCORE (s_dist) - PORTED FROM OLD ranker.py lines 107-113
        dist_cfg = weights["distance"]
        adist = abs(dist_from_level_bpct)
        if adist <= dist_cfg["near_bpct"]:
            s_dist = dist_cfg["near_score"]
        elif adist <= dist_cfg["ok_bpct"]:
            s_dist = dist_cfg["ok_score"]
        else:
            s_dist = dist_cfg["far_score"]

        # 8. SQUEEZE PERCENTILE SCORE (s_sq) - FROM OLD ranker.py lines 115-122
        squeeze_cfg = weights["squeeze"]
        if squeeze_pctile is not None:
            if squeeze_pctile <= 50:
                s_sq = squeeze_cfg["low_bonus"]
            elif squeeze_pctile <= 70:
                s_sq = squeeze_cfg["mid_bonus"]
            elif squeeze_pctile >= 90:
                s_sq = squeeze_cfg["high_penalty"]
            else:
                s_sq = 0.0
        else:
            s_sq = 0.0

        # 9. ACCEPTANCE STATUS SCORE (s_acc) - PORTED FROM OLD ranker.py lines 124-130
        acc_cfg = weights["acceptance"]
        if acceptance_status == "excellent":
            s_acc = acc_cfg["excellent_bonus"]
        elif acceptance_status == "good":
            s_acc = acc_cfg["good_bonus"]
        else:
            s_acc = 0.0

        # BASE SCORE = SUM OF ALL 9 COMPONENTS (exactly like OLD ranker.py line 132)
        base_score = s_vol + s_rsi + s_rsis + s_adx + s_adxs + s_vwap + s_dist + s_sq + s_acc

        # Extract setup_type from intraday_features (passed from run_pipeline)
        setup_type = intraday_features["setup_type"]

        # Regime multiplier from config - check strategy-specific first, then fallback
        regime_mult = self._get_strategy_regime_mult(setup_type, regime)

        # NOTE: Daily trend and HTF multipliers are applied ONLY in apply_universal_ranking_adjustments()
        # to match OLD ranker.py which applies them once in rank_candidates().
        # DO NOT apply them here - that would cause double-application!
        # (daily_trend and htf_context are passed to category ranking for logging only)
        _ = daily_trend  # Unused here - applied in universal adjustments
        _ = htf_context  # Unused here - applied in universal adjustments

        final_score = base_score * regime_mult

        logger.debug(f"[BREAKOUT] {symbol} score={final_score:.3f} (vol={s_vol:.2f}, rsi={s_rsi:.2f}, rsis={s_rsis:.2f}, adx={s_adx:.2f}, adxs={s_adxs:.2f}, vwap={s_vwap:.2f}, dist={s_dist:.2f}, sq={s_sq:.2f}, acc={s_acc:.2f}) * regime={regime_mult:.2f}")

        return RankingResult(
            score=final_score,
            components={
                "volume": s_vol,
                "rsi": s_rsi,
                "rsi_slope": s_rsis,
                "adx": s_adx,
                "adx_slope": s_adxs,
                "vwap": s_vwap,
                "distance": s_dist,
                "squeeze": s_sq,
                "acceptance": s_acc
            },
            multipliers={"regime": regime_mult}  # daily/htf applied in universal adjustments
        )

    # ======================== ENTRY ========================

    def calculate_entry(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        bias: str,
        levels: Dict[str, float],
        atr: float,
        setup_type: str
    ) -> EntryResult:
        """
        Breakout entry with setup-type-specific logic.

        BREAKOUT category setups:
        - orb_breakout: Enter AFTER breakout above ORH (longs) or below ORL (shorts)
        - orb_breakdown: Counter-trend bounce at ORL (long) or ORH (short)
        - break_of_structure: Momentum continuation
        - momentum_breakout: Aggressive entry with tight buffer

        Entry principles:
        - Breakouts: Enter AFTER break (ORH + buffer for long, ORL - buffer for short)
        - ORB breakdown (counter): Enter at support/resistance for bounce
        """
        logger.debug(f"[BREAKOUT] Calculating entry for {symbol} {setup_type} bias={bias}")

        current_close = float(df5m["close"].iloc[-1])
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)

        entry_cfg = self._get("entry")
        triggers = entry_cfg["triggers"]
        setup_lower = setup_type.lower()

        # Setup-type-specific entry logic
        if bias == "long":
            # ORB BREAKDOWN LONG - Counter-trend bounce at ORL
            if "orb" in setup_lower and "breakdown" in setup_lower:
                entry_ref = orl + (atr * 0.05)  # Just above support
                entry_trigger = "support_bounce"
            # ORB BREAKOUT LONG - Enter after breakout above ORH
            elif "orb" in setup_lower and "breakout" in setup_lower:
                entry_ref = orh + (atr * 0.05)  # 5% ATR buffer above ORH
                entry_trigger = triggers["long"]
            # MOMENTUM BREAKOUT - Tighter buffer
            elif "momentum" in setup_lower:
                entry_ref = orh + (atr * 0.03)  # 3% ATR buffer
                entry_trigger = triggers["long"]
            # BREAK OF STRUCTURE - At ORH
            elif "break_of_structure" in setup_lower or "bos" in setup_lower:
                entry_ref = orh
                entry_trigger = triggers["long"]
            # DEFAULT BREAKOUT LONG
            else:
                entry_ref = orh
                entry_trigger = triggers["long"]
        else:
            # ORB BREAKOUT SHORT - Counter-trend fade at ORH
            if "orb" in setup_lower and "breakout" in setup_lower:
                entry_ref = orh - (atr * 0.05)  # Just below resistance
                entry_trigger = triggers["short"]
            # ORB BREAKDOWN SHORT - Enter after breakdown below ORL
            elif "orb" in setup_lower and "breakdown" in setup_lower:
                entry_ref = orl - (atr * 0.05)  # 5% ATR buffer below ORL
                entry_trigger = triggers["short"]
            # MOMENTUM BREAKDOWN - Tighter buffer
            elif "momentum" in setup_lower:
                entry_ref = orl - (atr * 0.03)  # 3% ATR buffer
                entry_trigger = triggers["short"]
            # BREAK OF STRUCTURE - At ORL
            elif "break_of_structure" in setup_lower or "bos" in setup_lower:
                entry_ref = orl
                entry_trigger = triggers["short"]
            # DEFAULT BREAKOUT SHORT
            else:
                entry_ref = orl
                entry_trigger = triggers["short"]

        # Entry zone from config
        zone_mult = entry_cfg["zone_mult_atr"]
        zone_width = atr * zone_mult

        # Apply minimum zone width (as % of price) for large cap stocks with low ATR
        min_zone_pct = entry_cfg.get("min_zone_pct")
        if min_zone_pct > 0:
            min_zone_width = current_close * (min_zone_pct / 100.0)
            if zone_width < min_zone_width:
                logger.debug(f"[BREAKOUT] {symbol} zone widened from {zone_width:.3f} to {min_zone_width:.3f} (min_zone_pct={min_zone_pct}%)")
                zone_width = min_zone_width

        # Entry mode from config
        entry_mode = entry_cfg["mode"]

        # CRITICAL FIX: For immediate mode breakouts, use current price for entry zone
        # Reason: Breakouts are detected AFTER price has broken the level, so the entry_ref
        # (based on level) will be behind the current price. If entry_zone is around entry_ref,
        # the current price will be outside the zone and the trade will never trigger.
        #
        # Solution: For immediate mode, create entry zone around current_close instead.
        # This allows immediate execution at current market price.
        if entry_mode == "immediate":
            # Use current close as reference for entry zone (where we'll actually enter)
            entry_zone = (current_close - zone_width, current_close + zone_width)
            logger.debug(f"[BREAKOUT] Immediate mode: entry_zone around current={current_close:.2f} (ref={entry_ref:.2f})")
        else:
            # Conditional mode: use level-based entry_ref
            entry_zone = (entry_ref - zone_width, entry_ref + zone_width)

        return EntryResult(
            entry_zone=entry_zone,
            entry_ref_price=entry_ref,
            entry_trigger=entry_trigger,
            entry_mode=entry_mode
        )

    # ======================== TARGETS ========================

    def calculate_targets(
        self,
        symbol: str,
        entry_ref_price: float,
        hard_sl: float,
        bias: str,
        atr: float,
        levels: Dict[str, float],
        measured_move: float,
        setup_type: str = ""
    ) -> TargetResult:
        """
        Breakout targets: R:R based on measured move.

        Breakouts use aggressive targets since momentum should carry through:
        - T1: 1.5R (primary exit)
        - T2: 2.5R (runner)
        - T3: 3.5R (extended move)

        DATA-DRIVEN FIX: For ORB, use structure-based risk (ORH-ORL) instead of
        entry_price-SL risk. This results in tighter, more achievable targets.

        Problem: Current T1 at 4.15R (5.9% from entry) has only 13% hit rate.
        Fix: Using ORB structure risk brings T1 to ~2.23% from entry.
        Evidence: 25 EOD positive trades could book profit earlier with tighter targets.
        """
        # DATA-DRIVEN FIX: Use structure-based risk for ORB
        # When ORH and ORL are available, use them for risk calculation
        # This produces tighter, more realistic targets for intraday trading
        orh = levels.get("ORH", 0)
        orl = levels.get("ORL", 0)

        # Use structure-based risk when ORH/ORL available and valid
        if orh > 0 and orl > 0 and orh > orl:
            structure_risk = orh - orl
            default_risk = abs(entry_ref_price - hard_sl)

            # Use the smaller of structure risk or default risk
            # This ensures targets are achievable but not too aggressive
            risk_per_share = min(structure_risk, default_risk)
            logger.debug(f"[BREAKOUT] Using structure-based risk for {symbol}: ORH-ORL={structure_risk:.2f} vs entry-SL={default_risk:.2f}, using={risk_per_share:.2f}")
        else:
            # Fallback to default risk calculation
            risk_per_share = abs(entry_ref_price - hard_sl)

        logger.debug(f"[BREAKOUT] Calculating targets for {symbol} entry={entry_ref_price:.2f}, sl={hard_sl:.2f}, risk={risk_per_share:.2f}, mm={measured_move:.2f}")

        # T1/T2/T3 R:R ratios from config
        targets_cfg = self._get("targets")
        rr_ratios = targets_cfg["rr_ratios"]

        # Check for bias-specific targets first (long/short may have different T1/T2/T3)
        if bias in rr_ratios and isinstance(rr_ratios[bias], dict):
            t1_rr = rr_ratios[bias].get("t1", rr_ratios["t1"])
            t2_rr = rr_ratios[bias].get("t2", rr_ratios["t2"])
            t3_rr = rr_ratios[bias].get("t3", rr_ratios["t3"])
            logger.debug(f"[BREAKOUT] Using bias-specific targets for {bias}: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")
        else:
            t1_rr = rr_ratios["t1"]
            t2_rr = rr_ratios["t2"]
            t3_rr = rr_ratios["t3"]

        # Cap targets based on measured move from config
        caps = targets_cfg["caps"]
        cap1 = min(measured_move * caps["t1"]["measured_move_frac"], entry_ref_price * caps["t1"]["max_pct"])
        cap2 = min(measured_move * caps["t2"]["measured_move_frac"], entry_ref_price * caps["t2"]["max_pct"])
        cap3 = min(measured_move * caps["t3"]["measured_move_frac"], entry_ref_price * caps["t3"]["max_pct"])

        # PRE-TRADE REJECTION: If T1 cap < 0.8R, reject the setup
        # Low-volatility instruments (ETFs, liquid funds) can't hit viable targets
        min_t1_threshold = risk_per_share * 0.8
        if cap1 < min_t1_threshold:
            logger.info(f"[BREAKOUT] {symbol} rejected: T1 cap ({cap1:.4f}) < 0.8R ({min_t1_threshold:.4f}) - low volatility")
            return None

        # Calculate target distances with caps
        t1_dist = min(t1_rr * risk_per_share, cap1)
        t2_dist = min(t2_rr * risk_per_share, cap2)
        t3_dist = min(t3_rr * risk_per_share, cap3)

        if bias == "long":
            t1 = entry_ref_price + t1_dist
            t2 = entry_ref_price + t2_dist
            t3 = entry_ref_price + t3_dist
        else:
            t1 = entry_ref_price - t1_dist
            t2 = entry_ref_price - t2_dist
            t3 = entry_ref_price - t3_dist

        # Qty splits from config
        qty_splits = targets_cfg["qty_splits"]
        targets = [
            {"name": "T1", "level": round(t1, 2), "rr": round(t1_rr, 2), "qty_pct": qty_splits["t1"]},
            {"name": "T2", "level": round(t2, 2), "rr": round(t2_rr, 2), "qty_pct": qty_splits["t2"]},
            {"name": "T3", "level": round(t3, 2), "rr": round(t3_rr, 2), "qty_pct": qty_splits["t3"]},
        ]

        # Trail config from config file
        trail_config = targets_cfg["trail"]

        return TargetResult(
            targets=targets,
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            trail_config=trail_config
        )

    # ======================== RSI PENALTY ========================

    def _apply_rsi_penalty(self, rsi_val: float, bias: str) -> tuple:
        """
        BREAKOUT RSI penalty: Penalize weak momentum, reward strong momentum.

        Pro Trader Framework (Minervini, O'Neil):
        - High RSI = momentum confirmation = NO PENALTY (breakouts need momentum)
        - Low RSI = weak momentum = penalty (breakout may fail)
        """
        rsi_cfg = self.cfg["rsi_penalty"]
        long_weak = rsi_cfg["long_weak_threshold"]
        short_weak = rsi_cfg["short_weak_threshold"]
        penalty = rsi_cfg["penalty_mult"]

        if bias == "long" and rsi_val < long_weak:
            return (penalty, f"weak_momentum_rsi<{rsi_val:.0f}")
        elif bias == "short" and rsi_val > short_weak:
            return (penalty, f"weak_momentum_rsi>{rsi_val:.0f}")
        return (1.0, None)
