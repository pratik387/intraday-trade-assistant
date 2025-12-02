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
    TargetResult
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
                volume_ratio = features.get("volume_ratio", 1.0)

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

        return ScreeningResult(passed=passed, reasons=reasons, features=features)

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
        vol_mult: float
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

        # df5m reserved for future bar-level analysis
        _ = df5m

        reasons = []
        passed = True
        size_mult = 1.0
        min_hold = 0

        # Regime rules from config
        regime_cfg = self._get("gates", "regime_rules")
        if regime in regime_cfg:
            rule = regime_cfg[regime]
            if regime == "chop" and "orb" not in setup_type.lower():
                reasons.append("regime_penalty:chop")
                size_mult *= rule["size_mult"]
            else:
                size_mult *= rule["size_mult"]

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
        lookback = vol_cfg["long"]["lookback_bars"]
        if df1m is not None and len(df1m) >= 5:
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
        if df1m is not None and len(df1m) >= 5:
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

        # ADX gate from config
        adx_cfg = self._get("gates", "adx")
        min_adx = adx_cfg["min_value"]
        if adx < min_adx:
            reasons.append(f"adx_low:{adx:.1f}<{min_adx}")
            size_mult *= adx_cfg["low_adx_size_mult"]

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

        # Extract all features from intraday_features
        vol_ratio = float(intraday_features.get("volume_ratio") or 1.0)
        rsi = float(intraday_features.get("rsi") or 50.0)
        rsi_slope = float(intraday_features.get("rsi_slope") or 0.0)
        adx = float(intraday_features.get("adx") or 0.0)
        adx_slope = float(intraday_features.get("adx_slope") or 0.0)
        above_vwap = bool(intraday_features.get("above_vwap", True))
        dist_from_level_bpct = float(intraday_features.get("dist_from_level_bpct") or 9.99)
        squeeze_pctile = intraday_features.get("squeeze_pctile", None)
        acceptance_status = intraday_features.get("acceptance_status", "poor")
        bias = intraday_features.get("bias", "long")

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

        # 4. ADX SCORE (s_adx)
        adx_cfg = weights["adx"]
        s_adx = max((adx - adx_cfg["mid"]) / adx_cfg["divisor"], adx_cfg["floor"])

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
        setup_type = intraday_features.get("setup_type", "breakout")

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

        # Entry zone from config - tight for breakouts
        zone_mult = entry_cfg["zone_mult_atr"]
        zone_width = atr * zone_mult

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
        measured_move: float
    ) -> TargetResult:
        """
        Breakout targets: R:R based on measured move.

        Breakouts use aggressive targets since momentum should carry through:
        - T1: 1.5R (primary exit)
        - T2: 2.5R (runner)
        - T3: 3.5R (extended move)
        """
        # levels and atr reserved for future level-based target adjustments
        _ = levels
        _ = atr

        logger.debug(f"[BREAKOUT] Calculating targets for {symbol} entry={entry_ref_price:.2f}, sl={hard_sl:.2f}, mm={measured_move:.2f}")
        risk_per_share = abs(entry_ref_price - hard_sl)

        # T1/T2/T3 R:R ratios from config
        targets_cfg = self._get("targets")
        rr_ratios = targets_cfg["rr_ratios"]
        t1_rr = rr_ratios["t1"]
        t2_rr = rr_ratios["t2"]
        t3_rr = rr_ratios["t3"]

        # Cap targets based on measured move from config
        caps = targets_cfg["caps"]
        cap1 = min(measured_move * caps["t1"]["measured_move_frac"], entry_ref_price * caps["t1"]["max_pct"])
        cap2 = min(measured_move * caps["t2"]["measured_move_frac"], entry_ref_price * caps["t2"]["max_pct"])
        cap3 = min(measured_move * caps["t3"]["measured_move_frac"], entry_ref_price * caps["t3"]["max_pct"])

        if bias == "long":
            t1 = entry_ref_price + min(t1_rr * risk_per_share, cap1)
            t2 = entry_ref_price + min(t2_rr * risk_per_share, cap2)
            t3 = entry_ref_price + min(t3_rr * risk_per_share, cap3)
        else:
            t1 = entry_ref_price - min(t1_rr * risk_per_share, cap1)
            t2 = entry_ref_price - min(t2_rr * risk_per_share, cap2)
            t3 = entry_ref_price - min(t3_rr * risk_per_share, cap3)

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
