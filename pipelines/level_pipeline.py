# pipelines/level_pipeline.py
"""
Level Pipeline - Specialized pipeline for LEVEL category setups.

LEVEL setups are bounces/rejections at key price levels:
- support_bounce, resistance_bounce
- range_bounce, range_rejection
- vwap_reclaim, vwap_lose
- premium_zone, discount_zone
- orb_pullback
- order_block

Quality Metric: (retest_ok + hold_ok) * proximity_bonus
- Level acceptance is KEY: price must respect the level
- Retest without break confirms support/resistance
- Close above/below level confirms commitment

Key Filters:
- Price must be near the level (ATR-normalized distance)
- Retest should hold within acceptable tolerance
- Current close must be on correct side of level
- Volume spike confirms institutional interest

Regime Rules:
- Works in ALL regimes (levels are always relevant)
- Best in chop/range (levels define the range)
- Good in squeeze (accumulation/distribution zones)
"""

from typing import Dict, List, Optional, Any
import pandas as pd

from config.logging_config import get_agent_logger

from .base_pipeline import (
    BasePipeline,
    ScreeningResult,
    QualityResult,
    GateResult,
    RankingResult,
    EntryResult,
    TargetResult,
)

logger = get_agent_logger()


class LevelPipeline(BasePipeline):
    """Pipeline for LEVEL category setups."""

    def get_category_name(self) -> str:
        return "LEVEL"

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
        Level-specific screening filters.

        - Price must be near the target level
        - Recent bars should show level test behavior
        - No requirement for extreme momentum
        """
        # features reserved for future use
        _ = features

        logger.debug(f"[LEVEL] Screening {symbol} at {now}")

        reasons = []
        passed = True

        # Time window check from config (more permissive than breakouts)
        time_cfg = self._get("screening", "time_windows")
        morning_start = self.parse_time(time_cfg["morning_start"])
        morning_end = self.parse_time(time_cfg["morning_end"])
        afternoon_start = self.parse_time(time_cfg["afternoon_start"])
        afternoon_end = self.parse_time(time_cfg["afternoon_end"])

        md = now.hour * 60 + now.minute
        morning_in = morning_start <= md <= morning_end
        afternoon_in = afternoon_start <= md <= afternoon_end

        if not (morning_in or afternoon_in):
            reasons.append(f"time_window_fail:{md}")
            passed = False

        # Level proximity check
        current_close = float(df5m["close"].iloc[-1])
        # Use ATR from features (already calculated with fallback in run_pipeline)
        atr_fallback_pct = self.cfg.get("atr_fallback_pct")
        atr = features.get("atr") or self.calculate_atr(df5m) or (current_close * atr_fallback_pct)

        # Determine relevant level based on VWAP position
        vwap = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns else current_close
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)
        pdh = levels.get("PDH", orh)
        pdl = levels.get("PDL", orl)

        # Find nearest level
        candidate_levels = [orh, orl, vwap, pdh, pdl]
        candidate_levels = [l for l in candidate_levels if l > 0]

        if candidate_levels:
            nearest_level = min(candidate_levels, key=lambda x: abs(x - current_close))
            distance_to_level = abs(current_close - nearest_level) / max(atr, 1e-6)

            # Level plays need to be near the level (from config)
            max_distance = self._get("screening", "level_proximity", "max_distance_atr")
            if distance_to_level > max_distance:
                reasons.append(f"level_too_far:{distance_to_level:.2f}ATR")
                passed = False
            else:
                reasons.append(f"level_proximity_ok:{distance_to_level:.2f}ATR")

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
        Level quality: (retest_ok + hold_ok) * proximity_bonus

        From planner_internal.py lines 1179-1202:
        - Retest score: Did price test the level without breaking?
        - Hold score: Is current close on correct side?
        - Proximity bonus: Closer to level = higher quality
        """
        logger.debug(f"[LEVEL] Calculating quality for {symbol} bias={bias}")
        current_close = float(df5m["close"].iloc[-1])
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)

        # Target level based on bias
        target_level = orh if bias == "long" else orl

        # Distance to level (ATR-normalized)
        distance_to_level = abs(current_close - target_level) / max(atr, 1e-6)

        # Level acceptance scoring from config
        acc_cfg = self._get("quality", "acceptance")
        acc_bars = int(acc_cfg["bars"])
        acc_bpct = float(acc_cfg["retest_bpct"])

        win = df5m.tail(max(acc_bars, 2))

        if bias == "long":
            # For long bounce: low should not break below level
            retest_score = 1.0 if (win["low"].min() >= target_level * (1 - acc_bpct/100.0)) else 0.5
            hold_score = 1.0 if (win["close"].iloc[-1] >= target_level) else 0.3
        else:
            # For short bounce: high should not break above level
            retest_score = 1.0 if (win["high"].max() <= target_level * (1 + acc_bpct/100.0)) else 0.5
            hold_score = 1.0 if (win["close"].iloc[-1] <= target_level) else 0.3

        # Proximity bonus from config (closer = better)
        prox_cfg = self._get("quality", "proximity_bonus")
        proximity_bonus = max(prox_cfg["floor"], prox_cfg["max_bonus"] - distance_to_level * prox_cfg["decay_per_atr"])

        # Structural R:R for level plays
        quality_mult = self._get("quality", "quality_multiplier")
        structural_rr = (retest_score + hold_score) * proximity_bonus * quality_mult

        # Quality status based on acceptance from config
        quality_cfg = self._get("quality", "quality_thresholds")
        if retest_score >= quality_cfg["excellent"]["retest_score"] and hold_score >= quality_cfg["excellent"]["hold_score"]:
            quality_status = "excellent"
        elif retest_score >= quality_cfg["good"]["retest_score"] and hold_score >= quality_cfg["good"]["hold_score"]:
            quality_status = "good"
        elif retest_score >= quality_cfg["fair"]["retest_score"] or hold_score >= quality_cfg["fair"]["hold_score"]:
            quality_status = "fair"
        else:
            quality_status = "poor"

        metrics = {
            "retest_ok": retest_score >= 0.9,
            "hold_ok": hold_score >= 0.9,
            "distance_to_level": round(distance_to_level, 2),
            "proximity_bonus": round(proximity_bonus, 2),
        }

        reasons = [
            f"retest={retest_score:.1f}",
            f"hold={hold_score:.1f}",
            f"dist={distance_to_level:.2f}ATR"
        ]

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
        Level-specific gate validations - HARD GATES ONLY, NO PENALTIES.

        Level plays work in all regimes but have different requirements:
        - In trend: counter-trend requires stronger confirmation (HARD BLOCK if weak)
        - In chop: level respect is sufficient
        - ADX: high ADX is warning but level plays CAN work (no penalty)
        - Volume: nice-to-have, not required (no penalty)

        Pro Trader Approach (Linda Raschke):
        - Level plays are about price structure, not momentum
        - High ADX = trending = level might break through, but still tradeable
        - Volume surge = confirmation, but absence isn't disqualifying
        """
        # df5m, df1m reserved for future bar-level analysis
        _ = df5m
        _ = df1m

        logger.debug(f"[LEVEL] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, vol={vol_mult:.2f}")

        reasons = []
        passed = True
        size_mult = 1.0  # NO PENALTIES - hard gates only
        min_hold = 0

        # Regime rules from config - HARD GATES only
        regime_cfg = self._get("gates", "regime_rules")

        if regime in regime_cfg:
            rule = regime_cfg[regime]
            if not rule["allowed"]:
                reasons.append(f"regime_blocked:{regime}")
                passed = False
            else:
                reasons.append(f"regime_ok:{regime}")

                # Counter-trend check for trending regimes - HARD BLOCK if too weak
                if regime in ("trend_up", "trend_down"):
                    min_strength = rule["min_strength_counter_trend"]
                    is_long = "_long" in setup_type
                    is_counter = (regime == "trend_up" and not is_long) or (regime == "trend_down" and is_long)

                    if is_counter and strength < min_strength:
                        reasons.append(f"counter_trend_blocked:strength={strength:.2f}<{min_strength}")
                        passed = False
                    elif is_counter:
                        reasons.append(f"counter_trend_ok:strength={strength:.2f}")
                    else:
                        reasons.append("trend_aligned")

        # Volume confirmation - info only, no penalty
        vol_cfg = self._get("gates", "volume_confirmation")
        if vol_mult >= vol_cfg["threshold"]:
            reasons.append(f"volume_confirmed:{vol_mult:.2f}x")
        else:
            reasons.append(f"volume_low:{vol_mult:.2f}x")

        # ADX check - info only, no penalty (level plays work even in high ADX)
        adx_cfg = self._get("gates", "adx")
        if adx > adx_cfg["high_adx_threshold"]:
            reasons.append(f"high_adx_warning:{adx:.1f}")
        else:
            reasons.append(f"adx_ok:{adx:.1f}")

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
        Level-specific ranking - ALL 9 COMPONENTS FROM OLD ranker.py _intraday_strength().

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

        HTF Logic for LEVEL:
        - Level plays (bounces) often work AGAINST the HTF trend
        - +15% bonus if HTF trend OPPOSING (bounce into trend)
        - -5% penalty if HTF trend aligned (less important, not blocking)
        """
        logger.debug(f"[LEVEL] Calculating rank score for {symbol} in {regime}")

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

        # Regime multiplier from config - level plays excel in chop
        setup_type = intraday_features.get("setup_type", "")
        regime_mult = self._get_strategy_regime_mult(setup_type, regime)

        # NOTE: Daily trend and HTF multipliers are applied ONLY in apply_universal_ranking_adjustments()
        # to match OLD ranker.py which applies them once in rank_candidates().
        # DO NOT apply them here - that would cause double-application!
        _ = daily_trend  # Unused here - applied in universal adjustments
        _ = htf_context  # Unused here - applied in universal adjustments

        final_score = base_score * regime_mult

        logger.debug(f"[LEVEL] {symbol} score={final_score:.3f} (vol={s_vol:.2f}, rsi={s_rsi:.2f}, rsis={s_rsis:.2f}, adx={s_adx:.2f}, adxs={s_adxs:.2f}, vwap={s_vwap:.2f}, dist={s_dist:.2f}, sq={s_sq:.2f}, acc={s_acc:.2f}) * regime={regime_mult:.2f}")

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
        Level entry: Tight zones at the level.

        Level plays need entries AT or very near the level for optimal R:R.
        Entry zone is tighter than breakouts (0.10 ATR).
        """
        logger.debug(f"[LEVEL] Calculating entry for {symbol} {setup_type} bias={bias}")

        current_close = float(df5m["close"].iloc[-1])
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)
        pdh = levels.get("PDH", orh)
        pdl = levels.get("PDL", orl)
        vwap = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns else current_close

        entry_cfg = self._get("entry")
        triggers = entry_cfg["triggers"]
        mode_cfg = entry_cfg["mode"]
        ict_cfg = entry_cfg["ict_zones"]
        setup_lower = setup_type.lower()

        # Setup-type-specific entry logic for LEVEL category
        if "vwap" in setup_lower:
            # VWAP plays - enter at VWAP
            if bias == "long":
                entry_ref = vwap * 0.999  # Just below VWAP for long
            else:
                entry_ref = vwap * 1.001  # Just above VWAP for short
            entry_trigger = triggers["vwap"]
        elif "premium" in setup_lower or "discount" in setup_lower or "order_block" in setup_lower:
            # ICT zones - premium/discount
            if bias == "long":
                entry_ref = orl + (orh - orl) * ict_cfg["discount_zone_frac"]  # 30% into range
            else:
                entry_ref = orl + (orh - orl) * ict_cfg["premium_zone_frac"]  # 70% into range
            entry_trigger = triggers.get("premium_zone", triggers["default"])
        elif "pullback" in setup_lower or "retest" in setup_lower:
            # Pullback/Retest - enter at support (PDL) or resistance (PDH)
            if bias == "long":
                entry_ref = min(orl, pdl) if pdl > 0 else orl
            else:
                entry_ref = max(orh, pdh) if pdh > 0 else orh
            entry_trigger = triggers["default"]
        elif "reversal" in setup_lower:
            # Level reversal - enter at structure
            if bias == "long":
                entry_ref = min(orl, pdl) if pdl > 0 else orl
            else:
                entry_ref = max(orh, pdh) if pdh > 0 else orh
            entry_trigger = triggers["default"]
        elif "range" in setup_lower:
            # Range play - enter at range edge
            entry_ref = orl if bias == "long" else orh
            entry_trigger = triggers["default"]
        else:
            # Default LEVEL - entry at support/resistance
            entry_ref = orl if bias == "long" else orh
            entry_trigger = triggers["default"]

        # Tight entry zone for level plays from config
        zone_mult = entry_cfg["zone_mult_atr"]
        zone_width = atr * zone_mult

        entry_zone = (entry_ref - zone_width, entry_ref + zone_width)

        # Entry mode from config
        if "pullback" in setup_type.lower() or "retest" in setup_type.lower():
            entry_mode = mode_cfg["pullback"]
        else:
            entry_mode = mode_cfg["default"]

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
        Level targets: Conservative R:R since level plays are mean-reversion.

        Level plays typically don't run as far as breakouts:
        - T1: 1.2R (quick scalp)
        - T2: 2.0R (standard)
        - T3: 2.5R (extended - rare)
        """
        # levels and atr reserved for future level-based target adjustments
        _ = levels
        _ = atr

        logger.debug(f"[LEVEL] Calculating targets for {symbol} entry={entry_ref_price:.2f}, sl={hard_sl:.2f}, mm={measured_move:.2f}")
        risk_per_share = abs(entry_ref_price - hard_sl)

        # R:R ratios from config
        targets_cfg = self._get("targets")
        rr_ratios = targets_cfg["rr_ratios"]
        t1_rr = rr_ratios["t1"]
        t2_rr = rr_ratios["t2"]
        t3_rr = rr_ratios["t3"]

        # Cap targets from config (level plays shouldn't expect huge moves)
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

    # ======================== RSI PENALTY ========================

    def _apply_rsi_penalty(self, rsi_val: float, bias: str) -> tuple:
        """
        LEVEL RSI penalty: DISABLED - extreme RSI at levels is CONFIRMATION.

        Pro Trader Research (Linda Raschke, Al Brooks):
        - Extreme RSI + level = "probability increases dramatically"
        - Oversold at support = STRONG long signal
        - Overbought at resistance = STRONG short signal
        - Neutral RSI at level = weaker signal (no momentum confirmation)

        This is the OPPOSITE of what we previously thought!
        Extreme RSI is GOOD for level plays, not bad.
        """
        # RSI not used for level plays - extreme RSI is confirmation, not penalty
        _ = rsi_val
        _ = bias
        return (1.0, None)
