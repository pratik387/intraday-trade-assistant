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
    TargetResult
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
        atr = self.calculate_atr(df5m)

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
        Level-specific gate validations.

        Level plays work in all regimes but have different requirements:
        - In trend: need stronger confirmation
        - In chop: level respect is sufficient
        """
        # df5m, df1m reserved for future bar-level analysis
        _ = df5m
        _ = df1m

        logger.debug(f"[LEVEL] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, vol={vol_mult:.2f}")

        reasons = []
        passed = True
        size_mult = 1.0
        min_hold = 0

        # Regime rules from config
        regime_cfg = self._get("gates", "regime_rules")

        if regime in regime_cfg:
            rule = regime_cfg[regime]
            if rule.get("allowed", True):
                size_mult *= rule.get("size_mult", 1.0)
                reasons.append(f"regime_ok:{regime}")

                # Counter-trend check for trending regimes
                if regime in ("trend_up", "trend_down"):
                    min_strength = rule.get("min_strength_counter_trend", 0)
                    if strength < min_strength:
                        reasons.append(f"trend_counter_weak:strength={strength:.2f}<{min_strength}")
                        size_mult *= rule.get("counter_trend_size_mult", 0.7)
            else:
                reasons.append(f"regime_blocked:{regime}")
                passed = False

        # Volume confirmation from config
        vol_cfg = self._get("gates", "volume_confirmation")
        if vol_mult >= vol_cfg["threshold"]:
            reasons.append(f"volume_confirmation:{vol_mult:.2f}x")
            size_mult *= vol_cfg["bonus_mult"]

        # ADX check from config (high ADX is caution for level plays)
        adx_cfg = self._get("gates", "adx")
        if adx > adx_cfg["high_adx_threshold"]:
            reasons.append("high_adx_caution")
            size_mult *= adx_cfg["high_adx_size_mult"]

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
        Level-specific ranking.

        From ranker.py:
        - Acceptance status is PRIMARY factor
        - Distance to level matters
        - VWAP proximity bonus

        HTF Logic for LEVEL:
        - Level plays (bounces) often work AGAINST the HTF trend
        - +15% bonus if HTF trend OPPOSING (bounce into trend)
        - -5% penalty if HTF trend aligned (less important, not blocking)
        - HTF volume surge is neutral (doesn't help level plays much)
        """
        # daily_trend reserved for future use
        _ = daily_trend

        logger.debug(f"[LEVEL] Calculating rank score for {symbol} in {regime}")
        vol_ratio = float(intraday_features.get("volume_ratio", 1.0))
        dist_bpct = float(intraday_features.get("dist_from_level_bpct", 5.0))
        acceptance_status = intraday_features.get("acceptance_status", "fair")

        # Weights from config
        weights = self._get("ranking", "weights")

        # Acceptance is key for level plays
        acc_cfg = weights["acceptance"]
        if acceptance_status == "excellent":
            s_acc = acc_cfg["bonus"] * acc_cfg["excellent_mult"]
        elif acceptance_status == "good":
            s_acc = acc_cfg["bonus"] * acc_cfg["good_mult"]
        else:
            s_acc = acc_cfg["bonus"] * acc_cfg["fair_mult"]

        # Distance scoring from config (closer to level = better)
        dist_cfg = weights["distance"]
        adist = abs(dist_bpct)
        if adist <= dist_cfg["near_bpct"]:
            s_dist = dist_cfg["near_score"]
        elif adist <= dist_cfg["ok_bpct"]:
            s_dist = dist_cfg["ok_score"]
        else:
            s_dist = dist_cfg["far_score"]

        # Volume (secondary) from config
        vol_cfg = weights["volume"]
        s_vol = min(vol_ratio / vol_cfg["divisor"], vol_cfg["cap"])

        base_score = s_acc + s_dist + s_vol

        # Regime multiplier from config - level plays excel in chop
        regime_mults = self._get("ranking", "regime_multipliers")
        regime_mult = regime_mults.get(regime, 1.0)

        # HTF (15m) multiplier - LEVEL plays often work AGAINST HTF trend
        # A long bounce at support is better if HTF is trending down (mean reversion)
        htf_mult = 1.0
        bias = intraday_features.get("bias", "long")
        if htf_context:
            htf_trend = htf_context.get("htf_trend", "neutral")

            # For level plays, OPPOSING trend is a bonus (bounce into the trend)
            htf_aligned = (htf_trend == "up" and bias == "long") or (htf_trend == "down" and bias == "short")
            htf_opposing = (htf_trend == "down" and bias == "long") or (htf_trend == "up" and bias == "short")

            if htf_opposing:
                htf_mult = 1.15  # +15% for counter-trend bounce (bounce into larger trend)
            elif htf_aligned:
                htf_mult = 0.95  # -5% for aligned (less impactful, not blocking)
            # Neutral HTF = no adjustment

        final_score = base_score * regime_mult * htf_mult

        return RankingResult(
            score=final_score,
            components={"acceptance": s_acc, "distance": s_dist, "volume": s_vol},
            multipliers={"regime": regime_mult, "htf": htf_mult}
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
