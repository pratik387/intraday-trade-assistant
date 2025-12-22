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
    get_cap_segment,
    safe_level_get,
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
        orh = safe_level_get(levels, "ORH", current_close)
        orl = safe_level_get(levels, "ORL", current_close)
        pdh = safe_level_get(levels, "PDH", orh)
        pdl = safe_level_get(levels, "PDL", orl)

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

        # PRO TRADER APPROACH: Use the actual level detected by the structure detector
        # The detector finds nearest_support/nearest_resistance which may include:
        # - PDH/PDL (previous day)
        # - ORH/ORL (opening range)
        # - Computed range support/resistance (from intraday price action)
        #
        # Priority: detected_level > PDH/PDL > ORH/ORL
        detected_level = levels.get("detected_level")
        pdh = levels.get("PDH")
        pdl = levels.get("PDL")
        orh = levels.get("ORH")
        orl = levels.get("ORL")

        # Use detected_level if available (this is what the structure detector actually found)
        # Fall back to PDH/PDL (always available from daily cache)
        # Last resort: ORH/ORL (may be NaN on late starts)
        if detected_level is not None and not pd.isna(detected_level):
            target_level = detected_level
            logger.debug(f"[LEVEL] {symbol} using detected_level={detected_level:.2f}")
        elif bias == "long":
            # For long bounce, use support level (PDL preferred, then ORL)
            target_level = pdl if not pd.isna(pdl) else orl
            logger.debug(f"[LEVEL] {symbol} fallback to PDL/ORL={target_level}")
        else:
            # For short bounce, use resistance level (PDH preferred, then ORH)
            target_level = pdh if not pd.isna(pdh) else orh
            logger.debug(f"[LEVEL] {symbol} fallback to PDH/ORH={target_level}")

        # Reject if no valid target level - can't trade level setups without a level
        if target_level is None or pd.isna(target_level):
            logger.warning(f"[LEVEL] {symbol} rejected: no valid target level found")
            return None

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
        vol_mult: float,
        regime_diagnostics: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        Level-specific gate validations - HARD GATES ONLY, NO PENALTIES.

        Uses UNIFIED FILTER SYSTEM from base_pipeline:
        - apply_global_filters(): Global filters (short_min_adx, long_min_volume)
        - apply_setup_filters(): Per-setup filters with enabled flags

        Level plays work in all regimes but have different requirements:
        - In trend: counter-trend requires stronger confirmation (HARD BLOCK if weak)
        - In chop: level respect is sufficient
        - ADX: high ADX is warning but level plays CAN work (no penalty)
        - Volume: nice-to-have, not required (no penalty)
        """
        # df1m reserved for future bar-level analysis
        _ = df1m

        logger.debug(f"[LEVEL] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, vol={vol_mult:.2f}")

        # VWAP reclaim/lose re-validation: Ensure condition still holds at decision time
        setup_lower = setup_type.lower()
        if "vwap" in setup_lower and "vwap" in df5m.columns:
            current_price = float(df5m["close"].iloc[-1])
            current_vwap = float(df5m["vwap"].iloc[-1])

            if "reclaim" in setup_lower:
                # vwap_reclaim_long: Price MUST still be above VWAP
                if current_price <= current_vwap:
                    logger.debug(f"[LEVEL] {symbol} vwap_reclaim BLOCKED: price {current_price:.2f} <= VWAP {current_vwap:.2f}")
                    return GateResult(
                        passed=False,
                        reasons=[f"vwap_reclaim_invalid:price_{current_price:.2f}<=vwap_{current_vwap:.2f}"],
                        size_mult=1.0,
                        min_hold_bars=0
                    )
            elif "lose" in setup_lower:
                # vwap_lose_short: Price MUST still be below VWAP
                if current_price >= current_vwap:
                    logger.debug(f"[LEVEL] {symbol} vwap_lose BLOCKED: price {current_price:.2f} >= VWAP {current_vwap:.2f}")
                    return GateResult(
                        passed=False,
                        reasons=[f"vwap_lose_invalid:price_{current_price:.2f}>=vwap_{current_vwap:.2f}"],
                        size_mult=1.0,
                        min_hold_bars=0
                    )

            # ========== VWAP VOLUME FILTER (DATA-DRIVEN Dec 2024) ==========
            # From spike test: VWAP vol<50k: 18 trades (6W, 12L), blocking = +2,675 Rs
            # Low volume VWAP trades have poor execution and high slippage
            bar5_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0
            vwap_vol_cfg = self._get("gates", "vwap_volume")
            min_volume = vwap_vol_cfg.get("min_volume")

            if bar5_volume < min_volume:
                logger.debug(f"[LEVEL] {symbol} vwap BLOCKED: vol {bar5_volume/1000:.0f}k < {min_volume/1000:.0f}k")
                return GateResult(
                    passed=False,
                    reasons=[f"vwap_vol_blocked:{bar5_volume/1000:.0f}k<{min_volume/1000:.0f}k"],
                    size_mult=1.0,
                    min_hold_bars=0
                )

        reasons = []
        passed = True
        size_mult = 1.0  # NO PENALTIES - hard gates only
        min_hold = 0

        # Get common filter inputs
        bar5_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0
        is_long = "_long" in setup_type
        bias = "long" if is_long else "short"
        current_hour = df5m.index[-1].hour if hasattr(df5m.index[-1], 'hour') else 0
        cap_segment = get_cap_segment(symbol)

        # Get RSI for filters
        rsi_val = None
        if len(df5m) >= 14 and "close" in df5m.columns:
            close = df5m["close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                rsi_val = float(rsi.iloc[-1])

        # ========== UNIFIED FILTER SYSTEM ==========
        # 1. GLOBAL FILTERS (short_min_adx, long_min_volume)
        global_passed, global_reasons = self.apply_global_filters(
            setup_type=setup_type,
            symbol=symbol,
            bias=bias,
            adx=adx,
            rsi=rsi_val,
            volume=bar5_volume
        )
        reasons.extend(global_reasons)
        if not global_passed:
            logger.debug(f"[LEVEL] {symbol} {setup_type} BLOCKED by global filters: {global_reasons}")
            return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)

        # 2. SETUP-SPECIFIC FILTERS (from gates.setup_filters)
        setup_passed, setup_reasons, modifiers = self.apply_setup_filters(
            setup_type=setup_type,
            symbol=symbol,
            regime=regime,
            adx=adx,
            rsi=rsi_val,
            volume=bar5_volume,
            current_hour=current_hour,
            cap_segment=cap_segment,
            structural_rr=strength  # strength parameter is structural_rr
        )
        reasons.extend(setup_reasons)
        if not setup_passed:
            logger.debug(f"[LEVEL] {symbol} {setup_type} BLOCKED by setup filters: {setup_reasons}")
            return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)

        # Apply target modifiers from setup filters
        # Note: These are returned to the caller via the plan system
        if modifiers.get("sl_mult") != 1.0:
            reasons.append(f"sl_mult:{modifiers['sl_mult']}")
        if modifiers.get("t1_mult") != 1.0:
            reasons.append(f"t1_mult:{modifiers['t1_mult']}")
        if modifiers.get("t2_mult") != 1.0:
            reasons.append(f"t2_mult:{modifiers['t2_mult']}")

        # 3. SPECIALIZED FILTERS (require complex logic not in unified system)
        # BB width filter for resistance_bounce_short
        if setup_type == "resistance_bounce_short":
            specialized_filters = self._get("gates", "specialized_filters") or {}
            bb_cfg = specialized_filters.get("resistance_bounce_short_bb_width", {})
            if bb_cfg.get("enabled", False):
                max_bb_width = bb_cfg.get("max_bb_width", 0.10)
                if len(df5m) >= 20 and "close" in df5m.columns:
                    close = df5m["close"]
                    sma20 = close.rolling(window=20, min_periods=20).mean()
                    std20 = close.rolling(window=20, min_periods=20).std()
                    if pd.notna(sma20.iloc[-1]) and pd.notna(std20.iloc[-1]) and sma20.iloc[-1] > 0:
                        bb_width = (2 * std20.iloc[-1]) / sma20.iloc[-1]
                        if bb_width > max_bb_width:
                            reasons.append(f"bb_width_blocked:{bb_width:.3f}>{max_bb_width}")
                            logger.debug(f"[LEVEL] {symbol} resistance_bounce_short BLOCKED: BB width {bb_width:.3f} > {max_bb_width}")
                            return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
                        reasons.append(f"bb_width_ok:{bb_width:.3f}")

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
        LEVEL-SPECIFIC RANKING (Dec 2024 Recalibration)

        Pro trader research shows LEVEL plays need INVERTED scoring vs breakouts:
        - ADX: LOW is good (ranging market, levels respected)
        - RSI: EXTREME is good (oversold for long bounce, overbought for short bounce)
        - VWAP: Counter-VWAP is acceptable (reversal setup)
        - Volume: Less important than structure
        7 components:
        1. s_vol  - Volume ratio (reduced weight)
        2. s_rsi  - RSI extreme scoring (inverted logic)
        3. s_adx  - ADX low scoring (inverted logic)
        4. s_vwap - VWAP relaxed (counter allowed)
        5. s_dist - Distance from level (high weight)
        6. s_sq   - Squeeze percentile
        7. s_acc  - Acceptance bonus (excellent/good quality)
        """
        logger.debug(f"[LEVEL] Calculating rank score for {symbol} in {regime}")

        # REQUIRED features
        vol_ratio = float(intraday_features["volume_ratio"])
        rsi = float(intraday_features["rsi"])
        adx = float(intraday_features["adx"])
        above_vwap = bool(intraday_features["above_vwap"])
        bias = intraday_features["bias"]

        # OPTIONAL features
        dist_from_level_bpct = float(intraday_features.get("dist_from_level_bpct") or 9.99)
        squeeze_pctile = intraday_features.get("squeeze_pctile")
        acceptance_status = intraday_features.get("acceptance_status")

        weights = self._get("ranking", "weights")
        score_scale = self._get("ranking", "score_scale")

        # 1. VOLUME SCORE (reduced weight for level plays)
        vol_cfg = weights["volume"]
        s_vol = min(vol_ratio / vol_cfg["divisor"], vol_cfg["cap"])

        # 2. RSI SCORE - INVERTED: Extreme RSI = GOOD for bounces
        rsi_cfg = weights["rsi"]
        if bias == "long":
            # Long bounce wants oversold (low RSI)
            if rsi <= rsi_cfg["long_very_oversold_threshold"]:
                s_rsi = rsi_cfg["extreme_bonus"]  # Very oversold = excellent
            elif rsi <= rsi_cfg["long_oversold_threshold"]:
                s_rsi = rsi_cfg["good_bonus"]  # Oversold = good
            elif rsi >= rsi_cfg["long_overbought_penalty_threshold"]:
                s_rsi = rsi_cfg["penalty"]  # Overbought = bad for long bounce
            else:
                s_rsi = 0.0  # Neutral
        else:  # short
            # Short bounce wants overbought (high RSI)
            if rsi >= rsi_cfg["short_very_overbought_threshold"]:
                s_rsi = rsi_cfg["extreme_bonus"]  # Very overbought = excellent
            elif rsi >= rsi_cfg["short_overbought_threshold"]:
                s_rsi = rsi_cfg["good_bonus"]  # Overbought = good
            elif rsi <= rsi_cfg["short_oversold_penalty_threshold"]:
                s_rsi = rsi_cfg["penalty"]  # Oversold = bad for short bounce
            else:
                s_rsi = 0.0  # Neutral

        # 3. ADX SCORE - INVERTED: Low ADX = GOOD for bounces
        adx_cfg = weights["adx"]
        if adx <= adx_cfg["ideal_max"]:
            s_adx = adx_cfg["ideal_bonus"]  # Ranging market = excellent
        elif adx <= adx_cfg["acceptable_max"]:
            s_adx = adx_cfg["acceptable_bonus"]  # Low trend = good
        elif adx >= adx_cfg["penalty_threshold"]:
            s_adx = adx_cfg["penalty"]  # Strong trend = bad for bounce
        else:
            s_adx = 0.0  # Neutral

        # 4. VWAP SCORE - RELAXED: Counter-VWAP acceptable for reversals
        vwap_cfg = weights["vwap"]
        if bias == "long":
            s_vwap = vwap_cfg["aligned_bonus"] if above_vwap else vwap_cfg["counter_bonus"]
        else:
            s_vwap = vwap_cfg["aligned_bonus"] if not above_vwap else vwap_cfg["counter_bonus"]

        # 5. DISTANCE FROM LEVEL SCORE (high weight - critical for level plays)
        dist_cfg = weights["distance"]
        adist = abs(dist_from_level_bpct)
        if adist <= dist_cfg["near_bpct"]:
            s_dist = dist_cfg["near_score"]
        elif adist <= dist_cfg["ok_bpct"]:
            s_dist = dist_cfg["ok_score"]
        else:
            s_dist = dist_cfg["far_score"]

        # 6. SQUEEZE PERCENTILE SCORE
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

        # 7. ACCEPTANCE SCORE - Quality bonus for level trades
        acc_cfg = weights["acceptance"]
        if acc_cfg["enabled"]:
            if acceptance_status == "excellent":
                s_acc = acc_cfg["excellent_bonus"]
            elif acceptance_status == "good":
                s_acc = acc_cfg["good_bonus"]
            else:
                s_acc = 0.0
        else:
            s_acc = 0.0

        # WEIGHTED SUM (not simple addition) scaled to usable range
        weighted_sum = (
            s_vol * vol_cfg["weight"] +
            s_rsi * rsi_cfg["weight"] +
            s_adx * adx_cfg["weight"] +
            s_vwap * vwap_cfg["weight"] +
            s_dist * dist_cfg["weight"] +
            s_sq * squeeze_cfg["weight"] +
            s_acc * acc_cfg["weight"]
        )
        base_score = weighted_sum * score_scale

        # Regime multiplier - level plays excel in chop/squeeze
        setup_type = intraday_features["setup_type"]
        regime_mult = self._get_strategy_regime_mult(setup_type, regime)

        # HTF context handled in universal adjustments
        _ = daily_trend
        _ = htf_context

        final_score = base_score * regime_mult

        logger.debug(f"[LEVEL] {symbol} score={final_score:.3f} (weighted_sum={weighted_sum:.3f}*scale={score_scale}) * regime={regime_mult:.2f}")

        return RankingResult(
            score=final_score,
            components={
                "volume": s_vol,
                "rsi": s_rsi,
                "adx": s_adx,
                "vwap": s_vwap,
                "distance": s_dist,
                "squeeze": s_sq,
                "acceptance": s_acc
            },
            multipliers={"regime": regime_mult, "score_scale": score_scale}
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

        # Handle NaN properly - levels.get() returns NaN if the key exists but value is NaN
        orh = levels.get("ORH")
        orl = levels.get("ORL")
        pdh = levels.get("PDH")
        pdl = levels.get("PDL")
        detected_level = levels.get("detected_level")

        # Replace NaN with fallbacks
        if orh is None or pd.isna(orh):
            orh = current_close
        if orl is None or pd.isna(orl):
            orl = current_close
        if pdh is None or pd.isna(pdh):
            pdh = orh
        if pdl is None or pd.isna(pdl):
            pdl = orl
        vwap = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns else current_close

        entry_cfg = self._get("entry")
        triggers = entry_cfg["triggers"]
        mode_cfg = entry_cfg["mode"]
        ict_cfg = entry_cfg["ict_zones"]
        setup_lower = setup_type.lower()

        # Setup-type-specific entry logic for LEVEL category
        if "vwap" in setup_lower:
            # VWAP plays - differentiate by setup semantics
            if "reclaim" in setup_lower:
                # vwap_reclaim_long: Price just crossed ABOVE VWAP
                # Entry should be AT or ABOVE VWAP (confirming the reclaim)
                entry_ref = vwap * 1.001  # Just above VWAP
            elif "lose" in setup_lower:
                # vwap_lose_short: Price just crossed BELOW VWAP
                # Entry should be AT or BELOW VWAP (confirming the lose)
                entry_ref = vwap * 0.999  # Just below VWAP
            else:
                # vwap_mean_reversion: Entry toward VWAP
                if bias == "long":
                    entry_ref = vwap * 0.999  # Below VWAP for long (buy dip)
                else:
                    entry_ref = vwap * 1.001  # Above VWAP for short (sell rip)
            entry_trigger = triggers["vwap"]
        elif "premium" in setup_lower or "discount" in setup_lower or "order_block" in setup_lower:
            # ICT zones - premium/discount
            if bias == "long":
                entry_ref = orl + (orh - orl) * ict_cfg["discount_zone_frac"]  # 30% into range
            else:
                entry_ref = orl + (orh - orl) * ict_cfg["premium_zone_frac"]  # 70% into range
            entry_trigger = triggers["premium_zone"]
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
            # Range play - use detected_level (actual computed range level) if available
            if detected_level is not None and not pd.isna(detected_level):
                entry_ref = detected_level
                logger.debug(f"[LEVEL] {symbol} range using detected_level={detected_level:.2f}")
            else:
                entry_ref = orl if bias == "long" else orh
            entry_trigger = triggers["default"]
        else:
            # Default LEVEL (support_bounce, resistance_bounce, etc.)
            # Use detected_level first (actual level found by structure detector)
            if detected_level is not None and not pd.isna(detected_level):
                entry_ref = detected_level
                logger.debug(f"[LEVEL] {symbol} {setup_lower} using detected_level={detected_level:.2f}")
            elif bias == "long":
                entry_ref = pdl if pdl > 0 else orl
            else:
                entry_ref = pdh if pdh > 0 else orh
            entry_trigger = triggers["default"]

        # Final safety check - REJECT if entry_ref is NaN (don't risk money on bad data)
        if pd.isna(entry_ref):
            logger.warning(f"[LEVEL] {symbol} REJECTED: entry_ref is NaN - no valid level data")
            return None

        # Entry zone for level plays from config
        # Use ATR-based zone, but ensure minimum width for low-ATR large cap stocks
        zone_mult = entry_cfg["zone_mult_atr"]
        zone_width = atr * zone_mult

        # Apply minimum zone width (as % of price) for large cap stocks with low ATR
        min_zone_pct = entry_cfg.get("min_zone_pct")
        if min_zone_pct > 0:
            min_zone_width = entry_ref * (min_zone_pct / 100.0)
            if zone_width < min_zone_width:
                logger.debug(f"[LEVEL] {symbol} zone widened from {zone_width:.3f} to {min_zone_width:.3f} (min_zone_pct={min_zone_pct}%)")
                zone_width = min_zone_width

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
        measured_move: float,
        setup_type: str = ""
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

        # Check for setup_type specific targets first (e.g., support_bounce_long)
        # Then fallback to bias-specific (long/short), then to defaults
        if setup_type in rr_ratios and isinstance(rr_ratios[setup_type], dict):
            t1_rr = rr_ratios[setup_type].get("t1", rr_ratios["t1"])
            t2_rr = rr_ratios[setup_type].get("t2", rr_ratios["t2"])
            t3_rr = rr_ratios[setup_type].get("t3", rr_ratios["t3"])
            logger.debug(f"[LEVEL] Using setup-specific targets for {setup_type}: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")
        elif bias in rr_ratios and isinstance(rr_ratios[bias], dict):
            t1_rr = rr_ratios[bias].get("t1", rr_ratios["t1"])
            t2_rr = rr_ratios[bias].get("t2", rr_ratios["t2"])
            t3_rr = rr_ratios[bias].get("t3", rr_ratios["t3"])
            logger.debug(f"[LEVEL] Using bias-specific targets for {bias}: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")
        else:
            t1_rr = rr_ratios["t1"]
            t2_rr = rr_ratios["t2"]
            t3_rr = rr_ratios["t3"]

        # DATA-DRIVEN TARGET MULTIPLIERS (Dec 2024)
        # Some setups benefit from wider targets (e.g., orb_pullback_short t1_mult=1.5, t2_mult=1.5)
        setup_filters = self._get("gates", "setup_filters") or {}
        setup_filter_cfg = setup_filters.get(setup_type.lower(), {})
        t1_mult = setup_filter_cfg.get("t1_mult", 1.0)
        t2_mult = setup_filter_cfg.get("t2_mult", 1.0)

        if t1_mult != 1.0 or t2_mult != 1.0:
            t1_rr = t1_rr * t1_mult
            t2_rr = t2_rr * t2_mult
            t3_rr = t3_rr * t2_mult  # T3 scales with T2 to maintain T1 < T2 < T3 ordering
            logger.debug(f"[LEVEL] {symbol} {setup_type} TARGET MULTIPLIERS: t1_mult={t1_mult}, t2_mult={t2_mult} -> T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")

        # Cap targets from config (level plays shouldn't expect huge moves)
        caps = targets_cfg["caps"]
        cap1 = min(measured_move * caps["t1"]["measured_move_frac"], entry_ref_price * caps["t1"]["max_pct"])
        cap2 = min(measured_move * caps["t2"]["measured_move_frac"], entry_ref_price * caps["t2"]["max_pct"])
        cap3 = min(measured_move * caps["t3"]["measured_move_frac"], entry_ref_price * caps["t3"]["max_pct"])

        # PRE-TRADE REJECTION: If T1 cap < 0.8R, reject the setup
        # Low-volatility instruments (ETFs, liquid funds) can't hit viable targets
        # Better to reject upfront than trade with broken T1 (0R) or force T1 at 1R and hit SL
        min_t1_threshold = risk_per_share * 0.8
        if cap1 < min_t1_threshold:
            logger.info(f"[LEVEL] {symbol} rejected: T1 cap ({cap1:.4f}) < 0.8R ({min_t1_threshold:.4f}) - low volatility")
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
