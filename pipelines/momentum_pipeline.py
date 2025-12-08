# pipelines/momentum_pipeline.py
"""
Momentum Pipeline - Specialized pipeline for MOMENTUM category setups.

MOMENTUM setups are trend continuation plays:
- trend_continuation
- trend_pullback
- trend_reversal
- momentum_trend

Quality Metric: ADX * EMA_alignment
- ADX measures trend STRENGTH (not direction)
- EMA stack alignment confirms trend structure
- Go WITH the trend, not against it

Key Filters:
- ADX >= 25 for strong trend
- EMA stack must be aligned (price > EMA20 > EMA50 for long)
- Pullback should be shallow (not breaking structure)
- Volume should confirm trend

Regime Rules:
- ONLY works in trend_up/trend_down
- Blocked in chop (no trend to follow)
- Requires patience - wait for pullback to EMA
"""

from typing import Dict, List, Optional, Any, Tuple
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


class MomentumPipeline(BasePipeline):
    """Pipeline for MOMENTUM category setups."""

    def get_category_name(self) -> str:
        return "MOMENTUM"

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
        Momentum-specific screening filters.

        - ADX must show trending market (except FHM which uses RVOL)
        - EMA stack should be aligned (relaxed for FHM)
        - Not in choppy conditions
        """
        # levels reserved for future level-based screening
        _ = levels

        logger.debug(f"[MOMENTUM] Screening {symbol} at {now}")

        reasons = []
        passed = True

        # Check if this is an FHM (First Hour Momentum) setup
        setup_type = features.get("setup_type", "")
        is_fhm = "first_hour_momentum" in setup_type

        # ADX check from config - need trending market (bypassed for FHM)
        adx = float(df5m["adx"].iloc[-1]) if "adx" in df5m.columns else 0.0
        min_adx = self._get("screening", "adx", "min_value")

        if is_fhm:
            # FHM uses RVOL-based momentum instead of ADX
            # The FHM detector already validated RVOL >= 2.0x
            volume_ratio = features.get("volume_ratio", 1.0)
            if volume_ratio >= 1.5:
                reasons.append(f"fhm_volume_ok:{volume_ratio:.2f}x")
            else:
                reasons.append(f"fhm_volume_low:{volume_ratio:.2f}x")
                # Don't fail on volume - FHM detector already validated RVOL
            reasons.append(f"fhm_adx_bypass:{adx:.1f}")
        else:
            if adx < min_adx:
                reasons.append(f"adx_too_low:{adx:.1f}<{min_adx}")
                passed = False
            else:
                reasons.append(f"adx_ok:{adx:.1f}>={min_adx}")

        # EMA alignment check from config (relaxed for FHM)
        ema_required = self._get("screening", "ema_alignment", "required")
        current_close = float(df5m["close"].iloc[-1])
        ema20 = float(df5m["ema20"].iloc[-1]) if "ema20" in df5m.columns else current_close
        ema50 = float(df5m["ema50"].iloc[-1]) if "ema50" in df5m.columns else current_close

        # Determine trend direction from EMA stack
        if ema20 > ema50:
            trend_dir = "up"
            if current_close > ema20:
                reasons.append("ema_stack_bullish:price>ema20>ema50")
            else:
                reasons.append("pullback_to_ema20")
        elif ema20 < ema50:
            trend_dir = "down"
            if current_close < ema20:
                reasons.append("ema_stack_bearish:price<ema20<ema50")
            else:
                reasons.append("pullback_to_ema20")
        else:
            trend_dir = "flat"
            reasons.append("ema_stack_flat")
            # FHM doesn't require EMA alignment - early movers may not have formed stack yet
            if ema_required and not is_fhm:
                passed = False  # No clear trend

        return ScreeningResult(
            passed=passed,
            reasons=reasons,
            features={**features, "trend_direction": trend_dir}
        )

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
        Momentum quality: ADX_score * EMA_alignment

        From planner_internal.py lines 1224-1241:
        - ADX measures trend strength (>25 = strong)
        - EMA stack alignment (price > ema20 > ema50 for longs)
        - Combined score shows trend quality
        """
        # levels reserved for future level-based quality metrics
        _ = levels
        # atr reserved for future ATR-normalized metrics
        _ = atr

        logger.debug(f"[MOMENTUM] Calculating quality for {symbol} bias={bias}")

        current_close = float(df5m["close"].iloc[-1])

        # ADX score from config (normalized to 0-2 range)
        adx = float(df5m["adx"].iloc[-1]) if "adx" in df5m.columns else 20.0
        adx_cfg = self._get("quality", "adx_score")
        adx_score = min(adx / adx_cfg["base"], adx_cfg["cap"])

        # EMA alignment from config
        ema20 = float(df5m["ema20"].iloc[-1]) if "ema20" in df5m.columns else current_close
        ema50 = float(df5m["ema50"].iloc[-1]) if "ema50" in df5m.columns else current_close

        ema_scores = self._get("quality", "ema_alignment_scores")

        if bias == "long":
            scores = ema_scores["long"]
            if current_close > ema20 > ema50:
                ema_aligned = scores["price_above_ema20_above_ema50"]
            elif current_close > ema20:
                ema_aligned = scores["price_above_ema20"]
            elif ema20 > ema50:
                ema_aligned = scores["ema20_above_ema50"]
            else:
                ema_aligned = scores["counter_trend"]
        else:
            scores = ema_scores["short"]
            if current_close < ema20 < ema50:
                ema_aligned = scores["price_below_ema20_below_ema50"]
            elif current_close < ema20:
                ema_aligned = scores["price_below_ema20"]
            elif ema20 < ema50:
                ema_aligned = scores["ema20_below_ema50"]
            else:
                ema_aligned = scores["counter_trend"]

        # Structural R:R for momentum plays from config
        quality_mult = self._get("quality", "quality_multiplier")
        structural_rr = (adx_score * ema_aligned) * quality_mult

        # Quality status from config
        quality_cfg = self._get("quality", "quality_thresholds")
        if adx >= quality_cfg["excellent"]["min_adx"] and ema_aligned >= quality_cfg["excellent"]["min_ema_aligned"]:
            quality_status = "excellent"
        elif adx >= quality_cfg["good"]["min_adx"] and ema_aligned >= quality_cfg["good"]["min_ema_aligned"]:
            quality_status = "good"
        elif adx >= quality_cfg["fair"]["min_adx"]:
            quality_status = "fair"
        else:
            quality_status = "poor"

        metrics = {
            "adx": round(adx, 1),
            "adx_score": round(adx_score, 2),
            "ema_aligned": round(ema_aligned, 2),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
        }

        reasons = [
            f"adx={adx:.1f}",
            f"adx_score={adx_score:.2f}",
            f"ema_align={ema_aligned:.2f}"
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
        Momentum-specific gate validations.

        Momentum plays REQUIRE trending regime:
        - Block in chop (no trend to follow)
        - Boost in trend_up/trend_down

        EXCEPTION: FHM (First Hour Momentum) bypasses regime/ADX gates:
        - FHM uses RVOL-based momentum (already validated in detector)
        - FHM happens early session before trends establish
        - FHM only needs volume gate
        """
        # df1m reserved for future 1-minute analysis
        _ = df1m
        # strength reserved for future strength-based gating
        _ = strength

        # Check if this is an FHM setup
        is_fhm = "first_hour_momentum" in setup_type

        logger.debug(f"[MOMENTUM] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, vol={vol_mult:.2f}, is_fhm={is_fhm}")

        reasons = []
        passed = True
        size_mult = 1.0
        min_hold = 0

        # ========== CROSS-RUN VALIDATED FILTERS (Dec 2024) ==========
        # These filters were validated across both backtest_20251204 and backtest_20251205
        validated_filters = self._get("gates", "validated_filters") or {}

        # Get volume from 5m bar for volume filters
        bar5_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0
        is_long = "_long" in setup_type

        # FHM VOLUME GATE: FHM uses RVOL-based detection, but still needs reasonable volume
        # The FHM detector already validated RVOL >= 2.0x, so we use relaxed volume gate (50k)
        if is_fhm:
            fhm_min_volume = 50000  # Relaxed for early session plays
            if bar5_volume < fhm_min_volume:
                reasons.append(f"fhm_vol_blocked:{bar5_volume/1000:.0f}k<{fhm_min_volume/1000:.0f}k")
                logger.debug(f"[MOMENTUM] {symbol} FHM BLOCKED: Vol {bar5_volume/1000:.0f}k < {fhm_min_volume/1000:.0f}k")
                return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
            else:
                reasons.append(f"fhm_vol_ok:{bar5_volume/1000:.0f}k")
                logger.debug(f"[MOMENTUM] {symbol} FHM volume OK: {bar5_volume/1000:.0f}k")

        # 1. MOMENTUM_BREAKOUT_LONG: VOLUME < 100K FILTER
        # Evidence: R1: +848 Rs (14 trades) | R2: +998 Rs (13 trades) - Low volume lacks conviction
        elif setup_type == "momentum_breakout_long":
            filter_cfg = validated_filters.get("momentum_breakout_long_volume")
            min_volume = filter_cfg.get("min_volume")
            if bar5_volume < min_volume:
                reasons.append(f"momentum_breakout_long_blocked:vol{bar5_volume/1000:.0f}k<{min_volume/1000:.0f}k")
                logger.debug(f"[MOMENTUM] {symbol} momentum_breakout_long BLOCKED: Vol {bar5_volume/1000:.0f}k < {min_volume/1000:.0f}k")
                return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
            else:
                reasons.append(f"momentum_breakout_long_vol_ok:{bar5_volume/1000:.0f}k")

        # 2. LONG TRADE VOLUME FILTER (MIN 150K) - for OTHER long setups
        # Evidence: Winners have 2x volume (213k vs 107k)
        # Use elif to skip setups that already have specific volume filters above
        elif is_long:
            filter_cfg = validated_filters.get("long_trade_volume")
            min_volume = filter_cfg.get("min_volume")
            if bar5_volume < min_volume:
                reasons.append(f"long_vol_blocked:{bar5_volume/1000:.0f}k<{min_volume/1000:.0f}k")
                logger.debug(f"[MOMENTUM] {symbol} {setup_type} BLOCKED: Long vol {bar5_volume/1000:.0f}k < {min_volume/1000:.0f}k")
                return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=min_hold)
            else:
                reasons.append(f"long_vol_ok:{bar5_volume/1000:.0f}k")

        # Regime check from config - momentum NEEDS trend - HARD GATES only
        # FHM BYPASS: FHM happens early session before trends establish - allow in any regime
        regime_cfg = self._get("gates", "regime_rules")

        if is_fhm:
            # FHM bypasses regime gate - early movers often start before trend establishes
            reasons.append(f"fhm_regime_bypass:{regime}")
            logger.debug(f"[MOMENTUM] {symbol} FHM bypasses regime gate: {regime}")
        elif regime in regime_cfg:
            rule = regime_cfg[regime]
            if not rule["allowed"]:
                reasons.append(f"regime_blocked:{regime}")
                passed = False
            else:
                reasons.append(f"regime_ok:{regime}")

                # Check trend alignment for trending regimes - HARD GATE for counter-trend
                if regime in ("trend_up", "trend_down"):
                    is_long_trade = "_long" in setup_type
                    if (regime == "trend_up" and is_long_trade) or (regime == "trend_down" and not is_long_trade):
                        reasons.append("trend_aligned")
                    else:
                        # Counter-trend momentum is blocked (don't fight the trend)
                        reasons.append("counter_trend_blocked")
                        passed = False

        # ADX gate from config - strict for momentum - HARD GATE
        # FHM BYPASS: FHM uses RVOL-based momentum, not ADX
        # Note: ADX is also checked in screening, this is additional validation
        adx_cfg = self._get("gates", "adx")
        if is_fhm:
            # FHM bypasses ADX gate - uses RVOL instead
            reasons.append(f"fhm_adx_bypass:{adx:.1f}")
            logger.debug(f"[MOMENTUM] {symbol} FHM bypasses ADX gate: {adx:.1f}")
        elif adx < adx_cfg["min_value"]:
            reasons.append(f"adx_weak:{adx:.1f}<{adx_cfg['min_value']}")
            passed = False
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
        Momentum-specific ranking - ALL 9 COMPONENTS FROM OLD ranker.py _intraday_strength().

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

        HTF Logic for MOMENTUM:
        - Momentum REQUIRES HTF (15m) alignment
        - +25% bonus if HTF trend aligned (riding the wave)
        - -30% penalty if HTF trend opposing (fighting momentum)
        - +15% bonus if HTF momentum is strong (normalized momentum > 0.5)
        """
        logger.debug(f"[MOMENTUM] Calculating rank score for {symbol} in {regime}")

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

        # Regime multiplier from config - momentum needs trend
        setup_type = intraday_features["setup_type"]
        regime_mult = self._get_strategy_regime_mult(setup_type, regime)

        # NOTE: Daily trend and HTF multipliers are applied ONLY in apply_universal_ranking_adjustments()
        # to match OLD ranker.py which applies them once in rank_candidates().
        # DO NOT apply them here - that would cause double-application!
        _ = daily_trend  # Unused here - applied in universal adjustments
        _ = htf_context  # Unused here - applied in universal adjustments

        final_score = base_score * regime_mult

        logger.debug(f"[MOMENTUM] {symbol} score={final_score:.3f} (vol={s_vol:.2f}, rsi={s_rsi:.2f}, rsis={s_rsis:.2f}, adx={s_adx:.2f}, adxs={s_adxs:.2f}, vwap={s_vwap:.2f}, dist={s_dist:.2f}, sq={s_sq:.2f}, acc={s_acc:.2f}) * regime={regime_mult:.2f}")

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
        Momentum entry with setup-type-specific logic.

        MOMENTUM category setups:
        - trend_continuation: Enter at current price (riding momentum)
        - trend_pullback: Enter at EMA20 pullback
        - trend_reversal: Enter at structure (ORL/ORH)
        - momentum_trend: Enter at current with momentum

        Entry principles:
        - Continuation: Enter at current price with momentum
        - Pullback: Enter at EMA20 support/resistance
        - Reversal: Enter at structure level
        """
        logger.debug(f"[MOMENTUM] Calculating entry for {symbol} {setup_type} bias={bias}")

        current_close = float(df5m["close"].iloc[-1])
        ema20 = float(df5m["ema20"].iloc[-1]) if "ema20" in df5m.columns else current_close
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)

        entry_cfg = self._get("entry")
        triggers = entry_cfg["triggers"]
        setup_lower = setup_type.lower()

        # Setup-type-specific entry logic for MOMENTUM
        if "continuation" in setup_lower:
            # Trend continuation - enter at current price (riding momentum)
            entry_ref = current_close
            entry_trigger = triggers["long"] if bias == "long" else triggers["short"]
        elif "pullback" in setup_lower:
            # Trend pullback - enter at EMA20
            entry_ref = ema20
            entry_trigger = "ema_bounce"
        elif "reversal" in setup_lower:
            # Trend reversal - enter at structure
            if bias == "long":
                entry_ref = orl  # Reversal up from support
            else:
                entry_ref = orh  # Reversal down from resistance
            entry_trigger = "trend_reversal"
        elif "squeeze" in setup_lower:
            # Volatility squeeze - enter at structure
            entry_ref = orl if bias == "long" else orh
            entry_trigger = "squeeze_release"
        else:
            # Default momentum - use config reference (ema20 or current)
            reference = entry_cfg["reference"]
            if reference == "ema20":
                entry_ref = ema20
            else:
                entry_ref = current_close
            entry_trigger = triggers["long"] if bias == "long" else triggers["short"]

        # Entry zone from config
        zone_mult = entry_cfg["zone_mult_atr"]
        zone_width = atr * zone_mult
        entry_zone = (entry_ref - zone_width, entry_ref + zone_width)

        # Entry mode from config
        entry_mode = entry_cfg["mode"]

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
        Momentum targets: Trend-following aggressive targets.

        Trend continuation can run far:
        - T1: 1.5R (lock in profit)
        - T2: 3.0R (trend runner)
        - T3: 5.0R (home run - rare)
        """
        # levels reserved for future level-based targets
        _ = levels
        # atr reserved for future ATR-based target adjustments
        _ = atr

        logger.debug(f"[MOMENTUM] Calculating targets for {symbol} {setup_type} entry={entry_ref_price:.2f}, sl={hard_sl:.2f}, mm={measured_move:.2f}")

        risk_per_share = abs(entry_ref_price - hard_sl)

        # R:R ratios from config
        targets_cfg = self._get("targets")
        rr_ratios = targets_cfg["rr_ratios"]

        # Extract base strategy name (e.g., "first_hour_momentum" from "first_hour_momentum_long")
        base_strategy = setup_type.replace("_long", "").replace("_short", "") if setup_type else ""

        # Look up strategy-specific targets, fallback to default
        if base_strategy in rr_ratios:
            strategy_ratios = rr_ratios[base_strategy]
            # Check for bias-specific within strategy
            if bias in strategy_ratios and isinstance(strategy_ratios[bias], dict):
                t1_rr = strategy_ratios[bias].get("t1", strategy_ratios.get("t1"))
                t2_rr = strategy_ratios[bias].get("t2", strategy_ratios.get("t2"))
                t3_rr = strategy_ratios[bias].get("t3", strategy_ratios.get("t3"))
                logger.debug(f"[MOMENTUM] Using {base_strategy}/{bias} targets: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")
            else:
                t1_rr = strategy_ratios.get("t1")
                t2_rr = strategy_ratios.get("t2")
                t3_rr = strategy_ratios.get("t3")
                logger.debug(f"[MOMENTUM] Using {base_strategy} targets: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")
        else:
            # Fallback to default
            default_ratios = rr_ratios.get("default", {})
            if bias in default_ratios and isinstance(default_ratios[bias], dict):
                t1_rr = default_ratios[bias].get("t1", default_ratios.get("t1"))
                t2_rr = default_ratios[bias].get("t2", default_ratios.get("t2"))
                t3_rr = default_ratios[bias].get("t3", default_ratios.get("t3"))
                logger.debug(f"[MOMENTUM] Using default/{bias} targets: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")
            else:
                t1_rr = default_ratios.get("t1")
                t2_rr = default_ratios.get("t2")
                t3_rr = default_ratios.get("t3")
                logger.debug(f"[MOMENTUM] Using default targets: T1={t1_rr}R, T2={t2_rr}R, T3={t3_rr}R")

        # Targets can be extended in trending markets - caps from config
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
        MOMENTUM RSI penalty: Penalize weak momentum, reward strong momentum.

        Pro Trader Framework (Minervini, Weinstein):
        - High RSI = momentum confirmation = NO PENALTY (trend continuation needs momentum)
        - Low RSI = weak momentum = penalty (trend may be exhausting)
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
