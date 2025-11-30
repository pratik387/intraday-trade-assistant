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

        - ADX must show trending market
        - EMA stack should be aligned
        - Not in choppy conditions
        """
        # levels reserved for future level-based screening
        _ = levels

        logger.debug(f"[MOMENTUM] Screening {symbol} at {now}")

        reasons = []
        passed = True

        # ADX check from config - need trending market
        adx = float(df5m["adx"].iloc[-1]) if "adx" in df5m.columns else 0.0
        min_adx = self._get("screening", "adx", "min_value")

        if adx < min_adx:
            reasons.append(f"adx_too_low:{adx:.1f}<{min_adx}")
            passed = False
        else:
            reasons.append(f"adx_ok:{adx:.1f}>={min_adx}")

        # EMA alignment check from config
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
            if ema_required:
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
        """
        # df5m reserved for future bar-level analysis
        _ = df5m
        # df1m reserved for future 1-minute analysis
        _ = df1m
        # strength reserved for future strength-based gating
        _ = strength

        logger.debug(f"[MOMENTUM] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, vol={vol_mult:.2f}")

        reasons = []
        passed = True
        size_mult = 1.0
        min_hold = 0

        # Regime check from config - momentum NEEDS trend
        regime_cfg = self._get("gates", "regime_rules")

        if regime in regime_cfg:
            rule = regime_cfg[regime]
            if not rule.get("allowed", True):
                reasons.append(f"regime_block:{regime}_{rule.get('reason', 'not_allowed')}")
                passed = False
            else:
                size_mult *= rule.get("size_mult", 1.0)
                reasons.append(f"regime_ok:{regime}")

                # Check trend alignment for trending regimes
                if regime in ("trend_up", "trend_down"):
                    is_long = "_long" in setup_type
                    if (regime == "trend_up" and is_long) or (regime == "trend_down" and not is_long):
                        reasons.append("trend_aligned")
                        size_mult *= rule.get("aligned_bonus", 1.0)
                    else:
                        reasons.append("counter_trend_caution")
                        # Use trend_alignment config for counter-trend penalty
                        align_cfg = self._get("gates", "trend_alignment")
                        size_mult *= align_cfg["counter_trend_size_mult"]

        # ADX gate from config - strict for momentum
        adx_cfg = self._get("gates", "adx")
        if adx < adx_cfg["min_value"]:
            if adx_cfg.get("block_below", False):
                reasons.append(f"adx_weak:{adx:.1f}<{adx_cfg['min_value']}")
                passed = False
        elif adx >= adx_cfg["strong_threshold"]:
            reasons.append(f"adx_strong:{adx:.1f}")
            size_mult *= adx_cfg["strong_bonus_mult"]

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
        Momentum-specific ranking.

        From ranker.py:
        - ADX is PRIMARY factor
        - ADX slope: trend strengthening (from ranker.py lines 99-100)
        - RSI slope: momentum acceleration (from ranker.py lines 98)
        - Trend alignment bonus
        - Volume confirmation

        HTF Logic for MOMENTUM:
        - Momentum REQUIRES HTF (15m) alignment
        - +25% bonus if HTF trend aligned (riding the wave)
        - -30% penalty if HTF trend opposing (fighting momentum)
        - +15% bonus if HTF momentum is strong (normalized momentum > 0.5)
        """
        logger.debug(f"[MOMENTUM] Calculating rank score for {symbol} in {regime}")

        adx = float(intraday_features.get("adx") or 0.0)
        adx_slope = float(intraday_features.get("adx_slope") or 0.0)
        rsi_slope = float(intraday_features.get("rsi_slope") or 0.0)
        vol_ratio = float(intraday_features.get("volume_ratio") or 1.0)
        above_vwap = bool(intraday_features.get("above_vwap", True))
        bias = intraday_features.get("bias", "long")

        # Weights from config
        weights = self._get("ranking", "weights")

        # ADX is king for momentum from config
        adx_cfg = weights["adx"]
        s_adx = max((adx - adx_cfg["mid"]) / adx_cfg["divisor"], adx_cfg["floor"])
        s_adx = min(s_adx, adx_cfg["cap"])

        # ADX slope score (from ranker.py lines 99-100)
        # Positive ADX slope = trend strengthening = good for momentum
        adx_slope_cfg = weights.get("adx_slope", {})
        if adx_slope_cfg:
            slope_cap = adx_slope_cfg.get("cap", 0.5)
            slope_mult = adx_slope_cfg.get("multiplier", 0.1)
            if adx_slope > 0:
                s_adx_slope = min(adx_slope * slope_mult, slope_cap)
            else:
                s_adx_slope = 0.0  # Weakening trend not penalized, just not rewarded
        else:
            s_adx_slope = 0.0

        # RSI slope score (from ranker.py lines 98)
        # For momentum: RSI slope aligned with bias = good (accelerating)
        rsi_slope_cfg = weights.get("rsi_slope", {})
        if rsi_slope_cfg:
            slope_cap = rsi_slope_cfg.get("cap", 0.3)
            slope_mult = rsi_slope_cfg.get("multiplier", 0.1)
            if bias == "long" and rsi_slope > 0:
                # Long momentum with RSI accelerating up = good
                s_rsi_slope = min(rsi_slope * slope_mult, slope_cap)
            elif bias == "short" and rsi_slope < 0:
                # Short momentum with RSI accelerating down = good
                s_rsi_slope = min(abs(rsi_slope) * slope_mult, slope_cap)
            else:
                s_rsi_slope = 0.0
        else:
            s_rsi_slope = 0.0

        # Volume (secondary) from config
        vol_cfg = weights["volume"]
        s_vol = min((vol_ratio - 1.0) * vol_cfg["above_threshold_mult"], vol_cfg["cap"]) if vol_ratio > 1.0 else 0.0

        # VWAP alignment with trend from config
        vwap_cfg = weights["vwap"]
        if bias == "long":
            s_vwap = vwap_cfg["aligned_bonus"] if above_vwap else vwap_cfg["misaligned_penalty"]
        else:
            s_vwap = vwap_cfg["aligned_bonus"] if not above_vwap else vwap_cfg["misaligned_penalty"]

        base_score = s_adx + s_adx_slope + s_rsi_slope + s_vol + s_vwap

        # Regime multiplier from config - momentum needs trend
        # Use strategy-specific multipliers from baseline ranker.py if available
        setup_type = intraday_features.get("setup_type", "")
        regime_mult = self._get_strategy_regime_mult(setup_type, regime)

        # Daily trend alignment from config (strongest multiplier for momentum)
        daily_mults = self._get("ranking", "daily_trend_multipliers")
        daily_mult = daily_mults["neutral"]
        if daily_trend:
            if (daily_trend == "up" and bias == "long") or (daily_trend == "down" and bias == "short"):
                daily_mult = daily_mults["aligned"]
            elif (daily_trend == "up" and bias == "short") or (daily_trend == "down" and bias == "long"):
                daily_mult = daily_mults["counter"]

        # HTF (15m) multiplier - MOMENTUM requires HTF alignment
        htf_mult = 1.0
        if htf_context:
            htf_trend = htf_context.get("htf_trend", "neutral")
            htf_momentum = htf_context.get("htf_momentum", 0.0)

            # Check alignment: long wants up, short wants down
            htf_aligned = (htf_trend == "up" and bias == "long") or (htf_trend == "down" and bias == "short")
            htf_opposing = (htf_trend == "down" and bias == "long") or (htf_trend == "up" and bias == "short")

            if htf_aligned:
                htf_mult = 1.25  # +25% for aligned HTF trend (riding the wave)

                # Extra bonus for strong HTF momentum
                if abs(htf_momentum) > 0.5:
                    htf_mult *= 1.15  # +15% for strong momentum
            elif htf_opposing:
                htf_mult = 0.70  # -30% penalty for fighting HTF momentum

        final_score = base_score * regime_mult * daily_mult * htf_mult

        return RankingResult(
            score=final_score,
            components={
                "adx": s_adx,
                "adx_slope": s_adx_slope,
                "rsi_slope": s_rsi_slope,
                "volume": s_vol,
                "vwap": s_vwap
            },
            multipliers={"regime": regime_mult, "daily": daily_mult, "htf": htf_mult}
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
            reference = entry_cfg.get("reference", "ema20")
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
        measured_move: float
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

        logger.debug(f"[MOMENTUM] Calculating targets for {symbol} entry={entry_ref_price:.2f}, sl={hard_sl:.2f}, mm={measured_move:.2f}")

        risk_per_share = abs(entry_ref_price - hard_sl)

        # R:R ratios from config
        targets_cfg = self._get("targets")
        rr_ratios = targets_cfg["rr_ratios"]
        t1_rr = rr_ratios["t1"]
        t2_rr = rr_ratios["t2"]
        t3_rr = rr_ratios["t3"]

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
