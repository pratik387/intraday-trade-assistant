# pipelines/reversion_pipeline.py
"""
Reversion Pipeline - Specialized pipeline for REVERSION category setups.

REVERSION setups are mean reversion plays:
- failure_fade (fade failed breakouts)
- volume_spike_reversal
- gap_fill
- vwap_mean_reversion
- liquidity_sweep (ICT)
- fair_value_gap (ICT)

Quality Metric: vwap_distance + exhaustion_score + volume_exhaustion
- Extension from VWAP measures how overextended price is
- Exhaustion signals (RSI extremes) confirm reversal potential
- Volume spike on exhaustion = institutional capitulation

Key Filters:
- Price must be extended from VWAP (>1.5% typically)
- RSI in extreme territory (<30 for longs, >70 for shorts)
- Volume spike on the exhaustion move
- Recent failed breakout (for failure_fade)

Regime Rules:
- Best in chop (mean reversion is the play)
- Good after failed breakouts in trend
- Requires patience - entries on exhaustion candles
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


class ReversionPipeline(BasePipeline):
    """Pipeline for REVERSION category setups."""

    def get_category_name(self) -> str:
        return "REVERSION"

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
        Reversion-specific screening filters.

        - Price must be extended from VWAP
        - Look for exhaustion signals
        - Volume spike preferred
        """
        reasons = []
        passed = True

        # levels reserved for future level-based screening
        _ = levels

        logger.debug(f"[REVERSION] Screening {symbol} at {now}")

        current_close = float(df5m["close"].iloc[-1])
        vwap = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns else current_close
        atr = self.calculate_atr(df5m)

        # Extension from VWAP check from config
        vwap_distance_pct = abs(current_close - vwap) / max(vwap, 1e-6) * 100

        min_extension = self._get("screening", "extension", "min_vwap_extension_pct")

        if vwap_distance_pct < min_extension:
            reasons.append(f"not_extended:{vwap_distance_pct:.2f}%<{min_extension}%")
            passed = False
        else:
            reasons.append(f"extended_from_vwap:{vwap_distance_pct:.2f}%")

        # RSI extremes check from config
        rsi = float(df5m["rsi"].iloc[-1]) if "rsi" in df5m.columns else 50.0
        rsi_cfg = self._get("screening", "rsi_extremes")

        # For reversion, we want RSI in extreme territory
        # Long reversion when RSI < oversold (oversold bounce)
        # Short reversion when RSI > overbought (overbought fade)
        bias_hint = "long" if current_close < vwap else "short"

        if bias_hint == "long" and rsi > rsi_cfg["oversold_threshold"]:
            reasons.append(f"rsi_not_oversold:{rsi:.1f}>{rsi_cfg['oversold_threshold']}")
            passed = False
        elif bias_hint == "short" and rsi < rsi_cfg["overbought_threshold"]:
            reasons.append(f"rsi_not_overbought:{rsi:.1f}<{rsi_cfg['overbought_threshold']}")
            passed = False
        else:
            reasons.append(f"rsi_extreme_ok:{rsi:.1f}")

        # Volume spike check from config (reversion after capitulation)
        vol_ratio = features.get("volume_ratio", 1.0)
        vol_threshold = self._get("screening", "volume_spike_threshold")
        if vol_ratio >= vol_threshold:
            reasons.append(f"volume_spike_ok:{vol_ratio:.2f}x")
        else:
            reasons.append(f"volume_normal:{vol_ratio:.2f}x")

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
        Reversion quality: extension_from_vwap + exhaustion_score + volume_exhaustion

        From planner_internal.py lines 1204-1222:
        - VWAP distance (ATR-normalized) measures overextension
        - RSI exhaustion signals confirm reversal potential
        - Volume spike = institutional capitulation
        """
        # levels reserved for future level-based quality metrics
        _ = levels

        logger.debug(f"[REVERSION] Calculating quality for {symbol} bias={bias}")

        current_close = float(df5m["close"].iloc[-1])
        vwap = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns else current_close

        # VWAP distance (ATR-normalized)
        vwap_distance = abs(current_close - vwap) / max(atr, 1e-6)
        extension_pct = abs(current_close - vwap) / max(vwap, 1e-6) * 100

        # Exhaustion indicators from config
        rsi = float(df5m["rsi"].iloc[-1]) if "rsi" in df5m.columns else 50.0
        rsi_cfg = self._get("quality", "rsi_exhaustion")

        if bias == "long":
            # For long reversion, RSI < threshold = exhaustion (oversold bounce)
            threshold = rsi_cfg["long"]["threshold"]
            divisor = rsi_cfg["long"]["divisor"]
            exhaustion_score = max(0, (threshold - rsi) / divisor)
        else:
            # For short reversion, RSI > threshold = exhaustion (overbought fade)
            threshold = rsi_cfg["short"]["threshold"]
            divisor = rsi_cfg["short"]["divisor"]
            exhaustion_score = max(0, (rsi - threshold) / divisor)

        # Volume exhaustion (spike on the exhaustion move) from config
        vol_ratio = self.get_volume_ratio(df5m)
        vol_cfg = self._get("quality", "volume_exhaustion")
        volume_exhaustion = 1.0 + (vol_ratio - 1.0) * vol_cfg["multiplier"] if vol_ratio > vol_cfg["threshold"] else 1.0

        # Structural R:R for reversion plays from config
        vwap_weight = self._get("quality", "vwap_distance_weight")
        exhaust_weight = self._get("quality", "exhaustion_weight")
        structural_rr = (vwap_distance * vwap_weight + exhaustion_score * exhaust_weight) * volume_exhaustion

        # Quality status from config
        quality_cfg = self._get("quality", "quality_thresholds")
        if exhaustion_score >= quality_cfg["excellent"]["min_exhaustion_score"] and vwap_distance >= quality_cfg["excellent"]["min_vwap_distance_atr"]:
            quality_status = "excellent"
        elif exhaustion_score >= quality_cfg["good"]["min_exhaustion_score"] or vwap_distance >= quality_cfg["good"]["min_vwap_distance_atr"]:
            quality_status = "good"
        elif vwap_distance >= quality_cfg["fair"]["min_vwap_distance_atr"]:
            quality_status = "fair"
        else:
            quality_status = "poor"

        metrics = {
            "vwap_distance_atr": round(vwap_distance, 2),
            "extension_pct": round(extension_pct, 2),
            "rsi": round(rsi, 1),
            "exhaustion_score": round(exhaustion_score, 2),
            "volume_ratio": round(vol_ratio, 2),
        }

        reasons = [
            f"vwap_dist={vwap_distance:.2f}ATR",
            f"ext={extension_pct:.2f}%",
            f"rsi={rsi:.1f}",
            f"exhaust={exhaustion_score:.2f}"
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
        Reversion-specific gate validations.

        Reversion plays work best after extremes:
        - Need exhaustion signals
        - Volume spike confirms capitulation
        - Work in all regimes but timing matters
        """
        # df1m reserved for future 1-minute analysis
        _ = df1m

        logger.debug(f"[REVERSION] Validating gates for {symbol} {setup_type}: regime={regime}, adx={adx:.1f}, strength={strength:.2f}, vol={vol_mult:.2f}")

        reasons = []
        passed = True
        size_mult = 1.0

        # Min hold bars from config - reversion plays need patience
        min_hold = self._get("gates", "min_hold_bars")

        # Regime multipliers from config - reversion works best in chop
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
                        reasons.append(f"trend_counter_weak:strength={strength:.2f}")
                        size_mult *= rule.get("counter_trend_size_mult", 0.7)
            else:
                reasons.append(f"regime_blocked:{regime}")
                passed = False

        # Failure fade specific check from config
        if "failure_fade" in setup_type:
            ff_cfg = self._get("gates", "failure_fade")
            # Need to see a recent failed breakout
            if df5m is not None and len(df5m) >= 5:
                # Check for rejection (wick)
                last_bar = df5m.iloc[-1]
                body = abs(last_bar["close"] - last_bar["open"])
                total_range = last_bar["high"] - last_bar["low"]

                if total_range > 0:
                    body_ratio = body / total_range
                    if body_ratio < ff_cfg["rejection_candle_body_ratio_max"]:
                        reasons.append("rejection_candle_ok")
                        size_mult *= ff_cfg["rejection_candle_bonus_mult"]
                    else:
                        reasons.append("no_rejection_candle")

        # Volume spike is good for reversion from config
        vol_cfg = self._get("gates", "volume_capitulation")
        if vol_mult >= vol_cfg["threshold"]:
            reasons.append(f"capitulation_volume:{vol_mult:.2f}x")
            size_mult *= vol_cfg["bonus_mult"]

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
        Reversion-specific ranking.

        From ranker.py:
        - Extension from VWAP is primary
        - Exhaustion signals boost score (RSI extremes)
        - RSI score: penalize neutral RSI, reward extremes (opposite of trend-following)
        - RSI slope: momentum deceleration signals reversal
        - Volume spike confirms capitulation

        HTF Logic for REVERSION:
        - Reversion REQUIRES exhaustion on HTF timeframe
        - +25% bonus if HTF shows exhaustion (wick rejection)
        - +15% bonus if HTF trend opposing (trading against exhausted move)
        - No penalty for aligned HTF (just don't boost)
        """
        # daily_trend reserved for future HTF alignment
        _ = daily_trend

        logger.debug(f"[REVERSION] Calculating rank score for {symbol} in {regime}")

        vol_ratio = float(intraday_features.get("volume_ratio") or 1.0)
        vwap_distance = float(intraday_features.get("vwap_distance") or 0.0)
        rsi = float(intraday_features.get("rsi") or 50.0)
        rsi_slope = float(intraday_features.get("rsi_slope") or 0.0)

        # Weights from config
        weights = self._get("ranking", "weights")

        # Extension score from config (higher = more extended = better for reversion)
        ext_cfg = weights["extension"]
        s_ext = min(vwap_distance * ext_cfg["multiplier"], ext_cfg["cap"])

        # RSI score for REVERSION (from ranker.py lines 97-98)
        # For reversion: EXTREME RSI is GOOD (opposite of trend-following)
        # RSI < 30 or RSI > 70 is what we want
        rsi_cfg = weights.get("rsi", {})
        if rsi_cfg:
            rsi_extreme_bonus = rsi_cfg.get("extreme_bonus", 0.5)
            rsi_neutral_penalty = rsi_cfg.get("neutral_penalty", -0.2)
            if rsi < 30 or rsi > 70:
                s_rsi = rsi_extreme_bonus  # Extreme RSI is good for reversion
            elif 40 < rsi < 60:
                s_rsi = rsi_neutral_penalty  # Neutral RSI is bad for reversion
            else:
                s_rsi = 0.0  # Mild zone
        else:
            s_rsi = 0.0

        # RSI slope score for REVERSION (from ranker.py lines 98)
        # For reversion: we want RSI to be DECELERATING (slope opposing the extreme)
        # If RSI is oversold (<30), positive slope means reversal starting
        # If RSI is overbought (>70), negative slope means reversal starting
        rsi_slope_cfg = weights.get("rsi_slope", {})
        if rsi_slope_cfg:
            slope_cap = rsi_slope_cfg.get("cap", 0.3)
            slope_mult = rsi_slope_cfg.get("multiplier", 0.1)
            bias = intraday_features.get("bias", "long")
            if bias == "long" and rsi_slope > 0:
                # Long reversion with RSI turning up = good
                s_rsi_slope = min(rsi_slope * slope_mult, slope_cap)
            elif bias == "short" and rsi_slope < 0:
                # Short reversion with RSI turning down = good
                s_rsi_slope = min(abs(rsi_slope) * slope_mult, slope_cap)
            else:
                s_rsi_slope = 0.0
        else:
            s_rsi_slope = 0.0

        # Exhaustion score from config (extreme RSI)
        exhaust_cfg = weights["exhaustion"]
        if rsi < 30:
            s_exhaust = (30 - rsi) / 30 * exhaust_cfg["oversold_multiplier"]
        elif rsi > 70:
            s_exhaust = (rsi - 70) / 30 * exhaust_cfg["overbought_multiplier"]
        else:
            s_exhaust = 0.0

        # Volume spike from config (capitulation)
        vol_cfg = weights["volume"]
        s_vol = min((vol_ratio - 1.0) * vol_cfg["above_threshold_mult"], vol_cfg["cap"]) if vol_ratio > 1.0 else 0.0

        base_score = s_ext + s_exhaust + s_vol + s_rsi + s_rsi_slope

        # Regime multiplier from config - reversion thrives in chop
        regime_mults = self._get("ranking", "regime_multipliers")
        regime_mult = regime_mults.get(regime, 1.0)

        # HTF (15m) multiplier - REVERSION requires exhaustion confirmation
        htf_mult = 1.0
        bias = intraday_features.get("bias", "long")
        if htf_context:
            htf_trend = htf_context.get("htf_trend", "neutral")
            htf_exhaustion = htf_context.get("htf_exhaustion", False)

            # HTF exhaustion is the KEY signal for reversion
            if htf_exhaustion:
                htf_mult = 1.25  # +25% for HTF exhaustion (wick rejection)

            # Also bonus for opposing HTF trend (we're fading the move)
            htf_opposing = (htf_trend == "down" and bias == "long") or (htf_trend == "up" and bias == "short")
            if htf_opposing:
                htf_mult *= 1.15  # +15% for fading into exhausted move

            # No penalty for aligned HTF - just don't boost

        final_score = base_score * regime_mult * htf_mult

        return RankingResult(
            score=final_score,
            components={
                "extension": s_ext,
                "exhaustion": s_exhaust,
                "volume": s_vol,
                "rsi": s_rsi,
                "rsi_slope": s_rsi_slope
            },
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
        Reversion entry with setup-type-specific logic.

        REVERSION category setups:
        - failure_fade: Enter at failed breakout level (ORH for short, ORL for long)
        - vwap_mean_reversion: Enter at VWAP
        - volume_spike_reversal: Enter at exhaustion extreme
        - gap_fill: Enter at gap structure
        - liquidity_sweep / fair_value_gap (ICT): Enter at sweep level

        Entry principles:
        - Fades/Failures: Enter AT the rejection level (ORL for long bounce, ORH for short)
        - VWAP reversion: Enter at VWAP
        - Exhaustion: Enter at current extreme
        """
        logger.debug(f"[REVERSION] Calculating entry for {symbol} {setup_type} bias={bias}")

        current_close = float(df5m["close"].iloc[-1])
        vwap = float(df5m["vwap"].iloc[-1]) if "vwap" in df5m.columns else current_close
        orh = levels.get("ORH", current_close)
        orl = levels.get("ORL", current_close)

        entry_cfg = self._get("entry")
        triggers = entry_cfg["triggers"]
        setup_lower = setup_type.lower()

        # Setup-type-specific entry logic for REVERSION
        if "fade" in setup_lower or "failure" in setup_lower or "rejection" in setup_lower or "choc" in setup_lower:
            # Failure fade - enter AT the rejection level
            if bias == "long":
                entry_ref = orl  # Bounce off support
                entry_trigger = "support_bounce"
            else:
                entry_ref = orh  # Rejection at resistance
                entry_trigger = "resistance_rejection"
        elif "vwap" in setup_lower:
            # VWAP mean reversion - enter at VWAP
            if bias == "long":
                entry_ref = vwap * 0.999  # Just below VWAP
            else:
                entry_ref = vwap * 1.001  # Just above VWAP
            entry_trigger = "vwap_reversion"
        elif "gap" in setup_lower:
            # Gap fill - enter at gap structure
            entry_ref = orl if bias == "long" else orh
            entry_trigger = "gap_fill"
        elif "liquidity" in setup_lower or "sweep" in setup_lower:
            # ICT liquidity sweep - enter at swept level
            entry_ref = orl if bias == "long" else orh
            entry_trigger = "liquidity_sweep"
        elif "fair_value" in setup_lower or "fvg" in setup_lower:
            # ICT fair value gap - enter at FVG
            entry_ref = orl if bias == "long" else orh
            entry_trigger = "fvg_fill"
        else:
            # Default reversion - enter at extreme
            if bias == "long":
                entry_ref = min(current_close, orl)
                entry_trigger = triggers["long"]
            else:
                entry_ref = max(current_close, orh)
                entry_trigger = triggers["short"]

        # Wider entry zone for reversion - harder to time
        zone_mult = entry_cfg["zone_mult_atr"]
        zone_width = atr * zone_mult
        entry_zone = (entry_ref - zone_width, entry_ref + zone_width)

        # Reversion entries wait for confirmation
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
        Reversion targets: VWAP is the primary target.

        Mean reversion targets VWAP as the "mean":
        - T1: 50% move to VWAP
        - T2: VWAP level
        - T3: Overshoot past VWAP (rare)
        """
        # atr reserved for future ATR-based target caps
        _ = atr
        # measured_move reserved for future measured move targets
        _ = measured_move

        logger.debug(f"[REVERSION] Calculating targets for {symbol} entry={entry_ref_price:.2f}, sl={hard_sl:.2f}")

        risk_per_share = abs(entry_ref_price - hard_sl)

        targets_cfg = self._get("targets")
        vwap_cfg = targets_cfg["vwap_based"]
        fallback_rr = targets_cfg["fallback_rr_ratios"]

        # Get VWAP for target calculation
        vwap = levels.get("VWAP", entry_ref_price)

        # Calculate move to VWAP
        if bias == "long":
            move_to_vwap = vwap - entry_ref_price
        else:
            move_to_vwap = entry_ref_price - vwap

        move_to_vwap = max(move_to_vwap, 0)  # Ensure positive

        # Targets based on mean reversion from config
        if move_to_vwap > 0:
            t1_move = move_to_vwap * vwap_cfg["t1_frac"]
            t2_move = move_to_vwap * vwap_cfg["t2_frac"]
            t3_move = move_to_vwap * vwap_cfg["t3_frac"]
        else:
            # Fallback to R:R based from config
            t1_move = fallback_rr["t1"] * risk_per_share
            t2_move = fallback_rr["t2"] * risk_per_share
            t3_move = fallback_rr["t3"] * risk_per_share

        if bias == "long":
            t1 = entry_ref_price + t1_move
            t2 = entry_ref_price + t2_move
            t3 = entry_ref_price + t3_move
        else:
            t1 = entry_ref_price - t1_move
            t2 = entry_ref_price - t2_move
            t3 = entry_ref_price - t3_move

        # Calculate effective R:R
        t1_rr = t1_move / max(risk_per_share, 1e-6)
        t2_rr = t2_move / max(risk_per_share, 1e-6)
        t3_rr = t3_move / max(risk_per_share, 1e-6)

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
