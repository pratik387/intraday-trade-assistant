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
    TargetResult,
    get_cap_segment,
    safe_level_get
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
        vol_ratio = features["volume_ratio"]
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
        vol_mult: float,
        regime_diagnostics: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        Reversion-specific gate validations - HARD GATES ONLY, NO PENALTIES.

        Reversion plays work best after extremes:
        - RSI extremes already checked in screen() - HARD GATE there
        - Volume spike is nice-to-have, not required
        - Counter-trend trades need sufficient strength - HARD BLOCK if weak

        Pro Trader Approach (Larry Connors):
        - Reversion is about exhaustion, not momentum
        - Extremes are the SIGNAL, not a penalty condition
        - Rejection candle is confirmation, not requirement
        """
        # df1m reserved for future 1-minute analysis
        _ = df1m
        # adx not used for reversion gates
        _ = adx

        reasons = []
        passed = True
        size_mult = 1.0  # NO PENALTIES - hard gates only

        # ========== UNIFIED FILTER STRUCTURE (Dec 2024) ==========
        setup_filters = self._get("gates", "setup_filters") or {}
        setup_lower = setup_type.lower()
        is_long = "_long" in setup_type
        bias = "long" if is_long else "short"

        # Get current hour from df5m index
        current_time = df5m.index[-1] if hasattr(df5m.index[-1], 'hour') else None
        current_hour = current_time.hour if current_time else 0

        # Get volume from 5m bar for volume filters
        bar5_volume = float(df5m["volume"].iloc[-1]) if len(df5m) > 0 and "volume" in df5m.columns else 0

        # Get cap segment for filtering
        cap_segment = get_cap_segment(symbol)

        # Get RSI for global filters
        rsi_val = float(df5m["rsi"].iloc[-1]) if "rsi" in df5m.columns and not pd.isna(df5m["rsi"].iloc[-1]) else None

        # ========== 1. GLOBAL FILTERS ==========
        global_passed, global_reasons = self.apply_global_filters(
            setup_type=setup_type, symbol=symbol, bias=bias,
            adx=adx, rsi=rsi_val, volume=bar5_volume
        )
        reasons.extend(global_reasons)
        if not global_passed:
            logger.debug(f"[REVERSION] {symbol} {setup_type} BLOCKED by global filters: {global_reasons}")
            return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=0)

        # ========== 2. SETUP-SPECIFIC FILTERS (use base class unified filter) ==========
        setup_passed, setup_reasons, modifiers = self.apply_setup_filters(
            setup_type=setup_type,
            symbol=symbol,
            regime=regime,
            adx=adx,
            rsi=rsi_val,
            volume=bar5_volume,
            current_hour=current_hour,
            cap_segment=cap_segment,
            structural_rr=strength
        )
        reasons.extend(setup_reasons)
        if not setup_passed:
            logger.debug(f"[REVERSION] {symbol} {setup_type} BLOCKED by setup filters: {setup_reasons}")
            return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=0)

        # ========== 3. SPECIALIZED REVERSION FILTERS ==========
        setup_filter_cfg = setup_filters.get(setup_type, {})

        # 3a. Min volume ratio filter (for volume_spike_reversal_short - needs vol spike)
        if setup_filter_cfg.get("enabled", False):
            min_vol_ratio = setup_filter_cfg.get("min_volume_ratio")
            if min_vol_ratio:
                if vol_mult < min_vol_ratio:
                    reasons.append(f"setup_vol_ratio_blocked:{vol_mult:.1f}x<{min_vol_ratio}x")
                    logger.debug(f"[REVERSION] {symbol} {setup_type} BLOCKED: vol_ratio {vol_mult:.1f}x < {min_vol_ratio}x")
                    return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=0)
                else:
                    reasons.append(f"setup_vol_ratio_ok:{vol_mult:.1f}x")

            # 3b. Blocked daily regimes filter (for volume_spike_reversal_short)
            blocked_daily_regimes = setup_filter_cfg.get("blocked_daily_regimes", [])
            if blocked_daily_regimes:
                daily_regime = ""
                if regime_diagnostics and "daily" in regime_diagnostics:
                    daily_regime = (regime_diagnostics.get("daily") or {}).get("regime", "")
                if daily_regime in blocked_daily_regimes:
                    reasons.append(f"setup_daily_regime_blocked:{daily_regime}")
                    logger.debug(f"[REVERSION] {symbol} {setup_type} BLOCKED: daily_regime={daily_regime} in blocked list")
                    return GateResult(passed=False, reasons=reasons, size_mult=size_mult, min_hold_bars=0)
                else:
                    reasons.append(f"setup_daily_regime_ok:{daily_regime or 'unknown'}")

        # ========== VWAP_RECLAIM VOLUME FILTER (DATA-DRIVEN Dec 2024) ==========
        # From spike test: VWAP_RECLAIM vol<50k: 18 trades (6W, 12L), blocking = +2,675 Rs
        # Low volume VWAP reclaim trades have poor execution and high slippage
        if "vwap" in setup_lower or "mean_reversion" in setup_lower:
            # Get threshold from config
            # bar5_volume already calculated above for long trade filter
            vwap_reclaim_cfg = self._get("gates", "vwap_reclaim_volume")

            if vwap_reclaim_cfg["enabled"]:
                min_volume = vwap_reclaim_cfg["min_volume"]

                if bar5_volume < min_volume:
                    reasons.append(f"vwap_reclaim_blocked:vol{bar5_volume/1000:.0f}k<{min_volume/1000:.0f}k")
                    passed = False
                else:
                    reasons.append(f"vwap_reclaim_vol_ok:{bar5_volume/1000:.0f}k")

        # Min hold bars from config - reversion plays need patience
        min_hold = self._get("gates", "min_hold_bars")

        # Regime rules from config - HARD GATES only (can be disabled for A/B baseline)
        regime_cfg = self._get("gates", "regime_rules")
        regime_gates_enabled = regime_cfg["enabled"]

        if regime_gates_enabled and regime in regime_cfg:
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
        else:
            reasons.append(f"regime_gates_bypassed:{regime}")

        # Failure fade specific check - info only, no penalty
        if "failure_fade" in setup_type:
            ff_cfg = self._get("gates", "failure_fade")
            # Check for rejection (wick) - nice to have, not required
            if df5m is not None and len(df5m) >= 5:
                last_bar = df5m.iloc[-1]
                body = abs(last_bar["close"] - last_bar["open"])
                total_range = last_bar["high"] - last_bar["low"]

                if total_range > 0:
                    body_ratio = body / total_range
                    if body_ratio < ff_cfg["rejection_candle_body_ratio_max"]:
                        reasons.append("rejection_candle_ok")
                    else:
                        reasons.append("no_rejection_candle")

        # Volume spike - info only, no penalty (capitulation is nice, not required)
        vol_cfg = self._get("gates", "volume_capitulation")
        if vol_mult >= vol_cfg["threshold"]:
            reasons.append(f"capitulation_volume:{vol_mult:.2f}x")
        else:
            reasons.append(f"volume_normal:{vol_mult:.2f}x")

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
        REVERSION-SPECIFIC RANKING (Dec 2024 Recalibration)

        Pro trader research findings for REVERSION/MEAN-REVERSION plays:
        - Low ADX = GOOD (ranging market = mean reversion works)
        - Extreme RSI = GOOD (oversold/overbought = exhaustion signal)
        - Counter-VWAP = GOOD (extended from VWAP = reversion opportunity)
        - Far from level = GOOD (overextension = reversion setup)

        7 weighted components:
        1. Volume (10%): Moderate - exhaustion volume is key
        2. RSI (25%): Extreme RSI = exhaustion signal
        3. ADX (20%): Low ADX = ranging market
        4. VWAP (15%): Counter-VWAP = overextension
        5. Distance (10%): Far = overextension opportunity
        6. Squeeze (10%): Low squeeze = ranging market
        7. Acceptance (10%): Disabled - exhaustion matters more than structure
        """
        logger.debug(f"[REVERSION] Calculating rank score for {symbol} in {regime}")

        # REQUIRED features
        vol_ratio = float(intraday_features["volume_ratio"])
        rsi = float(intraday_features["rsi"])
        adx = float(intraday_features["adx"])
        above_vwap = bool(intraday_features["above_vwap"])
        bias = intraday_features["bias"]

        # OPTIONAL features (None = skip scoring, no hidden defaults)
        dist_from_level_bpct = intraday_features.get("dist_from_level_bpct")
        squeeze_pctile = intraday_features.get("squeeze_pctile")
        acceptance_status = intraday_features.get("acceptance_status")

        weights = self._get("ranking", "weights")
        score_scale = self._get("ranking", "score_scale")

        # 1. VOLUME SCORE - Moderate for reversion
        vol_cfg = weights["volume"]
        s_vol = min(vol_ratio / vol_cfg["divisor"], vol_cfg["cap"])

        # 2. RSI SCORE - EXTREME = GOOD for reversion (exhaustion signal)
        rsi_cfg = weights["rsi"]
        if bias == "long":
            if rsi <= rsi_cfg["long_very_oversold_threshold"]:
                s_rsi = rsi_cfg["extreme_bonus"]  # RSI < 25 = extreme oversold
            elif rsi <= rsi_cfg["long_oversold_threshold"]:
                s_rsi = rsi_cfg["good_bonus"]  # RSI < 35 = oversold
            elif rsi >= rsi_cfg["long_overbought_penalty_threshold"]:
                s_rsi = rsi_cfg["penalty"]  # RSI > 60 = wrong direction
            else:
                s_rsi = 0.0
        else:  # short
            if rsi >= rsi_cfg["short_very_overbought_threshold"]:
                s_rsi = rsi_cfg["extreme_bonus"]  # RSI > 75 = extreme overbought
            elif rsi >= rsi_cfg["short_overbought_threshold"]:
                s_rsi = rsi_cfg["good_bonus"]  # RSI > 65 = overbought
            elif rsi <= rsi_cfg["short_oversold_penalty_threshold"]:
                s_rsi = rsi_cfg["penalty"]  # RSI < 40 = wrong direction
            else:
                s_rsi = 0.0

        # 3. ADX SCORE - LOW = GOOD for reversion (ranging market)
        adx_cfg = weights["adx"]
        if adx <= adx_cfg["ideal_max"]:
            s_adx = adx_cfg["ideal_bonus"]  # ADX < 20 = ideal ranging
        elif adx <= adx_cfg["acceptable_max"]:
            s_adx = adx_cfg["acceptable_bonus"]  # ADX < 25 = acceptable
        elif adx >= adx_cfg["penalty_threshold"]:
            s_adx = adx_cfg["penalty"]  # ADX > 35 = trending (bad for reversion)
        else:
            s_adx = 0.0

        # 4. VWAP SCORE - COUNTER = GOOD for reversion (overextension)
        vwap_cfg = weights["vwap"]
        if bias == "long":
            # Long reversion: want to buy BELOW vwap (counter = extended down)
            s_vwap = vwap_cfg["counter_bonus"] if not above_vwap else vwap_cfg["aligned_bonus"]
        else:
            # Short reversion: want to sell ABOVE vwap (counter = extended up)
            s_vwap = vwap_cfg["counter_bonus"] if above_vwap else vwap_cfg["aligned_bonus"]

        # 5. DISTANCE SCORE - FAR = GOOD for reversion (overextension)
        dist_cfg = weights["distance"]
        if dist_from_level_bpct is not None:
            adist = abs(dist_from_level_bpct)
            if adist <= dist_cfg["near_bpct"]:
                s_dist = dist_cfg["near_score"]  # Near = less overextension
            elif adist <= dist_cfg["ok_bpct"]:
                s_dist = dist_cfg["ok_score"]  # Moderate extension
            else:
                s_dist = dist_cfg["far_score"]  # Far = good overextension
        else:
            s_dist = 0.0

        # 6. SQUEEZE SCORE - Low squeeze = good for reversion
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

        # 7. ACCEPTANCE SCORE - Disabled for reversion
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

        # Regime multiplier
        setup_type = intraday_features["setup_type"]
        regime_mult = self._get_strategy_regime_mult(setup_type, regime)

        # HTF context handled in universal adjustments
        _ = daily_trend
        _ = htf_context

        final_score = base_score * regime_mult

        logger.debug(f"[REVERSION] {symbol} score={final_score:.3f} (weighted_sum={weighted_sum:.3f}*scale={score_scale}) * regime={regime_mult:.2f}")

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
        orh = safe_level_get(levels, "ORH", current_close)
        orl = safe_level_get(levels, "ORL", current_close)

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

        # Apply minimum zone width (as % of price) for large cap stocks with low ATR
        min_zone_pct = entry_cfg.get("min_zone_pct")
        if min_zone_pct > 0:
            min_zone_width = entry_ref * (min_zone_pct / 100.0)
            if zone_width < min_zone_width:
                logger.debug(f"[REVERSION] {symbol} zone widened from {zone_width:.3f} to {min_zone_width:.3f} (min_zone_pct={min_zone_pct}%)")
                zone_width = min_zone_width

        entry_zone = (entry_ref - zone_width, entry_ref + zone_width)

        # Reversion entries wait for confirmation - check for setup-specific override
        entry_mode = entry_cfg["mode"]  # default: "retest"
        mode_overrides = entry_cfg.get("mode_overrides", {})

        # Check for setup-specific entry mode override
        for setup_pattern, override_mode in mode_overrides.items():
            if setup_pattern.startswith("_"):  # skip comments
                continue
            if setup_pattern.lower() in setup_lower:
                entry_mode = override_mode
                logger.debug(f"[REVERSION] {symbol} {setup_type} using entry_mode={override_mode} (override for {setup_pattern})")
                break

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
        vwap = safe_level_get(levels, "VWAP", entry_ref_price)

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

    # ======================== RSI PENALTY ========================

    def _apply_rsi_penalty(self, rsi_val: float, bias: str) -> tuple:
        """
        REVERSION RSI penalty: DISABLED - screening already checks RSI extremes.

        Pro Trader Framework (Larry Connors RSI(2)):
        - Extreme RSI is the SETUP CONDITION - already enforced in screen()
        - screen() blocks neutral RSI (hard gate), so no penalty needed here
        - Double-checking would be redundant

        The screen() method at lines 106-113 already blocks:
        - Long reversion when RSI > oversold_threshold
        - Short reversion when RSI < overbought_threshold
        """
        # RSI already checked as hard gate in screen() - no penalty needed
        _ = rsi_val
        _ = bias
        return (1.0, None)
