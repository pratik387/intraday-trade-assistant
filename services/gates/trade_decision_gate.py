from __future__ import annotations
"""
trade_decision_gate.py
----------------------
Central gate that combines:
  • Structure event detection (breakout/breakdown, VWAP reclaim/lose, squeeze release, failure/fade)
  • Market regime policy (index trend/chop/squeeze)
  • Event policy (macro windows, expiry, symbol events)
  • News spike adjustments (1-minute anomaly confirmation & sizing)

This module **does not** read config files. All thresholds/policies enter via the injected
components. Keep it pure and deterministic so backtests match live.

Public API
----------
class TradeDecisionGate:
    def __init__(self, *, structure_detector, regime_gate, event_policy_gate, news_spike_gate): ...
    def evaluate(self, *, symbol: str, now, df1m_tail, df5m_tail, index_df5m, levels) -> GateDecision: ...

Required component protocols (duck-typed):
- structure_detector.detect_setups(symbol, df5m_tail, levels) -> list[SetupCandidate]
- regime_gate.compute_regime(index_df5m) -> tuple[str, float]  # (regime, confidence 0..1)
- regime_gate.allow_setup(setup_type: str, regime: str, strength: float, adx_5m: float, vol_mult_5m: float) -> bool
- regime_gate.size_multiplier(regime: str) -> float  # optional; if missing, treated as 1.0
- event_policy_gate.decide_policy(now, symbol) -> (Policy, dict)  # Policy is defined in event_policy_gate
- news_spike_gate.has_symbol_spike(df1m_tail) -> (bool, NewsSignal)  # NewsSignal in news_spike_gate
- news_spike_gate.adjustment_for(signal) -> Adjustment            # Adjustment in news_spike_gate

Types
-----
SetupType: one of the literals below; extend in your structure detector if needed.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol, Literal

import pandas as pd

from .event_policy_gate import EventPolicyGate
from .news_spike_gate import NewsSpikeGate
from .market_sentiment_gate import MarketSentimentGate
from services.features import compute_hcet_features
from collections import defaultdict
from datetime import datetime, timedelta
from config.logging_config import get_agent_logger

logger = get_agent_logger()

SetupType = Literal[
    "breakout_long",
    "breakout_short",
    "vwap_reclaim_long",
    "vwap_lose_short",
    "squeeze_release_long",
    "squeeze_release_short",
    "failure_fade_long",
    "failure_fade_short",
    "gap_fill_long",
    "gap_fill_short",
    "flag_continuation_long",
    "flag_continuation_short",
    "support_bounce_long",
    "resistance_bounce_short",
    "orb_breakout_long",
    "orb_breakout_short",
    "vwap_mean_reversion_long",
    "vwap_mean_reversion_short",
    "volume_spike_reversal_long",
    "volume_spike_reversal_short",
    "trend_pullback_long",
    "trend_pullback_short",
    "range_rejection_long",
    "range_rejection_short",
    "momentum_breakout_long",
    "momentum_breakout_short",
    "trend_continuation_long",
    "trend_continuation_short",
    "order_block_long",
    "order_block_short",
    "fair_value_gap_long",
    "fair_value_gap_short",
    "liquidity_sweep_long",
    "liquidity_sweep_short",
    "premium_zone_short",
    "discount_zone_long",
    "equilibrium_breakout_long",
    "equilibrium_breakout_short",
    "break_of_structure_long",
    "break_of_structure_short",
    "change_of_character_long",
    "change_of_character_short",
    # INSTITUTIONAL RANGE TRADING - Profit from choppy markets
    "range_deviation_long",
    "range_deviation_short",
    "range_mean_reversion_long",
    "range_mean_reversion_short",
]


@dataclass(frozen=True)
class SetupCandidate:
    setup_type: SetupType
    strength: float  # arbitrary score from detector (higher = better)
    reasons: List[str]


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    reasons: List[str]
    setup_type: Optional[SetupType] = None  # DEPRECATED: use setup_candidates instead
    regime: Optional[str] = None
    regime_conf: float = 0.0
    size_mult: float = 1.0
    min_hold_bars: int = 0
    matched_rule: Optional[str] = None  # if you use rule miner/meta later
    p_breakout: Optional[float] = None  # placeholder for meta-prob models
    setup_candidates: Optional[List[SetupCandidate]] = None  # NEW: full structure detection results
    regime_diagnostics: Optional[dict] = None  # Phase 2: Multi-TF regime diagnostics


# ----------------------------- Component Protocols -----------------------------

class StructureDetector(Protocol):  # pragma: no cover (interface only)
    def detect_setups(self, symbol: str, df5m_tail: pd.DataFrame, levels: dict | None) -> List[SetupCandidate]:
        ...


class RegimeGate(Protocol):  # pragma: no cover (interface only)
    def compute_regime(self, index_df5m: pd.DataFrame) -> Tuple[str, float]:
        ...

    def allow_setup(
        self,
        setup_type: SetupType,
        regime: str,
        strength: float,
        adx_5m: float,
        vol_mult_5m: float,
    ) -> bool:
        ...

    # Optional sizing bias by regime
    def size_multiplier(self, regime: str) -> float:  # noqa: D401 (docstring not required)
        ...


# --------------------------------- Utility ------------------------------------

def _is_breakout(setup: SetupType) -> bool:
    return setup in {
        "breakout_long",
        "breakout_short",
        "squeeze_release_long",
        "squeeze_release_short",
        "orb_breakout_long",
        "orb_breakout_short",
        "flag_continuation_long",
        "flag_continuation_short",
        "equilibrium_breakout_long",
        "equilibrium_breakout_short",
        "break_of_structure_long",
        "break_of_structure_short",
        "change_of_character_long",
        "change_of_character_short",
    }


def _is_fade(setup: SetupType) -> bool:
    return setup in {
        "failure_fade_long",
        "failure_fade_short",
        "volume_spike_reversal_long",
        "volume_spike_reversal_short",
        "range_rejection_long",
        "range_rejection_short",
        "order_block_long",
        "order_block_short",
        "fair_value_gap_long",
        "fair_value_gap_short",
        "liquidity_sweep_long",
        "liquidity_sweep_short",
        "premium_zone_short",
        "discount_zone_long",
    }


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Regime allow-list helper (strict by default; HCET can bypass)
def _regime_allows(setup: str, regime: str) -> bool:
    if regime in {"choppy", "range"}:
        # keep chop tight: only allow orb_pullback_long; failure_fade_long only via HCET
        return setup in {"orb_pullback_long"}
    if regime == "trend_down":
        return setup in {"gap_fill_short", "range_break_retest_short", "breakout_short"}
    if regime == "trend_up":
        return setup in {"breakout_long", "orb_pullback_long"}
    return False


# ---------------------------- TradeDecisionGate --------------------------------

class TradeDecisionGate:
    """Combine structure + regime + event + news adjustments into one decision.

    All dependencies are injected so this class stays testable and config-free.
    """

    def __init__(
        self,
        *,
        structure_detector: StructureDetector,
        regime_gate: RegimeGate,
        event_policy_gate: EventPolicyGate,
        news_spike_gate: NewsSpikeGate,
        market_sentiment_gate=None,
        quality_filters: Optional[dict] = None,
    ) -> None:
        self.structure = structure_detector
        self.regime_gate = regime_gate
        self.event_gate = event_policy_gate
        self.news_gate = news_spike_gate
        self.sentiment_gate = market_sentiment_gate

        # Setup sequencing tracker - Enhancement 3
        self.setup_history = defaultdict(list)  # symbol -> [(timestamp, setup_type, success)]
        self.setup_sequence_patterns = {
            "failed_breakdown_to_bounce": ["breakout_short", "vwap_reclaim_long"],
            "failed_breakout_to_fade": ["breakout_long", "failure_fade_short"],
            "squeeze_to_breakout": ["squeeze_release_long", "breakout_long"],
            "fake_breakout_reversal": ["breakout_long", "breakout_short"]
        }
        self.quality_filters = quality_filters or {}

    # ------------------------------ SETUP SEQUENCING (Enhancement 3) ------
    def _track_setup(self, symbol: str, timestamp, setup_type: str):
        """Track a setup for sequence analysis."""
        # Keep only last 24 hours of history to avoid memory bloat
        cutoff = timestamp - timedelta(hours=24)
        self.setup_history[symbol] = [
            (ts, stype, success) for ts, stype, success in self.setup_history[symbol]
            if ts > cutoff
        ]
        # Add current setup (success will be determined later)
        self.setup_history[symbol].append((timestamp, setup_type, None))

    def _get_sequence_multiplier(self, symbol: str, current_setup: str) -> float:
        """
        Analyze recent setup sequence and return confidence multiplier.
        Returns > 1.0 for favorable sequences, < 1.0 for unfavorable.
        """
        history = self.setup_history.get(symbol, [])
        if len(history) < 2:
            return 1.0

        # Look at last setup (excluding current one)
        recent_setups = [setup_type for _, setup_type, _ in history[-3:-1]]  # Last 2 setups before current

        # Pattern recognition - failed setups leading to reversals
        sequence_bonus = 1.0

        for pattern_name, pattern_sequence in self.setup_sequence_patterns.items():
            if len(recent_setups) >= len(pattern_sequence) - 1:
                # Check if recent setups + current setup match a pattern
                full_sequence = recent_setups + [current_setup]
                if full_sequence[-len(pattern_sequence):] == pattern_sequence:
                    if "reversal" in pattern_name or "fade" in pattern_name:
                        sequence_bonus = 1.4  # 40% bonus for reversal patterns
                    elif "continuation" in pattern_name or "breakout" in pattern_name:
                        sequence_bonus = 1.2  # 20% bonus for continuation patterns

        # Additional logic: Multiple failed breakouts → higher reversal probability
        if current_setup in ["failure_fade_short", "vwap_reclaim_long"]:
            failed_breakouts = sum(1 for _, setup, _ in history[-5:] if setup == "breakout_long")
            if failed_breakouts >= 2:
                sequence_bonus = max(sequence_bonus, 1.3)  # Multiple failures = strong reversal

        return sequence_bonus

    # ------------------------------ HCET (High-Conviction Entry Template) ------
    def _is_high_conviction_candidate(self, setup_type: str, regime: str, features: Optional[dict]) -> tuple[bool, list[str]]:
        """Return (ok, reasons). Fires only if *all* confirms pass."""
        f = features or {}
        reasons = []

        # Shared confirmations
        sector_ok  = (f.get("sector_momentum", 0.0) * f.get("index_momentum", 0.0)) >= 0  # aligned sign
        vol_ok     = f.get("volume_ratio", 1.0) >= 1.2
        rr         = f.get("structural_rr", 0.0)
        rr_ok      = 1.2 <= rr <= 3.0  # avoid <1.2 and fantasy >3R
        news_clear = not f.get("news_spike_flag", False)

        if not (sector_ok and vol_ok and rr_ok and news_clear):
            return (False, ["hcet_shared_fail"])

        # Combo-specific confirmations
        if regime == "trend_down" and setup_type in {"gap_fill_short", "range_break_retest_short"}:
            gap_pct  = f.get("gap_pct", 0.0)
            fill_pct = f.get("gap_fill_pct", 1.0)
            wick_ok  = f.get("last_bar_upper_wick_bpct", 0.0) >= 35.0
            return (gap_pct >= 1.2 and fill_pct <= 0.5 and wick_ok, ["hcet_downtrend"])

        if regime in {"choppy", "range"} and setup_type == "orb_pullback_long":
            or_made      = f.get("orb_high_made", False)
            pb_frac      = f.get("pullback_frac_of_or", 1.0)   # 0.0–1.0
            vwap_reclaim = f.get("vwap_reclaim_bar", False)
            return (or_made and pb_frac <= 0.382 and vwap_reclaim, ["hcet_orb_pb_long"])

        if regime in {"choppy", "range"} and setup_type == "failure_fade_long":
            ret_z  = f.get("ret_z", 0.0)
            div_ok = f.get("rsi_divergence_2bar_bull", False)
            wick_ok = f.get("last_bar_lower_wick_bpct", 0.0) >= 40.0
            return (ret_z >= 2.5 and div_ok and wick_ok, ["hcet_ff_long"])

        return (False, ["hcet_no_match"])

    def _compute_features(
        self,
        symbol: str,
        df1m_tail: Optional[pd.DataFrame],
        df5m_tail: Optional[pd.DataFrame],
        index_df5m: Optional[pd.DataFrame],
        structural_rr: float
    ) -> Optional[dict]:
        """
        Compute HCET features internally using features.py module.

        Returns None if insufficient data for feature computation.
        Used ONLY for high-conviction setups when normal filters would block:
        - Early trading day (< 10 bars)
        - Regime restrictions
        - Time window restrictions

        This is an expensive computation, only called when HCET bypasses are needed.
        """
        try:
            # Need at least 5m data to compute meaningful features
            if df5m_tail is None or len(df5m_tail) == 0:
                return None

            # Index data is helpful but not strictly required
            if index_df5m is None or len(index_df5m) == 0:
                index_df5m = df5m_tail  # fallback to symbol's own data

            # Compute features using the centralized function
            features = compute_hcet_features(
                df1m_tail=df1m_tail,
                df5m_tail=df5m_tail,
                index_df5m=index_df5m,
                sector_df5m=None,  # Not available in gate context
                structural_rr=structural_rr
            )

            return features

        except Exception as e:
            # Log but don't fail - return None to indicate features unavailable
            # logger would need to be imported if we want to log here
            return None

    # ------------------------------ Public API ---------------------------------
    def evaluate(
        self,
        *,
        symbol: str,
        now,
        df1m_tail: pd.DataFrame,
        df5m_tail: pd.DataFrame,
        index_df5m: pd.DataFrame,
        levels: Optional[dict],
        plan: Optional[dict] = None,    # planner info (e.g., regime, rr, etc.)
        daily_df: Optional[pd.DataFrame] = None  # Phase 2: Multi-TF regime (210 days)
    ) -> GateDecision:
        reasons: List[str] = []

        # ---------------- FAST QUALITY FILTERS ----------------
        # Time window filtering with HCET override (we defer the final veto until after HCET computation)
        import pandas as pd
        try:
            ts = pd.to_datetime(now)
            minute_of_day = ts.hour * 60 + ts.minute
        except Exception:
            minute_of_day = None

        time_blocked = False
        if minute_of_day is not None:
            # Get time windows from config (defaults kept restrictive)
            tw = self.quality_filters.get('time_windows', {})

            def to_min(s: str, default_min: int) -> int:
                try:
                    if isinstance(s, str) and ':' in s:
                        hh, mm = map(int, s.split(':'))
                        return hh * 60 + mm
                except Exception:
                    pass
                return default_min

            morning_start   = to_min(tw.get('morning_start', '10:30'), 630)
            morning_end     = to_min(tw.get('morning_end', '12:30'),   750)
            afternoon_start = to_min(tw.get('afternoon_start','14:15'),855)
            afternoon_end   = to_min(tw.get('afternoon_end',  '15:00'),900)

            in_morning   = morning_start   <= minute_of_day <= morning_end
            in_afternoon = afternoon_start <= minute_of_day <= afternoon_end

            if not (in_morning or in_afternoon):
                # mark as blocked for now; allow HCET to override later
                time_blocked = True

        # Evidence-based pattern filters (momentum consolidation / range compression)
        pattern_filters_enabled = self.quality_filters.get('pattern_filters_enabled', True)

        # Skip feasible checks for very early data; allow HCET to decide later
        if df5m_tail is None or len(df5m_tail) < 10:
            reasons.append(f"skip_feasible_checks:insufficient_bars_{len(df5m_tail) if df5m_tail is not None else 0}<10")
            pattern_filters_enabled = False  # don't apply generic patterns on scarce data

        if pattern_filters_enabled and df5m_tail is not None and not df5m_tail.empty:
            pattern_reasons = []
            pattern_passed = True

            # Momentum consolidation
            if self.quality_filters.get('momentum_consolidation_enabled', True) and len(df5m_tail) >= 3:
                try:
                    close_3_bars_ago = df5m_tail["close"].iloc[-4] if len(df5m_tail) >= 4 else df5m_tail["close"].iloc[0]
                    current_close = df5m_tail["close"].iloc[-1]
                    momentum_15min = ((current_close - close_3_bars_ago) / close_3_bars_ago) * 100
                    momentum_threshold = self._get_strategy_momentum_threshold(None)
                    if abs(momentum_15min) > momentum_threshold:
                        pattern_reasons.append(f"momentum_consolidation_fail:{momentum_15min:.2f}%>{momentum_threshold}%")
                        pattern_passed = False
                    else:
                        pattern_reasons.append(f"momentum_consolidation_pass:{momentum_15min:.2f}%<={momentum_threshold}%")
                except Exception as e:
                    pattern_reasons.append(f"momentum_consolidation_error:{e.__class__.__name__}")

            # Range compression
            if self.quality_filters.get('range_compression_enabled', True) and len(df5m_tail) >= 20:
                try:
                    high = df5m_tail["high"]
                    low = df5m_tail["low"]
                    close_prev = df5m_tail["close"].shift(1)
                    tr1 = high - low
                    tr2 = abs(high - close_prev)
                    tr3 = abs(low - close_prev)
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
                    current_atr = atr_14.iloc[-1]
                    avg_atr_20 = atr_14.rolling(window=20, min_periods=20).mean().iloc[-1]
                    if pd.notna(current_atr) and pd.notna(avg_atr_20) and avg_atr_20 > 0:
                        atr_ratio = current_atr / avg_atr_20
                        compression_threshold = self.quality_filters.get('range_compression_threshold', 0.8)
                        if atr_ratio > compression_threshold:
                            pattern_reasons.append(f"range_compression_fail:{atr_ratio:.3f}>{compression_threshold}")
                            pattern_passed = False
                        else:
                            pattern_reasons.append(f"range_compression_pass:{atr_ratio:.3f}<={compression_threshold}")
                    else:
                        pattern_reasons.append("range_compression_insufficient_data")
                except Exception as e:
                    pattern_reasons.append(f"range_compression_error:{e.__class__.__name__}")

            if not pattern_passed:
                reasons.extend(pattern_reasons)
                return GateDecision(accept=False, reasons=reasons)
            else:
                reasons.extend(pattern_reasons)

        # ---------------- STRUCTURE ----------------
        logger.debug(f"TRADE_GATE: Calling structure.detect_setups for {symbol}")
        setups = self.structure.detect_setups(symbol, df5m_tail, levels)
        logger.debug(f"TRADE_GATE: Structure detection returned {len(setups)} setups for {symbol}")
        if not setups:
            logger.debug(f"TRADE_GATE: No structure events found for {symbol}, returning no_structure_event")
            return GateDecision(accept=False, reasons=["no_structure_event"])
        setups.sort(key=lambda s: s.strength, reverse=True)
        best = setups[0]
        reasons.extend([f"structure:{r}" for r in best.reasons])

        # Setup blacklist
        blacklisted_setups = self.quality_filters.get('blacklist_setups', [])
        if best.setup_type in blacklisted_setups:
            return GateDecision(accept=False, reasons=[f"blacklisted_setup:{best.setup_type}"])

        # ---------------- REGIME (Phase 2: Multi-timeframe) ----------------
        df_for_regime = index_df5m if index_df5m is not None and not index_df5m.empty else df5m_tail

        # Try multi-timeframe regime if available
        regime_diagnostics = None
        if hasattr(self.regime_gate, 'compute_regime_multi_tf') and daily_df is not None:
            try:
                regime, regime_confidence, regime_diagnostics = self.regime_gate.compute_regime_multi_tf(
                    df5=df_for_regime,
                    daily_df=daily_df,
                    symbol=symbol
                )
            except Exception as e:
                logger.warning(f"Multi-TF regime failed for {symbol}, falling back to 5m-only: {e}")
                regime, regime_confidence = self.regime_gate.compute_regime(df_for_regime)
        else:
            # Fallback to 5m-only regime
            regime, regime_confidence = self.regime_gate.compute_regime(df_for_regime)

        # Phase 3: Check if setup should be blocked by daily regime (evidence-based)
        # Linda Raschke MTF filtering: Block counter-trend setups when daily conf ≥ 0.70
        if regime_diagnostics and "daily" in regime_diagnostics:
            from services.gates.multi_timeframe_regime import DailyRegimeResult
            daily_data = regime_diagnostics["daily"]
            daily_result = DailyRegimeResult(
                regime=daily_data.get("regime", "chop"),
                confidence=daily_data.get("confidence", 0.0),
                trend_strength=daily_data.get("trend_strength", 0.0),
                metrics=daily_data.get("metrics", {})
            )

            # Check if multi-TF regime wants to block this setup
            if hasattr(self.regime_gate, 'multi_tf_regime') and hasattr(self.regime_gate.multi_tf_regime, 'should_block_setup'):
                should_block, block_reason = self.regime_gate.multi_tf_regime.should_block_setup(
                    setup_type=best.setup_type,
                    daily_result=daily_result,
                    min_daily_confidence=0.70
                )
                if should_block:
                    return GateDecision(
                        accept=False,
                        reasons=[f"blocked_by_daily_regime:{block_reason}"],
                        setup_type=best.setup_type,
                        regime=regime,
                        regime_conf=regime_confidence,
                        regime_diagnostics=regime_diagnostics
                    )

        # Evidence for regime gate
        strength = _safe_float(best.strength, 0.0)
        if df5m_tail is not None and not df5m_tail.empty:
            last5 = df5m_tail.iloc[-1]
            adx_5m = _safe_float(last5.get("adx", 0.0) if hasattr(last5, "get") else getattr(last5, "adx", 0.0), 0.0)
            if "volume" in df5m_tail.columns:
                recent_vol = df5m_tail["volume"].tail(24)
                median_vol = _safe_float(recent_vol.median(), 1.0) or 1.0
                vol_mult_5m = _safe_float(df5m_tail["volume"].iloc[-1], 0.0) / (median_vol or 1.0)
            else:
                vol_mult_5m = 1.0
        else:
            adx_5m = 0.0
            vol_mult_5m = 1.0

        # ---------------- HCET check (requires setup, regime, features) ---------
        # Only compute features if we might need HCET bypasses (early day, blocked regime, blocked time)
        features = None
        hc_ok, hc_reasons = False, ["hcet_not_needed"]

        # Check if HCET might be needed for bypasses
        insufficient_bars = (df5m_tail is None or len(df5m_tail) < 10)
        regime_blocked = not _regime_allows(best.setup_type, regime)
        time_blocked_check = time_blocked  # from earlier time window check

        if insufficient_bars or regime_blocked or time_blocked_check:
            # Only now compute features since HCET might be needed
            structural_rr = 0.0
            if plan and "quality" in plan and plan["quality"]:
                structural_rr = float(plan["quality"].get("structural_rr", 0.0) or 0.0)

            features = self._compute_features(
                symbol=symbol,
                df1m_tail=df1m_tail,
                df5m_tail=df5m_tail,
                index_df5m=index_df5m,
                structural_rr=structural_rr
            )

            hc_ok, hc_reasons = self._is_high_conviction_candidate(best.setup_type, regime, features)

        # Insufficient bars veto unless HCET
        if df5m_tail is None or len(df5m_tail) < 10:
            if hc_ok:
                reasons.append(f"hcet_enable_early({len(df5m_tail) if df5m_tail is not None else 0}bars)")
            else:
                return GateDecision(accept=False, reasons=reasons)

        # Regime allow-list (strict) unless HCET
        if not _regime_allows(best.setup_type, regime):
            # fallback to your injected regime_gate.allow_setup if you want both
            # Priority 2: Get cap_segment for cap-aware strategy filtering
            cap_segment = "unknown"
            try:
                import json
                from pathlib import Path
                nse_file = Path(__file__).parent.parent.parent / "nse_all.json"
                if nse_file.exists():
                    with nse_file.open() as f:
                        data = json.load(f)
                    cap_map = {item["symbol"]: item.get("cap_segment", "unknown") for item in data}
                    cap_segment = cap_map.get(symbol, "unknown")
            except Exception:
                cap_segment = "unknown"

            if hasattr(self.regime_gate, "allow_setup"):
                base_allow = bool(self.regime_gate.allow_setup(
                    best.setup_type, regime, strength, adx_5m, vol_mult_5m, cap_segment=cap_segment
                ))
            else:
                base_allow = False

            if not base_allow and not hc_ok:
                reasons.append(f"regime_block:{regime}[str={strength:.2f},adx={adx_5m:.2f},volx={vol_mult_5m:.2f},cap={cap_segment}]")
                return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
            elif not base_allow and hc_ok:
                reasons.append(f"hcet_bypass_regime:{','.join(hc_reasons)}")

        size_mult = 1.0
        if hasattr(self.regime_gate, "size_multiplier"):
            try:
                size_mult *= float(self.regime_gate.size_multiplier(regime))
            except Exception:
                pass
        reasons.append(f"regime:{regime}")

        # If time window blocked, allow only HCET to proceed
        if time_blocked and not hc_ok:
            return GateDecision(accept=False, reasons=[f"time_window_block:{minute_of_day}"])
        elif time_blocked and hc_ok:
            reasons.append(f"hcet_bypass_time_window:{minute_of_day}")

        # ---------------- EVENT POLICY (Phase 4: Enhanced with session/event thresholds) ----------------
        # Extract lane_type from candidates for fast scalp detection
        lane_type = None
        for candidate in setups:
            for reason in (candidate.reasons if hasattr(candidate, 'reasons') else []):
                if reason.startswith("lane:"):
                    lane_type = reason.split(":", 1)[1]
                    break
            if lane_type:
                break

        # Get enhanced policy with session/event thresholds
        policy, ctx = self.event_gate.decide_policy(
            now=now,
            symbol=symbol,
            adx_5m=adx_5m,
            vol_mult_5m=vol_mult_5m,
            strength=strength,
            lane_type=lane_type
        )
        # Check policy permissions
        if _is_breakout(best.setup_type) and not policy.allow_breakout:
            reasons.append("event_block:breakout")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
        if _is_fade(best.setup_type) and not policy.allow_fade:
            reasons.append("event_block:fade")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)

        # Phase 4: Check session/event threshold requirements
        if policy.min_adx and adx_5m < policy.min_adx:
            session_or_event = policy.session_type or policy.event_type or "policy"
            reasons.append(f"{session_or_event}:adx_fail:{adx_5m:.1f}<{policy.min_adx}")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)

        if policy.min_volume_mult and vol_mult_5m < policy.min_volume_mult:
            session_or_event = policy.session_type or policy.event_type or "policy"
            reasons.append(f"{session_or_event}:volume_fail:{vol_mult_5m:.2f}<{policy.min_volume_mult}")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)

        if policy.min_strength and strength < policy.min_strength:
            session_or_event = policy.session_type or policy.event_type or "policy"
            reasons.append(f"{session_or_event}:strength_fail:{strength:.2f}<{policy.min_strength}")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)

        # Phase 4: Check fast scalp lane permission (e.g., power hour blocks fast scalps)
        if lane_type == "fast_scalp_lane" and not policy.allow_fast_scalp:
            session_or_event = policy.session_type or policy.event_type or "policy"
            reasons.append(f"{session_or_event}:fast_scalp_rejected")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)

        # Apply policy adjustments
        size_mult *= float(policy.size_mult)
        min_hold = int(policy.min_hold_bars)
        if ctx:
            reasons.append("event_ctx:" + ",".join(sorted(ctx.keys())))

        # ---------------- NEWS SPIKE -------------------
        spike, sig = self.news_gate.has_symbol_spike(df1m_tail)
        if spike:
            adj = self.news_gate.adjustment_for(sig)
            min_hold += int(adj.require_hold_bars)
            size_mult *= float(adj.size_mult)
            reasons.append("news_spike:" + ";".join(sig.reasons))

        # ---------------- SENTIMENT --------------------
        if self.sentiment_gate is not None:
            try:
                banknifty_df = None  # can be wired from caller if available
                sentiment = self.sentiment_gate.analyze_sentiment(
                    nifty_df5=index_df5m,
                    banknifty_df5=banknifty_df,
                    breadth_data=None,
                    vix_level=None
                )
                if not self.sentiment_gate.should_trade_setup(best.setup_type, sentiment):
                    reasons.append(f"sentiment_block:{sentiment.sentiment_level.value}")
                    return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                sentiment_bias = self.sentiment_gate.get_setup_bias(best.setup_type, sentiment)
                size_mult *= sentiment_bias
                reasons.append(f"sentiment:{sentiment.sentiment_level.value}_{sentiment.market_trend.value}")
                reasons.append(f"sentiment_bias:{sentiment_bias:.2f}")
            except Exception as e:
                reasons.append(f"sentiment_error:{getattr(e, '__class__', type('E', (), {})).__name__}")

        # ---------------- ENTRY VALIDATION ------------
        entry_validation = self.quality_filters.get('entry_validation', {})

        # Volume confirmation
        if entry_validation.get('volume_confirmation', False) and df5m_tail is not None and len(df5m_tail) >= 5:
            try:
                volume_multiplier = entry_validation.get('volume_multiplier', 1.2)
                recent_volumes = df5m_tail["volume"].tail(10)
                avg_volume = recent_volumes.mean()
                current_volume = df5m_tail["volume"].iloc[-1]
                if current_volume < (avg_volume * volume_multiplier):
                    reasons.append(f"entry_validation_volume_fail:{current_volume:.0f}<{avg_volume*volume_multiplier:.0f}")
                    return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                else:
                    reasons.append(f"entry_validation_volume_pass:{current_volume:.0f}>={avg_volume*volume_multiplier:.0f}")
            except Exception as e:
                reasons.append(f"entry_validation_volume_error:{e.__class__.__name__}")

        # RSI band
        if entry_validation.get('rsi_min') is not None and entry_validation.get('rsi_max') is not None:
            try:
                if df5m_tail is not None and not df5m_tail.empty:
                    c = df5m_tail["close"].astype(float)
                    rsi14_last = float(_rsi(c, 14).iloc[-1])
                    rsi_min = entry_validation.get('rsi_min', 25)
                    rsi_max = entry_validation.get('rsi_max', 75)
                    if best.setup_type.endswith('_long') and (rsi14_last < rsi_min or rsi14_last > rsi_max):
                        reasons.append(f"entry_validation_rsi_long_fail:{rsi14_last:.1f}_not_in_{rsi_min}-{rsi_max}")
                        return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                    elif best.setup_type.endswith('_short') and (rsi14_last < rsi_min or rsi14_last > rsi_max):
                        reasons.append(f"entry_validation_rsi_short_fail:{rsi14_last:.1f}_not_in_{rsi_min}-{rsi_max}")
                        return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                    else:
                        reasons.append(f"entry_validation_rsi_pass:{rsi14_last:.1f}_in_{rsi_min}-{rsi_max}")
            except Exception as e:
                reasons.append(f"entry_validation_rsi_error:{e.__class__.__name__}")

        # Momentum threshold
        if entry_validation.get('momentum_threshold') is not None:
            try:
                if df5m_tail is not None and len(df5m_tail) >= 2:
                    current_close = df5m_tail["close"].iloc[-1]
                    prev_close = df5m_tail["close"].iloc[-2]
                    momentum = abs((current_close - prev_close) / prev_close)
                    momentum_threshold = entry_validation.get('momentum_threshold', 0.005)
                    if momentum < momentum_threshold:
                        reasons.append(f"entry_validation_momentum_fail:{momentum:.4f}<{momentum_threshold}")
                        return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                    else:
                        reasons.append(f"entry_validation_momentum_pass:{momentum:.4f}>={momentum_threshold}")
            except Exception as e:
                reasons.append(f"entry_validation_momentum_error:{e.__class__.__name__}")

        # Price action directionality
        if entry_validation.get('price_action_validation', False):
            try:
                if df5m_tail is not None and len(df5m_tail) >= 3:
                    last_3 = df5m_tail["close"].tail(3).values
                    if best.setup_type.endswith('_long'):
                        if last_3[-1] < last_3[-3]:
                            reasons.append("entry_validation_price_action_long_fail:declining_trend")
                            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                    elif best.setup_type.endswith('_short'):
                        if last_3[-1] > last_3[-3]:
                            reasons.append("entry_validation_price_action_short_fail:rising_trend")
                            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                    reasons.append("entry_validation_price_action_pass")
            except Exception as e:
                reasons.append(f"entry_validation_price_action_error:{e.__class__.__name__}")

        # ---------------- HCET sizing/hold tweaks ----------------
        if hc_ok:
            reasons.append("hcet:enabled")
            size_mult *= 1.15               # reward confluence, modestly
            min_hold = max(min_hold, 2)     # small confirmation hold
        else:
            # keep chop smaller by default
            if regime in {"choppy", "range"}:
                size_mult *= 0.8

        # ---------------- SETUP SEQUENCING (Enhancement 3) ----------------
        # Track this setup for future sequence analysis
        try:
            import pandas as pd
            current_time = pd.to_datetime(now) if hasattr(pd, 'to_datetime') else datetime.now()
            self._track_setup(symbol, current_time, best.setup_type)

            # Apply sequence-based confidence multiplier
            sequence_multiplier = self._get_sequence_multiplier(symbol, best.setup_type)
            if sequence_multiplier != 1.0:
                size_mult *= sequence_multiplier
                reasons.append(f"sequence_multiplier:{sequence_multiplier:.2f}")
        except Exception as e:
            reasons.append(f"sequence_error:{e.__class__.__name__}")

        # ---------------- FINAL ACCEPT ----------------
        return GateDecision(
            accept=True,
            reasons=reasons,
            setup_type=best.setup_type,  # DEPRECATED: for backward compatibility
            regime=regime,
            regime_conf=regime_confidence,
            size_mult=max(0.0, size_mult),
            min_hold_bars=max(0, min_hold),
            setup_candidates=setups,  # NEW: full setup candidates for structure system
            regime_diagnostics=regime_diagnostics,  # Phase 2: Multi-TF regime diagnostics
        )

    def _get_strategy_momentum_threshold(self, setup_type: Optional[str]) -> float:
        """Get strategy-specific momentum consolidation threshold"""
        if not setup_type:
            return self.quality_filters.get('momentum_consolidation_threshold', 1.0)

        strategy_filters = self.quality_filters.get('strategy_momentum_filters', {})
        for _, config in strategy_filters.items():
            if setup_type in config.get('strategies', []):
                return config['momentum_consolidation_threshold']  # KeyError if missing
        return self.quality_filters.get('momentum_consolidation_threshold', 1.0)
    