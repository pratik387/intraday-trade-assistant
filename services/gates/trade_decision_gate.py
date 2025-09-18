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
    setup_type: Optional[SetupType] = None
    regime: Optional[str] = None
    regime_conf: float = 0.0
    size_mult: float = 1.0
    min_hold_bars: int = 0
    matched_rule: Optional[str] = None  # if you use rule miner/meta later
    p_breakout: Optional[float] = None  # placeholder for meta-prob models


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
    }


def _is_fade(setup: SetupType) -> bool:
    return setup in {
        "failure_fade_long", 
        "failure_fade_short",
        "volume_spike_reversal_long",
        "volume_spike_reversal_short",
        "range_rejection_long",
        "range_rejection_short",
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
        self.quality_filters = quality_filters or {}

    # ------------------------------ Public API ------------------------------
    def evaluate(
        self,
        *,
        symbol: str,
        now,
        df1m_tail: pd.DataFrame,
        df5m_tail: pd.DataFrame,
        index_df5m: pd.DataFrame,
        levels: Optional[dict],
        plan: Optional[dict] = None,  # Add plan parameter for quality filters
        features: Optional[dict] = None,  # Add features for rank score
    ) -> GateDecision:
        reasons: List[str] = []

        # FAST QUALITY FILTERS - Applied first to save computation
        # Remove duplicate rank_score filter - already filtered at ranker/screener level
        # This was causing additional rejections after symbols already passed ranking

        
        # Remove duplicate structural_rr filter - already filtered at planner level
        # Time window filter (configurable)
        import datetime
        if hasattr(now, 'hour') and hasattr(now, 'minute'):
            minute_of_day = now.hour * 60 + now.minute

            # Get time windows from config (with fallback to restrictive defaults)
            time_windows = self.quality_filters.get('time_windows', {})

            # Parse time strings to minutes of day
            def time_to_minutes(time_str, default_minutes):
                try:
                    if isinstance(time_str, str) and ':' in time_str:
                        hour, minute = map(int, time_str.split(':'))
                        return hour * 60 + minute
                    return default_minutes
                except:
                    return default_minutes

            # Get configurable time windows (defaults match original hardcoded values)
            morning_start = time_to_minutes(time_windows.get('morning_start', '10:30'), 630)
            morning_end = time_to_minutes(time_windows.get('morning_end', '12:30'), 750)
            afternoon_start = time_to_minutes(time_windows.get('afternoon_start', '14:15'), 855)
            afternoon_end = time_to_minutes(time_windows.get('afternoon_end', '15:00'), 900)

            # Check if current time is within allowed trading windows
            in_morning_window = morning_start <= minute_of_day <= morning_end
            in_afternoon_window = afternoon_start <= minute_of_day <= afternoon_end

            if not (in_morning_window or in_afternoon_window):
                return GateDecision(accept=False, reasons=[f"time_window_block:{minute_of_day}_outside_{morning_start}-{morning_end}_or_{afternoon_start}-{afternoon_end}"])

        # EVIDENCE-BASED PATTERN FILTERS (from reverse engineering analysis)
        # These patterns showed 76% and 63% success rates respectively
        pattern_filters_enabled = self.quality_filters.get('pattern_filters_enabled', True)

        # Skip feasible checks for symbols with insufficient bars (early morning edge case)
        if df5m_tail is None or len(df5m_tail) < 10:
            reasons.append(f"skip_feasible_checks:insufficient_bars_{len(df5m_tail) if df5m_tail is not None else 0}<10")
            pattern_filters_enabled = False  # Skip pattern filters for early morning trades

        if pattern_filters_enabled and df5m_tail is not None and not df5m_tail.empty:
            pattern_reasons = []
            pattern_passed = True

            # MOMENTUM_CONSOLIDATION filter (76% success rate)
            # Look for -1% to +1% price momentum in 15-minute window (3 bars @ 5min each)
            momentum_filter_enabled = self.quality_filters.get('momentum_consolidation_enabled', True)
            if momentum_filter_enabled and len(df5m_tail) >= 3:
                try:
                    # Calculate 15-minute momentum using last 3 bars
                    close_3_bars_ago = df5m_tail["close"].iloc[-4] if len(df5m_tail) >= 4 else df5m_tail["close"].iloc[0]
                    current_close = df5m_tail["close"].iloc[-1]
                    momentum_15min = ((current_close - close_3_bars_ago) / close_3_bars_ago) * 100

                    # Get strategy-specific momentum threshold (setup_type unknown at this stage)
                    momentum_threshold = self._get_strategy_momentum_threshold(None)

                    if abs(momentum_15min) > momentum_threshold:
                        pattern_reasons.append(f"momentum_consolidation_fail:{momentum_15min:.2f}%>{momentum_threshold}%")
                        pattern_passed = False
                    else:
                        pattern_reasons.append(f"momentum_consolidation_pass:{momentum_15min:.2f}%<={momentum_threshold}%")

                except Exception as e:
                    pattern_reasons.append(f"momentum_consolidation_error:{e.__class__.__name__}")

            # RANGE_COMPRESSION filter (63% success rate)
            # Look for ATR ratio <= 0.8 (current ATR vs 20-period average ATR)
            compression_filter_enabled = self.quality_filters.get('range_compression_enabled', True)
            if compression_filter_enabled and len(df5m_tail) >= 20:
                try:
                    # Calculate ATR using True Range over 14 periods
                    high = df5m_tail["high"]
                    low = df5m_tail["low"]
                    close_prev = df5m_tail["close"].shift(1)

                    # True Range calculation
                    tr1 = high - low
                    tr2 = abs(high - close_prev)
                    tr3 = abs(low - close_prev)

                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                    # ATR calculation (14-period)
                    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
                    current_atr = atr_14.iloc[-1]

                    # Average ATR over last 20 periods
                    avg_atr_20 = atr_14.rolling(window=20, min_periods=20).mean().iloc[-1]

                    if pd.notna(current_atr) and pd.notna(avg_atr_20) and avg_atr_20 > 0:
                        atr_ratio = current_atr / avg_atr_20
                        compression_threshold = self.quality_filters.get('range_compression_threshold', 0.8)  # 0.8 default

                        if atr_ratio > compression_threshold:
                            pattern_reasons.append(f"range_compression_fail:{atr_ratio:.3f}>{compression_threshold}")
                            pattern_passed = False
                        else:
                            pattern_reasons.append(f"range_compression_pass:{atr_ratio:.3f}<={compression_threshold}")
                    else:
                        pattern_reasons.append("range_compression_insufficient_data")

                except Exception as e:
                    pattern_reasons.append(f"range_compression_error:{e.__class__.__name__}")

            # Reject if any pattern filter failed
            if not pattern_passed:
                reasons.extend(pattern_reasons)
                return GateDecision(accept=False, reasons=reasons)
            else:
                reasons.extend(pattern_reasons)  # Add pass reasons for logging

        # Structure: propose setups from closed 5m bars
        setups = self.structure.detect_setups(symbol, df5m_tail, levels)
        if not setups:
            return GateDecision(accept=False, reasons=["no_structure_event"])
        # pick the strongest for now (you can inject a ranker later)
        setups.sort(key=lambda s: s.strength, reverse=True)
        best = setups[0]
        reasons.extend([f"structure:{r}" for r in best.reasons])

        # Regime: classify index and check permissions with evidence
        df_for_regime = index_df5m if index_df5m is not None and not index_df5m.empty else df5m_tail
        regime = self.regime_gate.compute_regime(df_for_regime)

        # --- Evidence for regime gate ---
        strength = _safe_float(best.strength, 0.0)

        # last closed 5m row (Series) NA-safe
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
            
        try:
            if df5m_tail is not None and not df5m_tail.empty:
                c = df5m_tail["close"].astype(float)
                rsi14_last = float(_rsi(c, 14).iloc[-1])
                if best.setup_type in ("breakout_long", "vwap_reclaim_long", "squeeze_release_long") and rsi14_last > 65.0:
                    reasons.append(f"rsi_block_long:{rsi14_last:.1f}>65")
                    return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                if best.setup_type in ("breakout_short", "vwap_lose_short", "squeeze_release_short") and rsi14_last < 35.0:
                    reasons.append(f"rsi_block_short:{rsi14_last:.1f}<35")
                    return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
        except Exception:
            # RSI is a guardrail; if it fails, do not block the trade solely on this.
            pass

        if not self.regime_gate.allow_setup(best.setup_type, regime, strength, adx_5m, vol_mult_5m):
            reasons.append(f"regime_block:{regime}[str={strength:.2f},adx={adx_5m:.2f},volx={vol_mult_5m:.2f}]")
            return GateDecision(
                accept=False,
                reasons=reasons,
                setup_type=best.setup_type,
                regime=regime,
            )

        size_mult = 1.0
        if hasattr(self.regime_gate, "size_multiplier"):
            try:
                size_mult *= float(self.regime_gate.size_multiplier(regime))
            except Exception:
                pass
        reasons.append(f"regime:{regime}")

        # Event policy: macro/expiry/symbol windows → allow set & sizing/hold
        policy, ctx = self.event_gate.decide_policy(now, symbol)
        # map setup to breakout/fade permission
        if _is_breakout(best.setup_type) and not policy.allow_breakout:
            reasons.append("event_block:breakout")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
        if _is_fade(best.setup_type) and not policy.allow_fade:
            reasons.append("event_block:fade")
            return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
        size_mult *= float(policy.size_mult)
        min_hold = int(policy.min_hold_bars)
        if ctx:
            reasons.append("event_ctx:" + ",".join(sorted(ctx.keys())))

        # News spike adjustments from last closed 1m bar
        spike, sig = self.news_gate.has_symbol_spike(df1m_tail)
        if spike:
            adj = self.news_gate.adjustment_for(sig)
            min_hold += int(adj.require_hold_bars)
            size_mult *= float(adj.size_mult)
            reasons.append("news_spike:" + ";".join(sig.reasons))

        # Market Sentiment Analysis & Bias Application
        if self.sentiment_gate is not None:
            try:
                # Get BankNifty data if available (would need to be passed in)
                banknifty_df = None  # TODO: Pass BankNifty data from screener
                
                # Analyze market sentiment
                sentiment = self.sentiment_gate.analyze_sentiment(
                    nifty_df5=index_df5m,
                    banknifty_df5=banknifty_df,
                    breadth_data=None,  # TODO: Add breadth data integration
                    vix_level=None      # TODO: Add VIX data integration
                )
                
                # Apply sentiment filter - reject if setup not suitable for current sentiment
                if not self.sentiment_gate.should_trade_setup(best.setup_type, sentiment):
                    reasons.append(f"sentiment_block:{sentiment.sentiment_level.value}")
                    return GateDecision(accept=False, reasons=reasons, setup_type=best.setup_type, regime=regime)
                
                # Apply sentiment bias to size multiplier
                sentiment_bias = self.sentiment_gate.get_setup_bias(best.setup_type, sentiment)
                size_mult *= sentiment_bias
                
                # Add sentiment context to reasons
                reasons.append(f"sentiment:{sentiment.sentiment_level.value}_{sentiment.market_trend.value}")
                reasons.append(f"sentiment_bias:{sentiment_bias:.2f}")
                
            except Exception as e:
                # Sentiment analysis is enhancement; don't fail trades if it breaks
                if hasattr(e, '__class__'):
                    reasons.append(f"sentiment_error:{e.__class__.__name__}")
                else:
                    reasons.append("sentiment_error:unknown")

        # Accept with accumulated adjustments
        return GateDecision(
            accept=True,
            reasons=reasons,
            setup_type=best.setup_type,
            regime=regime,
            size_mult=max(0.0, size_mult),
            min_hold_bars=max(0, min_hold),
        )

    def _get_strategy_momentum_threshold(self, setup_type: Optional[str]) -> float:
        """Get strategy-specific momentum consolidation threshold"""
        if not setup_type:
            return self.quality_filters.get('momentum_consolidation_threshold', 1.0)

        strategy_filters = self.quality_filters.get('strategy_momentum_filters', {})

        # Check each strategy category
        for category, config in strategy_filters.items():
            if setup_type in config.get('strategies', []):
                return config.get('momentum_consolidation_threshold', 1.0)

        # Fallback to default if setup_type not found in any category
        return self.quality_filters.get('momentum_consolidation_threshold', 1.0)
