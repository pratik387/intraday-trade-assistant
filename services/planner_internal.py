# services/planner_internal.py
"""
Planner (intraday, 5m)
----------------------
Builds a config-driven intraday trade plan for a symbol using 5-minute OHLCV.

Key principles:
- **No code defaults**: every tunable comes from entry_config.json / exit_config.json.
- **Naive IST** timestamps for all DataFrames (see utils.time_util.ensure_naive_ist_index).
- **Observability**: rich logs; full stacktraces via logger.exception on errors.

High-level flow:
1) Normalize bars, slice latest session, compute core features (VWAP, EMA20/50, RSI, ADX, MACD).
2) Compute context (ATR, Opening Range, previous-day levels, gap).
3) Classify regime (simple CHOP vs trend heuristic).
4) Choose a structure-led strategy candidate (ORH/ORL/VWAP/PDH/PDL logic).
5) Compose exits & sizing strictly from config (targets by RR; entry zone vs ATR).
6) Apply feasibility caps and “late entry / intraday gate” soft size multipliers.

Outputs a serializable `plan` dict. No side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union

from config.logging_config import get_agent_logger, get_planning_logger
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index
from utils.time_util import _minute_of_day, _parse_hhmm_to_md
from services.gates.trade_decision_gate import SetupCandidate
from services.indicators.adx import calculate_ema, calculate_rsi, calculate_macd, calculate_adx, calculate_atr
from utils.level_utils import get_previous_day_levels
import datetime

logger = get_agent_logger()

# ----------------------------
# Helpers (math/indicators)
# ----------------------------

def get_date_range_from_df(df: pd.DataFrame) -> Tuple[str, str]:
    idx = pd.to_datetime(df.index)
    return (str(idx.min().date()), str(idx.max().date()))

def _ensure_session_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure naive IST index, add 'date' column, return only the latest session slice."""
    d = ensure_naive_ist_index(df.copy())
    if "date" not in d.columns:
        d["date"] = d.index.normalize()
    latest_day = d["date"].iloc[-1]
    return d[d["date"] == latest_day]

def _opening_range(df_sess: pd.DataFrame, orb_minutes: int) -> Tuple[float, float, pd.Timestamp]:
    """Opening Range using the first `orb_minutes` of the *current session*."""
    start_ts = df_sess.index.min()
    or_end = start_ts + pd.Timedelta(minutes=orb_minutes)
    or_df = df_sess.loc[(df_sess.index >= start_ts) & (df_sess.index < or_end)]
    return float(or_df["high"].max()), float(or_df["low"].min()), or_end

def _session_vwap(df_sess: pd.DataFrame) -> pd.Series:
    tp = (df_sess["high"] + df_sess["low"] + df_sess["close"]) / 3.0
    cum_pv = (tp * df_sess["volume"]).cumsum()
    cum_v = df_sess["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def _prev_day_levels(
    daily_df: Optional[pd.DataFrame],
    session_date: Optional[datetime.date],
) -> Dict[str, float]:
    """
    Return PDH/PDL/PDC from the last completed trading day strictly before session_date.

    This now uses the centralized implementation from utils.level_utils to avoid code duplication.
    """
    return get_previous_day_levels(daily_df, session_date, fallback_df=None, enable_fallback=False)


def _choppiness_index(df_sess: pd.DataFrame, lookback: int) -> float:
    """CHOP = 100 * log10(sum(TR) / (max(H)-min(L))) / log10(n)"""
    try:
        h, l, c = df_sess["high"], df_sess["low"], df_sess["close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        window = tr.tail(lookback)
        if len(window) < lookback or (window.sum() == 0):
            return 100.0
        hh = h.tail(lookback).max()
        ll = l.tail(lookback).min()
        denom = np.log10(lookback)
        numer = np.log10(window.sum() / max(hh - ll, 1e-9))
        return float(100.0 * numer / denom)
    except Exception as e:
        logger.exception(f"planner.choppiness error: {e}")
        return 100.0

def _pivot_swing_low(df_sess: pd.DataFrame, lookback: int) -> float:
    lows = df_sess["low"].tail(lookback + 2)
    return float(lows.min())

def _pivot_swing_high(df_sess: pd.DataFrame, lookback: int) -> float:
    highs = df_sess["high"].tail(lookback + 2)
    return float(highs.max())

def _detect_daily_trend(daily_df: Optional[pd.DataFrame], ema_period: int = 20) -> str:
    """Detect daily timeframe trend direction using EMA slope."""
    try:
        if daily_df is None or daily_df.empty or len(daily_df) < ema_period:
            return "neutral"

        daily_close = daily_df["close"].astype(float)
        daily_ema = calculate_ema(daily_close, ema_period)

        if len(daily_ema) < 3:
            return "neutral"

        # Check EMA slope over last 3 days
        recent_ema = daily_ema.tail(3)
        slope = (recent_ema.iloc[-1] - recent_ema.iloc[0]) / max(abs(recent_ema.iloc[0]), 1e-9)

        # Also check if price is above/below EMA
        price_vs_ema = daily_close.iloc[-1] > daily_ema.iloc[-1]

        if slope > 0.02 and price_vs_ema:  # 2% positive slope and above EMA
            return "up"
        elif slope < -0.02 and not price_vs_ema:  # 2% negative slope and below EMA
            return "down"
        else:
            return "neutral"

    except Exception as e:
        logger.exception(f"daily_trend detection error: {e}")
        return "neutral"

# ----------------------------
# Config dataclass + loader
# ----------------------------

@dataclass
class PlannerConfig:
    # session/time (context)
    session_open_hhmm: str
    session_close_hhmm: str
    orb_minutes: int
    bar_minutes: int

    # volatility/range context
    atr_period: int
    choppiness_lookback: int
    choppiness_high: float
    choppiness_low: float
    max_gap_pct_for_trend: float

    # entries
    entry_zone_atr_frac: float
    vwap_reclaim_min_bars_above: int

    # stops/targets (from exit_config.json)
    sl_atr_mult: float
    sl_below_swing_ticks: float
    t1_rr: float
    t2_rr: float
    trail_to: str

    # sizing/costs (from entry_config.json)
    risk_per_trade_rupees: float
    fees_slippage_bps: float

    # guardrails (context)
    enable_lunch_pause: bool
    lunch_start: str
    lunch_end: str

    # extras (nested knobs—optional but passed through)
    _extras: Dict[str, Any]


def _load_planner_config(user_overrides: Optional[Dict[str, Any]] = None) -> PlannerConfig:
    """
    Load all required knobs from JSON (and optional user overrides), no code defaults.
    """
    config = load_filters()

    # Required keys in entry/exit configs (fail fast if missing)
    required_entry = [
        "session_open_hhmm", "session_close_hhmm", "orb_minutes",
        "bar_5m_span_minutes",
        "planner_atr_period", "planner_choppiness_lookback",
        "planner_choppiness_high", "planner_choppiness_low",
        "planner_max_gap_pct_for_trend",
        "planner_entry_zone_atr_frac", "planner_vwap_reclaim_min_bars_above",
        "risk_per_trade_rupees", "fees_slippage_bps",
        "enable_lunch_pause", "lunch_start", "lunch_end"
    ]
    missing = [k for k in required_entry if k not in config]
    if missing:
        raise KeyError(f"entry_config.json missing keys for planner: {missing}")

    required_exit = ["sl_atr_mult", "sl_below_swing_ticks", "t1_rr", "t2_rr", "trail_to"]
    missing2 = [k for k in required_exit if k not in config]
    if missing2:
        raise KeyError(f"exit_config.json missing keys for planner: {missing2}")

    # Extras are optional but must be explicitly provided as dicts if needed
    extras = {}

    # Each extra section defaults to empty dict if not provided
    for key in ["intraday_gate", "late_entry_penalty", "planner_precision", "acceptance", "quality_filters"]:
        extras[key] = config.get(key, {})  # Empty dict for unused sections

    # Allow user overrides to update extras ONLY (strict knobs stay JSON-driven)
    if isinstance(user_overrides, dict):
        for k in extras.keys():
            if isinstance(user_overrides.get(k), dict):
                extras[k].update(user_overrides[k])

    cfg = PlannerConfig(
        session_open_hhmm=str(config["session_open_hhmm"]),
        session_close_hhmm=str(config["session_close_hhmm"]),
        orb_minutes=int(config["orb_minutes"]),
        bar_minutes=int(config["bar_5m_span_minutes"]),

        atr_period=int(config["planner_atr_period"]),
        choppiness_lookback=int(config["planner_choppiness_lookback"]),
        choppiness_high=float(config["planner_choppiness_high"]),
        choppiness_low=float(config["planner_choppiness_low"]),
        max_gap_pct_for_trend=float(config["planner_max_gap_pct_for_trend"]),

        entry_zone_atr_frac=float(config["planner_entry_zone_atr_frac"]),
        vwap_reclaim_min_bars_above=int(config["planner_vwap_reclaim_min_bars_above"]),

        sl_atr_mult=float(config["sl_atr_mult"]),
        sl_below_swing_ticks=float(config["sl_below_swing_ticks"]),
        t1_rr=float(config["t1_rr"]),
        t2_rr=float(config["t2_rr"]),
        trail_to=str(config["trail_to"]),

        risk_per_trade_rupees=float(config["risk_per_trade_rupees"]),
        fees_slippage_bps=float(config["fees_slippage_bps"]),

        enable_lunch_pause=bool(config["enable_lunch_pause"]),
        lunch_start=str(config["lunch_start"]),
        lunch_end=str(config["lunch_end"]),

        _extras=extras,
    )
    return cfg

# ----------------------------
# Regime + strategy selection
# ----------------------------

def _regime(df_sess: pd.DataFrame, cfg: PlannerConfig) -> str:
    """Simple regime classifier for planner context (independent of gates)."""
    chop = _choppiness_index(df_sess, cfg.choppiness_lookback)
    ema20 = calculate_ema(df_sess["close"], 20)
    ema50 = calculate_ema(df_sess["close"], 50)
    slope = (ema20.iloc[-1] - ema20.iloc[-5]) / max(abs(ema20.iloc[-5]), 1e-9)
    if chop <= cfg.choppiness_low and ema20.iloc[-1] > ema50.iloc[-1] and slope > 0:
        return "trend_up"
    if chop <= cfg.choppiness_low and ema20.iloc[-1] < ema50.iloc[-1] and slope < 0:
        return "trend_down"
    if chop >= cfg.choppiness_high:
        return "choppy"
    return "range"

def _plan_strategy_for_candidate(
    candidate: SetupCandidate,
    df_sess: pd.DataFrame,
    cfg: PlannerConfig,
    regime: str,
    orh: float,
    orl: float,
    or_end: pd.Timestamp,
    vwap: pd.Series,
    pd_levels: Dict[str, float],
    close: float,
    ema20: pd.Series,
    ema50: pd.Series,
    index_df5m: Optional[pd.DataFrame] = None,
    daily_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    NEW APPROACH: Plan trading strategy based on SetupCandidate from structure system.

    This function focuses ONLY on trade planning (stops, targets, sizing) and does NOT
    contain setup detection logic. The setup detection is handled by the structure system.
    """
    setup_type = str(candidate.setup_type)  # Convert SetupType to string
    strength = candidate.strength
    reasons = candidate.reasons

    # Determine bias from setup type
    bias = "long" if "_long" in setup_type else "short" if "_short" in setup_type else "flat"

    if bias == "flat":
        logger.warning(f"Unknown setup bias for {setup_type}, defaulting to flat")
        return {
            "name": "unknown_bias", "bias": "flat", "entry_trigger": "wait",
            "structure_stop": np.nan,
            "context": {"reason": [f"unknown_bias:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Calculate structure-based entry price first
    current_close = float(df_sess.iloc[-1]["close"])
    vwap_current = float(df_sess.iloc[-1]["vwap"]) if "vwap" in df_sess.columns and not pd.isna(df_sess.iloc[-1]["vwap"]) else None

    entry_price = _calculate_structure_entry(
        setup_type=setup_type,
        bias=bias,
        current_close=current_close,
        orh=orh,
        orl=orl,
        pdh=pd_levels.get("PDH", np.nan),
        pdl=pd_levels.get("PDL", np.nan),
        vwap_current=vwap_current,
        atr=calculate_atr(df_sess.tail(200), period=cfg.atr_period)
    )

    # Calculate ATR for stop calculation
    atr_for_stop = calculate_atr(df_sess.tail(200), period=cfg.atr_period)

    # Select stop loss strategy based on setup type category (using actual entry price)
    stop_structure = _calculate_structure_stop(setup_type, df_sess, orh, orl, vwap, entry_price, bias, ema20, atr_for_stop)

    # Generate entry trigger message
    entry_trigger = _generate_entry_trigger(setup_type, reasons, strength)

    # Create context with structure system information
    context = {
        "reason": reasons + [f"setup:{setup_type}", f"strength:{strength:.2f}", f"regime:{regime}"],
        "levels": {"ORH": orh, "ORL": orl, **pd_levels},
        "structure_strength": strength,
        "structure_reasons": reasons
    }

    return {
        "name": setup_type,
        "bias": bias,
        "entry_trigger": entry_trigger,
        "structure_stop": float(stop_structure),
        "context": context
    }

def _calculate_structure_stop(
    setup_type: str,
    df_sess: pd.DataFrame,
    orh: float,
    orl: float,
    vwap: pd.Series,
    entry_price: float,
    bias: str,
    ema20: Optional[pd.Series] = None,
    atr: float = None
) -> float:
    """
    PHASE 1 STOP LOSS FIX (Nov 2025): Uses entry-relative stops at 2.25× ATR distance.

    Analysis of 36 hard_sl exits (42.4% of all trades, -Rs.17,743) using 1m spike data showed:
    - 91.7% of SL hits AVOIDABLE with 1.5R stops (33/36 trades saved!)
    - 36.1% were FALSE STOP-OUTS - price reversed after SL hit
    - 22.2% hit T1 before SL - these could have been WINNING trades

    Previous stop: 1.5× ATR (1.0R)
    New stop: 2.25× ATR (1.5R) - 50% wider to avoid NSE intraday noise

    Expected impact: Save 33/36 SL trades, convert ~8 to T1 winners, gain Rs.15,000-18,000

    See: tools/analyze_stop_loss_problem.py for full analysis
    """
    # PHASE 1 FIX: Widen stops from 1.5× ATR to 2.25× ATR (1.5R vs 1.0R)
    if atr is None or np.isnan(atr):
        # Fallback: 0.75% of entry price if ATR unavailable (also widened from 0.5%)
        atr = entry_price * 0.0075

    if bias == "long":
        # For longs: stop 2.25× ATR below entry price (was 1.5× ATR)
        return entry_price - (atr * 2.25)
    elif bias == "short":
        # For shorts: stop 2.25× ATR above entry price (was 1.5× ATR)
        return entry_price + (atr * 2.25)

    return np.nan

def _generate_entry_trigger(setup_type: str, reasons: List[str], strength: float) -> str:
    """
    Generate human-readable entry trigger message based on setup type and structure reasons.
    """
    base_triggers = {
        "orb_breakout": "opening range breakout with volume expansion",
        "vwap_reclaim": "VWAP reclaim with volume confirmation",
        "vwap_mean_reversion": "mean reversion approach to VWAP",
        "trend_continuation": "trend continuation pattern",
        "trend_pullback": "pullback completion in established trend",
        "range_rejection": "rejection at range boundary",
        "support_bounce": "rejection at key support level",
        "resistance_bounce": "rejection at key resistance level",
        "momentum_breakout": "momentum breakout with volume surge",
        "squeeze_release": "volatility expansion breakout",
        "flag_continuation": "flag pattern breakout",
        "volume_spike_reversal": "reversal after climactic volume",
        "gap_fill": "gap fill approach with momentum weakness",
        "range_deviation": "range boundary deviation reversal",
        "order_block": "institutional order block reaction",
        "fair_value_gap": "fair value gap completion",
        "liquidity_sweep": "post-sweep reversal",
        "break_of_structure": "break of structure continuation"
    }

    # Find matching trigger
    trigger = "structure-based entry signal"
    for key, desc in base_triggers.items():
        if key in setup_type:
            trigger = desc
            break

    # Add strength and reason context
    if strength > 0.8:
        trigger += " (high confidence)"
    elif strength > 0.6:
        trigger += " (medium confidence)"

    if reasons:
        trigger += f" + {', '.join(reasons[:2])}"  # Add first 2 reasons

    return trigger

# STRATEGY SELECTION
# Strategy selection based on structure detection results from the professional structure system.

def _strategy_selector(
    df_sess: pd.DataFrame,
    cfg: PlannerConfig,
    regime: str,
    orh: float, orl: float, or_end: pd.Timestamp,
    vwap: pd.Series,
    pd_levels: Dict[str, float],
    setup_candidates: List[SetupCandidate],
    index_df5m: Optional[pd.DataFrame] = None,
    daily_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Pick a single best structure-led setup candidate."""
    if not setup_candidates:
        raise ValueError("setup_candidates is required for strategy selection")

    last = df_sess.iloc[-1]
    close = float(last["close"])

    ema20 = calculate_ema(df_sess["close"], 20)
    ema50 = calculate_ema(df_sess["close"], 50)

    # Select best setup candidate based on strength
    best_candidate = max(setup_candidates, key=lambda c: c.strength)
    return _plan_strategy_for_candidate(
        best_candidate, df_sess, cfg, regime, orh, orl, or_end,
        vwap, pd_levels, close, ema20, ema50, index_df5m, daily_df
    )

# ----------------------------
# Exits & sizing
# ----------------------------

def _compose_exits_and_size(
    price: float, bias: str, atr: float, structure_stop: float, cfg: PlannerConfig, qty_scale: float,
    cap_segment: str = "unknown"  # Priority 3: Cap-aware sizing
) -> Dict[str, Any]:
    """
    Compose hard stop, targets by RR, and position size. Strictly config-driven.
    Also enforces reasonable risk-per-share floors via cfg._extras['planner_precision'].

    Priority 3: Cap-aware sizing applies Van Tharp position sizing principles:
    - Large-caps: 1.2x size (lower volatility)
    - Mid-caps: 1.0x size (baseline)
    - Small-caps: 0.6x size + 1.3x wider stops (higher volatility)
    """
    try:
        pp = cfg._extras.get("planner_precision", {}) if hasattr(cfg, "_extras") else {}
        min_tick = float(pp.get("min_tick", 0.05))  # allowed to default within EXTRAS only

        if np.isnan(structure_stop):
            return {"eligible": False, "reason": "no_structure_stop", "targets": [], "qty": 0, "notional": 0.0}

        # ATR fallback if missing (warn + coarse estimate)
        fallback_zone = False
        if np.isnan(atr) or atr == 0.0:
            logger.warning(f"planner: ATR unavailable for price={price}, using fallback")
            atr = price * 0.005
            fallback_zone = True

        # Primary stop from structure, buffered by volatility (ATR multiple)
        # INSTITUTIONAL FIX: Use TIGHTER stop (min for long, max for short) for precision edge
        vol_stop = price - cfg.sl_atr_mult * atr if bias == "long" else price + cfg.sl_atr_mult * atr
        if bias == "long":
            # For LONG: SL below entry - use HIGHER value (closer to entry) = TIGHTER stop
            structure_sl = structure_stop - cfg.sl_below_swing_ticks
            hard_sl = max(structure_sl, vol_stop)  # Takes closer SL to entry
            rps = max(price - hard_sl, 0.0)
        else:
            # For SHORT: SL above entry - use LOWER value (closer to entry) = TIGHTER stop
            structure_sl = structure_stop + cfg.sl_below_swing_ticks
            hard_sl = min(structure_sl, vol_stop)  # Takes closer SL to entry
            rps = max(hard_sl - price, 0.0)

        # Floors for RPS to avoid too-tight stops
        min_rps_bpct = float(pp.get("min_rps_bpct", 0.45))  # % of price
        atr_rps_mult = float(pp.get("atr_rps_mult", 0.35))
        floor_by_px  = price * (min_rps_bpct / 100.0)
        floor_by_atr = (atr or 0.0) * atr_rps_mult
        rps_floor    = max(floor_by_px, floor_by_atr, 0.0)
        if rps < rps_floor:
            if bias == "long":
                hard_sl = price - rps_floor
            else:
                hard_sl = price + rps_floor
            rps = rps_floor

        # Tick-size guard
        if rps < min_tick:
            return {
                "eligible": False, "reason": f"risk_below_tick<{min_tick}",
                "risk_per_share": round(rps, 4), "hard_sl": round(hard_sl, 2),
                "targets": [], "trail": cfg.trail_to, "entry_zone": None, "qty": 0, "notional": 0.0
            }

        # Volatility-adjusted sizing
        volatility_multiplier = 1.0
        if hasattr(cfg, '_extras') and cfg._extras:
            volatility_config = cfg._extras.get('volatility_sizing', {})
            if volatility_config.get('enabled', False):
                try:
                    # Calculate volatility based on ATR as percentage of price
                    price_atr_ratio = (atr / price) * 100 if price > 0 and not np.isnan(atr) else 1.0

                    # Define volatility thresholds - must be explicitly configured
                    # KeyError if missing volatility parameters
                    low_vol_threshold = volatility_config['low_volatility_threshold']
                    high_vol_threshold = volatility_config['high_volatility_threshold']

                    # Volatility multipliers - must be explicitly configured
                    low_vol_mult = volatility_config['low_volatility_multiplier']
                    high_vol_mult = volatility_config['high_volatility_multiplier']
                    normal_vol_mult = volatility_config['normal_volatility_multiplier']

                    # Apply volatility-based adjustment
                    if price_atr_ratio < low_vol_threshold:
                        volatility_multiplier = low_vol_mult  # Low volatility: increase position size
                    elif price_atr_ratio > high_vol_threshold:
                        volatility_multiplier = high_vol_mult  # High volatility: decrease position size
                    else:
                        volatility_multiplier = normal_vol_mult  # Normal volatility

                    # Apply limits to prevent extreme sizing
                    max_adjustment = volatility_config.get('max_size_adjustment', 2.0)
                    min_adjustment = volatility_config.get('min_size_adjustment', 0.5)
                    volatility_multiplier = max(min_adjustment, min(max_adjustment, volatility_multiplier))

                except Exception:
                    # If volatility calculation fails, use default multiplier
                    volatility_multiplier = 1.0

        # === PRIORITY 3: CAP-AWARE SIZING (Van Tharp evidence) ===
        cap_size_mult = 1.0
        cap_sl_mult = 1.0

        if cap_segment != "unknown":
            cap_risk_cfg = load_filters().get("cap_risk_adjustments", {})
            if cap_risk_cfg.get("enabled", False):
                seg_cfg = cap_risk_cfg.get(cap_segment, {})
                cap_size_mult = seg_cfg.get("size_multiplier", 1.0)
                cap_sl_mult = seg_cfg.get("sl_atr_multiplier", 1.0)

                # Apply cap-specific SL multiplier (small-caps get wider stops)
                if cap_sl_mult != 1.0:
                    # Recalculate hard_sl with cap-specific multiplier
                    vol_stop_cap = price - (cfg.sl_atr_mult * cap_sl_mult * atr) if bias == "long" else price + (cfg.sl_atr_mult * cap_sl_mult * atr)
                    if bias == "long":
                        structure_sl = structure_stop - cfg.sl_below_swing_ticks
                        hard_sl = max(structure_sl, vol_stop_cap)
                        rps = max(price - hard_sl, 0.0)
                    else:
                        structure_sl = structure_stop + cfg.sl_below_swing_ticks
                        hard_sl = min(structure_sl, vol_stop_cap)
                        rps = max(hard_sl - price, 0.0)

                    logger.debug(f"CAP_SIZING: {cap_segment} cap_sl_mult={cap_sl_mult:.2f} → rps={rps:.4f}")

        # Sizing with volatility + cap adjustments
        base_qty = max(int(cfg.risk_per_trade_rupees // rps), 0)
        qty = max(int(base_qty * qty_scale * volatility_multiplier * cap_size_mult), 0)
        notional = qty * price

        if cap_size_mult != 1.0:
            logger.debug(f"CAP_SIZING: {cap_segment} size_mult={cap_size_mult:.2f} → qty={qty}")

        # Targets by RR
        t1 = price + (cfg.t1_rr * rps) if bias == "long" else price - (cfg.t1_rr * rps)
        t2 = price + (cfg.t2_rr * rps) if bias == "long" else price - (cfg.t2_rr * rps)

        # Entry zone width (ATR-based unless fallback)
        if fallback_zone:
            entry_zone = [round(price - 0.01, 2), round(price, 2)]
        else:
            entry_width = max(atr * cfg.entry_zone_atr_frac, 0.01)
            # For range strategies using current price is appropriate
            entry_zone = [round(price - entry_width, 2), round(price + entry_width, 2)]

        return {
            "eligible": qty > 0,
            "risk_per_share": round(rps, 4),
            "qty": int(qty),
            "notional": round(notional, 2),
            "hard_sl": round(hard_sl, 2),
            "targets": [
                {"name": "T1", "level": round(t1, 2), "rr": cfg.t1_rr, "action": "book_30%"},
                {"name": "T2", "level": round(t2, 2), "rr": cfg.t2_rr, "action": "trail_rest"},
            ],
            "trail": cfg.trail_to,
            "entry_zone": entry_zone,
        }

    except Exception as e:
        logger.exception(f"planner.compose_exits_size error: {e}")
        return {"eligible": False, "reason": "compose_error", "targets": [], "qty": 0, "notional": 0.0}

# ----------------------------
# Structure-Based Entry Calculation
# ----------------------------

def _calculate_structure_entry(
    setup_type: str,
    bias: str,
    current_close: float,
    orh: float,
    orl: float,
    pdh: float,
    pdl: float,
    vwap_current: Optional[float],
    atr: float
) -> float:
    """
    Calculate entry reference price based on setup type and structure levels.
    Uses institutional logic: enter at support for longs, resistance for shorts.

    Key principles:
    - Breakouts: Enter AFTER break (ORH + buffer for long, ORL - buffer for short)
    - Fades/Failures: Enter AT reversal level (ORL for long bounce, ORH for short rejection)
    - Gap Fill: Enter at structure (ORL for gap down/long, ORH for gap up/short)
    - VWAP: Enter at VWAP level with directional bias
    """
    setup_lower = setup_type.lower()

    if bias == "long":
        # LONG SETUPS - Enter at SUPPORT or after BREAKOUT above RESISTANCE

        # 1. BREAKOUT/BOS - Enter AFTER breakout above ORH
        if "breakout" in setup_lower or "break_of_structure" in setup_lower:
            if "orb" in setup_lower:
                return orh + (atr * 0.05)  # ORB breakout: 5% ATR buffer
            elif "momentum" in setup_lower:
                return orh + (atr * 0.03)  # Momentum breakout: 3% ATR buffer
            else:
                return orh  # Generic breakout at ORH

        # 2. FADE/FAILURE/REJECTION - Enter AT ORL (support bounce)
        elif any(kw in setup_lower for kw in ["fade", "failure", "rejection", "choc"]):
            return orl

        # 3. VWAP SETUPS - Enter at VWAP (with slight buffer)
        elif "vwap" in setup_lower:
            if vwap_current and vwap_current > 0:
                return vwap_current * 0.999  # Just below VWAP for long
            return current_close

        # 4. PULLBACK/RETEST - Enter at support (ORL or PDL)
        elif "pullback" in setup_lower or "retest" in setup_lower:
            return min(orl, pdl) if pdl > 0 else orl

        # 5. CONTINUATION - Enter at current price with momentum
        elif "continuation" in setup_lower:
            return current_close

        # 6. REVERSAL - Enter at support (ORL or PDL)
        elif "reversal" in setup_lower:
            return min(orl, pdl) if pdl > 0 else orl

        # 7. RANGE - Enter at range low (ORL)
        elif "range" in setup_lower:
            return orl

        # 8. SQUEEZE - Enter at support with volatility compression
        elif "squeeze" in setup_lower:
            return orl

        # 9. INSIDE BAR - Enter at inside bar low
        elif "inside" in setup_lower:
            return orl

        # 10. GAP FILL LONG - Gap DOWN scenario, enter at support (ORL)
        elif "gap" in setup_lower:
            return orl  # Gap down fill: enter long at support

        # 11. DEFAULT LONG - Entry at support (ORL)
        else:
            return orl

    else:  # SHORT setups
        # SHORT SETUPS - Enter at RESISTANCE or after BREAKDOWN below SUPPORT

        # 1. BREAKDOWN/BOS - Enter AFTER breakdown below ORL
        if "breakout" in setup_lower or "break_of_structure" in setup_lower:
            if "orb" in setup_lower:
                return orl - (atr * 0.05)  # ORB breakdown: 5% ATR buffer
            elif "momentum" in setup_lower:
                return orl - (atr * 0.03)  # Momentum breakdown: 3% ATR buffer
            else:
                return orl  # Generic breakdown at ORL

        # 2. FADE/FAILURE/REJECTION - Enter AT ORH (resistance rejection)
        elif any(kw in setup_lower for kw in ["fade", "failure", "rejection", "choc"]):
            return orh

        # 3. VWAP SETUPS - Enter at VWAP (with slight buffer)
        elif "vwap" in setup_lower:
            if vwap_current and vwap_current > 0:
                return vwap_current * 1.001  # Just above VWAP for short
            return current_close

        # 4. PULLBACK/RETEST - Enter at resistance (ORH or PDH)
        elif "pullback" in setup_lower or "retest" in setup_lower:
            return max(orh, pdh) if pdh > 0 else orh

        # 5. CONTINUATION - Enter at current price with momentum
        elif "continuation" in setup_lower:
            return current_close

        # 6. REVERSAL - Enter at resistance (ORH or PDH)
        elif "reversal" in setup_lower:
            return max(orh, pdh) if pdh > 0 else orh

        # 7. RANGE - Enter at range high (ORH)
        elif "range" in setup_lower:
            return orh

        # 8. SQUEEZE - Enter at resistance with volatility compression
        elif "squeeze" in setup_lower:
            return orh

        # 9. INSIDE BAR - Enter at inside bar high
        elif "inside" in setup_lower:
            return orh

        # 10. GAP FILL SHORT - Gap UP scenario, enter at resistance (ORH)
        elif "gap" in setup_lower:
            return orh  # Gap up fill: enter short at resistance

        # 11. DEFAULT SHORT - Entry at resistance (ORH)
        else:
            return orh

# ----------------------------
# Public API
# ----------------------------

def generate_trade_plan(
    df: pd.DataFrame,
    symbol: str,
    daily_df: Optional[pd.DataFrame] = None,
    setup_candidates: List[SetupCandidate] = None,
    # -------- OPTIONAL FEATURES for enhanced analysis ----------
    df1m_tail: Optional[pd.DataFrame] = None,
    index_df5m: Optional[pd.DataFrame] = None,
    sector_df5m: Optional[pd.DataFrame] = None,
    news_spike_gate: Any = None,
    prev_session_close: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build an intraday trade plan from a 5-min OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        5m bars with columns [open, high, low, close, volume]; naive IST index expected.
    symbol : str
    daily_df : pd.DataFrame | None
        Daily bars used to compute PDH/PDL/PDC.
    setup_candidates : List[SetupCandidate]
        Setup candidates from the structure detection system.
        Each candidate contains setup_type, strength, and reasons.
    df1m_tail : pd.DataFrame | None
        Optional 1m tail for spike/ret_z features (if available).
    index_df5m : pd.DataFrame | None
        Optional index 5m bars for index momentum.
    sector_df5m : pd.DataFrame | None
        Optional sector 5m bars for sector momentum.
    news_spike_gate : object | None
        Optional news spike detector; if provided, used to set news_spike_flag.
    prev_session_close : float | None
        Optional previous-day close override for gap metrics (falls back to PDC from daily_df).

    Returns
    -------
    plan : dict
        Serializable plan with eligibility, entry/stop/targets, sizing, context, and quality fields.
        Includes `plan["features"]` when optional inputs are provided, for Gate HCET consumption.
    """
    try:
        cfg = _load_planner_config()

        # Focus on the latest session
        df = ensure_naive_ist_index(df)
        sess = _ensure_session_df(df)

        # --- added: hard late-entry cutoff at planner level (tick/bar-ts aware) ---
        if not sess.empty:
            last_ts = sess.index[-1]
            cut_md = _parse_hhmm_to_md(load_filters().get("entry_cutoff_hhmm", "14:45"))
            if cut_md is not None:
                bar_ts = pd.Timestamp(last_ts)
                if _minute_of_day(bar_ts) >= cut_md:
                    return {"eligible": False, "reason": "after_entry_cutoff"}
        # --- /added ---

        # Warn if too few bars for a robust plan
        min_needed = max(20, cfg.orb_minutes // cfg.bar_minutes + 2)
        if len(sess) < min_needed:
            logger.warning(f"planner: {symbol} very few candles — plan may be weak (len={len(sess)}, need>={min_needed})")

        # Core features
        sess = sess.copy()
        sess["vwap"] = _session_vwap(sess)
        sess["ema20"] = calculate_ema(sess["close"], 20)
        sess["ema50"] = calculate_ema(sess["close"], 50)

        # Optional indicators for soft sizing / checklist
        rsi_len = int(cfg._extras.get("intraday_params", {}).get("rsi_len", 14))
        adx_len = int(cfg._extras.get("intraday_params", {}).get("adx_len", 14))
        sess["rsi14"] = calculate_rsi(sess["close"], period=rsi_len)
        sess["adx14"] = calculate_adx(sess, period=adx_len)
        macd_pack = calculate_macd(sess["close"])
        sess["macd"] = macd_pack["macd"]
        sess["macd_hist"] = macd_pack["histogram"]
        sess["vol_avg20"] = sess["volume"].rolling(20, min_periods=1).mean()
        sess["vol_ratio"] = sess["volume"] / sess["vol_avg20"]

        # Context
        atr = calculate_atr(df.tail(200), period=cfg.atr_period)
        orh, orl, or_end = _opening_range(sess, cfg.orb_minutes)
        session_date = sess.index[-1].date() if (sess is not None and not sess.empty) else None
        pd_levels = _prev_day_levels(daily_df, session_date)

        # Gap vs prev close (contextual)
        gap_pct = np.nan
        if not np.isnan(pd_levels.get("PDC", np.nan)):
            first_open = float(sess.iloc[0]["open"]) if not sess.empty else np.nan
            if not np.isnan(first_open):
                gap_pct = 100.0 * (first_open - pd_levels["PDC"]) / max(pd_levels["PDC"], 1e-9)

        regime = _regime(sess, cfg)
        planning_logger = get_planning_logger()

        if not setup_candidates:
            planning_logger.log_reject(
                symbol,
                "no_setup_candidates_provided",
                timestamp=last_ts.isoformat() if last_ts is not None else None,
                regime=regime,
                orh=orh,
                orl=orl
            )
            return {
                "eligible": False,
                "reason": "no_setup_candidates",
                "quality": {"rejection_reason": "no_setup_candidates_provided"},
                "notes": {"cautions": ["no_setup_candidates", f"regime_{regime}"]}
            }

        strat = _strategy_selector(sess, cfg, regime, orh, orl, or_end, sess["vwap"], pd_levels, setup_candidates, index_df5m, daily_df)

        if strat is None:
            logger.info(f"planner: {symbol} setup_rejected (regime={regime}, ORH={orh:.2f}, ORL={orl:.2f})")
            planning_logger.log_reject(
                symbol,
                "setup_conditions_not_met",
                timestamp=last_ts.isoformat() if last_ts is not None else None,
                regime=regime,
                orh=orh,
                orl=orl,
                setup_candidates_count=len(setup_candidates)
            )
            return {
                "eligible": False,
                "reason": "setup_conditions_not_met",
                "quality": {"rejection_reason": "setup_conditions_not_met"},
                "notes": {"cautions": ["rejected_setup", f"regime_{regime}"]}
            }

        if strat["name"] == "no_setup":
            logger.debug(f"planner: {symbol} no_setup (regime={regime}, ORH={orh:.2f}, ORL={orl:.2f})")
            planning_logger.log_reject(symbol, "no_setup_detected", timestamp=last_ts.isoformat() if last_ts is not None else None,
                                     strategy_type=str(strat["name"]) if strat["name"] is not None else None,
                                     regime=str(regime) if regime is not None else None,
                                     orh=float(orh) if orh is not None else None,
                                     orl=float(orl) if orl is not None else None)
            return {
                "eligible": False,
                "reason": "no_setup_detected",
                "quality": {"rejection_reason": "no_setup_detected"},
                "notes": {"cautions": ["no_setup", f"regime_{regime}"]}
            }

        # Soft sizing / checklist using EXTRAS (no hard gating)
        late_penalty = cfg._extras.get("late_entry_penalty", {})
        intraday_gate = cfg._extras.get("intraday_gate", {})

        last = sess.iloc[-1]
        qty_scale = 1.0
        cautions: List[str] = []
        must_checks: List[str] = []
        should_checks: List[str] = []

        # Late entry penalties
        rsi_above = late_penalty.get("rsi_above")
        macd_above = late_penalty.get("macd_above")
        if rsi_above is not None and float(last["rsi14"]) > float(rsi_above):
            qty_scale *= 0.6
            cautions.append(f"late_entry_rsi>{rsi_above}")
        if macd_above is not None and float(last["macd_hist"]) > float(macd_above):
            qty_scale *= 0.8
            cautions.append(f"late_entry_macd_hist>{macd_above}")

        # Intraday gate snapshot -> must/should checks (soft only)
        mv = intraday_gate.get("min_volume_ratio")
        if mv is not None:
            should_checks.append(f"vol_ratio>={mv}")
            if float(last["vol_ratio"]) < float(mv):
                qty_scale *= 0.8
                cautions.append("weak_volume_ratio")

        if intraday_gate.get("require_above_vwap", True):
            must_checks.append("price_above_vwap" if strat["bias"] == "long" else "price_below_vwap")

        rmin, rmax = intraday_gate.get("min_rsi"), intraday_gate.get("max_rsi")
        if rmin is not None and rmax is not None:
            should_checks.append(f"RSI in [{rmin},{rmax}]")
            rsi_now = float(last["rsi14"])
            if not (float(rmin) <= rsi_now <= float(rmax)):
                qty_scale *= 0.85
                cautions.append("rsi_out_of_band")

        adxmin, adxmax = intraday_gate.get("min_adx"), intraday_gate.get("max_adx")
        if adxmin is not None and adxmax is not None:
            should_checks.append(f"ADX in [{adxmin},{adxmax}]")
            adx_now = float(last["adx14"])
            if not (float(adxmin) <= adx_now <= float(adxmax)):
                qty_scale *= 0.9
                cautions.append("adx_out_of_band")

        # Calculate structure-based entry price instead of using current close
        current_close = float(sess.iloc[-1]["close"])
        vwap_current = float(sess.iloc[-1]["vwap"]) if "vwap" in sess.columns and not pd.isna(sess.iloc[-1]["vwap"]) else None

        entry_ref_price = _calculate_structure_entry(
            setup_type=strat["name"],
            bias=strat["bias"],
            current_close=current_close,
            orh=orh,
            orl=orl,
            pdh=pd_levels.get("PDH", np.nan),
            pdl=pd_levels.get("PDL", np.nan),
            vwap_current=vwap_current,
            atr=atr
        )

        # Priority 3: Load cap_segment for cap-aware sizing
        cap_segment = "unknown"
        try:
            import json
            from pathlib import Path
            nse_file = Path(__file__).parent.parent / "nse_all.json"
            if nse_file.exists():
                with nse_file.open() as f:
                    data = json.load(f)
                cap_map = {item["symbol"]: item.get("cap_segment", "unknown") for item in data}
                cap_segment = cap_map.get(symbol, "unknown")
        except Exception as e:
            logger.debug(f"CAP_SIZING: Failed to load cap_segment for {symbol}: {e}")
            cap_segment = "unknown"

        exits = _compose_exits_and_size(entry_ref_price, strat["bias"], atr, strat["structure_stop"], cfg, qty_scale=qty_scale, cap_segment=cap_segment)

        # Feasibility tightening (from planner_precision extras)
        pp = cfg._extras.get("planner_precision", {})
        logger.info(f"PLANNER_CONFIG: {symbol} planner_precision={pp}")
        measured_move = max(orh - orl, atr) if strat["bias"] in ("long","short") else atr
        # CRITICAL SAFETY FIX: Use conservative fallback instead of dangerous 1e-6
        # The 1e-6 fallback could create massive position sizes (risk_rupees / 1e-6)
        eps = max(entry_ref_price * 0.001, 0.01)  # 0.1% of price or ₹0.01 minimum
        rps = float(exits.get("risk_per_share", eps))
        rr_clip_max = float(pp.get("rr_clip_max", 6.0))

        t1_max_pct = float(pp.get("t1_max_pct", 100.0))
        t1_max_mm_frac = float(pp.get("t1_max_mm_frac", 1.0))
        t2_max_pct = float(pp.get("t2_max_pct", 100.0))
        t2_max_mm_frac = float(pp.get("t2_max_mm_frac", 1.0))
        t1_min_rr = float(pp.get("t1_min_rr", 1.0))

        def _cap_move(max_pct, max_mm_frac):
            return min(entry_ref_price * (max_pct / 100.0), measured_move * max_mm_frac)

        cap1 = _cap_move(t1_max_pct, t1_max_mm_frac)
        cap2 = _cap_move(t2_max_pct, t2_max_mm_frac)

        t1_orig = exits["targets"][0]["level"] if exits.get("targets") else np.nan
        t2_orig = exits["targets"][1]["level"] if exits.get("targets") and len(exits["targets"]) > 1 else np.nan

        if (not rps) or rps <= 0:
            # coarse salvage if RPS not computed - use the same eps value from above
            rps = eps

        if strat["bias"] == "long":
            t1_feasible = entry_ref_price + min(max(float(t1_orig) - entry_ref_price, 0.0), cap1) if not np.isnan(t1_orig) else entry_ref_price + cfg.t1_rr * rps
            t2_feasible = entry_ref_price + min(max(float(t2_orig) - entry_ref_price, 0.0), cap2) if not np.isnan(t2_orig) else entry_ref_price + cfg.t2_rr * rps
            t1_floor = entry_ref_price + t1_min_rr * rps
            t1_feasible = max(t1_feasible, t1_floor)
        else:
            t1_feasible = entry_ref_price - min(max(entry_ref_price - float(t1_orig), 0.0), cap1) if not np.isnan(t1_orig) else entry_ref_price - cfg.t1_rr * rps
            t2_feasible = entry_ref_price - min(max(entry_ref_price - float(t2_orig), 0.0), cap2) if not np.isnan(t2_orig) else entry_ref_price - cfg.t2_rr * rps
            t1_floor = entry_ref_price - t1_min_rr * rps
            t1_feasible = min(t1_feasible, t1_floor)

        t1_rr_eff = (t1_feasible - entry_ref_price) / rps if strat["bias"] == "long" else (entry_ref_price - t1_feasible) / rps
        t2_rr_eff = (t2_feasible - entry_ref_price) / rps if strat["bias"] == "long" else (entry_ref_price - t2_feasible) / rps

        if exits.get("targets"):
            exits["targets"][0]["level"] = round(float(t1_feasible), 2)
            exits["targets"][0]["rr"] = round(float(t1_rr_eff), 2)
        if exits.get("targets") and len(exits["targets"]) > 1:
            exits["targets"][1]["level"] = round(float(t2_feasible), 2)
            exits["targets"][1]["rr"] = round(float(t2_rr_eff), 2)

        # Log target levels for verification (Option B fix)
        logger.info(f"PLANNER_TARGETS: {symbol} entry={entry_ref_price:.2f} risk={rps:.2f} | T1={t1_feasible:.2f} ({t1_rr_eff:.2f}R) T2={t2_feasible:.2f} ({t2_rr_eff:.2f}R) | mm={measured_move:.2f} cap1={cap1:.2f} cap2={cap2:.2f}")

        # Calculate quality metric - DIFFERENT for breakouts vs fades
        # INSTITUTIONAL FIX (exit-opt-004-phase2c): Momentum-based metric for breakouts
        is_breakout_strategy = strat["name"] in {
            "breakout_long", "breakout_short",
            "orb_breakout_long", "orb_breakout_short",
            "flag_continuation_long", "flag_continuation_short"
        }

        if is_breakout_strategy:
            # MOMENTUM QUALITY for breakouts: volume * breakout_strength / risk
            # This measures STRENGTH of the break, not distance to arbitrary levels

            # Get volume confirmation
            vol_ratio = float(sess["vol_ratio"].iloc[-1]) if "vol_ratio" in sess.columns and not np.isnan(sess["vol_ratio"].iloc[-1]) else 1.0
            vol_ratio = max(vol_ratio, 0.5)  # Floor at 0.5 to avoid division issues

            # Calculate breakout strength (how far beyond the level)
            if strat["bias"] == "long":
                breakout_distance = max(current_close - orh, 0)  # Distance above ORH
                # Normalize by ATR (large ATR stocks naturally have larger breakout distances)
                breakout_strength = breakout_distance / max(atr, 1e-6)
            elif strat["bias"] == "short":
                breakout_distance = max(orl - current_close, 0)  # Distance below ORL
                breakout_strength = breakout_distance / max(atr, 1e-6)
            else:
                breakout_strength = 0.0

            # Structural R:R = volume * breakout_strength / risk
            # Higher volume + stronger break + tighter stop = higher quality
            # This is what professional traders look for in breakouts
            structural_rr = (vol_ratio * breakout_strength) / max(rps / atr, 1e-6)

            # Log the breakout quality for analysis
            logger.info(f"BREAKOUT_QUALITY: {symbol} vol_ratio={vol_ratio:.2f} breakout_dist={breakout_distance:.2f} "
                       f"breakout_strength={breakout_strength:.3f} risk_norm={rps/atr:.2f} structural_rr={structural_rr:.3f}")

        else:
            # DISTANCE-BASED R:R for fades/mean-reversion (works correctly for these)
            if strat["bias"] == "long":
                # For longs, next objective is ORH + 50% of measured move
                next_objective = orh + 0.5 * measured_move
                structural_rr = (next_objective - entry_ref_price) / max(rps, 1e-6)
            elif strat["bias"] == "short":
                # For shorts, next objective is ORL - 50% of measured move
                next_objective = orl - 0.5 * measured_move
                structural_rr = (entry_ref_price - next_objective) / max(rps, 1e-6)
            else:
                next_objective = float("nan")
                structural_rr = float("nan")

        # Handle negative/invalid values
        if not np.isnan(structural_rr):
            if structural_rr < 0:
                if is_breakout_strategy:
                    logger.warning(f"Negative momentum quality {structural_rr:.2f} for {symbol} - entry below breakout level")
                else:
                    logger.warning(f"Negative structural RR {structural_rr:.2f} for {symbol} - potential setup-strategy mismatch")
                structural_rr = 0.0
            else:
                structural_rr = float(np.clip(structural_rr, 0.0, rr_clip_max))

        acc_cfg = cfg._extras.get("acceptance", {})
        acc_bars = int(acc_cfg.get("bars", 2))
        acc_bpct = float(acc_cfg.get("retest_bpct", 0.5))
        need_vwap = bool(acc_cfg.get("need_vwap_hold", False))

        _lvl = orh if strat["bias"] == "long" else orl
        win = sess.tail(max(acc_bars, 2))

        if strat["bias"] == "long":
            retest_ok = (win["low"].min() >= _lvl * (1 - acc_bpct/100.0))
            hold_ok = (win["close"].iloc[-1] >= _lvl) and (not need_vwap or win["close"].iloc[-1] >= win["vwap"].iloc[-1])
        elif strat["bias"] == "short":
            retest_ok = (win["high"].max() <= _lvl * (1 + acc_bpct/100.0))
            hold_ok = (win["close"].iloc[-1] <= _lvl) and (not need_vwap or win["close"].iloc[-1] <= win["vwap"].iloc[-1])
        else:
            retest_ok = False
            hold_ok = False

        # Multi-tier acceptance status system
        if retest_ok and hold_ok:
            acceptance_status = "excellent"  # Both criteria met - highest quality
        elif retest_ok:
            acceptance_status = "good"       # Retest held but not holding current level
        elif hold_ok:
            acceptance_status = "fair"       # Holding level but had weak retest
        else:
            acceptance_status = "poor"       # Neither criteria met - lowest quality

        start_date, end_date = get_date_range_from_df(df)
        plan = {
            "symbol": symbol,
            "eligible": exits.get("eligible", False),
            "regime": regime,
            "strategy": strat["name"],
            "bias": strat["bias"],
            "entry": {
                "reference": round(entry_ref_price, 2),
                "trigger": strat["entry_trigger"],
                "zone": exits.get("entry_zone"),
                "must": must_checks + (["acceptance_quality"] if acceptance_status in ["good", "excellent"] else []),
                "should": should_checks,
                "filters": [
                    "above_VWAP" if strat["bias"] == "long" else "below_VWAP",
                    "ORB_context",
                    "volume_persistency>avg20",
                ],
            },
            "stop": {
                "hard": exits.get("hard_sl"),
                "type": "max(structure,ATR)",
                "structure": None if np.isnan(strat["structure_stop"]) else round(float(strat["structure_stop"]), 2),
            },
            "targets": exits.get("targets", []),
            "trail": exits.get("trail"),
            "sizing": {
                "risk_per_share": exits.get("risk_per_share"),
                "risk_rupees": cfg.risk_per_trade_rupees,
                "qty": exits.get("qty"),
                "notional": exits.get("notional"),
                "qty_scale": round(qty_scale, 2),
            },
            "levels": strat["context"]["levels"],
            "indicators": {
                "vwap": None if pd.isna(sess["vwap"].iloc[-1]) else round(float(sess["vwap"].iloc[-1]), 2),
                "ema20": None if pd.isna(sess["ema20"].iloc[-1]) else round(float(sess["ema20"].iloc[-1]), 2),
                "ema50": None if pd.isna(sess["ema50"].iloc[-1]) else round(float(sess["ema50"].iloc[-1]), 2),
                "atr": None if np.isnan(atr) else round(float(atr), 2),
                "rsi14": None if pd.isna(sess["rsi14"].iloc[-1]) else round(float(sess["rsi14"].iloc[-1]), 2),
                "adx14": None if pd.isna(sess["adx14"].iloc[-1]) else round(float(sess["adx14"].iloc[-1]), 2),
                "macd_hist": None if pd.isna(sess["macd_hist"].iloc[-1]) else round(float(sess["macd_hist"].iloc[-1]), 4),
                "vol_ratio": None if pd.isna(sess["vol_ratio"].iloc[-1]) else round(float(sess["vol_ratio"].iloc[-1]), 2),
            },
            "notes": {
                "gap_pct": None if np.isnan(gap_pct) else round(float(gap_pct), 2),
                "opening_range_end": str(or_end),
                "cautions": cautions,
            },
            "guardrails": [
                "avoid_entries <5m before/after lunch window" if cfg.enable_lunch_pause else None,
                "cancel trade after 45m if trigger not met",
            ],
            "date_range": {"start": start_date, "end": end_date},
            "quality": {
                "structural_rr": None if np.isnan(structural_rr) else round(float(structural_rr), 2),
                "acceptance_status": acceptance_status,
                "retest_ok": retest_ok,
                "hold_ok": hold_ok,
                "t1_feasible": bool(not np.isnan(t1_orig) and (abs(t1_feasible - t1_orig) < 1e-6 or (abs(t1_feasible - entry_ref_price) <= cap1 + 1e-9))),
                "t2_feasible": bool(not np.isnan(t2_orig) and (abs(t2_feasible - t2_orig) < 1e-6 or (abs(t2_feasible - entry_ref_price) <= cap2 + 1e-9)))
            },
        }
        plan["guardrails"] = [g for g in plan["guardrails"] if g]


        # APPLY QUALITY FILTERS - This was missing and causing poor performance!
        if plan["eligible"]:
            quality_filters = cfg._extras.get("quality_filters", {})
            if quality_filters.get("enabled", False):
                structural_rr_val = plan["quality"].get("structural_rr", 0)

                # INSTITUTIONAL FIX (exit-opt-004-phase2b): Strategy-specific R:R thresholds
                # Breakouts have momentum that blows through resistance, so use relaxed thresholds
                strategy_rr_overrides = quality_filters.get("strategy_structural_rr_overrides", {})
                strategy_type = plan.get("strategy", "")
                min_structural_rr = strategy_rr_overrides.get(strategy_type, quality_filters.get("min_structural_rr", 2.0))

                # Professional quality filters - reject poor setups at planning stage
                if plan["eligible"]:

                    # Filter: targets must be feasible
                    if not plan["quality"]["t1_feasible"]:
                        plan["eligible"] = False
                        plan["quality"]["rejection_reason"] = "T1 not feasible"
                        logger.debug(f"planner: {symbol} rejected due to T1 not feasible")
                        t1_level = plan["targets"][0].get("level") if plan["targets"] else None
                        entry_price = plan.get("entry_ref_price")
                        planning_logger.log_reject(symbol, "T1_not_feasible",
                                                 timestamp=last_ts.isoformat() if last_ts is not None else None,
                                                 strategy_type=str(plan["strategy"]) if plan["strategy"] is not None else None,
                                                 t1_orig=float(t1_level) if t1_level is not None else None,
                                                 entry_ref_price=float(entry_price) if entry_price is not None else None)

                    # Filter: structural R:R check (strategy-specific thresholds)
                    # INSTITUTIONAL FIX (exit-opt-004-phase2b): This filter was previously not enforced!
                    if plan["eligible"] and structural_rr_val < min_structural_rr:
                        plan["eligible"] = False
                        plan["quality"]["rejection_reason"] = f"structural_rr {structural_rr_val:.2f} < {min_structural_rr:.2f}"
                        logger.debug(f"planner: {symbol} rejected due to poor structural R:R {structural_rr_val:.2f} (strategy={strategy_type})")
                        planning_logger.log_reject(symbol, "structural_rr_too_low",
                                                 timestamp=last_ts.isoformat() if last_ts is not None else None,
                                                 strategy_type=str(plan["strategy"]) if plan["strategy"] is not None else None,
                                                 structural_rr=float(structural_rr_val),
                                                 min_required=float(min_structural_rr))

                    # Filter: T2 feasibility check (tiered approach)
                    # If T2 is infeasible, reduce position size to 50% (T1-only scalp mode)
                    if plan["eligible"] and not plan["quality"]["t2_feasible"]:
                        t2_level = plan["targets"][1].get("level") if len(plan["targets"]) > 1 else None
                        entry_price = plan.get("entry_ref_price")

                        # Reduce position size for T1-only exits
                        original_qty = plan["sizing"]["qty"]
                        plan["sizing"]["qty"] = max(1, int(original_qty * 0.5))
                        plan["quality"]["t2_exit_mode"] = "T1_only_scalp"

                        logger.info(f"planner: {symbol} T2 not feasible - reducing qty {original_qty}->{plan['sizing']['qty']} (T1-only mode)")
                        planning_logger.log_accept(symbol,
                                                 strategy_type=plan["strategy"],
                                                 timestamp=last_ts.isoformat() if last_ts is not None else None,
                                                 bias=plan["bias"],
                                                 qty=plan["sizing"]["qty"],
                                                 t1_rr=plan["targets"][0]["rr"] if plan["targets"] else None,
                                                 quality_status="T1_only_scalp",
                                                 t2_orig=float(t2_level) if t2_level is not None else None,
                                                 entry_ref_price=float(entry_price) if entry_price is not None else None)

                    # Filter: minimum target risk-reward ratios
                    if len(plan["targets"]) >= 2:
                        t1_rr = plan["targets"][0].get("rr", 0)

                        min_t1_rr = quality_filters.get("min_t1_rr", 1.0)
                        if t1_rr < min_t1_rr:
                            plan["eligible"] = False
                            plan["quality"]["rejection_reason"] = f"T1_rr {t1_rr:.1f} < {min_t1_rr}"
                            logger.debug(f"planner: {symbol} rejected due to low T1_rr {t1_rr:.1f}")
                            planning_logger.log_reject(symbol, "T1_rr_too_low", timestamp=last_ts.isoformat() if last_ts is not None else None,
                                                     strategy_type=str(plan["strategy"]) if plan["strategy"] is not None else None,
                                                     t1_rr=float(t1_rr) if t1_rr is not None else None,
                                                     min_t1_rr=float(min_t1_rr) if min_t1_rr is not None else None)

                    # Filter: ADX requirement for breakout setups
                    if plan["eligible"] and "breakout" in plan["strategy"]:
                        breakout_min_adx = quality_filters.get("breakout_min_adx", 20.0)
                        current_adx = plan["indicators"].get("adx14")

                        # Skip filter if ADX is not available
                        if current_adx is None:
                            current_adx = 0.0

                        if current_adx < breakout_min_adx:
                            plan["eligible"] = False
                            plan["quality"]["rejection_reason"] = f"ADX {current_adx:.1f} < {breakout_min_adx} (breakout filter)"
                            logger.debug(f"planner: {symbol} rejected breakout setup due to low ADX {current_adx:.1f}")
                            planning_logger.log_reject(symbol, "ADX_filter", timestamp=last_ts.isoformat() if last_ts is not None else None,
                                                     strategy_type=str(plan["strategy"]) if plan["strategy"] is not None else None,
                                                     current_adx=float(current_adx) if current_adx is not None else None,
                                                     min_adx=float(breakout_min_adx) if breakout_min_adx is not None else None)

        # Log planning decision
        if plan["eligible"]:
            planning_logger.log_accept(symbol,
                                     strategy_type=plan["strategy"],
                                     timestamp=last_ts.isoformat() if last_ts is not None else None,
                                     bias=plan["bias"],
                                     qty=plan["sizing"]["qty"],
                                     t1_rr=plan["targets"][0]["rr"] if plan["targets"] else None,
                                     structural_rr=plan["quality"].get("structural_rr"),
                                     entry_ref_price=plan.get("entry_ref_price"),
                                     quality_status=plan["quality"].get("acceptance_status"))

        logger.info(
            f"planner.plan sym={symbol} eligible={plan['eligible']} "
            f"bias={plan['bias']} strat={plan['strategy']} qty={plan['sizing']['qty']} "
            f"rr_t1={plan['targets'][0]['rr'] if plan['targets'] else None} "
            f"structural_rr={plan['quality'].get('structural_rr', 'N/A')} "
            f"last_ts={sess.index[-1]}"
        )
        return plan

    except Exception as e:
        logger.exception(f"planner.generate_trade_plan error for {symbol}: {e}")
        return {}
