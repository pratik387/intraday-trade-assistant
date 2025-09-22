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

from config.logging_config import get_agent_logger
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index
from utils.time_util import _minute_of_day, _parse_hhmm_to_md
import datetime

logger = get_agent_logger()

# ----------------------------
# Helpers (math/indicators)
# ----------------------------

def calculate_atr(df: pd.DataFrame, period: int) -> float:
    """Wilder-style ATR (EMA of True Range) approximated with rolling mean for speed."""
    try:
        h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        prev_close = np.r_[c[0], c[:-1]]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1]
        return float(atr)
    except Exception as e:
        logger.exception(f"planner.atr error: {e}")
        return float("nan")

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

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"line": macd_line, "signal": signal_line, "hist": hist}

def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx.bfill()

def _prev_day_levels(
    daily_df: Optional[pd.DataFrame],
    session_date: Optional[datetime.date],
) -> Dict[str, float]:
    """Return PDH/PDL/PDC from the last completed trading day strictly before session_date."""
    try:
        if daily_df is None or daily_df.empty:
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}
        d = daily_df.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date").set_index("date")
        else:
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()

        if session_date is not None:
            d = d[d.index.date < session_date]

        if "volume" in d.columns:
            d = d[d["volume"].fillna(0) > 0]
        d = d[d["high"].notna() & d["low"].notna()]

        if d.empty:
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}

        prev = d.iloc[-1]
        return {
            "PDH": float(prev["high"]),
            "PDL": float(prev["low"]),
            "PDC": float(prev.get("close", float("nan"))),
        }
    except Exception:
        return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}


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
        daily_ema = _ema(daily_close, ema_period)

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

    # Extras are optional but must be dicts if present
    extras = {
        "intraday_gate": config.get("intraday_gate", {}),
        "late_entry_penalty": config.get("late_entry_penalty", {}),
        "planner_precision": config.get("planner_precision", {}),
        "acceptance": config.get("acceptance", {}),
        "quality_filters": config.get("quality_filters", {}),
    }

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
    ema20 = _ema(df_sess["close"], 20)
    ema50 = _ema(df_sess["close"], 50)
    slope = (ema20.iloc[-1] - ema20.iloc[-5]) / max(abs(ema20.iloc[-5]), 1e-9)
    if chop <= cfg.choppiness_low and ema20.iloc[-1] > ema50.iloc[-1] and slope > 0:
        return "trend_up"
    if chop <= cfg.choppiness_low and ema20.iloc[-1] < ema50.iloc[-1] and slope < 0:
        return "trend_down"
    if chop >= cfg.choppiness_high:
        return "choppy"
    return "range"

def _select_long_strategy(
    df_sess: pd.DataFrame, cfg: PlannerConfig, regime: str,
    orh: float, orl: float, or_end: pd.Timestamp,
    vwap: pd.Series, pd_levels: Dict[str, float],
    close: float, ema20: pd.Series, ema50: pd.Series,
    setup_type: str, index_df5m: Optional[pd.DataFrame] = None,
    daily_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Select appropriate long strategy based on market conditions and setup type."""

    # Strategy 1: ORB Pullback Long - prioritize if setup is ORB-related
    if ("orb" in setup_type or "breakout_long" in setup_type) and regime == "trend_up" and close > orh and close > vwap.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        # Check time restriction for ORB pullback - only allow within first 60-90 minutes
        from config.filters_setup import load_filters
        config = load_filters()
        orb_max_minutes = config.get("orb_pullback_max_minutes", 90)

        session_start = df_sess.index.min()
        current_time = df_sess.index[-1]
        minutes_elapsed = (current_time - session_start).total_seconds() / 60.0

        if minutes_elapsed <= orb_max_minutes:
            # Check index alignment if required
            index_aligned = True
            index_momentum = 0.0

            # Check if we have index data and alignment requirement
            require_alignment = config.get("orb_require_index_alignment", True)
            if require_alignment and index_df5m is not None and len(index_df5m) > 0:
                from services.features import _momentum_5m
                index_momentum = _momentum_5m(index_df5m, 3)
                # For long ORB pullback, require positive index momentum
                index_aligned = index_momentum > 0

            if index_aligned:
                basis = "orb_pullback_long"
                stop_structure = max(orl, _pivot_swing_low(df_sess, 12))
                trigger_msg = f"pullback to ~ORH({orh:.2f}) with hold above VWAP (within {orb_max_minutes}min)"
                if require_alignment:
                    trigger_msg += f" (index aligned: {index_momentum:+.2f}%)"

                return {
                    "name": basis, "bias": "long",
                    "entry_trigger": trigger_msg,
                    "structure_stop": float(stop_structure),
                    "context": {"reason": ["trend_up", "above_ORH", "above_VWAP", "ema20>ema50", f"setup:{setup_type}", f"time_ok:{minutes_elapsed:.0f}min", f"index_aligned:{index_momentum:+.1f}%"],
                                "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
                }

    # Strategy 2: VWAP Reclaim Long - prioritize for VWAP-related setups
    if ("vwap" in setup_type or "reclaim" in setup_type) and close > vwap.iloc[-1] and df_sess["close"].tail(cfg.vwap_reclaim_min_bars_above).gt(
        vwap.tail(cfg.vwap_reclaim_min_bars_above)
    ).all():
        # Check distance to resistance levels (PDH/ORH) - don't buy if too close
        from config.filters_setup import load_filters
        config = load_filters()
        min_resistance_distance_atr = config.get("vwap_min_resistance_distance_atr", 0.2)

        # Calculate ATR for distance check
        atr_proxy = abs(df_sess["close"].pct_change().rolling(14, min_periods=5).mean().iloc[-1]) * close
        min_distance = min_resistance_distance_atr * atr_proxy

        # Find nearest resistance level above current price
        resistance_levels = []
        if pd_levels.get("PDH") and not np.isnan(pd_levels["PDH"]) and pd_levels["PDH"] > close:
            resistance_levels.append(pd_levels["PDH"])
        if orh > close:
            resistance_levels.append(orh)

        # Check if we have enough room to resistance
        distance_ok = True
        nearest_resistance = None
        if resistance_levels:
            nearest_resistance = min(resistance_levels)
            distance_to_resistance = nearest_resistance - close
            if distance_to_resistance < min_distance:
                distance_ok = False

        if distance_ok:
            # Check volume persistence - require last 3 bars volume > avg20
            volume_persistent = True
            vol_persistence_bars = config.get("vwap_volume_persistence_bars", 3)

            if config.get("vwap_reclaim_volume_confirmation", True):
                recent_volumes = df_sess["volume"].tail(vol_persistence_bars)
                avg_volume = df_sess["volume"].rolling(20, min_periods=10).mean()
                recent_avg_volumes = avg_volume.tail(vol_persistence_bars)

                # Check if each of the last N bars has volume > its 20-period average
                volume_persistent = (recent_volumes > recent_avg_volumes).all()

            if volume_persistent:
                # Check higher timeframe alignment if required
                htf_aligned = True
                daily_trend = "neutral"

                require_htf_alignment = config.get("require_higher_timeframe_alignment", True)
                if require_htf_alignment and daily_df is not None:
                    daily_ema_period = config.get("daily_trend_ema_period", 20)
                    daily_trend = _detect_daily_trend(daily_df, daily_ema_period)
                    # For long VWAP reclaim, require uptrend or neutral (not downtrend)
                    htf_aligned = daily_trend != "down"

                if htf_aligned:
                    basis = "vwap_reclaim_long"
                    stop_structure = _pivot_swing_low(df_sess, 8)
                    trigger_msg = "reclaim VWAP with N closes above, enter on minor dip"
                    if nearest_resistance:
                        trigger_msg += f" (resistance at {nearest_resistance:.2f})"
                    if config.get("vwap_reclaim_volume_confirmation", True):
                        trigger_msg += " + volume persistence"
                    if require_htf_alignment:
                        trigger_msg += f" + daily trend: {daily_trend}"

                    return {
                        "name": basis, "bias": "long",
                        "entry_trigger": trigger_msg,
                        "structure_stop": float(stop_structure),
                        "context": {"reason": ["VWAP_reclaim", f"setup:{setup_type}", "resistance_distance_ok", "volume_persistent", f"daily_trend:{daily_trend}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
                    }

    # Reject specific setups that don't meet their criteria (prevent fallback to generic)
    if ("vwap" in setup_type or "reclaim" in setup_type):
        logger.info(f"VWAP/reclaim setup {setup_type} rejected - conditions not met")
        return None

    if ("orb" in setup_type or "breakout_long" in setup_type):
        logger.debug(f"ORB/breakout setup {setup_type} rejected - conditions not met (regime/price/timing)")
        return None


    # Strategy 3: PDH Break + Hold Long - for range breakout setups
    if (pd_levels.get("PDH") is not None and not np.isnan(pd_levels.get("PDH"))) \
       and close > pd_levels["PDH"] and close > vwap.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        basis = "pdh_break_hold_long"
        stop_structure = max(orl, _pivot_swing_low(df_sess, 10))
        return {
            "name": basis, "bias": "long",
            "entry_trigger": f"retest/hold above PDH({pd_levels['PDH']:.2f})",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["above_PDH","above_VWAP","ema20>ema50", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 4: Momentum Breakout Long - for momentum-based setups
    if "momentum_breakout_long" in setup_type and close > vwap.iloc[-1]:
        basis = "momentum_breakout_long"
        stop_structure = _pivot_swing_low(df_sess, 6)  # Tighter stops for momentum
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "momentum breakout with volume surge, enter on minor pullback",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["momentum_surge", "above_VWAP", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 5: Trend Continuation Long - for trend following setups
    if "trend_continuation_long" in setup_type and regime == "trend_up" and ema20.iloc[-1] > ema50.iloc[-1]:
        basis = "trend_continuation_long"
        stop_structure = max(_pivot_swing_low(df_sess, 8), ema20.iloc[-1] * 0.985)  # EMA or swing low
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "trend continuation pattern, enter on minor dip to EMA20",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_up", "ema20>ema50", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # INSTITUTIONAL CONCEPTS - Long strategies
    if "order_block_long" in setup_type:
        basis = "order_block_long"
        stop_structure = _pivot_swing_low(df_sess, 5) * 0.998  # Tight stop below order block
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "order block support test with rejection",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["order_block_support", "institutional_accumulation", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "fair_value_gap_long" in setup_type:
        basis = "fair_value_gap_long"
        stop_structure = close * 0.996  # Stop below FVG zone
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "fair value gap fill completion",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["fair_value_gap_fill", "price_imbalance_correction", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "liquidity_sweep_long" in setup_type:
        basis = "liquidity_sweep_long"
        stop_structure = _pivot_swing_low(df_sess, 3) * 0.997  # Stop below swept level
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "post-sweep reversal with volume confirmation",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["liquidity_sweep_reversal", "stop_hunt_complete", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "discount_zone_long" in setup_type:
        basis = "discount_zone_long"
        stop_structure = _pivot_swing_low(df_sess, 8) * 0.995  # Wider stop for value zone
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "discount zone value buying opportunity",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["discount_zone_buy", "institutional_value_area", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "break_of_structure_long" in setup_type and close > vwap.iloc[-1]:
        basis = "break_of_structure_long"
        stop_structure = _pivot_swing_low(df_sess, 5) * 0.998  # Stop below broken structure
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "break of structure continuation",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["break_of_structure", "trend_change_up", "above_VWAP", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "change_of_character_long" in setup_type:
        basis = "change_of_character_long"
        stop_structure = _pivot_swing_low(df_sess, 3) * 0.997  # Tight stop for reversal
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "change of character reversal signal",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["change_of_character", "early_reversal_signal", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "equilibrium_breakout_long" in setup_type and close > vwap.iloc[-1]:
        basis = "equilibrium_breakout_long"
        stop_structure = vwap.iloc[-1] * 0.998  # Stop below VWAP/equilibrium
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "equilibrium breakout above balance",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["equilibrium_breakout", "balance_resolution_up", "above_VWAP", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # INSTITUTIONAL RANGE TRADING STRATEGIES - New professional setups for choppy markets
    if "range_deviation_long" in setup_type:
        basis = "range_deviation_long"
        stop_structure = _pivot_swing_low(df_sess, 5) * 0.995  # Tight stop below deviation point
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "range boundary deviation reversal (liquidity grab completed)",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["range_deviation_reversal", "institutional_accumulation", "liquidity_grab_reversal", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "range_mean_reversion_long" in setup_type:
        basis = "range_mean_reversion_long"
        stop_structure = _pivot_swing_low(df_sess, 8) * 0.992  # Wider stop for mean reversion
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "range mean reversion from oversold levels",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["range_mean_reversion", "oversold_bounce", "institutional_buying", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Squeeze Release Long (Volatility Expansion)
    if "squeeze_release_long" in setup_type:
        basis = "squeeze_release_long"
        stop_structure = _pivot_swing_low(df_sess, 15) * 0.995  # Tight stop after consolidation
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "volatility expansion breakout with volume confirmation",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["volatility_expansion", "squeeze_release", "institutional_breakout", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Gap Fill Long (Gap Fade Strategy)
    if "gap_fill_long" in setup_type:
        basis = "gap_fill_long"
        stop_structure = _pivot_swing_low(df_sess, 8) * 0.990  # Stop below gap low
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "gap fill approach with weakening momentum",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["gap_fill", "mean_reversion", "institutional_fade", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Flag Continuation Long (Trend Continuation)
    if "flag_continuation_long" in setup_type and regime == "trend_up":
        basis = "flag_continuation_long"
        stop_structure = _pivot_swing_low(df_sess, 10) * 0.992  # Stop below flag pattern
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "flag pattern breakout in trend direction",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_continuation", "flag_breakout", "institutional_follow_through", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Support Bounce Long (Key Level Reaction)
    if "support_bounce_long" in setup_type:
        basis = "support_bounce_long"
        stop_structure = _pivot_swing_low(df_sess, 6) * 0.985  # Tight stop below support
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "rejection candle at key support with volume",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["support_bounce", "level_reaction", "institutional_defense", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Volume Spike Reversal Long (Climactic Exhaustion)
    if "volume_spike_reversal_long" in setup_type:
        basis = "volume_spike_reversal_long"
        stop_structure = _pivot_swing_low(df_sess, 5) * 0.988  # Stop below spike low
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "reversal signal after climactic volume spike",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["volume_climax", "exhaustion_reversal", "institutional_absorption", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Trend Pullback Long (Trend Continuation)
    if "trend_pullback_long" in setup_type and regime == "trend_up":
        basis = "trend_pullback_long"
        stop_structure = _pivot_swing_low(df_sess, 12) * 0.994  # Stop below pullback low
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "pullback completion in established uptrend",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_pullback", "continuation", "institutional_accumulation", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Range Rejection Long (Range Boundary Defense)
    if "range_rejection_long" in setup_type:
        basis = "range_rejection_long"
        stop_structure = _pivot_swing_low(df_sess, 8) * 0.990  # Stop below range low
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "rejection at range boundary with institutional volume",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["range_rejection", "boundary_defense", "institutional_support", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: ORB Breakout Long (Opening Range Breakout)
    if "orb_breakout_long" in setup_type and close > orh:
        basis = "orb_breakout_long"
        stop_structure = max(orl, _pivot_swing_low(df_sess, 8)) * 0.995  # Stop below OR or swing low
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "opening range high breakout with volume expansion",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["orb_breakout", "volume_expansion", "institutional_breakout", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: VWAP Mean Reversion Long
    if "vwap_mean_reversion_long" in setup_type and close < vwap.iloc[-1]:
        basis = "vwap_mean_reversion_long"
        stop_structure = _pivot_swing_low(df_sess, 10) * 0.992  # Stop below recent swing
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "mean reversion from oversold levels to VWAP",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["vwap_mean_reversion", "oversold_bounce", "value_reversion", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # No fallback - log unhandled setup and return None
    logger.warning(f"UNHANDLED_LONG_SETUP: {setup_type} - no specific handler found, rejecting trade")
    return None

def _select_short_strategy(
    df_sess: pd.DataFrame, cfg: PlannerConfig, regime: str,
    orh: float, orl: float, or_end: pd.Timestamp,
    vwap: pd.Series, pd_levels: Dict[str, float],
    close: float, ema20: pd.Series, ema50: pd.Series,
    setup_type: str
) -> Dict[str, Any]:
    """Select appropriate short strategy based on market conditions and setup type."""

    # Strategy 1: PDL Break + Hold Short - prioritize for range breakdown setups
    if (pd_levels.get("PDL") is not None and not np.isnan(pd_levels.get("PDL"))) \
       and close < pd_levels["PDL"] and close < vwap.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        basis = "pdl_break_hold_short"
        stop_structure = min(orh, _pivot_swing_high(df_sess, 10))
        return {
            "name": basis, "bias": "short",
            "entry_trigger": f"retest/hold below PDL({pd_levels['PDL']:.2f})",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["below_PDL","below_VWAP","ema20<ema50", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 2: Range Break + Retest Short - for breakdown setups
    if regime == "trend_down" and close < orl and close < vwap.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        basis = "range_break_retest_short"
        stop_structure = min(orh, _pivot_swing_high(df_sess, 12))
        return {
            "name": basis, "bias": "short",
            "entry_trigger": f"retest of ~ORL({orl:.2f}) rejecting under VWAP",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_down", "below_ORL", "below_VWAP", "ema20<ema50", f"setup:{setup_type}"],
                        "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 3: Momentum Breakout Short - for momentum-based setups
    if "momentum_breakout_short" in setup_type and close < vwap.iloc[-1]:
        basis = "momentum_breakout_short"
        stop_structure = _pivot_swing_high(df_sess, 6)  # Tighter stops for momentum
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "momentum breakdown with volume surge, enter on minor bounce",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["momentum_breakdown", "below_VWAP", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 4: Trend Continuation Short - for trend following setups
    if "trend_continuation_short" in setup_type and regime == "trend_down" and ema20.iloc[-1] < ema50.iloc[-1]:
        basis = "trend_continuation_short"
        stop_structure = min(_pivot_swing_high(df_sess, 8), ema20.iloc[-1] * 1.015)  # EMA or swing high
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "trend continuation pattern, enter on minor bounce to EMA20",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_down", "ema20<ema50", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # INSTITUTIONAL CONCEPTS - Short strategies
    if "order_block_short" in setup_type:
        basis = "order_block_short"
        stop_structure = _pivot_swing_high(df_sess, 5) * 1.002  # Tight stop above order block
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "order block resistance test with rejection",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["order_block_resistance", "institutional_distribution", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "fair_value_gap_short" in setup_type:
        basis = "fair_value_gap_short"
        stop_structure = close * 1.004  # Stop above FVG zone
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "fair value gap fill completion",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["fair_value_gap_fill", "price_imbalance_correction", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "liquidity_sweep_short" in setup_type:
        basis = "liquidity_sweep_short"
        stop_structure = _pivot_swing_high(df_sess, 3) * 1.003  # Stop above swept level
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "post-sweep reversal with volume confirmation",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["liquidity_sweep_reversal", "stop_hunt_complete", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "premium_zone_short" in setup_type:
        basis = "premium_zone_short"
        stop_structure = _pivot_swing_high(df_sess, 8) * 1.005  # Wider stop for value zone
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "premium zone value selling opportunity",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["premium_zone_sell", "institutional_distribution_area", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "break_of_structure_short" in setup_type and close < vwap.iloc[-1]:
        basis = "break_of_structure_short"
        stop_structure = _pivot_swing_high(df_sess, 5) * 1.002  # Stop above broken structure
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "break of structure continuation",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["break_of_structure", "trend_change_down", "below_VWAP", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "change_of_character_short" in setup_type:
        basis = "change_of_character_short"
        stop_structure = _pivot_swing_high(df_sess, 3) * 1.003  # Tight stop for reversal
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "change of character reversal signal",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["change_of_character", "early_reversal_signal", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "equilibrium_breakout_short" in setup_type and close < vwap.iloc[-1]:
        basis = "equilibrium_breakout_short"
        stop_structure = vwap.iloc[-1] * 1.002  # Stop above VWAP/equilibrium
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "equilibrium breakout below balance",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["equilibrium_breakout", "balance_resolution_down", "below_VWAP", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # INSTITUTIONAL RANGE TRADING STRATEGIES - New professional setups for choppy markets
    if "range_deviation_short" in setup_type:
        basis = "range_deviation_short"
        stop_structure = _pivot_swing_high(df_sess, 5) * 1.005  # Tight stop above deviation point
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "range boundary deviation reversal (liquidity grab completed)",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["range_deviation_reversal", "institutional_distribution", "liquidity_grab_reversal", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    if "range_mean_reversion_short" in setup_type:
        basis = "range_mean_reversion_short"
        stop_structure = _pivot_swing_high(df_sess, 8) * 1.008  # Wider stop for mean reversion
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "range mean reversion from overbought levels",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["range_mean_reversion", "overbought_rejection", "institutional_selling", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Squeeze Release Short (Volatility Expansion)
    if "squeeze_release_short" in setup_type:
        basis = "squeeze_release_short"
        stop_structure = _pivot_swing_high(df_sess, 15) * 1.005  # Tight stop after consolidation
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "volatility expansion breakdown with volume confirmation",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["volatility_expansion", "squeeze_release", "institutional_breakdown", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Gap Fill Short (Gap Fade Strategy)
    if "gap_fill_short" in setup_type:
        basis = "gap_fill_short"
        stop_structure = _pivot_swing_high(df_sess, 8) * 1.010  # Stop above gap high
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "gap fill approach with weakening momentum",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["gap_fill", "mean_reversion", "institutional_fade", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Flag Continuation Short (Trend Continuation)
    if "flag_continuation_short" in setup_type and regime == "trend_down":
        basis = "flag_continuation_short"
        stop_structure = _pivot_swing_high(df_sess, 10) * 1.008  # Stop above flag pattern
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "flag pattern breakdown in trend direction",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_continuation", "flag_breakdown", "institutional_follow_through", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Resistance Bounce Short (Key Level Reaction)
    if "resistance_bounce_short" in setup_type:
        basis = "resistance_bounce_short"
        stop_structure = _pivot_swing_high(df_sess, 6) * 1.015  # Tight stop above resistance
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "rejection candle at key resistance with volume",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["resistance_bounce", "level_reaction", "institutional_defense", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Volume Spike Reversal Short (Climactic Exhaustion)
    if "volume_spike_reversal_short" in setup_type:
        basis = "volume_spike_reversal_short"
        stop_structure = _pivot_swing_high(df_sess, 5) * 1.012  # Stop above spike high
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "reversal signal after climactic volume spike",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["volume_climax", "exhaustion_reversal", "institutional_distribution", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Trend Pullback Short (Trend Continuation)
    if "trend_pullback_short" in setup_type and regime == "trend_down":
        basis = "trend_pullback_short"
        stop_structure = _pivot_swing_high(df_sess, 12) * 1.006  # Stop above pullback high
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "pullback completion in established downtrend",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_pullback", "continuation", "institutional_distribution", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: Range Rejection Short (Range Boundary Defense)
    if "range_rejection_short" in setup_type:
        basis = "range_rejection_short"
        stop_structure = _pivot_swing_high(df_sess, 8) * 1.010  # Stop above range high
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "rejection at range boundary with institutional volume",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["range_rejection", "boundary_defense", "institutional_resistance", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: ORB Breakout Short (Opening Range Breakdown)
    if "orb_breakout_short" in setup_type and close < orl:
        basis = "orb_breakout_short"
        stop_structure = min(orh, _pivot_swing_high(df_sess, 8)) * 1.005  # Stop above OR or swing high
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "opening range low breakdown with volume expansion",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["orb_breakdown", "volume_expansion", "institutional_breakdown", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Professional Setup: VWAP Mean Reversion Short
    if "vwap_mean_reversion_short" in setup_type and close > vwap.iloc[-1]:
        basis = "vwap_mean_reversion_short"
        stop_structure = _pivot_swing_high(df_sess, 10) * 1.008  # Stop above recent swing
        return {
            "name": basis, "bias": "short",
            "entry_trigger": "mean reversion from overbought levels to VWAP",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["vwap_mean_reversion", "overbought_rejection", "value_reversion", f"setup:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # No fallback - log unhandled setup and return None
    logger.warning(f"UNHANDLED_SHORT_SETUP: {setup_type} - no specific handler found, rejecting trade")
    return None

def _strategy_selector(
    df_sess: pd.DataFrame,
    cfg: PlannerConfig,
    regime: str,
    orh: float, orl: float, or_end: pd.Timestamp,
    vwap: pd.Series,
    pd_levels: Dict[str, float],
    setup_type: str,
    index_df5m: Optional[pd.DataFrame] = None,
    daily_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Pick a single best structure-led setup candidate."""
    last = df_sess.iloc[-1]
    close = float(last["close"])

    ema20 = _ema(df_sess["close"], 20)
    ema50 = _ema(df_sess["close"], 50)

    # SETUP-AWARE STRATEGY SELECTION - Match strategy bias to detected setup direction
    if not setup_type:
        raise ValueError("setup_type is required for strategy selection")

    # Long setups should only get long strategies
    if setup_type.endswith("_long"):
        return _select_long_strategy(df_sess, cfg, regime, orh, orl, or_end, vwap, pd_levels, close, ema20, ema50, setup_type, index_df5m, daily_df)
    # Short setups should only get short strategies
    elif setup_type.endswith("_short"):
        return _select_short_strategy(df_sess, cfg, regime, orh, orl, or_end, vwap, pd_levels, close, ema20, ema50, setup_type)
    else:
        # Handle edge case of unknown setup types
        return {
            "name": "unknown_setup", "bias": "flat", "entry_trigger": "wait",
            "structure_stop": np.nan,
            "context": {"reason": [f"unknown_setup_type:{setup_type}"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

# ----------------------------
# Exits & sizing
# ----------------------------

def _compose_exits_and_size(
    price: float, bias: str, atr: float, structure_stop: float, cfg: PlannerConfig, qty_scale: float
) -> Dict[str, Any]:
    """
    Compose hard stop, targets by RR, and position size. Strictly config-driven.
    Also enforces reasonable risk-per-share floors via cfg._extras['planner_precision'].
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
        vol_stop = price - cfg.sl_atr_mult * atr if bias == "long" else price + cfg.sl_atr_mult * atr
        if bias == "long":
            hard_sl = max(structure_stop - cfg.sl_below_swing_ticks, vol_stop)
            rps = max(price - hard_sl, 0.0)
        else:
            hard_sl = min(structure_stop + cfg.sl_below_swing_ticks, vol_stop)
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

                    # Define volatility thresholds
                    low_vol_threshold = volatility_config.get('low_volatility_threshold', 0.5)   # 0.5% ATR/price
                    high_vol_threshold = volatility_config.get('high_volatility_threshold', 2.0) # 2.0% ATR/price

                    # Volatility multipliers
                    low_vol_mult = volatility_config.get('low_volatility_multiplier', 1.3)   # Increase size in low vol
                    high_vol_mult = volatility_config.get('high_volatility_multiplier', 0.7)  # Decrease size in high vol
                    normal_vol_mult = volatility_config.get('normal_volatility_multiplier', 1.0)

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

        # Sizing with volatility adjustment
        base_qty = max(int(cfg.risk_per_trade_rupees // rps), 0)
        qty = max(int(base_qty * qty_scale * volatility_multiplier), 0)
        notional = qty * price

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
# Public API
# ----------------------------

def generate_trade_plan(
    df: pd.DataFrame,
    symbol: str,
    daily_df: Optional[pd.DataFrame] = None,
    setup_type: str = None,
    # -------- NEW OPTIONALS to compute HCET features ----------
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
    setup_type : str | None
        Detected setup type from structure events (e.g., "range_rejection_long", "breakout_short").
        Used to ensure strategy direction aligns with detected setup bias.
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
        **NEW**: includes `plan["features"]` when optional inputs are provided, for Gate HCET consumption.
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
        sess["ema20"] = _ema(sess["close"], 20)
        sess["ema50"] = _ema(sess["close"], 50)

        # Optional indicators for soft sizing / checklist
        rsi_len = int(cfg._extras.get("intraday_params", {}).get("rsi_len", 14))
        adx_len = int(cfg._extras.get("intraday_params", {}).get("adx_len", 14))
        sess["rsi14"] = _rsi(sess["close"], period=rsi_len)
        sess["adx14"] = _adx(sess, period=adx_len)
        macd_pack = _macd(sess["close"])
        sess["macd"] = macd_pack["line"]
        sess["macd_hist"] = macd_pack["hist"]
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
        strat = _strategy_selector(sess, cfg, regime, orh, orl, or_end, sess["vwap"], pd_levels, setup_type, index_df5m, daily_df)

        if strat is None:
            logger.info(f"planner: {symbol} setup_rejected={setup_type} (regime={regime}, ORH={orh:.2f}, ORL={orl:.2f})")
            return {
                "eligible": False,
                "reason": "setup_conditions_not_met",
                "quality": {"rejection_reason": f"setup_conditions_not_met:{setup_type}"},
                "notes": {"cautions": [f"rejected_{setup_type}", f"regime_{regime}"]}
            }

        if strat["name"] == "no_setup":
            logger.info(f"planner: {symbol} no_setup (regime={regime}, ORH={orh:.2f}, ORL={orl:.2f})")
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

        entry_ref_price = float(sess.iloc[-1]["close"])
        exits = _compose_exits_and_size(entry_ref_price, strat["bias"], atr, strat["structure_stop"], cfg, qty_scale=qty_scale)

        # Feasibility tightening (from planner_precision extras)
        pp = cfg._extras.get("planner_precision", {})
        measured_move = max(orh - orl, atr) if strat["bias"] in ("long","short") else atr
        rps = float(exits.get("risk_per_share", 1e-6))
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

        eps = max(entry_ref_price * 0.001, 0.01)
        if (not rps) or rps <= 0:
            # coarse salvage if RPS not computed
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

        # Calculate structural risk-reward based on expected measured move
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

        # Handle negative structural RR (indicates setup-strategy directional mismatch)
        if not np.isnan(structural_rr):
            if structural_rr < 0:
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
                "vwap": round(float(sess["vwap"].iloc[-1]), 2),
                "ema20": round(float(sess["ema20"].iloc[-1]), 2),
                "ema50": round(float(sess["ema50"].iloc[-1]), 2),
                "atr": None if np.isnan(atr) else round(float(atr), 2),
                "rsi14": round(float(sess["rsi14"].iloc[-1]), 2),
                "adx14": round(float(sess["adx14"].iloc[-1]), 2),
                "macd_hist": round(float(sess["macd_hist"].iloc[-1]), 4),
                "vol_ratio": round(float(sess["vol_ratio"].iloc[-1]), 2),
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
                min_structural_rr = quality_filters.get("min_structural_rr", 2.0)

                # Professional quality filters - reject poor setups at planning stage
                if plan["eligible"]:

                    # Filter: targets must be feasible
                    if not plan["quality"]["t1_feasible"]:
                        plan["eligible"] = False
                        plan["quality"]["rejection_reason"] = "T1 not feasible"
                        logger.info(f"planner: {symbol} rejected due to T1 not feasible")

                    # Filter: minimum target risk-reward ratios
                    if len(plan["targets"]) >= 2:
                        t1_rr = plan["targets"][0].get("rr", 0)

                        min_t1_rr = quality_filters.get("min_t1_rr", 1.0)
                        if t1_rr < min_t1_rr:
                            plan["eligible"] = False
                            plan["quality"]["rejection_reason"] = f"T1_rr {t1_rr:.1f} < {min_t1_rr}"
                            logger.info(f"planner: {symbol} rejected due to low T1_rr {t1_rr:.1f}")

                    # Filter: ADX requirement for breakout setups
                    if plan["eligible"] and "breakout" in plan["strategy"]:
                        breakout_min_adx = quality_filters.get("breakout_min_adx", 20.0)
                        current_adx = plan["indicators"].get("adx14", 0.0)

                        if current_adx < breakout_min_adx:
                            plan["eligible"] = False
                            plan["quality"]["rejection_reason"] = f"ADX {current_adx:.1f} < {breakout_min_adx} (breakout filter)"
                            logger.info(f"planner: {symbol} rejected breakout setup due to low ADX {current_adx:.1f}")

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
