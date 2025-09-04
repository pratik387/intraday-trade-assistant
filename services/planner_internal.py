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

from config.logging_config import get_loggers
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index

logger, _ = get_loggers()

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

def _prev_day_levels(daily_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Return PDH/PDL/PDC from the previous session (NaN if unavailable)."""
    try:
        if daily_df is None or daily_df.empty:
            return {"PDH": np.nan, "PDL": np.nan, "PDC": np.nan}
        d = daily_df.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date")
        elif isinstance(d.index, pd.DatetimeIndex):
            d = d.sort_index()
        else:
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()
        if len(d) < 2:
            return {"PDH": np.nan, "PDL": np.nan, "PDC": np.nan}
        prev = d.iloc[-1:]
        pdh = float(prev["high"].iloc[0])
        pdl = float(prev["low"].iloc[0])
        pdc = float(prev["close"].iloc[0])
        return {"PDH": pdh, "PDL": pdl, "PDC": pdc}
    except Exception as e:
        logger.exception(f"planner.prev_day_levels error: {e}")
        return {"PDH": np.nan, "PDL": np.nan, "PDC": np.nan}

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

def _strategy_selector(
    df_sess: pd.DataFrame,
    cfg: PlannerConfig,
    regime: str,
    orh: float, orl: float, or_end: pd.Timestamp,
    vwap: pd.Series,
    pd_levels: Dict[str, float]
) -> Dict[str, Any]:
    """Pick a single best structure-led setup candidate."""
    last = df_sess.iloc[-1]
    close = float(last["close"])

    ema20 = _ema(df_sess["close"], 20)
    ema50 = _ema(df_sess["close"], 50)

    # Strategy 1: ORB Pullback Long (trend_up)
    if regime == "trend_up" and close > orh and close > vwap.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        basis = "orb_pullback_long"
        stop_structure = max(orl, _pivot_swing_low(df_sess, 12))
        return {
            "name": basis,
            "bias": "long",
            "entry_trigger": f"pullback to ~ORH({orh:.2f}) with hold above VWAP",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_up", "above_ORH", "above_VWAP", "ema20>ema50"],
                        "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 2: VWAP Reclaim Long
    if close > vwap.iloc[-1] and df_sess["close"].tail(cfg.vwap_reclaim_min_bars_above).gt(
        vwap.tail(cfg.vwap_reclaim_min_bars_above)
    ).all():
        basis = "vwap_reclaim_long"
        stop_structure = _pivot_swing_low(df_sess, 8)
        return {
            "name": basis, "bias": "long",
            "entry_trigger": "reclaim VWAP with N closes above, enter on minor dip",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["VWAP_reclaim"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 2b: PDH Break + Hold Long
    if (pd_levels.get("PDH") is not None and not np.isnan(pd_levels.get("PDH"))) \
       and close > pd_levels["PDH"] and close > vwap.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        basis = "pdh_break_hold_long"
        stop_structure = max(orl, _pivot_swing_low(df_sess, 10))
        return {
            "name": basis, "bias": "long",
            "entry_trigger": f"retest/hold above PDH({pd_levels['PDH']:.2f})",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["above_PDH","above_VWAP","ema20>ema50"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 3b: PDL Break + Hold Short
    # Strategy 3b: PDL Break + Hold Short
    if (
        (pd_levels.get("PDL") is not None)
        and (not np.isnan(pd_levels.get("PDL")))
        and (close < pd_levels["PDL"])
        and (close < vwap.iloc[-1])
        and (ema20.iloc[-1] < ema50.iloc[-1])
    ):
        basis = "pdl_break_hold_short"
        stop_structure = min(orh, _pivot_swing_high(df_sess, 10))
        return {
            "name": basis, "bias": "short",
            "entry_trigger": f"retest/hold below PDL({pd_levels['PDL']:.2f})",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["below_PDL","below_VWAP","ema20<ema50"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Strategy 3: Range Break + Retest Short (trend_down)
    if regime == "trend_down" and close < orl and close < vwap.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        basis = "range_break_retest_short"
        stop_structure = min(orh, _pivot_swing_high(df_sess, 12))
        return {
            "name": basis, "bias": "short",
            "entry_trigger": f"retest of ~ORL({orl:.2f}) rejecting under VWAP",
            "structure_stop": float(stop_structure),
            "context": {"reason": ["trend_down", "below_ORL", "below_VWAP", "ema20<ema50"],
                        "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
        }

    # Fallback
    return {
        "name": "no_setup", "bias": "flat", "entry_trigger": "wait",
        "structure_stop": np.nan,
        "context": {"reason": ["no_clear_edge"], "levels": {"ORH": orh, "ORL": orl, **pd_levels}}
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

        # Sizing
        base_qty = max(int(cfg.risk_per_trade_rupees // rps), 0)
        qty = max(int(base_qty * qty_scale), 0)
        notional = qty * price

        # Targets by RR
        t1 = price + (cfg.t1_rr * rps) if bias == "long" else price - (cfg.t1_rr * rps)
        t2 = price + (cfg.t2_rr * rps) if bias == "long" else price - (cfg.t2_rr * rps)

        # Entry zone width (ATR-based unless fallback)
        if fallback_zone:
            entry_zone = [round(price - 0.01, 2), round(price, 2)]
        else:
            entry_width = max(atr * cfg.entry_zone_atr_frac, 0.01)
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
) -> Dict[str, Any]:
    """
    Build an intraday trade plan from a 5-min OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        5m bars with columns [open, high, low, close, volume]; naive IST index expected.
    symbol : str
    config : PlannerConfig | dict | None
        If dict, only extras (intraday_gate / planner_precision / late_entry_penalty / acceptance)
        are applied as overrides; strict knobs always come from JSON.
    daily_df : pd.DataFrame | None
        Daily bars used to compute PDH/PDL/PDC.

    Returns
    -------
    plan : dict
        Serializable plan with eligibility, entry/stop/targets, sizing, context, and quality fields.
    """
    try:
        cfg = _load_planner_config()

        # Focus on the latest session
        df = ensure_naive_ist_index(df)
        sess = _ensure_session_df(df)

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
        pd_levels = _prev_day_levels(daily_df)

        # Gap vs prev close (contextual)
        gap_pct = np.nan
        if not np.isnan(pd_levels.get("PDC", np.nan)):
            first_open = float(sess.iloc[0]["open"]) if not sess.empty else np.nan
            if not np.isnan(first_open):
                gap_pct = 100.0 * (first_open - pd_levels["PDC"]) / max(pd_levels["PDC"], 1e-9)

        regime = _regime(sess, cfg)
        strat = _strategy_selector(sess, cfg, regime, orh, orl, or_end, sess["vwap"], pd_levels)
        
        if strat["name"] == "no_setup":
            logger.info(f"planner: {symbol} no_setup (regime={regime}, ORH={orh:.2f}, ORL={orl:.2f})")
            return {}

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

        if strat["bias"] == "long":
            next_objective = orh + 0.5 * measured_move
            structural_rr = (next_objective - entry_ref_price) / max(rps, 1e-6)
        elif strat["bias"] == "short":
            next_objective = orl - 0.5 * measured_move
            structural_rr = (entry_ref_price - next_objective) / max(rps, 1e-6)
        else:
            next_objective = float("nan")
            structural_rr = float("nan")

        if not np.isnan(structural_rr):
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

        acceptance_ok = bool(retest_ok and hold_ok)

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
                "must": must_checks + (["acceptance_ok"] if acceptance_ok else []),
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
                "acceptance_ok": acceptance_ok,
                "t1_feasible": bool(not np.isnan(t1_orig) and (abs(t1_feasible - t1_orig) < 1e-6 or (abs(t1_feasible - entry_ref_price) <= cap1 + 1e-9))),
                "t2_feasible": bool(not np.isnan(t2_orig) and (abs(t2_feasible - t2_orig) < 1e-6 or (abs(t2_feasible - entry_ref_price) <= cap2 + 1e-9)))
            },
        }
        plan["guardrails"] = [g for g in plan["guardrails"] if g]

        logger.info(
            f"planner.plan sym={symbol} eligible={plan['eligible']} "
            f"bias={plan['bias']} strat={plan['strategy']} qty={plan['sizing']['qty']} "
            f"rr_t1={plan['targets'][0]['rr'] if plan['targets'] else None} "
            f"last_ts={sess.index[-1]}"
        )
        return plan

    except Exception as e:
        logger.exception(f"planner.generate_trade_plan error for {symbol}: {e}")
        return {}
