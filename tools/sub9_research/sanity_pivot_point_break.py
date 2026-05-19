"""Pure-math sanity check on the Daily Pivot Point Break pattern.

PATTERN (intraday 5m bars):
  Daily Pivot Points (classical):
    P  = (PDH + PDL + PDC) / 3
    R1 = 2P - PDL          S1 = 2P - PDH
    R2 = P + (PDH - PDL)   S2 = P - (PDH - PDL)
  where PDH/PDL/PDC are the prior session's high/low/close (derived
  from 5m bars aggregated by session_date).

  Trigger (LONG):  a 5m bar's CLOSE > R1 (first such bar in session)
                   AND bar's volume > 1.5x its symbol-specific
                       20-session 5m-bar avg volume
  Trigger (SHORT): a 5m bar's CLOSE < S1 AND same volume rule

  Entry: breakout bar's close.

  Hard SL: opposite extreme of PRIOR 4 BARS +/- 5 bps. Floor 0.5%.
    LONG  SL = min(low[i-4..i-1]) * (1 - 5bps);  enforce
              (entry - SL) >= 0.005 * entry
    SHORT SL = max(high[i-4..i-1]) * (1 + 5bps); enforce
              (SL - entry) >= 0.005 * entry

  R = |entry - hard_sl|
  T1 (50% qty): next pivot level (R2 if LONG, S2 if SHORT)  -- anchored
  T2 (50% qty): 2.0R from entry
  BE trail: active_sl = entry if t1_hit else hard_sl
  Time stop: 15:10 IST
  Latch: one fire per (symbol, session_date)

UNIVERSE:
  - nse_all.json + fno_liquid_200 + ind_nifty*list.csv (+ feather pool)
  - mis_leverage >= 1.0
  - cap_segment in {large_cap, mid_cap, small_cap}

PERIODS:
  - Discovery: 2024-09-01 .. 2025-09-30
  - OOS:       2025-10-01 .. 2026-04-30

GAUNTLET-V2 SHIP GATES (per cell):
  - n >= 125
  - NET PF >= 1.30
  - Daily Sharpe >= 0.5
  - Per-month winning >= 55%
  - Top-month NET share < 40%

SURVIVOR:
  - n >= 100 AND PF >= 1.20

CELL DIMS:
  - direction        (LONG / SHORT)
  - time_bucket      (morning / midsession / afternoon)
  - cap_segment      (small_cap / mid_cap / large_cap)
  - pivot_strength   (R1_only / broke_R2_S2_same_day)

OOM-SAFE: never use df["date"].dt.date on big frames. Use dt.floor("D")
or vectorized int64 day-floor.

Usage:
    python tools/sub9_research/sanity_pivot_point_break.py [--oos]
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Force UTF-8 stdout on Windows
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.regime_break_detector import check_window  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked pattern params -----------------------------------------------
SL_BUFFER = 0.0005           # 5 bps beyond prior-4-bar extreme
SL_MIN_PCT = 0.005           # 0.5% floor
T2_R_MULT = 2.0
T1_QTY_PCT = 0.5             # 50% out at T1, 50% runs to T2
USE_BE_TRAIL = True
TIME_STOP_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000
VOLUME_MULT = 1.5            # breakout bar vol > 1.5x 20-day avg
PRIOR_BAR_LOOKBACK = 4       # bars used to anchor hard SL
VOL_BASELINE_SESSIONS = 20   # rolling baseline window (sessions)

# Cell dimensions
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}
TIME_BUCKETS = [
    ("morning",   "09:30", "11:30"),
    ("midsession","11:30", "13:30"),
    ("afternoon", "13:30", "15:10"),
]

# Discovery / OOS windows
DISCOVERY_START = date(2024, 9, 1)
DISCOVERY_END   = date(2025, 9, 30)
OOS_START       = date(2025, 10, 1)
OOS_END         = date(2026, 4, 30)

# Ship gates
N_SHIP = 125
PF_SHIP = 1.30
SHARPE_SHIP = 0.5
WIN_MO_PCT_SHIP = 55.0
TOP_MO_PCT_SHIP = 40.0

# Survivor
N_SURV = 100
PF_SURV = 1.20

_KEEP_5M_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


# =========================================================================
# Universe construction
# =========================================================================

def _load_nse_all() -> Dict[str, dict]:
    path = _REPO / "nse_all.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, dict] = {}
    for row in raw:
        sym = str(row.get("symbol", ""))
        if sym.endswith(".NS"):
            sym = sym[:-3]
        if not sym:
            continue
        out[sym] = {
            "mis_leverage": float(row.get("mis_leverage", 0.0) or 0.0),
            "mis_enabled": bool(row.get("mis_enabled", False)),
            "cap_segment": str(row.get("cap_segment", "unknown")),
        }
    return out


def _load_universe_files() -> Set[str]:
    syms: Set[str] = set()
    assets = _REPO / "assets"
    fno = assets / "fno_liquid_200.csv"
    if fno.exists():
        df = pd.read_csv(fno)
        for c in df.columns:
            if "symbol" in c.lower():
                vals = df[c].dropna().astype(str).str.strip()
                vals = vals.str.replace(r"^NSE:", "", regex=True)
                syms |= set(vals.tolist())
    for fp in sorted(assets.glob("ind_nifty*list.csv")):
        try:
            df = pd.read_csv(fp)
            if "Symbol" in df.columns:
                vals = df["Symbol"].dropna().astype(str).str.strip()
                syms |= set(vals.tolist())
        except Exception:
            pass
    return syms


def _sample_feather_symbols() -> Set[str]:
    samples = [
        _REPO / "backtest-cache-download" / "monthly" / "2024_09_5m_enriched.feather",
        _REPO / "backtest-cache-download" / "monthly" / "2025_03_5m_enriched.feather",
        _REPO / "backtest-cache-download" / "monthly" / "2025_09_5m_enriched.feather",
    ]
    syms: Set[str] = set()
    for p in samples:
        if p.exists():
            try:
                df = pd.read_feather(p, columns=["symbol"])
                syms |= set(df["symbol"].astype(str).unique().tolist())
            except Exception:
                pass
    return syms


def build_universe() -> Tuple[Set[str], Dict[str, str]]:
    print("\n[universe] loading nse_all.json ...")
    meta = _load_nse_all()
    print(f"  nse_all entries: {len(meta):,}")

    print("[universe] aggregating fno_liquid_200 + ind_nifty*list ...")
    list_syms = _load_universe_files()
    print(f"  asset list symbols: {len(list_syms):,}")

    print("[universe] sampling broader pool from feathers ...")
    feath_syms = _sample_feather_symbols()
    print(f"  feather pool symbols: {len(feath_syms):,}")

    pool = list_syms | feath_syms
    print(f"  raw pool (union): {len(pool):,}")

    allowed: Set[str] = set()
    cap_map: Dict[str, str] = {}
    skipped_unknown = skipped_micro = skipped_mis = skipped_no_meta = 0
    for s in pool:
        m = meta.get(s)
        if m is None:
            skipped_no_meta += 1
            continue
        if m["mis_leverage"] < 1.0:
            skipped_mis += 1
            continue
        cap = m["cap_segment"]
        if cap == "unknown":
            skipped_unknown += 1
            continue
        if cap == "micro_cap":
            skipped_micro += 1
            continue
        if cap not in ALLOWED_CAPS:
            continue
        allowed.add(s)
        cap_map[s] = cap

    print(f"  drop no_meta:      {skipped_no_meta:,}")
    print(f"  drop mis_lev<1:    {skipped_mis:,}")
    print(f"  drop unknown_cap:  {skipped_unknown:,}")
    print(f"  drop micro_cap:    {skipped_micro:,}")
    print(f"  FINAL allowed:     {len(allowed):,}")
    cap_dist = pd.Series(list(cap_map.values())).value_counts().to_dict()
    print(f"  cap distribution: {cap_dist}")
    return allowed, cap_map


# =========================================================================
# Regime pre-flight
# =========================================================================

def regime_preflight(start: date, end: date, label: str):
    print(f"\n[regime] preflight check on {label} window {start} .. {end}")
    hits = check_window(
        "pivot_point_break",
        ["MIS_leverage", "STT_drag"],
        label,
        start,
        end,
        min_severity="high",
        raise_on_break=False,
    )
    if hits:
        print(f"  WARNING: {len(hits)} rule change(s) within window:")
        for r in hits:
            desc = r.description.encode("ascii", "replace").decode("ascii")
            print(f"    - {r.effective_date} [{r.severity.upper()}] {desc}")
    else:
        print("  clean (no high/critical rule changes)")


# =========================================================================
# Data loading -- OOM-safe (uses dt.floor("D"), not dt.date)
# =========================================================================

def _load_5m_month(yyyy: int, mm: int, allowed: Set[str],
                   cap_map: Dict[str, str]) -> pd.DataFrame:
    path = (_REPO / "backtest-cache-download" / "monthly"
            / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather")
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_feather(path, columns=_KEEP_5M_COLS)
    except Exception:
        return pd.DataFrame()
    df = df[df["symbol"].isin(allowed)]
    if df.empty:
        return df
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    df["cap_segment"] = df["symbol"].map(cap_map).astype("category")
    return df


def _months_between(start: date, end: date) -> List[Tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _months_between_with_buffer(start: date, end: date,
                                buffer_months: int) -> List[Tuple[int, int]]:
    """Adds N months before `start` so we have priors for pivot/vol baseline."""
    y, m = start.year, start.month
    for _ in range(buffer_months):
        m -= 1
        if m < 1:
            m = 12
            y -= 1
    return _months_between(date(y, m, 1), end)


def load_period_5m(start: date, end: date, allowed: Set[str],
                   cap_map: Dict[str, str],
                   include_prior_months: int = 2) -> pd.DataFrame:
    """Load 5m bars; include `include_prior_months` extra months before
    `start` to bootstrap PDH/PDL/PDC + 20-session vol baseline."""
    print(f"\n[load] 5m bars {start} .. {end} (with {include_prior_months}mo prior buffer) ...")
    parts: List[pd.DataFrame] = []
    for y, m in _months_between_with_buffer(start, end, include_prior_months):
        df = _load_5m_month(y, m, allowed, cap_map)
        if not df.empty:
            parts.append(df)
            print(f"    {y}-{m:02d}: {len(df):,} bars")
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)

    # Vectorized day-floor (OOM-safe — never use .dt.date on big df)
    ts_ns = big["date"].astype("datetime64[ns]")
    ns = ts_ns.values.view("int64")
    NS_PER_DAY = 86_400 * 1_000_000_000
    NS_PER_MIN = 60 * 1_000_000_000
    day_floor_ns = (ns // NS_PER_DAY) * NS_PER_DAY
    big["_day"] = pd.to_datetime(day_floor_ns)         # datetime64[ns]
    minutes = ((ns - day_floor_ns) // NS_PER_MIN).astype("int32")
    hours = (minutes // 60).astype("int8")
    mins = (minutes % 60).astype("int8")
    h_str = np.char.zfill(hours.astype("U2"), 2)
    m_str = np.char.zfill(mins.astype("U2"), 2)
    big["hhmm"] = np.char.add(np.char.add(h_str, ":"), m_str)
    # Filter to (start - buffer) .. end — we keep prior buffer in.
    # Note: reset_index(drop=True) was OOM'ing on 27M rows (it forces a deep
    # copy of the datetime64 array, ~209 MB). Sort by `date` only (smaller
    # memory footprint than 3-key sort) and skip the reset since downstream
    # uses groupby(symbol, _day) which doesn't care about index alignment.
    big.sort_values(["symbol", "date"], kind="mergesort", inplace=True)
    print(f"  total bars loaded: {len(big):,}  | symbols: {big['symbol'].nunique():,}")
    return big


# =========================================================================
# Session aggregates + pivots + vol baseline
# =========================================================================

def compute_session_aggs(big5m: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, _day): session_high, session_low, session_close,
    avg_5m_volume, then PDH/PDL/PDC + 20-session rolling vol baseline.

    Returns DataFrame indexed by (symbol, _day) with columns:
        pdh, pdl, pdc, P, R1, S1, R2, S2, vol_baseline_20d
    All shifted so they describe PRIOR-session info (safe for entry-day use).
    """
    print("\n[aggs] computing per-session H/L/C + pivots + vol baseline ...")
    grp = big5m.groupby(["symbol", "_day"], observed=True, sort=False)
    agg = grp.agg(
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_close=("close", "last"),
        avg_5m_vol=("volume", "mean"),
    ).reset_index()
    agg = agg.sort_values(["symbol", "_day"], kind="mergesort").reset_index(drop=True)

    # Shift by 1 per symbol -> prior session info
    by_sym = agg.groupby("symbol", observed=True, sort=False)
    agg["pdh"] = by_sym["session_high"].shift(1)
    agg["pdl"] = by_sym["session_low"].shift(1)
    agg["pdc"] = by_sym["session_close"].shift(1)

    # 20-session rolling mean of 5m-bar avg volume, shifted by 1 so the
    # baseline for session t comes from sessions t-20 .. t-1.
    agg["vol_baseline_20d"] = (
        by_sym["avg_5m_vol"]
        .apply(lambda s: s.shift(1).rolling(VOL_BASELINE_SESSIONS, min_periods=10).mean())
        .reset_index(level=0, drop=True)
    )

    # Pivots
    agg["P"]  = (agg["pdh"] + agg["pdl"] + agg["pdc"]) / 3.0
    agg["R1"] = 2 * agg["P"] - agg["pdl"]
    agg["S1"] = 2 * agg["P"] - agg["pdh"]
    agg["R2"] = agg["P"] + (agg["pdh"] - agg["pdl"])
    agg["S2"] = agg["P"] - (agg["pdh"] - agg["pdl"])

    # Keep only rows with pdh/pdl/pdc + baseline available
    before = len(agg)
    agg = agg.dropna(subset=["pdh", "pdl", "pdc", "vol_baseline_20d"]).reset_index(drop=True)
    print(f"  sessions w/ priors+baseline: {len(agg):,} (dropped {before - len(agg):,} for missing priors)")
    return agg


# =========================================================================
# Event detection: pivot break with volume confirmation
# =========================================================================

def _time_bucket(hhmm: str) -> Optional[str]:
    for label, lo, hi in TIME_BUCKETS:
        if lo <= hhmm < hi:
            return label
    return None


def find_events(big5m: pd.DataFrame, aggs: pd.DataFrame,
                period_start: date, period_end: date) -> pd.DataFrame:
    """Scan each (symbol, session) for first qualifying pivot break.

    Constraints:
      - Bar's close must cross R1 (LONG) or S1 (SHORT). "Cross" means
        prior bar's close was NOT beyond the level, current bar's close IS.
        (For the first bar of session we use just: close > R1 / close < S1.)
      - Bar's volume > VOLUME_MULT * vol_baseline_20d
      - Bar is within an allowed time bucket (09:30 .. 15:10)
      - At least PRIOR_BAR_LOOKBACK bars exist before the trigger in
        the SAME session (for SL anchoring)
      - Latched: first qualifying bar per (symbol, day)
    """
    print("\n[events] scanning pivot point breaks ...")

    # Build lookup: (symbol, day) -> agg row index
    aggs_idx = aggs.set_index(["symbol", "_day"])

    # Restrict bars to within the target period (drop the prior-month buffer).
    # OOM-safe: skip reset_index — boundary computation below uses .values
    # which is index-agnostic, so range-index reset is unnecessary.
    period_start_ts = pd.Timestamp(period_start)
    period_end_ts = pd.Timestamp(period_end)
    bars = big5m[(big5m["_day"] >= period_start_ts) & (big5m["_day"] <= period_end_ts)]
    print(f"  bars in target period: {len(bars):,}")

    # Group walk per (symbol, _day).
    # OOM-safe: avoid list(zip(...)) which materializes 23M Python tuples
    # (~900 MB). Use change-detection on each key array independently.
    sym_arr = bars["symbol"].values
    day_arr = bars["_day"].values
    # Boundary = where EITHER symbol OR day changes from prior row.
    sym_changed = np.r_[True, sym_arr[1:] != sym_arr[:-1]]
    day_changed = np.r_[True, day_arr[1:] != day_arr[:-1]]
    grp_starts = np.where(sym_changed | day_changed)[0]
    starts = grp_starts
    ends = np.concatenate([grp_starts[1:], [len(bars)]])

    high_arr = bars["high"].values.astype("float32")
    low_arr = bars["low"].values.astype("float32")
    close_arr = bars["close"].values.astype("float32")
    vol_arr = bars["volume"].values.astype("float32")
    hhmm_arr = bars["hhmm"].values
    ts_arr = bars["date"].values
    cap_arr = bars["cap_segment"].astype(str).values

    events: List[dict] = []
    n_no_priors = 0

    for grp_idx in range(len(starts)):
        s = int(starts[grp_idx]); e = int(ends[grp_idx])
        sym = str(sym_arr[s]); day = day_arr[s]
        try:
            row = aggs_idx.loc[(sym, day)]
        except KeyError:
            n_no_priors += 1
            continue
        R1 = float(row["R1"]); S1 = float(row["S1"])
        R2 = float(row["R2"]); S2 = float(row["S2"])
        vol_base = float(row["vol_baseline_20d"])
        vol_threshold = VOLUME_MULT * vol_base
        if not np.isfinite(R1) or not np.isfinite(S1) or vol_base <= 0:
            n_no_priors += 1
            continue

        latched = False
        for i in range(s + PRIOR_BAR_LOOKBACK, e):
            b_close = float(close_arr[i])
            b_vol = float(vol_arr[i])
            b_hhmm = hhmm_arr[i]
            tb = _time_bucket(b_hhmm)
            if tb is None:
                continue
            # Vol confirmation
            if b_vol <= vol_threshold:
                continue
            # Cross: prior close NOT beyond level AND current close IS beyond
            prev_close = float(close_arr[i - 1])
            direction = None
            if (prev_close <= R1) and (b_close > R1):
                direction = "LONG"
            elif (prev_close >= S1) and (b_close < S1):
                direction = "SHORT"
            else:
                continue

            # SL anchored on prior 4 bars
            lo_window = float(min(low_arr[i - PRIOR_BAR_LOOKBACK:i]))
            hi_window = float(max(high_arr[i - PRIOR_BAR_LOOKBACK:i]))
            entry = b_close
            if direction == "LONG":
                hard_sl = lo_window * (1 - SL_BUFFER)
                # Enforce 0.5% floor
                if entry - hard_sl < SL_MIN_PCT * entry:
                    hard_sl = entry * (1 - SL_MIN_PCT)
                stop_dist = entry - hard_sl
                if stop_dist <= 0:
                    continue
                t1 = R2                              # anchored to next pivot
                t2 = entry + T2_R_MULT * stop_dist
                # If R2 is below entry (degenerate pivot geometry), skip
                if t1 <= entry:
                    continue
            else:
                hard_sl = hi_window * (1 + SL_BUFFER)
                if hard_sl - entry < SL_MIN_PCT * entry:
                    hard_sl = entry * (1 + SL_MIN_PCT)
                stop_dist = hard_sl - entry
                if stop_dist <= 0:
                    continue
                t1 = S2
                t2 = entry - T2_R_MULT * stop_dist
                if t1 >= entry:
                    continue

            events.append({
                "symbol": sym,
                "_day": day,
                "cap_segment": cap_arr[i],
                "direction": direction,
                "time_bucket": tb,
                "entry_ts": ts_arr[i],
                "entry_hhmm": b_hhmm,
                "entry_price": entry,
                "P": float(row["P"]),
                "R1": R1, "S1": S1, "R2": R2, "S2": S2,
                "pdh": float(row["pdh"]),
                "pdl": float(row["pdl"]),
                "pdc": float(row["pdc"]),
                "vol_baseline_20d": vol_base,
                "breakout_vol": b_vol,
                "vol_ratio": b_vol / vol_base if vol_base > 0 else 0.0,
                "hard_sl": hard_sl,
                "stop_distance": stop_dist,
                "t1_target": t1,
                "t2_target": t2,
                "post_start_idx": i + 1,
                "session_end_idx": e,
                "trigger_idx": i,
                "session_start_idx": s,
            })
            latched = True
            break

    print(f"  events found (latched 1 per symbol-day): {len(events):,}")
    print(f"  groups skipped (no priors / bad baseline): {n_no_priors:,}")
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events)


# =========================================================================
# Pivot strength labeller (uses session-future data; informational only)
# =========================================================================

def label_pivot_strength(events: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    """For each event, tag whether the session also broke R2/S2.

    LONG event:  any bar at i_trigger .. session_end with close > R2.
    SHORT event: any bar at i_trigger .. session_end with close < S2.
    """
    if events.empty:
        return events

    # bars frame already lost when we restricted to period; rebuild by re-using
    # the aggregated big5m. But we have trigger_idx/session_end_idx referencing
    # the period-restricted `bars` view — which we don't carry here. To make
    # this robust, redo a tiny per-event scan via big5m subset.

    print("\n[pivot_strength] labelling R2/S2 same-day breaks ...")
    strengths: List[str] = []

    # Build per-(symbol,day) bar slice cache
    bm = big5m[["symbol", "_day", "close"]].copy()
    # We'll groupby and pick the relevant session quickly.
    bm_grp = bm.groupby(["symbol", "_day"], observed=True, sort=False)

    for _, ev in events.iterrows():
        sym = ev["symbol"]; day = ev["_day"]
        try:
            sub = bm_grp.get_group((sym, day))
        except KeyError:
            strengths.append("R1_only")
            continue
        closes = sub["close"].values.astype("float32")
        if ev["direction"] == "LONG":
            broke = (closes > float(ev["R2"])).any()
        else:
            broke = (closes < float(ev["S2"])).any()
        strengths.append("broke_R2_S2_same_day" if broke else "R1_only")

    events = events.copy()
    events["pivot_strength"] = strengths
    return events


# =========================================================================
# Simulator
# =========================================================================

def simulate(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Walk forward each event's post-bars (within bars view). bars is the
    period-restricted view used by find_events — we must pass that same
    frame for index alignment."""
    if events.empty:
        return events
    print(f"\n[sim] simulating {len(events):,} events ...")
    high_arr = bars["high"].values.astype("float32")
    low_arr = bars["low"].values.astype("float32")
    close_arr = bars["close"].values.astype("float32")
    hhmm_arr = bars["hhmm"].values
    ts_arr = bars["date"].values

    out: List[dict] = []
    for _, ev in events.iterrows():
        direction = ev["direction"]
        entry_price = float(ev["entry_price"])
        hard_sl = float(ev["hard_sl"])
        stop_dist = float(ev["stop_distance"])
        t1_target = float(ev["t1_target"])
        t2_target = float(ev["t2_target"])
        i_start = int(ev["post_start_idx"])
        i_end = int(ev["session_end_idx"])

        hit_t1 = False
        t1_exit = None
        exit_price = None
        exit_ts = None
        exit_reason = None

        for j in range(i_start, i_end):
            hj = float(high_arr[j]); lj = float(low_arr[j])
            cj = float(close_arr[j]); hhmm_j = hhmm_arr[j]
            ts_j = ts_arr[j]
            active_sl = entry_price if (hit_t1 and USE_BE_TRAIL) else hard_sl
            if direction == "LONG":
                if lj <= active_sl:
                    exit_price = active_sl; exit_ts = ts_j
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if (not hit_t1) and (hj >= t1_target):
                    hit_t1 = True
                    t1_exit = t1_target
                if hit_t1 and (hj >= t2_target):
                    exit_price = t2_target; exit_ts = ts_j; exit_reason = "t2"
                    break
            else:
                if hj >= active_sl:
                    exit_price = active_sl; exit_ts = ts_j
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if (not hit_t1) and (lj <= t1_target):
                    hit_t1 = True
                    t1_exit = t1_target
                if hit_t1 and (lj <= t2_target):
                    exit_price = t2_target; exit_ts = ts_j; exit_reason = "t2"
                    break
            if hhmm_j >= TIME_STOP_HHMM:
                exit_price = cj; exit_ts = ts_j; exit_reason = "time_stop"
                break

        if exit_price is None:
            last_j = i_end - 1
            if last_j < i_start:
                continue
            exit_price = float(close_arr[last_j])
            exit_ts = ts_arr[last_j]
            exit_reason = "session_end"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_dist, 1e-6)), 1)
        side = "BUY" if direction == "LONG" else "SELL"
        if hit_t1:
            qty_t1 = int(qty * T1_QTY_PCT)
            qty_t2 = qty - qty_t1
            if direction == "LONG":
                pnl_t1 = (t1_exit - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2
            else:
                pnl_t1 = (entry_price - t1_exit) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2
            realized = pnl_t1 + pnl_t2
            fee = (calc_fee(entry_price, t1_exit, qty_t1, side)
                   + calc_fee(entry_price, exit_price, qty_t2, side))
            blended = (t1_exit * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            if direction == "LONG":
                realized = (exit_price - entry_price) * qty
            else:
                realized = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, side)
            blended = exit_price
        net_pnl = realized - fee

        out.append({
            "T0_signal_date": pd.Timestamp(ev["_day"]).date(),
            "symbol": "NSE:" + ev["symbol"],
            "bare_symbol": ev["symbol"],
            "cap_segment": ev["cap_segment"],
            "direction": direction,
            "time_bucket": ev["time_bucket"],
            "pivot_strength": ev.get("pivot_strength", "R1_only"),
            "entry_ts": ev["entry_ts"],
            "entry_hhmm": ev["entry_hhmm"],
            "entry_price": entry_price,
            "P": ev["P"],
            "R1": ev["R1"], "S1": ev["S1"],
            "R2": ev["R2"], "S2": ev["S2"],
            "pdh": ev["pdh"], "pdl": ev["pdl"], "pdc": ev["pdc"],
            "vol_baseline_20d": ev["vol_baseline_20d"],
            "breakout_vol": ev["breakout_vol"],
            "vol_ratio": ev["vol_ratio"],
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "stop_distance": stop_dist,
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended,
            "exit_reason": exit_reason,
            "qty": qty,
            "realized_pnl": realized,
            "fee": fee,
            "net_pnl": net_pnl,
        })

    return pd.DataFrame(out)


# =========================================================================
# Reporting
# =========================================================================

def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _agg(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, pf=float("nan"), wr=float("nan"),
                    sharpe=float("nan"), net=0,
                    win_mo=float("nan"), top_mo=float("nan"), n_mo=0)
    pnl = df["net_pnl"]
    n = int(len(pnl))
    pf = _pf(pnl)
    wr = 100.0 * float((pnl > 0).mean())
    daily = df.groupby("T0_signal_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    net = float(pnl.sum())
    df = df.copy()
    df["_mo"] = pd.to_datetime(df["T0_signal_date"]).dt.strftime("%Y-%m")
    monthly = df.groupby("_mo")["net_pnl"].sum()
    n_mo = int(len(monthly))
    win_mo = 100.0 * float((monthly > 0).mean()) if n_mo > 0 else 0.0
    top_mo = (100.0 * float(monthly.abs().max()) / abs(net)) if abs(net) > 1e-6 else 0.0
    return dict(n=n, pf=pf, wr=wr, sharpe=sharpe, net=net,
                win_mo=win_mo, top_mo=top_mo, n_mo=n_mo)


def _ship_verdict(m: dict) -> Tuple[str, List[str]]:
    reasons = []
    if m["n"] < N_SHIP:
        reasons.append(f"n={m['n']}<{N_SHIP}")
    if (not np.isnan(m["pf"])) and m["pf"] < PF_SHIP:
        reasons.append(f"PF={m['pf']:.2f}<{PF_SHIP}")
    if (not np.isnan(m["sharpe"])) and m["sharpe"] < SHARPE_SHIP:
        reasons.append(f"Sh={m['sharpe']:.2f}<{SHARPE_SHIP}")
    if m["win_mo"] < WIN_MO_PCT_SHIP:
        reasons.append(f"win_mo={m['win_mo']:.0f}%<{WIN_MO_PCT_SHIP:.0f}%")
    if m["top_mo"] >= TOP_MO_PCT_SHIP:
        reasons.append(f"top_mo={m['top_mo']:.0f}%>={TOP_MO_PCT_SHIP:.0f}%")
    return ("SHIP", reasons) if not reasons else ("RETIRE", reasons)


def _cell_scan(trades: pd.DataFrame, dims: List[str], max_combo: int = 3):
    """1D / 2D / 3D scan across dims; print ship-eligible + survivor cells."""
    from itertools import combinations
    rows = []
    for k in range(1, max_combo + 1):
        for combo in combinations(dims, k):
            sub = trades.dropna(subset=list(combo))
            for cell_values, g in sub.groupby(list(combo), observed=True):
                m = _agg(g)
                v, _ = _ship_verdict(m)
                if not isinstance(cell_values, tuple):
                    cell_values = (cell_values,)
                cell = " | ".join(f"{c}={v}" for c, v in zip(combo, cell_values))
                rows.append({
                    "dims": ",".join(combo), "k": k, "cell": cell,
                    **m, "verdict": v,
                })
    return pd.DataFrame(rows)


def report(trades: pd.DataFrame, label: str):
    print("\n" + "=" * 78)
    print(f"REPORT -- pivot_point_break -- {label}")
    print("=" * 78)
    if trades.empty:
        print("  NO TRADES")
        return

    n_events = len(trades)
    cap_dist = trades["cap_segment"].value_counts().to_dict()
    dir_dist = trades["direction"].value_counts().to_dict()
    tb_dist = trades["time_bucket"].value_counts().to_dict()
    ps_dist = trades["pivot_strength"].value_counts().to_dict()
    print(f"\n  Funnel:")
    print(f"    traded events: {n_events:,}")
    print(f"    by cap: {cap_dist}")
    print(f"    by dir: {dir_dist}")
    print(f"    by time bucket: {tb_dist}")
    print(f"    by pivot strength: {ps_dist}")

    agg = _agg(trades)
    v, why = _ship_verdict(agg)
    print(f"\n  AGGREGATE:")
    print(f"    n={agg['n']:,}  PF={agg['pf']:.3f}  WR={agg['wr']:.1f}%  "
          f"Sharpe={agg['sharpe']:.3f}  NET=Rs.{agg['net']:,.0f}")
    print(f"    months={agg['n_mo']}  win_mo={agg['win_mo']:.1f}%  "
          f"top_mo={agg['top_mo']:.1f}%")
    print(f"    aggregate verdict: {v}  reasons: {';'.join(why) if why else 'all gates pass'}")

    # Pre-registered 4-dim per-cell table
    print(f"\n  Per-cell (direction x time_bucket x cap_segment x pivot_strength):")
    hdr = (f"  {'dir':<5} {'time':<11} {'cap':<10} {'strength':<22} {'n':>5} "
           f"{'PF':>6} {'WR':>5} {'Sh':>5} {'win_mo':>7} {'top_mo':>7} "
           f"{'NET':>10} {'verdict':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    grouper = ["direction", "time_bucket", "cap_segment", "pivot_strength"]
    for keys, g in trades.groupby(grouper, observed=True):
        dirn, tb, cap, ps = keys
        m = _agg(g)
        sv, _ = _ship_verdict(m)
        pf_str = f"{m['pf']:.2f}" if not np.isnan(m['pf']) and m['pf'] != float("inf") else "inf"
        print(f"  {dirn:<5} {tb:<11} {cap:<10} {ps:<22} {m['n']:>5} "
              f"{pf_str:>6} {m['wr']:>5.1f} {m['sharpe']:>5.2f} "
              f"{m['win_mo']:>6.1f}% {m['top_mo']:>6.1f}% "
              f"{m['net']:>10,.0f} {sv:>8}")

    # Cell mining 1D / 2D / 3D
    print(f"\n  CELL MINING (1D / 2D / 3D) across {grouper}:")
    cells = _cell_scan(trades, grouper, max_combo=3)
    ship_eligible = cells[cells["verdict"] == "SHIP"].sort_values(["pf", "n"], ascending=[False, False])
    survivors = cells[
        (cells["n"] >= N_SURV)
        & (~cells["pf"].isna())
        & (cells["pf"] >= PF_SURV)
        & (cells["verdict"] != "SHIP")
    ].sort_values(["pf", "n"], ascending=[False, False])

    print(f"\n  SHIP-ELIGIBLE cells (all 5 gauntlet-v2 gates): {len(ship_eligible)}")
    for _, r in ship_eligible.head(25).iterrows():
        print(f"    [{r['dims']}] {r['cell']}  "
              f"n={r['n']} PF={r['pf']:.2f} WR={r['wr']:.1f}% "
              f"Sh={r['sharpe']:.2f} win_mo={r['win_mo']:.0f}% "
              f"top_mo={r['top_mo']:.0f}% NET={r['net']:,.0f}")
    print(f"\n  SURVIVOR cells (n>={N_SURV}, PF>={PF_SURV}, not ship): {len(survivors)}")
    for _, r in survivors.head(25).iterrows():
        print(f"    [{r['dims']}] {r['cell']}  "
              f"n={r['n']} PF={r['pf']:.2f} WR={r['wr']:.1f}% "
              f"Sh={r['sharpe']:.2f} win_mo={r['win_mo']:.0f}% "
              f"top_mo={r['top_mo']:.0f}% NET={r['net']:,.0f}")

    # Monthly stability
    print(f"\n  Per-month NET (overall):")
    tcopy = trades.copy()
    tcopy["_mo"] = pd.to_datetime(tcopy["T0_signal_date"]).dt.strftime("%Y-%m")
    monthly = tcopy.groupby("_mo")["net_pnl"].sum()
    abs_net = float(abs(monthly.sum())) or 1.0
    for mo, net in monthly.items():
        share = 100.0 * abs(net) / abs_net
        flag = "  W" if net > 0 else "  L"
        print(f"    {mo}  NET=Rs.{net:>11,.0f}  share={share:5.1f}%  {flag}")

    print(f"\n  Exit-reason mix:")
    for rsn, g in trades.groupby("exit_reason"):
        n2 = len(g); net2 = g["net_pnl"].sum()
        print(f"    {rsn:<18} n={n2:>5} NET=Rs.{net2:>11,.0f}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos", action="store_true",
                        help="Also run OOS window after Discovery")
    args = parser.parse_args()

    print(">>> Daily Pivot Point Break -- PURE MATH SANITY <<<")

    allowed, cap_map = build_universe()
    if not allowed:
        print("[ABORT] empty universe")
        return 2

    regime_preflight(DISCOVERY_START, DISCOVERY_END, "Discovery")
    if args.oos:
        regime_preflight(OOS_START, OOS_END, "OOS")

    # ---- Discovery ----
    print("\n" + "#" * 78)
    print("# DISCOVERY")
    print("#" * 78)
    big5m = load_period_5m(DISCOVERY_START, DISCOVERY_END, allowed, cap_map,
                           include_prior_months=1)
    if big5m.empty:
        print("[ABORT] no discovery bars")
        return 2

    aggs = compute_session_aggs(big5m)

    # Period-restricted bars view used for index-aligned simulation
    period_start_ts = pd.Timestamp(DISCOVERY_START)
    period_end_ts = pd.Timestamp(DISCOVERY_END)
    bars_period = big5m[(big5m["_day"] >= period_start_ts) &
                        (big5m["_day"] <= period_end_ts)].reset_index(drop=True)
    print(f"  bars in target period: {len(bars_period):,}")

    events = find_events(big5m, aggs, DISCOVERY_START, DISCOVERY_END)
    if events.empty:
        print("[ABORT] no events in discovery")
        return 2

    # Label pivot strength
    events = label_pivot_strength(events, big5m)

    # Free memory: big5m no longer needed
    del big5m, aggs
    gc.collect()

    trades = simulate(events, bars_period)
    del bars_period, events
    gc.collect()

    out_dir = _REPO / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pivot_point_break_trades.csv"
    if not trades.empty:
        trades.to_csv(out_csv, index=False)
        print(f"\nTrade log: {out_csv}")

    report(trades, label="Discovery 2024-09 .. 2025-09")

    if not args.oos:
        print("\n[done] OOS not requested. Re-run with --oos if marginal.")
        return 0

    # ---- OOS ----
    print("\n" + "#" * 78)
    print("# OOS")
    print("#" * 78)
    big5m = load_period_5m(OOS_START, OOS_END, allowed, cap_map,
                           include_prior_months=1)
    if big5m.empty:
        print("[ABORT] no OOS bars")
        return 0
    aggs = compute_session_aggs(big5m)
    period_start_ts = pd.Timestamp(OOS_START)
    period_end_ts = pd.Timestamp(OOS_END)
    bars_period = big5m[(big5m["_day"] >= period_start_ts) &
                        (big5m["_day"] <= period_end_ts)].reset_index(drop=True)

    events = find_events(big5m, aggs, OOS_START, OOS_END)
    events = label_pivot_strength(events, big5m) if not events.empty else events
    del big5m, aggs
    gc.collect()

    trades_oos = simulate(events, bars_period) if not events.empty else pd.DataFrame()
    del bars_period, events
    gc.collect()

    out_csv_oos = out_dir / "pivot_point_break_trades_oos.csv"
    if not trades_oos.empty:
        trades_oos.to_csv(out_csv_oos, index=False)
        print(f"\nOOS trade log: {out_csv_oos}")
    report(trades_oos, label="OOS 2025-10 .. 2026-04")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
