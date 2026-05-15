"""Inside Bar Breakout — PROPER 3-period gauntlet (Discovery + OOS + Holdout).

v2 supersedes the original `sanity_inside_bar_breakout.py` which used a
13-month Discovery + treated 2025-10..2026-04 as "OOS" (it was actually
Holdout territory). This script uses the production schema:

  Discovery: 2023-01-01 .. 2024-12-31   (24 months)
  OOS:       2025-01-01 .. 2025-09-30   ( 9 months)
  Holdout:   2025-10-01 .. 2026-04-30   ( 7 months)

PATTERN (identical to v1 — locked):
  - Mother bar at T (5m bar with high H_m, low L_m)
  - Inside bar at T+5: high < H_m AND low > L_m (fully contained)
  - Breakout bar at T+10: closes above H_m (LONG) or below L_m (SHORT)
  - Entry: breakout bar's close
  - Hard SL: opposite extreme of mother + 5 bps buffer; min stop 0.5%
  - T1 (50% qty) = entry +/- 1.0R
  - T2 (50% qty) = entry +/- 2.0R
  - BE trail: active_sl = entry if t1_hit else hard_sl
  - Time stop: 15:10 IST
  - Latch: first qualifying trigger per (symbol, session_date)

OOM-SAFETY (this is the critical fix vs v1):
  v1 loaded ALL 13 months of 5m bars at once -> ~23M rows -> OOM'd Pivot
  and BB sister scripts. v2 processes ONE MONTH AT A TIME for trade
  generation (Inside Bar has no cross-day dependencies; intra-session only).
  Trades dictionaries (lightweight) are accumulated across months and only
  the final trade-log dataframe is held in memory.

  Other OOM rules:
    - use dt.floor("D") not .dt.date on big dataframes
    - sort by ("symbol","date") not ("symbol","d","date") — fewer keys
    - avoid list(zip(...)) on big arrays; use np.r_[True, arr[1:]!=arr[:-1]]
    - skip reset_index(drop=True) on big frames after sort

CELL MINING (ex-ante dimensions ONLY — no post-trade leak):
    direction, cap_segment, time_bucket, mother_size_bucket,
    inside_pos_bucket, R_size_bucket, dow, month, quarter, entry_hour_bucket
  max_combo=3 via `_cell_mine_tier_a.scan_cells`.

GAUNTLET-V2 SHIP GATES (per cell):
  n >= 125, PF >= 1.30, daily Sharpe >= 0.5, win_mo >= 55%, top_mo < 40%.

SL/T SWEEP:
  Run ONLY on Discovery cells that pass all gauntlet gates. Sweep:
    SL_BUFFER × T1_R × T2_R × T1_QTY × TIME_STOP = 4×3×4×4×3 = 576 combos.

Usage:
    python tools/sub9_research/sanity_inside_bar_breakout_v2.py
        [--skip-discovery] [--skip-oos] [--skip-holdout] [--no-sweep]
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
_SUB9 = _REPO / "tools" / "sub9_research"
if str(_SUB9) not in sys.path:
    sys.path.insert(0, str(_SUB9))

from services.regime_break_detector import check_window  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
from _cell_mine_tier_a import scan_cells  # noqa: E402


# ---- Locked DEFAULT pattern params (used in main Discovery run) ----------
SL_BUFFER = 0.0005           # 5 bps beyond mother's opposite extreme
SL_MIN_PCT = 0.005           # 0.5% floor — added vs v1 (was missing)
T1_R_MULT = 1.0
T2_R_MULT = 2.0
T1_QTY_PCT = 0.5
USE_BE_TRAIL = True
TIME_STOP_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000

ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}
TIME_BUCKETS = [
    ("morning",   "09:30", "11:30"),
    ("midsession","11:30", "13:30"),
    ("afternoon", "13:30", "15:10"),
]

# Production 3-period schema
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END   = date(2024, 12, 31)
OOS_START       = date(2025, 1, 1)
OOS_END         = date(2025, 9, 30)
HOLDOUT_START   = date(2025, 10, 1)
HOLDOUT_END     = date(2026, 4, 30)

# Gauntlet-v2 ship gates
N_SHIP = 125
PF_SHIP = 1.30
SHARPE_SHIP = 0.5
WIN_MO_PCT_SHIP = 55.0
TOP_MO_PCT_SHIP = 40.0

# Survivor thresholds (worth investigating)
N_SURV = 100
PF_SURV = 1.20

_KEEP_5M_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]
_OUT_DIR = _REPO / "reports" / "sub9_sanity"


# =========================================================================
# Universe (same as v1)
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
    """Pull broader symbol pool from a few feather files. Sampling 2023, 2024,
    2025 to ensure coverage across the full 24mo Discovery + OOS + Holdout."""
    samples = [
        _REPO / "backtest-cache-download" / "monthly" / "2023_06_5m_enriched.feather",
        _REPO / "backtest-cache-download" / "monthly" / "2024_06_5m_enriched.feather",
        _REPO / "backtest-cache-download" / "monthly" / "2025_06_5m_enriched.feather",
        _REPO / "backtest-cache-download" / "monthly" / "2026_02_5m_enriched.feather",
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
# Regime preflight
# =========================================================================

def regime_preflight(start: date, end: date, label: str):
    print(f"\n[regime] preflight check on {label} window {start} .. {end}")
    hits = check_window(
        "inside_bar_breakout_v2",
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
# OOM-safe per-month month loader
# =========================================================================

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


def _load_one_month(yyyy: int, mm: int, allowed: Set[str],
                    cap_map: Dict[str, str],
                    period_start: date, period_end: date) -> pd.DataFrame:
    """Load one month's 5m bars, filter to universe + clip to (period_start,
    period_end) bounds. Returns a single, sorted, OOM-safe frame.
    """
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

    # Dtypes
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    df["cap_segment"] = df["symbol"].map(cap_map).astype("category")

    # Vectorized day-floor + hhmm (OOM-safe — never .dt.date on big frame)
    ts_ns = df["date"].astype("datetime64[ns]")
    ns = ts_ns.values.view("int64")
    NS_PER_DAY = 86_400 * 1_000_000_000
    NS_PER_MIN = 60 * 1_000_000_000
    day_floor_ns = (ns // NS_PER_DAY) * NS_PER_DAY
    df["_day"] = pd.to_datetime(day_floor_ns)
    minutes = ((ns - day_floor_ns) // NS_PER_MIN).astype("int32")
    hours = (minutes // 60).astype("int8")
    mins = (minutes % 60).astype("int8")
    h_str = np.char.zfill(hours.astype("U2"), 2)
    m_str = np.char.zfill(mins.astype("U2"), 2)
    df["hhmm"] = np.char.add(np.char.add(h_str, ":"), m_str)

    # Clip to period bounds
    p_lo = pd.Timestamp(period_start)
    p_hi = pd.Timestamp(period_end)
    df = df[(df["_day"] >= p_lo) & (df["_day"] <= p_hi)]
    if df.empty:
        return df

    # Sort by (symbol, date). Skip reset_index — downstream uses .values
    # which is index-agnostic.
    df.sort_values(["symbol", "date"], kind="mergesort", inplace=True)
    return df


# =========================================================================
# Event detection + simulator on a single month frame
# =========================================================================

def _time_bucket(hhmm: str) -> Optional[str]:
    for label, lo, hi in TIME_BUCKETS:
        if lo <= hhmm < hi:
            return label
    return None


def _scan_and_simulate(
    bars: pd.DataFrame,
    sl_buffer: float = SL_BUFFER,
    sl_min_pct: float = SL_MIN_PCT,
    t1_r_mult: float = T1_R_MULT,
    t2_r_mult: float = T2_R_MULT,
    t1_qty_pct: float = T1_QTY_PCT,
    use_be_trail: bool = USE_BE_TRAIL,
    time_stop_hhmm: str = TIME_STOP_HHMM,
) -> List[dict]:
    """Combined event finder + simulator. Walks every (symbol, _day) group:
      1. find first qualifying mother+inside+breakout chain
      2. simulate exits forward
    Returns a list of trade dicts.

    Parameterized to support SL/T sweep without reloading bars.
    """
    if bars.empty:
        return []

    # Group boundaries — avoid list(zip(...)) which would OOM on big frames.
    sym_arr = bars["symbol"].values
    day_arr = bars["_day"].values
    sym_changed = np.r_[True, sym_arr[1:] != sym_arr[:-1]]
    day_changed = np.r_[True, day_arr[1:] != day_arr[:-1]]
    grp_starts = np.where(sym_changed | day_changed)[0]
    ends = np.concatenate([grp_starts[1:], [len(bars)]])

    high_arr = bars["high"].values.astype("float32")
    low_arr = bars["low"].values.astype("float32")
    close_arr = bars["close"].values.astype("float32")
    hhmm_arr = bars["hhmm"].values
    ts_arr = bars["date"].values
    cap_arr = bars["cap_segment"].astype(str).values

    trades: List[dict] = []

    for grp_idx in range(len(grp_starts)):
        s = int(grp_starts[grp_idx])
        e = int(ends[grp_idx])
        n_bars = e - s
        if n_bars < 4:  # need mother+inside+breakout+1 forward bar
            continue

        # Find first qualifying chain
        ev = None
        for i in range(s, e - 2):
            mh = float(high_arr[i]); ml = float(low_arr[i])
            ih = float(high_arr[i + 1]); il = float(low_arr[i + 1])
            if not (ih < mh and il > ml):
                continue
            b_close = float(close_arr[i + 2])
            b_hhmm = hhmm_arr[i + 2]
            tb = _time_bucket(b_hhmm)
            if tb is None:
                continue
            if b_close > mh:
                direction = "LONG"
            elif b_close < ml:
                direction = "SHORT"
            else:
                continue

            entry = b_close
            if direction == "LONG":
                hard_sl = ml * (1 - sl_buffer)
                # Enforce minimum 0.5% stop
                if entry - hard_sl < sl_min_pct * entry:
                    hard_sl = entry * (1 - sl_min_pct)
                stop_dist = entry - hard_sl
                if stop_dist <= 0:
                    continue
                t1 = entry + t1_r_mult * stop_dist
                t2 = entry + t2_r_mult * stop_dist
            else:
                hard_sl = mh * (1 + sl_buffer)
                if hard_sl - entry < sl_min_pct * entry:
                    hard_sl = entry * (1 + sl_min_pct)
                stop_dist = hard_sl - entry
                if stop_dist <= 0:
                    continue
                t1 = entry - t1_r_mult * stop_dist
                t2 = entry - t2_r_mult * stop_dist

            ev = dict(
                trigger_idx=i + 2,
                post_start=i + 3,
                session_end=e,
                direction=direction,
                time_bucket=tb,
                entry=entry,
                entry_hhmm=b_hhmm,
                entry_ts=ts_arr[i + 2],
                mh=mh, ml=ml, ih=ih, il=il,
                hard_sl=hard_sl,
                stop_dist=stop_dist,
                t1=t1, t2=t2,
                sym=str(sym_arr[i]),
                d=day_arr[i],
                cap=cap_arr[i],
            )
            break

        if ev is None:
            continue

        # Simulate exits
        direction = ev["direction"]
        entry = ev["entry"]
        hard_sl = ev["hard_sl"]
        stop_dist = ev["stop_dist"]
        t1_target = ev["t1"]; t2_target = ev["t2"]
        i_start = ev["post_start"]; i_end = ev["session_end"]

        hit_t1 = False
        t1_exit = None
        exit_price = None
        exit_ts = None
        exit_reason = None

        for j in range(i_start, i_end):
            hj = float(high_arr[j]); lj = float(low_arr[j])
            cj = float(close_arr[j]); hhmm_j = hhmm_arr[j]
            active_sl = entry if (hit_t1 and use_be_trail) else hard_sl
            if direction == "LONG":
                if lj <= active_sl:
                    exit_price = active_sl
                    exit_ts = ts_arr[j]
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if (not hit_t1) and (hj >= t1_target):
                    hit_t1 = True
                    t1_exit = t1_target
                if hit_t1 and (hj >= t2_target):
                    exit_price = t2_target
                    exit_ts = ts_arr[j]
                    exit_reason = "t2"
                    break
            else:
                if hj >= active_sl:
                    exit_price = active_sl
                    exit_ts = ts_arr[j]
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if (not hit_t1) and (lj <= t1_target):
                    hit_t1 = True
                    t1_exit = t1_target
                if hit_t1 and (lj <= t2_target):
                    exit_price = t2_target
                    exit_ts = ts_arr[j]
                    exit_reason = "t2"
                    break
            if hhmm_j >= time_stop_hhmm:
                exit_price = cj
                exit_ts = ts_arr[j]
                exit_reason = "time_stop"
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
            qty_t1 = int(qty * t1_qty_pct)
            qty_t2 = qty - qty_t1
            if qty_t1 == 0:  # t1_qty_pct == 0 — runner-only
                if direction == "LONG":
                    realized = (exit_price - entry) * qty_t2
                else:
                    realized = (entry - exit_price) * qty_t2
                fee = calc_fee(entry, exit_price, qty_t2, side)
                blended = exit_price
            else:
                if direction == "LONG":
                    pnl_t1 = (t1_exit - entry) * qty_t1
                    pnl_t2 = (exit_price - entry) * qty_t2
                else:
                    pnl_t1 = (entry - t1_exit) * qty_t1
                    pnl_t2 = (entry - exit_price) * qty_t2
                realized = pnl_t1 + pnl_t2
                fee = (calc_fee(entry, t1_exit, qty_t1, side)
                       + calc_fee(entry, exit_price, qty_t2, side))
                blended = (t1_exit * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            if direction == "LONG":
                realized = (exit_price - entry) * qty
            else:
                realized = (entry - exit_price) * qty
            fee = calc_fee(entry, exit_price, qty, side)
            blended = exit_price
        net_pnl = realized - fee

        trades.append({
            "T0_signal_date": ev["d"],
            "symbol": "NSE:" + ev["sym"],
            "bare_symbol": ev["sym"],
            "cap_segment": ev["cap"],
            "direction": direction,
            "time_bucket": ev["time_bucket"],
            "entry_ts": ev["entry_ts"],
            "entry_hhmm": ev["entry_hhmm"],
            "entry_price": entry,
            "mother_high": ev["mh"],
            "mother_low": ev["ml"],
            "inside_high": ev["ih"],
            "inside_low": ev["il"],
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

    return trades


# =========================================================================
# Period runner — month-by-month, OOM-safe
# =========================================================================

def run_period_month_by_month(
    label: str,
    start: date,
    end: date,
    allowed: Set[str],
    cap_map: Dict[str, str],
) -> pd.DataFrame:
    """Walk each month, generate trades, accumulate. Loads ONE month into
    memory at a time. Inside-bar pattern is intraday-only, no cross-day
    priors needed — so per-month chunking is exact, no boundary effects."""
    print(f"\n[{label}] running month-by-month {start} .. {end} ...")
    all_trades: List[dict] = []
    for y, m in _months_between(start, end):
        bars = _load_one_month(y, m, allowed, cap_map, start, end)
        if bars.empty:
            print(f"  {y}-{m:02d}: SKIPPED (no bars)")
            continue
        n_bars = len(bars)
        trades_this_month = _scan_and_simulate(bars)
        all_trades.extend(trades_this_month)
        print(f"  {y}-{m:02d}: bars={n_bars:>10,}  trades={len(trades_this_month):>5,}  cum={len(all_trades):>6,}")
        del bars
        gc.collect()
    print(f"\n[{label}] DONE — total trades: {len(all_trades):,}")
    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame(all_trades)


# =========================================================================
# Cell prep + aggregate metrics
# =========================================================================

def _bucket_pct(values: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(values, bins=edges, labels=labels, right=False, include_lowest=True)


def prep_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Add ex-ante (pre-trade) cell dimensions. No outcome leakage."""
    df = df.copy()
    d = pd.to_datetime(df["T0_signal_date"])
    df["_session_date"] = d.dt.date
    df["_month"] = d.dt.strftime("%Y-%m")
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")
    df["quarter"] = "Q" + d.dt.quarter.astype(str)

    mother_size_pct = (df["mother_high"] - df["mother_low"]) / df["mother_low"] * 100.0
    df["mother_size_bucket"] = _bucket_pct(
        mother_size_pct,
        edges=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    )

    rng = (df["mother_high"] - df["mother_low"]).replace(0, np.nan)
    inside_pos = (df["inside_low"] - df["mother_low"]) / rng
    df["inside_pos_bucket"] = _bucket_pct(
        inside_pos,
        edges=[-np.inf, 0.33, 0.67, np.inf],
        labels=["lower-third", "mid", "upper-third"],
    )

    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_size_bucket"] = _bucket_pct(
        r_pct,
        edges=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    )

    hhmm = df["entry_hhmm"].astype(str)
    df["entry_hour_bucket"] = hhmm.str.slice(0, 2) + ":00"

    return df


def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _aggregate_metrics(df: pd.DataFrame) -> dict:
    """Aggregate: n, PF, WR, daily Sharpe, NET, win_mo%, top_mo%, n_mo."""
    if df.empty:
        return dict(n=0, pf=float("nan"), wr=float("nan"),
                    sharpe=float("nan"), net=0.0,
                    win_mo=float("nan"), top_mo=float("nan"), n_mo=0)
    pnl = df["net_pnl"]
    n = int(len(pnl))
    pf = _pf(pnl)
    wr = 100.0 * float((pnl > 0).mean())
    daily = df.groupby("T0_signal_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    net = float(pnl.sum())
    mo = pd.to_datetime(df["T0_signal_date"]).dt.strftime("%Y-%m")
    monthly = df.groupby(mo)["net_pnl"].sum()
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


# =========================================================================
# Reporting + cell mining
# =========================================================================

CELL_DIMS = [
    "direction",
    "cap_segment",
    "time_bucket",
    "mother_size_bucket",
    "inside_pos_bucket",
    "R_size_bucket",
    "dow",
    "month",
    "quarter",
    "entry_hour_bucket",
]


def report_period(trades: pd.DataFrame, label: str,
                  cellmine: bool = True) -> Tuple[dict, pd.DataFrame]:
    """Aggregate report + (optional) deep cell mine. Returns (agg, ship_cells_df)."""
    print("\n" + "=" * 78)
    print(f"REPORT — inside_bar_breakout_v2 — {label}")
    print("=" * 78)
    if trades.empty:
        print("  NO TRADES")
        return {}, pd.DataFrame()

    n_total = len(trades)
    cap_dist = trades["cap_segment"].value_counts().to_dict()
    dir_dist = trades["direction"].value_counts().to_dict()
    tb_dist = trades["time_bucket"].value_counts().to_dict()
    print(f"\n  Funnel:  events={n_total:,}  by_cap={cap_dist}  by_dir={dir_dist}  by_tb={tb_dist}")

    agg = _aggregate_metrics(trades)
    v, why = _ship_verdict(agg)
    pf_str = f"{agg['pf']:.3f}" if (not np.isnan(agg["pf"]) and agg["pf"] != float("inf")) else "inf"
    print(f"\n  AGGREGATE:")
    print(f"    n={agg['n']:,}  PF={pf_str}  WR={agg['wr']:.1f}%  "
          f"Sharpe={agg['sharpe']:.3f}  NET=Rs.{agg['net']:,.0f}")
    print(f"    months={agg['n_mo']}  win_mo={agg['win_mo']:.1f}%  "
          f"top_mo={agg['top_mo']:.1f}%")
    print(f"    aggregate verdict: {v}  reasons: {';'.join(why) if why else 'all gates pass'}")

    # Monthly NET breakdown
    print(f"\n  Per-month NET breakdown:")
    _tmp = trades.copy()
    _tmp["_mo"] = pd.to_datetime(_tmp["T0_signal_date"]).dt.strftime("%Y-%m")
    monthly = _tmp.groupby("_mo")["net_pnl"].sum()
    abs_net = float(abs(monthly.sum())) or 1.0
    for mo, net in monthly.items():
        share = 100.0 * abs(net) / abs_net
        flag = "  W" if net > 0 else "  L"
        print(f"    {mo}  NET=Rs.{net:>11,.0f}  share={share:5.1f}%  {flag}")

    if not cellmine:
        return agg, pd.DataFrame()

    # Deep cell mine
    print(f"\n  [cell-mine] preparing dimensions + scanning ...")
    cells_df = prep_inside_bar(trades)
    dims_have = [d for d in CELL_DIMS if d in cells_df.columns]
    print(f"  dims scanned: {dims_have}")
    cells = scan_cells(
        cells_df, dims_have, "net_pnl", max_combo=3,
        date_col="_session_date", month_col="_month",
    )
    print(f"  total cells scanned: {len(cells):,}")

    # Apply gauntlet-v2 ship gates
    ship_mask = (
        (cells["n"] >= N_SHIP)
        & (cells["pf"] >= PF_SHIP)
        & (cells["sharpe"] >= SHARPE_SHIP)
        & (cells["win_mo_pct"] >= WIN_MO_PCT_SHIP)
        & (cells["top_mo_pct"] < TOP_MO_PCT_SHIP)
    )
    ship_eligible = cells[ship_mask].sort_values(["pf", "n"], ascending=[False, False]).copy()

    # Survivor gate
    surv_mask = (cells["n"] >= N_SURV) & (cells["pf"] >= PF_SURV) & (cells["sharpe"] > 0)
    survivors = cells[surv_mask].sort_values(["pf", "n"], ascending=[False, False])

    print(f"\n  SURVIVORS (n>={N_SURV}, PF>={PF_SURV}, Sh>0): {len(survivors):,}")
    for _, r in survivors.head(15).iterrows():
        print(f"    [{r['dims']}] {r['cell']}  n={r['n']:,} PF={r['pf']:.3f} "
              f"WR={r['wr']:.1f}% Sh={r['sharpe']:.2f} "
              f"win_mo={r['win_mo_pct']}% top_mo={r['top_mo_pct']}% NET={r['net']:,.0f}")

    print(f"\n  SHIP-ELIGIBLE (n>={N_SHIP}, PF>={PF_SHIP}, Sh>={SHARPE_SHIP}, "
          f"win_mo>={WIN_MO_PCT_SHIP}%, top_mo<{TOP_MO_PCT_SHIP}%): {len(ship_eligible):,}")
    for _, r in ship_eligible.head(20).iterrows():
        print(f"    [{r['dims']}] {r['cell']}  n={r['n']:,} PF={r['pf']:.3f} "
              f"Sh={r['sharpe']:.2f} win_mo={r['win_mo_pct']}% "
              f"top_mo={r['top_mo_pct']}% NET={r['net']:,.0f}")

    # Persist cell-mine results
    out_cells = _OUT_DIR / f"inside_bar_v2_cells_{label.lower()}.csv"
    cells.to_csv(out_cells, index=False)
    print(f"\n  wrote cell-mine CSV: {out_cells.relative_to(_REPO)}")

    if len(ship_eligible):
        out_ship = _OUT_DIR / f"inside_bar_v2_ship_cells_{label.lower()}.csv"
        ship_eligible.to_csv(out_ship, index=False)
        print(f"  wrote SHIP cells: {out_ship.relative_to(_REPO)} ({len(ship_eligible)} cells)")

    return agg, ship_eligible


# =========================================================================
# Cell-signature matching across periods
# =========================================================================

def _parse_cell_signature(cell_str: str) -> Dict[str, str]:
    """Parse 'dim1=val1 | dim2=val2 | ...' into a dict."""
    out: Dict[str, str] = {}
    for part in cell_str.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def filter_trades_to_cell(trades: pd.DataFrame, cell_dict: Dict[str, str]) -> pd.DataFrame:
    """Subset trades to those matching the cell signature."""
    prepped = prep_inside_bar(trades)
    mask = pd.Series(True, index=prepped.index)
    for k, v in cell_dict.items():
        if k not in prepped.columns:
            return pd.DataFrame()  # cell dim missing in this period
        col = prepped[k].astype(str)
        mask &= (col == v)
    return prepped[mask]


def validate_cells_on_period(ship_cells: pd.DataFrame, period_trades: pd.DataFrame,
                             period_label: str) -> pd.DataFrame:
    """Replay each ship-eligible cell on a new period; report whether all
    gauntlet-v2 gates hold."""
    print(f"\n  {'-' * 76}")
    print(f"  VALIDATING {len(ship_cells)} ship-eligible cells on {period_label}")
    print(f"  {'-' * 76}")
    rows: List[dict] = []
    for _, cell_row in ship_cells.iterrows():
        cell_dict = _parse_cell_signature(cell_row["cell"])
        sub = filter_trades_to_cell(period_trades, cell_dict)
        if sub.empty:
            print(f"    [{cell_row['dims']}] {cell_row['cell']} -> 0 trades in {period_label}  FAIL")
            rows.append({
                "cell": cell_row["cell"], "dims": cell_row["dims"],
                "period": period_label, "n": 0, "pf": float("nan"),
                "sharpe": float("nan"), "win_mo": float("nan"),
                "top_mo": float("nan"), "verdict": "RETIRE", "reasons": "no_trades",
            })
            continue
        m = _aggregate_metrics(sub)
        v, why = _ship_verdict(m)
        pf_str = f"{m['pf']:.2f}" if (not np.isnan(m["pf"]) and m["pf"] != float("inf")) else "inf"
        print(f"    [{cell_row['dims']}] {cell_row['cell']}")
        print(f"      {period_label}: n={m['n']:,} PF={pf_str} Sh={m['sharpe']:.2f} "
              f"win_mo={m['win_mo']:.0f}% top_mo={m['top_mo']:.0f}% NET={m['net']:,.0f}  {v}")
        if v == "RETIRE":
            print(f"      reasons: {';'.join(why)}")
        rows.append({
            "cell": cell_row["cell"], "dims": cell_row["dims"],
            "period": period_label, "n": m["n"], "pf": m["pf"],
            "sharpe": m["sharpe"], "win_mo": m["win_mo"], "top_mo": m["top_mo"],
            "net": m["net"], "verdict": v,
            "reasons": ";".join(why) if why else "all gates pass",
        })
    return pd.DataFrame(rows)


# =========================================================================
# SL / T1 / T2 / partial sweep — only on cell-event subsets
# =========================================================================

SWEEP_SL_BUFFERS = [0.003, 0.005, 0.007, 0.010]
SWEEP_T1_R = [0.5, 1.0, 1.5]
SWEEP_T2_R = [1.0, 1.5, 2.0, 3.0]
SWEEP_T1_QTY = [0.0, 0.33, 0.50, 0.67]
SWEEP_TIME_STOP = ["12:00", "13:30", "15:10"]


def _rebuild_bars_for_cell(allowed_symbols: Set[str], cell_dates: Set[date],
                            cap_map: Dict[str, str]) -> pd.DataFrame:
    """Reload only the months that contain cell trade dates. OOM-safe."""
    if not cell_dates:
        return pd.DataFrame()
    yms = sorted({(d.year, d.month) for d in cell_dates})
    print(f"  [sweep-load] reloading bars for {len(yms)} months covering {len(cell_dates)} sessions")
    parts: List[pd.DataFrame] = []
    for y, m in yms:
        df = _load_one_month(y, m, allowed_symbols, cap_map,
                             date(y, m, 1), date(y, m, 28) + pd.Timedelta(days=4).to_pytimedelta())
        if df.empty:
            continue
        # keep only cell symbols + cell dates
        df = df[df["symbol"].isin(allowed_symbols)]
        df = df[df["_day"].isin(pd.to_datetime(list(cell_dates)))]
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    bars = pd.concat(parts, ignore_index=True)
    bars.sort_values(["symbol", "date"], kind="mergesort", inplace=True)
    return bars


def sweep_cell(cell_str: str, disc_trades: pd.DataFrame,
               cap_map: Dict[str, str]) -> pd.DataFrame:
    """Run SL/T/partial/time sweep for one cell. Returns combos sorted by PF."""
    print(f"\n  [sweep] cell: {cell_str}")
    cell_dict = _parse_cell_signature(cell_str)
    sub = filter_trades_to_cell(disc_trades, cell_dict)
    if sub.empty:
        print("    no trades for cell — skipping sweep")
        return pd.DataFrame()
    symbols = set(sub["bare_symbol"].unique().tolist())
    dates = set(pd.to_datetime(sub["T0_signal_date"]).dt.date.unique().tolist())
    print(f"    cell trades: n={len(sub):,}  unique_syms={len(symbols)}  unique_sessions={len(dates)}")
    bars = _rebuild_bars_for_cell(symbols, dates, cap_map)
    if bars.empty:
        print("    bar reload empty — skipping sweep")
        return pd.DataFrame()

    rows: List[dict] = []
    n_combos = (len(SWEEP_SL_BUFFERS) * len(SWEEP_T1_R) *
                len(SWEEP_T2_R) * len(SWEEP_T1_QTY) * len(SWEEP_TIME_STOP))
    print(f"    sweeping {n_combos} parameter combinations ...")
    combo_idx = 0
    for sl_b in SWEEP_SL_BUFFERS:
        for t1r in SWEEP_T1_R:
            for t2r in SWEEP_T2_R:
                if t2r < t1r:
                    continue
                for t1q in SWEEP_T1_QTY:
                    for ts in SWEEP_TIME_STOP:
                        combo_idx += 1
                        trades_list = _scan_and_simulate(
                            bars,
                            sl_buffer=sl_b,
                            t1_r_mult=t1r,
                            t2_r_mult=t2r,
                            t1_qty_pct=t1q,
                            time_stop_hhmm=ts,
                        )
                        if not trades_list:
                            continue
                        tdf = pd.DataFrame(trades_list)
                        # Re-filter to cell membership at this SL combo
                        tdf_cell = filter_trades_to_cell(tdf, cell_dict)
                        if tdf_cell.empty:
                            continue
                        m = _aggregate_metrics(tdf_cell)
                        rows.append({
                            "sl_buffer": sl_b, "t1_r": t1r, "t2_r": t2r,
                            "t1_qty": t1q, "time_stop": ts,
                            "n": m["n"], "pf": m["pf"], "wr": m["wr"],
                            "sharpe": m["sharpe"], "net": m["net"],
                            "win_mo": m["win_mo"], "top_mo": m["top_mo"],
                        })
                        if combo_idx % 50 == 0:
                            print(f"      progress: {combo_idx}/{n_combos} combos")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["pf", "n"], ascending=[False, False]).reset_index(drop=True)
    return df


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-oos", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip SL/T sweep on ship-eligible cells")
    args = parser.parse_args()

    print(">>> Inside Bar Breakout v2 — PROPER 3-PERIOD GAUNTLET <<<")
    print(f"   Discovery: {DISCOVERY_START} .. {DISCOVERY_END}  ({len(_months_between(DISCOVERY_START, DISCOVERY_END))}mo)")
    print(f"   OOS:       {OOS_START} .. {OOS_END}  ({len(_months_between(OOS_START, OOS_END))}mo)")
    print(f"   Holdout:   {HOLDOUT_START} .. {HOLDOUT_END}  ({len(_months_between(HOLDOUT_START, HOLDOUT_END))}mo)")

    # ---- Universe + preflight ----
    allowed, cap_map = build_universe()
    if not allowed:
        print("[ABORT] empty universe")
        return 2

    regime_preflight(DISCOVERY_START, DISCOVERY_END, "Discovery")
    regime_preflight(OOS_START, OOS_END, "OOS")
    regime_preflight(HOLDOUT_START, HOLDOUT_END, "Holdout")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- DISCOVERY ----
    print("\n" + "#" * 78)
    print("# DISCOVERY  (2023-01-01 .. 2024-12-31)")
    print("#" * 78)
    disc_trades_path = _OUT_DIR / "inside_bar_v2_trades_discovery.csv"

    if args.skip_discovery and disc_trades_path.exists():
        print(f"  [skip] loading existing trades from {disc_trades_path}")
        disc_trades = pd.read_csv(disc_trades_path)
    else:
        disc_trades = run_period_month_by_month(
            "Discovery", DISCOVERY_START, DISCOVERY_END, allowed, cap_map)
        if not disc_trades.empty:
            disc_trades.to_csv(disc_trades_path, index=False)
            print(f"  wrote {disc_trades_path.relative_to(_REPO)} ({len(disc_trades):,} trades)")

    disc_agg, disc_ship = report_period(disc_trades, label="Discovery", cellmine=True)
    n_ship_disc = len(disc_ship)

    if n_ship_disc == 0:
        print("\n[VERDICT] Discovery has 0 ship-eligible cells. CONFIRM RETIRE.")
        # Still record the empty result
        return 0

    # ---- OOS ----
    print("\n" + "#" * 78)
    print("# OOS  (2025-01-01 .. 2025-09-30)")
    print("#" * 78)
    oos_trades_path = _OUT_DIR / "inside_bar_v2_trades_oos.csv"
    if args.skip_oos and oos_trades_path.exists():
        oos_trades = pd.read_csv(oos_trades_path)
    else:
        oos_trades = run_period_month_by_month(
            "OOS", OOS_START, OOS_END, allowed, cap_map)
        if not oos_trades.empty:
            oos_trades.to_csv(oos_trades_path, index=False)
            print(f"  wrote {oos_trades_path.relative_to(_REPO)} ({len(oos_trades):,} trades)")

    oos_agg, _ = report_period(oos_trades, label="OOS", cellmine=False)

    # Validate Discovery's ship-eligible cells on OOS
    oos_validation = validate_cells_on_period(disc_ship, oos_trades, "OOS")
    if not oos_validation.empty:
        oos_val_path = _OUT_DIR / "inside_bar_v2_oos_validation.csv"
        oos_validation.to_csv(oos_val_path, index=False)
        print(f"\n  wrote OOS validation: {oos_val_path.relative_to(_REPO)}")

    oos_ship = oos_validation[oos_validation["verdict"] == "SHIP"]
    print(f"\n  CELLS THAT PASSED DISCOVERY -> OOS: {len(oos_ship)}/{n_ship_disc}")
    if len(oos_ship) == 0:
        print("\n[VERDICT] No cells survived OOS. CONFIRM RETIRE.")
        return 0

    # ---- HOLDOUT ----
    print("\n" + "#" * 78)
    print("# HOLDOUT  (2025-10-01 .. 2026-04-30)")
    print("#" * 78)
    hold_trades_path = _OUT_DIR / "inside_bar_v2_trades_holdout.csv"
    if args.skip_holdout and hold_trades_path.exists():
        hold_trades = pd.read_csv(hold_trades_path)
    else:
        hold_trades = run_period_month_by_month(
            "Holdout", HOLDOUT_START, HOLDOUT_END, allowed, cap_map)
        if not hold_trades.empty:
            hold_trades.to_csv(hold_trades_path, index=False)
            print(f"  wrote {hold_trades_path.relative_to(_REPO)} ({len(hold_trades):,} trades)")

    hold_agg, _ = report_period(hold_trades, label="Holdout", cellmine=False)

    # Validate cells that passed OOS on Holdout
    survivors_oos = oos_validation[oos_validation["verdict"] == "SHIP"]
    surv_cells_df = disc_ship[disc_ship["cell"].isin(survivors_oos["cell"])]
    hold_validation = validate_cells_on_period(surv_cells_df, hold_trades, "Holdout")
    if not hold_validation.empty:
        hold_val_path = _OUT_DIR / "inside_bar_v2_holdout_validation.csv"
        hold_validation.to_csv(hold_val_path, index=False)
        print(f"\n  wrote Holdout validation: {hold_val_path.relative_to(_REPO)}")

    hold_ship = hold_validation[hold_validation["verdict"] == "SHIP"]
    print(f"\n  CELLS THAT PASSED DISC -> OOS -> HOLDOUT: {len(hold_ship)}/{len(oos_ship)}")

    if len(hold_ship) == 0:
        print("\n[VERDICT] No cells survived Holdout. CONFIRM RETIRE.")
        return 0

    # ---- SL/T sweep on triple-survivors ----
    print("\n" + "#" * 78)
    print(f"# SL/T SWEEP — {len(hold_ship)} triple-survivor cell(s)")
    print("#" * 78)

    if args.no_sweep:
        print("  [skip] --no-sweep flag set")
    else:
        for _, hr in hold_ship.iterrows():
            cell_str = hr["cell"]
            sweep_df = sweep_cell(cell_str, disc_trades, cap_map)
            if not sweep_df.empty:
                safe = cell_str.replace(" ", "").replace("|", "_").replace("=", "-")
                sweep_path = _OUT_DIR / f"inside_bar_v2_sweep_{safe[:80]}.csv"
                sweep_df.to_csv(sweep_path, index=False)
                print(f"\n  Top 10 sweep combos for {cell_str}:")
                print(sweep_df.head(10).to_string(index=False))
                print(f"  wrote {sweep_path.relative_to(_REPO)}")

    # ---- FINAL ----
    print("\n" + "#" * 78)
    print("# FINAL EVIDENCE DUMP — triple-survivor cells")
    print("#" * 78)
    for _, hr in hold_ship.iterrows():
        cell_str = hr["cell"]
        print(f"\n  CELL: {cell_str}")
        # Re-pull Discovery / OOS / Holdout numbers
        for period, tdf in [("Discovery", disc_trades),
                            ("OOS", oos_trades),
                            ("Holdout", hold_trades)]:
            cd = _parse_cell_signature(cell_str)
            sub = filter_trades_to_cell(tdf, cd)
            if sub.empty:
                print(f"    {period}: 0 trades")
                continue
            m = _aggregate_metrics(sub)
            pf_str = f"{m['pf']:.2f}" if (not np.isnan(m['pf']) and m['pf'] != float('inf')) else "inf"
            print(f"    {period:<9}: n={m['n']:>5,}  PF={pf_str:>5}  WR={m['wr']:>5.1f}%  "
                  f"Sh={m['sharpe']:>5.2f}  win_mo={m['win_mo']:>5.1f}%  "
                  f"top_mo={m['top_mo']:>5.1f}%  NET=Rs.{m['net']:>11,.0f}")

    print("\n[VERDICT] SHIP CANDIDATE — triple-period survival confirmed.")
    print("          Next step: draft §3.3 brief for live validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
