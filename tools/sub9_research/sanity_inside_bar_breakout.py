"""Pure-math sanity check on the Inside Bar breakout pattern.

NOT a §3.3 brief-gated test — user explicitly wants a mathematical
sanity that lets cell-mining find which (direction × time-of-day × cap)
cells have edge.

PATTERN (intraday 5m bars):
  - Mother bar at time T (5m bar with high H_m, low L_m)
  - Inside bar at T+5: high < H_m AND low > L_m (fully contained)
  - Breakout bar at T+10: closes above H_m (LONG) or below L_m (SHORT)
  - Entry: breakout bar's close
  - Hard SL: opposite extreme of mother + 5 bps buffer
       LONG SL = L_m * 0.9995
       SHORT SL = H_m * 1.0005
  - T1 (50% qty) = entry ± 1.0R
  - T2 (50% qty) = entry ± 2.0R
  - BE trail: active_sl = entry if t1_hit else hard_sl
  - Time stop: 15:10 IST
  - Latch: one fire per (symbol, session_date) — first qualifying trigger

UNIVERSE:
  - fno_liquid_200.csv + ind_nifty*list.csv + 5m feather symbol pool
  - Cross-ref with nse_all.json: mis_leverage >= 1.0
  - cap_segment in {large_cap, mid_cap, small_cap} (exclude micro/unknown)
  - Expected: ~600-1500 symbols

PERIODS:
  - Discovery: 2024-09-01 → 2025-09-30 (post-SEBI Aug, pre-Oct-1 MWPL)
  - OOS:       2025-10-01 → 2026-04-30 (post-Oct-1 MWPL + war)

GAUNTLET-V2 SHIP GATES (per cell):
  - n >= 125
  - NET PF >= 1.30
  - Daily Sharpe >= 0.5
  - Per-month winning >= 55%
  - Top-month NET share < 40%

SURVIVOR (investigate further, not ship):
  - n >= 100 AND PF >= 1.20

Usage:
    python tools/sub9_research/sanity_inside_bar_breakout.py [--oos]
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

# Force UTF-8 stdout on Windows to avoid cp1252 encode errors on rule-change descriptions
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
SL_BUFFER = 0.0005           # 5 bps beyond mother's opposite extreme
T1_R_MULT = 1.0
T2_R_MULT = 2.0
T1_QTY_PCT = 0.5             # 50% out at T1, 50% runs to T2
USE_BE_TRAIL = True
TIME_STOP_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000

# Cell dimensions (locked BEFORE seeing data)
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

# Gauntlet-v2 ship gates
N_SHIP = 125
PF_SHIP = 1.30
SHARPE_SHIP = 0.5
WIN_MO_PCT_SHIP = 55.0
TOP_MO_PCT_SHIP = 40.0

# Survivor (investigate)
N_SURV = 100
PF_SURV = 1.20

_KEEP_5M_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


# =========================================================================
# Universe construction
# =========================================================================

def _load_nse_all() -> Dict[str, dict]:
    """symbol (bare, no .NS) -> {mis_leverage, cap_segment}"""
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
    """Aggregate symbols from fno_liquid_200 + ind_nifty*list."""
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
    """Pull broader symbol pool from a few feather files to expand universe."""
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
    """Returns (allowed_bare_symbols, sym -> cap_segment)."""
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
# Pre-flight regime break check
# =========================================================================

def regime_preflight(start: date, end: date, label: str):
    print(f"\n[regime] preflight check on {label} window {start} .. {end}")
    hits = check_window(
        "inside_bar_breakout",
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
            # ASCII-only print on Windows cp1252 stdout
            desc = r.description.encode("ascii", "replace").decode("ascii")
            print(f"    - {r.effective_date} [{r.severity.upper()}] {desc}")
    else:
        print("  clean (no high/critical rule changes)")


# =========================================================================
# Data loading
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


def load_period_5m(start: date, end: date, allowed: Set[str],
                   cap_map: Dict[str, str]) -> pd.DataFrame:
    """Concat 5m bars across period months."""
    print(f"\n[load] 5m bars {start} .. {end} ...")
    parts: List[pd.DataFrame] = []
    for y, m in _months_between(start, end):
        df = _load_5m_month(y, m, allowed, cap_map)
        if not df.empty:
            parts.append(df)
            print(f"    {y}-{m:02d}: {len(df):,} bars")
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    # Vectorized date / hhmm
    ts_ns = big["date"].astype("datetime64[ns]")
    ns = ts_ns.values.view("int64")
    NS_PER_DAY = 86_400 * 1_000_000_000
    NS_PER_MIN = 60 * 1_000_000_000
    day_floor = (ns // NS_PER_DAY) * NS_PER_DAY
    big["d"] = pd.DatetimeIndex(pd.to_datetime(day_floor)).date
    minutes = ((ns - day_floor) // NS_PER_MIN).astype("int32")
    hours = (minutes // 60).astype("int8")
    mins = (minutes % 60).astype("int8")
    h_str = np.char.zfill(hours.astype("U2"), 2)
    m_str = np.char.zfill(mins.astype("U2"), 2)
    big["hhmm"] = np.char.add(np.char.add(h_str, ":"), m_str)
    big = big[(big["d"] >= start) & (big["d"] <= end)].reset_index(drop=True)
    big = big.sort_values(["symbol", "d", "date"], kind="mergesort").reset_index(drop=True)
    print(f"  total bars after clip: {len(big):,}  | symbols: {big['symbol'].nunique():,}")
    return big


# =========================================================================
# Event detection: find inside-bar breakouts
# =========================================================================

def _time_bucket(hhmm: str) -> Optional[str]:
    for label, lo, hi in TIME_BUCKETS:
        if lo <= hhmm < hi:
            return label
    return None


def find_events(big5m: pd.DataFrame) -> pd.DataFrame:
    """Locate (mother_idx, inside_idx, breakout_idx) tuples per (symbol, d)
    where:
        - mother bar at T (any 5m bar in 09:15..15:00)
        - inside bar at T+5: high < mother.high AND low > mother.low
        - breakout bar at T+10: close > mother.high (LONG) or close < mother.low (SHORT)
        - bars are within the same session_date AND consecutive 5m timestamps
        - latched: first qualifying (symbol, d) -> one event per session

    Returns DataFrame:
        symbol, d, cap_segment, direction, time_bucket,
        entry_ts, entry_price, mother_high, mother_low, hard_sl, stop_distance,
        t1_target, t2_target, post_bars_start_idx  -- (used for sim)
    """
    print("\n[events] scanning for inside-bar breakouts ...")
    # Group by (symbol, d) to walk bar-by-bar
    grp_key = pd.factorize(list(zip(big5m["symbol"], big5m["d"])))[0]
    boundaries = np.where(np.diff(grp_key) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(big5m)]])

    sym_arr = big5m["symbol"].values
    d_arr = big5m["d"].values
    high_arr = big5m["high"].values.astype("float32")
    low_arr = big5m["low"].values.astype("float32")
    close_arr = big5m["close"].values.astype("float32")
    hhmm_arr = big5m["hhmm"].values
    ts_arr = big5m["date"].values
    cap_arr = big5m["cap_segment"].astype(str).values

    events: List[dict] = []
    n_no_inside = n_no_breakout = n_latched = 0

    for grp_idx in range(len(starts)):
        s = int(starts[grp_idx]); e = int(ends[grp_idx])
        n_bars = e - s
        if n_bars < 3:
            continue
        # Bars within session, find first qualifying chain
        latched = False
        for i in range(s, e - 2):
            # bar i = mother; bar i+1 = inside; bar i+2 = breakout
            mh = float(high_arr[i]); ml = float(low_arr[i])
            ih = float(high_arr[i + 1]); il = float(low_arr[i + 1])
            # Inside bar test (strict): inside.high < mother.high AND inside.low > mother.low
            if not (ih < mh and il > ml):
                continue
            # Breakout bar
            b_close = float(close_arr[i + 2])
            b_hhmm = hhmm_arr[i + 2]
            tb = _time_bucket(b_hhmm)
            if tb is None:
                continue
            direction = None
            if b_close > mh:
                direction = "LONG"
            elif b_close < ml:
                direction = "SHORT"
            else:
                continue

            # Entry = breakout bar's close
            entry = b_close
            if direction == "LONG":
                hard_sl = ml * (1 - SL_BUFFER)
                stop_dist = entry - hard_sl
                if stop_dist <= 0:
                    continue
                t1 = entry + T1_R_MULT * stop_dist
                t2 = entry + T2_R_MULT * stop_dist
            else:
                hard_sl = mh * (1 + SL_BUFFER)
                stop_dist = hard_sl - entry
                if stop_dist <= 0:
                    continue
                t1 = entry - T1_R_MULT * stop_dist
                t2 = entry - T2_R_MULT * stop_dist

            events.append({
                "symbol": str(sym_arr[i]),
                "d": d_arr[i],
                "cap_segment": cap_arr[i],
                "direction": direction,
                "time_bucket": tb,
                "entry_ts": ts_arr[i + 2],
                "entry_hhmm": b_hhmm,
                "entry_price": entry,
                "mother_high": mh,
                "mother_low": ml,
                "inside_high": ih,
                "inside_low": il,
                "hard_sl": hard_sl,
                "stop_distance": stop_dist,
                "t1_target": t1,
                "t2_target": t2,
                "post_start_idx": i + 3,   # forward-walk starts at bar after breakout
                "session_end_idx": e,
            })
            n_latched += 1
            latched = True
            break
        if not latched:
            # Diagnostic: try to count near-misses
            pass

    print(f"  events found (latched first-per-session): {n_latched:,}")
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events)


# =========================================================================
# Simulator
# =========================================================================

def simulate(events: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    """Walk forward each event's post-bars to determine SL/T1/T2/time exits."""
    if events.empty:
        return events
    print(f"\n[sim] simulating {len(events):,} events ...")
    high_arr = big5m["high"].values.astype("float32")
    low_arr = big5m["low"].values.astype("float32")
    close_arr = big5m["close"].values.astype("float32")
    hhmm_arr = big5m["hhmm"].values
    ts_arr = big5m["date"].values

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
            # exhausted session bars (intraday-only)
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
            "T0_signal_date": ev["d"],
            "symbol": "NSE:" + ev["symbol"],
            "bare_symbol": ev["symbol"],
            "cap_segment": ev["cap_segment"],
            "direction": direction,
            "time_bucket": ev["time_bucket"],
            "entry_ts": ev["entry_ts"],
            "entry_hhmm": ev["entry_hhmm"],
            "entry_price": entry_price,
            "mother_high": ev["mother_high"],
            "mother_low": ev["mother_low"],
            "inside_high": ev["inside_high"],
            "inside_low": ev["inside_low"],
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


def report(trades: pd.DataFrame, label: str):
    print("\n" + "=" * 78)
    print(f"REPORT — inside_bar_breakout — {label}")
    print("=" * 78)
    if trades.empty:
        print("  NO TRADES")
        return

    n_events = len(trades)
    cap_dist = trades["cap_segment"].value_counts().to_dict()
    dir_dist = trades["direction"].value_counts().to_dict()
    tb_dist = trades["time_bucket"].value_counts().to_dict()
    print(f"\n  Funnel:")
    print(f"    traded events: {n_events:,}")
    print(f"    by cap: {cap_dist}")
    print(f"    by dir: {dir_dist}")
    print(f"    by time bucket: {tb_dist}")

    # Aggregate
    agg = _agg(trades)
    v, why = _ship_verdict(agg)
    print(f"\n  AGGREGATE:")
    print(f"    n={agg['n']:,}  PF={agg['pf']:.3f}  WR={agg['wr']:.1f}%  "
          f"Sharpe={agg['sharpe']:.3f}  NET=Rs.{agg['net']:,.0f}")
    print(f"    months={agg['n_mo']}  win_mo={agg['win_mo']:.1f}%  "
          f"top_mo={agg['top_mo']:.1f}%")
    print(f"    aggregate verdict: {v}  reasons: {';'.join(why) if why else 'all gates pass'}")

    # Per-cell (direction × time × cap)
    print(f"\n  Per-cell (direction × time_bucket × cap_segment):")
    hdr = (f"  {'direction':<6} {'time':<11} {'cap':<10} {'n':>5} "
           f"{'PF':>6} {'WR':>5} {'Sh':>5} {'win_mo':>7} {'top_mo':>7} "
           f"{'NET':>10} {'verdict':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    ship_cells: List[dict] = []
    survivor_cells: List[dict] = []
    cells = trades.groupby(["direction", "time_bucket", "cap_segment"], observed=True)
    for (dirn, tb, cap), g in cells:
        m = _agg(g)
        ship_v, ship_reasons = _ship_verdict(m)
        pf_str = f"{m['pf']:.2f}" if not np.isnan(m['pf']) and m['pf'] != float("inf") else "inf"
        print(f"  {dirn:<6} {tb:<11} {cap:<10} {m['n']:>5} "
              f"{pf_str:>6} {m['wr']:>5.1f} {m['sharpe']:>5.2f} "
              f"{m['win_mo']:>6.1f}% {m['top_mo']:>6.1f}% "
              f"{m['net']:>10,.0f} {ship_v:>8}")
        if ship_v == "SHIP":
            ship_cells.append({
                "cell": f"{dirn}|{tb}|{cap}", **m,
            })
        if (m["n"] >= N_SURV and not np.isnan(m["pf"]) and
                m["pf"] >= PF_SURV and ship_v != "SHIP"):
            survivor_cells.append({
                "cell": f"{dirn}|{tb}|{cap}", **m,
            })

    print(f"\n  Per-month winning %% breakdown (overall):")
    trades = trades.copy()
    trades["_mo"] = pd.to_datetime(trades["T0_signal_date"]).dt.strftime("%Y-%m")
    monthly = trades.groupby("_mo")["net_pnl"].sum()
    abs_net = float(abs(monthly.sum())) or 1.0
    for mo, net in monthly.items():
        share = 100.0 * abs(net) / abs_net
        flag = "  W" if net > 0 else "  L"
        print(f"    {mo}  NET=Rs.{net:>11,.0f}  share={share:5.1f}%  {flag}")

    print(f"\n  Exit-reason mix:")
    for rsn, g in trades.groupby("exit_reason"):
        n2 = len(g); net2 = g["net_pnl"].sum()
        print(f"    {rsn:<18} n={n2:>5} NET=Rs.{net2:>11,.0f}")

    print(f"\n  SHIP-ELIGIBLE CELLS (all 5 gauntlet-v2 gates pass): {len(ship_cells)}")
    for c in ship_cells:
        print(f"    {c['cell']}: n={c['n']} PF={c['pf']:.2f} Sh={c['sharpe']:.2f} "
              f"win_mo={c['win_mo']:.0f}% top_mo={c['top_mo']:.0f}%")
    print(f"\n  SURVIVOR CELLS (PF>={PF_SURV}, n>={N_SURV}, not ship-eligible): {len(survivor_cells)}")
    for c in survivor_cells:
        print(f"    {c['cell']}: n={c['n']} PF={c['pf']:.2f} Sh={c['sharpe']:.2f} "
              f"win_mo={c['win_mo']:.0f}% top_mo={c['top_mo']:.0f}%")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos", action="store_true",
                        help="Also run OOS window after Discovery")
    args = parser.parse_args()

    print(">>> Inside Bar breakout — PURE MATH SANITY <<<")

    # Universe
    allowed, cap_map = build_universe()
    if not allowed:
        print("[ABORT] empty universe")
        return 2

    # Regime-break preflight
    regime_preflight(DISCOVERY_START, DISCOVERY_END, "Discovery")
    if args.oos:
        regime_preflight(OOS_START, OOS_END, "OOS")

    # ---- Discovery ----
    print("\n" + "#" * 78)
    print("# DISCOVERY")
    print("#" * 78)
    big5m = load_period_5m(DISCOVERY_START, DISCOVERY_END, allowed, cap_map)
    if big5m.empty:
        print("[ABORT] no discovery bars")
        return 2

    events = find_events(big5m)
    trades = simulate(events, big5m)
    del big5m, events
    gc.collect()

    out_dir = _REPO / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "inside_bar_breakout_trades.csv"
    if not trades.empty:
        trades.to_csv(out_csv, index=False)
        print(f"\nTrade log: {out_csv}")

    report(trades, label="Discovery 2024-09 .. 2025-09")

    disc_agg = _agg(trades) if not trades.empty else None
    disc_v, disc_reasons = _ship_verdict(disc_agg) if disc_agg else ("RETIRE", ["no trades"])
    # Decision on OOS:
    #   - If aggregate fails clearly (PF < 1.10) AND no ship-eligible cells, skip OOS.
    #   - If marginal (PF 1.10..1.30 OR ship cells exist), run OOS.

    if not args.oos:
        print("\n[done] OOS not requested. Re-run with --oos if marginal.")
        return 0

    # ---- OOS ----
    print("\n" + "#" * 78)
    print("# OOS")
    print("#" * 78)
    big5m_oos = load_period_5m(OOS_START, OOS_END, allowed, cap_map)
    if big5m_oos.empty:
        print("[ABORT] no OOS bars")
        return 0
    events_oos = find_events(big5m_oos)
    trades_oos = simulate(events_oos, big5m_oos)
    del big5m_oos, events_oos
    gc.collect()

    out_csv_oos = out_dir / "inside_bar_breakout_trades_oos.csv"
    if not trades_oos.empty:
        trades_oos.to_csv(out_csv_oos, index=False)
        print(f"\nOOS trade log: {out_csv_oos}")
    report(trades_oos, label="OOS 2025-10 .. 2026-04")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
