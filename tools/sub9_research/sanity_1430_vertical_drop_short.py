"""14:30 IST Vertical-Drop SHORT — sanity gauntlet.

Indian-microstructure pattern (Candidate 2, specs/2026-05-15-research-
indian-trader-mechanics.md). Sourced from NSE working paper NSE-WP-2023-04
plus Vivek Bajaj (Stockedge). Two micro-flows converge at 14:30 IST on
NIFTY-50 / top-F&O constituents:

  1. MIS forced-squareoff drift — MIS longs begin unwinding 14:30..15:20.
  2. Institutional index hedging executes at 14:30 IST cutoffs.

Net-sell pressure on already-weak large-cap names. Distinct from the
retired `mis_unwind` setup (which used small/mid-cap + VWAP); this uses
LARGE-cap + a HARD 14:30 IST timestamp. Portfolio-complementary to
gap_fade_short (different time-of-day, different universe).

PATTERN — at end of bar N (5m bars; bar that ends at 14:30 IST i.e.
14:25-14:30):
  session_time(bar[N]) == 14:30 IST                          (hard timestamp)
  bar[N].close < bar[N-1].close                              (current bar down)
  bar[N-1].close < bar[N-2].close                            (2 consecutive down)
  bar[N].volume > avg(bar[N-3..N-12].vol) * 1.5              (vol spike, 10-bar baseline)
  symbol IN nifty_50 OR symbol IN top_30_by_FO_volume        (index-heavy only)

DIRECTION: SHORT at open of bar N+1 (14:35 IST entry).

EXITS:
  SL: bar[N].high + 0.10 * (bar[N].high - bar[N].low)
  T1 (1.0R): 50% qty exits
  T2 (1.5R): remainder, or trail-stop at break-above bar[N+1].high
  Time stop: 15:10 IST  (must close before 15:20 MIS squareoff)

PERIODS (production gauntlet schema):
  Discovery: 2023-01-01 .. 2024-12-31   (24mo)
  OOS:       2025-01-01 .. 2025-09-30   ( 9mo)
  Holdout:   2025-10-01 .. 2026-04-30   ( 7mo)

OOM-SAFE PATTERNS (Windows Python 3.10):
  - dt.floor("D") not .dt.date
  - sort_values inplace mergesort; avoid reset_index(drop=True) on big frames
  - np.r_[True, sym_arr[1:] != sym_arr[:-1]] for group boundaries
  - per-month chunking (intraday-only, no cross-day priors)
  - filter to universe BEFORE big load (universe is ~50-80 symbols)
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
_SUB9 = _REPO / "tools" / "sub9_research"
if str(_SUB9) not in sys.path:
    sys.path.insert(0, str(_SUB9))

from services.regime_break_detector import check_window  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
from _cell_mine_tier_a import scan_cells  # noqa: E402


# ---- Locked DEFAULT pattern params (Discovery run) -----------------------
TRIGGER_HHMM = "14:30"           # bar[N] ends at 14:30 IST (the 14:25-14:30 bar)
VOL_RATIO_MIN = 1.5              # bar[N].vol > 1.5 * avg(bar[N-3..N-12].vol)
VOL_BASELINE_LEN = 10            # 10-bar baseline (bar[N-3] .. bar[N-12])
VOL_BASELINE_GAP = 2             # skip immediate prior 2 bars to avoid contamination
SL_BUFFER_FRAC = 0.10            # SL = bar[N].high + 10% * range(bar[N])
T1_R_MULT = 1.0
T2_R_MULT = 1.5
T1_QTY_PCT = 0.5
USE_TRAIL_BREAK_HIGH = True      # after T1, trail stop = bar[N+1].high
TIME_STOP_HHMM = "15:10"         # exit before 15:20 MIS squareoff
RISK_PER_TRADE_RUPEES = 1000

# Top-N F&O symbols (by aggregate volume in the period) to UNION with NIFTY-50.
TOP_FO_N = 30

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

# Survivor thresholds
N_SURV = 100
PF_SURV = 1.20

_KEEP_5M_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]
_OUT_DIR = _REPO / "reports" / "sub9_sanity"


# =========================================================================
# Universe
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


def _load_nifty50_list() -> Set[str]:
    fp = _REPO / "assets" / "ind_nifty50list.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp)
    if "Symbol" not in df.columns:
        return set()
    syms = df["Symbol"].dropna().astype(str).str.strip()
    syms = syms.str.replace(r"^NSE:", "", regex=True)
    return set(syms.tolist())


def _load_fno_liquid_200() -> Set[str]:
    fp = _REPO / "assets" / "fno_liquid_200.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp)
    out: Set[str] = set()
    for c in df.columns:
        if "symbol" in c.lower():
            vals = df[c].dropna().astype(str).str.strip()
            vals = vals.str.replace(r"^NSE:", "", regex=True)
            out |= set(vals.tolist())
    return out


def _compute_top_fo_by_volume(
    nifty50: Set[str],
    fno_pool: Set[str],
    nse_meta: Dict[str, dict],
    discovery_start: date,
    discovery_end: date,
    top_n: int = TOP_FO_N,
) -> Set[str]:
    """Aggregate 5m bar volume over Discovery for FNO pool minus NIFTY-50,
    return top-N by total rupee-volume proxy (close*volume).

    Computed only ONCE per script run, on Discovery period — keeps universe
    static for OOS/Holdout (no look-ahead).
    """
    candidates = (fno_pool - nifty50)
    # Drop bad cap / MIS-ineligible upfront
    valid: Set[str] = set()
    for s in candidates:
        m = nse_meta.get(s)
        if m is None:
            continue
        if m["mis_leverage"] < 1.0:
            continue
        valid.add(s)
    if not valid:
        return set()

    print(f"  [top-FO] aggregating volume across {len(valid):,} F&O candidates over Discovery ...")
    agg: Dict[str, float] = {}
    for y, m in _months_between(discovery_start, discovery_end):
        path = (_REPO / "backtest-cache-download" / "monthly"
                / f"{y:04d}_{m:02d}_5m_enriched.feather")
        if not path.exists():
            continue
        try:
            df = pd.read_feather(path, columns=["symbol", "close", "volume"])
        except Exception:
            continue
        df = df[df["symbol"].isin(valid)]
        if df.empty:
            continue
        # Rupee-volume proxy = close * volume
        df["_rv"] = df["close"].astype("float64") * df["volume"].astype("float64")
        grp = df.groupby("symbol", observed=True)["_rv"].sum()
        for sym, v in grp.items():
            agg[sym] = agg.get(sym, 0.0) + float(v)
        del df
        gc.collect()
    if not agg:
        return set()
    sorted_syms = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    top = [s for s, _ in sorted_syms[:top_n]]
    print(f"  [top-FO] selected top {len(top)} by Discovery-period rupee-volume")
    print(f"  [top-FO] top 10 sample: {top[:10]}")
    return set(top)


def build_universe() -> Tuple[Set[str], Dict[str, str]]:
    """Universe = NIFTY-50 (from CSV) UNION top-30 F&O (by Discovery
    rupee-volume, excluding NIFTY-50 to avoid double-counting).

    Returns (allowed_symbols, membership_map) where membership_map[sym]
    is "nifty_50" or "top_30_fo_only".
    """
    print("\n[universe] loading nse_all.json ...")
    nse_meta = _load_nse_all()
    print(f"  nse_all entries: {len(nse_meta):,}")

    print("[universe] loading NIFTY-50 list ...")
    nifty50 = _load_nifty50_list()
    print(f"  nifty_50 raw: {len(nifty50):,}")

    print("[universe] loading FNO liquid 200 pool ...")
    fno_pool = _load_fno_liquid_200()
    print(f"  fno_liquid_200 raw: {len(fno_pool):,}")

    # Filter NIFTY-50 to those with valid MIS metadata
    nifty50_valid: Set[str] = set()
    membership: Dict[str, str] = {}
    drop_no_meta = drop_mis = 0
    for s in nifty50:
        m = nse_meta.get(s)
        if m is None:
            drop_no_meta += 1
            continue
        if m["mis_leverage"] < 1.0:
            drop_mis += 1
            continue
        nifty50_valid.add(s)
        membership[s] = "nifty_50"
    print(f"  nifty_50 valid:    {len(nifty50_valid):,}  (drop_no_meta={drop_no_meta}, drop_mis={drop_mis})")

    print("[universe] computing top-FO by Discovery rupee-volume ...")
    top_fo = _compute_top_fo_by_volume(
        nifty50_valid, fno_pool, nse_meta, DISCOVERY_START, DISCOVERY_END, TOP_FO_N)
    for s in top_fo:
        if s not in membership:
            membership[s] = "top_30_fo_only"

    allowed = nifty50_valid | top_fo
    print(f"  FINAL universe: {len(allowed):,}  (nifty_50={len(nifty50_valid)}, top_30_fo_only={len(top_fo)})")
    return allowed, membership


# =========================================================================
# Regime preflight
# =========================================================================

def regime_preflight(start: date, end: date, label: str):
    print(f"\n[regime] preflight check on {label} window {start} .. {end}")
    hits = check_window(
        "vertical_drop_1430_short",
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
# Month-by-month loader (OOM-safe)
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
                    membership: Dict[str, str],
                    period_start: date, period_end: date) -> pd.DataFrame:
    """Load one month's 5m bars, filter to universe + clip to (period_start,
    period_end) bounds. Universe is small (~50-80), so the loaded frame is
    far smaller than for whole-NSE setups.
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

    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    df["nifty_membership"] = df["symbol"].map(membership).astype("category")

    # Vectorized day-floor + hhmm (OOM-safe)
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

    p_lo = pd.Timestamp(period_start)
    p_hi = pd.Timestamp(period_end)
    df = df[(df["_day"] >= p_lo) & (df["_day"] <= p_hi)]
    if df.empty:
        return df

    df.sort_values(["symbol", "date"], kind="mergesort", inplace=True)
    return df


# =========================================================================
# Event detection + simulator
# =========================================================================

def _scan_and_simulate(
    bars: pd.DataFrame,
    vol_ratio_min: float = VOL_RATIO_MIN,
    vol_baseline_len: int = VOL_BASELINE_LEN,
    vol_baseline_gap: int = VOL_BASELINE_GAP,
    sl_buffer_frac: float = SL_BUFFER_FRAC,
    t1_r_mult: float = T1_R_MULT,
    t2_r_mult: float = T2_R_MULT,
    t1_qty_pct: float = T1_QTY_PCT,
    use_trail_break_high: bool = USE_TRAIL_BREAK_HIGH,
    time_stop_hhmm: str = TIME_STOP_HHMM,
    trigger_hhmm: str = TRIGGER_HHMM,
) -> List[dict]:
    """At end of bar that closes at 14:30 IST (bar[N]):
      - bar[N].close < bar[N-1].close < bar[N-2].close (2 consecutive down)
      - bar[N].volume > 1.5 * avg(bar[N-3..N-12].vol)
      - session_open computed from FIRST bar of the day
    Entry: SHORT at open of bar[N+1] (14:35 IST).
    SL: bar[N].high + 10% * range. T1=1R (50%), T2=1.5R, trail
    bar[N+1].high after T1. Time stop = 15:10 IST.
    """
    if bars.empty:
        return []

    sym_arr = bars["symbol"].values
    day_arr = bars["_day"].values
    sym_changed = np.r_[True, sym_arr[1:] != sym_arr[:-1]]
    day_changed = np.r_[True, day_arr[1:] != day_arr[:-1]]
    grp_starts = np.where(sym_changed | day_changed)[0]
    ends = np.concatenate([grp_starts[1:], [len(bars)]])

    open_arr = bars["open"].values.astype("float32")
    high_arr = bars["high"].values.astype("float32")
    low_arr = bars["low"].values.astype("float32")
    close_arr = bars["close"].values.astype("float32")
    vol_arr = bars["volume"].values.astype("float32")
    hhmm_arr = bars["hhmm"].values
    ts_arr = bars["date"].values
    memb_arr = bars["nifty_membership"].astype(str).values

    # Minimum bars before bar[N] = vol_baseline_gap + vol_baseline_len (10+2=12)
    min_lookback = vol_baseline_gap + vol_baseline_len  # 12

    trades: List[dict] = []

    for grp_idx in range(len(grp_starts)):
        s = int(grp_starts[grp_idx])
        e = int(ends[grp_idx])
        n_bars = e - s
        if n_bars < min_lookback + 3:
            continue

        # Find bar[N] = bar at trigger_hhmm
        # Iterate from s+min_lookback to e-2 (need bar[N+1] for entry)
        bar_n_abs = None
        for i in range(s + min_lookback, e - 1):
            if hhmm_arr[i] == trigger_hhmm:
                bar_n_abs = i
                break
            elif hhmm_arr[i] > trigger_hhmm:
                # Trigger time missed (gap in bars across 14:30 IST)
                break

        if bar_n_abs is None:
            continue

        i = bar_n_abs

        # 2 consecutive down closes: bar[N].close < bar[N-1].close < bar[N-2].close
        c_n = float(close_arr[i])
        c_n1 = float(close_arr[i - 1])
        c_n2 = float(close_arr[i - 2])
        if not (c_n < c_n1 < c_n2):
            continue

        # Volume spike: bar[N].vol > 1.5 * avg(bar[N-3..N-12].vol)
        # i.e. baseline indices [i-12, i-11, ..., i-3]  (10 bars, skip 2 bars)
        # vol_baseline_gap=2 -> skip i-1 and i-2
        # vol_baseline_len=10 -> use i-3..i-12
        base_lo = i - (vol_baseline_gap + vol_baseline_len)  # i - 12
        base_hi = i - vol_baseline_gap                        # i - 2 (exclusive)
        if base_lo < s:
            continue
        vol_base = float(vol_arr[base_lo:base_hi].mean())
        if vol_base <= 0:
            continue
        vol_n = float(vol_arr[i])
        vol_ratio = vol_n / vol_base
        if vol_ratio < vol_ratio_min:
            continue

        # Need bar[N+1] for entry
        entry_idx = i + 1
        if entry_idx >= e:
            continue
        if hhmm_arr[entry_idx] != "14:35":
            # Bar after 14:30 must be 14:35 — if gap in feed, skip
            continue

        h_n = float(high_arr[i])
        l_n = float(low_arr[i])
        bar_n_range = h_n - l_n
        if bar_n_range <= 0:
            continue

        entry = float(open_arr[entry_idx])

        hard_sl = h_n + sl_buffer_frac * bar_n_range
        stop_dist = hard_sl - entry
        if stop_dist <= 0:
            continue

        t1_target = entry - t1_r_mult * stop_dist
        t2_target = entry - t2_r_mult * stop_dist

        # Session open + intraday weakness severity
        sess_open = float(open_arr[s])
        intraday_open_to_1430_pct = (c_n - sess_open) / sess_open * 100.0 if sess_open > 0 else 0.0

        # Simulate exits from entry_idx
        bar_n1_high = float(high_arr[entry_idx])
        i_end = e

        hit_t1 = False
        t1_exit = None
        exit_price = None
        exit_ts = None
        exit_reason = None

        for j in range(entry_idx, i_end):
            hj = float(high_arr[j])
            lj = float(low_arr[j])
            cj = float(close_arr[j])
            hhmm_j = hhmm_arr[j]

            if hit_t1 and use_trail_break_high:
                # After T1, trail-stop = bar[N+1].high (break-ABOVE = exit
                # for a SHORT; price rallying back above N+1.high signals
                # exhaustion of the down-move).
                active_sl = bar_n1_high
            else:
                active_sl = hard_sl

            # SHORT: SL hit if high >= active_sl
            if hj >= active_sl:
                exit_price = active_sl
                exit_ts = ts_arr[j]
                exit_reason = "trail_break_above" if (hit_t1 and use_trail_break_high) else "stop"
                break

            # T1 / T2 hit checks
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
            if last_j < entry_idx:
                continue
            exit_price = float(close_arr[last_j])
            exit_ts = ts_arr[last_j]
            exit_reason = "session_end"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_dist, 1e-6)), 1)
        side = "SELL"
        if hit_t1:
            qty_t1 = int(qty * t1_qty_pct)
            qty_t2 = qty - qty_t1
            if qty_t1 == 0:
                realized = (entry - exit_price) * qty_t2
                fee = calc_fee(entry, exit_price, qty_t2, side)
                blended = exit_price
            else:
                pnl_t1 = (entry - t1_exit) * qty_t1
                pnl_t2 = (entry - exit_price) * qty_t2
                realized = pnl_t1 + pnl_t2
                fee = (calc_fee(entry, t1_exit, qty_t1, side)
                       + calc_fee(entry, exit_price, qty_t2, side))
                blended = (t1_exit * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            realized = (entry - exit_price) * qty
            fee = calc_fee(entry, exit_price, qty, side)
            blended = exit_price
        net_pnl = realized - fee

        # Prior-day close drift bucket: approximate by using session-open of
        # current day as a proxy if we don't load prior-day data. We don't
        # have the prior daily close here; use intraday_open_to_1430 as the
        # primary "weakness" dimension instead. We add a session_open_vs_lo
        # bucket as a secondary structural dim.

        trades.append({
            "T0_signal_date": pd.Timestamp(day_arr[i]).to_pydatetime().date(),
            "symbol": "NSE:" + str(sym_arr[i]),
            "bare_symbol": str(sym_arr[i]),
            "nifty_membership": memb_arr[i],
            "direction": "SHORT",
            "bar_n_hhmm": str(hhmm_arr[i]),
            "entry_ts": ts_arr[entry_idx],
            "entry_hhmm": str(hhmm_arr[entry_idx]),
            "entry_price": entry,
            "session_open": sess_open,
            "intraday_open_to_1430_pct": intraday_open_to_1430_pct,
            "bar_n_high": h_n,
            "bar_n_low": l_n,
            "bar_n_close": c_n,
            "bar_n_range": bar_n_range,
            "bar_n1_high": bar_n1_high,
            "vol_ratio": vol_ratio,
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
# Period runner
# =========================================================================

def run_period_month_by_month(
    label: str,
    start: date,
    end: date,
    allowed: Set[str],
    membership: Dict[str, str],
) -> pd.DataFrame:
    print(f"\n[{label}] running month-by-month {start} .. {end} ...")
    all_trades: List[dict] = []
    for y, m in _months_between(start, end):
        bars = _load_one_month(y, m, allowed, membership, start, end)
        if bars.empty:
            print(f"  {y}-{m:02d}: SKIPPED (no bars)")
            continue
        n_bars = len(bars)
        trades_this_month = _scan_and_simulate(bars)
        all_trades.extend(trades_this_month)
        print(f"  {y}-{m:02d}: bars={n_bars:>9,}  trades={len(trades_this_month):>5,}  cum={len(all_trades):>6,}")
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


def prep_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add ex-ante (pre-trade) cell dimensions. No outcome leakage.

    Cell dims (no exit_reason, no hit_t1, no exit_price, no realized_pnl):
      - nifty_membership (nifty_50 / top_30_fo_only)
      - dow / month
      - R_pct_bucket (stop_distance / entry_price)
      - bar_N_range_pct_bucket (bar_n_range / bar_n_close)
      - vol_ratio_bucket
      - intraday_open_to_1430_pct_bucket  (intraday weakness severity)
    """
    df = df.copy()
    d = pd.to_datetime(df["T0_signal_date"])
    df["_session_date"] = d.dt.date
    df["_month"] = d.dt.strftime("%Y-%m")
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")

    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_pct_bucket"] = _bucket_pct(
        r_pct,
        edges=[-np.inf, 0.4, 0.8, 1.5, np.inf],
        labels=["<0.4%", "0.4-0.8%", "0.8-1.5%", "1.5%+"],
    )

    bn_range_pct = df["bar_n_range"] / df["bar_n_close"] * 100.0
    df["bar_N_range_pct_bucket"] = _bucket_pct(
        bn_range_pct,
        edges=[-np.inf, 0.3, 0.6, 1.2, np.inf],
        labels=["<0.3%", "0.3-0.6%", "0.6-1.2%", "1.2%+"],
    )

    df["vol_ratio_bucket"] = _bucket_pct(
        df["vol_ratio"],
        edges=[1.5, 2.0, 3.0, np.inf],
        labels=["1.5-2.0", "2.0-3.0", "3.0+"],
    )

    df["intraday_open_to_1430_pct_bucket"] = _bucket_pct(
        df["intraday_open_to_1430_pct"],
        edges=[-np.inf, -2.0, -1.0, -0.3, 0.3, np.inf],
        labels=["<-2%", "-2..-1%", "-1..-0.3%", "-0.3..+0.3%", ">+0.3%"],
    )

    return df


def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _aggregate_metrics(df: pd.DataFrame) -> dict:
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
# Cell mining (ex-ante dims only)
# =========================================================================

CELL_DIMS = [
    "nifty_membership",
    "dow",
    "month",
    "R_pct_bucket",
    "bar_N_range_pct_bucket",
    "vol_ratio_bucket",
    "intraday_open_to_1430_pct_bucket",
]


def report_period(trades: pd.DataFrame, label: str,
                  cellmine: bool = True) -> Tuple[dict, pd.DataFrame, int]:
    print("\n" + "=" * 78)
    print(f"REPORT — vertical_drop_1430_short — {label}")
    print("=" * 78)
    if trades.empty:
        print("  NO TRADES")
        return {}, pd.DataFrame(), 0

    n_total = len(trades)
    mem_dist = trades["nifty_membership"].value_counts().to_dict()
    er_dist = trades["exit_reason"].value_counts().to_dict()
    print(f"\n  Funnel: events={n_total:,}")
    print(f"    by_membership: {mem_dist}")
    print(f"    by_exit_reason: {er_dist}")

    agg = _aggregate_metrics(trades)
    v, why = _ship_verdict(agg)
    pf_str = f"{agg['pf']:.3f}" if (not np.isnan(agg["pf"]) and agg["pf"] != float("inf")) else "inf"
    print(f"\n  AGGREGATE:")
    print(f"    n={agg['n']:,}  PF={pf_str}  WR={agg['wr']:.1f}%  "
          f"Sharpe={agg['sharpe']:.3f}  NET=Rs.{agg['net']:,.0f}")
    print(f"    months={agg['n_mo']}  win_mo={agg['win_mo']:.1f}%  "
          f"top_mo={agg['top_mo']:.1f}%")
    print(f"    aggregate verdict: {v}  reasons: {';'.join(why) if why else 'all gates pass'}")

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
        return agg, pd.DataFrame(), 0

    print(f"\n  [cell-mine] preparing dimensions + scanning ...")
    cells_df = prep_cells(trades)
    dims_have = [d for d in CELL_DIMS if d in cells_df.columns]
    print(f"  dims scanned: {dims_have}")
    cells = scan_cells(
        cells_df, dims_have, "net_pnl", max_combo=3,
        date_col="_session_date", month_col="_month",
    )
    cells_scanned = int(len(cells))
    print(f"  total cells scanned: {cells_scanned:,}")

    ship_mask = (
        (cells["n"] >= N_SHIP)
        & (cells["pf"] >= PF_SHIP)
        & (cells["sharpe"] >= SHARPE_SHIP)
        & (cells["win_mo_pct"] >= WIN_MO_PCT_SHIP)
        & (cells["top_mo_pct"] < TOP_MO_PCT_SHIP)
    )
    ship_eligible = cells[ship_mask].sort_values(["pf", "n"], ascending=[False, False]).copy()

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

    out_cells = _OUT_DIR / f"vertical_drop_1430_cells_{label.lower()}.csv"
    cells.to_csv(out_cells, index=False)
    print(f"\n  wrote cell-mine CSV: {out_cells.relative_to(_REPO)}")

    if len(ship_eligible):
        out_ship = _OUT_DIR / f"vertical_drop_1430_ship_cells_{label.lower()}.csv"
        ship_eligible.to_csv(out_ship, index=False)
        print(f"  wrote SHIP cells: {out_ship.relative_to(_REPO)} ({len(ship_eligible)} cells)")

    return agg, ship_eligible, cells_scanned


# =========================================================================
# Cell-signature matching across periods
# =========================================================================

def _parse_cell_signature(cell_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in cell_str.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def filter_trades_to_cell(trades: pd.DataFrame, cell_dict: Dict[str, str]) -> pd.DataFrame:
    prepped = prep_cells(trades)
    mask = pd.Series(True, index=prepped.index)
    for k, v in cell_dict.items():
        if k not in prepped.columns:
            return pd.DataFrame()
        col = prepped[k].astype(str)
        mask &= (col == v)
    return prepped[mask]


def validate_cells_on_period(ship_cells: pd.DataFrame, period_trades: pd.DataFrame,
                             period_label: str) -> pd.DataFrame:
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
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-oos", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()

    print(">>> 14:30 IST Vertical-Drop SHORT — PROPER 3-PERIOD GAUNTLET <<<")
    print(f"   Discovery: {DISCOVERY_START} .. {DISCOVERY_END}  ({len(_months_between(DISCOVERY_START, DISCOVERY_END))}mo)")
    print(f"   OOS:       {OOS_START} .. {OOS_END}  ({len(_months_between(OOS_START, OOS_END))}mo)")
    print(f"   Holdout:   {HOLDOUT_START} .. {HOLDOUT_END}  ({len(_months_between(HOLDOUT_START, HOLDOUT_END))}mo)")
    print(f"   Trigger: bar[N]@{TRIGGER_HHMM} IST, 2 down closes, vol > {VOL_RATIO_MIN}x 10-bar baseline (skip last 2 bars)")
    print(f"   Direction: SHORT @ open(bar[N+1] = 14:35 IST)")
    print(f"   SL: bar[N].high + {SL_BUFFER_FRAC:.0%} * bar[N].range")
    print(f"   T1={T1_R_MULT}R @ {T1_QTY_PCT:.0%} qty, T2={T2_R_MULT}R, trail-break-high after T1, "
          f"time_stop={TIME_STOP_HHMM}")
    print(f"   Universe: NIFTY-50 UNION top-{TOP_FO_N}-F&O-by-volume")

    # ---- Universe + preflight ----
    allowed, membership = build_universe()
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
    disc_trades_path = _OUT_DIR / "vertical_drop_1430_trades_discovery.csv"

    if args.skip_discovery and disc_trades_path.exists():
        print(f"  [skip] loading existing trades from {disc_trades_path}")
        disc_trades = pd.read_csv(disc_trades_path)
    else:
        disc_trades = run_period_month_by_month(
            "Discovery", DISCOVERY_START, DISCOVERY_END, allowed, membership)
        if not disc_trades.empty:
            disc_trades.to_csv(disc_trades_path, index=False)
            print(f"  wrote {disc_trades_path.relative_to(_REPO)} ({len(disc_trades):,} trades)")

    disc_agg, disc_ship, disc_cells_scanned = report_period(disc_trades, label="Discovery", cellmine=True)
    n_ship_disc = len(disc_ship)

    expected_fp = disc_cells_scanned * 0.05
    print(f"\n  [bonferroni] {disc_cells_scanned:,} cells x alpha=0.05 -> expected "
          f"{expected_fp:.0f} false positives at chance. ship_eligible={n_ship_disc}.")

    # Regime observation: per-quarter / per-half-year breakdown
    if not disc_trades.empty:
        print(f"\n  [regime-observation] per-half-year aggregate (regime-conditional check):")
        _t = disc_trades.copy()
        _t["_dt"] = pd.to_datetime(_t["T0_signal_date"])
        _t["_half"] = _t["_dt"].dt.year.astype(str) + "-H" + ((_t["_dt"].dt.month - 1) // 6 + 1).astype(str)
        for h, sub in _t.groupby("_half"):
            m = _aggregate_metrics(sub)
            pf_str = f"{m['pf']:.2f}" if (not np.isnan(m['pf']) and m['pf'] != float('inf')) else "inf"
            print(f"    {h}: n={m['n']:>4,} PF={pf_str:>5} WR={m['wr']:>5.1f}% "
                  f"Sh={m['sharpe']:>5.2f} NET=Rs.{m['net']:>11,.0f}")

    if n_ship_disc == 0:
        print("\n[VERDICT] Discovery has 0 ship-eligible cells. RETIRE — no OOS/Holdout run.")
        return 0

    # ---- OOS ----
    print("\n" + "#" * 78)
    print("# OOS  (2025-01-01 .. 2025-09-30)")
    print("#" * 78)
    oos_trades_path = _OUT_DIR / "vertical_drop_1430_trades_oos.csv"
    if args.skip_oos and oos_trades_path.exists():
        oos_trades = pd.read_csv(oos_trades_path)
    else:
        oos_trades = run_period_month_by_month(
            "OOS", OOS_START, OOS_END, allowed, membership)
        if not oos_trades.empty:
            oos_trades.to_csv(oos_trades_path, index=False)
            print(f"  wrote {oos_trades_path.relative_to(_REPO)} ({len(oos_trades):,} trades)")

    oos_agg, _, _ = report_period(oos_trades, label="OOS", cellmine=False)

    oos_validation = validate_cells_on_period(disc_ship, oos_trades, "OOS")
    if not oos_validation.empty:
        oos_val_path = _OUT_DIR / "vertical_drop_1430_oos_validation.csv"
        oos_validation.to_csv(oos_val_path, index=False)
        print(f"\n  wrote OOS validation: {oos_val_path.relative_to(_REPO)}")

    oos_ship = oos_validation[oos_validation["verdict"] == "SHIP"]
    print(f"\n  CELLS THAT PASSED DISCOVERY -> OOS: {len(oos_ship)}/{n_ship_disc}")
    if len(oos_ship) == 0:
        print("\n[VERDICT] No cells survived OOS. RETIRE.")
        return 0

    # ---- HOLDOUT ----
    print("\n" + "#" * 78)
    print("# HOLDOUT  (2025-10-01 .. 2026-04-30)")
    print("#" * 78)
    hold_trades_path = _OUT_DIR / "vertical_drop_1430_trades_holdout.csv"
    if args.skip_holdout and hold_trades_path.exists():
        hold_trades = pd.read_csv(hold_trades_path)
    else:
        hold_trades = run_period_month_by_month(
            "Holdout", HOLDOUT_START, HOLDOUT_END, allowed, membership)
        if not hold_trades.empty:
            hold_trades.to_csv(hold_trades_path, index=False)
            print(f"  wrote {hold_trades_path.relative_to(_REPO)} ({len(hold_trades):,} trades)")

    hold_agg, _, _ = report_period(hold_trades, label="Holdout", cellmine=False)

    survivors_oos = oos_validation[oos_validation["verdict"] == "SHIP"]
    surv_cells_df = disc_ship[disc_ship["cell"].isin(survivors_oos["cell"])]
    hold_validation = validate_cells_on_period(surv_cells_df, hold_trades, "Holdout")
    if not hold_validation.empty:
        hold_val_path = _OUT_DIR / "vertical_drop_1430_holdout_validation.csv"
        hold_validation.to_csv(hold_val_path, index=False)
        print(f"\n  wrote Holdout validation: {hold_val_path.relative_to(_REPO)}")

    hold_ship = hold_validation[hold_validation["verdict"] == "SHIP"]
    print(f"\n  CELLS THAT PASSED DISC -> OOS -> HOLDOUT: {len(hold_ship)}/{len(oos_ship)}")

    if len(hold_ship) == 0:
        print("\n[VERDICT] No cells survived Holdout. RETIRE.")
        return 0

    # ---- FINAL evidence dump ----
    print("\n" + "#" * 78)
    print("# FINAL EVIDENCE DUMP — triple-survivor cells")
    print("#" * 78)
    for _, hr in hold_ship.iterrows():
        cell_str = hr["cell"]
        print(f"\n  CELL: {cell_str}")
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

    print(f"\n[VERDICT] SHIP CANDIDATE — triple-period survival on {len(hold_ship)} cell(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
