"""Open=High Sustained Weakness SHORT — sanity gauntlet.

Indian-trader-sourced (Zerodha Varsity Module 5 §12.3 "Open=High signature";
SEBI 2021 working paper on pre-open auction over-discovery; Hitesh Patel,
Smallcase 2024 blog). Folk-known signature: when the day's open turns out to
be the day's high within tight tolerance for the first ~hour, the pre-open
auction over-discovered (institutional supply > retail demand at the auction
price) and the rest of the session is structurally distribution.

PATTERN — at end of bar N where bar[N].hhmm IN {10:00, 10:05, 10:10, 10:15}
(bars are left-labeled; 10:00 label = bar covering 10:00-10:05 IST):
  day_open       = open of 09:15 bar
  day_high_sofar = max(high) of bars 09:15 .. bar[N] inclusive
  |day_open - day_high_sofar| / day_open  <= 0.0010     (within 0.10%)
  bar[N].close < bar[N-1].close                          (weakness confirmation)
  cap_segment IN {small_cap, mid_cap}                    (retail-FOMO universe)

DIRECTION: SHORT at open of bar N+1.

EXITS:
  SL: day_high_sofar + 0.10 * (bar[N].high - bar[N].low)
  T1 (1.0R): 50% qty exits
  T2 (1.5R): remainder, or trail-stop at break-above bar[N+1].high after T1
  Time stop: 14:00 IST

PERIODS (production gauntlet schema):
  Discovery: 2023-01-01 .. 2024-12-31   (24mo)
  OOS:       2025-01-01 .. 2025-09-30   ( 9mo)
  Holdout:   2025-10-01 .. 2026-04-30   ( 7mo)

OOM-SAFE PATTERNS:
  - dt.floor("D") via int64 (no .dt.date on 27M-row frames)
  - groupby + transform("first") for day_open, .cummax() for day_high_sofar
  - per-month chunking
  - mergesort inplace; avoid reset_index(drop=True) on big frames
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
OPEN_HIGH_TOL_FRAC = 0.0010       # |day_open - day_high_sofar| / day_open <= 0.10%
TRIGGER_HHMM_WINDOW = {"10:00", "10:05", "10:10", "10:15"}  # bar-N labels (4-bar window)
SL_BUFFER_FRAC = 0.10              # SL = day_high_sofar + 10% * bar[N].range
T1_R_MULT = 1.0
T2_R_MULT = 1.5
T1_QTY_PCT = 0.5
USE_TRAIL_BREAK_HIGH = True        # after T1, trail-stop at bar[N+1].high (break-above = exit)
TIME_STOP_HHMM = "14:00"
RISK_PER_TRADE_RUPEES = 1000

ALLOWED_CAPS = {"small_cap", "mid_cap"}

# Time bucket for hhmm at trigger (bar[N].hhmm) — the 4-bar trigger window only.
HHMM_BUCKETS = [
    ("10:00", "10:00"),
    ("10:05", "10:05"),
    ("10:10", "10:10"),
    ("10:15", "10:15"),
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
    skipped_unknown = skipped_micro = skipped_large = skipped_mis = skipped_no_meta = 0
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
        if cap == "large_cap":
            # Open=High signature is a small/mid-cap pre-open auction
            # over-discovery phenomenon; large-caps are too efficient.
            skipped_large += 1
            continue
        if cap not in ALLOWED_CAPS:
            continue
        allowed.add(s)
        cap_map[s] = cap

    print(f"  drop no_meta:      {skipped_no_meta:,}")
    print(f"  drop mis_lev<1:    {skipped_mis:,}")
    print(f"  drop unknown_cap:  {skipped_unknown:,}")
    print(f"  drop micro_cap:    {skipped_micro:,}")
    print(f"  drop large_cap:    {skipped_large:,}  (open=high requires small/mid)")
    print(f"  FINAL allowed:     {len(allowed):,}")
    cap_dist = pd.Series(list(cap_map.values())).value_counts().to_dict()
    print(f"  cap distribution: {cap_dist}")
    return allowed, cap_map


# =========================================================================
# Daily context (prior-session close, 20d ADV, prior daily return)
# =========================================================================

def build_daily_context(allowed: Set[str]) -> pd.DataFrame:
    """Load consolidated daily feather and compute per-(symbol, d):
       - pdc  : prior session close
       - prior_day_return_pct : (close / pdc - 1) * 100  computed at session
                                d itself (= "yesterday's daily move" as seen
                                at time of intraday trigger today). The
                                value attached to a trade ON date D is the
                                row at (symbol, D-1) = previous session's
                                close/pdc — i.e. the return of the session
                                BEFORE D. We deliver this by shifting once.
       - adv_20d_cr : 20d trailing avg traded value in Rs Cr (shifted by 1
                      day so it's strictly priors-only).
       - vol_20d_avg : 20d trailing avg daily volume (shifted) — used to
                       normalize the first-hour volume ratio.

    Window: 2022-09-01 .. 2026-04-30 (need warmup for 20d).
    """
    print("\n[daily] loading consolidated_daily.feather ...")
    path = _REPO / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not path.exists():
        raise FileNotFoundError(f"missing daily cache: {path}")
    df = pd.read_feather(path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[df["symbol"].astype(str).isin(allowed)].copy()
    df = df[(df["d"] >= date(2022, 9, 1)) & (df["d"] <= HOLDOUT_END)].copy()
    df = df[["symbol", "d", "close", "volume"]]
    df["traded_value"] = df["close"] * df["volume"]
    df.sort_values(["symbol", "d"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # prior session close — strictly previous session per symbol
    df["pdc"] = df.groupby("symbol", observed=True)["close"].shift(1)

    # daily return realized on session d (today's close vs pdc).
    # When we look up trigger-date D, we want the daily return realized on
    # session D-1 (yesterday's move). For that we shift this column by 1
    # additional row → "lagged_daily_return" attached to D = return on D-1.
    df["daily_return_pct"] = (df["close"] / df["pdc"] - 1.0) * 100.0
    df["prior_session_return_pct"] = df.groupby("symbol", observed=True)[
        "daily_return_pct"
    ].shift(1)

    # 20d adv (strictly prior — shift before rolling)
    grp_tv = df.groupby("symbol", observed=True)["traded_value"]
    df["adv_20d_cr"] = grp_tv.transform(
        lambda v: v.shift(1).rolling(20, min_periods=10).mean()
    ) / 1e7
    grp_vol = df.groupby("symbol", observed=True)["volume"]
    df["vol_20d_avg"] = grp_vol.transform(
        lambda v: v.shift(1).rolling(20, min_periods=10).mean()
    )

    keep = ["symbol", "d", "pdc", "adv_20d_cr", "vol_20d_avg", "prior_session_return_pct"]
    out = df[keep].copy()
    n_sym = out["symbol"].nunique()
    n_rows = len(out)
    print(f"  daily context rows: {n_rows:,}  symbols: {n_sym:,}")
    return out


# =========================================================================
# Regime preflight
# =========================================================================

def regime_preflight(start: date, end: date, label: str):
    print(f"\n[regime] preflight check on {label} window {start} .. {end}")
    hits = check_window(
        "open_eq_high_short",
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
                    cap_map: Dict[str, str],
                    period_start: date, period_end: date) -> pd.DataFrame:
    """Load one month's 5m bars, filter to universe + clip to period."""
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
    daily_ctx: pd.DataFrame,
    open_high_tol_frac: float = OPEN_HIGH_TOL_FRAC,
    trigger_hhmm: Set[str] = TRIGGER_HHMM_WINDOW,
    sl_buffer_frac: float = SL_BUFFER_FRAC,
    t1_r_mult: float = T1_R_MULT,
    t2_r_mult: float = T2_R_MULT,
    t1_qty_pct: float = T1_QTY_PCT,
    use_trail_break_high: bool = USE_TRAIL_BREAK_HIGH,
    time_stop_hhmm: str = TIME_STOP_HHMM,
) -> List[dict]:
    """Walk every (symbol, _day) group; find first qualifying open=high
    weakness bar in trigger window; SHORT at open of next bar; simulate.
    LATCH: first qualifying event per (symbol, session_date).
    """
    if bars.empty:
        return []

    # Per-(symbol, _day) state — done once per group via groupby transforms.
    grp = bars.groupby(["symbol", "_day"], sort=False, observed=True)
    bars = bars.copy()
    bars["day_open"] = grp["open"].transform("first").astype("float32")
    bars["day_high_sofar"] = grp["high"].cummax().astype("float32")
    bars["day_low_sofar"] = grp["low"].cummin().astype("float32")
    # Cumulative volume — needed later for first-hour vol ratio.
    bars["day_cum_vol"] = grp["volume"].cumsum().astype("float64")

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
    cap_arr = bars["cap_segment"].astype(str).values
    day_open_arr = bars["day_open"].values.astype("float32")
    day_high_arr = bars["day_high_sofar"].values.astype("float32")
    day_low_arr = bars["day_low_sofar"].values.astype("float32")
    day_cum_vol_arr = bars["day_cum_vol"].values.astype("float64")

    # Pre-index daily context for fast (symbol, d) lookup.
    daily_idx = daily_ctx.set_index(["symbol", "d"])

    trades: List[dict] = []

    for grp_idx in range(len(grp_starts)):
        s = int(grp_starts[grp_idx])
        e = int(ends[grp_idx])
        n_bars = e - s
        # Need at least: bar at idx 0 (day_open established) + a bar in
        # trigger window + at least one forward bar for entry + sim.
        if n_bars < 10:
            continue

        ev = None
        for i in range(s + 1, e - 1):
            n_hhmm = hhmm_arr[i]
            if n_hhmm not in trigger_hhmm:
                # Iterate strictly within trigger window. Bars are
                # ordered by time ascending — once we pass 10:15 we can
                # short-circuit.
                if n_hhmm > "10:15":
                    break
                continue

            do = float(day_open_arr[i])
            dhs = float(day_high_arr[i])
            if do <= 0:
                continue
            # |day_open - day_high_sofar| / day_open <= tol
            diff_frac = abs(do - dhs) / do
            if diff_frac > open_high_tol_frac:
                continue

            # bar[N].close < bar[N-1].close
            c_n = float(close_arr[i])
            c_nm1 = float(close_arr[i - 1])
            if not (c_n < c_nm1):
                continue

            # Entry at OPEN of bar[N+1]
            entry_idx = i + 1
            if entry_idx >= e:
                continue
            entry_hhmm = hhmm_arr[entry_idx]
            entry = float(open_arr[entry_idx])

            h_n = float(high_arr[i])
            l_n = float(low_arr[i])
            bar_n_range = h_n - l_n
            if bar_n_range <= 0:
                # degenerate bar
                continue

            # SL = day_high_sofar + 10% * bar[N].range
            hard_sl = dhs + sl_buffer_frac * bar_n_range
            stop_dist = hard_sl - entry
            if stop_dist <= 0:
                # entry already above SL — invalidated by gap
                continue

            t1_target = entry - t1_r_mult * stop_dist
            t2_target = entry - t2_r_mult * stop_dist

            # Day-level metrics captured at trigger time (ex-ante).
            # intraday range pct at trigger = (high_sofar - low_sofar) / day_open
            dl_at_trigger = float(day_low_arr[i])
            intraday_range_pct = (dhs - dl_at_trigger) / do * 100.0
            # first-hour cum vol AT this bar (since trigger bars are in
            # the 10:00-10:15 window, this is "~first 60 min" cumulative).
            first_hour_cum_vol = float(day_cum_vol_arr[i])

            ev = dict(
                bar_n_idx=i,
                entry_idx=entry_idx,
                session_end=e,
                entry=entry,
                entry_hhmm=entry_hhmm,
                entry_ts=ts_arr[entry_idx],
                bar_n_hhmm=n_hhmm,
                bar_n_high=h_n,
                bar_n_low=l_n,
                bar_n_close=c_n,
                bar_n_range=bar_n_range,
                bar_n1_high=float(high_arr[entry_idx]),
                day_open=do,
                day_high_sofar=dhs,
                day_low_sofar=dl_at_trigger,
                open_high_diff_frac=diff_frac,
                intraday_range_pct=intraday_range_pct,
                first_hour_cum_vol=first_hour_cum_vol,
                hard_sl=hard_sl,
                stop_dist=stop_dist,
                t1=t1_target,
                t2=t2_target,
                sym=str(sym_arr[i]),
                d=day_arr[i],
                cap=cap_arr[i],
            )
            break  # latch: first qualifier per session

        if ev is None:
            continue

        # Attach daily context (pdc, adv, prior_return)
        sym = ev["sym"]
        d_ts = pd.Timestamp(ev["d"]).date()
        try:
            ctx = daily_idx.loc[(sym, d_ts)]
            if isinstance(ctx, pd.DataFrame):
                ctx = ctx.iloc[-1]
            pdc = float(ctx["pdc"]) if pd.notna(ctx["pdc"]) else np.nan
            adv_cr = float(ctx["adv_20d_cr"]) if pd.notna(ctx["adv_20d_cr"]) else np.nan
            vol_20d_avg = float(ctx["vol_20d_avg"]) if pd.notna(ctx["vol_20d_avg"]) else np.nan
            prior_ret_pct = (float(ctx["prior_session_return_pct"])
                             if pd.notna(ctx["prior_session_return_pct"]) else np.nan)
        except KeyError:
            pdc = adv_cr = vol_20d_avg = prior_ret_pct = np.nan

        # gap_pct = (day_open vs pdc) — open vs prior close
        if not np.isnan(pdc) and pdc > 0:
            gap_pct = (ev["day_open"] / pdc - 1.0) * 100.0
        else:
            gap_pct = np.nan

        # first-hour vol ratio vs prior 20d daily volume / ~13 (first hour
        # bars out of ~75 session bars). Crude but ex-ante.
        if not np.isnan(vol_20d_avg) and vol_20d_avg > 0:
            # 09:15-10:15 = 12 5m bars (about 16% of session). Expected
            # first-hour vol ~ vol_20d_avg * (12/75) = vol_20d_avg * 0.16.
            expected_first_hour = vol_20d_avg * (12.0 / 75.0)
            vol_first_hour_ratio = ev["first_hour_cum_vol"] / expected_first_hour
        else:
            vol_first_hour_ratio = np.nan

        # ---- Simulate exits forward ----
        entry = ev["entry"]
        hard_sl = ev["hard_sl"]
        stop_dist = ev["stop_dist"]
        t1_target = ev["t1"]
        t2_target = ev["t2"]
        bar_n1_high = ev["bar_n1_high"]
        i_start = ev["entry_idx"]
        i_end = ev["session_end"]

        hit_t1 = False
        t1_exit = None
        exit_price = None
        exit_ts = None
        exit_reason = None

        for j in range(i_start, i_end):
            hj = float(high_arr[j])
            lj = float(low_arr[j])
            cj = float(close_arr[j])
            hhmm_j = hhmm_arr[j]

            # After T1, trail stop to bar[N+1].high (any break above = exit).
            if hit_t1 and use_trail_break_high:
                active_sl = bar_n1_high
            else:
                active_sl = hard_sl

            # SHORT: SL hit if intra-bar high >= active_sl
            if hj >= active_sl:
                exit_price = active_sl
                exit_ts = ts_arr[j]
                exit_reason = "trail_break_above" if (hit_t1 and use_trail_break_high) else "stop"
                break

            # T1/T2 check via intra-bar low
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
        side = "SELL"  # SHORT entry
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

        trades.append({
            "T0_signal_date": ev["d"],
            "symbol": "NSE:" + ev["sym"],
            "bare_symbol": ev["sym"],
            "cap_segment": ev["cap"],
            "direction": "SHORT",
            "bar_n_hhmm": ev["bar_n_hhmm"],
            "entry_ts": ev["entry_ts"],
            "entry_hhmm": ev["entry_hhmm"],
            "entry_price": entry,
            "bar_n_high": ev["bar_n_high"],
            "bar_n_low": ev["bar_n_low"],
            "bar_n_close": ev["bar_n_close"],
            "bar_n_range": ev["bar_n_range"],
            "bar_n1_high": bar_n1_high,
            "day_open": ev["day_open"],
            "day_high_sofar": ev["day_high_sofar"],
            "day_low_sofar": ev["day_low_sofar"],
            "open_high_diff_frac": ev["open_high_diff_frac"],
            "intraday_range_pct": ev["intraday_range_pct"],
            "first_hour_cum_vol": ev["first_hour_cum_vol"],
            "pdc": pdc,
            "gap_pct": gap_pct,
            "adv_20d_cr": adv_cr,
            "vol_20d_avg": vol_20d_avg,
            "vol_first_hour_ratio": vol_first_hour_ratio,
            "prior_session_return_pct": prior_ret_pct,
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
    cap_map: Dict[str, str],
    daily_ctx: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n[{label}] running month-by-month {start} .. {end} ...")
    all_trades: List[dict] = []
    for y, m in _months_between(start, end):
        bars = _load_one_month(y, m, allowed, cap_map, start, end)
        if bars.empty:
            print(f"  {y}-{m:02d}: SKIPPED (no bars)")
            continue
        n_bars = len(bars)
        trades_this_month = _scan_and_simulate(bars, daily_ctx)
        all_trades.extend(trades_this_month)
        print(f"  {y}-{m:02d}: bars={n_bars:>10,}  trades={len(trades_this_month):>5,}  cum={len(all_trades):>6,}")
        del bars
        gc.collect()
    print(f"\n[{label}] DONE - total trades: {len(all_trades):,}")
    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame(all_trades)


# =========================================================================
# Cell prep + aggregate metrics
# =========================================================================

def _bucket_cut(values: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(values, bins=edges, labels=labels, right=False, include_lowest=True)


def prep_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add ex-ante (pre-trade) cell dimensions only. No outcome leakage:
    no exit_reason, no hit_t1, no exit_price, no realized_pnl.

    Dims:
      cap_segment, hhmm_bucket, dow, month,
      R_pct_bucket, gap_pct_bucket, intraday_range_pct_bucket,
      vol_first_hour_bucket, prior_session_close_pct_bucket, adv_bucket
    """
    df = df.copy()
    d = pd.to_datetime(df["T0_signal_date"])
    df["_session_date"] = d.dt.date
    df["_month"] = d.dt.strftime("%Y-%m")
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")

    # hhmm bucket = the trigger bar's hhmm (already in trigger window)
    df["hhmm_bucket"] = df["bar_n_hhmm"].astype(str)

    # R% = stop_distance / entry_price * 100
    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_pct_bucket"] = _bucket_cut(
        r_pct,
        edges=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2%+"],
    )

    # gap_pct = (day_open vs pdc) — open vs prior close
    df["gap_pct_bucket"] = _bucket_cut(
        df["gap_pct"],
        edges=[-np.inf, -1.0, 0.0, 1.0, 3.0, np.inf],
        labels=["<-1%", "-1to0%", "0to+1%", "+1to+3%", "+3%+"],
    )

    # intraday range at trigger (day_high_sofar - day_low_sofar)/day_open*100
    df["intraday_range_pct_bucket"] = _bucket_cut(
        df["intraday_range_pct"],
        edges=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2%+"],
    )

    # vol first hour ratio (cum vol up to bar[N]) / expected_first_hour_vol
    df["vol_first_hour_bucket"] = _bucket_cut(
        df["vol_first_hour_ratio"],
        edges=[-np.inf, 0.5, 1.0, 1.5, np.inf],
        labels=["<0.5x", "0.5-1x", "1-1.5x", "1.5x+"],
    )

    # prior session daily return (yesterday's move, on session D-1)
    df["prior_session_close_pct_bucket"] = _bucket_cut(
        df["prior_session_return_pct"],
        edges=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=["<-1.5%", "-1.5to-0.5%", "-0.5to+0.5%", "+0.5to+1.5%", ">+1.5%"],
    )

    # adv (20d trailing avg traded value in Rs Cr)
    df["adv_bucket"] = _bucket_cut(
        df["adv_20d_cr"],
        edges=[-np.inf, 5.0, 15.0, 50.0, np.inf],
        labels=["<5Cr", "5-15Cr", "15-50Cr", "50Cr+"],
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
    "cap_segment",
    "hhmm_bucket",
    "dow",
    "month",
    "R_pct_bucket",
    "gap_pct_bucket",
    "intraday_range_pct_bucket",
    "vol_first_hour_bucket",
    "prior_session_close_pct_bucket",
    "adv_bucket",
]


def report_period(trades: pd.DataFrame, label: str,
                  cellmine: bool = True) -> Tuple[dict, pd.DataFrame, int]:
    print("\n" + "=" * 78)
    print(f"REPORT - open_eq_high_short - {label}")
    print("=" * 78)
    if trades.empty:
        print("  NO TRADES")
        return {}, pd.DataFrame(), 0

    n_total = len(trades)
    cap_dist = trades["cap_segment"].value_counts().to_dict()
    tb_dist = trades["bar_n_hhmm"].value_counts().to_dict()
    er_dist = trades["exit_reason"].value_counts().to_dict()
    print(f"\n  Funnel: events={n_total:,}")
    print(f"    by_cap: {cap_dist}")
    print(f"    by_trigger_hhmm: {tb_dist}")
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

    # 1D gap_pct breakdown — special focus per task: gap-flat days should
    # work better than gap-up days.
    print(f"\n  [gap-context] 1D gap_pct_bucket aggregate:")
    if "gap_pct_bucket" in cells_df.columns:
        for bucket, sub in cells_df.groupby("gap_pct_bucket", observed=True):
            if len(sub) == 0:
                continue
            m = _aggregate_metrics(sub)
            pf_s = f"{m['pf']:.2f}" if (not np.isnan(m['pf']) and m['pf'] != float('inf')) else "inf"
            print(f"    gap={bucket}  n={m['n']:>5,}  PF={pf_s:>5}  "
                  f"WR={m['wr']:>5.1f}%  Sh={m['sharpe']:>5.2f}  NET={m['net']:>11,.0f}")

    out_cells = _OUT_DIR / f"open_eq_high_cells_{label.lower()}.csv"
    cells.to_csv(out_cells, index=False)
    print(f"\n  wrote cell-mine CSV: {out_cells.relative_to(_REPO)}")

    if len(ship_eligible):
        out_ship = _OUT_DIR / f"open_eq_high_ship_cells_{label.lower()}.csv"
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

    print(">>> Open=High Sustained Weakness SHORT - PROPER 3-PERIOD GAUNTLET <<<")
    print(f"   Discovery: {DISCOVERY_START} .. {DISCOVERY_END}  ({len(_months_between(DISCOVERY_START, DISCOVERY_END))}mo)")
    print(f"   OOS:       {OOS_START} .. {OOS_END}  ({len(_months_between(OOS_START, OOS_END))}mo)")
    print(f"   Holdout:   {HOLDOUT_START} .. {HOLDOUT_END}  ({len(_months_between(HOLDOUT_START, HOLDOUT_END))}mo)")
    print(f"   Trigger: |day_open - day_high_sofar|/day_open <= {OPEN_HIGH_TOL_FRAC:.4f} "
          f"AND close<close_prev AND bar_N.hhmm IN {sorted(TRIGGER_HHMM_WINDOW)}")
    print(f"   Direction: SHORT @ open(bar[N+1])")
    print(f"   SL: day_high_sofar + {SL_BUFFER_FRAC:.0%} * bar[N].range")
    print(f"   T1={T1_R_MULT}R @ {T1_QTY_PCT:.0%} qty, T2={T2_R_MULT}R, trail-break-above bar[N+1].high "
          f"after T1, time_stop={TIME_STOP_HHMM}")

    # ---- Universe + daily context + preflight ----
    allowed, cap_map = build_universe()
    if not allowed:
        print("[ABORT] empty universe")
        return 2

    daily_ctx = build_daily_context(allowed)

    regime_preflight(DISCOVERY_START, DISCOVERY_END, "Discovery")
    regime_preflight(OOS_START, OOS_END, "OOS")
    regime_preflight(HOLDOUT_START, HOLDOUT_END, "Holdout")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- DISCOVERY ----
    print("\n" + "#" * 78)
    print("# DISCOVERY  (2023-01-01 .. 2024-12-31)")
    print("#" * 78)
    disc_trades_path = _OUT_DIR / "open_eq_high_trades_discovery.csv"

    if args.skip_discovery and disc_trades_path.exists():
        print(f"  [skip] loading existing trades from {disc_trades_path}")
        disc_trades = pd.read_csv(disc_trades_path)
    else:
        disc_trades = run_period_month_by_month(
            "Discovery", DISCOVERY_START, DISCOVERY_END, allowed, cap_map, daily_ctx)
        if not disc_trades.empty:
            disc_trades.to_csv(disc_trades_path, index=False)
            print(f"  wrote {disc_trades_path.relative_to(_REPO)} ({len(disc_trades):,} trades)")

    disc_agg, disc_ship, disc_cells_scanned = report_period(disc_trades, label="Discovery", cellmine=True)
    n_ship_disc = len(disc_ship)

    expected_fp = disc_cells_scanned * 0.05
    print(f"\n  [bonferroni] {disc_cells_scanned:,} cells x alpha=0.05 -> expected "
          f"{expected_fp:.0f} false positives at chance. ship_eligible={n_ship_disc}.")

    if n_ship_disc == 0:
        print("\n[VERDICT] Discovery has 0 ship-eligible cells. RETIRE - no OOS/Holdout run.")
        return 0

    # ---- OOS ----
    print("\n" + "#" * 78)
    print("# OOS  (2025-01-01 .. 2025-09-30)")
    print("#" * 78)
    oos_trades_path = _OUT_DIR / "open_eq_high_trades_oos.csv"
    if args.skip_oos and oos_trades_path.exists():
        oos_trades = pd.read_csv(oos_trades_path)
    else:
        oos_trades = run_period_month_by_month(
            "OOS", OOS_START, OOS_END, allowed, cap_map, daily_ctx)
        if not oos_trades.empty:
            oos_trades.to_csv(oos_trades_path, index=False)
            print(f"  wrote {oos_trades_path.relative_to(_REPO)} ({len(oos_trades):,} trades)")

    oos_agg, _, _ = report_period(oos_trades, label="OOS", cellmine=False)

    oos_validation = validate_cells_on_period(disc_ship, oos_trades, "OOS")
    if not oos_validation.empty:
        oos_val_path = _OUT_DIR / "open_eq_high_oos_validation.csv"
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
    hold_trades_path = _OUT_DIR / "open_eq_high_trades_holdout.csv"
    if args.skip_holdout and hold_trades_path.exists():
        hold_trades = pd.read_csv(hold_trades_path)
    else:
        hold_trades = run_period_month_by_month(
            "Holdout", HOLDOUT_START, HOLDOUT_END, allowed, cap_map, daily_ctx)
        if not hold_trades.empty:
            hold_trades.to_csv(hold_trades_path, index=False)
            print(f"  wrote {hold_trades_path.relative_to(_REPO)} ({len(hold_trades):,} trades)")

    hold_agg, _, _ = report_period(hold_trades, label="Holdout", cellmine=False)

    survivors_oos = oos_validation[oos_validation["verdict"] == "SHIP"]
    surv_cells_df = disc_ship[disc_ship["cell"].isin(survivors_oos["cell"])]
    hold_validation = validate_cells_on_period(surv_cells_df, hold_trades, "Holdout")
    if not hold_validation.empty:
        hold_val_path = _OUT_DIR / "open_eq_high_holdout_validation.csv"
        hold_validation.to_csv(hold_val_path, index=False)
        print(f"\n  wrote Holdout validation: {hold_val_path.relative_to(_REPO)}")

    hold_ship = hold_validation[hold_validation["verdict"] == "SHIP"]
    print(f"\n  CELLS THAT PASSED DISC -> OOS -> HOLDOUT: {len(hold_ship)}/{len(oos_ship)}")

    if len(hold_ship) == 0:
        print("\n[VERDICT] No cells survived Holdout. RETIRE.")
        return 0

    print("\n" + "#" * 78)
    print("# FINAL EVIDENCE DUMP - triple-survivor cells")
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

    print(f"\n[VERDICT] SHIP CANDIDATE - triple-period survival on {len(hold_ship)} cell(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
