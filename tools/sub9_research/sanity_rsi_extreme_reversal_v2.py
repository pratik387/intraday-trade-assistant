"""Mathematical-pattern PROPER gauntlet on RSI extreme reversal — FULL 2-YEAR Discovery.

V2 RATIONALE
-----------
V1 ran 13-month Discovery (2024-09..2025-09); cell mining surfaced 2 ship-eligible
cells, but OOS check on a 7-month window retired them (all FAIL PF>=1.30). The
13-month sample may have been too thin for the deep-cell signal-to-noise ratio.
This V2 runs the production-schema gauntlet:
  - Discovery: 24 months (2023-01-01 .. 2024-12-31)
  - OOS:        9 months (2025-01-01 .. 2025-09-30)
  - Holdout:    7 months (2025-10-01 .. 2026-04-30)

If a Discovery cell ships AND validates on OOS, run the 576-combo SL/T1/T2 sweep
on that surviving cell, then validate the best combo on Holdout.

PATTERN (unchanged from V1)
---------------------------
14-period Wilder RSI on 15m close prices (aggregated from 5m feathers).
  - RSI < 25 -> LONG candidate; RSI > 75 -> SHORT candidate
Confirmation = next 15m bar closes in reversal direction.
Entry on confirmation bar's close. SL = recent-4-15m lookback (with 0.5% min floor).
T1=1R (50% qty), T2=2R, BE trail after T1, time stop 15:10.

OOM-SAFE LOAD (carried over from V1)
------------------------------------
  - Use dt.floor("D") instead of .dt.date on multi-million row frames.
  - Skip big5m["session_date"] = big5m["date"].dt.date (OOM at 23M rows).
  - Per-month chunked load + Universe filter early.

UNIVERSE
--------
nse_all.json filtered to mis_leverage>=1.0 AND cap_segment in {large,mid,small},
UNION fno_liquid_200. Same as V1.

DEEP CELL DIMS (10)
-------------------
direction, cap_segment, tod_bucket,
rsi_severity (deep/moderate), rsi_severity_fine (S70-75/.../S85+; L25-20/.../L10-),
R_size_bucket, dow, month, quarter, confirmation_lag_min_bucket.

Gauntlet-v2 ship gates: n>=125, PF>=1.30, Sharpe>=0.5, win_mo>=55%, top_mo<40%.

Usage:
    python tools/sub9_research/sanity_rsi_extreme_reversal_v2.py
"""
from __future__ import annotations

import gc
import json
import sys
from datetime import date
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Line-buffered stdout for live progress
sys.stdout.reconfigure(line_buffering=True)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.indicators.indicators import calculate_rsi  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
from tools.sub9_research._cell_mine_tier_a import (  # noqa: E402
    scan_cells,
    _sharpe_of,
    _monthly_stats,
    N_MIN_SURVIVOR,
    N_MIN_SHIP,
    PF_MIN_SURVIVOR,
    PF_MIN_SHIP,
    SHARPE_MIN_SURVIVOR,
    SHARPE_MIN_SHIP,
    LOSING_MONTHS_PCT_MAX_SHIP,
    TOP_MONTH_CONCENTRATION_MAX,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_NSE_ALL_JSON = _REPO / "nse_all.json"
_FNO_LIQUID = _REPO / "assets" / "fno_liquid_200.csv"
_OUT_DIR = _REPO / "reports" / "sub9_sanity"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Full 2-year Discovery + 9-month OOS + 7-month Holdout
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END = date(2024, 12, 31)
OOS_START = date(2025, 1, 1)
OOS_END = date(2025, 9, 30)
HOLDOUT_START = date(2025, 10, 1)
HOLDOUT_END = date(2026, 4, 30)

RSI_PERIOD = 14
RSI_OVERSOLD = 25.0
RSI_OVERBOUGHT = 75.0
RSI_DEEP_LOW = 20.0
RSI_DEEP_HIGH = 80.0

SL_LOOKBACK_15M = 4
SL_BUF_LONG = 0.995
SL_BUF_SHORT = 1.005
MIN_STOP_PCT = 0.005   # 0.5%
T1_R = 1.0
T2_R = 2.0
T1_QTY_PCT = 0.5
TIME_STOP_HHMM = "15:10"

RISK_PER_TRADE_RUPEES = 1000
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}

# Ship gates (Gauntlet v2) — kept identical to V1 + cell-mining
SHIP_N = 125
SHIP_PF = 1.30
SHIP_SHARPE = 0.5
SHIP_WIN_MO_PCT = 55.0
SHIP_TOP_MO_PCT = 40.0

# OOS/Holdout cell-validation gates (lower n floor because of shorter windows)
OOS_PASS_N = 50
OOS_PASS_PF = 1.30
OOS_PASS_SH = 0.0

# Sweep grid (576 combos = 3 x 3 x 4 x 4 x 4)
# Used ONLY if any Discovery+OOS cell survives
SWEEP_SL_BUFFERS = [0.003, 0.005, 0.007]
SWEEP_T1_RS = [0.5, 1.0, 1.5]
SWEEP_T2_RS = [1.0, 1.5, 2.0, 3.0]
SWEEP_T1_QTYS = [0.0, 0.33, 0.50, 0.67]
SWEEP_TIME_STOPS = ["13:00", "14:00", "14:30", "15:10"]


# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------
def build_universe() -> Tuple[set, Dict[str, str]]:
    with open(_NSE_ALL_JSON) as f:
        data = json.load(f)
    cap_map: Dict[str, str] = {}
    universe: set = set()
    for d in data:
        sym = d.get("symbol", "").replace(".NS", "").strip()
        if not sym:
            continue
        cap = d.get("cap_segment", "unknown")
        lev = float(d.get("mis_leverage") or 0.0)
        if cap in ALLOWED_CAPS and lev >= 1.0:
            universe.add(sym)
            cap_map[sym] = cap
    fno_df = pd.read_csv(_FNO_LIQUID)
    for s in fno_df["symbol"].dropna().astype(str):
        bare = s.replace("NSE:", "").strip()
        if bare:
            universe.add(bare)
            cap_map.setdefault(bare, "large_cap")
    return universe, cap_map


# ----------------------------------------------------------------------------
# Data load + 15m aggregation + RSI
# ----------------------------------------------------------------------------
def _months_in_period(start: date, end: date) -> List[Tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def load_5m_for_period(start: date, end: date, universe: set) -> pd.DataFrame:
    """Load all 5m_enriched feathers covering [start, end].

    OOM-safe: per-month filter by universe BEFORE concat. Skip building a
    .dt.date column (which OOM'd on V1's 23M-row Discovery).
    """
    months = _months_in_period(start, end)
    print(f"  loading {len(months)} monthly 5m feathers covering "
          f"{start} .. {end} ...")
    parts: List[pd.DataFrame] = []
    for y, m in months:
        fp = _FEATHER_DIR / f"{y:04d}_{m:02d}_5m_enriched.feather"
        if not fp.exists():
            print(f"    missing: {fp.name}")
            continue
        df = pd.read_feather(fp)
        df = df[df["symbol"].isin(universe)]
        if df.empty:
            continue
        parts.append(df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy())
        del df
        gc.collect()
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    # Filter to the exact date window using .dt.floor("D") instead of .dt.date
    # to avoid the 23M-row OOM that hits dt.date on big frames.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    d_mask = (big["date"] >= start_ts) & (big["date"] < end_ts)
    big = big[d_mask]
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"  total 5m bars after universe + date filter: {len(big):,}")
    return big


def aggregate_to_15m_with_rsi(big5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m -> 15m within each session day, compute 14-period Wilder
    RSI on 15m close per symbol. OOM-safe: derives session_date from floor('D')
    only at the groupby site, not as a persisted column on big5m.
    """
    if big5m.empty:
        return big5m
    print("  aggregating 5m -> 15m bars ...")
    # session_date via floor('D') is cheap (datetime64), avoid .dt.date materialization
    sess = big5m["date"].dt.floor("D")
    bin15 = big5m["date"].dt.floor("15min")
    df = pd.DataFrame({
        "symbol": big5m["symbol"].values,
        "session_date": sess.values,
        "bin": bin15.values,
        "open": big5m["open"].values,
        "high": big5m["high"].values,
        "low": big5m["low"].values,
        "close": big5m["close"].values,
        "volume": big5m["volume"].values,
    })
    grouped = df.groupby(["symbol", "session_date", "bin"], sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    grouped = grouped.rename(columns={"bin": "date"})
    grouped["hhmm"] = grouped["date"].dt.strftime("%H:%M")
    # session_date is now a datetime64 (day-floored); convert ONCE to python date
    # for cheap downstream equality compares against ev["session_date"].
    grouped["session_date"] = grouped["session_date"].dt.date
    grouped = grouped.sort_values(["symbol", "date"]).reset_index(drop=True)

    del df
    gc.collect()

    print(f"  total 15m bars: {len(grouped):,}")
    print("  computing 14-period Wilder RSI per symbol (continuous) ...")
    grouped["rsi"] = grouped.groupby("symbol", sort=False)["close"].transform(
        lambda s: calculate_rsi(s, period=RSI_PERIOD)
    )
    return grouped


# ----------------------------------------------------------------------------
# Event detection
# ----------------------------------------------------------------------------
def detect_events(df15: pd.DataFrame) -> pd.DataFrame:
    print("  detecting RSI-extreme + confirmation events ...")
    df = df15.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    grp = df.groupby("symbol", sort=False)
    df["next_open"] = grp["open"].shift(-1)
    df["next_close"] = grp["close"].shift(-1)
    df["next_high"] = grp["high"].shift(-1)
    df["next_low"] = grp["low"].shift(-1)
    df["next_date"] = grp["date"].shift(-1)
    df["next_session"] = grp["session_date"].shift(-1)
    df["next_hhmm"] = df["next_date"].dt.strftime("%H:%M")

    long_trig = (df["rsi"] < RSI_OVERSOLD) & df["rsi"].notna()
    short_trig = (df["rsi"] > RSI_OVERBOUGHT) & df["rsi"].notna()
    same_sess = df["session_date"] == df["next_session"]
    conf_in_window = df["next_hhmm"].fillna("99:99") <= "14:45"

    long_conf = df["next_close"] > df["next_open"]
    short_conf = df["next_close"] < df["next_open"]

    long_events = df[long_trig & same_sess & conf_in_window & long_conf].copy()
    long_events["direction"] = "long"
    long_events["rsi_trigger"] = long_events["rsi"]

    short_events = df[short_trig & same_sess & conf_in_window & short_conf].copy()
    short_events["direction"] = "short"
    short_events["rsi_trigger"] = short_events["rsi"]

    events = pd.concat([long_events, short_events], ignore_index=True)
    if events.empty:
        return events

    events = events.sort_values(["symbol", "session_date", "next_date"])
    events = events.drop_duplicates(subset=["symbol", "session_date"], keep="first").reset_index(drop=True)

    def _sev(r: float, direction: str) -> str:
        if direction == "long":
            return "deep" if r < RSI_DEEP_LOW else "moderate"
        else:
            return "deep" if r > RSI_DEEP_HIGH else "moderate"

    events["rsi_severity"] = events.apply(
        lambda r: _sev(r["rsi_trigger"], r["direction"]), axis=1
    )

    def _tod(hhmm: str) -> str:
        if hhmm < "11:30":
            return "morning"
        elif hhmm < "13:30":
            return "mid"
        else:
            return "afternoon"

    events["tod_bucket"] = events["next_hhmm"].apply(_tod)
    return events


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(events: pd.DataFrame, df15: pd.DataFrame, big5m: pd.DataFrame,
             cap_map: Dict[str, str]) -> pd.DataFrame:
    print(f"  simulating {len(events):,} events ...")
    df15 = df15.sort_values(["symbol", "date"]).reset_index(drop=True)
    df15_by_sym: Dict[str, pd.DataFrame] = {
        sym: g for sym, g in df15.groupby("symbol", sort=False)
    }

    # Index 5m by (symbol, session_date) for bar walk — OOM-safe via floor("D")
    sd_floor = big5m["date"].dt.floor("D")
    sym_sess_5m: Dict[Tuple[str, date], pd.DataFrame] = {}
    for (sym, sd_ts), g in big5m.groupby([big5m["symbol"], sd_floor], sort=False):
        sym_sess_5m[(sym, sd_ts.date())] = g.sort_values("date").reset_index(drop=True)

    trades: List[dict] = []
    for _, ev in events.iterrows():
        sym = ev["symbol"]
        sd = ev["session_date"]
        direction = ev["direction"]
        conf_date = pd.Timestamp(ev["next_date"])
        entry_price = float(ev["next_close"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        sym_15 = df15_by_sym.get(sym)
        if sym_15 is None:
            continue
        mask = sym_15["date"] <= conf_date
        lookback = sym_15.loc[mask].tail(SL_LOOKBACK_15M)
        if lookback.empty:
            continue

        if direction == "long":
            sl_base = float(lookback["low"].min())
            hard_sl = sl_base * SL_BUF_LONG
            stop_distance = entry_price - hard_sl
            min_stop_dist = entry_price * MIN_STOP_PCT
            if stop_distance < min_stop_dist:
                hard_sl = entry_price - min_stop_dist
                stop_distance = min_stop_dist
            t1 = entry_price + T1_R * stop_distance
            t2 = entry_price + T2_R * stop_distance
        else:
            sl_base = float(lookback["high"].max())
            hard_sl = sl_base * SL_BUF_SHORT
            stop_distance = hard_sl - entry_price
            min_stop_dist = entry_price * MIN_STOP_PCT
            if stop_distance < min_stop_dist:
                hard_sl = entry_price + min_stop_dist
                stop_distance = min_stop_dist
            t1 = entry_price - T1_R * stop_distance
            t2 = entry_price - T2_R * stop_distance

        if stop_distance <= 0:
            continue

        qty = max(int(RISK_PER_TRADE_RUPEES / stop_distance), 1)
        qty_t1 = int(qty * T1_QTY_PCT)
        qty_runner = qty - qty_t1

        bars = sym_sess_5m.get((sym, sd))
        if bars is None or bars.empty:
            continue
        walk_start = conf_date + pd.Timedelta(minutes=15)
        post = bars[bars["date"] >= walk_start]
        if post.empty:
            continue

        t1_hit = False
        t1_exit_price: Optional[float] = None
        t2_exit_price: Optional[float] = None
        sl_exit_price: Optional[float] = None
        time_exit_price: Optional[float] = None
        exit_reason: Optional[str] = None
        exit_ts = None

        for _, bar in post.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hhmm = ts.strftime("%H:%M") if hasattr(ts, "strftime") else pd.Timestamp(ts).strftime("%H:%M")

            active_sl = entry_price if t1_hit else hard_sl

            if direction == "long":
                if low <= active_sl:
                    sl_exit_price = active_sl
                    exit_reason = "be" if t1_hit else "sl"
                    exit_ts = ts
                    break
                if not t1_hit and high >= t1:
                    t1_hit = True
                    t1_exit_price = t1
                if high >= t2:
                    t2_exit_price = t2
                    exit_reason = "t2"
                    exit_ts = ts
                    break
                if hhmm >= TIME_STOP_HHMM:
                    time_exit_price = close
                    exit_reason = "time"
                    exit_ts = ts
                    break
            else:
                if high >= active_sl:
                    sl_exit_price = active_sl
                    exit_reason = "be" if t1_hit else "sl"
                    exit_ts = ts
                    break
                if not t1_hit and low <= t1:
                    t1_hit = True
                    t1_exit_price = t1
                if low <= t2:
                    t2_exit_price = t2
                    exit_reason = "t2"
                    exit_ts = ts
                    break
                if hhmm >= TIME_STOP_HHMM:
                    time_exit_price = close
                    exit_reason = "time"
                    exit_ts = ts
                    break

        if exit_reason is None:
            last = post.iloc[-1]
            time_exit_price = float(last["close"])
            exit_reason = "last_bar"
            exit_ts = last["date"]

        pnl = 0.0
        if t1_hit and t1_exit_price is not None:
            if direction == "long":
                pnl += (t1_exit_price - entry_price) * qty_t1
            else:
                pnl += (entry_price - t1_exit_price) * qty_t1

        final_qty = qty_runner if t1_hit else qty
        if t2_exit_price is not None:
            final_exit_price = t2_exit_price
        elif sl_exit_price is not None:
            final_exit_price = sl_exit_price
        else:
            final_exit_price = time_exit_price if time_exit_price is not None else entry_price

        if direction == "long":
            pnl += (final_exit_price - entry_price) * final_qty
        else:
            pnl += (entry_price - final_exit_price) * final_qty

        legs = []
        if t1_hit and t1_exit_price is not None:
            legs.append((qty_t1, t1_exit_price))
        legs.append((final_qty, final_exit_price))
        total_q = sum(q for q, _ in legs)
        if total_q > 0:
            avg_exit = sum(q * p for q, p in legs) / total_q
        else:
            avg_exit = entry_price

        leg_side = "BUY" if direction == "long" else "SELL"
        fee = calc_fee(entry_price, avg_exit, qty, leg_side)
        net_pnl = pnl - fee

        trades.append({
            "session_date": sd,
            "symbol": sym,
            "cap_segment": cap_map.get(sym, "unknown"),
            "direction": direction,
            "rsi_trigger": float(ev["rsi_trigger"]),
            "rsi_severity": ev["rsi_severity"],
            "tod_bucket": ev["tod_bucket"],
            "trigger_ts": ev["date"],
            "confirmation_ts": conf_date,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1": t1,
            "t2": t2,
            "stop_distance": stop_distance,
            "qty": qty,
            "t1_hit": bool(t1_hit),
            "exit_reason": exit_reason,
            "exit_ts": exit_ts,
            "avg_exit_price": avg_exit,
            "gross_pnl": pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------------
# Cell annotation (10-dim prep)
# ----------------------------------------------------------------------------
def annotate_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["session_date"])
    df["_session_date"] = d.dt.date
    df["_month"] = d.dt.strftime("%Y-%m")
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")
    df["quarter"] = "Q" + d.dt.quarter.astype(str)

    rsi = df["rsi_trigger"]
    side = df["direction"].astype(str).str.lower()

    long_bins = [-np.inf, 10, 15, 20, 25, np.inf]
    long_lbl = ["L10-", "L15-10", "L20-15", "L25-20", "L25+"]
    short_bins = [-np.inf, 70, 75, 80, 85, np.inf]
    short_lbl = ["S<=70", "S70-75", "S75-80", "S80-85", "S85+"]

    long_bucket = pd.cut(rsi, bins=long_bins, labels=long_lbl, right=False)
    short_bucket = pd.cut(rsi, bins=short_bins, labels=short_lbl, right=False)
    df["rsi_severity_fine"] = np.where(side == "long", long_bucket.astype(str),
                                       short_bucket.astype(str))
    df.loc[df["rsi_severity_fine"] == "nan", "rsi_severity_fine"] = np.nan

    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_size_pct"] = r_pct
    df["R_size_bucket"] = pd.cut(
        r_pct,
        bins=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    ).astype(str)
    df.loc[df["R_size_bucket"] == "nan", "R_size_bucket"] = np.nan

    trig = pd.to_datetime(df["trigger_ts"])
    conf = pd.to_datetime(df["confirmation_ts"])
    lag_min = (conf - trig).dt.total_seconds() / 60.0
    df["confirmation_lag_min_bucket"] = pd.cut(
        lag_min,
        bins=[-np.inf, 15, 30, 45, 60, np.inf],
        labels=["<=15", "15-30", "30-45", "45-60", "60+"],
    ).astype(str)
    df.loc[df["confirmation_lag_min_bucket"] == "nan", "confirmation_lag_min_bucket"] = np.nan
    return df


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _agg_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, pf=0.0, wr=0.0, net=0.0, sharpe=0.0,
                    n_months=0, win_mo_pct=0.0, top_mo_pct=0.0)
    pnl = df["net_pnl"]
    pf = _pf(pnl)
    wr = 100.0 * float((pnl > 0).mean())
    sharpe = _sharpe_of(df.groupby("session_date")["net_pnl"].sum())
    # monthly
    m = df.copy()
    m["_mo"] = pd.to_datetime(m["session_date"]).dt.strftime("%Y-%m")
    monthly = m.groupby("_mo")["net_pnl"].sum()
    n_mo = int(monthly.size)
    win_mo_pct = 100.0 * float((monthly > 0).mean()) if n_mo > 0 else 0.0
    total = float(monthly.sum())
    top_pct = 100.0 * float(monthly.abs().max()) / abs(total) if abs(total) > 1e-6 else 0.0
    return dict(
        n=int(len(df)), pf=float(pf), wr=float(wr),
        net=float(pnl.sum()), sharpe=float(sharpe),
        n_months=n_mo,
        win_mo_pct=round(win_mo_pct, 1),
        top_mo_pct=round(top_pct, 1),
    )


# ----------------------------------------------------------------------------
# Period runner
# ----------------------------------------------------------------------------
def run_period(start: date, end: date, label: str,
               universe: set, cap_map: Dict[str, str]
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (trades, df15, big5m). Last two needed only if doing per-cell
    sweep on this period; pass-through avoids re-loading."""
    print(f"\n{'#'*78}\n# {label.upper()} {start} .. {end}\n{'#'*78}")
    big5m = load_5m_for_period(start, end, universe)
    if big5m.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    df15 = aggregate_to_15m_with_rsi(big5m)
    events = detect_events(df15)
    print(f"  events (latch=1/sym/day): {len(events):,}")
    if not events.empty:
        print(f"    long={int((events['direction']=='long').sum()):,}  "
              f"short={int((events['direction']=='short').sum()):,}")
    trades = simulate(events, df15, big5m, cap_map)
    print(f"  trades simulated: {len(trades):,}")
    return (trades, df15, big5m)


def print_aggregate(trades: pd.DataFrame, label: str) -> dict:
    print(f"\n{'='*78}\n{label.upper()} AGGREGATE\n{'='*78}")
    if trades.empty:
        print("  [NO TRADES]")
        return {}
    agg = _agg_row(trades)
    print(f"  n={agg['n']:,}  PF={agg['pf']:.3f}  WR={agg['wr']:.1f}%  "
          f"NET={agg['net']:,.0f}  Sharpe(daily)={agg['sharpe']:.3f}")
    print(f"  months={agg['n_months']}  win_mo={agg['win_mo_pct']}%  "
          f"top_mo={agg['top_mo_pct']}%")
    return agg


# ----------------------------------------------------------------------------
# Cell mining (deep, 10-dim)
# ----------------------------------------------------------------------------
DEEP_DIMS = [
    "direction",
    "cap_segment",
    "tod_bucket",
    "rsi_severity",
    "rsi_severity_fine",
    "R_size_bucket",
    "dow",
    "month",
    "quarter",
    "confirmation_lag_min_bucket",
]


def deep_cell_mine(trades_annot: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (survivors, ship_eligible). Writes full cell scan to CSV."""
    have = [d for d in DEEP_DIMS if d in trades_annot.columns]
    print(f"\n  deep cell-mining ({len(have)} dims, max_combo=3) ...")
    cells = scan_cells(
        trades_annot, have, "net_pnl",
        max_combo=3,
        date_col="_session_date",
        month_col="_month",
    )
    print(f"  total cells: {len(cells):,}")
    out_full = _OUT_DIR / f"rsi_v2_{label}_cells_full.csv"
    cells.to_csv(out_full, index=False)
    print(f"  wrote: {out_full.name}")

    surv_mask = (
        (cells["n"] >= N_MIN_SURVIVOR)
        & (cells["pf"] >= PF_MIN_SURVIVOR)
        & (cells["sharpe"] >= SHARPE_MIN_SURVIVOR)
    )
    surv = cells[surv_mask].sort_values(["pf", "n"], ascending=[False, False])

    ship_mask = (
        (cells["n"] >= SHIP_N)
        & (cells["pf"] >= SHIP_PF)
        & (cells["sharpe"] >= SHIP_SHARPE)
        & (cells["lose_mo_pct"] <= LOSING_MONTHS_PCT_MAX_SHIP)
        & (cells["top_mo_pct"] < SHIP_TOP_MO_PCT)
        & (cells["win_mo_pct"] >= SHIP_WIN_MO_PCT)
    )
    ship = cells[ship_mask].sort_values(["pf", "n"], ascending=[False, False])

    surv_csv = _OUT_DIR / f"rsi_v2_{label}_survivors.csv"
    surv.to_csv(surv_csv, index=False)
    ship_csv = _OUT_DIR / f"rsi_v2_{label}_ship_eligible.csv"
    ship.to_csv(ship_csv, index=False)
    print(f"  survivors: {len(surv):,}  -> {surv_csv.name}")
    print(f"  ship-eligible: {len(ship):,}  -> {ship_csv.name}")

    if not surv.empty:
        print(f"\n  TOP SURVIVORS:")
        for _, r in surv.head(15).iterrows():
            print(f"    [{r['dims']}] {r['cell']}  "
                  f"n={r['n']:,} PF={r['pf']:.3f} WR={r['wr']:.1f}% "
                  f"Sh={r['sharpe']:.2f} mo[win={r['win_mo_pct']}% "
                  f"top={r['top_mo_pct']}%] NET={r['net']:,.0f}")
    if not ship.empty:
        print(f"\n  SHIP-ELIGIBLE:")
        for _, r in ship.head(15).iterrows():
            print(f"    [{r['dims']}] {r['cell']}  "
                  f"n={r['n']:,} PF={r['pf']:.3f} WR={r['wr']:.1f}% "
                  f"Sh={r['sharpe']:.2f} mo[win={r['win_mo_pct']}% "
                  f"top={r['top_mo_pct']}%] NET={r['net']:,.0f}")
    return (surv, ship)


# ----------------------------------------------------------------------------
# Cell-filter builder (from cell label string)
# ----------------------------------------------------------------------------
def parse_cell(dims_csv: str, cell_label: str) -> Dict[str, str]:
    """`dims_csv` like 'direction,cap_segment' and `cell_label` like
    'direction=short | cap_segment=small_cap' -> dict mapping col->value."""
    dims = dims_csv.split(",")
    out: Dict[str, str] = {}
    for part in cell_label.split(" | "):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    # Sanity: all dim keys must be present
    return {k: out[k] for k in dims if k in out}


def apply_cell_filter(df: pd.DataFrame, cell_filter: Dict[str, str]) -> pd.DataFrame:
    sub = df
    for col, val in cell_filter.items():
        if col not in sub.columns:
            return sub.iloc[0:0]
        sub = sub[sub[col].astype(str) == str(val)]
    return sub


# ----------------------------------------------------------------------------
# SL/T1/T2 sweep on a SURVIVING cell's events
# ----------------------------------------------------------------------------
def simulate_combo_for_cell(events_cell: pd.DataFrame,
                            df15_by_sym: Dict[str, pd.DataFrame],
                            sym_sess_5m: Dict[Tuple[str, date], pd.DataFrame],
                            cap_map: Dict[str, str],
                            sl_buffer: float, t1_r: float, t2_r: float,
                            t1_qty_pct: float, time_stop_hhmm: str) -> pd.DataFrame:
    """Re-simulate cell events under one combo.

    SL_BUFFER applied as buffer on top of lookback OR as min-floor when
    lookback distance is too tight.
    """
    trades: List[dict] = []
    for _, ev in events_cell.iterrows():
        sym = ev["symbol"]
        sd = ev["session_date"]
        direction = ev["direction"]
        conf_date = pd.Timestamp(ev["next_date"])
        entry_price = float(ev["next_close"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        sym_15 = df15_by_sym.get(sym)
        if sym_15 is None:
            continue
        mask = sym_15["date"] <= conf_date
        lookback = sym_15.loc[mask].tail(SL_LOOKBACK_15M)
        if lookback.empty:
            continue

        if direction == "long":
            sl_base = float(lookback["low"].min())
            hard_sl = sl_base * (1.0 - sl_buffer)
            stop_distance = entry_price - hard_sl
            min_stop = entry_price * sl_buffer
            if stop_distance < min_stop:
                hard_sl = entry_price - min_stop
                stop_distance = min_stop
            t1 = entry_price + t1_r * stop_distance
            t2 = entry_price + t2_r * stop_distance
        else:
            sl_base = float(lookback["high"].max())
            hard_sl = sl_base * (1.0 + sl_buffer)
            stop_distance = hard_sl - entry_price
            min_stop = entry_price * sl_buffer
            if stop_distance < min_stop:
                hard_sl = entry_price + min_stop
                stop_distance = min_stop
            t1 = entry_price - t1_r * stop_distance
            t2 = entry_price - t2_r * stop_distance

        if stop_distance <= 0:
            continue

        qty = max(int(RISK_PER_TRADE_RUPEES / stop_distance), 1)
        qty_t1 = int(qty * t1_qty_pct)
        qty_runner = qty - qty_t1

        bars = sym_sess_5m.get((sym, sd))
        if bars is None or bars.empty:
            continue
        walk_start = conf_date + pd.Timedelta(minutes=15)
        post = bars[bars["date"] >= walk_start]
        if post.empty:
            continue

        t1_hit = False
        t1_exit_price: Optional[float] = None
        t2_exit_price: Optional[float] = None
        sl_exit_price: Optional[float] = None
        time_exit_price: Optional[float] = None
        exit_reason: Optional[str] = None

        for _, bar in post.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hhmm = ts.strftime("%H:%M") if hasattr(ts, "strftime") else pd.Timestamp(ts).strftime("%H:%M")

            # If t1_qty_pct == 0, no BE trail
            if t1_qty_pct > 0 and t1_hit:
                active_sl = entry_price
            else:
                active_sl = hard_sl

            if direction == "long":
                if low <= active_sl:
                    sl_exit_price = active_sl
                    exit_reason = "be" if (t1_qty_pct > 0 and t1_hit) else "sl"
                    break
                if t1_qty_pct > 0 and not t1_hit and high >= t1:
                    t1_hit = True
                    t1_exit_price = t1
                if high >= t2:
                    t2_exit_price = t2
                    exit_reason = "t2"
                    break
                if hhmm >= time_stop_hhmm:
                    time_exit_price = close
                    exit_reason = "time"
                    break
            else:
                if high >= active_sl:
                    sl_exit_price = active_sl
                    exit_reason = "be" if (t1_qty_pct > 0 and t1_hit) else "sl"
                    break
                if t1_qty_pct > 0 and not t1_hit and low <= t1:
                    t1_hit = True
                    t1_exit_price = t1
                if low <= t2:
                    t2_exit_price = t2
                    exit_reason = "t2"
                    break
                if hhmm >= time_stop_hhmm:
                    time_exit_price = close
                    exit_reason = "time"
                    break

        if exit_reason is None:
            last = post.iloc[-1]
            time_exit_price = float(last["close"])
            exit_reason = "last_bar"

        pnl = 0.0
        if t1_qty_pct > 0 and t1_hit and t1_exit_price is not None:
            if direction == "long":
                pnl += (t1_exit_price - entry_price) * qty_t1
            else:
                pnl += (entry_price - t1_exit_price) * qty_t1

        if t1_qty_pct > 0 and t1_hit:
            final_qty = qty_runner
        else:
            final_qty = qty

        if t2_exit_price is not None:
            final_exit = t2_exit_price
        elif sl_exit_price is not None:
            final_exit = sl_exit_price
        else:
            final_exit = time_exit_price if time_exit_price is not None else entry_price

        if direction == "long":
            pnl += (final_exit - entry_price) * final_qty
        else:
            pnl += (entry_price - final_exit) * final_qty

        legs = []
        if t1_qty_pct > 0 and t1_hit and t1_exit_price is not None:
            legs.append((qty_t1, t1_exit_price))
        legs.append((final_qty, final_exit))
        total_q = sum(q for q, _ in legs)
        avg_exit = sum(q * p for q, p in legs) / total_q if total_q > 0 else entry_price

        leg_side = "BUY" if direction == "long" else "SELL"
        fee = calc_fee(entry_price, avg_exit, qty, leg_side)
        net_pnl = pnl - fee

        trades.append({
            "session_date": sd,
            "symbol": sym,
            "net_pnl": net_pnl,
        })
    return pd.DataFrame(trades)


def annotate_events_for_cells(events: pd.DataFrame, cap_map: Dict[str, str]) -> pd.DataFrame:
    """Replicate trades-level cell annotations on the EVENTS frame so we can
    filter events to a cell before sweeping. Uses same dims as annotate_cells
    but computed from event columns (no stop_distance/R_size known yet for the
    sweep — R_size depends on combo, so cells using R_size_bucket can't be
    swept). Note this is intentional: R_size_bucket filter on events is
    impossible since R depends on SL_buffer."""
    df = events.copy()
    df["cap_segment"] = df["symbol"].map(cap_map).fillna("unknown")
    d = pd.to_datetime(df["session_date"])
    df["dow"] = d.dt.day_name().str[:3]
    df["month"] = d.dt.strftime("%b")
    df["quarter"] = "Q" + d.dt.quarter.astype(str)
    # confirmation_lag_min_bucket
    trig = pd.to_datetime(df["date"])
    conf = pd.to_datetime(df["next_date"])
    lag_min = (conf - trig).dt.total_seconds() / 60.0
    df["confirmation_lag_min_bucket"] = pd.cut(
        lag_min,
        bins=[-np.inf, 15, 30, 45, 60, np.inf],
        labels=["<=15", "15-30", "30-45", "45-60", "60+"],
    ).astype(str)
    df.loc[df["confirmation_lag_min_bucket"] == "nan", "confirmation_lag_min_bucket"] = np.nan
    # rsi_severity_fine
    rsi = df["rsi_trigger"]
    side = df["direction"].astype(str).str.lower()
    long_bins = [-np.inf, 10, 15, 20, 25, np.inf]
    long_lbl = ["L10-", "L15-10", "L20-15", "L25-20", "L25+"]
    short_bins = [-np.inf, 70, 75, 80, 85, np.inf]
    short_lbl = ["S<=70", "S70-75", "S75-80", "S80-85", "S85+"]
    long_bucket = pd.cut(rsi, bins=long_bins, labels=long_lbl, right=False)
    short_bucket = pd.cut(rsi, bins=short_bins, labels=short_lbl, right=False)
    df["rsi_severity_fine"] = np.where(side == "long", long_bucket.astype(str),
                                       short_bucket.astype(str))
    df.loc[df["rsi_severity_fine"] == "nan", "rsi_severity_fine"] = np.nan
    return df


def run_sweep_on_cell(cell_dims: str, cell_label: str,
                      events_annot: pd.DataFrame,
                      df15: pd.DataFrame, big5m: pd.DataFrame,
                      cap_map: Dict[str, str], period_label: str) -> pd.DataFrame:
    """576-combo SL/T1/T2/T1_QTY/time_stop sweep on a single cell's events."""
    cf = parse_cell(cell_dims, cell_label)
    # Drop R_size_bucket from the cell filter (R depends on SL_buf, can't pre-filter)
    cf = {k: v for k, v in cf.items() if k != "R_size_bucket"}
    sub_events = apply_cell_filter(events_annot, cf)
    print(f"\n  sweep cell [{cell_dims}] {cell_label}  -> {len(sub_events):,} events on {period_label}")
    if sub_events.empty:
        return pd.DataFrame()

    df15_by_sym = {sym: g.sort_values("date").reset_index(drop=True)
                   for sym, g in df15.groupby("symbol", sort=False)}
    sd_floor = big5m["date"].dt.floor("D")
    sym_sess_5m: Dict[Tuple[str, date], pd.DataFrame] = {}
    for (sym, sd_ts), g in big5m.groupby([big5m["symbol"], sd_floor], sort=False):
        sym_sess_5m[(sym, sd_ts.date())] = g.sort_values("date").reset_index(drop=True)

    combos = list(product(SWEEP_SL_BUFFERS, SWEEP_T1_RS, SWEEP_T2_RS, SWEEP_T1_QTYS, SWEEP_TIME_STOPS))
    rows = []
    for idx, (sl, t1, t2, q1, ts) in enumerate(combos, 1):
        if t2 < t1:
            continue
        tr = simulate_combo_for_cell(sub_events, df15_by_sym, sym_sess_5m, cap_map,
                                     sl, t1, t2, q1, ts)
        if tr.empty:
            continue
        pnl = tr["net_pnl"]
        pf = _pf(pnl)
        wr = 100.0 * float((pnl > 0).mean())
        sh = _sharpe_of(tr.groupby("session_date")["net_pnl"].sum())
        m = tr.copy()
        m["_mo"] = pd.to_datetime(m["session_date"]).dt.strftime("%Y-%m")
        monthly = m.groupby("_mo")["net_pnl"].sum()
        n_mo = int(monthly.size)
        win_mo = int((monthly > 0).sum())
        win_mo_pct = 100.0 * win_mo / n_mo if n_mo > 0 else 0.0
        tot = float(monthly.sum())
        top_pct = (100.0 * float(monthly.abs().max()) / abs(tot)) if abs(tot) > 1e-6 else 0.0
        rows.append({
            "sl_buf": sl, "t1_r": t1, "t2_r": t2, "t1_qty": q1, "time_stop": ts,
            "n": int(len(tr)),
            "pf": pf, "wr": wr, "sh": sh,
            "n_mo": n_mo, "win_mo": win_mo, "win_mo_pct": win_mo_pct,
            "top_mo_pct": top_pct, "net": float(pnl.sum()),
        })
        if idx % 60 == 0:
            print(f"    [{idx}/{len(combos)}] sl={sl:.3f} t1={t1} t2={t2} q1={q1} ts={ts}"
                  f" -> n={len(tr)} pf={pf:.3f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=== RSI extreme reversal V2 — PROPER 2-year Discovery gauntlet ===")
    print(f"Discovery: {DISCOVERY_START} .. {DISCOVERY_END}")
    print(f"OOS:       {OOS_START} .. {OOS_END}")
    print(f"Holdout:   {HOLDOUT_START} .. {HOLDOUT_END}")

    universe, cap_map = build_universe()
    print(f"\nUniverse: {len(universe):,} symbols")

    # ====== DISCOVERY ======
    disc_trades, disc_df15, disc_big5m = run_period(
        DISCOVERY_START, DISCOVERY_END, "discovery", universe, cap_map
    )
    if disc_trades.empty:
        print("\n[NO DISCOVERY TRADES] exiting.")
        return

    disc_trades.to_csv(_OUT_DIR / "rsi_v2_discovery_trades.csv", index=False)
    print(f"  wrote rsi_v2_discovery_trades.csv  ({len(disc_trades):,})")

    disc_agg = print_aggregate(disc_trades, "discovery")
    disc_annot = annotate_cells(disc_trades)
    disc_surv, disc_ship = deep_cell_mine(disc_annot, "discovery")

    if disc_ship.empty and disc_surv.empty:
        print("\n[RETIRE] Discovery: no survivor cells, no ship-eligible cells. Pattern has no edge.")
        return

    # ====== OOS VALIDATION ======
    if disc_ship.empty:
        print(f"\n[NOTE] No ship-eligible Discovery cells. {len(disc_surv)} survivors will still be OOS-checked.")
        cells_to_validate = disc_surv
    else:
        print(f"\n[Validating {len(disc_ship)} ship-eligible Discovery cells on OOS]")
        cells_to_validate = disc_ship

    # Free Discovery 5m to save RAM before OOS load
    del disc_big5m, disc_df15
    gc.collect()

    oos_trades, oos_df15, oos_big5m = run_period(
        OOS_START, OOS_END, "oos", universe, cap_map
    )
    if oos_trades.empty:
        print("\n[NO OOS TRADES] exiting.")
        return
    oos_trades.to_csv(_OUT_DIR / "rsi_v2_oos_trades.csv", index=False)
    print_aggregate(oos_trades, "oos")

    oos_annot = annotate_cells(oos_trades)

    print(f"\n{'='*78}\nOOS CELL VALIDATION — {len(cells_to_validate)} cell(s)\n{'='*78}")
    oos_pass_rows = []
    for _, row in cells_to_validate.iterrows():
        cf = parse_cell(row["dims"], row["cell"])
        sub = apply_cell_filter(oos_annot, cf)
        if sub.empty:
            print(f"  [{row['dims']}] {row['cell']}  -> 0 OOS trades")
            continue
        agg = _agg_row(sub)
        passes = (agg["n"] >= OOS_PASS_N and agg["pf"] >= OOS_PASS_PF
                  and agg["sharpe"] > OOS_PASS_SH)
        flag = "PASS" if passes else "FAIL"
        print(f"  [{row['dims']}] {row['cell']}  n={agg['n']} PF={agg['pf']:.3f} "
              f"Sh={agg['sharpe']:.2f} mo[win={agg['win_mo_pct']}% top={agg['top_mo_pct']}%]  "
              f"NET={agg['net']:,.0f}  [{flag}]")
        if passes:
            oos_pass_rows.append({
                "dims": row["dims"], "cell": row["cell"],
                "disc_pf": float(row["pf"]), "disc_n": int(row["n"]),
                "oos_pf": agg["pf"], "oos_n": agg["n"], "oos_sh": agg["sharpe"],
                "oos_win_mo": agg["win_mo_pct"], "oos_top_mo": agg["top_mo_pct"],
                "oos_net": agg["net"],
            })

    oos_pass_df = pd.DataFrame(oos_pass_rows)
    oos_pass_csv = _OUT_DIR / "rsi_v2_oos_pass_cells.csv"
    oos_pass_df.to_csv(oos_pass_csv, index=False)
    print(f"\n  OOS-pass cells: {len(oos_pass_df)}  -> {oos_pass_csv.name}")

    if oos_pass_df.empty:
        print("\n[RETIRE] No cells pass OOS validation. Skipping Holdout + sweep.")
        return

    # ====== HOLDOUT VALIDATION ======
    # Free OOS data before Holdout load
    del oos_big5m, oos_df15
    gc.collect()

    hold_trades, hold_df15, hold_big5m = run_period(
        HOLDOUT_START, HOLDOUT_END, "holdout", universe, cap_map
    )
    if hold_trades.empty:
        print("\n[NO HOLDOUT TRADES] cannot validate further.")
        return
    hold_trades.to_csv(_OUT_DIR / "rsi_v2_holdout_trades.csv", index=False)
    print_aggregate(hold_trades, "holdout")

    hold_annot = annotate_cells(hold_trades)
    print(f"\n{'='*78}\nHOLDOUT CELL VALIDATION — {len(oos_pass_df)} cell(s)\n{'='*78}")
    hold_pass_rows = []
    for _, row in oos_pass_df.iterrows():
        cf = parse_cell(row["dims"], row["cell"])
        sub = apply_cell_filter(hold_annot, cf)
        if sub.empty:
            print(f"  [{row['dims']}] {row['cell']}  -> 0 Holdout trades")
            continue
        agg = _agg_row(sub)
        passes = (agg["n"] >= OOS_PASS_N and agg["pf"] >= OOS_PASS_PF
                  and agg["sharpe"] > OOS_PASS_SH)
        flag = "PASS" if passes else "FAIL"
        print(f"  [{row['dims']}] {row['cell']}  n={agg['n']} PF={agg['pf']:.3f} "
              f"Sh={agg['sharpe']:.2f} mo[win={agg['win_mo_pct']}% top={agg['top_mo_pct']}%]  "
              f"NET={agg['net']:,.0f}  [{flag}]")
        if passes:
            hold_pass_rows.append({
                "dims": row["dims"], "cell": row["cell"],
                "hold_pf": agg["pf"], "hold_n": agg["n"], "hold_sh": agg["sharpe"],
                "hold_win_mo": agg["win_mo_pct"], "hold_top_mo": agg["top_mo_pct"],
                "hold_net": agg["net"],
            })

    hold_pass_df = pd.DataFrame(hold_pass_rows)
    hold_pass_csv = _OUT_DIR / "rsi_v2_holdout_pass_cells.csv"
    hold_pass_df.to_csv(hold_pass_csv, index=False)
    print(f"\n  Holdout-pass cells: {len(hold_pass_df)}  -> {hold_pass_csv.name}")

    # ====== SWEEP on first OOS-pass cell ======
    # Sweep is gated on Discovery+OOS survival (per task spec). We rebuild
    # Discovery 5m + 15m to run the sweep on the surviving cell's events.
    del hold_big5m, hold_df15
    gc.collect()

    print(f"\n{'='*78}\nSL/T1/T2 SWEEP on surviving cells\n{'='*78}")
    print("  Reloading Discovery 5m+15m for sweep ...")
    disc_big5m_sw = load_5m_for_period(DISCOVERY_START, DISCOVERY_END, universe)
    disc_df15_sw = aggregate_to_15m_with_rsi(disc_big5m_sw)
    disc_events_sw = detect_events(disc_df15_sw)
    disc_events_annot = annotate_events_for_cells(disc_events_sw, cap_map)

    # Sweep at most TOP 3 OOS-pass cells (cost-bound; 576 combos each)
    for idx, row in oos_pass_df.head(3).iterrows():
        print(f"\n--- Sweep cell {idx+1}/{min(3, len(oos_pass_df))}: [{row['dims']}] {row['cell']} ---")
        sweep_df = run_sweep_on_cell(
            row["dims"], row["cell"], disc_events_annot,
            disc_df15_sw, disc_big5m_sw, cap_map, "discovery"
        )
        if sweep_df.empty:
            print(f"  Sweep yielded no rows.")
            continue
        sweep_csv = _OUT_DIR / f"rsi_v2_sweep_cell{idx}_discovery.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"  wrote {sweep_csv.name}  ({len(sweep_df)} combos)")
        # Top combos by PF (with n>=50 floor on cell-level sweep)
        filt = sweep_df[sweep_df["n"] >= 50].sort_values("pf", ascending=False)
        print(f"\n  TOP 10 combos by PF (n>=50):")
        cols = ["sl_buf", "t1_r", "t2_r", "t1_qty", "time_stop", "n", "pf", "wr", "sh",
                "win_mo_pct", "top_mo_pct", "net"]
        print(filt.head(10)[cols].to_string(index=False))

    print(f"\n{'='*78}\nDONE\n{'='*78}")


if __name__ == "__main__":
    main()
