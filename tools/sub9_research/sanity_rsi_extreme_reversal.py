"""Mathematical-pattern sanity check for RSI extreme reversal.

Pure-math test (NOT a sub-9 §3.3 brief-gated test) — let cell mining find
which cells, if any, have edge.

PATTERN
-------
14-period RSI on 15m close prices (rolling).
  - RSI < 25 (oversold) -> LONG signal
  - RSI > 75 (overbought) -> SHORT signal
Confirmation = next 15m bar closes in the direction of expected reversal:
  - bullish 15m bar after RSI<25 -> LONG entry
  - bearish 15m bar after RSI>75 -> SHORT entry
Entry on confirmation bar's close.

DATA
----
5m feathers carry a precomputed 14-period RSI on 5m close (Wilder smoothing,
see `services/indicators/indicators.py:calculate_rsi`). To match the pattern
specified in the brief (RSI on 15m close, 14-period), the 5m frame is
aggregated to 15m bars (groupby symbol/15min-bin: open=first, high=max,
low=min, close=last, volume=sum) and a fresh 14-period Wilder RSI is computed
on the 15m close series. This avoids the noise of 5m RSI and keeps the
pattern definition canonical.

MECHANICS
---------
Entry  = confirmation bar's close
Hard SL:
  SHORT -> recent (last 4 15m bars incl. confirmation) HIGH * 1.005
  LONG  -> recent (last 4 15m bars incl. confirmation) LOW  * 0.995
  Min stop distance: 0.5% of entry
R = |entry - hard_sl|
T1 (50% qty) = entry +/- 1.0R
T2 (50% qty) = entry +/- 2.0R
Time stop = 15:10 IST  (last 5m bar before MIS auto-square)
BE trail: active_sl = entry if t1_hit else hard_sl
Latch: one fire per (symbol, session_date)

UNIVERSE
--------
Full NSE x MIS-enabled: nse_all.json filtered to
  mis_leverage >= 1.0
  cap_segment in {large_cap, mid_cap, small_cap}
plus union with assets/fno_liquid_200.csv.

PERIOD
------
Discovery: 2024-09-01 .. 2025-09-30
OOS:       2025-10-01 .. 2026-04-30 (only if discovery passes/marginal)

GAUNTLET-V2 SHIP GATES
----------------------
n >= 125, NET PF >= 1.30, Sharpe >= 0.5,
per-month winning >= 55%, top-month NET < 40%.

REGIME-BREAK PRE-FLIGHT
-----------------------
check_window depends_on=["MIS_leverage","STT_drag"]
window = Discovery, min_severity=high, raise_on_break=False (informational).

Usage:
    python tools/sub9_research/sanity_rsi_extreme_reversal.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.indicators.indicators import calculate_rsi  # noqa: E402
from services.regime_break_detector import check_window, GauntletRegimeBreak  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_NSE_ALL_JSON = _REPO / "nse_all.json"
_FNO_LIQUID = _REPO / "assets" / "fno_liquid_200.csv"
_OUT_TRADES = _REPO / "reports" / "sub9_sanity" / "rsi_extreme_reversal_trades.csv"
_OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)

DISCOVERY_START = date(2024, 9, 1)
DISCOVERY_END = date(2025, 9, 30)
OOS_START = date(2025, 10, 1)
OOS_END = date(2026, 4, 30)

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

# Ship gates (Gauntlet v2)
SHIP_N = 125
SHIP_PF = 1.30
SHIP_SHARPE = 0.5
SHIP_WIN_MO_PCT = 55.0
SHIP_TOP_MO_PCT = 40.0

# Survivor (cell mining)
SURV_N = 100
SURV_PF = 1.20
SURV_SHARPE = 0.0


# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------
def build_universe() -> Tuple[set, Dict[str, str]]:
    """Return (set of bare symbols, dict symbol->cap_segment).

    Universe = (nse_all.json filtered to mis_leverage >= 1.0 AND cap_segment
    in {large/mid/small}) UNION fno_liquid_200 (these are large-cap F&O names
    that may not all be in nse_all.json's cap classification).
    """
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

    # Union with fno_liquid_200
    fno_df = pd.read_csv(_FNO_LIQUID)
    for s in fno_df["symbol"].dropna().astype(str):
        bare = s.replace("NSE:", "").strip()
        if bare:
            universe.add(bare)
            cap_map.setdefault(bare, "large_cap")  # fno_liquid is large-cap by definition
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
    """Load all 5m_enriched feathers covering [start, end]."""
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
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    # Filter to the exact date window
    d_mask = (big["date"].dt.date >= start) & (big["date"].dt.date <= end)
    big = big[d_mask].copy()
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"  total 5m bars after universe + date filter: {len(big):,}")
    return big


def aggregate_to_15m_with_rsi(big5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m -> 15m within each session day, compute 14-period
    Wilder RSI on 15m close per symbol (continuous across sessions, like
    production indicator state).

    Returns frame with columns: symbol, date (15m bar timestamp - start of
    bin), open, high, low, close, volume, rsi, session_date, hhmm.
    """
    if big5m.empty:
        return big5m
    print("  aggregating 5m -> 15m bars ...")
    df = big5m.copy()
    df["session_date"] = df["date"].dt.date
    # 15m bin start: floor to 15min
    df["bin"] = df["date"].dt.floor("15min")
    grouped = df.groupby(["symbol", "session_date", "bin"], sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    grouped = grouped.rename(columns={"bin": "date"})
    grouped["hhmm"] = grouped["date"].dt.strftime("%H:%M")
    # Drop incomplete trailing bins after 15:15 (NSE intraday close is 15:30,
    # last full 15m bar starts at 15:15 covering 15:15-15:30). We keep 15:15
    # because trade time stop is 15:10 and we exit on 5m bar walk.
    grouped = grouped.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"  total 15m bars: {len(grouped):,}")
    print("  computing 14-period Wilder RSI per symbol (continuous) ...")
    # Compute RSI per symbol — Wilder smoothing is recursive so it MUST be
    # computed per-symbol, not vectorized across the whole frame.
    grouped["rsi"] = grouped.groupby("symbol", sort=False)["close"].transform(
        lambda s: calculate_rsi(s, period=RSI_PERIOD)
    )
    return grouped


# ----------------------------------------------------------------------------
# Event detection
# ----------------------------------------------------------------------------
def detect_events(df15: pd.DataFrame) -> pd.DataFrame:
    """Find RSI-extreme + confirmation events.

    Trigger bar: RSI < RSI_OVERSOLD (long) or RSI > RSI_OVERBOUGHT (short).
    Confirmation bar: NEXT 15m bar in same session_date, close > open for long,
                      close < open for short.

    Returns one row per (symbol, session_date) — latch=one fire per day, even
    if multiple triggers fire intra-day. Keeps the FIRST qualifying event.

    Excludes:
      - Trigger bar with hhmm < 09:30 (no 09:15-09:29 trigger; need first 15m bar
        to be 09:15-09:29 for warmup, first valid trigger bar is 09:30 onward;
        plus we want confirmation bar to land in the trading window, not before
        market open).
      - Trigger bar with hhmm >= 14:30 (confirmation would be 14:45 — entry at
        14:45 close leaves only 25 min before 15:10 time stop; allow it but
        gate against 14:45 trigger which makes 15:00 confirmation).
        Actually keep all triggers up to 14:30 inclusive (confirmation 14:45,
        entry 14:45 close, 25 min until 15:10 stop). Drop triggers >= 14:45.
    """
    print("  detecting RSI-extreme + confirmation events ...")
    df = df15.copy()
    # Per-symbol shift to get next bar's open/close & date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    grp = df.groupby("symbol", sort=False)
    df["next_open"] = grp["open"].shift(-1)
    df["next_close"] = grp["close"].shift(-1)
    df["next_high"] = grp["high"].shift(-1)
    df["next_low"] = grp["low"].shift(-1)
    df["next_date"] = grp["date"].shift(-1)
    df["next_session"] = grp["session_date"].shift(-1)
    df["next_hhmm"] = df["next_date"].dt.strftime("%H:%M")

    # Trigger conditions
    long_trig = (df["rsi"] < RSI_OVERSOLD) & df["rsi"].notna()
    short_trig = (df["rsi"] > RSI_OVERBOUGHT) & df["rsi"].notna()

    # Confirmation must be in same session
    same_sess = df["session_date"] == df["next_session"]
    # Confirmation bar must have hhmm <= "14:45" (last entry allowed; 25 min to
    # time stop). Triggers at 14:30 -> conf 14:45 (entry). Triggers >= 14:45
    # would put confirmation at >= 15:00, leaving < 10 min — drop.
    conf_in_window = df["next_hhmm"].fillna("99:99") <= "14:45"

    # Confirmation direction
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

    # Latch: one fire per (symbol, session_date) — keep first by next_date
    events = events.sort_values(["symbol", "session_date", "next_date"])
    events = events.drop_duplicates(subset=["symbol", "session_date"], keep="first").reset_index(drop=True)

    # Severity bucket on the trigger RSI
    def _sev(r: float, direction: str) -> str:
        if direction == "long":
            return "deep" if r < RSI_DEEP_LOW else "moderate"
        else:
            return "deep" if r > RSI_DEEP_HIGH else "moderate"

    events["rsi_severity"] = events.apply(
        lambda r: _sev(r["rsi_trigger"], r["direction"]), axis=1
    )

    # Time-of-day bucket on the confirmation bar (entry time)
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
    """For each event, compute SL/T1/T2 and walk 5m bars from confirmation
    bar close onward to find exit.

    SL lookback: last SL_LOOKBACK_15M 15m bars INCLUDING the confirmation bar.
    """
    print(f"  simulating {len(events):,} events ...")

    # Index 15m by symbol for SL lookback
    df15 = df15.sort_values(["symbol", "date"]).reset_index(drop=True)
    df15_by_sym: Dict[str, pd.DataFrame] = {
        sym: g for sym, g in df15.groupby("symbol", sort=False)
    }

    # Index 5m by (symbol, session_date) for bar walk
    big5m = big5m.copy()
    big5m["session_date"] = big5m["date"].dt.date
    big5m["hhmm"] = big5m["date"].dt.strftime("%H:%M")
    sym_sess_5m: Dict[Tuple[str, date], pd.DataFrame] = {
        k: g.sort_values("date").reset_index(drop=True)
        for k, g in big5m.groupby(["symbol", "session_date"], sort=False)
    }

    trades: List[dict] = []

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        sd = ev["session_date"]
        direction = ev["direction"]
        conf_date = pd.Timestamp(ev["next_date"])
        entry_price = float(ev["next_close"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        # SL lookback: last 4 15m bars including confirmation
        sym_15 = df15_by_sym.get(sym)
        if sym_15 is None:
            continue
        # Take the last SL_LOOKBACK_15M bars with date <= conf_date
        mask = sym_15["date"] <= conf_date
        lookback = sym_15.loc[mask].tail(SL_LOOKBACK_15M)
        if lookback.empty:
            continue

        if direction == "long":
            sl_base = float(lookback["low"].min())
            hard_sl = sl_base * SL_BUF_LONG
            stop_distance = entry_price - hard_sl
            # Min-stop floor
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

        # 5m bar walk from confirmation bar close onward
        bars = sym_sess_5m.get((sym, sd))
        if bars is None or bars.empty:
            continue
        # Confirmation 15m bar covers [conf_date, conf_date + 15m). Entry is
        # at the confirmation bar's CLOSE, which is the close of the 5m bar
        # ending at conf_date + 15m. So we want bars with date > conf_date+10m
        # actually with date >= conf_date+15m (the bin that STARTS at +15m).
        walk_start = conf_date + pd.Timedelta(minutes=15)
        post = bars[bars["date"] >= walk_start]
        if post.empty:
            # No 5m bars after confirmation — accept the next available close as exit
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
            hhmm = bar["hhmm"]

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
            # Ran out of bars before time stop — exit at last close
            last = post.iloc[-1]
            time_exit_price = float(last["close"])
            exit_reason = "last_bar"
            exit_ts = last["date"]

        # PnL — partial fills
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

        # Composite avg exit for fee calc (volume-weighted)
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
# Reporting + cell mining
# ----------------------------------------------------------------------------
def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _sharpe_daily(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    daily = df.groupby("session_date")["net_pnl"].sum()
    if daily.size < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std())


def _monthly_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n_months=0, win_mo_pct=0.0, top_mo_pct=0.0)
    m = df.copy()
    m["_mo"] = pd.to_datetime(m["session_date"]).dt.strftime("%Y-%m")
    monthly = m.groupby("_mo")["net_pnl"].sum()
    n_mo = int(monthly.size)
    win_mo_pct = 100.0 * float((monthly > 0).mean()) if n_mo > 0 else 0.0
    total = float(monthly.sum())
    if abs(total) > 1e-6:
        top_pct = 100.0 * float(monthly.abs().max()) / abs(total)
    else:
        top_pct = 0.0
    return dict(n_months=n_mo, win_mo_pct=round(win_mo_pct, 1),
                top_mo_pct=round(top_pct, 1))


def _agg_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, pf=0.0, wr=0.0, net=0.0, sharpe=0.0,
                    n_months=0, win_mo_pct=0.0, top_mo_pct=0.0)
    pnl = df["net_pnl"]
    pf = _pf(pnl)
    wr = 100.0 * float((pnl > 0).mean())
    sharpe = _sharpe_daily(df)
    m = _monthly_stats(df)
    return dict(
        n=int(len(df)), pf=float(pf), wr=float(wr),
        net=float(pnl.sum()), sharpe=float(sharpe),
        n_months=m["n_months"], win_mo_pct=m["win_mo_pct"],
        top_mo_pct=m["top_mo_pct"],
    )


def _ship_pass(row: dict) -> bool:
    return (
        row["n"] >= SHIP_N
        and row["pf"] >= SHIP_PF
        and row["sharpe"] >= SHIP_SHARPE
        and row["win_mo_pct"] >= SHIP_WIN_MO_PCT
        and row["top_mo_pct"] < SHIP_TOP_MO_PCT
    )


def _surv_pass(row: dict) -> bool:
    return (
        row["n"] >= SURV_N
        and row["pf"] >= SURV_PF
        and row["sharpe"] >= SURV_SHARPE
    )


def cell_mine(trades: pd.DataFrame) -> pd.DataFrame:
    """Scan 1D/2D/3D combos of cell dims; return aggregate metrics per cell."""
    dims = ["direction", "tod_bucket", "cap_segment", "rsi_severity"]
    dims = [d for d in dims if d in trades.columns]

    rows = []
    for k in range(1, len(dims) + 1):
        for combo in combinations(dims, k):
            for cell_vals, sub in trades.groupby(list(combo), observed=True):
                if not isinstance(cell_vals, tuple):
                    cell_vals = (cell_vals,)
                cell = " | ".join(f"{c}={v}" for c, v in zip(combo, cell_vals))
                agg = _agg_row(sub)
                rows.append({
                    "dims": ",".join(combo),
                    "k": k,
                    "cell": cell,
                    **agg,
                })
    return pd.DataFrame(rows)


def regime_preflight() -> None:
    print("\n=== regime_break pre-flight ===")
    print(f"  strategy:    rsi_extreme_reversal")
    print(f"  depends_on:  ['MIS_leverage', 'STT_drag']")
    print(f"  window:      {DISCOVERY_START} .. {DISCOVERY_END}  (Discovery)")
    try:
        hits = check_window(
            strategy_name="rsi_extreme_reversal",
            depends_on=["MIS_leverage", "STT_drag"],
            window_label="Discovery",
            start=DISCOVERY_START,
            end=DISCOVERY_END,
            min_severity="high",
            raise_on_break=False,
        )
        if not hits:
            print("  PASS (no high/critical rule changes in window).")
        else:
            print(f"  NOTE: {len(hits)} high+ rule change(s) in window (informational):")
            for r in hits:
                desc = (r.description or "")[:90].encode("ascii", errors="replace").decode("ascii")
                print(f"    - {r.effective_date} [{r.severity}] {desc}")
    except GauntletRegimeBreak as e:
        print(f"  WARN: {e}")


def print_report(trades: pd.DataFrame, period_label: str) -> Tuple[dict, pd.DataFrame]:
    print(f"\n{'='*78}")
    print(f"{period_label.upper()} — {DISCOVERY_START if period_label=='discovery' else OOS_START}"
          f" .. {DISCOVERY_END if period_label=='discovery' else OOS_END}")
    print('=' * 78)
    if trades.empty:
        print("  [NO TRADES]")
        return ({}, pd.DataFrame())

    agg = _agg_row(trades)
    print(f"\nAGGREGATE:")
    print(f"  n={agg['n']:,}  PF={agg['pf']:.3f}  WR={agg['wr']:.1f}%  "
          f"NET={agg['net']:,.0f}  Sharpe(daily)={agg['sharpe']:.3f}")
    print(f"  months={agg['n_months']}  win_mo={agg['win_mo_pct']}%  "
          f"top_mo={agg['top_mo_pct']}%")

    # Per direction × tod × cap × severity 1D breakdown
    print("\nPer cell (1D):")
    for dim in ["direction", "tod_bucket", "cap_segment", "rsi_severity"]:
        print(f"\n  -- {dim} --")
        for val, sub in trades.groupby(dim):
            r = _agg_row(sub)
            print(f"    {dim}={val:<12} n={r['n']:>5} PF={r['pf']:.3f} "
                  f"WR={r['wr']:.1f}% Sh={r['sharpe']:.2f} "
                  f"win_mo={r['win_mo_pct']}% top_mo={r['top_mo_pct']}% "
                  f"NET={r['net']:,.0f}")

    # Cell mine
    cells = cell_mine(trades)
    surv = cells[cells.apply(_surv_pass, axis=1)].sort_values(["pf", "n"], ascending=[False, False])
    ship = cells[cells.apply(_ship_pass, axis=1)].sort_values(["pf", "n"], ascending=[False, False])

    print(f"\nCELL MINING:")
    print(f"  Survivors (n>={SURV_N}, PF>={SURV_PF}, Sh>={SURV_SHARPE}): {len(surv):,}")
    for _, r in surv.head(15).iterrows():
        print(f"    [{r['dims']}] {r['cell']}  n={r['n']} "
              f"PF={r['pf']:.3f} WR={r['wr']:.1f}% Sh={r['sharpe']:.2f} "
              f"win_mo={r['win_mo_pct']}% top_mo={r['top_mo_pct']}% "
              f"NET={r['net']:,.0f}")

    print(f"\n  Ship-eligible (n>={SHIP_N}, PF>={SHIP_PF}, Sh>={SHIP_SHARPE}, "
          f"win_mo>={SHIP_WIN_MO_PCT}%, top_mo<{SHIP_TOP_MO_PCT}%): {len(ship):,}")
    for _, r in ship.head(15).iterrows():
        print(f"    [{r['dims']}] {r['cell']}  n={r['n']} "
              f"PF={r['pf']:.3f} WR={r['wr']:.1f}% Sh={r['sharpe']:.2f} "
              f"win_mo={r['win_mo_pct']}% top_mo={r['top_mo_pct']}% "
              f"NET={r['net']:,.0f}")

    return (agg, ship)


def print_verdict(agg: dict, ship: pd.DataFrame) -> str:
    print("\n--- SHIP-GATE VERDICT (Discovery aggregate) ---")
    if _ship_pass(agg):
        verdict = "PASS"
        print(f"  PASS — aggregate clears all Gauntlet-v2 gates.")
    elif len(ship) > 0:
        verdict = "CELL-LOCKED"
        print(f"  CELL-LOCKED — aggregate fails, but {len(ship)} cell(s) ship-eligible.")
    elif (agg.get("n", 0) >= SURV_N and agg.get("pf", 0) >= SURV_PF
          and agg.get("sharpe", 0) >= SURV_SHARPE):
        verdict = "MARGINAL"
        print(f"  MARGINAL — clears survivor floor; may warrant OOS test.")
    else:
        verdict = "RETIRE"
        print(f"  RETIRE — fails ship + survivor floors. Pattern has no edge.")
    return verdict


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_period(start: date, end: date, label: str,
               universe: set, cap_map: Dict[str, str]) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    print(f"\n{'#'*78}\n# {label.upper()} {start} .. {end}\n{'#'*78}")
    big5m = load_5m_for_period(start, end, universe)
    if big5m.empty:
        print("  [empty 5m frame]")
        return (pd.DataFrame(), {}, pd.DataFrame())
    df15 = aggregate_to_15m_with_rsi(big5m)
    events = detect_events(df15)
    print(f"  events (RSI-extreme + confirmation, latch=1/sym/day): {len(events):,}")
    if not events.empty:
        print(f"    long={int((events['direction']=='long').sum()):,}  "
              f"short={int((events['direction']=='short').sum()):,}")
    trades = simulate(events, df15, big5m, cap_map)
    print(f"  trades simulated: {len(trades):,}")
    agg, ship = print_report(trades, label)
    return (trades, agg, ship)


def main():
    print("=== RSI extreme reversal -- sanity check ===")

    # Build universe
    universe, cap_map = build_universe()
    print(f"\nUniverse: {len(universe):,} symbols "
          f"(nse_all.json filtered + fno_liquid_200 union)")

    # Regime pre-flight
    regime_preflight()

    # Discovery
    disc_trades, disc_agg, disc_ship = run_period(
        DISCOVERY_START, DISCOVERY_END, "discovery", universe, cap_map
    )

    if disc_trades.empty:
        print("\n[NO DISCOVERY TRADES]  exiting.")
        return

    disc_trades.to_csv(_OUT_TRADES, index=False)
    print(f"\nDiscovery trades written: {_OUT_TRADES}")

    verdict = print_verdict(disc_agg, disc_ship)

    # OOS decision
    print(f"\n--- OOS DECISION ---")
    run_oos = verdict in ("PASS", "CELL-LOCKED", "MARGINAL")
    if not run_oos:
        print("  SKIP OOS — Discovery retired.")
        return

    print(f"  Running OOS ({OOS_START} .. {OOS_END}) ...")
    oos_trades, oos_agg, oos_ship = run_period(
        OOS_START, OOS_END, "oos", universe, cap_map
    )
    if not oos_trades.empty:
        oos_path = _OUT_TRADES.parent / "rsi_extreme_reversal_trades_oos.csv"
        oos_trades.to_csv(oos_path, index=False)
        print(f"\nOOS trades written: {oos_path}")
        _print_verdict_oos = print_verdict(oos_agg, oos_ship)


if __name__ == "__main__":
    main()
