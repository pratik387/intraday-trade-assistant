"""SL / T1 / T2 / partial-pct sweep for C2 (F&O ban-entry T+1 fade).

Same mechanics as the sister sanity (sanity_fno_ban_entry_t1_fade.py):
  - Entry on T+1 at 10:00 IST 5m bar close.
  - Hard SL = high of 09:30..10:00 window on T+1, with configurable buffer.
  - T1 / T2 either R-multiple OR structural (PDC + half-fade-of-prior-day).
  - Time stop configurable.

Extended grid (audit-driven — most existing sweeps lock T1_QTY_PCT at {0, 0.5};
this sweep adds 0.33 / 0.67 splits and a parallel target_mode dimension):

    SL_BUFFER_PCTS  = [0.3, 0.5, 0.7]              (3)
    T1_R_MULTIPLES  = [0.25, 0.5, 1.0]             (3)
    T2_R_MULTIPLES  = [1.0, 1.5, 2.0]              (3)
    T1_QTY_PCTS     = [0.0, 0.33, 0.50, 0.67]      (4)  <-- KEY NEW DIM
    BE_TRAIL        = [True, False]                 (2)  (moot when T1_QTY=0)
    TIME_STOPS      = ["13:00", "14:30", "15:10"]   (3)
    TARGET_MODES    = ["r_multiple", "structural_pdc"]  (2)

Total = 3*3*3*4*2*3*2 = 1296 combos.

STUB MODE: if data/fno_ban_history/fno_ban_events.parquet is missing,
synthesize ~10 events and run end-to-end. Results in STUB will be
meaningless (n=10 << gauntlet floor of 125) — point is just to verify the
sweep iterates without crashing.

Gauntlet-v2 ship-eligible thresholds (mirrors _cell_mine_tier_a.py):
    n >= 125, NET PF >= 1.30, Sharpe >= 0.5,
    losing_months_pct <= 40%, top_month_concentration < 40%

Output: reports/sub9_sanity/_fno_ban_t1_sweep/sweep_results.csv

Usage:
    python tools/sub9_research/_fno_ban_entry_t1_fade_sl_target_sweep.py
    python tools/sub9_research/_fno_ban_entry_t1_fade_sl_target_sweep.py --period discovery-b
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Line-buffered stdout so progress is visible during long runs
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
from services.regime_break_detector import (  # noqa: E402
    check_window,
    GauntletRegimeBreak,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FNO_BAN_PARQUET = _REPO / "data" / "fno_ban_history" / "fno_ban_events.parquet"
_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_DAILY_FEATHER = _REPO / "cache" / "preaggregate" / "consolidated_daily.feather"
_OUT_DIR = _REPO / "reports" / "sub9_sanity" / "_fno_ban_t1_sweep"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_HHMM = "10:00"      # T+1 5m bar close
WINDOW_START_HHMM = "09:30"
WINDOW_END_HHMM = "10:00"
RISK_PER_TRADE_RUPEES = 1000

STRATEGY_NAME = "fno_ban_entry_t1_fade"
DEPENDS_ON = ["MWPL", "intraday_ban", "single_stock_FO", "F&O_speculation"]

# Brief Cell A primary filters
GAP_BAND_PCT = (-2.0, 2.0)      # T+1 09:15 gap_pct ∈ [-2%, +2%]
PRIOR_DAY_RETURN_MIN_PCT = 1.5  # entry day was up-move into the ban

# Periods (post-rule windows by construction; all post-Nov-3-2025)
_PERIODS = {
    "discovery-b": (date(2025, 11, 3), date(2025, 12, 31)),
    "oos-prewar": (date(2026, 1, 1), date(2026, 2, 27)),
    "oos-war":    (date(2026, 2, 28), date(2026, 4, 8)),
    "oos-postwar": (date(2026, 4, 9), date(2026, 4, 30)),
    "full":       (date(2025, 11, 3), date(2026, 4, 30)),
}

# ---- Sweep grid ----
SL_BUFFER_PCTS = [0.3, 0.5, 0.7]
T1_R_MULTIPLES = [0.25, 0.5, 1.0]
T2_R_MULTIPLES = [1.0, 1.5, 2.0]
T1_QTY_PCTS    = [0.0, 0.33, 0.50, 0.67]
BE_TRAIL_OPTS  = [True, False]
TIME_STOPS     = ["13:00", "14:30", "15:10"]
TARGET_MODES   = ["r_multiple", "structural_pdc"]

# Gauntlet-v2 ship gates
SHIP_N = 125
SHIP_PF = 1.30
SHIP_SHARPE = 0.5
SHIP_LOSE_MO_PCT_MAX = 40.0
SHIP_TOP_MO_PCT_MAX = 40.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pf_of(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def _to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return pd.Timestamp(x).date()


# ---------------------------------------------------------------------------
# Stub event generator (used when ban parquet is missing)
# ---------------------------------------------------------------------------

def _synthesize_stub_events(n: int = 10) -> pd.DataFrame:
    """Generate ~n synthetic ban events so the sweep can run end-to-end.

    This is ONLY for plumbing verification. The trades produced are
    meaningless — they exist purely to confirm all combos iterate.
    """
    rng = np.random.default_rng(seed=42)
    syms = ["IDEA", "YESBANK", "RBLBANK", "HINDCOPPER", "GMRINFRA",
            "BSOFT", "NMDC", "SAIL", "GRANULES", "BHEL"]
    base = date(2025, 11, 10)
    rows = []
    # Spread events across ~5 months so we can compute monthly metrics
    for i in range(n):
        sym = syms[i % len(syms)]
        ban_date = base + timedelta(days=int(rng.integers(0, 120)))
        # T+1 = next weekday
        t1 = ban_date + timedelta(days=1)
        while t1.weekday() >= 5:
            t1 = t1 + timedelta(days=1)
        # Synthesize realistic price levels for stub
        pdc = float(rng.uniform(50.0, 500.0))            # prior day close (=ban day close)
        prior_day_ret = float(rng.uniform(1.5, 5.0))     # +1.5 .. +5%
        t1_open_gap = float(rng.uniform(-1.5, 1.5))      # gap-band qualifying
        t1_open = pdc * (1.0 + t1_open_gap / 100.0)

        # 09:30..10:00 window: random walk; pick a window_high and entry_price
        window_high = max(t1_open, pdc) * float(rng.uniform(1.001, 1.012))
        # Entry at 10:00 5m bar close — slightly below window_high typically
        entry_price = window_high * float(rng.uniform(0.985, 0.998))

        # Forward path: simulate ~75 5m bars from 10:00..15:25
        forward = []
        cur = entry_price
        for k in range(75):
            # Slight short-fade drift (mechanism); add noise
            drift = -float(rng.uniform(-0.0002, 0.0010)) * cur
            noise = float(rng.normal(0.0, 0.0035)) * cur
            cur = max(cur + drift + noise, 1.0)
            high = cur * (1.0 + abs(float(rng.normal(0.0, 0.0020))))
            low = cur * (1.0 - abs(float(rng.normal(0.0, 0.0020))))
            close_b = float(rng.uniform(low, high))
            # bar timestamp: starting 10:00, +5 min each
            ts = datetime.combine(t1, datetime.min.time()) + timedelta(
                hours=10, minutes=k * 5
            )
            forward.append({
                "date": pd.Timestamp(ts),
                "hhmm": ts.strftime("%H:%M"),
                "high": float(high),
                "low": float(low),
                "close": float(close_b),
            })
        rows.append({
            "symbol": sym,
            "ban_date": ban_date,
            "t1_date": t1,
            "event_type": "eod" if i % 2 == 0 else "intraday",
            "cap_segment": "small_cap" if i % 3 != 0 else "mid_cap",
            "pdc": pdc,
            "prior_day_return_pct": prior_day_ret,
            "t1_open": t1_open,
            "t1_gap_pct": t1_open_gap,
            "window_high": window_high,
            "entry_price": entry_price,
            "forward_bars": forward,    # list[dict]
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-data event loader (skeleton — not exercised in STUB MODE)
# ---------------------------------------------------------------------------

def _load_real_events(date_lo: date, date_hi: date) -> Optional[pd.DataFrame]:
    """Load ban events parquet and join with 5m feathers to produce qualified events.

    NOTE: this is a skeleton for real-data runs. The parquet does not yet exist
    on disk per the brief's data-requirements section. If parquet is missing
    or join fails, returns None and the caller falls back to STUB MODE.
    """
    if not _FNO_BAN_PARQUET.exists():
        return None

    try:
        bans = pd.read_parquet(_FNO_BAN_PARQUET)
    except Exception as e:
        print(f"  WARN: failed to read {_FNO_BAN_PARQUET.name}: {e}")
        return None

    # Schema (per brief): [symbol, ban_date, ban_entry_time, ban_exit_time,
    #                      mwpl_pct_at_entry, event_type, entry_snapshot_index]
    bans["ban_date"] = pd.to_datetime(bans["ban_date"]).dt.date
    bans = bans[(bans["ban_date"] >= date_lo) & (bans["ban_date"] <= date_hi)]
    if bans.empty:
        return None

    # Daily PDC for prior-day-return and gap calc
    if not _DAILY_FEATHER.exists():
        print(f"  WARN: missing {_DAILY_FEATHER}")
        return None
    daily = pd.read_feather(_DAILY_FEATHER)
    daily["ts"] = pd.to_datetime(daily["ts"])
    if daily["ts"].dt.tz is not None:
        daily["ts"] = daily["ts"].dt.tz_localize(None)
    daily["d"] = daily["ts"].dt.date

    # For each ban event, locate T+1 (next trading day) and build event row.
    # NOTE: real-data implementation is intentionally minimal — full
    # implementation lives in the sister sanity script. Sweep tool is
    # exercised in STUB MODE only per task scope.
    print("  NOTE: real-data event-load is a skeleton; production-grade")
    print("        implementation lives in sanity_fno_ban_entry_t1_fade.py.")
    print("        For sweep-only verification, prefer STUB MODE.")
    return None


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_one_event(
    ev: Dict,
    sl_buffer_pct: float,
    t1_r: float,
    t2_r: float,
    t1_qty_pct: float,
    use_be_trail: bool,
    time_stop_hhmm: str,
    target_mode: str,
) -> Optional[Dict]:
    """Replay one event under a parameter combo.

    BE-trail correctness (mirrors _circuit_t1_sl_target_sweep.py):
        active_sl = entry_price if (t1_hit and use_be_trail) else hard_sl

    Single-target (T1_QTY=0) case: trade rides 100% to T2 / SL / time_stop.
    """
    entry_price = float(ev["entry_price"])
    window_high = float(ev["window_high"])
    pdc = float(ev["pdc"])
    prior_day_ret = float(ev["prior_day_return_pct"])

    # Hard SL: window-high + buffer (this is a SHORT, so SL is ABOVE entry)
    hard_sl = window_high * (1.0 + sl_buffer_pct / 100.0)
    stop_distance = hard_sl - entry_price
    if stop_distance <= 0:
        return None
    R = stop_distance

    # Targets
    if target_mode == "r_multiple":
        t1_target = entry_price - t1_r * R
        t2_target = entry_price - t2_r * R
    else:  # structural_pdc
        # T1 = PDC anchor (close of ban-day, == prior-day-close of T+1)
        t1_target = pdc
        # T2 = PDC * (1 - 0.5 * prior_day_return_pct / 100)
        t2_target = pdc * (1.0 - 0.5 * prior_day_ret / 100.0)
        # Sanity: ensure targets are below entry (SHORT). If structural target
        # is above entry (e.g., entry well below PDC already), fall back to R.
        if t1_target >= entry_price:
            t1_target = entry_price - t1_r * R
        if t2_target >= entry_price:
            t2_target = entry_price - t2_r * R

    qty = max(int(RISK_PER_TRADE_RUPEES / max(R, 1e-6)), 1)
    use_partial = t1_qty_pct > 0.0
    qty_at_t1 = int(qty * t1_qty_pct) if use_partial else 0
    qty_runner = qty - qty_at_t1

    forward = ev.get("forward_bars")
    if not forward:
        return None

    t1_hit = False
    t1_exit_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    for bar in forward:
        h = bar["hhmm"]
        # Skip entry bar itself
        if h <= ENTRY_HHMM:
            continue
        high = float(bar["high"])
        low = float(bar["low"])
        close_b = float(bar["close"])

        # active_sl: BE trail kicks in only AFTER T1 is hit AND use_be_trail
        active_sl = entry_price if (t1_hit and use_be_trail) else hard_sl

        # SHORT: stop triggers on HIGH crossing active_sl from below
        if high >= active_sl:
            exit_price = active_sl
            exit_reason = "breakeven_trail" if (t1_hit and use_be_trail) else "stop"
            break

        # T1 partial (SHORT: low crossing below t1_target)
        if use_partial and (not t1_hit) and (low <= t1_target):
            t1_hit = True
            t1_exit_price = t1_target

        # T2 (SHORT: low crossing below t2_target)
        if low <= t2_target:
            exit_price = t2_target
            exit_reason = "t2"
            break

        # Time stop
        if h >= time_stop_hhmm:
            exit_price = close_b
            exit_reason = "time_stop"
            break

    if exit_price is None:
        # EOD exit
        exit_price = float(forward[-1]["close"])
        exit_reason = "eod"

    # PnL: SHORT
    if use_partial and t1_hit and t1_exit_price is not None:
        pnl_t1 = (entry_price - t1_exit_price) * qty_at_t1
        pnl_runner = (entry_price - exit_price) * qty_runner
        gross_pnl = pnl_t1 + pnl_runner
        fee = (calc_fee(entry_price, t1_exit_price, qty_at_t1, "SELL")
               + calc_fee(entry_price, exit_price, qty_runner, "SELL"))
    else:
        gross_pnl = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "SELL")
    net_pnl = gross_pnl - fee

    return {
        "symbol": ev["symbol"],
        "t1_date": ev["t1_date"],
        "exit_reason": exit_reason,
        "t1_hit": bool(t1_hit),
        "net_pnl": float(net_pnl),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _evaluate_combo(events: pd.DataFrame, params: Dict) -> Optional[Dict]:
    trades = []
    for _, ev in events.iterrows():
        t = simulate_one_event(
            ev.to_dict(),
            sl_buffer_pct=params["sl_pct"],
            t1_r=params["t1r"],
            t2_r=params["t2r"],
            t1_qty_pct=params["t1q_pct"],
            use_be_trail=params["be_trail"],
            time_stop_hhmm=params["time_stop"],
            target_mode=params["target_mode"],
        )
        if t is not None:
            trades.append(t)
    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    net = float(tdf["net_pnl"].sum())
    pf = pf_of(tdf["net_pnl"])
    wr = float((tdf["net_pnl"] > 0).mean() * 100.0)

    # Daily Sharpe
    daily = tdf.groupby("t1_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if (daily.std() and daily.std() > 0) else 0.0

    # Monthly metrics
    tdf["month"] = pd.to_datetime(tdf["t1_date"]).dt.strftime("%Y-%m")
    monthly = tdf.groupby("month")["net_pnl"].sum()
    total_months = max(len(monthly), 1)
    win_mo = int((monthly > 0).sum())
    lose_mo = int((monthly <= 0).sum())
    win_mo_pct = 100.0 * win_mo / total_months
    lose_mo_pct = 100.0 * lose_mo / total_months
    top_mo_pct = (100.0 * float(monthly.abs().max()) / abs(net)) if abs(net) > 1e-6 else 0.0

    out = dict(params)
    out.update(dict(
        n=n, pf=pf, wr=wr, sharpe=sharpe,
        win_mo_pct=win_mo_pct, lose_mo_pct=lose_mo_pct, top_mo_pct=top_mo_pct,
        net=net,
    ))
    return out


def _build_grid() -> List[Dict]:
    """Build the full combo grid. Deduplicates BE_TRAIL when T1_QTY=0
    (BE trail is moot without a T1 partial)."""
    combos: List[Dict] = []
    seen_no_partial: set = set()
    for sl_pct, t1r, t2r, t1q, be, ts, tm in product(
        SL_BUFFER_PCTS, T1_R_MULTIPLES, T2_R_MULTIPLES,
        T1_QTY_PCTS, BE_TRAIL_OPTS, TIME_STOPS, TARGET_MODES,
    ):
        if t1q == 0.0:
            # BE trail is moot — keep just one copy in the OUTPUT but still
            # iterate over both BE values so the grid count matches the spec.
            key = (sl_pct, t1r, t2r, 0.0, ts, tm)
            seen_no_partial.add(key)
        combos.append(dict(
            sl_pct=sl_pct, t1r=t1r, t2r=t2r, t1q_pct=t1q,
            be_trail=be, time_stop=ts, target_mode=tm,
        ))
    return combos


def _dedupe_no_partial(df: pd.DataFrame) -> pd.DataFrame:
    """When T1_QTY=0, the be_trail flag is moot. Keep be_trail=False rows
    only (one canonical copy) in the deduplicated view."""
    keep_partial = df[df["t1q_pct"] > 0.0]
    no_partial = df[(df["t1q_pct"] == 0.0) & (~df["be_trail"])]
    return pd.concat([keep_partial, no_partial], ignore_index=True)


def _ship_eligible_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["n"] >= SHIP_N)
        & (df["pf"] >= SHIP_PF)
        & (df["sharpe"] >= SHIP_SHARPE)
        & (df["lose_mo_pct"] <= SHIP_LOSE_MO_PCT_MAX)
        & (df["top_mo_pct"] < SHIP_TOP_MO_PCT_MAX)
    )


def _print_top(df: pd.DataFrame, n: int, label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    if df.empty:
        print("  (none)")
        return
    cols = ["sl_pct", "t1r", "t2r", "t1q_pct", "be_trail", "target_mode",
            "time_stop", "n", "pf", "wr", "sharpe",
            "win_mo_pct", "lose_mo_pct", "top_mo_pct", "net"]
    head = df.head(n)[cols].copy()
    for c in ("pf", "wr", "sharpe", "win_mo_pct", "lose_mo_pct", "top_mo_pct"):
        head[c] = head[c].round(3)
    head["net"] = head["net"].round(0)
    print(head.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=list(_PERIODS.keys()), default="full",
                    help="Window for regime-break preflight + real-data filter.")
    args = ap.parse_args()

    date_lo, date_hi = _PERIODS[args.period]
    print(f"Period: {args.period} ({date_lo}..{date_hi})", flush=True)
    print(f"Strategy: {STRATEGY_NAME}", flush=True)
    print(f"Depends-on: {DEPENDS_ON}", flush=True)

    # ---- Regime-break pre-flight ----
    print("\n[preflight] regime_break_detector.check_window ...", flush=True)
    hits = check_window(
        strategy_name=STRATEGY_NAME,
        depends_on=DEPENDS_ON,
        window_label=args.period,
        start=date_lo,
        end=date_hi,
        min_severity="critical",   # only refuse on critical rule changes
        raise_on_break=False,       # report-only for research sweep
    )
    if not hits:
        print("  OK -- no critical rule changes in window.")
    else:
        # Pretty-print (ASCII-safe) — log but do not raise; the brief
        # pre-registers pre/post-Apr-1 sub-window splits as the canonical
        # remediation. Sweep is research-only, not the gauntlet itself.
        print(f"  REGIME BREAK detected for window {date_lo}..{date_hi}:")
        for h in hits:
            desc = (h.description or "").encode("ascii", "replace").decode("ascii")
            affects = sorted(set(DEPENDS_ON) & h.affects)
            print(f"    - {h.effective_date} [{h.severity.upper()}] "
                  f"{desc}  (affects {affects})")
        print("  NOTE: continuing — sweep is research-only. Production gauntlet "
              "must split pre/post-Apr-1.")

    # ---- Load events (real or stub) ----
    print("\n[load_events] attempting real-data load ...", flush=True)
    events = _load_real_events(date_lo, date_hi)
    if events is None or events.empty:
        print(f"  WARN: ban event parquet missing or empty at {_FNO_BAN_PARQUET}")
        print(f"  WARN: STUB MODE — synthesizing ~10 events for plumbing test")
        events = _synthesize_stub_events(n=10)

    print(f"  events available for sweep: {len(events)}", flush=True)
    if len(events) < SHIP_N:
        print(f"  NOTE: n={len(events)} < SHIP_N={SHIP_N}; ship-eligible top will be empty.")

    # ---- Build grid ----
    combos = _build_grid()
    print(f"\n[grid] total combos: {len(combos)}", flush=True)

    # ---- Run sweep ----
    rows: List[Dict] = []
    progress_every = max(len(combos) // 20, 1)
    for i, params in enumerate(combos):
        r = _evaluate_combo(events, params)
        if r is not None:
            rows.append(r)
        if (i + 1) % progress_every == 0 or (i + 1) == len(combos):
            print(f"  [{i+1}/{len(combos)}]  last_pf="
                  f"{(r['pf'] if r else float('nan')):.3f}", flush=True)

    if not rows:
        print("\nERROR: no combos produced any trades.")
        return

    df = pd.DataFrame(rows)
    # Dedupe view (collapse moot be_trail when t1q_pct=0)
    df_view = _dedupe_no_partial(df)
    df_view = df_view.sort_values("pf", ascending=False).reset_index(drop=True)

    # ---- Save full results ----
    out_csv = _OUT_DIR / "sweep_results.csv"
    out_cols = ["sl_pct", "t1r", "t2r", "t1q_pct", "be_trail", "target_mode",
                "time_stop", "n", "pf", "wr", "sharpe",
                "win_mo_pct", "lose_mo_pct", "top_mo_pct", "net"]
    df_view[out_cols].to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}", flush=True)
    print(f"  rows in CSV (deduped): {len(df_view)}")
    print(f"  rows in raw grid:     {len(df)}")

    # ---- Top-10 by PF (with n >= SHIP_N filter) ----
    top_by_pf = df_view[df_view["n"] >= SHIP_N].sort_values("pf", ascending=False)
    _print_top(top_by_pf, 10, f"TOP 10 BY PF (n>={SHIP_N})")

    # ---- Top-10 ship-eligible (all gauntlet-v2 gates) ----
    ship = df_view[_ship_eligible_mask(df_view)].sort_values("pf", ascending=False)
    _print_top(ship, 10, "TOP 10 SHIP-ELIGIBLE (all gauntlet-v2 gates)")

    # ---- Gate-pass-rate summary ----
    print("\n=== GATE PASS COUNTS (independent, deduped view) ===", flush=True)
    print(f"  combos evaluated:                  {len(df_view)}")
    print(f"  n >= {SHIP_N}:                          "
          f"{int((df_view['n'] >= SHIP_N).sum())}")
    print(f"  NET PF >= {SHIP_PF}:                      "
          f"{int((df_view['pf'] >= SHIP_PF).sum())}")
    print(f"  Sharpe >= {SHIP_SHARPE}:                       "
          f"{int((df_view['sharpe'] >= SHIP_SHARPE).sum())}")
    print(f"  losing_months_pct <= {SHIP_LOSE_MO_PCT_MAX}%:           "
          f"{int((df_view['lose_mo_pct'] <= SHIP_LOSE_MO_PCT_MAX).sum())}")
    print(f"  top_month_concentration < {SHIP_TOP_MO_PCT_MAX}%:     "
          f"{int((df_view['top_mo_pct'] < SHIP_TOP_MO_PCT_MAX).sum())}")
    print(f"  ALL GATES (ship-eligible):         {int(_ship_eligible_mask(df_view).sum())}")

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
