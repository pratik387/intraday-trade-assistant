"""Gap-fade-short sanity + SL/target sweep — streaming version.

Streams 5m feathers month-by-month to avoid loading 24 months at once.
Trigger detection only needs bars 09:15-09:30 (drops 95%+ of data on
load). Forward bars loaded only for trigger (symbol, day) pairs.

CRITICAL: gap_fade_short has the SAME architecture as circuit_t1
(T1 partial qty_pct=0.5 + breakeven trail). The circuit_t1 sweep found
this DESTROYS the strategy. This sweep verifies whether gap_fade has the
same defect.

Periods:
  discovery: 2023-01..2024-12
  oos:       2025-01..2025-09
  holdout:   2025-10..2026-03

Usage:
    python tools/sub9_research/_gap_fade_short_sl_target_sweep.py --period discovery
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure stdout is line-buffered so progress is visible during long runs
sys.stdout.reconfigure(line_buffering=True)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.symbol_metadata import get_cap_segment            # noqa: E402
from services.state.zerodha_mis_fetcher import ZerodhaMISFetcher  # noqa: E402


_MIS_FETCHER = None


def _get_mis_allowed_set() -> set:
    """Production-parity MIS filter — current Zerodha MIS list as proxy."""
    global _MIS_FETCHER
    if _MIS_FETCHER is None:
        _MIS_FETCHER = ZerodhaMISFetcher()
        if not _MIS_FETCHER.load_from_zerodha():
            print("  WARN: MIS list load failed — proceeding without MIS filter")
            return set()
        print(f"  MIS list loaded: {_MIS_FETCHER.count()} symbols")
    return set(_MIS_FETCHER._mis_symbols.keys())
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402

RISK_PER_TRADE_RUPEES = 1000
ALLOWED_CAPS = {"small_cap"}      # production locked

# ---- Production detector params ----
MIN_GAP_PCT_ABOVE_PDC = 1.5
MAX_GAP_PCT_ABOVE_PDC = 8.0
MIN_UPPER_WICK_RATIO = 0.5
MAX_BODY_SIZE_PCT = 30.0
REQUIRE_VOL_DECLINE = True
ACTIVE_START = "09:15"
ACTIVE_END = "09:30"

# ---- Sweep grid (smaller — focus on partial mode + key SL/time) ----
STOP_BUFFER_PCTS = [0.10, 0.25, 0.50, 1.0]
ATR_STOP_MULTS   = [1.0, 1.5, 2.0]
TIME_STOPS       = ["10:15", "10:45", "11:30", "13:00", "15:10"]
PARTIAL_MODES    = ["all_in", "partial_50_be_trail", "partial_50_no_trail"]
PROD_CURRENT = (0.25, 1.5, "10:15", "partial_50_be_trail")


def pf_of(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def get_period(period: str):
    if period == "discovery":
        return [(2023, m) for m in range(1, 13)] + [(2024, m) for m in range(1, 13)], date(2023, 1, 1), date(2024, 12, 31)
    if period == "oos":
        return [(2025, m) for m in range(1, 10)], date(2025, 1, 1), date(2025, 9, 30)
    if period == "holdout":
        return [(2025, m) for m in range(10, 13)] + [(2026, m) for m in range(1, 4)], date(2025, 10, 1), date(2026, 3, 31)
    raise ValueError(period)


def load_daily_pdc(date_lo: date, date_hi: date) -> Dict[Tuple[str, date], float]:
    fp = _REPO / "cache" / "preaggregate" / "consolidated_daily.feather"
    df = pd.read_feather(fp)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date_lo - timedelta(days=5)) & (df["d"] <= date_hi)][["symbol", "d", "close"]]
    df = df.sort_values(["symbol", "d"])
    df["pdc"] = df.groupby("symbol")["close"].shift(1)
    # Build dict (symbol, d) -> pdc
    pdc = {}
    for sym, g in df.groupby("symbol", sort=False):
        for d, p in zip(g["d"], g["pdc"]):
            if pd.notna(p) and p > 0:
                pdc[(sym, d)] = float(p)
    return pdc


# ---------------------------------------------------------------------------
# Pass 1: Detect triggers — only need active-window bars 09:15-09:30
# ---------------------------------------------------------------------------

def detect_triggers_streaming(
    periods: List[Tuple[int, int]],
    date_lo: date,
    date_hi: date,
    pdc_map: Dict[Tuple[str, date], float],
    cap_map: Dict[str, str],
) -> List[Dict]:
    triggers: List[Dict] = []
    cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    for yyyy, mm in periods:
        fp = _REPO / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not fp.exists():
            continue
        print(f"  scan {fp.name}", flush=True)
        m = pd.read_feather(fp, columns=cols)
        m["d"] = m["date"].dt.date
        m["hhmm"] = m["date"].dt.strftime("%H:%M")
        m = m[(m["d"] >= date_lo) & (m["d"] <= date_hi)]
        m = m[(m["hhmm"] >= ACTIVE_START) & (m["hhmm"] <= ACTIVE_END)]
        # Pre-filter by cap_segment via cap_map
        m["cap_segment"] = m["symbol"].map(cap_map)
        m = m[m["cap_segment"].isin(ALLOWED_CAPS)]
        # MIS-eligibility filter (production parity — Zerodha MIS list)
        _mis_set = _get_mis_allowed_set()
        if _mis_set:
            m = m[m["symbol"].isin(_mis_set)]
        if m.empty:
            del m
            continue
        # Process per (symbol, d)
        for (sym, d), g in m.groupby(["symbol", "d"], sort=False):
            pdc = pdc_map.get((sym, d))
            if pdc is None:
                continue
            g = g.sort_values("date").reset_index(drop=True)
            opening = g.iloc[0]
            gap_open = float(opening["open"])
            gap_pct = ((gap_open - pdc) / pdc) * 100.0
            if not (MIN_GAP_PCT_ABOVE_PDC <= gap_pct <= MAX_GAP_PCT_ABOVE_PDC):
                continue
            gap_high = float(opening["high"])
            opening_vol = float(opening["volume"])

            for _, bar in g.iterrows():
                bar_open = float(bar["open"])
                bar_close = float(bar["close"])
                bar_high = float(bar["high"])
                bar_vol = float(bar["volume"])
                body = abs(bar_close - bar_open)
                candle_top = max(bar_open, bar_close)
                upper_wick = bar_high - candle_top
                wick_ratio = (upper_wick / body) if body > 1e-8 else float("inf")
                body_pct = (body / bar_open) * 100.0 if bar_open > 0 else 0.0
                if wick_ratio < MIN_UPPER_WICK_RATIO:
                    continue
                if body_pct > MAX_BODY_SIZE_PCT:
                    continue
                if REQUIRE_VOL_DECLINE and bar_vol >= opening_vol:
                    continue
                triggers.append({
                    "symbol": sym, "session_date": d,
                    "cap_segment": g.iloc[0]["cap_segment"],
                    "pdc": pdc, "gap_open": gap_open, "gap_high": gap_high,
                    "gap_pct": gap_pct,
                    "entry_ts": str(bar["date"]),
                    "entry_hhmm": bar["hhmm"],
                    "entry_price": bar_close,
                })
                break   # latch: one fire per (symbol, day)
        del m
        print(f"    cumulative triggers: {len(triggers)}", flush=True)
    return triggers


# ---------------------------------------------------------------------------
# Pass 2: For each (symbol, month) that has triggers, load forward bars
# ---------------------------------------------------------------------------

def load_forward_bars_streaming(
    triggers: List[Dict],
    periods: List[Tuple[int, int]],
) -> Dict[Tuple[str, date], pd.DataFrame]:
    # Map (year, month) -> set of symbols that triggered that month
    month_syms: Dict[Tuple[int, int], set] = {}
    for t in triggers:
        d = t["session_date"]
        key = (d.year, d.month)
        month_syms.setdefault(key, set()).add(t["symbol"])

    # ATR pre-computation requires bars from 09:15-09:30 (already captured during trigger detect)
    # Forward simulation needs bars from entry_hhmm to 15:30
    fw_map: Dict[Tuple[str, date], pd.DataFrame] = {}
    cols = ["date", "symbol", "high", "low", "close"]
    for yyyy, mm in periods:
        if (yyyy, mm) not in month_syms:
            continue
        fp = _REPO / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not fp.exists():
            continue
        print(f"  fw_bars {fp.name}", flush=True)
        m = pd.read_feather(fp, columns=cols)
        m = m[m["symbol"].isin(month_syms[(yyyy, mm)])]
        m["d"] = m["date"].dt.date
        m["hhmm"] = m["date"].dt.strftime("%H:%M")
        # Only need bars from 09:15 onwards (we'll filter further per trigger)
        m = m[m["hhmm"] >= ACTIVE_START]
        for (sym, d), g in m.groupby(["symbol", "d"], sort=False):
            fw_map[(sym, d)] = g.sort_values("date")[["hhmm", "high", "low", "close", "date"]].reset_index(drop=True)
        del m
    return fw_map


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_trade(
    trig: Dict, forward: pd.DataFrame,
    stop_buffer_pct: float, atr_stop_mult: float,
    time_stop_hhmm: str, partial_mode: str,
) -> Optional[Dict]:
    if forward is None or forward.empty:
        return None
    entry_price = trig["entry_price"]
    entry_hhmm = trig["entry_hhmm"]
    gap_high = trig["gap_high"]
    pdc = trig["pdc"]

    # ATR proxy from pre-entry bars
    pre = forward[forward["hhmm"] <= entry_hhmm]
    if len(pre) == 0:
        return None
    atr_val = float((pre["high"] - pre["low"]).mean())

    sl_from_gap = gap_high * (1.0 + stop_buffer_pct / 100.0)
    sl_from_atr = entry_price + atr_val * atr_stop_mult
    hard_sl = max(sl_from_gap, sl_from_atr)
    stop_distance = hard_sl - entry_price
    if stop_distance <= 0:
        return None

    t1_target = (entry_price + pdc) / 2.0
    t2_target = pdc
    if t1_target >= entry_price:
        t1_target = entry_price - stop_distance * 0.5
    if t2_target >= entry_price:
        t2_target = entry_price - stop_distance

    qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

    after = forward[forward["hhmm"] > entry_hhmm].reset_index(drop=True)
    if after.empty:
        return None

    use_partial = partial_mode != "all_in"
    use_be_trail = partial_mode == "partial_50_be_trail"

    hit_t1 = False
    t1_exit_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    fw_hhmm = after["hhmm"].values
    fw_high = after["high"].values.astype(float)
    fw_low = after["low"].values.astype(float)
    fw_close = after["close"].values.astype(float)

    for i in range(len(after)):
        h = fw_hhmm[i]; high = fw_high[i]; low = fw_low[i]; close_b = fw_close[i]
        active_sl = entry_price if (hit_t1 and use_be_trail) else hard_sl
        if high >= active_sl:
            exit_price = active_sl
            exit_reason = "breakeven_trail" if (hit_t1 and use_be_trail) else "stop"
            break
        if use_partial and (not hit_t1) and (low <= t1_target):
            hit_t1 = True; t1_exit_price = t1_target
        if low <= t2_target:
            exit_price = t2_target; exit_reason = "t2"; break
        if h >= time_stop_hhmm:
            exit_price = close_b; exit_reason = "time_stop"; break

    if exit_price is None:
        exit_price = float(fw_close[-1]); exit_reason = "eod"

    if use_partial and hit_t1:
        qty_t1 = qty // 2; qty_t2 = qty - qty_t1
        pnl = (entry_price - t1_exit_price) * qty_t1 + (entry_price - exit_price) * qty_t2
        fee = (calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
               + calc_fee(entry_price, exit_price, qty_t2, "SELL"))
    else:
        pnl = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "SELL")
    net_pnl = pnl - fee
    return {"session_date": trig["session_date"], "exit_reason": exit_reason, "net_pnl": net_pnl}


def sweep(triggers: List[Dict], fw_map: Dict) -> pd.DataFrame:
    combos = list(product(STOP_BUFFER_PCTS, ATR_STOP_MULTS, TIME_STOPS, PARTIAL_MODES))
    print(f"\nSweep: {len(combos)} combos x {len(triggers)} trades", flush=True)
    rows = []
    for combo_i, (sb, am, ts, pm) in enumerate(combos):
        trades = []
        for trig in triggers:
            fw = fw_map.get((trig["symbol"], trig["session_date"]))
            t = simulate_trade(trig, fw, sb, am, ts, pm)
            if t is not None:
                trades.append(t)
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        net = tdf["net_pnl"].sum()
        pf = pf_of(tdf["net_pnl"])
        daily = tdf.groupby("session_date")["net_pnl"].sum()
        sharpe = (daily.mean() / daily.std()) if daily.std() > 0 else 0.0
        ex = tdf["exit_reason"].value_counts(normalize=True) * 100
        rows.append(dict(
            stop_buffer_pct=sb, atr_stop_mult=am, time_stop=ts, partial_mode=pm,
            n=len(tdf), net=net, pf=pf,
            wr=(tdf["net_pnl"] > 0).mean() * 100,
            sharpe=sharpe,
            pct_stop=ex.get("stop", 0) + ex.get("breakeven_trail", 0),
            pct_t2=ex.get("t2", 0),
            pct_time=ex.get("time_stop", 0) + ex.get("eod", 0),
            is_prod=(sb, am, ts, pm) == PROD_CURRENT,
        ))
        if (combo_i + 1) % 30 == 0:
            print(f"  [{combo_i+1}/{len(combos)}] sb={sb} am={am} ts={ts} pm={pm} PF={pf:.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=["discovery", "oos", "holdout"])
    args = ap.parse_args()
    periods, date_lo, date_hi = get_period(args.period)
    print(f"Period: {args.period} ({date_lo}..{date_hi})  {len(periods)} months", flush=True)

    print("Loading daily PDC...", flush=True)
    pdc_map = load_daily_pdc(date_lo, date_hi)
    print(f"  PDC entries: {len(pdc_map):,}", flush=True)

    # Build cap_segment map for all symbols we'll see
    syms_in_pdc = set(s for (s, _) in pdc_map.keys())
    print(f"  Building cap_segment map for {len(syms_in_pdc)} syms...", flush=True)
    cap_map = {s: get_cap_segment(f"NSE:{s}") for s in syms_in_pdc}
    print(f"  small_cap syms: {sum(1 for v in cap_map.values() if v == 'small_cap')}", flush=True)

    print("\nPass 1: detect triggers from active-window bars only", flush=True)
    triggers = detect_triggers_streaming(periods, date_lo, date_hi, pdc_map, cap_map)
    if not triggers:
        print("No triggers!", flush=True)
        return
    trig_csv = _REPO / "reports" / "sub9_sanity" / f"_gap_fade_short_triggers_{args.period}.csv"
    pd.DataFrame(triggers).to_csv(trig_csv, index=False)
    print(f"\nTotal triggers: {len(triggers)}  saved: {trig_csv.name}", flush=True)

    print("\nPass 2: load forward bars per trigger", flush=True)
    fw_map = load_forward_bars_streaming(triggers, periods)
    print(f"  fw_map entries: {len(fw_map):,}", flush=True)

    results = sweep(triggers, fw_map)
    results = results.sort_values("pf", ascending=False).reset_index(drop=True)
    out = _REPO / "reports" / "sub9_sanity" / f"_gap_fade_short_sl_target_sweep_{args.period}.csv"
    results.to_csv(out, index=False)
    print(f"\nResults: {out}", flush=True)

    print("\n=== PROD CURRENT ===", flush=True)
    prod = results[results["is_prod"]]
    print(f"{'sb':>4} {'am':>4} {'ts':>6} {'partial_mode':>22} {'n':>5} {'PF':>6} {'WR':>5} {'Sharpe':>7} {'%stop':>6} {'%t2':>5} {'%time':>6} {'net':>14}")
    for _, r in prod.iterrows():
        print(f"{r['stop_buffer_pct']:>4} {r['atr_stop_mult']:>4} {r['time_stop']:>6} {r['partial_mode']:>22} {int(r['n']):>5} {r['pf']:>6.3f} {r['wr']:>4.1f}% {r['sharpe']:>7.3f} {r['pct_stop']:>5.1f}% {r['pct_t2']:>4.1f}% {r['pct_time']:>5.1f}% Rs.{r['net']:>11,.0f}")

    print("\n=== TOP 25 BY PF ===", flush=True)
    for _, r in results.head(25).iterrows():
        tag = " <-PROD" if r["is_prod"] else ""
        print(f"{r['stop_buffer_pct']:>4} {r['atr_stop_mult']:>4} {r['time_stop']:>6} {r['partial_mode']:>22} {int(r['n']):>5} {r['pf']:>6.3f} {r['wr']:>4.1f}% {r['sharpe']:>7.3f} {r['pct_stop']:>5.1f}% {r['pct_t2']:>4.1f}% {r['pct_time']:>5.1f}% Rs.{r['net']:>11,.0f}{tag}")


if __name__ == "__main__":
    main()
