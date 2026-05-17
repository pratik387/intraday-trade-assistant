"""Pre-coding sanity check for C-02 Round-Number Stop-Cluster Sweep + Recovery.

Candidate spec: `specs/2026-05-16-new-setup-candidates.md` -> CANDIDATE-02.

THESIS: Indian retail traders cluster stop-losses at round-number prices
(Rs.100, Rs.250, Rs.500, Rs.1000, Rs.1500, Rs.2000) far more than at PDH/PDL
because retail-education courses teach this (Subasish Pani, Powerof Stocks,
Zerodha Varsity). When intraday price pokes through such a level briefly and
reverses, that's the stop-cluster being cleared. After the clear, natural
buying/selling interest at the round number reasserts, producing a tradeable
mean-reversion signal.

Distinct from retired `pdh_pdl_sweep_reclaim` (ICT institutional-stop-hunt
framing). The mechanism here is RETAIL stop-clustering at psychological levels,
not institutional manipulation.

UNIVERSE + FILTERS:
  - cap_segment in {small_cap, mid_cap} (where retail concentration is highest)
  - MIS-eligible
  - Symbol in 5m feather cache

ROUND-NUMBER DEFINITION (Indian retail psychology):
  - price < 100: multiples of Rs.25 (e.g., 25, 50, 75)
  - 100 <= price < 500: multiples of Rs.50 (e.g., 100, 150, 200, 500)
  - 500 <= price < 2000: multiples of Rs.100 (e.g., 500, 600, 1000, 1500)
  - price >= 2000: multiples of Rs.250 (e.g., 2000, 2250, 2500)
  The most "magnetic" levels are Rs.100, 250, 500, 1000, 2000 (whole-thousands etc)
  but we test the broader set in v1 to establish baseline.

SWEEP DETECTION (per 5m bar):
  - Upside sweep (SHORT signal): bar.high >= RN * 1.0015 (0.15% above)
    AND bar.close <= RN (closed back below)
  - Downside sweep (LONG signal): bar.low <= RN * 0.9985
    AND bar.close >= RN (closed back above)
  - Volume on sweep bar >= 2x session-cumulative-average

RECOVERY CONFIRMATION:
  - Next 1 bar's close stays on the recovery side (above RN for LONG, below for SHORT)

ACTIVE WINDOW: 09:30 - 15:00 IST (avoid first 15 min noise + last 10 min MIS rush)

TRADE GEOMETRY:
  - Entry at confirmation-bar's close
  - SL = 0.5% beyond sweep extreme (above sweep_high for SHORT, below sweep_low for LONG)
  - T1 = entry +/- 1R, T2 = entry +/- 2R
  - Time stop = 15:10
  - One fire per (symbol, day, RN level) - latched

Usage:
    .venv/Scripts/python tools/sub9_research/sanity_round_number_sweep.py
    .venv/Scripts/python tools/sub9_research/sanity_round_number_sweep.py --window oos
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment, get_mis_info  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee       # noqa: E402


# ----- Config knobs -----

ALLOWED_CAPS = {"small_cap", "mid_cap"}

ACTIVE_START_HHMM = "09:30"
ACTIVE_END_HHMM = "15:00"

POKE_PCT = 0.15           # bar pokes >= POKE_PCT% beyond RN
VOL_RATIO_MIN = 2.0       # bar.volume >= VOL_RATIO_MIN * session_cumulative_average

SL_PCT_BEYOND_SWEEP = 0.5
T1_R = 1.0
T2_R = 2.0
EXIT_BAR_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000

WINDOWS = {
    "discovery": (date(2023, 1, 1), date(2024, 12, 31)),
    "oos":       (date(2025, 1, 1), date(2025, 9, 30)),
    "holdout":   (date(2025, 10, 1), date(2026, 4, 30)),
}

WINDOW_START = WINDOWS["discovery"][0]
WINDOW_END = WINDOWS["discovery"][1]
WINDOW_LABEL = "discovery"


def round_numbers_near(price: float, tol_pct: float = 2.0) -> List[float]:
    """Return list of round-number levels within ±tol_pct of price.

    Increment magnitude depends on price band (Indian retail psychology):
      - price < 100: multiples of Rs.25
      - 100 <= price < 500: multiples of Rs.50
      - 500 <= price < 2000: multiples of Rs.100
      - price >= 2000: multiples of Rs.250
    """
    if price < 100:
        increment = 25
    elif price < 500:
        increment = 50
    elif price < 2000:
        increment = 100
    else:
        increment = 250

    low_band = price * (1.0 - tol_pct / 100.0)
    high_band = price * (1.0 + tol_pct / 100.0)

    # Find nearest round numbers on either side
    near_below = (price // increment) * increment
    candidates = [near_below - increment, near_below, near_below + increment, near_below + 2 * increment]
    return [c for c in candidates if low_band <= c <= high_band and c > 0]


# ----- Loaders -----

def _months_in_window() -> List[tuple]:
    months: List[tuple] = []
    y, m = WINDOW_START.year, WINDOW_START.month
    while (y, m) <= (WINDOW_END.year, WINDOW_END.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    p = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_feather(p)


def _is_mis_eligible(bare_symbol: str) -> bool:
    nse_sym = f"NSE:{bare_symbol}"
    try:
        return bool(get_mis_info(nse_sym).get("mis_enabled", False))
    except Exception:
        return False


def _cap_segment(bare_symbol: str) -> str:
    try:
        return get_cap_segment(f"NSE:{bare_symbol}")
    except Exception:
        return "unknown"


# ----- Trade simulation -----

def _simulate_day(symbol: str, day_bars: pd.DataFrame, cap_segment: str) -> List[dict]:
    """Walk day bars, fire on first qualifying round-number sweep+recovery.

    One trade per (symbol, day, RN level) - latched.
    """
    if day_bars.empty:
        return []

    bars = day_bars.copy().reset_index(drop=True)
    bars["hhmm"] = bars["date"].dt.strftime("%H%M").astype(int)
    active_start = int(ACTIVE_START_HHMM.replace(":", ""))
    active_end = int(ACTIVE_END_HHMM.replace(":", ""))
    exit_cutoff = int(EXIT_BAR_HHMM.replace(":", ""))

    # Session cumulative volume mean for vol-ratio check
    bars["cum_vol_mean"] = bars["volume"].expanding(min_periods=2).mean().shift(1)

    trades = []
    fired_keys = set()  # (symbol, day, RN)

    for i in range(len(bars) - 1):  # -1 because we need next bar for confirmation
        bar = bars.iloc[i]
        next_bar = bars.iloc[i + 1]
        hhmm = int(bar["hhmm"])
        if hhmm < active_start or hhmm > active_end:
            continue
        if pd.isna(bar["cum_vol_mean"]) or bar["cum_vol_mean"] <= 0:
            continue

        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_volume = float(bar["volume"])

        # Volume confirmation
        vol_ratio = bar_volume / float(bar["cum_vol_mean"])
        if vol_ratio < VOL_RATIO_MIN:
            continue

        # Build active round numbers around the bar's price range
        # Use the mid-price of the bar to centre the search
        rns = round_numbers_near((bar_high + bar_low) / 2.0)
        if not rns:
            continue

        for rn in rns:
            key = (symbol, bar["date"].date(), rn)
            if key in fired_keys:
                continue

            # Upside sweep check (SHORT signal)
            poke_threshold_up = rn * (1.0 + POKE_PCT / 100.0)
            if bar_high >= poke_threshold_up and bar_close <= rn:
                # Confirm: next bar's close stays below RN
                if float(next_bar["close"]) < rn:
                    direction = "short"
                    sweep_extreme = bar_high
                    entry_price = float(next_bar["close"])
                    hard_sl = sweep_extreme * (1.0 + SL_PCT_BEYOND_SWEEP / 100.0)
                    R = hard_sl - entry_price
                    if R <= 0:
                        continue
                    trade = _path_walk(bars, i + 1, direction, entry_price, hard_sl, R,
                                       exit_cutoff, symbol, bar["date"].date(), cap_segment,
                                       rn, sweep_extreme, vol_ratio)
                    if trade is not None:
                        trades.append(trade)
                        fired_keys.add(key)
                        return trades  # one trade per day, latch

            # Downside sweep check (LONG signal)
            poke_threshold_dn = rn * (1.0 - POKE_PCT / 100.0)
            if bar_low <= poke_threshold_dn and bar_close >= rn:
                if float(next_bar["close"]) > rn:
                    direction = "long"
                    sweep_extreme = bar_low
                    entry_price = float(next_bar["close"])
                    hard_sl = sweep_extreme * (1.0 - SL_PCT_BEYOND_SWEEP / 100.0)
                    R = entry_price - hard_sl
                    if R <= 0:
                        continue
                    trade = _path_walk(bars, i + 1, direction, entry_price, hard_sl, R,
                                       exit_cutoff, symbol, bar["date"].date(), cap_segment,
                                       rn, sweep_extreme, vol_ratio)
                    if trade is not None:
                        trades.append(trade)
                        fired_keys.add(key)
                        return trades

    return trades


def _path_walk(bars, entry_idx, direction, entry_price, hard_sl, R,
               exit_cutoff, symbol, trade_date, cap_segment,
               rn, sweep_extreme, vol_ratio):
    """Walk subsequent bars; return trade dict with baseline exit + MFE/MAE."""
    if direction == "long":
        t1_target = entry_price + T1_R * R
        t2_target = entry_price + T2_R * R
    else:
        t1_target = entry_price - T1_R * R
        t2_target = entry_price - T2_R * R

    after = bars.iloc[entry_idx:].copy()
    after = after[after["hhmm"] <= exit_cutoff]
    if after.empty:
        return None

    mfe_price = entry_price
    mae_price = entry_price
    closes_at_hhmm: Dict[int, float] = {}

    baseline_exit_ts = None
    baseline_exit_price = None
    baseline_exit_reason = None

    for bar in after.itertuples(index=False):
        ts = bar.date
        hi = float(bar.high)
        lo = float(bar.low)
        cl = float(bar.close)
        hhmm = int(bar.hhmm)

        if direction == "long":
            mfe_price = max(mfe_price, hi)
            mae_price = min(mae_price, lo)
        else:
            mfe_price = min(mfe_price, lo)
            mae_price = max(mae_price, hi)
        closes_at_hhmm[hhmm] = cl

        if baseline_exit_price is None:
            if direction == "long":
                if lo <= hard_sl:
                    baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, hard_sl, "stop"
                elif hi >= t2_target:
                    baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, t2_target, "t2_full"
                elif ts.strftime("%H:%M") >= EXIT_BAR_HHMM:
                    baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, cl, "time_stop"
            else:
                if hi >= hard_sl:
                    baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, hard_sl, "stop"
                elif lo <= t2_target:
                    baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, t2_target, "t2_full"
                elif ts.strftime("%H:%M") >= EXIT_BAR_HHMM:
                    baseline_exit_ts, baseline_exit_price, baseline_exit_reason = ts, cl, "time_stop"

    if baseline_exit_price is None:
        last = after.iloc[-1]
        baseline_exit_ts = last["date"]
        baseline_exit_price = float(last["close"])
        baseline_exit_reason = "last_bar"

    if direction == "long":
        mfe_r = (mfe_price - entry_price) / R
        mae_r = (entry_price - mae_price) / R
    else:
        mfe_r = (entry_price - mfe_price) / R
        mae_r = (mae_price - entry_price) / R

    def _close_at(target_hhmm: int) -> float:
        eligible = [v for k, v in closes_at_hhmm.items() if k <= target_hhmm]
        return float(eligible[-1]) if eligible else float("nan")

    qty = max(int(RISK_PER_TRADE_RUPEES / max(R, 1e-6)), 1)
    if direction == "long":
        realized_pnl = (baseline_exit_price - entry_price) * qty
        fee = calc_fee(entry_price, baseline_exit_price, qty, "BUY")
    else:
        realized_pnl = (entry_price - baseline_exit_price) * qty
        fee = calc_fee(entry_price, baseline_exit_price, qty, "SELL")
    net_pnl = realized_pnl - fee

    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "side": direction.upper(),
        "signal_type": f"round_number_sweep_{direction}",
        "cap_segment": cap_segment,
        "rn_level": rn,
        "sweep_extreme": sweep_extreme,
        "vol_ratio": vol_ratio,
        "entry_ts": bars.iloc[entry_idx]["date"],
        "entry_price": entry_price,
        "hard_sl": hard_sl,
        "t1_target": t1_target,
        "t2_target": t2_target,
        "R_per_share": R,
        "qty": qty,
        "exit_ts": baseline_exit_ts,
        "exit_price": baseline_exit_price,
        "exit_reason": baseline_exit_reason,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "close_at_1300": _close_at(1300),
        "close_at_1400": _close_at(1400),
        "close_at_1500": _close_at(1500),
        "realized_pnl": realized_pnl,
        "fee": fee,
        "net_pnl": net_pnl,
    }


# ----- Driver -----

def main() -> int:
    global WINDOW_START, WINDOW_END, WINDOW_LABEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=list(WINDOWS.keys()), default="discovery")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    WINDOW_LABEL = args.window
    WINDOW_START, WINDOW_END = WINDOWS[WINDOW_LABEL]

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "reports" / "sub9_sanity"
        / f"_round_number_sweep_trades_{WINDOW_LABEL}.csv"
    )

    print(f"== sanity_round_number_sweep ({WINDOW_LABEL}: {WINDOW_START} -> {WINDOW_END}) ==")

    # Walk 5m feathers month-by-month
    print(f"  loading 5m feathers + walking month-by-month ...")
    all_trades: List[dict] = []

    for (y, m) in _months_in_window():
        mdf = _load_5m_for_month(y, m)
        if mdf.empty:
            continue
        mdf["d"] = mdf["date"].dt.date
        mdf = mdf[(mdf["d"] >= WINDOW_START) & (mdf["d"] <= WINDOW_END)]
        if mdf.empty:
            continue

        # Per-symbol cap+MIS filter (computed once per month batch)
        syms_in_month = set(mdf["symbol"].unique())
        sym_cap = {s: _cap_segment(s) for s in syms_in_month}
        sym_mis = {s: _is_mis_eligible(s) for s in syms_in_month}
        keep_syms = {s for s in syms_in_month if sym_cap.get(s) in ALLOWED_CAPS and sym_mis.get(s)}
        mdf = mdf[mdf["symbol"].isin(keep_syms)]
        if mdf.empty:
            continue

        print(f"    {y:04d}-{m:02d}: scanning {mdf['symbol'].nunique()} symbols, {mdf['d'].nunique()} days ...")

        for (sym, d), grp in mdf.groupby(["symbol", "d"], sort=False):
            day_bars = grp.sort_values("date").reset_index(drop=True)
            cap = sym_cap.get(sym, "unknown")
            trades = _simulate_day(sym, day_bars, cap)
            all_trades.extend(trades)
        del mdf

    print(f"    trades fired total: {len(all_trades):,}")

    if not all_trades:
        print("  NO TRADES FIRED - exit")
        return 0

    tdf = pd.DataFrame(all_trades)

    # Aggregate verdict
    n = len(tdf)
    wins = int((tdf["net_pnl"] > 0).sum())
    wr = 100.0 * wins / n
    net = float(tdf["net_pnl"].sum())
    gross_wins = float(tdf[tdf["net_pnl"] > 0]["net_pnl"].sum())
    gross_losses = -float(tdf[tdf["net_pnl"] < 0]["net_pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_win = gross_wins / wins if wins else 0.0
    avg_loss = gross_losses / max(n - wins, 1)
    sharpe = (tdf["net_pnl"].mean() / tdf["net_pnl"].std(ddof=1)) * (252 ** 0.5) if tdf["net_pnl"].std(ddof=1) > 0 else 0.0
    exit_dist = tdf["exit_reason"].value_counts(normalize=True).to_dict()

    print()
    print("=" * 80)
    print(f"  C-02 ROUND-NUMBER SWEEP - {WINDOW_LABEL.upper()} VERDICT")
    print("=" * 80)
    print(f"  n trades:               {n:,}")
    print(f"  win rate:               {wr:.1f}%  ({wins} wins / {n - wins} losses)")
    print(f"  Profit Factor:          {pf:.3f}")
    print(f"  NET PnL (after fees):   Rs. {net:>+12,.0f}")
    print(f"  Avg win / avg loss:     Rs. {avg_win:>+8,.0f}  /  Rs. {avg_loss:>+8,.0f}")
    print(f"  Annualized Sharpe:      {sharpe:.2f}")
    print(f"  Exit reason mix:        {exit_dist}")
    print()

    print("  Per-direction breakdown:")
    dir_grp = tdf.groupby("side").agg(
        n=("net_pnl", "count"),
        net=("net_pnl", "sum"),
        wr=("net_pnl", lambda s: 100.0 * (s > 0).sum() / len(s)),
    )
    dir_grp["pf"] = tdf.groupby("side").apply(
        lambda s: (s[s["net_pnl"] > 0]["net_pnl"].sum() /
                   max(-s[s["net_pnl"] < 0]["net_pnl"].sum(), 1e-9))
    )
    print(dir_grp.round(2).to_string())
    print()

    print("  Per-cap-segment breakdown:")
    cap_grp = tdf.groupby("cap_segment").agg(
        n=("net_pnl", "count"),
        net=("net_pnl", "sum"),
        wr=("net_pnl", lambda s: 100.0 * (s > 0).sum() / len(s)),
    )
    cap_grp["pf"] = tdf.groupby("cap_segment").apply(
        lambda s: (s[s["net_pnl"] > 0]["net_pnl"].sum() /
                   max(-s[s["net_pnl"] < 0]["net_pnl"].sum(), 1e-9))
    )
    print(cap_grp.round(2).to_string())
    print()

    if n < 200:
        verdict = "STRUCTURAL RETIRE-PRE-DATA (n < 200)"
    elif pf >= 1.10:
        verdict = "STRONG PROCEED -> cell-mine + R-sweep"
    elif pf >= 1.0:
        verdict = "MARGINAL -> cell selection may rescue"
    else:
        verdict = "THESIS RETIRE (PF < 1.0)"
    print(f"  VERDICT: {verdict}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(out_path, index=False)
    print(f"  trades CSV saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
