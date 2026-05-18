"""Pre-coding sanity check for C-08 Last-Hour VWAP-Mean-Revert SHORT (MIS-Unwind Window).

Candidate spec: `specs/2026-05-16-new-setup-candidates.md` -> CANDIDATE-08.

THESIS: During the 14:30-15:15 MIS auto-square-off window, retail long positions
on stocks trading above their session VWAP face forced sell pressure (broker
auto-liquidation by 15:20 SEBI rule). This creates mean-reversion pressure back
toward VWAP. Short the extension, target VWAP retest, hard time-stop at 15:15.

Distinct from retired `mis_unwind_short` (sub-7): same THESIS, different MECHANIC.
The retired version used VWAP-cross-signal + momentum gate. This uses VWAP-
EXTENSION + RSI overbought as the entry signal.

UNIVERSE + FILTERS:
  - cap_segment in {small_cap, mid_cap} (where retail concentration is highest)
  - MIS-eligible

ACTIVE WINDOW: 14:30 - 15:10 IST (MIS auto-square window starts 15:20, exit by 15:10)

SIGNAL (per 5m bar):
  - current_price / session_vwap >= 1.005 (>= 50 bps above VWAP)
  - RSI(14) >= 65 (overbought confirmation)
  - bar volume >= 2x session-cumulative-average (volume confirms reversal-imminent)

ENTRY:
  - SHORT at signal bar's close
  - SL = entry * (1 + 0.4%)  (0.4% above)
  - T1 = session_vwap (the magnet)
  - T2 = session_vwap * 0.997 (slightly below VWAP)
  - Hard time stop = 15:10

R-MULTIPLE FORM (for consistent comparison):
  - R = SL distance = entry * 0.004
  - T1 = entry - 1R
  - T2 = entry - 2R

DECISION:
  n_total < 200 -> RETIRE-PRE-DATA
  PF >= 1.10 + n >= 200 -> PROCEED
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment, get_mis_info  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee       # noqa: E402


ALLOWED_CAPS = {"small_cap", "mid_cap"}

ACTIVE_START_HHMM = "14:30"
ACTIVE_END_HHMM = "15:00"  # 2026-05-17: tightened from 15:10 (matches production fix)

VWAP_EXTENSION_PCT = 0.5    # current_price / vwap - 1 >= 0.5%
RSI_OVERBOUGHT = 65
VOL_RATIO_MIN = 2.0         # bar.volume >= 2x session-cumulative-avg

SL_PCT_ABOVE_ENTRY = 0.4
T1_R = 1.0
T2_R = 2.0
EXIT_BAR_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000

# Entry-zone semantics (mode A only). Default Mode B = next-bar-open.
ENTRY_MODE = "B"                # "A" | "B" — set by --entry-mode
ENTRY_ZONE_PCT = 0.2            # symmetric, matches setups.mis_unwind_vwap_revert_short.entry_zone_pct
TRIGGER_EXPIRY_BARS = 3         # 15 minutes = 3 x 5m bars

WINDOWS = {
    "discovery": (date(2023, 1, 1), date(2024, 12, 31)),
    "oos":       (date(2025, 1, 1), date(2025, 9, 30)),
    "holdout":   (date(2025, 10, 1), date(2026, 4, 30)),
}

WINDOW_START = WINDOWS["discovery"][0]
WINDOW_END = WINDOWS["discovery"][1]
WINDOW_LABEL = "discovery"


def _months_in_window():
    months = []
    y, m = WINDOW_START.year, WINDOW_START.month
    while (y, m) <= (WINDOW_END.year, WINDOW_END.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _load_5m_for_month(yyyy, mm):
    p = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_feather(p)


def _is_mis_eligible(bare_symbol):
    try:
        return bool(get_mis_info(f"NSE:{bare_symbol}").get("mis_enabled", False))
    except Exception:
        return False


def _cap_segment(bare_symbol):
    try:
        return get_cap_segment(f"NSE:{bare_symbol}")
    except Exception:
        return "unknown"


def _simulate_day(symbol, day_bars, cap_segment):
    if day_bars.empty or "vwap" not in day_bars.columns or "rsi" not in day_bars.columns:
        return None
    bars = day_bars.copy().reset_index(drop=True)
    bars["hhmm"] = bars["date"].dt.strftime("%H%M").astype(int)
    bars["cum_vol_mean"] = bars["volume"].expanding(min_periods=2).mean().shift(1)

    active_start = int(ACTIVE_START_HHMM.replace(":", ""))
    active_end = int(ACTIVE_END_HHMM.replace(":", ""))
    exit_cutoff = int(EXIT_BAR_HHMM.replace(":", ""))

    for i in range(len(bars)):
        bar = bars.iloc[i]
        hhmm = int(bar["hhmm"])
        if hhmm < active_start or hhmm > active_end:
            continue
        vwap = bar.get("vwap")
        rsi = bar.get("rsi")
        if pd.isna(vwap) or pd.isna(rsi) or vwap <= 0:
            continue
        if pd.isna(bar["cum_vol_mean"]) or bar["cum_vol_mean"] <= 0:
            continue

        close_px = float(bar["close"])
        vwap_val = float(vwap)
        ext_pct = (close_px / vwap_val - 1.0) * 100.0
        if ext_pct < VWAP_EXTENSION_PCT:
            continue
        if float(rsi) < RSI_OVERBOUGHT:
            continue

        bar_volume = float(bar["volume"])
        vol_ratio = bar_volume / float(bar["cum_vol_mean"])
        if vol_ratio < VOL_RATIO_MIN:
            continue

        # Signal qualifies - fire SHORT
        # Entry semantics — see ENTRY_MODE module constant.
        # Mode B (default, idealized): fill at next-bar OPEN.
        # Mode A (tick-zone-touch): walk subsequent bars; fill when range
        #   intersects entry_zone (signal_close ± ENTRY_ZONE_PCT). EXPIRE
        #   after TRIGGER_EXPIRY_BARS without touch.
        signal_close_px = close_px  # store for metadata

        if ENTRY_MODE == "B":
            if i + 1 >= len(bars):
                continue
            entry_bar = bars.iloc[i + 1]
            entry_ts_actual = entry_bar["date"]
            entry_price = float(entry_bar["open"])
            entry_offset = 1
        else:
            # Mode A
            zone_min = signal_close_px * (1.0 - ENTRY_ZONE_PCT / 100.0)
            zone_max = signal_close_px * (1.0 + ENTRY_ZONE_PCT / 100.0)
            entry_price = None
            entry_ts_actual = None
            entry_offset = None
            for j in range(1, min(1 + TRIGGER_EXPIRY_BARS, len(bars) - i)):
                cand = bars.iloc[i + j]
                cand_open = float(cand["open"])
                cand_high = float(cand["high"])
                cand_low = float(cand["low"])
                if zone_min <= cand_open <= zone_max:
                    entry_price = cand_open
                elif cand_low <= zone_max and cand_high >= zone_min:
                    entry_price = max(cand_low, zone_min)
                if entry_price is not None:
                    entry_ts_actual = cand["date"]
                    entry_offset = j
                    break
            if entry_price is None:
                continue  # EXPIRED — no zone touch within trigger window

        hard_sl = entry_price * (1.0 + SL_PCT_ABOVE_ENTRY / 100.0)
        R = hard_sl - entry_price
        if R <= 0:
            continue

        t1_target = entry_price - T1_R * R
        t2_target = entry_price - T2_R * R

        after = bars.iloc[i + entry_offset:].copy()
        after = after[after["hhmm"] <= exit_cutoff]
        if after.empty:
            return None

        mfe_price = entry_price
        mae_price = entry_price
        closes_at_hhmm = {}
        baseline_exit_ts = baseline_exit_price = baseline_exit_reason = None

        for ar_bar in after.itertuples(index=False):
            ts = ar_bar.date
            hi = float(ar_bar.high)
            lo = float(ar_bar.low)
            cl = float(ar_bar.close)
            hhmm_b = int(ar_bar.hhmm)

            mfe_price = min(mfe_price, lo)
            mae_price = max(mae_price, hi)
            closes_at_hhmm[hhmm_b] = cl

            if baseline_exit_price is None:
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

        mfe_r = (entry_price - mfe_price) / R
        mae_r = (mae_price - entry_price) / R

        qty = max(int(RISK_PER_TRADE_RUPEES / max(R, 1e-6)), 1)
        realized_pnl = (entry_price - baseline_exit_price) * qty
        fee = calc_fee(entry_price, baseline_exit_price, qty, "SELL")
        net_pnl = realized_pnl - fee

        return {
            "trade_date": bar["date"].date(),
            "signal_ts": bar["date"],
            "signal_close": signal_close_px,
            "symbol": symbol,
            "side": "SHORT",
            "signal_type": "mis_unwind_vwap_revert_short",
            "cap_segment": cap_segment,
            "vwap": vwap_val,
            "rsi": float(rsi),
            "vwap_ext_pct": ext_pct,
            "vol_ratio": vol_ratio,
            "entry_ts": entry_ts_actual,
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
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        }
    return None


def main():
    global WINDOW_START, WINDOW_END, WINDOW_LABEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=list(WINDOWS.keys()), default="discovery")
    parser.add_argument("--out", default=None)
    parser.add_argument("--entry-mode", choices=["B", "A"], default="B",
                        help="B=next-bar-open (default, idealized); "
                             "A=tick-zone-touch (matches a tick-aware production trigger)")
    args = parser.parse_args()
    global ENTRY_MODE
    ENTRY_MODE = args.entry_mode
    WINDOW_LABEL = args.window
    WINDOW_START, WINDOW_END = WINDOWS[WINDOW_LABEL]

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "reports" / "sub9_sanity"
        / f"_mis_unwind_vwap_revert_trades_{WINDOW_LABEL}.csv"
    )

    print(f"== sanity_mis_unwind_vwap_revert ({WINDOW_LABEL}: {WINDOW_START} -> {WINDOW_END}) ==")
    print("  loading 5m feathers + walking month-by-month ...")

    all_trades = []
    for (y, m) in _months_in_window():
        mdf = _load_5m_for_month(y, m)
        if mdf.empty:
            continue
        mdf["d"] = mdf["date"].dt.date
        mdf = mdf[(mdf["d"] >= WINDOW_START) & (mdf["d"] <= WINDOW_END)]
        if mdf.empty:
            continue
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
            trade = _simulate_day(sym, day_bars, cap)
            if trade is not None:
                all_trades.append(trade)
        del mdf

    print(f"    trades fired total: {len(all_trades):,}")

    if not all_trades:
        print("  NO TRADES FIRED - exit")
        return 0

    tdf = pd.DataFrame(all_trades)
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
    print(f"  C-08 MIS-UNWIND VWAP-MEAN-REVERT SHORT - {WINDOW_LABEL.upper()} VERDICT")
    print("=" * 80)
    print(f"  n trades:               {n:,}")
    print(f"  win rate:               {wr:.1f}%  ({wins} wins / {n - wins} losses)")
    print(f"  Profit Factor:          {pf:.3f}")
    print(f"  NET PnL (after fees):   Rs. {net:>+12,.0f}")
    print(f"  Avg win / avg loss:     Rs. {avg_win:>+8,.0f}  /  Rs. {avg_loss:>+8,.0f}")
    print(f"  Annualized Sharpe:      {sharpe:.2f}")
    print(f"  Exit reason mix:        {exit_dist}")
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
