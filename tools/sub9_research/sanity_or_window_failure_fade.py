"""Pre-coding sanity check for C-10 OR-Window Failure Fade.

Candidate spec: `specs/2026-05-16-new-setup-candidates.md` -> CANDIDATE-10.

THESIS: During the Initial Balance (IB) window 09:30-10:30 IST, retail breakout
traders are taught to enter on ORH/ORL pierces. When those pierces FAIL (close
back through the level within 1-2 bars), retail entries get trapped and create
cascading stop-out flow in the opposite direction. Fading the failed pierce
captures the cascade.

Distinct from retired `pdh_pdl_sweep_reclaim` (sub-8, ICT framing at PDH/PDL).
This uses TODAY's forming OR levels (not yesterday's PDH/PDL) during IB window
specifically. Different participant (retail OR breakout traders during
establishment) and different timing.

UNIVERSE + FILTERS:
  - cap_segment in {small_cap, mid_cap}
  - MIS-eligible
  - Symbol in 5m feather cache

OR LEVEL COMPUTATION:
  - Opening Range = first 3 5m bars (09:15-09:30)
  - ORH = max(high) of those 3 bars
  - ORL = min(low) of those 3 bars

DETECTION (active window 09:30 - 10:30 IST):
  - Bar pierces ORH (UPside) by >= 0.15% AND close back below ORH (rejection)
  - Or bar pierces ORL (DOWNside) by >= 0.15% AND close back above ORL (rejection)
  - Sweep bar's volume >= 2x session-cumulative-average
  - Next 1 bar's close stays on recovery side

ENTRY:
  - Upside pierce -> SHORT (cascade-down trade)
  - Downside pierce -> LONG (cascade-up trade)
  - Entry at confirmation bar's close
  - SL = 0.3% beyond sweep extreme
  - T1 = entry +/- 1R, T2 = entry +/- 2R
  - Time stop = 15:10

DECISION CRITERION:
  n_total < 200 -> STRUCTURAL RETIRE-PRE-DATA
  PF >= 1.10 + n >= 200 -> STRONG PROCEED
  Else MARGINAL or RETIRE
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


ALLOWED_CAPS = {"small_cap", "mid_cap"}

OR_BARS = 3                  # opening range = first 3 5m bars (09:15-09:30)
ACTIVE_START_HHMM = "09:30"  # signals can fire from 09:30 onwards
ACTIVE_END_HHMM = "10:30"    # IB window ends 10:30

POKE_PCT = 0.15              # bar pokes >= POKE_PCT% beyond OR level
VOL_RATIO_MIN = 2.0          # bar.volume >= VOL_RATIO_MIN * session_cumulative_average

SL_PCT_BEYOND_SWEEP = 0.3
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


def _simulate_day(symbol: str, day_bars: pd.DataFrame, cap_segment: str) -> Optional[dict]:
    """Walk day bars. Compute OR from first 3 bars, then look for failed pierce in 09:30-10:30."""
    if day_bars.empty:
        return None

    bars = day_bars.copy().reset_index(drop=True)
    bars["hhmm"] = bars["date"].dt.strftime("%H%M").astype(int)

    if len(bars) < OR_BARS + 2:
        return None

    or_bars = bars.iloc[:OR_BARS]
    orh = float(or_bars["high"].max())
    orl = float(or_bars["low"].min())

    active_start = int(ACTIVE_START_HHMM.replace(":", ""))
    active_end = int(ACTIVE_END_HHMM.replace(":", ""))
    exit_cutoff = int(EXIT_BAR_HHMM.replace(":", ""))

    bars["cum_vol_mean"] = bars["volume"].expanding(min_periods=2).mean().shift(1)

    # Scan bars in active window for pierces (start from OR_BARS to skip OR bars themselves)
    for i in range(OR_BARS, len(bars) - 1):
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

        vol_ratio = bar_volume / float(bar["cum_vol_mean"])
        if vol_ratio < VOL_RATIO_MIN:
            continue

        # Upside ORH pierce + failure -> SHORT
        if bar_high >= orh * (1.0 + POKE_PCT / 100.0) and bar_close < orh:
            if float(next_bar["close"]) < orh:
                direction = "short"
                sweep_extreme = bar_high
                entry_price = float(next_bar["close"])
                hard_sl = sweep_extreme * (1.0 + SL_PCT_BEYOND_SWEEP / 100.0)
                R = hard_sl - entry_price
                if R <= 0:
                    continue
                or_level = orh
                return _path_walk(bars, i + 1, direction, entry_price, hard_sl, R,
                                  exit_cutoff, symbol, bar["date"].date(), cap_segment,
                                  or_level, "ORH", sweep_extreme, vol_ratio)

        # Downside ORL pierce + failure -> LONG
        if bar_low <= orl * (1.0 - POKE_PCT / 100.0) and bar_close > orl:
            if float(next_bar["close"]) > orl:
                direction = "long"
                sweep_extreme = bar_low
                entry_price = float(next_bar["close"])
                hard_sl = sweep_extreme * (1.0 - SL_PCT_BEYOND_SWEEP / 100.0)
                R = entry_price - hard_sl
                if R <= 0:
                    continue
                or_level = orl
                return _path_walk(bars, i + 1, direction, entry_price, hard_sl, R,
                                  exit_cutoff, symbol, bar["date"].date(), cap_segment,
                                  or_level, "ORL", sweep_extreme, vol_ratio)

    return None


def _path_walk(bars, entry_idx, direction, entry_price, hard_sl, R,
               exit_cutoff, symbol, trade_date, cap_segment,
               or_level, or_name, sweep_extreme, vol_ratio):
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
        "signal_type": f"or_window_failure_fade_{direction}",
        "cap_segment": cap_segment,
        "or_name": or_name,
        "or_level": or_level,
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
        "close_at_1100": _close_at(1100),
        "close_at_1200": _close_at(1200),
        "close_at_1300": _close_at(1300),
        "close_at_1500": _close_at(1500),
        "realized_pnl": realized_pnl,
        "fee": fee,
        "net_pnl": net_pnl,
    }


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
        / f"_or_window_failure_fade_trades_{WINDOW_LABEL}.csv"
    )

    print(f"== sanity_or_window_failure_fade ({WINDOW_LABEL}: {WINDOW_START} -> {WINDOW_END}) ==")
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
    print(f"  C-10 OR-WINDOW FAILURE FADE - {WINDOW_LABEL.upper()} VERDICT")
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
